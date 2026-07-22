"""LLM-as-coordinator baseline (SPEC §1.3 R11, Table 8).

SPEC **R11** — *"Trained coordinator > LLM-as-coordinator"* — is, as the merged verifier
:mod:`trinity.analysis.coordinator_vs_llm` puts it, *"the whole thesis of TRINITY"*: a
tiny (<20K-param) SLM + linear head trained with sep-CMA-ES should route better than
simply **prompting an LLM to act as the coordinator**. If a frozen LLM picking the
models and roles matched the trained head, the evolved coordinator would not be earning
its keep.

The verifier for R11 is merged (#360) and so is its report script, but nothing in
``src/`` could ever *produce* an LLM-as-coordinator run — so the baseline accuracy it
grades had to be typed in from the paper. This module is that producer.

**It is a drop-in, and changes nothing.** :mod:`trinity.orchestration.session` already
routes through a ``Policy`` protocol — ``decide(transcript_text, *, sample, rng) ->
(agent_idx, Role)`` — precisely so alternative coordinators can be swapped in (its own
docstring notes "tests pass a mock"). :class:`LLMCoordinatorPolicy` implements that
protocol, so measuring R11 needs **no change to the session loop, the trained policy, or
the submission path**.

**Why the client is synchronous.** ``Policy.decide`` is sync — the trained coordinator is
a local forward pass — and ``run_trajectory`` calls it without awaiting. This policy
therefore takes a plain ``chat`` callable rather than the async pool, and a live run
supplies a sync bridge to it. Offline tests pass :class:`StubDecider`, mirroring how
:class:`trinity.fugu.conductor.StubConductor` backs ``PromptedConductor``.

**Two integrity properties the comparison depends on.**

1. *A parse failure is not a routing decision.* When the LLM emits something
   unparseable the policy falls back deterministically, but it **counts** every fallback.
   A baseline that mostly fell back is not an LLM-as-coordinator at all — it is a
   constant policy wearing its name — so :attr:`DecisionStats.fallback_rate` is exposed
   and :meth:`DecisionStats.is_representative` gates whether the run should be compared
   at all. Reporting a fallback-dominated run as "the LLM coordinator" would make R11
   pass for the wrong reason.
2. *The baseline is not free.* The trained head costs ~0 output tokens per decision; an
   LLM coordinator spends a full completion **per turn**, on top of the answering call.
   :class:`DecisionStats` accumulates that spend so an R11/R12 comparison can be
   budget-matched instead of accuracy-only — an accuracy-only read flatters the baseline
   by ignoring the tokens it burns to make each choice.

No torch, no network at module scope: pure stdlib over an injected callable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from trinity.types import ROLE_ORDER, Role

__all__ = [
    "DEFAULT_FALLBACK_ROLE",
    "LLM_COORDINATOR_SYSTEM",
    "ChatFn",
    "Decision",
    "DecisionStats",
    "LLMCoordinatorPolicy",
    "StubDecider",
    "build_decision_prompt",
    "parse_decision",
]

#: Role used when the model's reply cannot be parsed. Worker is the only role that can
#: produce an answer on its own, so a fallback to Worker keeps the trajectory able to
#: terminate with content rather than stalling on Thinker/Verifier turns.
DEFAULT_FALLBACK_ROLE: Role = Role.WORKER

#: A run whose decisions are mostly fallbacks is not an LLM-as-coordinator.
MAX_FALLBACK_RATE = 0.10

LLM_COORDINATOR_SYSTEM = (
    "You are the COORDINATOR of a multi-agent system. You do NOT solve the task "
    "yourself. Each turn you choose which agent from the pool acts next, and in which "
    "role.\n\n"
    "ROLES:\n"
    "- THINKER: produces meta-level guidance — a plan, a decomposition, or a critique "
    "of the partial solution. Does not solve the task end-to-end.\n"
    "- WORKER: acts directly on the task and produces the actual content (derivation, "
    "code, numerical result).\n"
    "- VERIFIER: checks whether the accumulated solution is correct and complete, and "
    "emits ACCEPT or REVISE. Selecting VERIFIER is how a trajectory can terminate "
    "early, so do not select it before a WORKER has produced something to check.\n\n"
    "Reply with EXACTLY these two lines and nothing else:\n"
    "AGENT: <the agent's name, exactly as listed>\n"
    "ROLE: <THINKER or WORKER or VERIFIER>"
)

_AGENT_RE = re.compile(r"^\s*AGENT\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ROLE_RE = re.compile(r"^\s*ROLE\s*:\s*([A-Za-z]+)\s*$", re.IGNORECASE | re.MULTILINE)


class ChatFn(Protocol):
    """A synchronous chat call: messages in, assistant text out.

    A live run adapts the async pool to this shape; tests pass :class:`StubDecider`.
    """

    def __call__(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class Decision:
    """One parsed coordinator decision."""

    agent_idx: int
    role: Role
    raw: str
    parsed: bool

    def as_tuple(self) -> tuple[int, Role]:
        """The ``Policy.decide`` return shape."""
        return (self.agent_idx, self.role)


@dataclass
class DecisionStats:
    """Accounting for a baseline run — what it chose, and what it cost to choose."""

    calls: int = 0
    parse_failures: int = 0
    client_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    role_counts: dict[str, int] = field(default_factory=dict)
    agent_counts: dict[int, int] = field(default_factory=dict)

    @property
    def fallbacks(self) -> int:
        """Decisions the LLM did not actually make (bad parse or a failed call)."""
        return self.parse_failures + self.client_errors

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.calls if self.calls else 0.0

    def is_representative(self, *, max_fallback_rate: float = MAX_FALLBACK_RATE) -> bool:
        """Is this run actually an LLM-as-coordinator, or mostly the fallback constant?

        R11 compares the trained head against *an LLM making the choices*. If most
        decisions came from the fallback, the run does not support that comparison.
        """
        return self.calls > 0 and self.fallback_rate <= max_fallback_rate

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view."""
        return {
            "calls": self.calls,
            "parse_failures": self.parse_failures,
            "client_errors": self.client_errors,
            "fallbacks": self.fallbacks,
            "fallback_rate": self.fallback_rate,
            "is_representative": self.is_representative(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "role_counts": dict(self.role_counts),
            "agent_counts": {str(k): v for k, v in self.agent_counts.items()},
        }


def build_decision_prompt(
    transcript_text: str,
    pool_models: Sequence[str],
    *,
    turn: int | None = None,
    max_turns: int | None = None,
) -> list[dict[str, str]]:
    """Chat messages asking a model to pick the next ``(agent, role)``.

    The agent menu is listed by **name**, and the reply is expected to name the agent
    rather than index it — an index invites an off-by-one between the model's 1-based
    reading and the pool's 0-based indexing. :func:`parse_decision` still accepts a
    1-based index as a courtesy, exactly as advertised here.

    Parameters
    ----------
    transcript_text:
        The same coordinator input the trained head sees (``session._transcript_text``),
        so both coordinators route on identical context.
    pool_models:
        Pool model names in index order.
    turn, max_turns:
        Optional budget context; when both are given the prompt states how many turns
        remain, which is information the trained head has implicitly through training.
    """
    if not pool_models:
        raise ValueError("pool_models must be non-empty")

    menu = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(pool_models))
    budget = ""
    if turn is not None and max_turns is not None:
        remaining = max(0, max_turns - turn + 1)
        budget = (
            f"\n\nThis is turn {turn} of at most {max_turns} "
            f"({remaining} remaining, including this one)."
        )

    user = (
        f"AGENT POOL (choose exactly one, by name):\n{menu}\n\n"
        f"CONVERSATION SO FAR:\n{transcript_text}{budget}\n\n"
        "Choose the next agent and role."
    )
    return [
        {"role": "system", "content": LLM_COORDINATOR_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_decision(text: str, pool_models: Sequence[str]) -> tuple[int, Role] | None:
    """Parse ``AGENT:``/``ROLE:`` out of a reply; ``None`` if either is unusable.

    Tolerant in the ways that do not change meaning — case, surrounding whitespace,
    markdown emphasis, and extra prose around the two lines — and strict in the way that
    does: an agent that is not in the pool, or a role that is not one of the three, is a
    parse **failure**, never a guess. Silently coercing an out-of-pool name to index 0
    would fabricate a routing decision the model never made.

    The **last** occurrence of each field wins, so a model that reasons aloud and then
    states its final answer is read correctly.
    """
    agent_hits = _AGENT_RE.findall(text or "")
    role_hits = _ROLE_RE.findall(text or "")
    if not agent_hits or not role_hits:
        return None

    role_name = role_hits[-1].strip().upper()
    role = {r.name: r for r in ROLE_ORDER}.get(role_name)
    if role is None:
        return None

    raw_agent = agent_hits[-1].strip().strip("*_`\"'").strip()
    agent_idx = _resolve_agent(raw_agent, pool_models)
    if agent_idx is None:
        return None
    return (agent_idx, role)


def _resolve_agent(raw: str, pool_models: Sequence[str]) -> int | None:
    """Map an agent token to a pool index: exact name, then 1-based index."""
    lowered = [m.lower() for m in pool_models]
    token = raw.lower()
    if token in lowered:
        return lowered.index(token)

    # A bare "2." / "2" is the 1-based menu position advertised in the prompt.
    m = re.fullmatch(r"(\d+)\.?", raw)
    if m:
        one_based = int(m.group(1))
        if 1 <= one_based <= len(pool_models):
            return one_based - 1
        return None

    # "2. gpt-5" — a menu line echoed verbatim.
    m = re.match(r"(\d+)\.\s*(.+)$", raw)
    if m and m.group(2).lower() in lowered:
        return lowered.index(m.group(2).lower())
    return None


class StubDecider:
    """Offline stand-in for a chat model (mirrors ``fugu.conductor.StubConductor``).

    Give it a fixed reply, or a callable taking the messages and returning a reply, to
    drive parse/fallback paths deterministically with no network.
    """

    def __init__(self, reply: str | Callable[[list[dict[str, str]]], str]) -> None:
        self._reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if callable(self._reply):
            return self._reply(messages)
        return self._reply


class LLMCoordinatorPolicy:
    """Prompt an LLM to route, satisfying ``session.Policy`` (SPEC R11, Table 8).

    Parameters
    ----------
    chat:
        Synchronous chat callable (see :class:`ChatFn`).
    pool_models:
        Pool model names in index order — the same list ``run_trajectory`` receives, so
        ``agent_idx`` means the same thing to both coordinators.
    max_turns:
        Optional horizon, surfaced to the model as remaining-budget context.
    fallback_agent_idx, fallback_role:
        The deterministic decision used when a call or parse fails. Deterministic on
        purpose: a random fallback would inject noise into the baseline that R11 would
        then attribute to the LLM's routing.
    token_counter:
        Optional ``(messages, reply) -> (prompt_tokens, completion_tokens)`` so a run can
        account for the coordinator's own spend (see the module docstring).

    Notes
    -----
    ``sample`` and ``rng`` are accepted for protocol compatibility and ignored: the
    baseline's stochasticity lives in the model's own decoding, not in a policy-side
    categorical, so there is no distribution here to sample from.
    """

    def __init__(
        self,
        chat: ChatFn,
        pool_models: Sequence[str],
        *,
        max_turns: int | None = None,
        fallback_agent_idx: int = 0,
        fallback_role: Role = DEFAULT_FALLBACK_ROLE,
        token_counter: Callable[[list[dict[str, str]], str], tuple[int, int]] | None = None,
    ) -> None:
        if not pool_models:
            raise ValueError("pool_models must be non-empty")
        if not 0 <= fallback_agent_idx < len(pool_models):
            raise ValueError(
                f"fallback_agent_idx {fallback_agent_idx} out of range for "
                f"{len(pool_models)} pool models"
            )
        self.chat = chat
        self.pool_models = list(pool_models)
        self.max_turns = max_turns
        self.fallback_agent_idx = fallback_agent_idx
        self.fallback_role = fallback_role
        self.token_counter = token_counter
        self.stats = DecisionStats()
        self.history: list[Decision] = []
        self._turn = 0

    def reset(self) -> None:
        """Clear per-trajectory turn state. Stats and history are cumulative."""
        self._turn = 0

    def decide(
        self, transcript_text: str, *, sample: bool = False, rng: Any = None
    ) -> tuple[int, Role]:
        """Choose ``(agent_idx, role)`` for the next turn — the ``Policy`` protocol."""
        self._turn += 1
        self.stats.calls += 1

        messages = build_decision_prompt(
            transcript_text,
            self.pool_models,
            turn=self._turn if self.max_turns is not None else None,
            max_turns=self.max_turns,
        )

        try:
            reply = self.chat(messages)
        except Exception:
            # A dead call must not abort the whole baseline run; it is recorded as a
            # non-decision so fallback_rate reflects it.
            self.stats.client_errors += 1
            return self._record(self._fallback(""), parsed=False)

        if self.token_counter is not None:
            p_tok, c_tok = self.token_counter(messages, reply)
            self.stats.prompt_tokens += int(p_tok)
            self.stats.completion_tokens += int(c_tok)

        parsed = parse_decision(reply, self.pool_models)
        if parsed is None:
            self.stats.parse_failures += 1
            return self._record(self._fallback(reply), parsed=False)
        return self._record(Decision(parsed[0], parsed[1], reply, True), parsed=True)

    def _fallback(self, raw: str) -> Decision:
        return Decision(self.fallback_agent_idx, self.fallback_role, raw, False)

    def _record(self, decision: Decision, *, parsed: bool) -> tuple[int, Role]:
        self.history.append(decision)
        self.stats.role_counts[decision.role.value] = (
            self.stats.role_counts.get(decision.role.value, 0) + 1
        )
        self.stats.agent_counts[decision.agent_idx] = (
            self.stats.agent_counts.get(decision.agent_idx, 0) + 1
        )
        return decision.as_tuple()
