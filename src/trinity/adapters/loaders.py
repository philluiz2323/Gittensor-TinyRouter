"""Raw benchmark loaders for the built-in benchmark adapters.

Moved out of ``trinity.orchestration.dataset`` (issue #10) so the per-benchmark
dataset parsing lives behind the adapter interface. Each ``_load_*`` returns a
list of :class:`~trinity.types.Task` (or ``None`` on any failure), with a tiny
offline toy fallback so smoke tests run with zero network. ``load_split`` is the
single canonical "raw-or-toy + deterministic shuffle + truncate" path shared by
the concrete adapters and the back-compat ``dataset.load_tasks`` shim.

This module imports only the stdlib and :class:`trinity.types` (``datasets`` is
lazy/guarded), so it has no dependency on the adapters package or the dataset
module and cannot introduce an import cycle.

HuggingFace dataset ids (when ``datasets`` + network are available):
- math500       : ``HuggingFaceH4/MATH-500`` (fallback ``qwedsacf/competition_math``)
- mmlu          : ``cais/mmlu`` (config ``all``; train -> ``auxiliary_train``)
- gpqa          : ``Idavidrein/gpqa`` (config ``gpqa_diamond``)
- livecodebench : ``livecodebench/code_generation_lite`` (V1 train / V6 eval)

Logical splits ("train"/"test") are resolved to each dataset's native split name
via ``_SPLIT_ALIASES`` -- upstream names do not always match, and a mismatch used
to demote the caller to the toy set silently. ``load_split`` emits a
``ToyFallbackWarning`` whenever the toy set stands in for real data.
"""
from __future__ import annotations

import random
import warnings
from typing import Any

from trinity.types import Task

#: Benchmarks with a dedicated raw loader in this module.
SUPPORTED_BENCHMARKS: tuple[str, ...] = ("math500", "mmlu", "gpqa", "livecodebench")

_CHOICE_LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")

# Logical split -> dataset-native split name, per benchmark.
#
# A logical split ("train"/"test") is *not* always the name the upstream dataset
# uses. `cais/mmlu` ships {test, validation, dev, auxiliary_train} and has no
# "train" at all, so forwarding "train" verbatim makes `load_dataset` raise and
# silently demotes the caller to the toy set. Benchmarks absent from this table
# pass their split through untouched; LiveCodeBench keeps its own mapping in
# `_lcb_version_for_split` because it resolves to a release *config*, not a split.
_SPLIT_ALIASES: dict[str, dict[str, str]] = {
    # MMLU's official training pool is `auxiliary_train`; `test` is the eval split.
    "mmlu": {"train": "auxiliary_train"},
}


class ToyFallbackWarning(UserWarning):
    """Warns that a benchmark degraded to the built-in toy set.

    The toy set exists so offline smoke tests run without a network. It must
    never stand in for real data unnoticed: a wrong split name or a gated
    repository would otherwise look exactly like intentional offline dev.
    """


def _resolve_split(benchmark: str, split: str) -> str:
    """Map a logical split onto the split name the upstream dataset actually has.

    Args:
        benchmark: One of :data:`SUPPORTED_BENCHMARKS`.
        split: The logical split, e.g. ``"train"`` or ``"test"``.

    Returns:
        The dataset-native split name, or ``split`` unchanged when the benchmark
        needs no aliasing.
    """
    key = (split or "").strip().lower()
    return _SPLIT_ALIASES.get(benchmark, {}).get(key, split)

def _try_load_hf(
    path: str,
    *,
    name: str | None = None,
    split: str | None = None,
) -> Any | None:
    """Attempt ``datasets.load_dataset``; return ``None`` on any failure.

    The import is lazy (so the module imports fine on a box without ``datasets``)
    and any error -- missing package, no network, unknown dataset id, gated repo --
    is swallowed so that callers fall back to the offline toy set. Failures are
    intentionally silent here; the caller decides whether the fallback is loud.

    Parameters
    ----------
    path:
        HuggingFace dataset repository id.
    name:
        Optional dataset config name (e.g. ``"all"`` for MMLU).
    split:
        Optional split string passed straight to ``load_dataset``.

    Returns
    -------
    The loaded dataset object, or ``None`` if loading was not possible.
    """
    try:
        from datasets import load_dataset  # type: ignore import-not-found
    except Exception:
        return None
    try:
        return load_dataset(path, name=name, split=split)
    except Exception:
        return None

def _row_get(row: Any, *keys: str, default: Any = None) -> Any:
    """Return the first present key from a (dict-like) dataset row."""
    for k in keys:
        try:
            if k in row and row[k] is not None:
                return row[k]
        except TypeError:
            # Non-mapping row; give up.
            break
    return default

def _load_math500_hf(split: str) -> list[Task] | None:
    """MATH-500 loader. answer = reference final answer string."""
    ds = _try_load_hf("HuggingFaceH4/MATH-500", split=split or "test")
    src = "HuggingFaceH4/MATH-500"
    if ds is None:
        # Fallback dataset uses a different schema (uses "solution" only).
        ds = _try_load_hf("qwedsacf/competition_math", split=split or "test")
        src = "qwedsacf/competition_math"
    if ds is None:
        return None
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        problem = _row_get(row, "problem", "question", default="")
        answer = _row_get(row, "answer", "solution", default="")
        if not problem:
            continue
        tasks.append(
            Task(
                task_id=f"math500-{i}",
                benchmark="math500",
                prompt=str(problem),
                answer=str(answer),
                meta={
                    "source": src,
                    "subject": _row_get(row, "subject", "type"),
                    "level": _row_get(row, "level"),
                },
            )
        )
    return tasks or None

def _load_mmlu_hf(split: str) -> list[Task] | None:
    """MMLU loader. answer = correct option LETTER ("A".."D").

    ``cais/mmlu`` exposes {test, validation, dev, auxiliary_train}; the logical
    ``"train"`` split is resolved to ``auxiliary_train`` by :func:`_resolve_split`.
    """
    ds = _try_load_hf("cais/mmlu", name="all", split=_resolve_split("mmlu", split or "test"))
    if ds is None:
        return None
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        question = _row_get(row, "question", default="")
        choices = _row_get(row, "choices", default=None)
        answer_idx = _row_get(row, "answer", default=None)
        if not question or not choices or answer_idx is None:
            continue
        try:
            answer_idx = int(answer_idx)
        except (TypeError, ValueError):
            continue
        if not (0 <= answer_idx < len(_CHOICE_LETTERS)):
            continue
        tasks.append(
            Task(
                task_id=f"mmlu-{i}",
                benchmark="mmlu",
                prompt=_format_mcq(str(question), list(choices)),
                answer=_CHOICE_LETTERS[answer_idx],
                meta={
                    "source": "cais/mmlu",
                    "subject": _row_get(row, "subject"),
                    "choices": list(choices),
                },
            )
        )
    return tasks or None

def _load_gpqa_hf(split: str) -> list[Task] | None:
    """GPQA-Diamond loader.

    GPQA stores the correct answer plus three distractors as separate columns.
    We shuffle them deterministically (per-row seeded by index) into A-D and
    record the resulting correct letter as the answer.
    """
    ds = _try_load_hf("Idavidrein/gpqa", name="gpqa_diamond", split=split or "train")
    if ds is None:
        return None
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        question = _row_get(row, "Question", "question", default="")
        correct = _row_get(row, "Correct Answer", default=None)
        incorrect = [
            _row_get(row, "Incorrect Answer 1"),
            _row_get(row, "Incorrect Answer 2"),
            _row_get(row, "Incorrect Answer 3"),
        ]
        incorrect = [c for c in incorrect if c is not None]
        if not question or correct is None or len(incorrect) < 3:
            continue
        options = [str(correct)] + [str(c) for c in incorrect[:3]]
        # Deterministic per-row shuffle so option positions are stable.
        order = list(range(len(options)))
        random.Random(i).shuffle(order)
        shuffled = [options[j] for j in order]
        correct_pos = order.index(0)  # original index 0 == correct answer
        tasks.append(
            Task(
                task_id=f"gpqa-{i}",
                benchmark="gpqa",
                prompt=_format_mcq(str(question), shuffled),
                answer=_CHOICE_LETTERS[correct_pos],
                meta={
                    "source": "Idavidrein/gpqa",
                    "config": "gpqa_diamond",
                    "choices": shuffled,
                },
            )
        )
    return tasks or None

def _load_livecodebench_hf(split: str) -> list[Task] | None:
    """LiveCodeBench loader.

    Per SPEC §6.1 the in-distribution split is V1 (train, 400) and V6
    (eval, 175). We map ``split`` -> release version:
        "train" / "v1" -> release_v1
        "test"  / "v6" -> release_v6

    answer is a dict test spec consumed by the sandboxed pass@1 executor:
        {"tests": [{"input": str, "output": str}, ...],
         "fn_name": str | None,
         "starter_code": str | None}
    """
    version = _lcb_version_for_split(split)
    ds = _try_load_hf(
        "livecodebench/code_generation_lite",
        name=version,
        split="test",
    )
    if ds is None:
        # Some mirrors expose the version via `split` rather than config name.
        ds = _try_load_hf("livecodebench/code_generation_lite", split=version)
    if ds is None:
        return None
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        question = _row_get(
            row, "question_content", "question", "problem", default=""
        )
        if not question:
            continue
        tests = _parse_lcb_tests(row)
        tasks.append(
            Task(
                task_id=str(_row_get(row, "question_id", default=f"lcb-{i}")),
                benchmark="livecodebench",
                prompt=str(question),
                answer={
                    "tests": tests,
                    "fn_name": _row_get(row, "fn_name", "func_name"),
                    "starter_code": _row_get(row, "starter_code"),
                },
                meta={
                    "source": "livecodebench/code_generation_lite",
                    "version": version,
                    "platform": _row_get(row, "platform"),
                    "difficulty": _row_get(row, "difficulty"),
                },
            )
        )
    return tasks or None

def _format_mcq(question: str, choices: list[Any]) -> str:
    """Render a multiple-choice question with lettered options.

    The prompt explicitly asks the pool model to end with a single answer
    letter so the reward checker's letter extraction is reliable.
    """
    lines = [question.strip(), ""]
    for letter, choice in zip(_CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer with the single letter of the correct option.")
    return "\n".join(lines)

def _lcb_version_for_split(split: str) -> str:
    """Map a logical split string onto a LiveCodeBench release config name."""
    s = (split or "").strip().lower()
    if s in ("test", "eval", "v6", "release_v6"):
        return "release_v6"
    # Default / train -> V1 (the SPEC training split).
    return "release_v1"

def _parse_lcb_tests(row: Any) -> list[dict[str, str]]:
    """Best-effort extraction of LiveCodeBench public test cases.

    LiveCodeBench schemas vary across mirrors. We accept either a JSON-encoded
    string or an already-parsed list under several common keys, and normalise to
    a list of ``{"input": ..., "output": ...}`` dicts. Returns ``[]`` if nothing
    parseable is found (the reward checker treats empty tests as unscoreable).
    """
    import json

    raw = _row_get(row, "public_test_cases", "test_cases", "tests", default=None)
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    tests: list[dict[str, str]] = []
    for case in raw:
        if isinstance(case, dict):
            inp = case.get("input", case.get("stdin", ""))
            out = case.get("output", case.get("expected_output", ""))
            tests.append({"input": str(inp), "output": str(out)})
    return tests

def _toy_tasks(benchmark: str) -> list[Task]:
    """Hand-written tiny task set so smoke tests run without ``datasets``/network.

    Each set has 2-3 deterministic, self-contained items whose ``answer`` matches
    the format the corresponding reward checker expects.
    """
    if benchmark == "math500":
        return [
            Task(
                task_id="math500-toy-0",
                benchmark="math500",
                prompt="What is 2 + 2? Give the final answer in \\boxed{}.",
                answer="4",
                meta={"source": "toy"},
            ),
            Task(
                task_id="math500-toy-1",
                benchmark="math500",
                prompt=(
                    "A train travels 60 miles in 1.5 hours. What is its average "
                    "speed in miles per hour? Put the answer in \\boxed{}."
                ),
                answer="40",
                meta={"source": "toy"},
            ),
            Task(
                task_id="math500-toy-2",
                benchmark="math500",
                prompt="Compute 7 * 8. Give the final answer in \\boxed{}.",
                answer="56",
                meta={"source": "toy"},
            ),
        ]
    if benchmark == "mmlu":
        return [
            Task(
                task_id="mmlu-toy-0",
                benchmark="mmlu",
                prompt=_format_mcq(
                    "What is the chemical symbol for water?",
                    ["CO2", "H2O", "O2", "NaCl"],
                ),
                answer="B",
                meta={"source": "toy", "choices": ["CO2", "H2O", "O2", "NaCl"]},
            ),
            Task(
                task_id="mmlu-toy-1",
                benchmark="mmlu",
                prompt=_format_mcq(
                    "Which planet is closest to the Sun?",
                    ["Venus", "Earth", "Mercury", "Mars"],
                ),
                answer="C",
                meta={
                    "source": "toy",
                    "choices": ["Venus", "Earth", "Mercury", "Mars"],
                },
            ),
        ]
    if benchmark == "gpqa":
        return [
            Task(
                task_id="gpqa-toy-0",
                benchmark="gpqa",
                prompt=_format_mcq(
                    "Which fundamental force binds quarks inside a proton?",
                    [
                        "Electromagnetic force",
                        "The strong nuclear force",
                        "Gravity",
                        "The weak nuclear force",
                    ],
                ),
                answer="B",
                meta={"source": "toy"},
            ),
            Task(
                task_id="gpqa-toy-1",
                benchmark="gpqa",
                prompt=_format_mcq(
                    "What is the approximate speed of light in a vacuum?",
                    [
                        "3 x 10^6 m/s",
                        "3 x 10^8 m/s",
                        "3 x 10^10 m/s",
                        "3 x 10^4 m/s",
                    ],
                ),
                answer="B",
                meta={"source": "toy"},
            ),
        ]
    if benchmark == "livecodebench":
        return [
            Task(
                task_id="lcb-toy-0",
                benchmark="livecodebench",
                prompt=(
                    "Read an integer n from standard input and print n * n.\n"
                    "Input: a single integer.\nOutput: the square of the integer."
                ),
                answer={
                    "tests": [
                        {"input": "3\n", "output": "9"},
                        {"input": "5\n", "output": "25"},
                    ],
                    "fn_name": None,
                    "starter_code": None,
                },
                meta={"source": "toy"},
            ),
            Task(
                task_id="lcb-toy-1",
                benchmark="livecodebench",
                prompt=(
                    "Read two integers a and b on one line separated by a space "
                    "and print their sum."
                ),
                answer={
                    "tests": [
                        {"input": "2 3\n", "output": "5"},
                        {"input": "10 -4\n", "output": "6"},
                    ],
                    "fn_name": None,
                    "starter_code": None,
                },
                meta={"source": "toy"},
            ),
        ]
    raise ValueError(
        f"Unknown benchmark {benchmark!r}. Supported: {SUPPORTED_BENCHMARKS}"
    )

_HF_LOADERS = {
    "math500": _load_math500_hf,
    "mmlu": _load_mmlu_hf,
    "gpqa": _load_gpqa_hf,
    "livecodebench": _load_livecodebench_hf,
}

def load_split(
    benchmark: str,
    split: str,
    max_items: int | None,
    seed: int = 0,
    allow_toy_fallback: bool = True,
) -> list[Task]:
    """Load a benchmark split as a deterministic list of :class:`Task`.

    The canonical loading path: resolve the logical split to the dataset's native
    split name (see :func:`_resolve_split`), try the benchmark's HuggingFace
    loader (lazy/guarded); on any failure fall back to the built-in toy set; then
    apply a ``seed``-seeded shuffle and truncate to ``max_items``. Repeated calls
    with identical arguments return identical lists.

    Falling back always emits a :class:`ToyFallbackWarning`, so a genuine load
    failure can never masquerade as real data.

    Args:
        benchmark: One of :data:`SUPPORTED_BENCHMARKS`.
        split: Logical split, e.g. ``"train"`` / ``"test"``.
        max_items: Cap on the number of tasks returned; ``None`` means all.
        seed: Seed controlling the deterministic shuffle.
        allow_toy_fallback: When ``True`` (the default) an unavailable dataset
            degrades to the toy set with a warning. When ``False`` it raises
            instead -- use this for training and hidden-benchmark builds, where
            toy data is never acceptable.

    Raises:
        ValueError: If ``benchmark`` has no registered raw loader.
        RuntimeError: If the dataset could not be loaded and ``allow_toy_fallback``
            is ``False``.
    """
    key = (benchmark or "").strip().lower()
    if key not in _HF_LOADERS:
        raise ValueError(
            f"Unknown benchmark {benchmark!r}. Supported: {SUPPORTED_BENCHMARKS}"
        )

    tasks = _HF_LOADERS[key](split)
    if not tasks:
        # Offline / failed load -> built-in toy set. Never do this quietly: a
        # wrong split name looks identical to intentional offline dev.
        if not allow_toy_fallback:
            raise RuntimeError(
                f"Could not load benchmark {key!r} split {split!r} from "
                f"HuggingFace, and allow_toy_fallback=False. Install `datasets`, "
                f"check network access, and verify the split exists upstream."
            )
        warnings.warn(
            f"Benchmark {key!r} split {split!r} could not be loaded from "
            f"HuggingFace; falling back to the built-in toy set "
            f"({len(_toy_tasks(key))} tasks). Results are not meaningful.",
            ToyFallbackWarning,
            stacklevel=2,
        )
        tasks = _toy_tasks(key)

    rng = random.Random(seed)
    tasks = list(tasks)
    rng.shuffle(tasks)

    if max_items is not None:
        tasks = tasks[: max(0, int(max_items))]
    return tasks
