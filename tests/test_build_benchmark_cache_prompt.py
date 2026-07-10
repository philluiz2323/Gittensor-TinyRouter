"""Offline test: cached benchmark answers use the WORKER single-turn prompt.

The hidden benchmark's cached model answers back the 70%-weighted single-turn
score, so they must be produced with the same single-turn invocation the router
is trained and live-evaluated against — one WORKER-role turn with an empty
transcript (`build_messages(Role.WORKER, question, [])`), exactly as
`trinity.eval._score_single_model` and `session.run_trajectory` do. Caching with
a bare user message would query a different prompt than the pipeline ever uses.

No network: a stub pool records the messages it is handed.
"""
import asyncio
import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

# build_benchmark lives under scripts/ and needs benchmark_protocol importable.
_SCRIPT = _REPO / "scripts" / "build_benchmark.py"
_spec = importlib.util.spec_from_file_location("build_benchmark", _SCRIPT)
build_benchmark = importlib.util.module_from_spec(_spec)
sys.modules["build_benchmark"] = build_benchmark
_spec.loader.exec_module(build_benchmark)

from trinity.roles.prompts import WORKER_SYSTEM, build_messages  # noqa: E402
from trinity.types import Role  # noqa: E402


class _StubResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _RecordingPool:
    """Records every (model, messages) it is asked to chat with; no network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict]]] = []

    async def chat(self, model, messages, **kwargs):
        self.calls.append((model, messages))
        return _StubResult(f"answer-from-{model}")


def _run(items, pool, models):
    asyncio.run(build_benchmark._cache_answers(items, pool, models))


def test_cache_uses_worker_role_messages():
    items = [{"question_id": "q0", "question_text": "What is 2+2?", "model_answers": {}}]
    pool = _RecordingPool()
    models = ["m-a", "m-b"]

    _run(items, pool, models)

    # One call per (item, model).
    assert len(pool.calls) == len(models)
    for model, messages in pool.calls:
        # Must be the WORKER single-turn layout, identical to the baseline path.
        assert messages == build_messages(Role.WORKER, "What is 2+2?", [])
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == WORKER_SYSTEM
        assert "What is 2+2?" in messages[1]["content"]
        # Regression guard: never a bare single user message.
        assert not (len(messages) == 1 and messages[0]["role"] == "user")

    # Answers are cached back per model.
    assert items[0]["model_answers"] == {"m-a": "answer-from-m-a", "m-b": "answer-from-m-b"}


def test_cache_skips_already_cached_models():
    items = [{"question_id": "q0", "question_text": "Q?", "model_answers": {"m-a": "prior"}}]
    pool = _RecordingPool()
    _run(items, pool, ["m-a", "m-b"])

    # m-a already had an answer -> only m-b is queried.
    assert [m for m, _ in pool.calls] == ["m-b"]
    assert items[0]["model_answers"]["m-a"] == "prior"
