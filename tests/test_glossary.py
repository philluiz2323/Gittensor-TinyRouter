"""Doc-consistency tests for docs/GLOSSARY.md (issue #34).

The glossary defines the oracle-ceiling metrics (oracle, headroom, gap_closed, and their
supporting terms). These tests keep it honest without any API calls, GPU, or network: every
ceiling metric the glossary names must resolve to a real symbol in scripts/oracle_ceiling.py
(a CeilingStats field or a module-level definition), and the glossary must be linked from the
README so it is actually discoverable.
"""
import dataclasses
import importlib.util
import sys
from pathlib import Path

# Load the analysis script as a module (it lives under scripts/, not the importable package).
_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "oracle_ceiling.py"
_spec = importlib.util.spec_from_file_location("oracle_ceiling", _SCRIPT)
oc = importlib.util.module_from_spec(_spec)
sys.modules["oracle_ceiling"] = oc
_spec.loader.exec_module(oc)

_GLOSSARY = _ROOT / "docs" / "GLOSSARY.md"
_README = _ROOT / "README.md"

# The ceiling metrics the glossary must define. Each must map to a real symbol in
# oracle_ceiling.py so the doc cannot silently drift from the code it describes.
CORE_METRICS = [
    "best_single",
    "best_single_crossfit",
    "routing_oracle",
    "routing_oracle_naive",
    "clairvoyant_any",
    "routing_headroom",
    "unroutable_noise",
    "router_gap_closed",
]


def _code_symbols() -> set[str]:
    """Names a documented metric may resolve to: a CeilingStats field or a module-level name."""
    fields = {f.name for f in dataclasses.fields(oc.CeilingStats)}
    return fields | set(dir(oc))


def test_glossary_exists() -> None:
    assert _GLOSSARY.is_file(), "docs/GLOSSARY.md is missing"


def test_glossary_linked_from_readme() -> None:
    readme = _README.read_text(encoding="utf-8")
    assert "docs/GLOSSARY.md" in readme, "README must link to docs/GLOSSARY.md so it is discoverable"


def test_documented_metrics_are_real_code_symbols() -> None:
    text = _GLOSSARY.read_text(encoding="utf-8")
    symbols = _code_symbols()
    for metric in CORE_METRICS:
        assert f"`{metric}`" in text, f"{metric} is not documented in docs/GLOSSARY.md"
        assert metric in symbols, f"{metric} is documented but is not a real oracle_ceiling symbol"
