"""Offline tests for the R9 (each ablation hurts) verifier. No network, no GPU."""
from __future__ import annotations

import pytest

from trinity.analysis.ablations import analyze, render


def _spec_shape():
    # SPEC Table 2 shape: full tops every ablation; tri-role and the last-token
    # (penultimate) choice have the largest drops.
    return {
        "full": 0.7044,
        "no_svf": 0.685,
        "no_thinker": 0.690,
        "no_trirole": 0.640,     # big drop
        "last_token": 0.585,     # biggest drop (SPEC: 61.46 -> 50.85 on LCB)
    }


def test_r9_holds_when_every_ablation_hurts():
    report = analyze(_spec_shape())
    assert report["n_ablations"] == 4 and report["n_hurt"] == 4
    assert report["r9_holds"] is True
    assert report["did_not_hurt"] == []


def test_matter_most_is_the_two_largest_drops():
    report = analyze(_spec_shape())
    assert report["matter_most"] == ["last_token", "no_trirole"]
    # drops are sorted descending.
    drops = [d["drop"] for d in report["drops"]]
    assert drops == sorted(drops, reverse=True)


def test_r9_violated_when_an_ablation_does_not_hurt():
    accs = {"full": 0.70, "no_svf": 0.68, "no_thinker": 0.71}  # removing thinker helps?!
    report = analyze(accs)
    assert report["r9_holds"] is False
    assert report["did_not_hurt"] == ["no_thinker"]
    assert report["n_hurt"] == 1


def test_a_tie_is_not_a_hurt():
    report = analyze({"full": 0.70, "no_svf": 0.70})
    assert report["r9_holds"] is False
    assert report["did_not_hurt"] == ["no_svf"]


def test_drop_is_full_minus_ablation():
    report = analyze({"full": 0.70, "no_svf": 0.64})
    d = report["drops"][0]
    assert d["ablation"] == "no_svf" and d["drop"] == pytest.approx(0.06)


def test_custom_full_key_and_non_numeric_skipped():
    report = analyze({"baseline": 0.70, "no_svf": 0.66, "broken": None}, full="baseline")
    assert [d["ablation"] for d in report["drops"]] == ["no_svf"]
    assert report["r9_holds"] is True


def test_missing_full_key_raises():
    with pytest.raises(KeyError):
        analyze({"no_svf": 0.66, "no_thinker": 0.65})


def test_no_ablations_does_not_hold():
    assert analyze({"full": 0.70})["r9_holds"] is False


def test_render_reports_verdict_and_matter_most():
    md = render(_spec_shape())
    assert "R9 (removing each component hurts): HOLDS" in md
    assert "4/4 ablations hurt" in md
    assert "matter most (largest drops): last_token, no_trirole" in md

    bad = render({"full": 0.70, "no_thinker": 0.71})
    assert "VIOLATED" in bad and "did NOT hurt: no_thinker" in bad


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
