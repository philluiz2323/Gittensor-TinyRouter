"""Math answers wrapped in \\textbf/\\textit/\\emph/\\textrm must still grade (issue #435)."""
from __future__ import annotations

from trinity.orchestration.reward import score_text


def test_textbf_textit_emph_textrm_unwrap_like_mathbf():
    assert score_text("math500", r"\boxed{\textbf{5}}", "5") == 1.0
    assert score_text("math500", r"\boxed{\textit{5}}", "5") == 1.0
    assert score_text("math500", r"\boxed{\emph{5}}", "5") == 1.0
    assert score_text("math500", r"\boxed{\textrm{5}}", "5") == 1.0
    assert score_text("math500", r"\boxed{\mathbf{5}}", "5") == 1.0
