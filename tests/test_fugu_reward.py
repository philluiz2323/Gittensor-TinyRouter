"""Offline tests for the Conductor reward: the false-positive/negative guards."""
from __future__ import annotations

from trinity.fugu.reward import committed_answer, is_correct, training_reward
from trinity.fugu.workflow import StepResult, WorkflowRun
from trinity.types import Task


def _run(final, *, parsed=True, steps=None):
    return WorkflowRun(
        workflow=None, parsed_ok=parsed, steps=steps or [], final_answer=final,
    )


def _step(out):
    return StepResult(step=0, model_id=0, model_name="minimax-m3", subtask="", output=out)


MATH = Task(task_id="m", benchmark="math500", prompt="2+2", answer="4")
MMLU = Task(task_id="c", benchmark="mmlu", prompt="pick", answer="C")


def test_parse_fail_is_incorrect_and_zero_reward():
    run = _run("\\boxed{4}", parsed=False)
    assert is_correct(run, MATH) == 0
    assert training_reward(run, MATH) == 0.0


def test_correct_and_wrong():
    assert is_correct(_run("\\boxed{4}"), MATH) == 1
    assert training_reward(_run("\\boxed{4}"), MATH) == 1.0
    assert is_correct(_run("\\boxed{5}"), MATH) == 0
    # parsed but wrong gets partial credit in TRAINING only.
    assert training_reward(_run("\\boxed{5}"), MATH) == 0.5


def test_committed_answer_recovers_from_a_non_boxed_final_step():
    # Final step rephrased without re-boxing; an earlier step had the boxed answer.
    run = _run("So the result follows.", steps=[_step("\\boxed{4}")])
    assert committed_answer("math500", run) == "\\boxed{4}"
    assert is_correct(run, MATH) == 1  # not a false negative


def test_empty_boxed_final_does_not_shadow_a_real_earlier_answer():
    # A later turn that re-boxes empty ("\boxed{}") must not be selected as the
    # committed answer over an earlier turn that boxed the real value.
    run = _run("Reformatting: \\boxed{}", steps=[_step("\\boxed{4}")])
    assert committed_answer("math500", run) == "\\boxed{4}"
    assert is_correct(run, MATH) == 1  # the produced answer 4 is not thrown away


def test_dollar_amount_is_not_a_false_negative():
    # Regression: "$18.90" gold vs "18.90" answer must grade correct (the shared
    # grader stripped bare "$" before "\\$", leaving "\\18.90"). Surfaced by the
    # math500 Conductor baseline (task math500-459).
    from trinity.orchestration import reward as R

    assert R.math_equal("18.90", "\\$18.90") is True
    assert R.score_text("math500", "The total cost is $18.90.", "\\$18.90") == 1.0
    run = _run("So the cost is $18.90.")
    assert is_correct(run, Task(task_id="d", benchmark="math500", prompt="", answer="\\$18.90")) == 1


def test_thousands_separator_is_not_a_false_negative():
    # Regression: a thousands-separated math answer ("2,000") must grade equal to
    # the same value without the separator. normalize_math_answer kept the comma
    # while extract_last_number stripped it, so the two halves of the grader
    # disagreed and any answer >= 1000 written with grouping scored 0.
    from trinity.orchestration import reward as R

    assert R.math_equal("2,000", "2000") is True
    assert R.math_equal("1,000,000", "1000000") is True
    assert R.score_text("math500", "Thus the answer is \\boxed{2,000}.", "2000") == 1.0
    assert R.score_text("math500", "The answer is 2000.", "2,000") == 1.0
    # A short comma-separated list is not a number and must stay unequal.
    assert R.math_equal("1,2,3", "123") is False
    run = _run("So the final count is \\boxed{12,345}.")
    assert is_correct(run, Task(task_id="t", benchmark="math500", prompt="", answer="12345")) == 1


def test_no_false_positive_from_prose_letter():
    # "A nice approach" must NOT be read as choice 'A' (the JOURNAL P2 bug).
    run = _run("A nice approach to this question.")
    assert is_correct(run, MMLU) == 0
    # but an explicit answer line is graded correctly.
    assert is_correct(_run("A nice approach, but the answer is C."), MMLU) == 1


def test_choice_takes_final_committed_answer_not_the_first():
    # A self-correcting response commits to its LAST stated choice. The grader
    # must read the committed answer, not the discarded first guess -- consistent
    # with extract_boxed / verifier.parse_verdict, which also take the last match.
    from trinity.orchestration import reward as R

    assert R.extract_choice_letter(
        "At first the answer is A. Wait, reconsidering, the answer is C."
    ) == "C"
    # End-to-end through the scorer: revised-to-correct is not a false negative,
    assert R.score_text("mmlu", "The answer is B. Actually the answer is D.", "D") == 1.0
    # and a revised-away first guess is not a false positive.
    assert R.score_text("mmlu", "The answer is D. Actually the answer is B.", "D") == 0.0


def test_choice_last_committed_answer_wins_across_different_phrasings():
    # "Last committed wins" must hold ACROSS the commitment patterns, not only
    # within one. A model instructed to box its final answer commonly reasons in
    # prose ("the answer is B") and then commits a *different*, boxed final (D).
    # The boxed final is the committed answer; the earlier prose guess is discarded.
    from trinity.orchestration import reward as R

    assert R.extract_choice_letter("The answer is B.\n\nFinal answer: \\boxed{D}") == "D"
    assert R.extract_choice_letter("So the answer is C. \\boxed{A}") == "A"
    # End-to-end: boxing the correct final after a wrong prose guess is not a false
    # negative, and the discarded prose guess is not a false positive.
    assert R.score_text("mmlu", "The answer is B. Final answer: \\boxed{D}", "D") == 1.0
    assert R.score_text("mmlu", "The answer is B. Final answer: \\boxed{D}", "B") == 0.0


def test_choice_weak_cues_never_override_a_committed_answer():
    # The fix must not regress the discussion / option-echo cases: a committed
    # answer outranks a later "option X" mention and a trailing option list.
    from trinity.orchestration import reward as R

    # "option B is wrong" is discussion, not a commitment -> the committed C wins.
    assert R.extract_choice_letter("The answer is C. Option B is wrong.") == "C"
    # An echoed option list after the commit must not override it.
    assert R.extract_choice_letter("The answer is B.\nA) one\nB) two\nC) three\nD) four") == "B"
    # With no committed-answer phrasing, the weaker cues still resolve a letter.
    assert R.extract_choice_letter("Option C") == "C"
    assert R.extract_choice_letter("C)") == "C"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("ALL PASS")
