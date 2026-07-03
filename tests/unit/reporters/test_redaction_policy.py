"""Tests for the shared redaction policy (design §12, plan Task 13)."""

from conftest import _run_with_pass_error_timeout_and_provenance

from agentic_evalkit.reporters import RedactionPolicy, apply_redaction


def test_redaction_removes_configured_evidence_keys() -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    policy = RedactionPolicy(evidence_keys=("actual",))
    redacted = apply_redaction(run, policy)
    passed_sample = redacted.samples[0]
    assert passed_sample.grade is not None
    assert "actual" not in passed_sample.grade.evidence
    assert passed_sample.grade.evidence == {"expected": "42"}


def test_redaction_replaces_secret_pattern_matches_in_string_evidence() -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    leaking_grade = run.samples[0].grade
    assert leaking_grade is not None
    leaking_grade = leaking_grade.model_copy(
        update={"evidence": {"note": "token=sk-abc123 leaked in output"}}
    )
    leaking_sample = run.samples[0].model_copy(update={"grade": leaking_grade})
    run = run.model_copy(update={"samples": (leaking_sample, *run.samples[1:])})

    policy = RedactionPolicy(secret_patterns=(r"sk-[a-zA-Z0-9]+",))
    redacted = apply_redaction(run, policy)
    redacted_grade = redacted.samples[0].grade
    assert redacted_grade is not None
    assert redacted_grade.evidence == {"note": "token=[REDACTED] leaked in output"}


def test_redaction_does_not_mutate_the_original_run() -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    original_evidence = run.samples[0].grade.evidence  # type: ignore[union-attr]
    policy = RedactionPolicy(evidence_keys=("actual",))
    apply_redaction(run, policy)
    assert run.samples[0].grade.evidence == original_evidence  # type: ignore[union-attr]


def test_redaction_with_empty_policy_returns_equivalent_but_new_model() -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    redacted = apply_redaction(run, RedactionPolicy())
    assert redacted == run
    assert redacted is not run


def test_redaction_leaves_samples_without_grades_untouched() -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    policy = RedactionPolicy(evidence_keys=("actual",))
    redacted = apply_redaction(run, policy)
    assert redacted.samples[1].grade is None
    assert redacted.samples[2].grade is None
