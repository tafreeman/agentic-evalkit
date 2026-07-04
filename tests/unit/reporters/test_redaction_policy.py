"""Tests for the shared redaction policy (design §12, plan Task 13)."""

from agentic_evalkit.models import EvalRunResult
from agentic_evalkit.reporters import DEFAULT_REDACTION_POLICY, RedactionPolicy, apply_redaction


def test_redaction_removes_configured_evidence_keys(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    run = pass_error_timeout_and_provenance_run
    policy = RedactionPolicy(evidence_keys=("actual",))
    redacted = apply_redaction(run, policy)
    passed_sample = redacted.samples[0]
    assert passed_sample.grade is not None
    assert "actual" not in passed_sample.grade.evidence
    assert passed_sample.grade.evidence == {"expected": "42"}


def test_redaction_replaces_secret_pattern_matches_in_string_evidence(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    run = pass_error_timeout_and_provenance_run
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


def test_redaction_does_not_mutate_the_original_run(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    run = pass_error_timeout_and_provenance_run
    original_evidence = run.samples[0].grade.evidence  # type: ignore[union-attr]
    policy = RedactionPolicy(evidence_keys=("actual",))
    apply_redaction(run, policy)
    assert run.samples[0].grade.evidence == original_evidence  # type: ignore[union-attr]


def test_redaction_with_empty_policy_returns_equivalent_but_new_model(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    run = pass_error_timeout_and_provenance_run
    redacted = apply_redaction(run, RedactionPolicy())
    assert redacted == run
    assert redacted is not run


def test_redaction_leaves_samples_without_grades_untouched(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    run = pass_error_timeout_and_provenance_run
    policy = RedactionPolicy(evidence_keys=("actual",))
    redacted = apply_redaction(run, policy)
    assert redacted.samples[1].grade is None
    assert redacted.samples[2].grade is None


def test_default_policy_redacts_known_credential_shapes(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    run = pass_error_timeout_and_provenance_run
    leaking_grade = run.samples[0].grade
    assert leaking_grade is not None
    leaking_grade = leaking_grade.model_copy(
        update={
            "evidence": {
                "hf": "hub token hf_AbCdEfGh0123456789 captured in output",
                "openai": "sk-proj-abcDEF0123456789xy",
                "header": "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload",
            }
        }
    )
    leaking_sample = run.samples[0].model_copy(update={"grade": leaking_grade})
    run = run.model_copy(update={"samples": (leaking_sample, *run.samples[1:])})

    redacted = apply_redaction(run, DEFAULT_REDACTION_POLICY)
    redacted_grade = redacted.samples[0].grade
    assert redacted_grade is not None
    rendered = str(redacted_grade.evidence)
    assert "hf_AbCdEfGh0123456789" not in rendered
    assert "sk-proj-abcDEF0123456789xy" not in rendered
    assert "eyJhbGciOiJIUzI1NiJ9" not in rendered
    assert "[REDACTED]" in rendered


def test_default_policy_leaves_benign_evidence_untouched(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    # Length guards must keep ordinary prose intact: "task-manager" contains
    # a literal "sk-", "hf_hub" starts like a token, "the bearer is here" has
    # a short word after "bearer", and "authorization" as an evidence *key*
    # is never pattern-scanned (patterns apply to string values only).
    run = pass_error_timeout_and_provenance_run
    benign = {
        "note": "hf_hub lookup for task-manager passed; the bearer is here",
        "expected": "42",
        "authorization": "granted",
    }
    grade = run.samples[0].grade
    assert grade is not None
    grade = grade.model_copy(update={"evidence": benign})
    sample = run.samples[0].model_copy(update={"grade": grade})
    run = run.model_copy(update={"samples": (sample, *run.samples[1:])})

    redacted = apply_redaction(run, DEFAULT_REDACTION_POLICY)
    redacted_grade = redacted.samples[0].grade
    assert redacted_grade is not None
    assert redacted_grade.evidence == benign
