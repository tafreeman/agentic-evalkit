"""Tests for the shared redaction policy (design §12, plan Task 13)."""

import re

import pytest

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


# --- Story 2.4 (R-002): RedactionPolicy construction edges ------------------
#
# The default pattern set must redact each representative credential shape in
# isolation, and a malformed policy must fail loudly rather than silently
# under-redact. See the report note on the "rejected at construction" AC:
# ``RedactionPolicy()`` (empty) is the documented *opt-out* (the runner
# defaults to ``DEFAULT_REDACTION_POLICY``, with empty as the explicit
# opt-out), and patterns are compiled at the write boundary
# (``apply_redaction``), not in the pydantic constructor -- so an invalid
# regex is rejected at *write* time, not construction time. These guards pin
# the behaviour the source actually provides: fail-loud, never silent
# under-redaction.


def _with_single_evidence_value(run: EvalRunResult, value: str) -> EvalRunResult:
    """Return ``run`` with one grade's evidence replaced by ``{"note": value}``."""
    grade = run.samples[0].grade
    assert grade is not None
    grade = grade.model_copy(update={"evidence": {"note": value}})
    sample = run.samples[0].model_copy(update={"grade": grade})
    return run.model_copy(update={"samples": (sample, *run.samples[1:])})


@pytest.mark.parametrize(
    ("secret", "sentinel"),
    [
        ("hf_AbCdEfGh0123456789", "hf_AbCdEfGh0123456789"),
        ("sk-proj-abcDEF0123456789xy", "sk-proj-abcDEF0123456789xy"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1Ni.payload", "eyJhbGciOiJIUzI1Ni.payload"),
    ],
    ids=["hf-token", "openai-sk", "bearer"],
)
def test_each_default_pattern_redacts_its_representative_secret_in_isolation(
    pass_error_timeout_and_provenance_run: EvalRunResult,
    secret: str,
    sentinel: str,
) -> None:
    """Every default secret pattern redacts its representative credential when
    that credential is the only thing in the evidence value, so no default
    pattern is dead or mis-anchored.
    """
    run = _with_single_evidence_value(
        pass_error_timeout_and_provenance_run, f"captured {secret} in output"
    )
    redacted = apply_redaction(run, DEFAULT_REDACTION_POLICY)
    redacted_grade = redacted.samples[0].grade
    assert redacted_grade is not None
    rendered = str(redacted_grade.evidence["note"])
    assert sentinel not in rendered
    assert "[REDACTED]" in rendered


def test_empty_policy_is_a_valid_opt_out_not_a_construction_error() -> None:
    """``RedactionPolicy()`` (no patterns, no evidence keys) is a valid,
    constructible opt-out -- it is the documented way to disable redaction,
    not a malformed policy. Construction must succeed and leave both tuples
    empty.
    """
    policy = RedactionPolicy()
    assert policy.secret_patterns == ()
    assert policy.evidence_keys == ()


def test_invalid_regex_pattern_fails_loudly_at_the_write_boundary(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """A policy carrying an unparseable regex must not silently under-redact:
    it fails loudly (``re.error``) when applied at the write boundary. (Note:
    the source compiles patterns in ``apply_redaction``, so this is enforced
    at write time, not in the constructor -- see the Story 2.4 report note.)
    """
    bad_policy = RedactionPolicy(secret_patterns=("[unterminated",))
    with pytest.raises(re.error):
        apply_redaction(pass_error_timeout_and_provenance_run, bad_policy)


def test_construction_accepts_an_invalid_regex_but_write_rejects_it(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """Documents the AC deviation explicitly: an invalid regex is accepted at
    construction (pydantic does not compile it) and only rejected when the
    *same* policy is used at the write boundary. This exercises both halves of
    its name: construction must not raise, and applying that constructed policy
    must raise ``re.error`` rather than silently under-redacting.
    """
    policy = RedactionPolicy(secret_patterns=("(",))
    # Constructed fine; the pattern is stored verbatim as a string.
    assert policy.secret_patterns == ("(",)
    # The same policy, applied at the write boundary, fails loudly when the
    # pattern is finally compiled -- not a silent no-op.
    with pytest.raises(re.error):
        apply_redaction(pass_error_timeout_and_provenance_run, policy)
