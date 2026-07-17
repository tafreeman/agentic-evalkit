"""Tests for the shared redaction policy -- the rules for scrubbing secrets
out of a report before it's written (design doc §12, plan Task 13).
"""

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


def test_redaction_covers_the_system_under_tests_raw_output_too(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """A secret-looking value inside ``execution.output`` (the raw output from
    the system being tested, not from our own grading code) must get redacted
    when the report is written, exactly like grade evidence does -- the
    tested system's own words don't get a free pass.

    This test closes a specific gap: elsewhere in the code, a docstring about
    "spilling" (writing an output that's too big to keep inline out to its
    own separate file) assumed this redaction step already covered ordinary,
    non-spilled outputs. But an output small enough to never trigger
    spilling -- or one that only shrinks below the spill-size threshold after
    being redacted -- still needs to pass through this same redaction
    function, or it would reach the final report with its secrets intact.
    """
    run = pass_error_timeout_and_provenance_run
    leaking_execution = run.samples[0].execution.model_copy(
        update={"output": {"answer": "42", "note": "captured sk-abc123 in transit"}}
    )
    leaking_sample = run.samples[0].model_copy(update={"execution": leaking_execution})
    run = run.model_copy(update={"samples": (leaking_sample, *run.samples[1:])})

    redacted = apply_redaction(run, RedactionPolicy(secret_patterns=(r"sk-[a-zA-Z0-9]+",)))
    redacted_output = redacted.samples[0].execution.output
    assert redacted_output is not None
    assert redacted_output == {"answer": "42", "note": "captured [REDACTED] in transit"}
    # The grade (untouched by this policy's evidence_keys) survives unchanged.
    assert redacted.samples[0].grade == run.samples[0].grade


def test_redaction_covers_execution_error_payloads(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """When something goes wrong while running a sample (a timeout, a crash,
    etc.), the ``error`` field can capture debugging text -- a stack trace, or
    a Python repr of some arguments -- that might happen to contain a
    credential (an API key or token). That text must be redacted exactly the
    same way a successful execution's output would be."""
    run = pass_error_timeout_and_provenance_run
    leaking_execution = run.samples[1].execution.model_copy(
        update={"error": {"message": "connect failed with Bearer eyJhbGciOiJIUzI1NiJ9.tok"}}
    )
    leaking_sample = run.samples[1].model_copy(update={"execution": leaking_execution})
    run = run.model_copy(update={"samples": (run.samples[0], leaking_sample, run.samples[2])})

    redacted = apply_redaction(run, DEFAULT_REDACTION_POLICY)
    redacted_error = redacted.samples[1].execution.error
    assert redacted_error is not None
    assert "eyJhbGciOiJIUzI1NiJ9" not in str(redacted_error)
    assert "[REDACTED]" in str(redacted_error)


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
    # Every default pattern requires a minimum length before it counts as a
    # match, so ordinary text that merely resembles the start of a secret is
    # left alone: "task-manager" happens to contain the literal characters
    # "sk-", "hf_hub" starts the way a Hugging Face token does, and "the
    # bearer is here" has only a short, harmless word after "bearer". Also,
    # "authorization" here is used as a dictionary *key*, not a value -- these
    # patterns only scan string values, never keys -- so a key literally
    # named "authorization" doesn't trigger anything either.
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


# --- Story 2.4 (R-002): RedactionPolicy construction edge cases -------------
#
# Two things are checked below: (1) each default secret pattern must redact
# its own kind of credential even when tested completely on its own, and (2) a
# broken policy (e.g. a bad regex) must fail with a loud error rather than
# quietly redacting nothing. This refers back to a review note on requirement
# R-002's "rejected at construction" acceptance criterion: an empty
# ``RedactionPolicy()`` is the documented, intentional way to turn redaction
# off (the CLI normally defaults to ``DEFAULT_REDACTION_POLICY``, so you'd
# have to pass an empty one on purpose), and regex patterns are only compiled
# -- and therefore only checked for validity -- inside ``apply_redaction`` (at
# write time), not inside the policy's constructor. So an invalid regex is
# rejected when a report is written, not when the ``RedactionPolicy`` object
# is created. The tests below pin down (lock in, as a regression check) that
# real behavior: fail loudly, never silently redact less than requested.


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
    """Each default secret-detecting pattern actually redacts the credential
    it's meant to catch, when that credential is the only text in the
    evidence value. This proves none of the default regex patterns are dead
    (never matching anything) or subtly broken (written to match a slightly
    wrong shape).
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
    """``RedactionPolicy()`` with no patterns and no evidence keys is a valid
    way to build the object -- it's the documented way to turn redaction off
    entirely, not a mistake or a malformed policy. Creating it this way must
    succeed, leaving both tuples empty.
    """
    policy = RedactionPolicy()
    assert policy.secret_patterns == ()
    assert policy.evidence_keys == ()


def test_invalid_regex_pattern_fails_loudly_at_the_write_boundary(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """A policy holding a regex pattern that Python can't even parse must not
    quietly redact nothing -- it should raise ``re.error`` (a loud failure)
    the moment it's actually used to write a report. (The code only compiles
    -- and therefore only validates -- patterns inside ``apply_redaction``,
    not inside the ``RedactionPolicy`` constructor, so this failure happens
    at write time. See the Story 2.4 note above for why.)
    """
    bad_policy = RedactionPolicy(secret_patterns=("[unterminated",))
    with pytest.raises(re.error):
        apply_redaction(pass_error_timeout_and_provenance_run, bad_policy)


def test_construction_accepts_an_invalid_regex_but_write_rejects_it(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """Spells out, on purpose, a difference from the original acceptance
    criteria: building a ``RedactionPolicy`` with an invalid regex succeeds
    (Pydantic doesn't try to compile the string, so it can't detect the
    problem yet) -- it's only rejected later, when that same policy is
    actually used to write a report. This test checks both halves: creating
    the policy must NOT raise, and then using it to redact a run must raise
    ``re.error`` instead of quietly redacting less than it should.
    """
    policy = RedactionPolicy(secret_patterns=("(",))
    # Creating the policy succeeds -- the pattern is just stored as-is, as a
    # plain string, not compiled yet.
    assert policy.secret_patterns == ("(",)
    # Using that same policy to write a report fails loudly once the pattern
    # is finally compiled -- it does not just silently do nothing.
    with pytest.raises(re.error):
        apply_redaction(pass_error_timeout_and_provenance_run, policy)
