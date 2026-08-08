"""Tests for the shared redaction policy -- the rules for scrubbing secrets
out of a report before it's written (design doc §12, plan Task 13).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.models import (
    DatasetRef,
    EvalRunManifest,
    EvalRunResult,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    SampleResult,
    is_output_spill_error_record,
)
from agentic_evalkit.reporters import DEFAULT_REDACTION_POLICY, RedactionPolicy, apply_redaction
from agentic_evalkit.reporters.base import _is_artifact_digest

if TYPE_CHECKING:
    from pathlib import Path


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


def test_redaction_covers_execution_artifacts(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """``artifacts`` used to hold nothing but a harness-authored digest, so
    the redaction sweep skipped it. It no longer does: a spill that fails
    records the artifact store's own exception message under
    ``output_spill_error``, and a store that talks to something remote is
    free to echo a URL, a header, or a credential into that text. The runner
    redacts that message on the way in, but a caller may have switched the
    runner's own spill redaction off (``redaction_policy=None`` is a
    documented opt-out), which leaves this report boundary as the only thing
    standing between that text and the written file."""
    run = pass_error_timeout_and_provenance_run
    leaking_execution = run.samples[1].execution.model_copy(
        update={
            "artifacts": {
                "output_spill_error": {
                    "type": "OSError",
                    "code": "output_spill_failed",
                    "message": "upload rejected for Bearer eyJhbGciOiJIUzI1NiJ9.tok",
                }
            }
        }
    )
    leaking_sample = run.samples[1].model_copy(update={"execution": leaking_execution})
    run = run.model_copy(update={"samples": (run.samples[0], leaking_sample, run.samples[2])})

    redacted = apply_redaction(run, DEFAULT_REDACTION_POLICY)
    redacted_artifacts = redacted.samples[1].execution.artifacts
    assert "eyJhbGciOiJIUzI1NiJ9" not in str(redacted_artifacts)
    assert "[REDACTED]" in str(redacted_artifacts)
    # The structural keys around the message survive -- only the secret-shaped
    # substring is replaced, so the record stays machine-readable.
    spill_record = redacted_artifacts["output_spill_error"]
    assert isinstance(spill_record, dict)
    assert spill_record["code"] == "output_spill_failed"


def test_redaction_leaves_a_spilled_output_digest_intact(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """Sweeping ``artifacts`` must not damage what normally lives there. An
    ``output_ref`` digest is hex, matches no credential shape, and is the
    only way to find a spilled payload again -- rewriting one character of it
    would orphan the artifact it points at."""
    run = pass_error_timeout_and_provenance_run
    digest = "sha256:" + "ab12" * 16
    referencing_execution = run.samples[0].execution.model_copy(
        update={"output": None, "artifacts": {"output_ref": digest}}
    )
    referencing_sample = run.samples[0].model_copy(update={"execution": referencing_execution})
    run = run.model_copy(update={"samples": (referencing_sample, *run.samples[1:])})

    redacted = apply_redaction(run, DEFAULT_REDACTION_POLICY)

    assert redacted.samples[0].execution.artifacts == {"output_ref": digest}


def test_a_caller_supplied_pattern_never_rewrites_an_output_ref_digest(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """The sweep over ``artifacts`` was justified by the observation that no
    *default* pattern matches a hex digest -- but ``apply_redaction`` is
    public API and takes whatever policy a caller hands it, and "a long hex
    string looks like a key" is an entirely reasonable rule for someone to
    add. The digest is the only pointer back to a payload already written to
    the artifact store, so rewriting one character of it orphans those bytes
    permanently: the report would reference an artifact nobody can look up.
    It is therefore restored verbatim after the sweep.
    """
    run = pass_error_timeout_and_provenance_run
    digest = "sha256:" + "ab12" * 16
    referencing_execution = run.samples[0].execution.model_copy(
        update={"output": None, "artifacts": {"output_ref": digest}}
    )
    referencing_sample = run.samples[0].model_copy(update={"execution": referencing_execution})
    run = run.model_copy(update={"samples": (referencing_sample, *run.samples[1:])})

    # A generic "long hex blob" rule, which does match this digest.
    hostile = RedactionPolicy(secret_patterns=(r"[A-Fa-f0-9]{32,}",))
    redacted = apply_redaction(run, hostile)

    assert redacted.samples[0].execution.artifacts["output_ref"] == digest


def test_a_target_cannot_smuggle_a_secret_through_the_output_ref_exemption(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """The exemption is gated on the value's shape, not on the key name.

    ``artifacts`` is target-controlled and ``output_ref`` is not a reserved
    name -- the same premise that makes ``is_output_spill_error_record``
    necessary one field over. If holding that key were enough to earn the
    exemption, a target returning ``artifacts={"output_ref": "hf_…"}`` would
    walk a live credential straight through this sweep and into the canonical
    report, under the *default* policy the CLI writes every report with. Only
    a value shaped like something ``ArtifactStore`` actually minted is
    exempt; anything else is target text and gets swept like the rest.
    """
    run = pass_error_timeout_and_provenance_run
    smuggled = "hf_" + "A" * 34
    execution = run.samples[0].execution.model_copy(
        update={"output": None, "artifacts": {"output_ref": smuggled}}
    )
    sample = run.samples[0].model_copy(update={"execution": execution})
    run = run.model_copy(update={"samples": (sample, *run.samples[1:])})

    redacted = apply_redaction(run, DEFAULT_REDACTION_POLICY)

    assert redacted.samples[0].execution.artifacts["output_ref"] == "[REDACTED]"


def test_the_digest_exemption_does_not_shield_the_rest_of_artifacts(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """Exempting ``output_ref`` must be exactly that narrow: every other key
    in ``artifacts`` still gets swept, including in a record that sits beside
    a digest in the same dict.
    """
    run = pass_error_timeout_and_provenance_run
    digest = "sha256:" + "ab12" * 16
    execution = run.samples[0].execution.model_copy(
        update={
            "output": None,
            "artifacts": {
                "output_ref": digest,
                "store_note": "upload used Bearer eyJhbGciOiJIUzI1NiJ9.tok",
            },
        }
    )
    sample = run.samples[0].model_copy(update={"execution": execution})
    run = run.model_copy(update={"samples": (sample, *run.samples[1:])})

    redacted = apply_redaction(run, DEFAULT_REDACTION_POLICY)
    artifacts = redacted.samples[0].execution.artifacts

    assert artifacts["output_ref"] == digest  # still exempt
    assert "eyJhbGciOiJIUzI1NiJ9" not in str(artifacts["store_note"])
    assert "[REDACTED]" in str(artifacts["store_note"])


def test_a_caller_supplied_pattern_never_rewrites_the_spill_failure_code(
    pass_error_timeout_and_provenance_run: EvalRunResult,
) -> None:
    """The record's ``code`` is a machine-readable contract, not prose.

    ``is_output_spill_error_record`` is what every consumer -- ``HarnessGrader``
    deciding a result cannot be re-graded, the CLI's dropped-output warning --
    uses to tell a genuine spill failure from something a target wrote. The
    sweep recurses into the record and rewrites its strings like any other, so
    a caller policy broad enough to match ``output_spill_failed`` would leave
    the redacted report carrying a record no consumer recognises: re-grading
    it falls back to the "produced no output" ``UNAVAILABLE`` this whole
    change exists to eliminate.

    The digest got a shape-gated exemption for the same class of reason. This
    is the other half of it. Restoring the code smuggles nothing: it is
    rewritten to the fixed constant, and only for a record that already
    carried it before the sweep -- ``message`` and ``type``, which are the
    parts that can hold store-authored text, stay redacted.
    """
    run = pass_error_timeout_and_provenance_run
    execution = run.samples[1].execution.model_copy(
        update={
            "output": None,
            "artifacts": {
                "output_spill_error": {
                    "type": "OSError",
                    "code": "output_spill_failed",
                    # A long lowercase run, so the very pattern that would
                    # have eaten the code matches in here too.
                    "message": "rejected by internalstoragebackend",
                }
            },
        }
    )
    sample = run.samples[1].model_copy(update={"execution": execution})
    run = run.model_copy(update={"samples": (run.samples[0], sample, run.samples[2])})

    # A broad identifier rule, which does match "output_spill_failed".
    hostile = RedactionPolicy(secret_patterns=(r"[a-z_]{16,}",))
    redacted = apply_redaction(run, hostile)

    record = redacted.samples[1].execution.artifacts["output_spill_error"]
    assert isinstance(record, dict)
    assert record["code"] == "output_spill_failed"  # still recognisable
    assert is_output_spill_error_record(record)
    # The exemption is exactly that narrow: the same pattern still rewrites
    # the message, which is where store-authored text lives.
    assert "[REDACTED]" in str(record["message"])
    assert "internalstoragebackend" not in str(record["message"])


def test_the_digest_exemption_matches_what_the_artifact_store_actually_mints(
    tmp_path: Path,
) -> None:
    """The exemption's shape gate and the store's digest format are two
    independent spellings of one contract, and nothing else pins them
    together.

    ``_OUTPUT_REF_DIGEST_RE`` hard-codes ``sha256:`` plus 64 lowercase hex
    characters; ``ArtifactStore._digest_of`` is what actually produces the
    value. Every other test here hand-writes the literal (``"sha256:" +
    "ab12" * 16``), so all of them would keep passing if the store switched
    to sha512, to an uppercase digest, or to a different prefix -- while
    every real digest silently lost the exemption and got rewritten by any
    caller policy matching long hex, orphaning the artifact it points at.
    This is the same drift ``test_the_spill_failure_code_constant_matches_
    the_error_taxonomy`` closes for the taxonomy code, applied to the other
    constant this change introduced.
    """
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_bytes(b'{"answer": "42"}', media_type="application/json")

    assert _is_artifact_digest(ref.digest)


def test_the_sweep_covers_every_free_form_field_a_run_carries() -> None:
    """The sweep is only worth anything if it reaches everywhere a secret can sit.

    For a long time it did not. It swept the target's ``output``,
    ``structured_output``, ``error`` and ``artifacts``, plus a grade's
    ``evidence`` -- the *output* side -- on the tacit assumption that
    everything else was benign harness or dataset content. Six fields were
    left untouched, and each is a real hiding place:

    * ``sample.input`` is the prompt, which routinely contains the very
      credential the agent under test is meant to use, or customer data
      lifted from a production trace to make a regression case;
    * ``sample.metadata`` and ``sample.expected_artifacts`` are
      adapter-authored and carry whatever the adapter put there;
    * ``sample.reference`` is dataset content, which is not automatically
      safe;
    * ``execution.tool_calls`` records the arguments the target sent to a
      tool, which is exactly where an API key travels;
    * ``execution.environment_metadata`` records the environment, which is
      where connection strings live;
    * ``grade.oracle_provenance`` records how an external checker was
      invoked -- a command line, an image reference, a registry URL.

    This is parametrised over the whole set rather than written per field so
    that a future field added to any of these models is a conscious decision
    about redaction rather than a silent omission.
    """
    token = "hf_" + "z" * 30
    at = datetime(2026, 8, 4, tzinfo=UTC)
    run = EvalRunResult(
        run_id="run-001",
        manifest=EvalRunManifest(
            run_name="n",
            dataset_ref=DatasetRef(provider="p", dataset_id="d"),
            adapter="a",
            grader="g",
            target_name="t",
        ),
        resolved_dataset=ResolvedDataset(dataset_id="d", revision="v1"),
        samples=(
            SampleResult(
                sample=EvalSample(
                    sample_id="s0",
                    input={"prompt": f"use {token} to authenticate"},
                    reference=f"reference {token}",
                    metadata={"note": f"meta {token}"},
                    expected_artifacts={"spec": f"artifact {token}"},
                    source_digest="sha256:x",
                    adapter="a",
                ),
                execution=NormalizedExecutionResult(
                    sample_id="s0",
                    attempt=1,
                    output={"answer": f"out {token}"},
                    tool_calls=({"name": "http", "args": f"header {token}"},),
                    environment_metadata={"conn": f"env {token}"},
                    status=ExecutionStatus.COMPLETED,
                    started_at=at,
                    finished_at=at,
                ),
                grade=GradeResult(
                    sample_id="s0",
                    grader="g",
                    status=GradeStatus.PASS,
                    evidence={"reason": f"ev {token}"},
                    oracle_provenance={"cmd": f"docker run --token {token}"},
                    created_at=at,
                ),
            ),
        ),
        started_at=at,
    )

    redacted = apply_redaction(run, DEFAULT_REDACTION_POLICY).samples[0]

    leaked = [
        name
        for name, value in (
            ("sample.input", redacted.sample.input),
            ("sample.reference", redacted.sample.reference),
            ("sample.metadata", redacted.sample.metadata),
            ("sample.expected_artifacts", redacted.sample.expected_artifacts),
            ("execution.output", redacted.execution.output),
            ("execution.tool_calls", redacted.execution.tool_calls),
            ("execution.environment_metadata", redacted.execution.environment_metadata),
            ("grade.evidence", redacted.grade.evidence),
            ("grade.oracle_provenance", redacted.grade.oracle_provenance),
        )
        if token in repr(value)
    ]
    assert leaked == [], f"secret survived redaction in: {leaked}"

    # tool_calls must still be a tuple afterwards: every collection on a wire
    # model is a tuple by contract (ADR-0002), and the generic JSON sweep
    # turns any sequence into a list, so this field needs its own pass.
    assert isinstance(redacted.execution.tool_calls, tuple)
