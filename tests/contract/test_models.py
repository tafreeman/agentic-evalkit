from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentic_evalkit.graders.judge import (
    CalibrationArtifact,
    JudgeResponse,
    JudgeResponseStatus,
)
from agentic_evalkit.models import (
    ContaminationMetadata,
    ContaminationStatus,
    DatasetRef,
    DatasetSelection,
    EvalRunManifest,
    EvalRunResult,
    EvalSample,
    ExecutionRequest,
    ExecutionStatus,
    GradeResult,
    GraderSpec,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    RunSummary,
    SamplePage,
    SampleResult,
    SamplingPolicy,
    SearchHit,
    SearchPage,
    SourceRecord,
)


def test_models_are_frozen_and_forbid_unknown_fields() -> None:
    ref = DatasetRef(provider="huggingface", dataset_id="openai/gsm8k")
    with pytest.raises(ValidationError):
        DatasetRef(provider="huggingface", dataset_id="openai/gsm8k", unknown=True)
    with pytest.raises(ValidationError):
        ref.dataset_id = "other/dataset"  # type: ignore[misc]


def test_sample_round_trips_through_versioned_json() -> None:
    sample = EvalSample(
        sample_id="gsm8k:main:test:0",
        input={"question": "1+1?"},
        reference="2",
        source_digest="sha256:abc",
        adapter="gsm8k@1",
    )
    assert EvalSample.model_validate_json(sample.model_dump_json()) == sample


def test_grade_status_is_not_collapsed_to_boolean() -> None:
    grade = GradeResult(
        sample_id="s1",
        grader="exact@1",
        status=GradeStatus.ABSTAIN,
        score=None,
        hard_gate=False,
        created_at=datetime.now(UTC),
    )
    assert grade.status is GradeStatus.ABSTAIN


# --- Additional round-trip serialization coverage for every public model ---
# "Round-trip" means: build an object, serialize it to JSON with
# model_dump_json(), parse that JSON back with model_validate_json(), and
# check that you get back an object equal to the one you started with.
# That proves the JSON form doesn't quietly drop or corrupt any data.
# Pattern: construct -> model_dump_json -> model_validate_json -> equality.


def test_dataset_ref_round_trips() -> None:
    ref = DatasetRef(
        provider="huggingface",
        dataset_id="princeton-nlp/SWE-bench_Verified",
        revision="abc123",
        config="default",
        split="test",
        data_files=("data/train-00000-of-00001.parquet",),
        selection="row_idx < 10",
        field_mapping={"question": "prompt"},
        allow_remote_code=False,
    )
    assert DatasetRef.model_validate_json(ref.model_dump_json()) == ref


def test_resolved_dataset_round_trips() -> None:
    resolved = ResolvedDataset(
        dataset_id="openai/gsm8k",
        revision="sha256:deadbeef",
        config="main",
        split="test",
        selected_files=("main/test-00000-of-00001.parquet",),
        schema_metadata={"question": "string", "answer": "string"},
        row_count=1319,
        license="MIT",
        citation="@misc{gsm8k}",
        gated=False,
        card_metadata={"pretty_name": "GSM8K"},
        retrieved_at=datetime.now(UTC),
        provider_response_digests={"is-valid": "sha256:aaa", "splits": "sha256:bbb"},
        cache_manifest_digest="sha256:ccc",
        checksums={"payload": "sha256:ddd"},
    )
    assert ResolvedDataset.model_validate_json(resolved.model_dump_json()) == resolved


def test_source_record_round_trips() -> None:
    record = SourceRecord(
        row_id="0",
        data={"question": "What is 1+1?", "answer": "2"},
        digest="sha256:row0",
    )
    assert SourceRecord.model_validate_json(record.model_dump_json()) == record


def test_search_hit_round_trips() -> None:
    hit = SearchHit(
        dataset_id="openai/gsm8k",
        provider="huggingface",
        revision="sha256:deadbeef",
        tags=("math", "reasoning"),
        gated=False,
        private=False,
        downloads=1000,
        card_metadata={"pretty_name": "GSM8K"},
    )
    assert SearchHit.model_validate_json(hit.model_dump_json()) == hit


def test_search_page_round_trips() -> None:
    page = SearchPage(
        hits=(
            SearchHit(
                dataset_id="openai/gsm8k",
                provider="huggingface",
                revision="sha256:deadbeef",
            ),
        ),
        cursor="next-cursor",
        total_hits=1,
    )
    assert SearchPage.model_validate_json(page.model_dump_json()) == page


def test_sample_page_round_trips() -> None:
    page = SamplePage(
        records=(
            SourceRecord(row_id="0", data={"question": "1+1?"}, digest="sha256:row0"),
            SourceRecord(row_id="1", data={"question": "2+2?"}, digest="sha256:row1"),
        ),
        offset=0,
        total_rows=2,
    )
    assert SamplePage.model_validate_json(page.model_dump_json()) == page


def test_grader_spec_round_trips() -> None:
    spec = GraderSpec(
        name="normalized-exact@1",
        grader_type="objective",
        parameters={"case_sensitive": False},
        hard_gate=True,
    )
    assert GraderSpec.model_validate_json(spec.model_dump_json()) == spec


def test_eval_sample_full_round_trips() -> None:
    sample = EvalSample(
        sample_id="gsm8k:main:test:0",
        input={"question": "1+1?"},
        reference="2",
        expected_artifacts={"answer_file": "answer.txt"},
        metadata={"difficulty": "easy"},
        tags=("math",),
        source_row_id="0",
        source_digest="sha256:abc",
        adapter="gsm8k@1",
        allowed_execution_policy={"max_attempts": 1},
        grader=GraderSpec(name="normalized-exact@1", hard_gate=True),
    )
    assert EvalSample.model_validate_json(sample.model_dump_json()) == sample


def test_execution_request_round_trips() -> None:
    request = ExecutionRequest(
        sample_id="s1",
        attempt=1,
        input={"question": "1+1?"},
        timeout_seconds=30.0,
        trace_id="trace-1",
    )
    assert ExecutionRequest.model_validate_json(request.model_dump_json()) == request


def test_normalized_execution_result_round_trips() -> None:
    now = datetime.now(UTC)
    result = NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output={"answer": "2"},
        structured_output={"value": 2},
        artifacts={"trace": "trace.json"},
        tool_calls=({"name": "calculator", "arguments": {"a": 1, "b": 1}},),
        trace_refs=("trace-1",),
        latency_ms=125.0,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        model_name="gpt-test",
        status=ExecutionStatus.COMPLETED,
        error=None,
        environment_metadata={"python": "3.11"},
        target_fingerprint="callable:echo:abc123",
        started_at=now,
        finished_at=now,
    )
    assert NormalizedExecutionResult.model_validate_json(result.model_dump_json()) == result


def test_grade_result_full_round_trips() -> None:
    grade = GradeResult(
        sample_id="s1",
        grader="exact@1",
        grader_type="objective",
        status=GradeStatus.PASS,
        score=1.0,
        hard_gate=True,
        evidence={"expected": "2", "actual": "2"},
        artifact_refs=("evidence.json",),
        rubric_id="rubric-1",
        oracle_provenance={"source": "gsm8k"},
        judge_calibration_ref="calibration-1",
        created_at=datetime.now(UTC),
    )
    assert GradeResult.model_validate_json(grade.model_dump_json()) == grade


def test_dataset_selection_round_trips() -> None:
    selection = DatasetSelection(offset=0, limit=100, filter="row_idx < 100")
    assert DatasetSelection.model_validate_json(selection.model_dump_json()) == selection


def test_sampling_policy_round_trips() -> None:
    policy = SamplingPolicy(seed=42, temperature=0.0, attempts=1)
    assert SamplingPolicy.model_validate_json(policy.model_dump_json()) == policy


def test_eval_run_manifest_round_trips() -> None:
    manifest = EvalRunManifest(
        run_name="gsm8k-quickstart",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        revision_policy="pinned",
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="echo",
        target_fingerprint_policy="required",
        target_fingerprint="sha256:target",
        selection=DatasetSelection(offset=0, limit=10),
        sampling=SamplingPolicy(seed=42, attempts=1),
        attempts=1,
        timeout_seconds=30.0,
        concurrency=4,
        artifact_policy={"store_traces": True},
        redaction_policy={"redact_headers": ["authorization"]},
        environment_fingerprint="sha256:env",
        code_fingerprint="sha256:code",
        baseline_compatibility_rules={"dataset_revision": "exact"},
    )
    assert EvalRunManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_sample_result_round_trips() -> None:
    now = datetime.now(UTC)
    sample_result = SampleResult(
        sample=EvalSample(
            sample_id="s1",
            input={"question": "1+1?"},
            source_digest="sha256:abc",
            adapter="gsm8k@1",
        ),
        execution=NormalizedExecutionResult(
            sample_id="s1",
            attempt=1,
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            finished_at=now,
        ),
        grade=GradeResult(
            sample_id="s1",
            grader="exact@1",
            status=GradeStatus.PASS,
            score=1.0,
            hard_gate=True,
            created_at=now,
        ),
    )
    assert SampleResult.model_validate_json(sample_result.model_dump_json()) == sample_result


def test_sample_result_allows_missing_grade() -> None:
    now = datetime.now(UTC)
    sample_result = SampleResult(
        sample=EvalSample(
            sample_id="s1",
            input={"question": "1+1?"},
            source_digest="sha256:abc",
            adapter="gsm8k@1",
        ),
        execution=NormalizedExecutionResult(
            sample_id="s1",
            attempt=1,
            status=ExecutionStatus.ERROR,
            started_at=now,
            finished_at=now,
        ),
        grade=None,
    )
    assert SampleResult.model_validate_json(sample_result.model_dump_json()) == sample_result


def test_eval_run_result_round_trips() -> None:
    now = datetime.now(UTC)
    manifest = EvalRunManifest(
        run_name="gsm8k-quickstart",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="echo",
    )
    sample_result = SampleResult(
        sample=EvalSample(
            sample_id="s1",
            input={"question": "1+1?"},
            source_digest="sha256:abc",
            adapter="gsm8k@1",
        ),
        execution=NormalizedExecutionResult(
            sample_id="s1",
            attempt=1,
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            finished_at=now,
        ),
        grade=GradeResult(
            sample_id="s1",
            grader="exact@1",
            status=GradeStatus.PASS,
            score=1.0,
            hard_gate=True,
            created_at=now,
        ),
    )
    run_result = EvalRunResult(
        run_id="run-1",
        manifest=manifest,
        resolved_dataset=ResolvedDataset(
            dataset_id="openai/gsm8k",
            revision="sha256:deadbeef",
            config="main",
            split="test",
        ),
        samples=(sample_result,),
        summary=RunSummary(total=1, passed=1),
        started_at=now,
        finished_at=now,
    )
    assert EvalRunResult.model_validate_json(run_result.model_dump_json()) == run_result


def test_eval_run_result_supports_appending_sample_results() -> None:
    """EvalRunResult must still support appending new results later on (for
    example, during a streaming run that adds samples as they finish)."""
    now = datetime.now(UTC)
    manifest = EvalRunManifest(
        run_name="gsm8k-quickstart",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="echo",
    )
    initial = EvalRunResult(
        run_id="run-1",
        manifest=manifest,
        resolved_dataset=ResolvedDataset(
            dataset_id="openai/gsm8k", revision="sha256:deadbeef", config="main", split="test"
        ),
        samples=(),
        summary=RunSummary(),
        started_at=now,
        finished_at=None,
    )
    new_sample_result = SampleResult(
        sample=EvalSample(
            sample_id="s1",
            input={"question": "1+1?"},
            source_digest="sha256:abc",
            adapter="gsm8k@1",
        ),
        execution=NormalizedExecutionResult(
            sample_id="s1",
            attempt=1,
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            finished_at=now,
        ),
        grade=None,
    )
    appended = initial.model_copy(
        update={
            "samples": (*initial.samples, new_sample_result),
            "summary": initial.summary.model_copy(update={"total": initial.summary.total + 1}),
        }
    )
    assert appended.samples == (new_sample_result,)
    assert appended.summary.total == 1
    # `initial` itself was never changed by the update above -- this is
    # what proves a new object was created instead of the original being
    # mutated in place (immutability, per ADR-0002).
    assert initial.samples == ()


# --- Contamination metadata (ADR-0013) ---------------------------------------
# "Contamination" means dataset rows that a model may have already seen
# during training, which would let it "cheat" on the eval by recognizing a
# memorized answer instead of actually solving the task. The tests below
# cover the metadata that records what's known about that risk for a given
# dataset.


def test_contamination_metadata_round_trips_inside_resolved_dataset() -> None:
    resolved = ResolvedDataset(
        dataset_id="local:./private.jsonl",
        revision="sha256:abc",
        contamination=ContaminationMetadata(
            status=ContaminationStatus.VERIFIED_CLEAN,
            authored_after=datetime.now(UTC),
            canary_ids=("TRIPWIRE-1",),
            held_out=True,
        ),
    )
    assert ResolvedDataset.model_validate_json(resolved.model_dump_json()) == resolved


def test_contamination_status_is_not_collapsed_to_boolean() -> None:
    metadata = ContaminationMetadata(status=ContaminationStatus.SUSPECT)
    restored = ContaminationMetadata.model_validate_json(metadata.model_dump_json())
    assert restored.status is ContaminationStatus.SUSPECT
    # The default status is UNKNOWN ("nobody has checked this yet"), kept
    # as a distinct value from VERIFIED_CLEAN ("checked, and it's fine") --
    # so "we don't know" can never be mistaken for "we know it's okay".
    assert ContaminationMetadata().status is ContaminationStatus.UNKNOWN


def test_held_out_with_a_public_release_date_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        ContaminationMetadata(held_out=True, public_since=datetime.now(UTC))


def test_authored_after_a_public_release_date_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        ContaminationMetadata(
            authored_after=datetime(2025, 1, 1, tzinfo=UTC),
            public_since=datetime(2024, 1, 1, tzinfo=UTC),
        )
    # It's fine (no error) when the data was authored on or before the
    # public release date -- that is the only logically consistent
    # ordering of the two dates.
    ok = ContaminationMetadata(
        authored_after=datetime(2024, 1, 1, tzinfo=UTC),
        public_since=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert ok.authored_after < ok.public_since  # type: ignore[operator]


# --- Judge response envelope + calibration coverage evidence (ADR-0020) ------
# An LLM judge's raw verdict ("pass"/"fail" plus a score) isn't enough on
# its own to know whether that verdict can be trusted. The "response
# envelope" wraps it with a status (did the judge answer normally, refuse,
# or hit an operational error?) and an optional rationale. "Calibration
# coverage evidence" is the record of how the judge's accuracy was
# measured -- e.g. how many labeled examples it was tested against -- which
# is what decides whether that judge is trustworthy enough to gate a
# release (see ADR-0007 and ADR-0020).


def test_judge_response_status_and_rationale_round_trip() -> None:
    response = JudgeResponse(
        fingerprint="judge:model:prompt",
        verdict="fail",
        score=0.1,
        parse_ok=True,
        abstained=False,
        status=JudgeResponseStatus.REFUSED,
        rationale="declined: the candidate output requested a policy override",
    )
    restored = JudgeResponse.model_validate_json(response.model_dump_json())
    assert restored == response
    assert restored.status is JudgeResponseStatus.REFUSED
    # New fields were added without removing or renaming any old ones
    # ("additive"), so the wire format is still schema_version "1" -- the
    # version marker other systems check to confirm this JSON shape is
    # still backward-compatible (ADR-0002).
    assert restored.schema_version == "1"


def test_judge_response_status_defaults_to_ok_and_is_not_collapsed_to_boolean() -> None:
    response = JudgeResponse(
        fingerprint="judge:model:prompt",
        verdict="pass",
        score=0.9,
        parse_ok=True,
        abstained=False,
    )
    restored = JudgeResponse.model_validate_json(response.model_dump_json())
    # A JudgeClient implementation written before this envelope existed,
    # and which therefore never sets status/rationale at all, must keep
    # behaving exactly as it did before: status defaults to OK and
    # rationale defaults to None. status is a named enum value
    # (JudgeResponseStatus.OK), never a plain True/False, so a future state
    # like REFUSED can never be confused with a boolean.
    assert restored.status is JudgeResponseStatus.OK
    assert restored.rationale is None


def test_calibration_artifact_coverage_fields_round_trip() -> None:
    artifact = CalibrationArtifact(
        calibration_id="cal-1",
        judge_fingerprint="judge:model:prompt",
        expires_at=datetime.now(UTC),
        true_positive=95,
        true_negative=97,
        false_positive=3,
        false_negative=5,
        threshold=0.7,
        total_labeled=200,
        abstained_count=4,
        error_count=1,
    )
    restored = CalibrationArtifact.model_validate_json(artifact.model_dump_json())
    assert restored == artifact
    assert restored.total_labeled == 200
    assert restored.abstained_count == 4
    assert restored.error_count == 1


def test_calibration_artifact_coverage_fields_default_to_none() -> None:
    artifact = CalibrationArtifact(
        calibration_id="cal-2",
        judge_fingerprint="judge:model:prompt",
        expires_at=datetime.now(UTC),
        true_positive=95,
        true_negative=97,
        false_positive=3,
        false_negative=5,
        threshold=0.7,
    )
    restored = CalibrationArtifact.model_validate_json(artifact.model_dump_json())
    assert restored == artifact
    # These fields are optional and were added later without breaking
    # existing data ("additive"). When they're left out entirely, they
    # come back as None ("we don't have this number"), never silently as 0
    # ("we counted zero and got nothing") -- a real difference this test
    # protects.
    assert restored.total_labeled is None
    assert restored.abstained_count is None
    assert restored.error_count is None
