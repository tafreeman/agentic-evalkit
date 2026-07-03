"""Shared fixtures for reporter tests.

All reporter tests (JSON, JSONL, Markdown, HTML) exercise the same frozen
three-sample :class:`~agentic_evalkit.models.EvalRunResult` so that every
reporter is proven against one identical, provenance-carrying run.
"""

from datetime import UTC, datetime

from agentic_evalkit.models import (
    DatasetRef,
    DatasetSelection,
    EvalRunManifest,
    EvalRunResult,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    RunSummary,
    SampleResult,
    SamplingPolicy,
)

_STARTED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 7, 2, 12, 5, 0, tzinfo=UTC)


def _sample(sample_id: str) -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        input={"question": f"question for {sample_id}"},
        reference="42",
        source_digest=f"sha256:{sample_id}",
        adapter="gsm8k@1",
    )


def _execution(
    sample_id: str, *, status: ExecutionStatus, output: dict[str, object] | None = None
) -> NormalizedExecutionResult:
    return NormalizedExecutionResult(
        sample_id=sample_id,
        attempt=1,
        output=output,
        status=status,
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _grade(sample_id: str) -> GradeResult:
    return GradeResult(
        sample_id=sample_id,
        grader="normalized-exact@1",
        status=GradeStatus.PASS,
        score=1.0,
        hard_gate=False,
        evidence={"expected": "42", "actual": "42"},
        created_at=_FINISHED_AT,
    )


def _run_with_pass_error_timeout_and_provenance() -> EvalRunResult:
    """A frozen three-sample run: one pass, one error, one timeout.

    ``resolved_dataset.revision`` is pinned to ``"abc"`` so every reporter
    test can assert on a stable, known provenance value.
    """
    passed = SampleResult(
        sample=_sample("gsm8k:main:test:0"),
        execution=_execution(
            "gsm8k:main:test:0",
            status=ExecutionStatus.COMPLETED,
            output={"answer": "42"},
        ),
        grade=_grade("gsm8k:main:test:0"),
    )
    errored = SampleResult(
        sample=_sample("gsm8k:main:test:1"),
        execution=_execution("gsm8k:main:test:1", status=ExecutionStatus.ERROR),
        grade=None,
    )
    timed_out = SampleResult(
        sample=_sample("gsm8k:main:test:2"),
        execution=_execution("gsm8k:main:test:2", status=ExecutionStatus.TIMEOUT),
        grade=None,
    )
    manifest = EvalRunManifest(
        run_name="gsm8k-smoke",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="echo-target",
        selection=DatasetSelection(offset=0, limit=3),
        sampling=SamplingPolicy(seed=7, attempts=1),
        attempts=1,
        timeout_seconds=30.0,
        concurrency=1,
        environment_fingerprint="env:sha256:deadbeef",
        code_fingerprint="code:sha256:cafef00d",
    )
    resolved_dataset = ResolvedDataset(
        dataset_id="openai/gsm8k",
        revision="abc",
        config="main",
        split="test",
        row_count=3,
        retrieved_at=_STARTED_AT,
    )
    return EvalRunResult(
        run_id="run-001",
        manifest=manifest,
        resolved_dataset=resolved_dataset,
        samples=(passed, errored, timed_out),
        summary=RunSummary(total=3, passed=1, failed=0, errors=1, timeouts=1),
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )
