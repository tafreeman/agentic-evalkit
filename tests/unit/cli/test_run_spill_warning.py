"""The count behind the ``run`` command's dropped-output warning.

A sample whose oversized output the artifact store refused keeps the status
and the grade it genuinely earned, so it appears in the ``outcomes`` line as
an ordinary pass or fail and changes no count. That is the right accounting
-- a storage failure is not a task failure and not an operational one -- but
it means a run that quietly lost some of its evidence would otherwise print
as completely clean. ``_failed_spill_count`` is what lets the command say so
out loud instead.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from agentic_evalkit.cli.runs import _failed_spill_count
from agentic_evalkit.models import (
    DatasetRef,
    DatasetSelection,
    EvalRunManifest,
    EvalRunResult,
    EvalSample,
    ExecutionStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    RunSummary,
    SampleResult,
    SamplingPolicy,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

_AT = datetime(2026, 7, 30, 12, 0, 0, tzinfo=UTC)

_SPILL_RECORD: JsonValue = {
    "type": "ArtifactStoreLimitExceeded",
    "code": "output_spill_failed",
    "message": "artifact payload exceeds the configured maximum",
}


def _sample_result(sample_id: str, *, artifacts: dict[str, JsonValue]) -> SampleResult:
    return SampleResult(
        sample=EvalSample(
            sample_id=sample_id,
            input={"question": "q"},
            reference="42",
            source_digest=f"sha256:{sample_id}",
            adapter="identity@1",
        ),
        execution=NormalizedExecutionResult(
            sample_id=sample_id,
            attempt=1,
            output=None,
            artifacts=artifacts,
            status=ExecutionStatus.COMPLETED,
            started_at=_AT,
            finished_at=_AT,
        ),
        grade=None,
    )


def _run(*samples: SampleResult) -> EvalRunResult:
    return EvalRunResult(
        run_id="run-1",
        manifest=EvalRunManifest(
            run_name="spill-warning",
            dataset_ref=DatasetRef(provider="local", dataset_id="fixtures/tiny"),
            adapter="identity@1",
            grader="exact@1",
            target_name="fake",
            selection=DatasetSelection(offset=0, limit=len(samples)),
            sampling=SamplingPolicy(seed=1, attempts=1),
            attempts=1,
            timeout_seconds=30.0,
            concurrency=1,
        ),
        resolved_dataset=ResolvedDataset(
            dataset_id="fixtures/tiny",
            revision="abc",
            config=None,
            split="test",
            row_count=len(samples),
        ),
        samples=samples,
        summary=RunSummary(total=len(samples)),
        started_at=_AT,
        finished_at=_AT,
    )


def test_a_run_with_no_failed_spills_counts_zero() -> None:
    """Nothing must be warned about on the ordinary path -- including for a
    sample whose output spilled successfully, which also has ``output=None``
    but is not a loss: the bytes are on disk under ``output_ref``."""
    run = _run(
        _sample_result("identity:0", artifacts={}),
        _sample_result("identity:1", artifacts={"output_ref": "sha256:" + "ab" * 32}),
    )

    assert _failed_spill_count(run) == 0


def test_every_sample_whose_output_could_not_be_stored_is_counted() -> None:
    """The count is per sample, not per run: a store that has stopped
    accepting writes fails every oversized sample it sees, and the warning
    has to say how many were lost, not merely that some were."""
    run = _run(
        _sample_result("identity:0", artifacts={"output_spill_error": _SPILL_RECORD}),
        _sample_result("identity:1", artifacts={"output_ref": "sha256:" + "cd" * 32}),
        _sample_result("identity:2", artifacts={"output_spill_error": _SPILL_RECORD}),
    )

    assert _failed_spill_count(run) == 2


def test_repeated_attempts_at_one_sample_are_counted_once() -> None:
    """``result.samples`` holds one entry per *attempt*, not per sample. A run
    configured with ``attempts=3`` against a store that has stopped accepting
    writes produces three records for the same sample -- and the warning,
    whose entire job is to size the loss accurately, must say one output was
    lost, not three.
    """
    run = _run(
        _sample_result("identity:0", artifacts={"output_spill_error": _SPILL_RECORD}),
        _sample_result("identity:0", artifacts={"output_spill_error": _SPILL_RECORD}),
        _sample_result("identity:0", artifacts={"output_spill_error": _SPILL_RECORD}),
    )

    assert _failed_spill_count(run) == 1


def test_a_recorded_failure_that_kept_its_output_inline_is_not_counted() -> None:
    """A spill-failure record does not by itself mean anything was lost.

    ``_spill_failure_result`` records the failure whenever the spill boundary
    raises, but drops the output only when it was genuinely oversized. A
    boundary that fails *before* the size check -- a malformed
    ``secret_patterns``, which is a per-runner setting and so fails every
    sample in the run -- leaves the record beside an output still sitting in
    the report. Counting those would make the warning claim a total loss of
    evidence on a run that lost none: the exact overstatement this count
    exists to avoid, at maximum scale.
    """
    kept = _sample_result("identity:0", artifacts={"output_spill_error": _SPILL_RECORD})
    kept = kept.model_copy(
        update={"execution": kept.execution.model_copy(update={"output": {"answer": "42"}})}
    )
    run = _run(
        kept,
        _sample_result("identity:1", artifacts={"output_spill_error": _SPILL_RECORD}),
    )

    assert _failed_spill_count(run) == 1


def test_a_target_writing_the_key_itself_is_not_counted() -> None:
    """``artifacts`` is target-controlled, so the key alone means nothing. A
    target that keeps its own upload diagnostics under that name must not
    make a healthy run print a warning about evidence it never lost -- only a
    record carrying the taxonomy code the runner writes is counted.
    """
    run = _run(
        _sample_result("identity:0", artifacts={"output_spill_error": "my own note"}),
        _sample_result("identity:1", artifacts={"output_spill_error": {"note": "no code here"}}),
        _sample_result("identity:2", artifacts={"output_spill_error": {"code": "something_else"}}),
    )

    assert _failed_spill_count(run) == 0
