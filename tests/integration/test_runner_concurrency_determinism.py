"""Story 5.2 (R-007): multi-attempt x concurrency determinism.

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 5, Story 5.2) and
``agentic_evalkit.runner``.

The AC combines ``attempts>1`` with ``concurrency>1`` -- a shape the existing
``tests/integration/test_runner.py`` concurrency/cancellation tests (all
``attempts=1``) never cover. ``EvalRunner`` builds a sample-major,
attempt-minor task plan and collects into a pre-sized list indexed by task
order, so results must come back in that fixed order regardless of which task
finishes first; the in-flight count must never exceed
``Semaphore(concurrency)``; and a cancel mid-run must leave no task still
running.

This lives in its own module (rather than extending ``test_runner.py``, which
is already near the file-size cap) with self-contained fakes, so it neither
grows that file past the limit nor collides with its module-scoped helpers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.models import (
    DatasetRef,
    DatasetSelection,
    EvalRunManifest,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    SamplingPolicy,
    SourceRecord,
)
from agentic_evalkit.runner import EvalRunner
from agentic_evalkit.targets.base import ExecutionTarget

# --- Self-contained deterministic fakes -------------------------------------


class _DeterminismCatalog:
    """Structurally satisfies the runner's local catalog protocol."""

    def __init__(self, records: tuple[SourceRecord, ...]) -> None:
        self._records = records

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        return ResolvedDataset(
            dataset_id=ref.dataset_id,
            revision="sha256:" + "0" * 64,
            config=ref.config,
            split=ref.split,
            row_count=len(self._records),
        )

    async def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        end = len(self._records) if limit is None else offset + limit
        for record in self._records[offset:end]:
            yield record


class _DeterminismAdapter:
    api_version = "1"
    name = "identity@1"

    def prepare(self, record: SourceRecord) -> EvalSample:
        question = record.data["question"]
        assert isinstance(question, str)
        answer = record.data.get("answer")
        reference = answer if isinstance(answer, str) else None
        return EvalSample(
            sample_id=f"identity:{record.row_id}",
            input={"question": question},
            reference=reference,
            source_row_id=record.row_id,
            source_digest=record.digest,
            adapter=self.name,
        )


class _DeterminismGrader:
    """Grades a completed execution by exact match against the reference."""

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        now = datetime.now(UTC)
        output = execution.output
        answer = output.get("answer") if output else None
        is_match = answer is not None and answer == sample.reference
        return GradeResult(
            sample_id=sample.sample_id,
            grader="exact@1",
            status=GradeStatus.PASS if is_match else GradeStatus.FAIL,
            score=1.0 if is_match else 0.0,
            created_at=now,
        )


def _determinism_records(count: int) -> tuple[SourceRecord, ...]:
    return tuple(
        SourceRecord(
            row_id=str(i), data={"question": f"q{i}", "answer": "x"}, digest=f"sha256:r{i}"
        )
        for i in range(count)
    )


def _determinism_manifest(*, concurrency: int, attempts: int) -> EvalRunManifest:
    return EvalRunManifest(
        run_name="determinism-run",
        dataset_ref=DatasetRef(provider="local", dataset_id="fixture.jsonl"),
        adapter="identity@1",
        grader="exact@1",
        target_name="fake",
        selection=DatasetSelection(),
        sampling=SamplingPolicy(attempts=attempts),
        attempts=attempts,
        concurrency=concurrency,
    )


def _determinism_runner(*, records: int, target: ExecutionTarget, tmp_path: Path) -> EvalRunner:
    return EvalRunner(
        catalog=_DeterminismCatalog(_determinism_records(records)),
        adapters={"identity@1": _DeterminismAdapter()},
        targets={"fake": target},
        graders={"exact@1": _DeterminismGrader()},
        artifact_store=ArtifactStore(tmp_path),
    )


# --- Tests ------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_attempt_results_are_ordered_by_task_not_completion(
    tmp_path: Path,
) -> None:
    """With ``attempts>1`` and ``concurrency>1``, results are indexed by task
    order (sample-major, attempt-minor), even when a target deliberately
    finishes them out of order. The pre-sized result list must place each
    (sample, attempt) at its task-order slot, not its completion-order slot.
    """

    class _ScrambledOrderTarget:
        """Sleeps by a delay keyed to (sample, attempt) so completion order is
        deliberately unrelated to task order: sample 0 finishes *after*
        sample 1, and later attempts finish before earlier ones. A
        completion-order collector would visibly misorder the results; the
        runner's pre-sized, task-indexed list must not.
        """

        async def execute(
            self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
        ) -> NormalizedExecutionResult:
            # sample 1 is faster than sample 0; within a sample, later attempts
            # are faster than earlier ones -- the reverse of task order.
            sample_rank = 0 if sample.sample_id == "identity:1" else 1
            delay = 0.01 + 0.02 * sample_rank + 0.005 * (3 - attempt)
            await asyncio.sleep(delay)
            now = datetime.now(UTC)
            return NormalizedExecutionResult(
                sample_id=sample.sample_id,
                attempt=attempt,
                output={"answer": sample.reference},
                status=ExecutionStatus.COMPLETED,
                started_at=now,
                finished_at=now,
            )

    runner = _determinism_runner(records=2, target=_ScrambledOrderTarget(), tmp_path=tmp_path)
    result = await runner.run(_determinism_manifest(concurrency=3, attempts=3))

    # 2 samples x 3 attempts = 6 results, in sample-major, attempt-minor order.
    assert result.summary.total == 6
    observed = [(item.sample.sample_id, item.execution.attempt) for item in result.samples]
    assert observed == [
        ("identity:0", 1),
        ("identity:0", 2),
        ("identity:0", 3),
        ("identity:1", 1),
        ("identity:1", 2),
        ("identity:1", 3),
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_attempt_in_flight_count_never_exceeds_the_semaphore(
    tmp_path: Path,
) -> None:
    """A watermark instrument proves the semaphore bounds concurrency even
    when the task count (samples x attempts) far exceeds the limit.
    """
    max_in_flight = 0
    in_flight = 0
    lock = asyncio.Lock()

    class _WatermarkTarget:
        async def execute(
            self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
        ) -> NormalizedExecutionResult:
            nonlocal max_in_flight, in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            now = datetime.now(UTC)
            return NormalizedExecutionResult(
                sample_id=sample.sample_id,
                attempt=attempt,
                output={"answer": sample.reference},
                status=ExecutionStatus.COMPLETED,
                started_at=now,
                finished_at=now,
            )

    runner = _determinism_runner(records=4, target=_WatermarkTarget(), tmp_path=tmp_path)
    # 4 samples x 3 attempts = 12 tasks, but concurrency is capped at 2.
    result = await runner.run(_determinism_manifest(concurrency=2, attempts=3))
    assert result.summary.total == 12
    assert max_in_flight <= 2
    assert max_in_flight >= 2  # the limit is actually reached, not trivially under


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancelling_a_multi_attempt_run_leaves_no_task_running(
    tmp_path: Path,
) -> None:
    """Cancelling a run with ``attempts>1`` and ``concurrency>1`` mid-flight
    propagates ``CancelledError`` and leaves no started attempt still running:
    every task that began either finished or observed cancellation. Guards
    against orphaned background tasks under the multi-attempt task plan.
    """
    started = 0
    finished = 0
    cancelled_observed = 0
    lock = asyncio.Lock()
    first_started = asyncio.Event()

    class _BlockingTarget:
        async def execute(
            self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
        ) -> NormalizedExecutionResult:
            nonlocal started, finished, cancelled_observed
            async with lock:
                started += 1
            first_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                async with lock:
                    cancelled_observed += 1
                raise
            async with lock:
                finished += 1
            now = datetime.now(UTC)
            return NormalizedExecutionResult(
                sample_id=sample.sample_id,
                attempt=attempt,
                output={"answer": sample.reference},
                status=ExecutionStatus.COMPLETED,
                started_at=now,
                finished_at=now,
            )

    runner = _determinism_runner(records=3, target=_BlockingTarget(), tmp_path=tmp_path)
    run_task = asyncio.ensure_future(runner.run(_determinism_manifest(concurrency=2, attempts=2)))
    await asyncio.wait_for(first_started.wait(), timeout=2.0)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    # Every task that started must have either finished or observed the
    # cancellation -- none left running as an orphan. A brief yield lets the
    # TaskGroup finish unwinding its children before we read the counters.
    await asyncio.sleep(0)
    assert started == finished + cancelled_observed
    assert finished == 0  # the 10s sleep never completes before cancellation
