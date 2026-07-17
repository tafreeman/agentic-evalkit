"""Story 5.2 (R-007): proving that results still come back in a predictable
order even when a run has multiple attempts per sample AND runs several
things at once (concurrency).

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 5, Story 5.2) and
``agentic_evalkit.runner``.

The acceptance criteria for this story require testing ``attempts>1``
together with ``concurrency>1`` at the same time -- a combination the
existing tests in ``tests/integration/test_runner.py`` never cover (their
concurrency and cancellation tests all use ``attempts=1``, i.e. only one
attempt per sample).

Here is the guarantee being tested: ``EvalRunner`` decides, up front, on a
fixed order for every (sample, attempt) pair to run in -- ordered first by
sample, then by attempt number within that sample (called "sample-major,
attempt-minor" order) -- and reserves one slot per pair in a list sized to
fit all of them from the start. Because each result is written into its own
pre-assigned slot, the final list always comes back in that same fixed
order, no matter which (sample, attempt) pair's task actually finishes first
in real time. On top of that: the number of attempts actually running at
the same time must never exceed the ``concurrency`` limit (enforced by an
``asyncio.Semaphore``, a counter that blocks new tasks from starting once
the limit is reached) -- and if the run is cancelled partway through, no
task is allowed to keep running in the background afterward.

This lives in its own file rather than being added to ``test_runner.py``
(which is already close to this project's maximum file-length guideline),
so it doesn't push that file over the limit, and so its own helper
classes/functions -- defined at module level and reused across its tests --
can't accidentally clash by name with the similarly-named helpers already
defined at module level in ``test_runner.py``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from agentic_evalkit.targets.base import ExecutionTarget

# --- Fakes: hand-written stand-ins used only in this file -------------------


class _DeterminismCatalog:
    """A fake dataset catalog with the ``resolve``/``iter_records`` methods
    the runner's catalog protocol expects, so it can stand in for a real
    catalog in these tests."""

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
    """When there is more than one attempt per sample AND more than one thing
    can run at once, results must still come back ordered the way they were
    planned -- first by sample, then by attempt number within that sample --
    even when the fake target below deliberately finishes them in a
    scrambled, different order. The runner's result list has one slot
    reserved per (sample, attempt) pair from the start, at a fixed position,
    so each result lands in its planned slot, not wherever it would land
    based on when it actually finished.
    """
    # Records each (sample_id, attempt) at the moment its execution returns.
    # All tasks run on one event loop, so a plain-list append (no await mid-
    # append) is safe without a lock.
    completion_order: list[tuple[str, int]] = []

    class _ScrambledOrderTarget:
        """A fake target that waits a different amount of time depending on
        which (sample, attempt) pair it's handling, chosen specifically so
        the order things actually finish in has nothing to do with the order
        they were planned in -- within one sample, later attempts are made
        to finish before earlier ones. Simply collecting results in the
        order they finish would give them back scrambled; the runner's
        result list, since each attempt writes into its own pre-assigned
        slot, must not be scrambled. Each call also records when it actually
        finished, so the test can directly prove the scramble really
        happened, instead of just assuming it did.
        """

        async def execute(
            self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
        ) -> NormalizedExecutionResult:
            # Sample "identity:1" finishes faster than sample "identity:0";
            # and within a single sample, later attempts finish faster than
            # earlier ones. Both of these are the reverse of task order.
            sample_rank = 0 if sample.sample_id == "identity:1" else 1
            delay = 0.01 + 0.02 * sample_rank + 0.005 * (3 - attempt)
            await asyncio.sleep(delay)
            completion_order.append((sample.sample_id, attempt))
            now = datetime.now(UTC)
            return NormalizedExecutionResult(
                sample_id=sample.sample_id,
                attempt=attempt,
                output={"answer": sample.reference},
                status=ExecutionStatus.COMPLETED,
                started_at=now,
                finished_at=now,
            )

    submission_order = [
        ("identity:0", 1),
        ("identity:0", 2),
        ("identity:0", 3),
        ("identity:1", 1),
        ("identity:1", 2),
        ("identity:1", 3),
    ]

    runner = _determinism_runner(records=2, target=_ScrambledOrderTarget(), tmp_path=tmp_path)
    result = await runner.run(_determinism_manifest(concurrency=3, attempts=3))

    # 2 samples x 3 attempts = 6 results total, ordered by sample first and
    # attempt second (sample-major, attempt-minor order).
    assert result.summary.total == 6
    observed = [(item.sample.sample_id, item.execution.attempt) for item in result.samples]
    assert observed == submission_order

    # This proves the scramble genuinely happened, rather than the in-order
    # result above being a coincidence: every planned task did finish
    # (`sorted(completion_order) == sorted(submission_order)`), just not in
    # the order it was submitted (`completion_order != submission_order`).
    # Exactly what order things finish in, past sample 0's attempts, depends
    # on semaphore timing and can vary a little between runs, so instead of
    # asserting one fixed completion sequence (which could be flaky --
    # inconsistently fail from run to run), we only assert this weaker fact,
    # which is always true and is still enough to prove the scramble
    # happened.
    assert sorted(completion_order) == sorted(submission_order)
    assert completion_order != submission_order


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_attempt_in_flight_count_never_exceeds_the_semaphore(
    tmp_path: Path,
) -> None:
    """Uses a "high-water mark" counter -- one that remembers the highest
    value it ever reached, not just its current value -- to prove that the
    semaphore actually keeps concurrency within the configured limit, even
    when the total number of tasks (samples times attempts) is much larger
    than that limit.
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
    """Checks what happens if you cancel a run partway through when it has
    more than one attempt per sample and more than one thing running at
    once: the cancellation (``asyncio.CancelledError``) must propagate out
    normally, and every attempt that had already started must end up either
    finished or having noticed the cancellation -- none should be left
    running in the background, abandoned, after the run itself has already
    returned or raised.
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
    try:
        # If waiting for `first_started` takes longer than its 2-second
        # budget (which can happen on a slow CI machine) and raises
        # `TimeoutError`, the `finally` block below still runs: it cancels
        # and cleans up `run_task` so this test can never leave a task
        # running in the background, and so a stray "pending task" warning
        # during teardown never hides the real `TimeoutError` that actually
        # failed the test.
        await asyncio.wait_for(first_started.wait(), timeout=2.0)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
    finally:
        run_task.cancel()
        # This swallows the CancelledError (and anything else raised during
        # cleanup) on purpose -- we don't care about the exception here, only
        # about making sure the task has actually stopped. The real
        # pass/fail verdict comes from the assertions below, not from this
        # cleanup step.
        await asyncio.gather(run_task, return_exceptions=True)

    # Every task that started must have ended up either finished normally or
    # having caught the cancellation -- none left behind still running. The
    # `asyncio.sleep(0)` below just yields control briefly, giving the
    # TaskGroup (the object that runs and tracks all the child tasks) a
    # chance to finish shutting down its children before we check the
    # counters.
    await asyncio.sleep(0)
    assert started == finished + cancelled_observed
    assert finished == 0  # the 10s sleep never completes before cancellation
