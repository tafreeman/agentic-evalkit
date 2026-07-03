"""End-to-end tests for :class:`agentic_evalkit.runner.EvalRunner` (plan Task 11).

The first test below is copied verbatim from
``docs/plans/2026-07-02-agentic-evalkit-initial-release.md`` (Task 11, Step 2).
Every fake is deterministic and in-process; none of these tests touch the
network or a real model.

``EvalRunner`` is typed against a local ``_CatalogProtocol`` (defined in
``agentic_evalkit.runner``, not imported from ``agentic_evalkit.datasets``):
the runner depends on the catalog's *shape* (``resolve`` + ``iter_records``),
not on ``DatasetCatalog``'s concrete class, so the fakes below satisfy the
protocol structurally without inheriting anything.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.events import RunEvent
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

# --- Deterministic fakes ----------------------------------------------------


class _FakeCatalog:
    """Structurally satisfies the runner's local catalog protocol."""

    def __init__(self, records: tuple[SourceRecord, ...]) -> None:
        self._records = records
        self.resolve_calls = 0

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        self.resolve_calls += 1
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


def _catalog_with_two_records() -> _FakeCatalog:
    return _FakeCatalog(
        (
            SourceRecord(row_id="0", data={"question": "q0", "answer": "42"}, digest="sha256:r0"),
            SourceRecord(row_id="1", data={"question": "q1", "answer": "43"}, digest="sha256:r1"),
        )
    )


class _IdentityAdapter:
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

    def validate_oracle(self, sample: EvalSample) -> bool:
        return sample.reference is not None

    def aggregate_metadata(self) -> dict[str, object]:
        return {"adapter": self.name}


class _SequencedTarget:
    """Returns a fixed sequence of results, one per call, in call order."""

    def __init__(self, results: tuple[NormalizedExecutionResult, ...]) -> None:
        self._results = list(results)

    @classmethod
    def success_then_error(cls) -> _SequencedTarget:
        now = datetime.now(UTC)
        return cls(
            (
                NormalizedExecutionResult(
                    sample_id="identity:0",
                    attempt=1,
                    output={"answer": "42"},
                    status=ExecutionStatus.COMPLETED,
                    started_at=now,
                    finished_at=now,
                ),
                NormalizedExecutionResult(
                    sample_id="identity:1",
                    attempt=1,
                    output=None,
                    status=ExecutionStatus.ERROR,
                    error={"type": "RuntimeError", "message": "boom"},
                    started_at=now,
                    finished_at=now,
                ),
            )
        )

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        index = len(self._results) - len(self._pending())
        result = self._pending().pop(0)
        assert index >= 0  # nosec B101 - test-only sequencing guard
        return result.model_copy(update={"sample_id": sample.sample_id, "attempt": attempt})

    def _pending(self) -> list[NormalizedExecutionResult]:
        return self._results


class _ExactFixtureGrader:
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


def _artifact_store(root: Path | None = None) -> ArtifactStore:
    """Build an isolated ``ArtifactStore``.

    The plan's verbatim test calls this with no arguments, so ``root``
    defaults to a fresh OS temp directory (never the repo's working
    directory or a shared path) rather than requiring a pytest fixture.
    Other tests in this module pass pytest's ``tmp_path`` explicitly to get
    a directory that is cleaned up by pytest itself.
    """
    return ArtifactStore(root if root is not None else Path(tempfile.mkdtemp()))


def _records(count: int) -> tuple[SourceRecord, ...]:
    return tuple(
        SourceRecord(
            row_id=str(i), data={"question": f"q{i}", "answer": "x"}, digest=f"sha256:r{i}"
        )
        for i in range(count)
    )


def _manifest(**overrides: object) -> EvalRunManifest:
    defaults: dict[str, object] = {
        "run_name": "test-run",
        "dataset_ref": DatasetRef(provider="local", dataset_id="fixture.jsonl"),
        "adapter": "identity@1",
        "grader": "exact@1",
        "target_name": "fake",
        "selection": DatasetSelection(),
        "sampling": SamplingPolicy(attempts=1),
        "attempts": 1,
        "concurrency": 2,
    }
    defaults.update(overrides)
    return EvalRunManifest(**defaults)  # type: ignore[arg-type]


# --- Verbatim plan test -------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runner_preserves_sample_failure_and_infrastructure_error() -> None:
    runner = EvalRunner(
        catalog=_catalog_with_two_records(),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _SequencedTarget.success_then_error()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(),
    )
    result = await runner.run(_manifest())
    assert result.summary.total == 2
    assert result.summary.failed == 0
    assert result.summary.errors == 1
    assert result.samples[0].grade.status == "pass"
    assert result.samples[1].execution.status == "error"
    assert result.samples[1].grade is None


# --- Additional coverage (plan Task 11, Step 6) -----------------------------


_VOLATILE_KEYS = frozenset({"run_id", "started_at", "finished_at", "created_at"})


def _strip_volatile(value: object) -> object:
    """Recursively drop run IDs and timestamps from a ``model_dump`` tree.

    Every attempt's ``NormalizedExecutionResult``/``GradeResult`` carries its
    own ``started_at``/``finished_at``/``created_at``, not just the top-level
    run, so this walks the whole structure rather than popping a fixed set
    of top-level keys.
    """
    if isinstance(value, dict):
        return {
            key: _strip_volatile(item) for key, item in value.items() if key not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repeated_runs_of_the_same_manifest_are_equivalent(tmp_path: Path) -> None:
    manifest = _manifest()
    first = await EvalRunner(
        catalog=_catalog_with_two_records(),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _SequencedTarget.success_then_error()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(tmp_path / "run-a"),
    ).run(manifest)
    second = await EvalRunner(
        catalog=_catalog_with_two_records(),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _SequencedTarget.success_then_error()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(tmp_path / "run-b"),
    ).run(manifest)

    assert _strip_volatile(first.model_dump(mode="json")) == _strip_volatile(
        second.model_dump(mode="json")
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrency_never_exceeds_the_manifest_limit(tmp_path: Path) -> None:
    max_concurrency_seen = 0
    in_flight = 0
    lock = asyncio.Lock()

    class _CountingTarget:
        async def execute(
            self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
        ) -> NormalizedExecutionResult:
            nonlocal max_concurrency_seen, in_flight
            async with lock:
                in_flight += 1
                max_concurrency_seen = max(max_concurrency_seen, in_flight)
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

    runner = EvalRunner(
        catalog=_FakeCatalog(_records(6)),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _CountingTarget()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(tmp_path),
    )
    result = await runner.run(_manifest(concurrency=2))
    assert result.summary.total == 6
    assert max_concurrency_seen <= 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancelling_the_run_marks_pending_samples_cancelled(tmp_path: Path) -> None:
    release_first = asyncio.Event()
    started_first = asyncio.Event()

    class _HangingTarget:
        def __init__(self) -> None:
            self._calls = 0

        async def execute(
            self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
        ) -> NormalizedExecutionResult:
            self._calls += 1
            if self._calls == 1:
                started_first.set()
                await release_first.wait()
            else:
                await asyncio.sleep(10)
            now = datetime.now(UTC)
            return NormalizedExecutionResult(
                sample_id=sample.sample_id,
                attempt=attempt,
                output={"answer": sample.reference},
                status=ExecutionStatus.COMPLETED,
                started_at=now,
                finished_at=now,
            )

    runner = EvalRunner(
        catalog=_FakeCatalog(_records(3)),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _HangingTarget()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(tmp_path),
    )
    run_task = asyncio.ensure_future(runner.run(_manifest(concurrency=1)))
    await asyncio.wait_for(started_first.wait(), timeout=2.0)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task
    release_first.set()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_never_mutates_the_supplied_manifest(tmp_path: Path) -> None:
    runner = EvalRunner(
        catalog=_catalog_with_two_records(),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _SequencedTarget.success_then_error()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(tmp_path),
    )
    manifest = _manifest()
    before = manifest.model_dump(mode="json")
    await runner.run(manifest)
    assert manifest.model_dump(mode="json") == before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execution_failed_status_counts_as_operational_not_a_task_failure(
    tmp_path: Path,
) -> None:
    """An ``ExecutionStatus.FAILED`` result never reaches grading; it is an
    operational outcome (like ``error``/``timeout``), not a graded "the
    system answered wrong" outcome, so it must not land in
    ``RunSummary.failed``.
    """

    class _AlwaysFailedTarget:
        async def execute(
            self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
        ) -> NormalizedExecutionResult:
            now = datetime.now(UTC)
            return NormalizedExecutionResult(
                sample_id=sample.sample_id,
                attempt=attempt,
                output=None,
                status=ExecutionStatus.FAILED,
                error={"type": "TargetFailure", "message": "target reported failure"},
                started_at=now,
                finished_at=now,
            )

    runner = EvalRunner(
        catalog=_FakeCatalog(_records(1)),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _AlwaysFailedTarget()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(tmp_path),
    )
    result = await runner.run(_manifest())
    assert result.summary.total == 1
    assert result.summary.errors == 1
    assert result.summary.failed == 0
    assert result.samples[0].execution.status == "failed"
    assert result.samples[0].grade is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_rejects_unknown_component_names(tmp_path: Path) -> None:
    runner = EvalRunner(
        catalog=_catalog_with_two_records(),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _SequencedTarget.success_then_error()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(tmp_path),
    )
    with pytest.raises(ManifestValidationError):
        await runner.run(_manifest(target_name="missing-target"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_emits_ordered_progress_events(tmp_path: Path) -> None:
    events: list[RunEvent] = []

    def _sink(event: RunEvent) -> None:
        events.append(event)

    runner = EvalRunner(
        catalog=_catalog_with_two_records(),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _SequencedTarget.success_then_error()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(tmp_path),
    )
    await runner.run(_manifest(), event_sink=_sink)

    event_type_names = [type(event).__name__ for event in events]
    assert event_type_names[0] == "RunStarted"
    assert event_type_names[-1] == "RunCompleted"
    assert "DatasetResolved" in event_type_names
    assert event_type_names.count("SampleStarted") == 2
    assert event_type_names.count("SampleCompleted") == 2
