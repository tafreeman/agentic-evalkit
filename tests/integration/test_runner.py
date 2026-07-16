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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from agentic_evalkit.artifacts import ArtifactRef, ArtifactStore
from agentic_evalkit.benchmarks.harness import FakeHarnessExecutor, HarnessResult, HarnessStatus
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.events import RunEvent, RunFailed
from agentic_evalkit.graders.harness import HarnessGrader
from agentic_evalkit.graders.judge import JudgeGrader, JudgeRequest, JudgeResponse
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
from agentic_evalkit.reporters.base import RedactionPolicy
from agentic_evalkit.runner import EvalRunner

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import JsonValue

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
        result = self._results.pop(0)
        return result.model_copy(update={"sample_id": sample.sample_id, "attempt": attempt})


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


#: Keeps each no-arg ``_artifact_store`` ``TemporaryDirectory`` alive for the
#: whole session; its exit finalizer then removes it, unlike a bare
#: ``mkdtemp()``, which left one orphaned OS temp directory per session.
_NO_ARG_STORE_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _artifact_store(root: Path | None = None) -> ArtifactStore:
    """Build an isolated ``ArtifactStore``.

    The plan's verbatim test calls this with no arguments, so ``root``
    defaults to a fresh OS temp directory (never the repo's working
    directory or a shared path) rather than requiring a pytest fixture; the
    directory is registered in ``_NO_ARG_STORE_DIRS`` so it outlives the
    test but is still deleted at interpreter exit. Other tests in this
    module pass pytest's ``tmp_path`` explicitly to get a directory that is
    cleaned up by pytest itself.
    """
    if root is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="evalkit-artifact-store-")
        _NO_ARG_STORE_DIRS.append(temp_dir)
        root = Path(temp_dir.name)
    return ArtifactStore(root)


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
    """Asserts the EXACT full event-type sequence, not just counts and endpoints.

    ``concurrency=1`` makes the two samples' sub-sequences run strictly one
    after another rather than interleaved, so the full sequence -- not just
    each sample's internal sub-sequence -- is deterministic. The fixture
    target (``success_then_error``) grades sample 0 (``COMPLETED`` ->
    ``GradeCompleted`` fires) and does not grade sample 1 (``ERROR`` ->
    requirement 6 skips grading), which is why ``GradeCompleted`` appears
    only once even though there are two samples.
    """
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
    await runner.run(_manifest(concurrency=1), event_sink=_sink)

    event_type_names = [type(event).__name__ for event in events]
    assert event_type_names == [
        "RunStarted",
        "DatasetResolved",
        "SampleStarted",
        "ExecutionCompleted",
        "GradeCompleted",
        "SampleCompleted",
        "SampleStarted",
        "ExecutionCompleted",
        "SampleCompleted",
        "RunCompleted",
    ]

    run_id = events[0].run_id
    assert all(event.run_id == run_id for event in events)


# --- RunFailed emission (defect 1) ------------------------------------------


class _ResolveRaisesCatalog:
    """Structurally satisfies the runner's catalog protocol; ``resolve`` always fails.

    ``iter_records`` is never expected to be called -- the runner must abort
    before reaching sample preparation -- so it raises if it ever is, making
    a wrongly-ordered runner change fail loudly rather than silently pass.
    """

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        raise RuntimeError("dataset provider unreachable")

    async def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        raise AssertionError("iter_records must not be called after resolve() failed")
        yield  # pragma: no cover - unreachable; makes this an async generator


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dataset_resolution_failure_emits_exactly_one_run_failed(tmp_path: Path) -> None:
    """A catalog whose ``resolve`` raises is an infrastructure-level abort.

    Exactly one ``RunFailed`` is emitted (naming the original exception's
    type), no ``RunCompleted`` follows it, and the original exception -- not
    a wrapped or replaced one -- is what the caller observes.
    """
    events: list[RunEvent] = []

    def _sink(event: RunEvent) -> None:
        events.append(event)

    runner = EvalRunner(
        catalog=_ResolveRaisesCatalog(),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _SequencedTarget.success_then_error()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(tmp_path),
    )
    with pytest.raises(RuntimeError, match="dataset provider unreachable"):
        await runner.run(_manifest(), event_sink=_sink)

    event_type_names = [type(event).__name__ for event in events]
    assert event_type_names == ["RunStarted", "RunFailed"]
    assert "RunCompleted" not in event_type_names

    run_failed = events[-1]
    assert isinstance(run_failed, RunFailed)
    assert run_failed.error_type == "RuntimeError"
    assert run_failed.message == "dataset provider unreachable"
    assert run_failed.run_id == events[0].run_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancellation_during_the_run_emits_exactly_one_run_failed(tmp_path: Path) -> None:
    """``asyncio.CancelledError`` is itself an infrastructure-level abort.

    Mirrors ``test_cancelling_the_run_marks_pending_samples_cancelled``'s
    hanging-target fixture, but additionally asserts the ``RunFailed`` event
    this defect fix adds: cancellation must not end the run with no
    terminal event at all.
    """
    started_first = asyncio.Event()
    events: list[RunEvent] = []

    def _sink(event: RunEvent) -> None:
        events.append(event)

    class _HangsForeverTarget:
        async def execute(
            self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
        ) -> NormalizedExecutionResult:
            started_first.set()
            await asyncio.sleep(10)
            raise AssertionError("unreachable - the run is cancelled before this sleep returns")

    runner = EvalRunner(
        catalog=_FakeCatalog(_records(1)),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _HangsForeverTarget()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(tmp_path),
    )
    run_task = asyncio.ensure_future(runner.run(_manifest(concurrency=1), event_sink=_sink))
    await asyncio.wait_for(started_first.wait(), timeout=2.0)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    event_type_names = [type(event).__name__ for event in events]
    assert "RunFailed" in event_type_names
    assert "RunCompleted" not in event_type_names
    run_failed = next(event for event in events if isinstance(event, RunFailed))
    assert run_failed.error_type == "CancelledError"


# --- Redacted spill (defect 2) -----------------------------------------------


class _PlantedTokenTarget:
    """Returns one execution whose output is large enough to spill and
    contains a planted fake Hugging Face token, so a redaction policy that
    matches ``hf_...`` tokens has something real to catch.
    """

    def __init__(self, *, token: str, padding_chars: int) -> None:
        self._token = token
        self._padding_chars = padding_chars

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        now = datetime.now(UTC)
        padding = "x" * self._padding_chars
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output={"answer": sample.reference, "log": f"token={self._token} {padding}"},
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            finished_at=now,
        )


_PLANTED_TOKEN = "hf_AbCdEfGh0123456789"
#: Enough filler so the serialized output clears ``_LARGE_OUTPUT_THRESHOLD_BYTES``
#: (8192 bytes) and is guaranteed to spill regardless of the token's own length.
_SPILL_PADDING_CHARS = 8300


@pytest.mark.integration
@pytest.mark.asyncio
async def test_spill_redacts_a_planted_secret_when_a_policy_is_supplied(tmp_path: Path) -> None:
    policy = RedactionPolicy(secret_patterns=(r"hf_[A-Za-z0-9]{16,}",))
    artifact_store = _artifact_store(tmp_path)
    runner = EvalRunner(
        catalog=_FakeCatalog(_records(1)),
        adapters={"identity@1": _IdentityAdapter()},
        targets={
            "fake": _PlantedTokenTarget(token=_PLANTED_TOKEN, padding_chars=_SPILL_PADDING_CHARS)
        },
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=artifact_store,
        redaction_policy=policy,
    )
    result = await runner.run(_manifest())

    execution = result.samples[0].execution
    assert execution.output is None  # spilled, not left inline
    digest = execution.artifacts["output_ref"]
    assert isinstance(digest, str)

    ref = ArtifactRef(digest=digest, media_type="application/json", byte_count=0)
    payload = artifact_store.read(ref).decode("utf-8")
    metadata = artifact_store.metadata(ref)

    assert _PLANTED_TOKEN not in payload
    assert "[REDACTED]" in payload
    assert metadata.redacted is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_spill_redacts_a_planted_secret_by_default(tmp_path: Path) -> None:
    """With no explicit ``redaction_policy``, the runner now defaults to
    ``DEFAULT_REDACTION_POLICY`` (Story 2.1 / R-002): the planted ``hf_``
    token is stripped from the spilled artifact and the artifact is recorded
    as redacted, so a real run never spills raw secrets to disk.
    """
    artifact_store = _artifact_store(tmp_path)
    runner = EvalRunner(
        catalog=_FakeCatalog(_records(1)),
        adapters={"identity@1": _IdentityAdapter()},
        targets={
            "fake": _PlantedTokenTarget(token=_PLANTED_TOKEN, padding_chars=_SPILL_PADDING_CHARS)
        },
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=artifact_store,
    )
    result = await runner.run(_manifest())

    execution = result.samples[0].execution
    assert execution.output is None
    digest = execution.artifacts["output_ref"]
    assert isinstance(digest, str)

    ref = ArtifactRef(digest=digest, media_type="application/json", byte_count=0)
    payload = artifact_store.read(ref).decode("utf-8")
    metadata = artifact_store.metadata(ref)

    assert _PLANTED_TOKEN not in payload
    assert "[REDACTED]" in payload
    assert metadata.redacted is True


# --- Grade before spill (defect 3) -------------------------------------------
#
# ``_execute_and_grade`` used to spill *before* grading, so any grader
# handling an execution large enough to spill was handed ``output=None``
# instead of the real content (ADR-0017). These tests prove the fix through
# the real ``EvalRunner.run`` path: a grader must see the full, intact
# output at grade time, and the runner must still spill that same execution
# for storage afterwards -- spilling moved, it did not disappear.


class _OutputCapturingGrader:
    """Records the exact ``execution.output`` it was handed at grade time.

    The captured value is asserted on directly (not just the grade status),
    so this proves the grader saw real content rather than merely that it
    didn't error.
    """

    def __init__(self) -> None:
        self.seen_outputs: list[dict[str, JsonValue] | None] = []

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        now = datetime.now(UTC)
        self.seen_outputs.append(execution.output)
        return GradeResult(
            sample_id=sample.sample_id,
            grader="output-capturing@1",
            status=GradeStatus.PASS,
            score=1.0,
            evidence={"observed_output_was_none": execution.output is None},
            created_at=now,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_grader_sees_the_full_output_before_it_is_spilled_for_storage(
    tmp_path: Path,
) -> None:
    """A large execution (the planted-token/padding technique from the spill
    tests above) must still be graded against its real, intact output -- not
    an ``output=None`` spill placeholder -- and must still end up spilled in
    the persisted result, exactly as before the fix.
    """
    artifact_store = _artifact_store(tmp_path)
    grader = _OutputCapturingGrader()
    runner = EvalRunner(
        catalog=_FakeCatalog(_records(1)),
        adapters={"identity@1": _IdentityAdapter()},
        targets={
            "fake": _PlantedTokenTarget(token=_PLANTED_TOKEN, padding_chars=_SPILL_PADDING_CHARS)
        },
        graders={"exact@1": grader},
        artifact_store=artifact_store,
    )
    result = await runner.run(_manifest())

    # The grader was handed the full, intact output: the planted token is
    # present in what it captured, not a spill placeholder.
    assert len(grader.seen_outputs) == 1
    seen_output = grader.seen_outputs[0]
    assert seen_output is not None
    assert _PLANTED_TOKEN in str(seen_output)

    # The grade reflects real content, not a spill-placeholder error.
    grade = result.samples[0].grade
    assert grade is not None
    assert grade.status == GradeStatus.PASS
    assert grade.evidence["observed_output_was_none"] is False

    # Spilling still happens for storage, just after grading: the FINAL
    # persisted execution is spilled, same assertion shape as the redaction
    # tests above.
    execution = result.samples[0].execution
    assert execution.output is None
    digest = execution.artifacts["output_ref"]
    assert isinstance(digest, str)


class _CapturingHarnessPredictor:
    """A ``HarnessPredictor`` that records the ``execution.output`` it saw.

    ``HarnessGrader`` itself has no seam to observe from outside, so the
    injected predictor callable -- which ``HarnessGrader.grade`` always
    invokes with the same ``(sample, execution)`` it received -- is what
    proves what the grader actually saw.
    """

    def __init__(self) -> None:
        self.seen_outputs: list[dict[str, JsonValue] | None] = []

    def __call__(
        self, sample: EvalSample, execution: NormalizedExecutionResult
    ) -> dict[str, JsonValue]:
        self.seen_outputs.append(execution.output)
        output = execution.output or {}
        return {
            "instance_id": sample.sample_id,
            "model_name_or_path": "test-model",
            "model_patch": output.get("model_patch", ""),
        }


class _LargePatchTarget:
    """Returns one execution whose output is a SWE-bench-shaped patch large
    enough to spill -- ``_PlantedTokenTarget``'s padding technique, under the
    ``model_patch`` key ``HarnessGrader``'s predictor actually reads.
    """

    def __init__(self, *, marker: str, padding_chars: int) -> None:
        self._marker = marker
        self._padding_chars = padding_chars

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        now = datetime.now(UTC)
        padding = "+" * self._padding_chars
        patch = f"--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+{self._marker}{padding}\n"
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output={"model_patch": patch},
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            finished_at=now,
        )


_PATCH_MARKER = "planted-fix-marker"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_harness_grader_sees_the_full_patch_before_it_is_spilled(tmp_path: Path) -> None:
    """The grader that actually motivated ADR-0017: a SWE-bench patch large
    enough to spill must still reach ``HarnessGrader`` (via its predictor)
    intact, and must grade to a real, hard-gated verdict -- not the
    defensive spilled-output ERROR path.
    """
    artifact_store = _artifact_store(tmp_path)
    predictor = _CapturingHarnessPredictor()
    harness_grader = HarnessGrader(
        executor=FakeHarnessExecutor(
            default_result=HarnessResult(
                status=HarnessStatus.COMPLETED, resolved=True, message="ok"
            )
        ),
        predictor=predictor,
        benchmark="swebench-verified@1",
        name="swebench-harness@1",
    )
    runner = EvalRunner(
        catalog=_FakeCatalog(_records(1)),
        adapters={"identity@1": _IdentityAdapter()},
        targets={
            "fake": _LargePatchTarget(marker=_PATCH_MARKER, padding_chars=_SPILL_PADDING_CHARS)
        },
        graders={"swebench-harness@1": harness_grader},
        artifact_store=artifact_store,
    )
    result = await runner.run(_manifest(grader="swebench-harness@1"))

    # The predictor -- and therefore HarnessGrader -- saw the full, intact
    # patch, not a spill placeholder.
    assert len(predictor.seen_outputs) == 1
    seen_output = predictor.seen_outputs[0]
    assert seen_output is not None
    assert _PATCH_MARKER in str(seen_output)

    # A real, earned, hard-gated verdict -- not the spilled-output ERROR path.
    grade = result.samples[0].grade
    assert grade is not None
    assert grade.status == GradeStatus.PASS
    assert grade.hard_gate is True
    assert "spilled" not in str(grade.evidence).lower()

    # Spilling still happens for storage, just after grading.
    execution = result.samples[0].execution
    assert execution.output is None
    digest = execution.artifacts["output_ref"]
    assert isinstance(digest, str)


# --- Judge transport isolation (ADR-0020) ------------------------------------


class _AlwaysCompletedTarget:
    """Returns a COMPLETED execution for every sample, so grading always runs.

    Unlike ``_SequencedTarget.success_then_error``, both samples reach the
    grader here -- exactly what the judge-transport-isolation test below needs,
    since a non-completed execution would skip grading (requirement 6) and mask
    whether the judge itself was reached.
    """

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        now = datetime.now(UTC)
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output={"answer": sample.reference},
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            finished_at=now,
        )


class _RaisingOnOneSampleJudge:
    """A ``JudgeClient`` whose transport raises for exactly one ``sample_id``.

    Proves ADR-0020's transport isolation end to end through ``EvalRunner``:
    one raising judge yields one graded ERROR sample, and the run finishes
    normally rather than aborting with a ``RunFailed``. Every other sample gets
    a clean, parseable (advisory) verdict.
    """

    fingerprint = "judge:model:prompt"

    def __init__(self, *, raising_sample_id: str) -> None:
        self._raising_sample_id = raising_sample_id

    async def judge(self, request: JudgeRequest) -> JudgeResponse:
        if request.sample_id == self._raising_sample_id:
            raise RuntimeError("judge provider unreachable")
        return JudgeResponse(
            fingerprint=self.fingerprint,
            verdict="pass",
            score=0.9,
            parse_ok=True,
            abstained=False,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_completes_when_the_judge_raises_on_one_sample(tmp_path: Path) -> None:
    """A ``JudgeClient`` that raises on one sample must not abort the whole run
    (ADR-0020): the run finishes, the affected sample is graded ERROR carrying
    ``judge_transport_error`` evidence, the other sample grades normally, and no
    ``RunFailed`` is emitted.
    """
    events: list[RunEvent] = []

    def _sink(event: RunEvent) -> None:
        events.append(event)

    judge_grader = JudgeGrader(
        _RaisingOnOneSampleJudge(raising_sample_id="identity:1"),
        calibration=None,
        gate=False,
    )
    runner = EvalRunner(
        catalog=_catalog_with_two_records(),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _AlwaysCompletedTarget()},
        graders={"judge@1": judge_grader},
        artifact_store=_artifact_store(tmp_path),
    )
    result = await runner.run(_manifest(grader="judge@1"), event_sink=_sink)

    assert result.summary.total == 2
    graded_by_id = {sample.sample.sample_id: sample.grade for sample in result.samples}

    errored = graded_by_id["identity:1"]
    assert errored is not None
    assert errored.status is GradeStatus.ERROR
    assert errored.hard_gate is False
    assert errored.evidence["judge_transport_error"] == "RuntimeError"

    other = graded_by_id["identity:0"]
    assert other is not None
    assert other.status is GradeStatus.PASS  # advisory verdict, never gating
    assert other.hard_gate is False

    event_names = [type(event).__name__ for event in events]
    assert "RunCompleted" in event_names
    assert "RunFailed" not in event_names
