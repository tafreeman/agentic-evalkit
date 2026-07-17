"""End-to-end tests for :class:`agentic_evalkit.runner.EvalRunner` (plan Task 11).

The first test below is copied exactly, word for word, from
``docs/plans/2026-07-02-agentic-evalkit-initial-release.md`` (Task 11, Step
2) -- it was part of the original spec for this feature. Every fake object
used in this file behaves the same way every time it's called (no real
randomness, no network, no real AI model) -- these tests only exercise
in-process, hand-written stand-ins.

``EvalRunner`` doesn't depend on the real dataset catalog class. Instead,
its type hints reference a small local ``_CatalogProtocol`` (defined in
``agentic_evalkit.runner``, not imported from ``agentic_evalkit.datasets``)
-- a description of the two methods (``resolve`` and ``iter_records``) that
any catalog-like object must have. Because of this, the fake catalog
classes below can satisfy the runner just by having those two methods,
without needing to inherit from ``DatasetCatalog`` or any other real class.
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

# --- Fakes: hand-written stand-ins that always behave the same way ----------


class _FakeCatalog:
    """A fake dataset catalog: it has the same ``resolve``/``iter_records``
    methods the runner's ``_CatalogProtocol`` expects, so the runner accepts
    it in place of a real catalog, without this class needing to inherit
    from anything."""

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


#: Holds onto every ``TemporaryDirectory`` created by calling
#: ``_artifact_store()`` with no arguments, so Python doesn't garbage-collect
#: (and thereby delete) the directory before the test finishes using it.
#: When the test process exits, each ``TemporaryDirectory`` cleans itself up
#: automatically. Before this list existed, using the lower-level
#: ``mkdtemp()`` function instead left one leftover, never-deleted temp
#: directory on disk behind every time the test suite ran.
_NO_ARG_STORE_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _artifact_store(root: Path | None = None) -> ArtifactStore:
    """Build a fresh, isolated ``ArtifactStore`` for a test to use.

    The test copied word-for-word from the plan document calls this
    function with no arguments at all, so ``root`` needs a sensible default
    -- it falls back to a brand-new OS-level temp directory (never this
    repo's working directory or any shared path) instead of requiring every
    caller to pass in a pytest ``tmp_path`` fixture. That fallback directory
    is tracked in ``_NO_ARG_STORE_DIRS`` so it survives for the whole test,
    and only gets deleted when the test process exits. Other tests in this
    file instead pass pytest's own ``tmp_path`` fixture directly, which
    pytest already cleans up on its own.
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


# --- Test copied exactly from the original plan document --------------------


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
    """Walk a dict/list tree (as produced by Pydantic's ``model_dump``) and
    remove every run ID and timestamp field from it, at every level -- not
    just the top level.

    Every individual attempt has its own ``NormalizedExecutionResult`` and
    ``GradeResult``, each carrying its own ``started_at``/``finished_at``/
    ``created_at`` fields -- not only the top-level run has these fields. So
    this function recurses into the whole structure instead of only
    removing a fixed set of keys from the top.
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
    """When the system being tested reports ``ExecutionStatus.FAILED`` (it
    broke while running), that result never even reaches the grading step.
    ``FAILED`` -- just like ``error`` or ``timeout`` -- means something went
    wrong with running the system itself, which is a completely different
    situation from a grader looking at a real answer and deciding it's
    wrong. So a ``FAILED`` result must never be counted in
    ``RunSummary.failed``, which is reserved specifically for "the system
    ran fine, but its answer was wrong."
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
    """Checks the exact, full list of event types emitted, in order -- not
    just how many events fired or which ones fired first and last.

    Setting ``concurrency=1`` forces the two samples to run one completely
    after the other instead of overlapping, so their events can't get
    interleaved -- which means the entire event sequence is predictable, not
    just the handful of events belonging to any one sample. The fake target
    used here (``success_then_error``) makes sample 0 succeed (status
    ``COMPLETED``, so a ``GradeCompleted`` event fires for it) and makes
    sample 1 come back as ``ERROR``. Since an ``ERROR`` execution skips
    grading entirely (see requirement 6 in the runner), sample 1 never
    produces its own ``GradeCompleted`` event -- which is exactly why the
    list below has only one ``GradeCompleted``, even though there are two
    samples total.
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


# --- the RunFailed event fires correctly (defect 1) --------------------------


class _ResolveRaisesCatalog:
    """A fake catalog that has the two methods the runner needs, but whose
    ``resolve`` method always raises an error instead of returning data.

    This test expects the runner to give up immediately when ``resolve``
    fails, before it ever gets to preparing samples -- so ``iter_records``
    should never be called at all. To make sure of that, ``iter_records``
    itself raises an error if it's ever called. That way, if a future change
    to the runner accidentally called it anyway (running things in the
    wrong order), this test would fail loudly and obviously, instead of
    quietly passing when it shouldn't.
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
    """If the dataset catalog's ``resolve`` method raises an error, that
    counts as our own infrastructure breaking, not the AI system under test
    giving a wrong answer.

    This test checks that exactly one ``RunFailed`` event fires (recording
    the original exception's type by name), that no ``RunCompleted`` event
    ever follows it, and that the exception the caller actually sees is the
    original one -- not some wrapped or replaced substitute.
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
    """Being cancelled (``asyncio.CancelledError``) counts as our own
    infrastructure aborting the run, the same as any other unexpected error.

    This test uses the same "hanging target" trick as
    ``test_cancelling_the_run_marks_pending_samples_cancelled`` above, but
    additionally checks the ``RunFailed`` event that this bug fix added:
    cancelling a run must never leave it with no final event at all --
    there must always be some event recording how the run ended.
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


# --- secrets get blanked out ("redacted") before a big output gets spilled --
# --- to disk (defect 2) -------------------------------------------------------


class _PlantedTokenTarget:
    """A fake target whose output is deliberately both (1) large enough that
    the runner will "spill" it out to a separate file instead of keeping it
    inline, and (2) contains a fake Hugging Face access token planted inside
    it -- giving a redaction policy that looks for ``hf_...``-shaped tokens
    something real to actually find and blank out.
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
#: The number of filler characters to add so the output, once serialized, is
#: bigger than ``_LARGE_OUTPUT_THRESHOLD_BYTES`` (8192 bytes) -- the size
#: limit that triggers a spill to disk. This guarantees the output spills no
#: matter how long the planted token itself happens to be.
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
    """If the caller doesn't explicitly pass a ``redaction_policy``, the
    runner now falls back to ``DEFAULT_REDACTION_POLICY`` automatically
    (tracked as Story 2.1 / R-002). This test checks that, even with no
    policy explicitly given, the planted ``hf_`` token still gets stripped
    out of the spilled file, and the file gets marked as redacted -- so a
    real run can never accidentally spill a raw secret to disk just because
    nobody remembered to configure a redaction policy.
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


# --- grading happens before spilling now (defect 3) --------------------------
#
# ``_execute_and_grade`` used to spill the output to disk *before* grading
# ran. That meant any output big enough to spill would be replaced with
# ``output=None`` before the grader ever got to look at it -- so the grader
# ended up judging an empty placeholder instead of the AI system's real
# answer (fixed per ADR-0017). The tests below prove the fix by running the
# real ``EvalRunner.run`` path end to end: a grader must see the full, real
# output at the moment it grades, and the runner must still spill that same
# output afterwards for storage. In other words, spilling still happens --
# it just moved to occur after grading instead of before.


class _OutputCapturingGrader:
    """A fake grader that just records the exact ``execution.output`` value
    it was given each time it's asked to grade something.

    The test checks this recorded value directly (not just whether grading
    succeeded without an error), which is what actually proves the grader
    saw the AI system's real answer -- not just that nothing crashed.
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
    """Even for an output large enough to spill (using the same
    planted-token-plus-padding trick as the spill tests above), grading must
    still see the real, complete output -- not the ``output=None``
    placeholder that a spilled result gets. And the final saved result must
    still end up spilled, exactly as it did before this fix -- only the
    order relative to grading changed.
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

    # Spilling still happens for storage, just after grading now: the FINAL
    # saved execution result is spilled, checked the same way as in the
    # redaction tests above.
    execution = result.samples[0].execution
    assert execution.output is None
    digest = execution.artifacts["output_ref"]
    assert isinstance(digest, str)


class _CapturingHarnessPredictor:
    """A fake ``HarnessPredictor`` that just records the ``execution.output``
    value it was given.

    There's no direct way to peek inside ``HarnessGrader`` from a test to
    see what it received. But ``HarnessGrader.grade`` always calls its
    injected predictor callable with the exact same ``(sample, execution)``
    values it was itself given -- so recording what this fake predictor
    sees is how the test proves what the grader actually saw.
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
    """A fake target whose output looks like a real SWE-bench submission: a
    code patch (a diff describing a code change) big enough to trigger a
    spill to disk, using the same padding trick as ``_PlantedTokenTarget``
    above. The patch is stored under the ``model_patch`` key, which is the
    specific key that ``HarnessGrader``'s predictor actually reads from.
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
    """``HarnessGrader`` is the actual grader that motivated ADR-0017's fix
    in the first place. This test checks that a SWE-bench code patch large
    enough to spill still reaches ``HarnessGrader`` (through its predictor)
    completely intact, and still earns a real, hard-gated (able to block a
    release) verdict -- not the fallback ``ERROR`` verdict that would
    result if the grader had only seen an already-spilled, emptied-out
    placeholder.
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

    # A real, earned, hard-gated verdict here -- not the fallback ERROR that
    # would happen if the output had been spilled before grading.
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


# --- one judge-model failure must not take down the whole run (ADR-0020) ----


class _AlwaysCompletedTarget:
    """A fake target that always reports ``COMPLETED``, for every sample --
    so every sample always makes it to grading.

    Unlike ``_SequencedTarget.success_then_error`` (which deliberately fails
    one sample), both samples here reach the grader. That's exactly what the
    test below needs: if an execution weren't ``COMPLETED``, grading would
    be skipped entirely (per requirement 6), which would hide whether the
    judge model itself was ever actually reached.
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
    """A fake judge-model client that raises an error -- as if the network
    call to the judge model had failed -- for exactly one specific
    ``sample_id``, and behaves normally for every other sample.

    This proves, end to end through the real ``EvalRunner``, that
    ADR-0020's fix actually works: when the judge model fails for one
    sample, only that one sample gets graded as ``ERROR`` -- the run as a
    whole still finishes normally instead of aborting with a ``RunFailed``
    event. Every other sample still gets a normal, successfully-parsed
    verdict (advisory only, meaning it's purely informational and can't
    block a release).
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
    """When the judge-model client raises an error for just one sample, that
    must not abort the entire run (ADR-0020). Instead: the run still
    finishes, the one affected sample is graded ``ERROR`` and records
    evidence showing it was a ``judge_transport_error`` (a failure in
    calling the judge, not a real verdict), the other sample is graded
    normally, and no ``RunFailed`` event ever fires.
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
    assert other.status is GradeStatus.PASS  # informational only, can't gate a release
    assert other.hard_gate is False

    event_names = [type(event).__name__ for event in events]
    assert "RunCompleted" in event_names
    assert "RunFailed" not in event_names
