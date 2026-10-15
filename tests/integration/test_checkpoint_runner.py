"""A cancelled runner preserves completed work and requires an explicit retry."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.checkpoints import RunCheckpoint
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.models import (
    DatasetRef,
    EvalRunManifest,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    SourceRecord,
)
from agentic_evalkit.runner import EvalRunner

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


class _Catalog:
    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        return ResolvedDataset(dataset_id=ref.dataset_id, revision="sha256:fixture")

    async def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        records = (
            SourceRecord(row_id="one", data={"answer": "one"}, digest="sha256:one"),
            SourceRecord(row_id="two", data={"answer": "two"}, digest="sha256:two"),
        )
        for record in records[offset:limit]:
            yield record


class _Adapter:
    def prepare(self, record: SourceRecord) -> EvalSample:
        return EvalSample(
            sample_id=record.row_id,
            input={},
            reference=record.data["answer"],
            source_digest=record.digest,
            adapter="fixture",
        )


class _Grader:
    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        return GradeResult(
            sample_id=sample.sample_id,
            grader="fixture",
            status=GradeStatus.PASS,
            score=1.0,
            created_at=datetime.now(UTC),
        )


class _Target:
    def __init__(self) -> None:
        self.first_complete = asyncio.Event()
        self.second_started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[str] = []

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        self.calls.append(sample.sample_id)
        if sample.sample_id == "two":
            self.second_started.set()
            await self.release.wait()
        now = datetime.now(UTC)
        result = NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            status=ExecutionStatus.COMPLETED,
            output={"answer": sample.reference},
            started_at=now,
            finished_at=now,
        )
        if sample.sample_id == "one":
            self.first_complete.set()
        return result


def _manifest() -> EvalRunManifest:
    return EvalRunManifest(
        run_name="checkpoint",
        dataset_ref=DatasetRef(provider="local", dataset_id="fixture.jsonl"),
        adapter="fixture",
        grader="fixture",
        target_name="fixture",
        concurrency=1,
    )


@pytest.mark.asyncio
async def test_cancelled_run_does_not_replay_completed_attempts_without_opt_in(
    tmp_path: Path,
) -> None:
    target = _Target()
    runner = EvalRunner(
        catalog=_Catalog(),
        adapters={"fixture": _Adapter()},
        targets={"fixture": target},
        graders={"fixture": _Grader()},
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
    )
    checkpoint_dir = tmp_path / "checkpoint"
    with RunCheckpoint(checkpoint_dir) as checkpoint:
        task = asyncio.create_task(runner.run(_manifest(), checkpoint=checkpoint))
        await asyncio.wait_for(target.second_started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert set(checkpoint.completed()) == {("one", 1)}

    with (
        RunCheckpoint(checkpoint_dir, resume=True) as checkpoint,
        pytest.raises(ManifestValidationError, match="interrupted"),
    ):
        checkpoint.begin(
            _manifest(),
            ResolvedDataset(dataset_id="fixture.jsonl", revision="sha256:fixture"),
            (
                _Adapter().prepare(
                    SourceRecord(row_id="one", data={"answer": "one"}, digest="sha256:one")
                ),
                _Adapter().prepare(
                    SourceRecord(row_id="two", data={"answer": "two"}, digest="sha256:two")
                ),
            ),
        )
