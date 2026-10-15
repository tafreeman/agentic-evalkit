"""Crash recovery must preserve completed evidence and reject changed inputs."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentic_evalkit.checkpoints import RunCheckpoint
from agentic_evalkit.errors import DatasetIntegrityError, ManifestValidationError
from agentic_evalkit.models import (
    DatasetRef,
    EvalRunManifest,
    EvalSample,
    ExecutionStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    SampleResult,
)


def _manifest() -> EvalRunManifest:
    return EvalRunManifest(
        run_name="test",
        dataset_ref=DatasetRef(provider="local", dataset_id="data"),
        adapter="mapped",
        grader="exact",
        target_name="target",
    )


def _sample() -> EvalSample:
    return EvalSample(
        sample_id="case", input={"question": "hi"}, source_digest="sha256:one", adapter="mapped"
    )


def _result() -> SampleResult:
    now = datetime.now(UTC)
    return SampleResult(
        sample=_sample(),
        execution=NormalizedExecutionResult(
            sample_id="case",
            attempt=1,
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            finished_at=now,
            output={"answer": "ok", "token": "sk-abcdefghijklmnopqrstuv"},
        ),
    )


def _begin(store: RunCheckpoint, manifest: EvalRunManifest | None = None) -> None:
    store.begin(
        manifest or _manifest(),
        ResolvedDataset(dataset_id="data", revision="sha256:data"),
        (_sample(),),
    )


def test_resume_reuses_saved_run_identity_and_completed_result(tmp_path: Path) -> None:
    with RunCheckpoint(tmp_path) as store:
        _begin(store)
        run_id = store.run_id
        store.start("case", 1)
        store.save(_result())
    with RunCheckpoint(tmp_path, resume=True) as store:
        _begin(store)
        assert store.run_id == run_id
        saved = store.completed()["case", 1]
        assert saved.execution.output["answer"] == "ok"
        assert "abcdefghijklmnopqrstuv" not in saved.model_dump_json()


def test_inflight_work_requires_explicit_retry(tmp_path: Path) -> None:
    with RunCheckpoint(tmp_path) as store:
        _begin(store)
        store.start("case", 1)
    with (
        RunCheckpoint(tmp_path, resume=True) as store,
        pytest.raises(ManifestValidationError, match="interrupted"),
    ):
        _begin(store)
    with RunCheckpoint(tmp_path, resume=True, retry_interrupted=True) as store:
        _begin(store)
        assert store.completed() == {}


def test_changed_manifest_refuses_resume(tmp_path: Path) -> None:
    with RunCheckpoint(tmp_path) as store:
        _begin(store)
    with RunCheckpoint(tmp_path, resume=True) as store, pytest.raises(DatasetIntegrityError):
        _begin(store, _manifest().model_copy(update={"grader": "different"}))


def test_existing_checkpoint_requires_resume(tmp_path: Path) -> None:
    with RunCheckpoint(tmp_path) as store:
        _begin(store)
    with pytest.raises(ManifestValidationError, match="resume"), RunCheckpoint(tmp_path):
        pass


def test_only_one_writer_can_own_a_checkpoint(tmp_path: Path) -> None:
    with (
        RunCheckpoint(tmp_path),
        pytest.raises(ManifestValidationError, match="locked"),
        RunCheckpoint(tmp_path, resume=True),
    ):
        pass


def test_checksum_corruption_is_detected_before_a_result_is_reused(tmp_path: Path) -> None:
    with RunCheckpoint(tmp_path) as store:
        _begin(store)
        store.start("case", 1)
        store.save(_result())

    with RunCheckpoint(tmp_path, resume=True) as store:
        store.connection.execute("UPDATE attempts SET payload='{}' WHERE sample_id='case'")
        store.connection.commit()
        _begin(store)
        with pytest.raises(DatasetIntegrityError, match="checksum"):
            store.completed()
