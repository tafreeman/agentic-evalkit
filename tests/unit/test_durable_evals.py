"""Regression cases found while preparing durable experiment execution."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_evalkit.cli import app
from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.errors import DatasetIntegrityError
from agentic_evalkit.graders.composite import CompositeGrader, WeightedGrader
from agentic_evalkit.models import (
    DatasetRef,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
)

runner = CliRunner()


@pytest.mark.asyncio
async def test_local_dataset_cannot_change_after_resolution(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question":"original"}\n', encoding="utf-8")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    dataset = await provider.resolve(DatasetRef(provider="local", dataset_id=str(path)))
    path.write_text('{"question":"changed"}\n', encoding="utf-8")
    with pytest.raises(DatasetIntegrityError):
        await provider.preview(dataset)


class FixedGrader:
    def __init__(self, status: GradeStatus) -> None:
        self.status = status

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        return GradeResult(
            sample_id=sample.sample_id,
            grader="fixed",
            status=self.status,
            score=1.0 if self.status is GradeStatus.PASS else None,
            created_at=datetime.now(UTC),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [GradeStatus.UNAVAILABLE, GradeStatus.ERROR, GradeStatus.ABSTAIN]
)
async def test_missing_hard_gate_cannot_be_replaced_by_sanity_pass(status: GradeStatus) -> None:
    sample = EvalSample(sample_id="case", input={}, adapter="test", source_digest="sha256:test")
    now = datetime.now(UTC)
    execution = NormalizedExecutionResult(
        sample_id="case",
        attempt=1,
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
        output={"answer": "ok"},
    )
    grader = CompositeGrader(
        name="required-oracle",
        graders=(
            WeightedGrader(FixedGrader(GradeStatus.PASS), weight=1, hard_gate=False),
            WeightedGrader(FixedGrader(status), weight=1, hard_gate=True),
        ),
    )
    result = await grader.grade(sample, execution)
    assert result.status is (
        GradeStatus.ERROR if status is GradeStatus.ERROR else GradeStatus.UNAVAILABLE
    )
    assert result.score is None
    assert result.hard_gate is False


@pytest.mark.parametrize("weight", [float("nan"), float("inf")])
def test_composite_rejects_nonfinite_weights(weight: float) -> None:
    with pytest.raises(ValueError):
        WeightedGrader(FixedGrader(GradeStatus.PASS), weight=weight, hard_gate=False)


def test_run_resume_flags_require_a_checkpoint_directory(tmp_path: Path) -> None:
    manifest = tmp_path / "run.yaml"
    manifest.write_text(
        """
run_name: checkpoint-contract
dataset:
  provider: local
  dataset_id: cases.jsonl
adapter: gsm8k@1
grader: normalized-exact@1
target:
  kind: callable
  import_string: agentic_evalkit.examples.zero_target:zero_target
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", str(manifest), "--resume"])

    assert result.exit_code != 0
    assert "require --checkpoint-dir" in result.output
