"""Exercise the saved-config CLI using local data and actual subprocesses."""

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agentic_evalkit.cli import app

runner = CliRunner()


def _workspace(path: Path, *, arms: int = 2) -> Path:
    (path / "cases.jsonl").write_text(
        '{"id":"one","prompt":"zero","expected":"0"}\n{"id":"two","prompt":"one","expected":"1"}\n',
        encoding="utf-8",
    )
    manifest = {
        "run_name": "local",
        "dataset": {"provider": "local", "dataset_id": "cases.jsonl"},
        "adapter": "field-map@1",
        "grader": "normalized-exact@1",
        "field_map": {
            "sample_id": ["id"],
            "input": {"question": ["prompt"]},
            "reference": ["expected"],
        },
        "target": {
            "kind": "callable",
            "import_string": "agentic_evalkit.examples.zero_target:zero_target",
        },
    }
    (path / "run.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    config = {
        "name": "local-test",
        "benchmark": {"name": "fixture", "reviewed_on": str(date.today())},
        "arms": [{"name": f"arm-{i}", "manifest": "run.yaml"} for i in range(arms)],
    }
    config_path = path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_run_status_resume_and_compare(tmp_path: Path) -> None:
    config = _workspace(tmp_path)
    run_dir = tmp_path / "results"
    result = runner.invoke(app, ["experiments", "run", str(config), "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    state = json.loads((run_dir / "state.json").read_text())
    assert [arm["status"] for arm in state["arms"]] == ["completed", "completed"]
    original = (run_dir / "arm-0" / state["arms"][0]["report"]).read_bytes()
    resumed = runner.invoke(app, ["experiments", "resume", str(run_dir)])
    assert resumed.exit_code == 0, resumed.output
    assert (run_dir / "arm-0" / state["arms"][0]["report"]).read_bytes() == original
    status = runner.invoke(app, ["experiments", "status", str(run_dir), "--json"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["name"] == "local-test"
    compared = runner.invoke(app, ["experiments", "compare", str(run_dir)])
    assert compared.exit_code == 0, compared.output
    assert json.loads(compared.output)["paired_instances"] == 2


def test_resume_refuses_modified_dataset(tmp_path: Path) -> None:
    config = _workspace(tmp_path, arms=1)
    run_dir = tmp_path / "results"
    assert (
        runner.invoke(app, ["experiments", "run", str(config), "--run-dir", str(run_dir)]).exit_code
        == 0
    )
    (tmp_path / "cases.jsonl").write_text('{"id":"changed","prompt":"x","expected":"0"}\n')
    result = runner.invoke(app, ["experiments", "resume", str(run_dir)])
    assert result.exit_code != 0
    assert "changed" in result.output.lower()


def test_study_report_exports_all_arms_without_inventing_labels(tmp_path: Path) -> None:
    config = _workspace(tmp_path, arms=3)
    run_dir = tmp_path / "results"
    assert (
        runner.invoke(app, ["experiments", "run", str(config), "--run-dir", str(run_dir)]).exit_code
        == 0
    )
    output_dir = tmp_path / "study-report"
    result = runner.invoke(
        app, ["experiments", "report", str(run_dir), "--output-dir", str(output_dir)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((output_dir / "report.json").read_text())
    assert set(payload["arms"]) == {"arm-0", "arm-1", "arm-2"}
    assert payload["schema_version"] == "1"
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "report.md").is_file()
    assert (
        runner.invoke(
            app, ["experiments", "report", str(run_dir), "--output-dir", str(output_dir)]
        ).exit_code
        != 0
    )
    state = json.loads((run_dir / "state.json").read_text())
    report = run_dir / "arm-0" / state["arms"][0]["report"]
    report.write_bytes(report.read_bytes() + b" ")
    assert runner.invoke(app, ["experiments", "report", str(run_dir)]).exit_code != 0


def test_duplicate_sample_ids_fail_before_execution(tmp_path: Path) -> None:
    config = _workspace(tmp_path, arms=1)
    (tmp_path / "cases.jsonl").write_text('{"id":"x","prompt":"x","expected":"0"}\n' * 2)
    result = runner.invoke(
        app, ["experiments", "run", str(config), "--run-dir", str(tmp_path / "results")]
    )
    assert result.exit_code != 0
    assert "duplicate" in result.output.lower()


def test_regression_dataset_cannot_be_labeled_capability(tmp_path: Path) -> None:
    path = _workspace(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["purpose"] = "capability"
    path.write_text(yaml.safe_dump(config))
    result = runner.invoke(app, ["experiments", "validate", str(path)])
    assert result.exit_code == 2
    assert "regression-only" in result.output


def test_inputs_changed_during_execution_cannot_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentic_evalkit.cli import experiments

    config = _workspace(tmp_path, arms=1)
    original = experiments._run_native

    async def changing_run(*args, **kwargs):
        report = await original(*args, **kwargs)
        (tmp_path / "cases.jsonl").write_text("{}\n")
        return report

    monkeypatch.setattr(experiments, "_run_native", changing_run)
    run_dir = tmp_path / "results"
    result = runner.invoke(app, ["experiments", "run", str(config), "--run-dir", str(run_dir)])
    assert result.exit_code != 0
    state = json.loads((run_dir / "state.json").read_text())
    assert state["arms"][0]["status"] == "failed"
