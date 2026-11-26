"""The ``integrity`` command and the experiment ``integrity:`` block."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003

import pytest
import yaml
from _integrity_fixtures import (
    SOLUTION_SHA,
    TEST_EDIT_PATCH,
    run_result,
    sample_result,
)
from typer.testing import CliRunner

from agentic_evalkit.cli import app
from agentic_evalkit.cli.experiments import _audit_arms, _integrity_summaries
from agentic_evalkit.cli.integrity import REPORT_NAME, find_run_report, run_integrity_audit
from agentic_evalkit.cli.runs import write_canonical_report
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.experiment import (
    ArmState,
    ArmStatus,
    ExperimentState,
    load_experiment,
)
from agentic_evalkit.integrity import IntegrityReport

runner = CliRunner()


def _shell_call(command: str) -> dict[str, object]:
    return {"tool": "shell_run", "arguments": {"command": command}}


def _write_run(directory: Path, *, cheat: bool = True) -> Path:
    samples = (
        sample_result("good", tool_calls=(_shell_call("pytest -q"),)),
        sample_result(
            "bad",
            patch=TEST_EDIT_PATCH if cheat else sample_result("x").execution.output["patch"],  # type: ignore[index]
            tool_calls=(_shell_call(f"git show {SOLUTION_SHA}"),),
        ),
    )
    return write_canonical_report(run_result(samples), directory)


def _context(directory: Path) -> Path:
    path = directory / "integrity_context.json"
    path.write_text(
        json.dumps({"instances": {"bad": {"solution_shas": [SOLUTION_SHA]}}}), encoding="utf-8"
    )
    return path


# --- CLI --------------------------------------------------------------------


def test_cli_writes_a_report_with_raw_and_clean_rates(tmp_path: Path) -> None:
    _write_run(tmp_path)
    _context(tmp_path)
    outcome = runner.invoke(app, ["integrity", str(tmp_path)])
    assert outcome.exit_code == 0, outcome.output
    assert "1 flagged" in outcome.output
    assert "clean_pass_rate=0.500" in outcome.output
    report = IntegrityReport.model_validate_json((tmp_path / REPORT_NAME).read_text("utf-8"))
    assert (report.raw_pass_rate, report.clean_pass_rate) == (1.0, 0.5)
    bad = next(item for item in report.instances if item.instance_id == "bad")
    assert {flag.rule for flag in bad.flags} == {"graded_file_edited", "solution_sha"}


def test_cli_does_not_modify_the_run_report(tmp_path: Path) -> None:
    report_path = _write_run(tmp_path)
    before = report_path.read_bytes()
    runner.invoke(app, ["integrity", str(tmp_path)])
    assert report_path.read_bytes() == before


def test_cli_reports_a_directory_without_a_run_report(tmp_path: Path) -> None:
    outcome = runner.invoke(app, ["integrity", str(tmp_path)])
    assert outcome.exit_code == 2
    assert "exactly one run report" in outcome.output


def test_find_run_report_ignores_its_own_output(tmp_path: Path) -> None:
    path = _write_run(tmp_path)
    (tmp_path / REPORT_NAME).write_text("{}", encoding="utf-8")
    assert find_run_report(tmp_path) == path


def test_cli_rejects_a_malformed_context_file(tmp_path: Path) -> None:
    path = _write_run(tmp_path)
    (tmp_path / "integrity_context.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ManifestValidationError):
        run_integrity_audit(path, tmp_path)


def test_cli_reads_tool_calls_jsonl_next_to_the_report(tmp_path: Path) -> None:
    path = _write_run(tmp_path)
    record = {
        "run_id": "run-1",
        "sample_id": "good",
        "call_id": "1",
        "tool": "shell_run",
        "arguments": {"command": f"git show {SOLUTION_SHA}"},
    }
    (tmp_path / "tool_calls.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / "integrity_context.json").write_text(
        json.dumps({"instances": {"good": {"solution_shas": [SOLUTION_SHA]}}}), encoding="utf-8"
    )
    report = run_integrity_audit(path, tmp_path)
    assert next(i for i in report.instances if i.instance_id == "good").flags


def test_cli_carries_preflight_checks_from_the_context(tmp_path: Path) -> None:
    path = _write_run(tmp_path, cheat=False)
    (tmp_path / "integrity_context.json").write_text(
        json.dumps({"preflight": [{"name": "egress", "status": "failed", "detail": "open"}]}),
        encoding="utf-8",
    )
    report = run_integrity_audit(path, tmp_path)
    assert [(c.name, c.passed) for c in report.preflight] == [("egress", False)]


# --- experiment block -------------------------------------------------------


def _experiment_yaml(directory: Path, **extra: object) -> Path:
    (directory / "evidence.md").write_text("x", encoding="utf-8")
    path = directory / "exp.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "swe",
                "benchmark": {"name": "swe", "reviewed_on": datetime.now(UTC).date().isoformat()},
                "arms": [{"name": "a", "manifest": "a.yaml"}],
                **extra,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_experiment_without_an_integrity_block_is_unchanged(tmp_path: Path) -> None:
    assert load_experiment(_experiment_yaml(tmp_path)).integrity is None


def test_experiment_integrity_block_requires_its_context_file(tmp_path: Path) -> None:
    path = _experiment_yaml(tmp_path, integrity={"context": "ctx.json"})
    with pytest.raises(ManifestValidationError, match=r"ctx\.json"):
        load_experiment(path)
    (tmp_path / "ctx.json").write_text("{}", encoding="utf-8")
    assert load_experiment(path).integrity is not None


def test_experiment_integrity_block_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError):
        load_experiment(_experiment_yaml(tmp_path, integrity={"conext": "typo.json"}))


def _state(arm_dir: Path, report: Path, status: ArmStatus) -> ExperimentState:
    return ExperimentState(
        name="swe",
        config_path="exp.yaml",
        input_digest="sha256:x",
        started_at=datetime.now(UTC),
        arms=(ArmState(name="a", status=status, report=report.name),),
    )


def test_audit_arms_writes_a_report_for_each_completed_native_arm(tmp_path: Path) -> None:
    arm_dir = tmp_path / "run" / "a"
    report = _write_run(arm_dir)
    from agentic_evalkit.experiment import file_digest

    state = _state(arm_dir, report, ArmStatus.COMPLETED)
    state = state.model_copy(
        update={"arms": (state.arms[0].model_copy(update={"report_digest": file_digest(report)}),)}
    )
    config = load_experiment(_experiment_yaml(tmp_path, integrity={}))
    _audit_arms(config, tmp_path / "exp.yaml", tmp_path / "run", state)
    assert (arm_dir / REPORT_NAME).is_file()
    summary = _integrity_summaries(tmp_path / "run", state)
    assert summary["a"]["total"] == 2  # type: ignore[index]


def test_audit_arms_skips_arms_that_did_not_complete(tmp_path: Path) -> None:
    arm_dir = tmp_path / "run" / "a"
    report = _write_run(arm_dir)
    state = _state(arm_dir, report, ArmStatus.FAILED)
    config = load_experiment(_experiment_yaml(tmp_path, integrity={}))
    _audit_arms(config, tmp_path / "exp.yaml", tmp_path / "run", state)
    assert not (arm_dir / REPORT_NAME).exists()
    assert _integrity_summaries(tmp_path / "run", state) == {}


# --- end to end ---------------------------------------------------------------


def _e2e_experiment(directory: Path) -> Path:
    (directory / "cases.jsonl").write_text(
        '{"question": "What is 2+2?", "answer": "so 4 #### 4"}\n', encoding="utf-8"
    )
    (directory / "arm.yaml").write_text(
        "run_name: e2e\n"
        "dataset:\n  provider: local\n  dataset_id: cases.jsonl\n"
        "adapter: gsm8k@1\ngrader: normalized-exact@1\n"
        "target:\n  kind: callable\n"
        "  import_string: agentic_evalkit.examples.zero_target:zero_target\n",
        encoding="utf-8",
    )
    (directory / "integrity_context.json").write_text('{"instances": {}}', encoding="utf-8")
    path = directory / "exp.yaml"
    path.write_text(
        "name: e2e\n"
        f"benchmark: {{name: local, reviewed_on: {datetime.now(UTC).date().isoformat()}}}\n"
        "arms:\n  - {name: a, manifest: arm.yaml}\n"
        "integrity:\n  context: integrity_context.json\n",
        encoding="utf-8",
    )
    return path


def test_experiments_run_writes_an_integrity_report_per_arm(tmp_path: Path) -> None:
    experiment = _e2e_experiment(tmp_path)
    run_dir = tmp_path / "out"
    outcome = runner.invoke(app, ["experiments", "run", str(experiment), "--run-dir", str(run_dir)])
    assert "integrity 1 instance(s)" in outcome.output, outcome.output
    report = IntegrityReport.model_validate_json((run_dir / "a" / REPORT_NAME).read_text("utf-8"))
    assert report.total == 1
    assert report.context_digest is not None


def test_changing_the_context_file_means_a_new_run_directory(tmp_path: Path) -> None:
    experiment = _e2e_experiment(tmp_path)
    run_dir = tmp_path / "out"
    runner.invoke(app, ["experiments", "run", str(experiment), "--run-dir", str(run_dir)])
    (tmp_path / "integrity_context.json").write_text(
        '{"instances": {}, "preflight": []}', encoding="utf-8"
    )
    outcome = runner.invoke(app, ["experiments", "resume", str(run_dir)])
    assert outcome.exit_code != 0
    assert "start a new run directory" in outcome.output


def test_experiments_run_without_a_context_ignores_files_in_the_run_directory(
    tmp_path: Path,
) -> None:
    experiment = _e2e_experiment(tmp_path)
    text = experiment.read_text(encoding="utf-8").replace(
        "integrity:\n  context: integrity_context.json\n", "integrity: {}\n"
    )
    experiment.write_text(text, encoding="utf-8")
    run_dir = tmp_path / "out"
    # A context planted in the arm directory afterwards must not be read: it is not part
    # of the run's input identity.
    runner.invoke(app, ["experiments", "run", str(experiment), "--run-dir", str(run_dir)])
    (run_dir / "a" / "integrity_context.json").write_text(
        '{"preflight": [{"name": "x", "status": "failed"}]}', encoding="utf-8"
    )
    runner.invoke(app, ["experiments", "resume", str(run_dir)])
    report = IntegrityReport.model_validate_json((run_dir / "a" / REPORT_NAME).read_text("utf-8"))
    assert report.preflight == ()
