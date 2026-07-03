"""Integration tests for the runnable objective-only CLI (plan Task 14, Steps 1-7).

The first two tests below are copied verbatim from
``docs/plans/2026-07-02-agentic-evalkit-initial-release.md`` (Task 14, Step 1).
Everything else in this module is additional coverage for the ``doctor``,
``datasets``, ``init``/``validate``, and ``run`` commands built in Steps 5-7,
plus the exit-code policy from Step 4.

``test_provider_failure_has_nonzero_exit_and_error_code`` legitimately hits
the real Hugging Face Dataset Viewer. The Viewer returns HTTP 401 (not 404)
for anonymous requests to any nonexistent dataset -- deliberate
anti-enumeration so private-dataset existence cannot be probed -- which the
provider correctly maps to ``dataset_access_denied`` (ADR-0003 / Task 6). The
test therefore asserts the exit code (4, provider error) plus that a stable
provider error code is surfaced, not a specific one. This module is not marked
``live``; it relies on one specific, stable negative-path HTTP response.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from agentic_evalkit.cli import app
from agentic_evalkit.models import (
    DatasetRef,
    EvalRunManifest,
    EvalRunResult,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    RunSummary,
    SampleResult,
    SamplingPolicy,
)
from agentic_evalkit.reporters.json import JsonReporter

runner = CliRunner()

# --- Deterministic run fixtures for compare/report (no network) -------------

_STARTED_AT = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 7, 3, 12, 5, 0, tzinfo=UTC)


def _sample_result(sample_id: str, *, passed: bool) -> SampleResult:
    grade_status = GradeStatus.PASS if passed else GradeStatus.FAIL
    return SampleResult(
        sample=EvalSample(
            sample_id=sample_id,
            input={"question": f"q-{sample_id}"},
            reference="42",
            source_digest=f"sha256:{sample_id}",
            adapter="gsm8k@1",
        ),
        execution=NormalizedExecutionResult(
            sample_id=sample_id,
            attempt=1,
            output={"answer": "42" if passed else "0"},
            status=ExecutionStatus.COMPLETED,
            started_at=_STARTED_AT,
            finished_at=_FINISHED_AT,
        ),
        grade=GradeResult(
            sample_id=sample_id,
            grader="normalized-exact@1",
            status=grade_status,
            score=1.0 if passed else 0.0,
            created_at=_FINISHED_AT,
        ),
    )


def _run_result(
    *,
    run_id: str,
    dataset_revision: str = "rev-abc",
    passed_flags: tuple[bool, ...] = (True, False),
) -> EvalRunResult:
    samples = tuple(
        _sample_result(f"gsm8k:{index}", passed=flag) for index, flag in enumerate(passed_flags)
    )
    manifest = EvalRunManifest(
        run_name=f"{run_id}-fixture",
        dataset_ref=DatasetRef(
            provider="huggingface", dataset_id="openai/gsm8k", config="main", split="test"
        ),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="cli-target",
        sampling=SamplingPolicy(seed=7, temperature=0.0, attempts=1),
        attempts=1,
    )
    resolved = ResolvedDataset(
        dataset_id="openai/gsm8k",
        revision=dataset_revision,
        config="main",
        split="test",
    )
    passed = sum(passed_flags)
    return EvalRunResult(
        run_id=run_id,
        manifest=manifest,
        resolved_dataset=resolved,
        samples=samples,
        summary=RunSummary(total=len(samples), passed=passed, failed=len(samples) - passed),
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _write_run(tmp_path: Path, name: str, run: EvalRunResult) -> Path:
    destination = tmp_path / f"{name}.json"
    JsonReporter().write(run, destination, generated_at="2026-07-03T12:05:00+00:00")
    return destination


def test_curated_and_init_work_without_manual_import(tmp_path) -> None:  # type: ignore[no-untyped-def]
    listed = runner.invoke(app, ["datasets", "curated", "--format", "json"])
    assert listed.exit_code == 0
    assert "swe-bench-verified" in listed.stdout
    destination = tmp_path / "eval.yaml"
    created = runner.invoke(app, ["init", "--preset", "gsm8k", "--output", str(destination)])
    assert created.exit_code == 0
    assert destination.exists()
    validated = runner.invoke(app, ["validate", str(destination)])
    assert validated.exit_code == 0
    assert "valid" in validated.stdout.lower()


def test_provider_failure_has_nonzero_exit_and_error_code() -> None:
    result = runner.invoke(app, ["datasets", "inspect", "hf:missing/not-found"])
    assert result.exit_code == 4
    # HF returns 401 (anti-enumeration) rather than 404 for a nonexistent
    # dataset, which maps to dataset_access_denied; either provider error code
    # is a correct, stable, nonzero-exit outcome for an unresolvable dataset.
    assert ("dataset_access_denied" in result.stdout) or ("dataset_not_found" in result.stdout)


# --- Additional coverage (plan Task 14, Steps 5-7) --------------------------


def test_root_app_shows_help_without_a_subcommand() -> None:
    # Typer/Click's no_args_is_help=True prints help and exits 2 (its
    # standard "no command given" usage-error convention) rather than 0;
    # this still surfaces every command name so a user immediately sees
    # what is available.
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "doctor" in result.stdout
    assert "run" in result.stdout


def test_version_flag_prints_the_installed_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_doctor_runs_offline_and_reports_checks() -> None:
    result = runner.invoke(app, ["doctor", "--offline", "--format", "json"])
    assert result.exit_code in (0, 3)
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert all("status" in entry for entry in payload)


def test_datasets_curated_table_format_lists_both_presets() -> None:
    result = runner.invoke(app, ["datasets", "curated"])
    assert result.exit_code == 0
    assert "gsm8k" in result.stdout
    assert "swe-bench-verified" in result.stdout


def test_init_without_preset_or_dataset_is_invalid_input() -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 2


def test_init_refuses_to_overwrite_existing_file_without_force(tmp_path: Path) -> None:
    destination = tmp_path / "eval.yaml"
    destination.write_text("existing: true\n")
    result = runner.invoke(app, ["init", "--preset", "gsm8k", "--output", str(destination)])
    assert result.exit_code == 2
    assert destination.read_text() == "existing: true\n"


def test_validate_rejects_malformed_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "bad.yaml"
    destination.write_text("run_name: only-one-field\n")
    result = runner.invoke(app, ["validate", str(destination)])
    assert result.exit_code == 2
    assert "manifest_validation_error" in result.stdout


def test_validate_rejects_python_tagged_yaml(tmp_path: Path) -> None:
    destination = tmp_path / "unsafe.yaml"
    destination.write_text("run_name: !!python/object/apply:os.system ['echo hi']\n")
    result = runner.invoke(app, ["validate", str(destination)])
    assert result.exit_code == 2


def test_run_executes_gsm8k_against_the_zero_target_and_writes_canonical_json(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "eval.yaml"
    created = runner.invoke(app, ["init", "--preset", "gsm8k", "--output", str(manifest_path)])
    assert created.exit_code == 0

    output_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "run",
            str(manifest_path),
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.stdout

    report_files = list(output_dir.glob("*.json"))
    assert len(report_files) == 1
    envelope = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert envelope["summary"]["total"] == 1
    assert len(envelope["samples"]) == 1
    sample = envelope["samples"][0]
    assert sample["execution"]["status"] == "completed"
    assert sample["grade"]["status"] in ("pass", "fail")
    assert str(report_files[0]) in result.stdout


def test_run_missing_manifest_file_is_invalid_input(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", str(tmp_path / "does-not-exist.yaml"), "--yes"],
    )
    assert result.exit_code == 2


def test_datasets_search_supports_json_format() -> None:
    result = runner.invoke(
        app,
        ["datasets", "search", "gsm8k", "--provider", "huggingface", "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    assert "hits" in payload


# --- compare (plan Task 14 Step 10) -----------------------------------------


def test_compare_two_compatible_runs_reports_estimate_and_seed(tmp_path: Path) -> None:
    left = _write_run(tmp_path, "left", _run_result(run_id="run-a", passed_flags=(True, False)))
    right = _write_run(tmp_path, "right", _run_result(run_id="run-b", passed_flags=(True, True)))

    result = runner.invoke(
        app,
        ["compare", str(left), str(right), "--bootstrap-samples", "200", "--seed", "13"],
    )
    assert result.exit_code == 0, result.stdout
    assert "estimate" in result.stdout.lower()
    assert "seed" in result.stdout.lower()
    assert "13" in result.stdout


def test_compare_json_format_carries_percentiles_and_paired_count(tmp_path: Path) -> None:
    left = _write_run(tmp_path, "left", _run_result(run_id="run-a"))
    right = _write_run(tmp_path, "right", _run_result(run_id="run-b"))

    result = runner.invoke(
        app,
        [
            "compare",
            str(left),
            str(right),
            "--bootstrap-samples",
            "200",
            "--seed",
            "5",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["seed"] == 5
    assert payload["paired_count"] == 2
    assert "estimate" in payload
    assert "lower_percentile" in payload
    assert "upper_percentile" in payload


def test_compare_incompatible_runs_lists_every_mismatch_and_exits_two(tmp_path: Path) -> None:
    left = _write_run(tmp_path, "left", _run_result(run_id="run-a", dataset_revision="rev-abc"))
    right = _write_run(tmp_path, "right", _run_result(run_id="run-b", dataset_revision="rev-def"))

    result = runner.invoke(app, ["compare", str(left), str(right), "--seed", "1"])
    assert result.exit_code == 2
    assert "incompatible_runs" in result.stdout
    assert "dataset revision" in result.stdout
    assert "rev-abc" in result.stdout
    assert "rev-def" in result.stdout


def test_compare_rejects_bootstrap_samples_out_of_range(tmp_path: Path) -> None:
    left = _write_run(tmp_path, "left", _run_result(run_id="run-a"))
    right = _write_run(tmp_path, "right", _run_result(run_id="run-b"))

    too_low = runner.invoke(
        app, ["compare", str(left), str(right), "--bootstrap-samples", "50", "--seed", "1"]
    )
    assert too_low.exit_code == 2

    too_high = runner.invoke(
        app, ["compare", str(left), str(right), "--bootstrap-samples", "20000", "--seed", "1"]
    )
    assert too_high.exit_code == 2


def test_compare_missing_run_file_is_invalid_input(tmp_path: Path) -> None:
    right = _write_run(tmp_path, "right", _run_result(run_id="run-b"))
    result = runner.invoke(app, ["compare", str(tmp_path / "nope.json"), str(right), "--seed", "1"])
    assert result.exit_code == 2


# --- report (plan Task 14 Step 10) ------------------------------------------


def test_report_regenerates_jsonl(tmp_path: Path) -> None:
    source = _write_run(tmp_path, "run", _run_result(run_id="run-a"))
    destination = tmp_path / "out.jsonl"
    result = runner.invoke(
        app, ["report", str(source), "--format", "jsonl", "--output", str(destination)]
    )
    assert result.exit_code == 0, result.stdout
    assert destination.exists()
    lines = destination.read_text(encoding="utf-8").strip().splitlines()
    # header + one record per sample + trailer
    assert len(lines) == 4
    header = json.loads(lines[0])
    assert header["record_type"] == "header"
    assert header["run_id"] == "run-a"


def test_report_regenerates_markdown(tmp_path: Path) -> None:
    source = _write_run(tmp_path, "run", _run_result(run_id="run-a"))
    destination = tmp_path / "out.md"
    result = runner.invoke(
        app, ["report", str(source), "--format", "markdown", "--output", str(destination)]
    )
    assert result.exit_code == 0, result.stdout
    content = destination.read_text(encoding="utf-8")
    assert "# Evaluation Run" in content
    assert "run-a" in content


def test_report_regenerates_self_contained_html(tmp_path: Path) -> None:
    source = _write_run(tmp_path, "run", _run_result(run_id="run-a"))
    destination = tmp_path / "out.html"
    result = runner.invoke(
        app, ["report", str(source), "--format", "html", "--output", str(destination)]
    )
    assert result.exit_code == 0, result.stdout
    content = destination.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    # self-contained: no remote script/style/font references
    assert "http://" not in content
    assert "https://" not in content


def test_report_missing_run_file_is_invalid_input(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "report",
            str(tmp_path / "nope.json"),
            "--format",
            "jsonl",
            "--output",
            str(tmp_path / "x.jsonl"),
        ],
    )
    assert result.exit_code == 2
