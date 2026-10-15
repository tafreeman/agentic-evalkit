"""One saved workflow for single-target, multi-arm, and native HELM evaluations."""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.checkpoints import ProcessLock, RunCheckpoint, digest
from agentic_evalkit.cli.app import ExitCode, app, console, run_cli_command, safe_text
from agentic_evalkit.cli.datasets import build_catalog
from agentic_evalkit.cli.reports import load_run_result
from agentic_evalkit.cli.runs import (
    _RunnerCatalogAdapter,
    _with_contamination,
    _with_provenance_fingerprints,
    build_target_for_document,
    components_for_document,
    write_canonical_report,
)
from agentic_evalkit.errors import DatasetIntegrityError, ManifestValidationError
from agentic_evalkit.events import RunEvent, SampleCompleted
from agentic_evalkit.experiment import (
    ArmState,
    ArmStatus,
    ExperimentConfig,
    ExperimentState,
    atomic_json,
    file_digest,
    load_experiment,
)
from agentic_evalkit.helm import helm_version, run_helm
from agentic_evalkit.manifest import ManifestDocument, load_manifest
from agentic_evalkit.provenance import compute_environment_fingerprint
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY
from agentic_evalkit.runner import EvalRunner

if TYPE_CHECKING:
    from agentic_evalkit.experiment import ExperimentArm
    from agentic_evalkit.models import EvalRunResult

experiments_app = typer.Typer(help="Validate, run, resume, and compare saved experiments.")
app.add_typer(experiments_app, name="experiments")


def _document(path: Path) -> ManifestDocument:
    document = load_manifest(path)
    if document.manifest.dataset_ref.provider == "local":
        ref = document.manifest.dataset_ref
        ref = ref.model_copy(update={"dataset_id": str((path.parent / ref.dataset_id).resolve())})
        document = document.model_copy(
            update={"manifest": document.manifest.model_copy(update={"dataset_ref": ref})}
        )
    return document


def _input_identity(config: ExperimentConfig, path: Path) -> str:
    """Measure code and declared dependencies, including uncommitted source edits."""
    files = {str(path): file_digest(path)}
    base = path.parent
    for item in (*config.provenance_files, *config.benchmark.validation_evidence):
        dependency = (base / item).resolve()
        files[str(dependency)] = file_digest(dependency)
    for arm in config.arms:
        if arm.manifest is not None:
            manifest_path = (base / arm.manifest).resolve()
            files[str(manifest_path)] = file_digest(manifest_path)
            document = _document(manifest_path)
            if document.components is not None:
                dependency = Path(document.components.path)
                files[str(dependency)] = file_digest(dependency)
            if document.manifest.dataset_ref.provider == "local":
                dependency = Path(document.manifest.dataset_ref.dataset_id)
                files[str(dependency)] = file_digest(dependency)
            # A callable target is executable input too.  Its manifest entry
            # identifies the symbol, but not the bytes that implement it.
            # Record the module file when Python can resolve it; built-in and
            # extension modules have no stable source file and must instead
            # be supplied through ``provenance_files`` by the experiment.
            target = document.target
            if target.kind == "callable":
                module_name, _, _ = target.import_string.partition(":")
                spec = importlib.util.find_spec(module_name)
                origin = spec.origin if spec is not None else None
                if origin in (None, "built-in", "frozen"):
                    raise ManifestValidationError(
                        message=(
                            "callable target cannot be source-fingerprinted; "
                            "declare its immutable dependency in provenance_files"
                        )
                    )
                dependency = Path(cast("str", origin)).resolve()
                files[str(dependency)] = file_digest(dependency)
        elif arm.helm is not None:
            helm_version(arm.helm)
    package = Path(__file__).resolve().parent.parent
    source = {
        str(item.relative_to(package)): file_digest(item) for item in sorted(package.rglob("*.py"))
    }
    return digest(
        {"files": files, "source": source, "environment": compute_environment_fingerprint()}
    )


def _load_state(directory: Path) -> ExperimentState:
    try:
        return ExperimentState.model_validate_json(
            (directory / "state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ManifestValidationError(
            message="run directory has no readable experiment state"
        ) from error


def _save_state(directory: Path, state: ExperimentState) -> None:
    atomic_json(directory / "state.json", state.model_dump(mode="json"))


def _update_arm(state: ExperimentState, arm: ArmState) -> ExperimentState:
    return state.model_copy(
        update={"arms": tuple(arm if item.name == arm.name else item for item in state.arms)}
    )


def _error_text(error: BaseException) -> str:
    text = str(error)[:4000]
    for pattern in DEFAULT_REDACTION_POLICY.secret_patterns:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


async def _run_native(
    arm: ExperimentArm, base: Path, directory: Path, *, retry_interrupted: bool
) -> Path:
    if arm.manifest is None:
        raise ManifestValidationError(message="native arm has no manifest")
    document = _document((base / arm.manifest).resolve())
    manifest, adapters, graders = components_for_document(document)
    manifest = _with_provenance_fingerprints(manifest, document.target)
    ref = manifest.dataset_ref
    catalog = _RunnerCatalogAdapter(
        build_catalog(offline=False, allowed_local_roots=(base.resolve(),)),  # noqa: ASYNC240
        ref,
    )
    target = build_target_for_document(document)
    runner = EvalRunner(
        catalog=catalog,
        adapters=adapters,
        graders=graders,
        targets={manifest.target_name: target},
        artifact_store=ArtifactStore(directory / "artifacts"),
    )

    def progress(event: RunEvent) -> None:
        if isinstance(event, SampleCompleted):
            console.print(
                safe_text(f"{arm.name}: completed {event.sample_id} attempt {event.attempt}")
            )

    with RunCheckpoint(
        directory,
        resume=(directory / "checkpoint.sqlite3").exists(),
        retry_interrupted=retry_interrupted,
    ) as checkpoint:
        result = await runner.run(manifest, event_sink=progress, checkpoint=checkpoint)
    result = _with_contamination(result, manifest)
    return write_canonical_report(result, directory)


def _verify_saved_report(directory: Path, arm: ArmState) -> Path:
    if arm.report is None:
        raise DatasetIntegrityError(message=f"completed arm {arm.name} has no report")
    path = (directory / arm.name / arm.report).resolve()
    if not path.is_relative_to((directory / arm.name).resolve()):
        raise DatasetIntegrityError(message="report path escapes arm directory")
    if not path.is_file() or file_digest(path) != arm.report_digest:
        raise DatasetIntegrityError(message=f"saved report for {arm.name} changed or is missing")
    if path.name == "helm-result.json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["evidence"]:
            evidence = (path.parent / item["path"]).resolve()
            if (
                not evidence.is_relative_to(path.parent)
                or not evidence.is_file()
                or file_digest(evidence) != item["sha256"]
            ):
                raise DatasetIntegrityError(message="native HELM evidence changed or is missing")
    return path


async def _execute(
    config_path: Path, directory: Path, *, resume: bool, retry_interrupted: bool
) -> ExperimentState:
    config_path = config_path.resolve()  # noqa: ASYNC240
    config = load_experiment(config_path)
    directory = directory.resolve()  # noqa: ASYNC240
    identity = _input_identity(config, config_path)
    with ProcessLock(directory / ".experiment.lock"):
        state_path = directory / "state.json"
        if resume:
            state = _load_state(directory)
            if state.input_digest != identity:
                raise DatasetIntegrityError(
                    message="experiment inputs, dataset, or code changed; start a new run directory"
                )
        else:
            entries = (
                tuple(entry for entry in directory.iterdir() if entry.name != ".experiment.lock")
                if directory.exists()
                else ()
            )
            if state_path.exists() or entries:
                raise ManifestValidationError(
                    message="run directory exists; use experiments resume"
                )
            state = ExperimentState(
                name=config.name,
                config_path=str(config_path),
                input_digest=identity,
                started_at=datetime.now(UTC),
                arms=tuple(ArmState(name=arm.name) for arm in config.arms),
            )
            _save_state(directory, state)
        for arm in config.arms:
            previous = next(item for item in state.arms if item.name == arm.name)
            if previous.status in (ArmStatus.COMPLETED, ArmStatus.PREPARED):
                _verify_saved_report(directory, previous)
                continue
            if (
                arm.helm is not None
                and previous.status in (ArmStatus.RUNNING, ArmStatus.INTERRUPTED)
                and not retry_interrupted
            ):
                raise ManifestValidationError(
                    message="HELM was interrupted; inspect its log then use --retry-interrupted"
                )
            current = previous.model_copy(update={"status": ArmStatus.RUNNING, "error": None})
            state = _update_arm(state, current)
            _save_state(directory, state)
            arm_dir = directory / arm.name
            try:
                if _input_identity(config, config_path) != identity:
                    raise DatasetIntegrityError(
                        message="experiment inputs changed before execution"
                    )
                report = (
                    await run_helm(arm.helm, arm_dir, arm.name)
                    if arm.helm is not None
                    else await _run_native(
                        arm, config_path.parent, arm_dir, retry_interrupted=retry_interrupted
                    )
                )
                if _input_identity(config, config_path) != identity:
                    raise DatasetIntegrityError(
                        message="experiment inputs changed during execution"
                    )
                completed = (
                    ArmStatus.PREPARED
                    if arm.helm is not None and arm.helm.phase == "prepare"
                    else ArmStatus.COMPLETED
                )
                current = current.model_copy(
                    update={
                        "status": completed,
                        "report": report.name,
                        "report_digest": file_digest(report),
                    }
                )
            except BaseException as error:
                status = (
                    ArmStatus.INTERRUPTED
                    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
                    else ArmStatus.FAILED
                )
                current = current.model_copy(update={"status": status, "error": _error_text(error)})
                state = _update_arm(state, current)
                _save_state(directory, state)
                raise
            state = _update_arm(state, current)
            _save_state(directory, state)
        return state


@experiments_app.command("validate")
def validate_experiment(
    config: Annotated[Path, typer.Argument(help="Experiment YAML or JSON.")],
) -> None:
    """Check config, files, benchmark eligibility, and HELM version without calling a target."""

    def action() -> None:
        definition = load_experiment(config)
        identity = _input_identity(definition, config.resolve())
        console.print(
            safe_text(
                f"valid: {definition.name}; {len(definition.arms)} arm(s); "
                f"{definition.purpose}; {identity}"
            )
        )

    run_cli_command(action, debug=False)


@experiments_app.command("run")
def run_experiment(
    config: Annotated[Path, typer.Argument()],
    run_dir: Annotated[
        Path, typer.Option("--run-dir", help="A new directory for durable state and results.")
    ],
) -> None:
    """Run all configured arms, checkpointing every completed native attempt."""
    state = run_cli_command(
        lambda: asyncio.run(_execute(config, run_dir, resume=False, retry_interrupted=False)),
        debug=False,
    )
    _print_finished(run_dir, state)


@experiments_app.command("resume")
def resume_experiment(
    run_dir: Annotated[Path, typer.Argument()],
    retry_interrupted: Annotated[bool, typer.Option("--retry-interrupted")] = False,
) -> None:
    """Continue an unchanged experiment without replaying completed attempts."""
    saved = run_cli_command(lambda: _load_state(run_dir), debug=False)
    state = run_cli_command(
        lambda: asyncio.run(
            _execute(
                Path(saved.config_path), run_dir, resume=True, retry_interrupted=retry_interrupted
            )
        ),
        debug=False,
    )
    _print_finished(run_dir, state)


def _print_finished(directory: Path, state: ExperimentState) -> None:
    issues = False
    for arm in state.arms:
        console.print(
            safe_text(f"{arm.name}: {arm.status}; {directory / arm.name / (arm.report or '')}"),
            soft_wrap=True,
        )
        if arm.report is not None and arm.report != "helm-result.json":
            result = load_run_result(directory / arm.name / arm.report)
            summary = result.summary
            console.print(
                safe_text(
                    f"  pass={summary.passed} fail={summary.failed} error={summary.errors} "
                    f"timeout={summary.timeouts} unavailable={summary.unavailable}"
                )
            )
            issues |= bool(
                summary.errors or summary.timeouts or summary.cancelled or summary.unavailable
            )
    if issues:
        raise typer.Exit(int(ExitCode.INFRASTRUCTURE_ERROR))


@experiments_app.command("status")
def experiment_status(
    run_dir: Annotated[Path, typer.Argument()],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Read saved progress; no target calls or dataset downloads."""
    state = run_cli_command(lambda: _load_state(run_dir), debug=False)
    if as_json:
        typer.echo(state.model_dump_json(indent=2))
    else:
        for arm in state.arms:
            console.print(safe_text(f"{arm.name}: {arm.status}"))


@experiments_app.command("compare")
def compare_experiment(run_dir: Annotated[Path, typer.Argument()]) -> None:
    """Compare two completed native arms with the target as the declared treatment."""
    from agentic_evalkit.stats.experiment import compare_experiment_arms

    def action() -> dict[str, object]:
        state = _load_state(run_dir)
        if len(state.arms) != 2 or any(arm.status is not ArmStatus.COMPLETED for arm in state.arms):
            raise ManifestValidationError(
                message="comparison requires exactly two completed native arms"
            )
        results: list[EvalRunResult] = []
        for arm in state.arms:
            path = _verify_saved_report(run_dir, arm)
            if path.name == "helm-result.json":
                raise ManifestValidationError(
                    message="use native HELM metrics; cross-backend comparison is not supported"
                )
            results.append(load_run_result(path))
        return compare_experiment_arms(
            results[0], results[1], names=(state.arms[0].name, state.arms[1].name)
        )

    typer.echo(json.dumps(run_cli_command(action, debug=False), indent=2, allow_nan=False))


def _metric_rows(value: object, prefix: str = "") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        return [
            row
            for key, item in value.items()
            for row in _metric_rows(item, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, list | tuple):
        return [row for i, item in enumerate(value) for row in _metric_rows(item, f"{prefix}[{i}]")]
    return [(prefix, value)]


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if isinstance(value, str) and text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


@experiments_app.command("report")
def study_report(
    run_dir: Annotated[Path, typer.Argument()],
    output_dir: Annotated[
        Path | None, typer.Option(help="New directory for JSON, CSV and Markdown.")
    ] = None,
) -> None:
    """Summarize every native arm without running targets or inferring missing telemetry."""
    from agentic_evalkit.stats.study import summarize_study_run

    def action() -> dict[str, object]:
        state = _load_state(run_dir)
        arms: dict[str, object] = {}
        for arm in state.arms:
            if arm.status is not ArmStatus.COMPLETED:
                raise ManifestValidationError(message="report requires completed native arms")
            path = _verify_saved_report(run_dir, arm)
            if path.name == "helm-result.json":
                raise ManifestValidationError(message="use native HELM metrics for HELM arms")
            arms[arm.name] = summarize_study_run(load_run_result(path))
        report: dict[str, object] = {
            "schema_version": "1",
            "experiment": state.name,
            "input_identity": state.input_digest,
            "arms": arms,
            "qualification": (
                "Descriptive per-arm measures, not a winner claim. Missing labels and telemetry "
                "are not zeros. Use paired comparisons for declared target contrasts."
            ),
        }
        if output_dir is not None:
            rows = _metric_rows(report)
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            writer.writerow(("metric", "value"))
            writer.writerows((_csv_cell(key), _csv_cell(value)) for key, value in rows)

            def escape(value: object) -> str:
                return (
                    ("unavailable" if value is None else str(value))
                    .replace("|", "\\|")
                    .replace("\n", " ")
                    .replace("\r", " ")
                )

            markdown = "# Study measures\n\n| Metric | Value |\n|---|---|\n"
            markdown += "".join(f"| {escape(key)} | {escape(value)} |\n" for key, value in rows)
            # A new directory prevents accidental replacement of source reports or earlier exports.
            output_dir.mkdir(parents=True, exist_ok=False)
            atomic_json(output_dir / "report.json", report)
            (output_dir / "metrics.csv").write_text(stream.getvalue(), encoding="utf-8")
            (output_dir / "report.md").write_text(markdown, encoding="utf-8")
        return report

    typer.echo(json.dumps(run_cli_command(action, debug=False), indent=2, allow_nan=False))
