"""``agentic-evalkit init``/``validate``/``run``: manifest lifecycle and execution.

``init`` writes a starting manifest from a curated preset (or an explicit
dataset/adapter/grader triple); when the developer supplies no target of
their own, ``init --preset gsm8k`` wires in the packaged
:mod:`agentic_evalkit.examples.zero_target` smoke target so the pipeline is
runnable immediately (plan Task 14 Step 7). ``validate`` round-trips a
manifest file through :func:`~agentic_evalkit.manifest.load_manifest`
without running anything. ``run`` loads a manifest, resolves its CLI target
block into a concrete :class:`~agentic_evalkit.targets.base.ExecutionTarget`,
prints a preflight summary, streams Rich progress from
:class:`~agentic_evalkit.runner.EvalRunner` events, and writes the canonical
run JSON report.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.text import Text

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.benchmarks.gsm8k import Gsm8kAdapter
from agentic_evalkit.cli.app import ExitCode, app, console, run_cli_command, safe_text
from agentic_evalkit.cli.datasets import build_catalog
from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.presets import BUILTIN_PRESETS, DatasetPreset
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.events import ExecutionCompleted, RunEvent, SampleCompleted, SampleStarted
from agentic_evalkit.graders.exact import ExactMatchGrader
from agentic_evalkit.manifest import (
    CallableTargetConfig,
    HttpTargetConfig,
    ManifestDocument,
    SubprocessTargetConfig,
    dump_manifest,
    load_manifest,
)
from agentic_evalkit.models import (
    DatasetRef,
    DatasetSelection,
    EvalRunManifest,
    ResolvedDataset,
    SamplingPolicy,
    SourceRecord,
)
from agentic_evalkit.reporters.json import JsonReporter
from agentic_evalkit.runner import EvalRunner
from agentic_evalkit.targets.base import ExecutionTarget
from agentic_evalkit.targets.callable import CallableTarget
from agentic_evalkit.targets.http import HttpTarget
from agentic_evalkit.targets.subprocess import SubprocessTarget

__all__ = ["build_target_for_document", "init", "run", "validate"]


class _RunnerCatalogAdapter:
    """Adapts a real :class:`DatasetCatalog` to ``EvalRunner``'s narrower catalog shape.

    ``EvalRunner`` depends only on a local, minimal protocol: ``resolve(ref)
    -> ResolvedDataset`` plus ``iter_records(dataset, *, offset, limit) ->
    AsyncIterator[SourceRecord]`` (see ``agentic_evalkit.runner._CatalogProtocol``).
    ``DatasetCatalog.iter_records`` additionally requires the original
    ``DatasetRef`` for provider routing (a page/record-iteration call has no
    other way to know which provider a ``ResolvedDataset`` came from). This
    adapter closes over the one ``ref`` a CLI run always has fixed for its
    whole duration and forwards to the real catalog with it, so
    ``DatasetCatalog`` never needs to satisfy a protocol it cannot
    structurally satisfy on its own.
    """

    def __init__(self, catalog: DatasetCatalog, ref: DatasetRef) -> None:
        self._catalog = catalog
        self._ref = ref

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        return await self._catalog.resolve(ref)

    def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        return self._catalog.iter_records(self._ref, dataset, offset=offset, limit=limit)


_DEMO_TARGET_IMPORT_STRING = "agentic_evalkit.examples.zero_target:zero_target"

#: Adapters/graders the runnable objective-only CLI knows how to construct
#: by name. This module intentionally hardcodes this small, fully-tested
#: table rather than doing dynamic plugin discovery for adapters/graders
#: (that is out of Task 14's scope); every name here matches a value used by
#: a ``BUILTIN_PRESETS`` entry, so any preset-based manifest resolves.
_KNOWN_ADAPTERS = {"gsm8k@1": Gsm8kAdapter()}


def _extract_answer(output: object) -> str:
    if isinstance(output, dict):
        value = output.get("answer")
        if value is not None:
            return str(value)
    return ""


_KNOWN_GRADERS = {
    "normalized-exact@1": ExactMatchGrader(name="normalized-exact@1", extractor=_extract_answer)
}


# --- init ---------------------------------------------------------------


@app.command()
def init(
    preset: Annotated[
        str | None, typer.Option("--preset", help="A curated preset name, e.g. 'gsm8k'.")
    ] = None,
    output: Annotated[
        Path, typer.Option("--output", help="Where to write the manifest YAML.")
    ] = Path("eval.yaml"),
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing manifest file.")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Write a starting manifest from a curated preset."""

    def _action() -> ManifestDocument:
        if preset is None:
            raise ManifestValidationError(
                message="agentic-evalkit init requires --preset (e.g. --preset gsm8k)",
                context={"errors": ({"path": "preset", "message": "required"},)},
            )
        if preset not in BUILTIN_PRESETS:
            raise ManifestValidationError(
                message=f"unknown preset {preset!r}; run 'agentic-evalkit datasets curated'",
                context={"errors": ({"path": "preset", "message": f"unknown preset {preset!r}"},)},
            )
        if output.exists() and not force:
            raise ManifestValidationError(
                message=f"{output} already exists; pass --force to overwrite it",
                context={"errors": ({"path": "output", "message": "file already exists"},)},
            )
        return _manifest_document_for_preset(BUILTIN_PRESETS[preset])

    document = run_cli_command(_action, debug=debug)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dump_manifest(document), encoding="utf-8")
    console.print(safe_text(f"Wrote manifest for preset {preset!r} to {output}"), soft_wrap=True)


def _manifest_document_for_preset(preset: DatasetPreset) -> ManifestDocument:
    manifest = EvalRunManifest(
        run_name=f"{preset.name}-quickstart",
        dataset_ref=preset.ref,
        adapter=preset.adapter,
        grader=preset.grader,
        target_name="cli-target",
        selection=DatasetSelection(limit=10),
        sampling=SamplingPolicy(attempts=1),
        attempts=1,
        timeout_seconds=30.0,
        concurrency=1,
    )
    return ManifestDocument(
        manifest=manifest,
        target=CallableTargetConfig(import_string=_DEMO_TARGET_IMPORT_STRING),
    )


# --- validate -------------------------------------------------------------


@app.command()
def validate(
    manifest_path: Annotated[Path, typer.Argument(help="Path to a manifest YAML file.")],
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Validate a manifest file without running anything."""
    document = run_cli_command(lambda: load_manifest(manifest_path), debug=debug)
    prefix = Text("valid", style="bold green")
    detail = safe_text(f": {manifest_path} (run_name={document.manifest.run_name!r})")
    console.print(Text.assemble(prefix, detail), soft_wrap=True)


# --- run --------------------------------------------------------------------


def _load_callable_target(config: CallableTargetConfig) -> ExecutionTarget:
    module_name, _, attr_name = config.import_string.partition(":")
    if not module_name or not attr_name:
        raise ManifestValidationError(
            message=(
                f"callable target import string {config.import_string!r} must be of the form "
                "'module.path:attribute'"
            ),
            context={"import_string": config.import_string},
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ManifestValidationError(
            message=f"could not import module {module_name!r} for target: {error}",
            context={"import_string": config.import_string},
        ) from error
    try:
        func = getattr(module, attr_name)
    except AttributeError as error:
        raise ManifestValidationError(
            message=f"module {module_name!r} has no attribute {attr_name!r}",
            context={"import_string": config.import_string},
        ) from error
    return CallableTarget(func, name=config.import_string)


def _load_subprocess_target(config: SubprocessTargetConfig) -> ExecutionTarget:
    return SubprocessTarget(command=config.argv)


def _load_http_target(config: HttpTargetConfig) -> ExecutionTarget:
    # The manifest never carries a literal credential (design §12); a named
    # credential_hook resolves to an environment variable read here, at run
    # time, not stored in or dumped back into the manifest file.
    headers = None
    if config.credential_hook:
        import os

        token = os.environ.get(config.credential_hook)
        if token:

            def _headers() -> dict[str, str]:
                return {"Authorization": f"Bearer {token}"}

            headers = _headers
    client = httpx.AsyncClient(timeout=30.0)
    return HttpTarget(client=client, url=config.url, name=config.url, headers=headers)


def _build_target(
    config: CallableTargetConfig | SubprocessTargetConfig | HttpTargetConfig,
) -> ExecutionTarget:
    if isinstance(config, CallableTargetConfig):
        return _load_callable_target(config)
    if isinstance(config, SubprocessTargetConfig):
        return _load_subprocess_target(config)
    return _load_http_target(config)


def _require_known_component(name: str, table: dict[str, object], *, kind: str) -> None:
    if name not in table:
        raise ManifestValidationError(
            message=(
                f"manifest references {kind} {name!r}, which this CLI does not know how to "
                f"construct; known {kind}s: {sorted(table)}"
            ),
            context={"errors": ({"path": kind, "message": f"unknown {kind} {name!r}"},)},
        )


def _preflight_summary(manifest: EvalRunManifest, target_config: object, *, ref: DatasetRef) -> str:
    return (
        f"dataset={ref.provider}:{ref.dataset_id} "
        f"config={ref.config or '<auto>'} split={ref.split or '<auto>'} "
        f"adapter={manifest.adapter} grader={manifest.grader} "
        f"target={type(target_config).__name__} "
        f"limit={manifest.selection.limit} attempts={manifest.attempts} "
        f"concurrency={manifest.concurrency}"
    )


def _default_output_dir() -> Path:
    return Path("agentic-evalkit-runs")


def _raise_yes_required() -> None:
    raise ManifestValidationError(
        message="run is noninteractive; pass --yes to proceed without a prompt",
        context={"errors": ({"path": "yes", "message": "required when noninteractive"},)},
    )


@app.command()
def run(
    manifest_path: Annotated[Path, typer.Argument(help="Path to a manifest YAML file.")],
    limit: Annotated[
        int | None, typer.Option("--limit", help="Override the manifest's sample limit.")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", help="Directory for the JSON report.")
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", help="Skip the interactive confirmation prompt.")
    ] = False,
    offline: Annotated[
        bool, typer.Option("--offline", help="Only use exact cached dataset pages.")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Run a manifest end to end and write a canonical JSON report."""
    document = run_cli_command(lambda: load_manifest(manifest_path), debug=debug)
    manifest = document.manifest
    if limit is not None:
        manifest = manifest.model_copy(
            update={"selection": manifest.selection.model_copy(update={"limit": limit})}
        )

    ref = manifest.dataset_ref
    prefix = Text("preflight", style="bold")
    detail = safe_text(f": {_preflight_summary(manifest, document.target, ref=ref)}")
    console.print(Text.assemble(prefix, detail), soft_wrap=True)

    if not yes and not sys.stdin.isatty():
        run_cli_command(_raise_yes_required, debug=debug)
    elif not yes:
        confirmed = typer.confirm("Proceed with this run?", default=True)
        if not confirmed:
            console.print("[yellow]cancelled[/yellow]")
            raise typer.Exit(code=int(ExitCode.CANCELLED))

    def _validate_components() -> None:
        _require_known_component(manifest.adapter, _KNOWN_ADAPTERS, kind="adapter")  # type: ignore[arg-type]
        _require_known_component(manifest.grader, _KNOWN_GRADERS, kind="grader")  # type: ignore[arg-type]

    run_cli_command(_validate_components, debug=debug)
    target = run_cli_command(lambda: _build_target(document.target), debug=debug)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    task_id = progress.add_task("running", total=manifest.selection.limit or None)

    def _sink(event: RunEvent) -> None:
        # Progress.update's `description` is typed as plain `str` (Rich does
        # not accept a Text/RenderableType here), so it cannot use safe_text
        # the way table cells and console.print calls above do. This is an
        # accepted, narrower risk than the rest of the module: sample_id
        # here is always adapter-generated (e.g. "gsm8k:0" from
        # Gsm8kAdapter.prepare's f-string over a row index), never raw
        # provider/dataset-card text, so it cannot contain "[...]".
        if isinstance(event, SampleStarted):
            progress.update(
                task_id, description=f"sample {event.sample_id} (attempt {event.attempt})"
            )
        elif isinstance(event, ExecutionCompleted):
            progress.update(task_id, description=f"executed {event.sample_id}: {event.status}")
        elif isinstance(event, SampleCompleted):
            progress.advance(task_id)

    def _action():  # type: ignore[no-untyped-def]
        catalog = _RunnerCatalogAdapter(build_catalog(offline=offline), ref)
        artifact_store = ArtifactStore(
            (output_dir or _default_output_dir()) / "artifacts" / _run_stamp()
        )
        runner = EvalRunner(
            catalog=catalog,
            adapters=_KNOWN_ADAPTERS,
            targets={"cli-target": target},
            graders=_KNOWN_GRADERS,
            artifact_store=artifact_store,
        )
        with progress:
            return asyncio.run(runner.run(manifest, event_sink=_sink))

    result = run_cli_command(_action, debug=debug)

    destination_dir = output_dir or _default_output_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)
    report_path = destination_dir / f"{result.run_id}.json"
    JsonReporter().write(result, report_path)

    summary = result.summary
    console.print(
        f"[bold]outcomes[/bold]: total={summary.total} passed={summary.passed} "
        f"failed={summary.failed} partial={summary.partial} errors={summary.errors} "
        f"timeouts={summary.timeouts} cancelled={summary.cancelled} "
        f"abstained={summary.abstained} unavailable={summary.unavailable}"
    )
    # soft_wrap=True: a report path must appear on stdout as one unbroken
    # substring (scripts and tests grep for the exact path), never wrapped
    # mid-string at the console width the way default word-wrapping would.
    console.print(safe_text(f"report: {report_path}"), soft_wrap=True)

    if summary.errors > 0 or summary.timeouts > 0:
        raise typer.Exit(code=int(ExitCode.INFRASTRUCTURE_ERROR))


def _run_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


def build_target_for_document(document: ManifestDocument) -> ExecutionTarget:
    """Construct the concrete execution target ``document.target`` describes.

    Exposed so tests/tools can build the same target a ``run`` invocation
    would use without duplicating the ``_build_target`` dispatch table.
    """
    return _build_target(document.target)
