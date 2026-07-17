"""``agentic-evalkit init``/``validate``/``run``: create, check, and run an evaluation manifest.

A "manifest" is the config file that describes one evaluation run: which
dataset, which adapter, which grader, and which target (the system being
tested) to use.

``init`` writes a starting manifest from a curated preset (a ready-made
template) -- or from an explicit dataset/adapter/grader combination if one is
named directly. If the developer doesn't point it at their own system to
test, ``init --preset gsm8k`` wires in the packaged
:mod:`agentic_evalkit.examples.zero_target` "smoke target" -- a trivial
stand-in system under test, not a real one -- purely so the whole pipeline
can be run immediately, end to end, with no setup (plan Task 14 Step 7).
``validate`` loads a manifest file through
:func:`~agentic_evalkit.manifest.load_manifest` and confirms it's
well-formed, without actually running anything. ``run`` loads a manifest,
turns its target section into a concrete, callable
:class:`~agentic_evalkit.targets.base.ExecutionTarget` (the actual system
being evaluated), prints a summary of what it's about to do before doing it,
streams live progress (via the Rich library) from
:class:`~agentic_evalkit.runner.EvalRunner` as samples finish, and writes the
standard run report as JSON with secrets stripped out per the default
redaction policy (design §12).

``_KNOWN_GRADERS`` (below) also registers two graders that a manifest has to
opt into by name -- they are never used unless a manifest explicitly asks for
them -- built on the packaged
:class:`~agentic_evalkit.examples.reference_judge.ReferenceJudgeClient`.
Naming ``"judge-reference@1"`` or ``"composite-reference@1"`` in a manifest
lets you exercise the full "judge" grading pipeline (an LLM grading another
system's output) end to end, without needing a real LLM provider configured.
Both are permanently *uncalibrated* -- meaning nobody has proven, using
held-out test cases, that this judge is accurate enough to trust -- so
neither one can ever "hard-gate" a release (i.e. neither can single-handedly
fail a build; design §9). They exist only to prove the judge/composite code
paths actually run, the same way the ``gsm8k`` preset's ``zero_target``
proves the execution path runs -- not to produce a grade anyone should act
on. Using either is always a deliberate, explicit choice a manifest author
makes by naming it; the default preset-generated manifests
(``init --preset ...``) never do.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import typer
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.text import Text

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.benchmarks.grounding import GroundedCitationAdapter
from agentic_evalkit.benchmarks.gsm8k import Gsm8kAdapter
from agentic_evalkit.benchmarks.swebench import SweBenchVerifiedAdapter
from agentic_evalkit.benchmarks.swebench_docker import (
    SweBenchDockerHarnessExecutor,
    swebench_prediction,
)
from agentic_evalkit.cli.app import ExitCode, app, console, run_cli_command, safe_text
from agentic_evalkit.cli.datasets import build_catalog
from agentic_evalkit.datasets.presets import BUILTIN_PRESETS, DatasetPreset
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.events import ExecutionCompleted, RunEvent, SampleCompleted, SampleStarted
from agentic_evalkit.examples.reference_judge import ReferenceJudgeClient
from agentic_evalkit.graders.composite import CompositeGrader, WeightedGrader
from agentic_evalkit.graders.exact import ExactMatchGrader
from agentic_evalkit.graders.grounding import build_grounded_citation_grader
from agentic_evalkit.graders.harness import HarnessGrader
from agentic_evalkit.graders.judge import JudgeGrader
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
    EvalRunResult,
    ResolvedDataset,
    SamplingPolicy,
    SourceRecord,
)
from agentic_evalkit.provenance import (
    compute_code_fingerprint,
    compute_environment_fingerprint,
    compute_target_fingerprint,
)
from agentic_evalkit.reporters import REPORTER_FORMATS
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY, apply_redaction
from agentic_evalkit.runner import EvalRunner
from agentic_evalkit.stats import build_report_aggregates
from agentic_evalkit.targets.callable import CallableTarget
from agentic_evalkit.targets.http import HttpTarget
from agentic_evalkit.targets.subprocess import SubprocessTarget

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agentic_evalkit.benchmarks.base import BenchmarkAdapter
    from agentic_evalkit.datasets.catalog import DatasetCatalog
    from agentic_evalkit.graders.base import Grader
    from agentic_evalkit.targets.base import ExecutionTarget

__all__ = ["build_target_for_document", "init", "run", "validate", "write_canonical_report"]


class _RunnerCatalogAdapter:
    """Bridges a real :class:`DatasetCatalog` to the smaller interface ``EvalRunner`` expects.

    ``EvalRunner`` only knows about a minimal, local interface: something
    with a ``resolve(ref) -> ResolvedDataset`` method and an
    ``iter_records(dataset, *, offset, limit) -> AsyncIterator[SourceRecord]``
    method (see ``agentic_evalkit.runner._CatalogProtocol`` for that exact
    shape). The real ``DatasetCatalog.iter_records`` needs more than that: it
    also needs the original ``DatasetRef`` on every call, because that's the
    only way it can tell which provider (e.g. Hugging Face vs. local files) a
    given ``ResolvedDataset`` came from. It also takes an ``offline`` flag on
    every call, and ``EvalRunner``'s minimal interface has no place to pass
    one.

    This class exists to close that gap without changing either side. A CLI
    run picks one ``ref`` and one ``offline`` setting at the start and keeps
    them fixed for the whole run (ADR-0010), so this adapter just remembers
    those two values once (in its constructor) and supplies them itself on
    every call into the real catalog. That way neither ``DatasetCatalog`` nor
    ``EvalRunner``/``_CatalogProtocol`` has to change its method signatures
    to match the other -- this class quietly absorbs the difference.
    """

    def __init__(self, catalog: DatasetCatalog, ref: DatasetRef, *, offline: bool = False) -> None:
        self._catalog = catalog
        self._ref = ref
        self._offline = offline

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        return await self._catalog.resolve(ref, offline=self._offline)

    def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        return self._catalog.iter_records(
            self._ref, dataset, offset=offset, limit=limit, offline=self._offline
        )


_DEMO_TARGET_IMPORT_STRING = "agentic_evalkit.examples.zero_target:zero_target"

#: Adapters/graders the CLI knows how to build, indexed by name. This module
#: deliberately hardcodes this small, fully-tested lookup table instead of
#: scanning for and auto-loading plugins at runtime (more flexible, but
#: harder to test and reason about). Every name a curated preset can
#: reference already resolves here (both ``gsm8k`` and ``swe-bench-verified``
#: do); ``grounded-citation-tasks@1`` has no curated preset yet, so it's only
#: reachable if someone hand-writes a manifest that names it directly
#: (ADR-0012).
_KNOWN_ADAPTERS: dict[str, BenchmarkAdapter] = {
    "gsm8k@1": Gsm8kAdapter(),
    "grounded-citation-tasks@1": GroundedCitationAdapter(),
    "swebench-verified@1": SweBenchVerifiedAdapter(),
}


def _extract_answer(output: object) -> str:
    if isinstance(output, dict):
        value = output.get("answer")
        if value is not None:
            return str(value)
    return ""


def _build_known_graders() -> dict[str, Grader]:
    """Construct every grader name the CLI can resolve from a manifest.

    A factory function (rather than a bare module-level dict literal, the
    prior shape) purely for readability now that building the composite
    entry takes a few steps; behavior and the resulting mapping's shape are
    unchanged, still evaluated once at import time.
    """
    exact_match = ExactMatchGrader(name="normalized-exact@1", extractor=_extract_answer)
    # calibration=None means nobody has verified this judge is accurate
    # enough to trust (there's no track record proving it against
    # known-correct/incorrect examples). Because of that, JudgeGrader.grade
    # always forces hard_gate=False -- "never let this result block a
    # release" -- no matter what gate=True below might otherwise suggest
    # (design §9). So choosing this grader can never make a run's pass/fail
    # outcome depend on an unproven judge's opinion.
    judge_reference = JudgeGrader(
        ReferenceJudgeClient(), calibration=None, gate=True, name="judge-reference@1"
    )
    composite_reference = CompositeGrader(
        name="composite-reference@1",
        graders=(
            # The exact-match check (a deterministic, rule-based grader) is
            # the one allowed to hard-gate this composite -- i.e. it alone
            # can fail the run (design §9's rule: "a model judge is never the
            # first check for anything an objective, rule-based grader can
            # decide"). The judge grader alongside it only ever contributes
            # an advisory score: it adds to the overall number but can never
            # by itself fail the run.
            WeightedGrader(exact_match, weight=0.7, hard_gate=True),
            WeightedGrader(
                JudgeGrader(
                    ReferenceJudgeClient(), calibration=None, gate=False, name="judge-reference@1"
                ),
                weight=0.3,
                hard_gate=False,
            ),
        ),
    )
    # A composite check for the "grounded citation" task, where the
    # rule-based (deterministic) part does almost all the work (ADR-0012).
    # It has two components: a deterministic grader that checks the basic
    # mechanics of citation grounding, and an LLM-judge component (here, the
    # packaged reference client) that would separately judge citation
    # "sufficiency." The judge component is permanently uncalibrated here --
    # nobody has proven it's accurate -- so it is wired at weight 0.0: it
    # still runs and its verdict gets recorded, but it can never move the
    # score or block a run. The deterministic component is the only one
    # allowed to hard-gate (i.e. actually fail the run).
    grounded_citation = build_grounded_citation_grader(judge_client=ReferenceJudgeClient())
    # Real (not a placeholder/reference) grader for the SWE-bench benchmark
    # (ADR-0014) -- it actually checks whether a code patch fixes the issue,
    # using a Docker-based test harness. The module that defines the executor
    # can be imported even on a plain install (no extra packages needed), but
    # at run time it reports status UNAVAILABLE unless both the optional
    # ``agentic-evalkit[swebench]`` extra packages and a running Docker
    # daemon are actually present. So just registering this grader here
    # never forces someone who installed the base package (without the
    # SWE-bench extras) to have docker/swebench importable.
    swebench_harness = HarnessGrader(
        executor=SweBenchDockerHarnessExecutor(),
        predictor=swebench_prediction,
        benchmark="swebench-verified@1",
        name="swebench-harness@1",
    )
    return {
        "normalized-exact@1": exact_match,
        "judge-reference@1": judge_reference,
        "composite-reference@1": composite_reference,
        "grounded-citation@1": grounded_citation,
        "swebench-harness@1": swebench_harness,
    }


_KNOWN_GRADERS: dict[str, Grader] = _build_known_graders()


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
        # Copy the preset's data-contamination warning label onto the
        # manifest too, so it later reaches the run report's
        # resolved_dataset field (ADR-0013) instead of sitting unused on the
        # preset object alone. ("Contamination" = the risk that the system
        # under test already saw this dataset during its own training,
        # which would make a high score misleading -- see
        # _with_contamination's docstring below for the full explanation.)
        contamination=preset.contamination,
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
    # The manifest file itself never contains an actual secret/API key
    # (design §12). Instead it can name a credential_hook -- a string this
    # code looks up as an environment-variable name, right here, at run
    # time. The resulting token is never written into or saved back to the
    # manifest file.
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


def _with_provenance_fingerprints(
    manifest: EvalRunManifest,
    target_config: CallableTargetConfig | SubprocessTargetConfig | HttpTargetConfig,
) -> EvalRunManifest:
    """Return a copy of ``manifest`` with all three "provenance" fingerprints filled in.

    A "fingerprint" here is a hash that captures exactly what ran: which
    Python interpreter/package versions, which code, and which target.
    Recording these is what lets us later prove two runs are truly
    comparable (or catch it when they aren't) -- that proof is what
    "provenance" means throughout this codebase.

    Design §5.6 describes ``environment_fingerprint``/``code_fingerprint``/
    ``target_fingerprint`` as the fields that pin down "environment and code
    fingerprints" and "execution target and target fingerprint" for a run.
    But before this function was added, nothing in the CLI actually called
    the generator functions in :mod:`agentic_evalkit.provenance` -- so every
    real ``run`` wrote ``None`` into all three fields, no matter what a
    hand-written manifest file put there. Whatever this function computes
    always overrides anything already in the manifest: a fingerprint is a
    claim about what *this specific run* actually used, and that isn't
    something a person can honestly fill in by hand in advance -- it has to
    be measured at run time.

    This function is called exactly once, right before the preflight summary
    and the run itself, so that everything downstream -- the preflight
    message printed to the console, ``EvalRunner``, the JSON report, and
    that report's provenance section -- all see the exact same fingerprinted
    manifest. There's no second, separate fingerprinting step anywhere else
    that could drift out of sync with this one.

    This function does not modify ``manifest`` in place (we never mutate
    data in this codebase -- see ADR-0002): it returns a brand-new
    ``EvalRunManifest`` via ``model_copy``, the same pattern used one line
    above, at this function's only call site, for the ``--limit`` override.
    """
    return manifest.model_copy(
        update={
            "environment_fingerprint": compute_environment_fingerprint(),
            "code_fingerprint": compute_code_fingerprint(),
            "target_fingerprint": compute_target_fingerprint(target_config.model_dump(mode="json")),
        }
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


def _with_contamination(result: EvalRunResult, manifest: EvalRunManifest) -> EvalRunResult:
    """Copy the manifest's "contamination" warning label onto the run report's resolved dataset.

    "Contamination" here means: has this dataset been public, and mirrored
    elsewhere, for long enough that the system under test might have already
    seen these exact questions and answers during its own training? If so, a
    high score doesn't prove real capability -- it might just mean the
    answers were memorized. ADR-0013 lets a preset carry a best-effort
    ``SUSPECT`` label flagging that risk.

    That label lives on ``DatasetPreset`` and gets copied onto the manifest
    by ``_manifest_document_for_preset``, but the dataset provider's
    ``resolve`` method has no way to see the original preset. So without
    this function, the run report's ``resolved_dataset.contamination`` field
    would always end up ``None`` -- silently dropping the warning the label
    exists to surface. A value the provider resolved on its own always wins
    over the manifest's value; the manifest's label only fills in when the
    provider didn't already set one. As always in this codebase, nothing is
    modified in place (ADR-0002): this returns a new ``EvalRunResult`` via
    ``model_copy``.
    """
    if manifest.contamination is None or result.resolved_dataset.contamination is not None:
        return result
    return result.model_copy(
        update={
            "resolved_dataset": result.resolved_dataset.model_copy(
                update={"contamination": manifest.contamination}
            )
        }
    )


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
    manifest = _with_provenance_fingerprints(manifest, document.target)

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
        # Rich's Progress.update() expects `description` to be a plain `str`
        # (it won't accept the Text/RenderableType wrapper that safe_text()
        # produces), so this can't be routed through safe_text() the way the
        # table cells and console.print() calls elsewhere in this file are.
        # safe_text() exists to stop a string like "[bold]" from being
        # misread as Rich markup instead of literal text -- normally a
        # concern for text sourced from an external provider or dataset
        # card. Skipping it here is safe because sample_id is always
        # generated by our own adapter code (e.g. "gsm8k:0", built from an
        # f-string over a row index in Gsm8kAdapter.prepare), never raw text
        # from a provider or dataset card, so it can never contain a
        # "[...]"-style sequence that Rich would misinterpret.
        if isinstance(event, SampleStarted):
            progress.update(
                task_id, description=f"sample {event.sample_id} (attempt {event.attempt})"
            )
        elif isinstance(event, ExecutionCompleted):
            progress.update(task_id, description=f"executed {event.sample_id}: {event.status}")
        elif isinstance(event, SampleCompleted):
            progress.advance(task_id)

    def _action():  # type: ignore[no-untyped-def]
        catalog = _RunnerCatalogAdapter(build_catalog(offline=offline), ref, offline=offline)
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
    result = _with_contamination(result, manifest)

    report_path = write_canonical_report(result, output_dir or _default_output_dir())

    summary = result.summary
    console.print(
        f"[bold]outcomes[/bold]: total={summary.total} passed={summary.passed} "
        f"failed={summary.failed} partial={summary.partial} errors={summary.errors} "
        f"timeouts={summary.timeouts} cancelled={summary.cancelled} "
        f"abstained={summary.abstained} unavailable={summary.unavailable}"
    )
    # soft_wrap=True because the report path must appear on stdout as one
    # single, unbroken piece of text (both scripts and our own tests search
    # the output for the exact path string) -- it must never get broken
    # across multiple lines the way Rich's normal word-wrapping would do
    # once it hits the console's width.
    console.print(safe_text(f"report: {report_path}"), soft_wrap=True)

    if summary.errors > 0 or summary.timeouts > 0:
        raise typer.Exit(code=int(ExitCode.INFRASTRUCTURE_ERROR))


def _run_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


def write_canonical_report(result: EvalRunResult, destination_dir: Path) -> Path:
    """Write ``result`` to disk as the standard run-report JSON, with secrets stripped out first.

    The secret-stripping step ("redaction" -- see design §12) happens
    exactly once, right here, before anything is written to disk. That
    guarantees every downstream reader of this file -- the
    ``report``/``compare`` commands and anything else that opens it later --
    only ever sees data that never contained an un-redacted API key or
    token. This function is exposed publicly (like
    :func:`build_target_for_document`) specifically so tests can check that
    redaction actually happens, without needing to run a whole evaluation
    first.

    ``aggregates`` is computed via
    :func:`agentic_evalkit.stats.build_report_aggregates`, from the result
    *before* redaction. It includes things like a "Wilson-bounded" pass rate
    (a statistically conservative version of the raw pass rate that accounts
    for how small the sample was), how resources like time/tokens were
    distributed across samples, and "pass@k" (the fraction of samples where
    at least one of several repeated attempts succeeded, when the manifest
    configured more than one attempt per sample). None of that touches the
    actual evidence/output text that redaction is scrubbing -- aggregation
    only ever looks at counts, scores, and resource numbers -- so computing
    it before or after redaction gives the exact same numbers either way.
    It's done before redaction here simply so that this function's one call
    to the redaction step stays the single, obvious place in the code where
    secrets are ever removed.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    report_path = destination_dir / f"{result.run_id}.json"
    aggregates = build_report_aggregates(result)
    # Look up the reporter through the central REPORTER_FORMATS registry
    # rather than constructing one directly, so that only formats we've
    # actually registered here -- and therefore verified go through
    # redaction -- can be written. Redaction itself is still applied exactly
    # once, immediately before the write (design §12).
    reporter = REPORTER_FORMATS["json"]()
    reporter.write(
        apply_redaction(result, DEFAULT_REDACTION_POLICY), report_path, aggregates=aggregates
    )
    return report_path


def build_target_for_document(document: ManifestDocument) -> ExecutionTarget:
    """Construct the concrete execution target ``document.target`` describes.

    Exposed so tests/tools can build the same target a ``run`` invocation
    would use without duplicating the ``_build_target`` dispatch table.
    """
    return _build_target(document.target)
