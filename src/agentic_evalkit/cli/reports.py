"""``agentic-evalkit compare``/``report``: analytics over canonical run JSON.

Both commands consume the canonical run JSON that ``run`` writes (Slice 4a's
single source of truth). :func:`load_run_result` is the inverse of
``reporters.json.build_envelope``: it reads that envelope's top-level keys
back into a real :class:`~agentic_evalkit.models.EvalRunResult` via
``model_validate``, so ``compare`` and ``report`` operate on the same
immutable contract a Python caller would, with no re-execution and no
network access.

``compare`` runs the Task 12 paired-bootstrap comparison
(:func:`agentic_evalkit.stats.compare_runs`) between two runs; incompatible
runs are surfaced as invalid *user input* (exit 2) with every mismatch
listed, since the user chose two runs that cannot be meaningfully compared.
``report`` regenerates a JSONL, Markdown, or self-contained HTML report from
one run's canonical JSON using the Task 13 reporters -- the canonical JSON
stays the source of truth and every other format derives from it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from agentic_evalkit.cli.app import app, console, print_output, run_cli_command, safe_text
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.models import EvalRunResult
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY, Reporter, apply_redaction
from agentic_evalkit.reporters.html import HtmlReporter
from agentic_evalkit.reporters.jsonl import JsonlReporter
from agentic_evalkit.reporters.markdown import MarkdownReporter
from agentic_evalkit.stats import ComparisonResult, compare_runs

__all__ = ["compare", "load_run_result", "report"]

#: The subset of ``build_envelope``'s top-level keys that reconstruct an
#: ``EvalRunResult``. ``provenance`` and ``generated_at`` are derived/echoed
#: fields in the envelope (not model fields), so they are intentionally not
#: read back here -- ``model_validate`` rebuilds provenance from the manifest
#: and resolved dataset it already carries.
_REQUIRED_ENVELOPE_KEYS = (
    "run_id",
    "manifest",
    "resolved_dataset",
    "summary",
    "samples",
    "started_at",
    "finished_at",
)


def load_run_result(path: str | Path) -> EvalRunResult:
    """Reconstruct an :class:`EvalRunResult` from a canonical run JSON file.

    The exact inverse of ``reporters.json.build_envelope``: it validates the
    envelope's ``run_id``/``manifest``/``resolved_dataset``/``summary``/
    ``samples``/``started_at``/``finished_at`` keys back into the frozen
    model via ``model_validate`` (which re-parses the nested manifest,
    resolved-dataset, sample, and summary submodels). ``schema_version`` is
    validated as part of ``EvalRunResult`` itself.

    Raises:
        ManifestValidationError: The file is missing/unreadable, is not JSON,
            does not decode to a mapping, is missing a required envelope key,
            or fails ``EvalRunResult`` validation. ``context["errors"]``
            names the problem so the failure is actionable at the CLI
            boundary rather than a raw traceback.
    """
    resolved_path = Path(path)
    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestValidationError(
            message=f"could not read run file {resolved_path}: {error}",
            context={
                "path": str(resolved_path),
                "errors": ({"path": "<file>", "message": str(error)},),
            },
        ) from error

    try:
        envelope = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ManifestValidationError(
            message=f"run file {resolved_path} is not valid JSON: {error}",
            context={
                "path": str(resolved_path),
                "errors": ({"path": "<root>", "message": str(error)},),
            },
        ) from error

    if not isinstance(envelope, dict):
        raise ManifestValidationError(
            message=f"run file {resolved_path} must decode to a JSON object",
            context={
                "path": str(resolved_path),
                "errors": ({"path": "<root>", "message": "expected a JSON object"},),
            },
        )

    missing = [key for key in _REQUIRED_ENVELOPE_KEYS if key not in envelope]
    if missing:
        raise ManifestValidationError(
            message=(
                f"run file {resolved_path} is not a canonical run report (missing keys: {missing})"
            ),
            context={
                "path": str(resolved_path),
                "errors": tuple(
                    {"path": key, "message": "required envelope key"} for key in missing
                ),
            },
        )

    model_input = {key: envelope[key] for key in _REQUIRED_ENVELOPE_KEYS}
    if "schema_version" in envelope:
        model_input["schema_version"] = envelope["schema_version"]
    try:
        return EvalRunResult.model_validate(model_input)
    except ValueError as error:
        raise ManifestValidationError(
            message=f"run file {resolved_path} failed run-result validation",
            context={
                "path": str(resolved_path),
                "errors": ({"path": "<root>", "message": str(error)},),
            },
        ) from error


# --- compare ----------------------------------------------------------------


def _validate_bootstrap_samples(value: int) -> None:
    if not (100 <= value <= 10_000):
        raise ManifestValidationError(
            message=(f"--bootstrap-samples must be in [100, 10000], got {value}"),
            context={
                "errors": ({"path": "bootstrap_samples", "message": "must be in [100, 10000]"},)
            },
        )


@app.command()
def compare(
    left: Annotated[Path, typer.Argument(help="Baseline canonical run JSON file.")],
    right: Annotated[Path, typer.Argument(help="Candidate canonical run JSON file.")],
    bootstrap_samples: Annotated[
        int,
        typer.Option("--bootstrap-samples", help="Bootstrap resamples (100-10000)."),
    ] = 1000,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the deterministic bootstrap.")] = 0,
    format_: Annotated[
        str, typer.Option("--format", help="Output format: table or json.")
    ] = "table",
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Compare two runs' paired success rates with a seeded bootstrap interval.

    Incompatible runs (different dataset revision, adapter, grader, target or
    sampling policy) are an invalid *choice of inputs*, so they exit 2 with
    every mismatch listed -- not a provider/infrastructure error.
    """

    def _action() -> ComparisonResult:
        _validate_bootstrap_samples(bootstrap_samples)
        left_run = load_run_result(left)
        right_run = load_run_result(right)
        return compare_runs(left_run, right_run, bootstrap_samples=bootstrap_samples, seed=seed)

    comparison = run_cli_command(_action, debug=debug)
    if format_ == "json":
        print_output(comparison.model_dump(mode="json"), format_=format_)
        return
    console.print(
        safe_text(
            f"estimate={comparison.estimate:.4f} "
            f"2.5th={comparison.lower_percentile:.4f} "
            f"97.5th={comparison.upper_percentile:.4f} "
            f"paired={comparison.paired_count} "
            f"seed={comparison.seed}"
        ),
        soft_wrap=True,
    )


# --- report -----------------------------------------------------------------

_REPORTERS: dict[str, Reporter] = {
    "jsonl": JsonlReporter(),
    "markdown": MarkdownReporter(),
    "html": HtmlReporter(),
}


@app.command()
def report(
    source: Annotated[Path, typer.Argument(help="Canonical run JSON file to regenerate from.")],
    format_: Annotated[
        str, typer.Option("--format", help="Output format: jsonl, markdown, or html.")
    ] = "markdown",
    output: Annotated[
        Path | None, typer.Option("--output", help="Where to write the regenerated report.")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Regenerate a JSONL, Markdown, or self-contained HTML report from run JSON.

    The default redaction policy is re-applied before rendering: ``run``
    already writes redacted canonical JSON, so this is defense in depth for
    run files produced by older tools or edited by hand -- a credential-shaped
    evidence value can never reach a regenerated report either way.
    """

    def _action() -> Path:
        if format_ not in _REPORTERS:
            raise ManifestValidationError(
                message=(f"unknown report format {format_!r}; choose one of {sorted(_REPORTERS)}"),
                context={"errors": ({"path": "format", "message": f"unknown format {format_!r}"},)},
            )
        run = apply_redaction(load_run_result(source), DEFAULT_REDACTION_POLICY)
        destination = output if output is not None else source.with_suffix(_SUFFIXES[format_])
        return _REPORTERS[format_].write(run, destination)

    destination = run_cli_command(_action, debug=debug)
    console.print(safe_text(f"report: {destination}"), soft_wrap=True)


#: Default output suffix per format when ``--output`` is omitted.
_SUFFIXES = {"jsonl": ".jsonl", "markdown": ".md", "html": ".html"}
