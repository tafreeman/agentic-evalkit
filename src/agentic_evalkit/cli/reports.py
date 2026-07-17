"""``agentic-evalkit compare``/``report``: turn a saved run into a comparison or a report.

(A few comments below cite internal project-planning labels like "Slice
4a", "Task 12", or "Task 13" -- these just point back to the
implementation phase that made or built a given decision; they're not
concepts you need to look anything up to understand.)

When ``agentic-evalkit run`` finishes, it writes one JSON file that this
project treats as the single, authoritative record of that run (a choice
made in Slice 4a) -- both commands in this module read that file back
rather than re-running anything. :func:`load_run_result` undoes what
``reporters.json.build_envelope`` did when that file was first written: it
reads the same top-level keys back out of the file and rebuilds them, via
Pydantic's ``model_validate``, into a real
:class:`~agentic_evalkit.models.EvalRunResult` object. So ``compare`` and
``report`` both end up working with the exact same frozen, structured
object a Python caller would get directly from the library -- with no
re-running of the evaluation and no network access needed.

``compare`` (implemented as Task 12) statistically compares two runs'
success rates using a paired bootstrap -- a resampling technique for
estimating how much two matched sets of results could plausibly differ
just by chance, implemented in :func:`agentic_evalkit.stats.compare_runs`.
If the two runs turn out not to be comparable at all (for example, they
used different graders or datasets), that counts as bad input from the
user rather than a system failure: the command exits with code 2 and
lists every specific mismatch it found, since the user is the one who
chose two runs that cannot be meaningfully compared. ``report``
(implemented as Task 13) instead regenerates a JSONL, Markdown, or
self-contained HTML report file from one run's saved JSON. Either way,
that saved JSON file stays the one source of truth, and everything else
here -- comparison statistics, alternate report formats -- is derived
from it, never the other way around.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from agentic_evalkit.cli.app import app, console, print_output, run_cli_command, safe_text
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.models import EvalRunResult
from agentic_evalkit.reporters import REPORTER_FORMATS
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY, Reporter, apply_redaction
from agentic_evalkit.stats import ComparisonResult, build_report_aggregates, compare_runs

__all__ = ["compare", "load_run_result", "report"]

#: The top-level keys, out of everything ``build_envelope`` writes, that are
#: actually needed to rebuild an ``EvalRunResult`` object. The envelope also
#: contains ``provenance`` and ``generated_at``, but those two are only
#: extra information echoed into the file for a human or another tool to
#: read -- they aren't fields ``EvalRunResult`` itself has, so they're
#: deliberately left out of this tuple. ``model_validate`` recomputes
#: provenance on its own anyway, from the manifest and resolved-dataset
#: values it gets back from the keys listed here.
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
    """Read a saved run's JSON file back into a real :class:`EvalRunResult` object.

    This does exactly the reverse of what ``reporters.json.build_envelope``
    did when it first wrote that file: it takes the envelope's ``run_id``,
    ``manifest``, ``resolved_dataset``, ``summary``, ``samples``,
    ``started_at``, and ``finished_at`` keys and validates them back into
    the frozen ``EvalRunResult`` model via Pydantic's ``model_validate``
    (which, along the way, also re-parses the nested manifest,
    resolved-dataset, sample, and summary objects that those keys contain).
    ``schema_version`` gets checked too, as part of ``EvalRunResult``'s own
    validation.

    Raises:
        ManifestValidationError: Raised if the file can't be found or read,
            isn't valid JSON, doesn't decode to a JSON object, is missing
            one of the required keys listed above, or fails
            ``EvalRunResult``'s own validation. ``context["errors"]`` spells
            out exactly what went wrong, so a caller sees an actionable
            message at the CLI level instead of a raw Python traceback.
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
    left: Annotated[
        Path, typer.Argument(help="Baseline run's saved JSON file (from 'agentic-evalkit run').")
    ],
    right: Annotated[
        Path, typer.Argument(help="Candidate run's saved JSON file (from 'agentic-evalkit run').")
    ],
    bootstrap_samples: Annotated[
        int,
        typer.Option(
            "--bootstrap-samples",
            help="Number of resamples to draw for the bootstrap confidence interval (100-10000).",
        ),
    ] = 1000,
    seed: Annotated[
        int,
        typer.Option(
            "--seed",
            help="Random seed for resampling, so the same seed always reproduces the same result.",
        ),
    ] = 0,
    allow_cross_environment: Annotated[
        bool,
        typer.Option(
            "--allow-cross-environment",
            help=(
                "Allow comparing two runs even if they were produced on different "
                "machines/environments or different code versions (i.e., their "
                "environment_fingerprint and/or code_fingerprint differ), instead of refusing "
                "the comparison outright (ADR-0015). Every other consistency check between the "
                "two runs is still enforced."
            ),
        ),
    ] = False,
    format_: Annotated[
        str, typer.Option("--format", help="Output format: table or json.")
    ] = "table",
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Compare two runs' paired success rates, with a confidence interval from a seeded bootstrap.

    If the two runs are not actually compatible with each other -- say, they
    used a different dataset revision, adapter, grader, target, or sampling
    policy -- that counts as the user choosing two things that can't be
    meaningfully compared, not a provider or infrastructure problem. So the
    command exits with code 2 and lists every specific mismatch it found.
    ``--allow-cross-environment`` narrowly waives just an
    environment_fingerprint and/or code_fingerprint mismatch -- meaning the
    two runs were produced on different machines/environments or different
    code versions (ADR-0015) -- while every other consistency check between
    the two runs still applies. Any field waived this way is reported back
    to you, never silently dropped.
    """

    def _action() -> ComparisonResult:
        _validate_bootstrap_samples(bootstrap_samples)
        left_run = load_run_result(left)
        right_run = load_run_result(right)
        return compare_runs(
            left_run,
            right_run,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            allow_cross_environment=allow_cross_environment,
        )

    comparison = run_cli_command(_action, debug=debug)
    if format_ == "json":
        print_output(comparison.model_dump(mode="json"), format_=format_)
        return
    waived_segment = (
        f" waived={','.join(comparison.waived_provenance_fields)}"
        if comparison.waived_provenance_fields
        else ""
    )
    console.print(
        safe_text(
            f"estimate={comparison.estimate:.4f} "
            f"2.5th={comparison.lower_percentile:.4f} "
            f"97.5th={comparison.upper_percentile:.4f} "
            f"paired={comparison.paired_count} "
            f"seed={comparison.seed}"
            f"{waived_segment}"
        ),
        soft_wrap=True,
    )


# --- report -----------------------------------------------------------------

# This dict is built directly from REPORTER_FORMATS -- the one master list
# (defined in reporters/__init__.py) of every report format this project
# knows how to write -- instead of listing the formats again by hand here.
# That way this command can never quietly drift out of sync with that
# master list: add a new format to REPORTER_FORMATS and this command can
# already regenerate it, and nothing can be written by some other path
# without also being registered there (that "every writable format must
# come from the one registry" rule is tracked as Story 2.2 / R-002). "json"
# itself is left out here because this command's whole job is to
# regenerate other formats starting from a run's canonical JSON file --
# "canonical" meaning that saved file is already treated as the one
# authoritative record of the run -- so regenerating "json" from "json"
# would be pointless.
_REPORTERS: dict[str, Reporter] = {
    name: reporter_type() for name, reporter_type in REPORTER_FORMATS.items() if name != "json"
}


@app.command()
def report(
    source: Annotated[
        Path,
        typer.Argument(
            help="The run's saved JSON file to build a report from (from 'agentic-evalkit run')."
        ),
    ],
    format_: Annotated[
        str, typer.Option("--format", help="Output format: jsonl, markdown, or html.")
    ] = "markdown",
    output: Annotated[
        Path | None, typer.Option("--output", help="Where to write the regenerated report.")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Regenerate a JSONL, Markdown, or self-contained HTML report from a run's saved JSON.

    "Self-contained" for the HTML report means that one HTML file has
    everything it needs embedded in it (styles, data, and so on) -- it
    doesn't load anything else, so it can be opened directly or emailed
    around. Before rendering, this command re-applies the project's
    default redaction policy (the step that blanks out anything that looks
    like a secret). ``run`` already writes out redacted JSON in the first
    place, so doing it again here is just a defense-in-depth safety net --
    for example, a run file produced by an older version of this tool, or
    hand-edited afterward, might not already be clean. Either way, a
    credential-shaped piece of evidence can never make it into a
    regenerated report.

    ``aggregates`` (the summary statistics shown in the report) is
    recomputed here, via :func:`agentic_evalkit.stats.build_report_aggregates`,
    rather than simply copied from the source file's own ``"aggregates"``
    key -- which only exists on JSON reports written after that field was
    added to the format. Recomputing it fresh from the reconstructed
    ``EvalRunResult`` means every regenerated report includes these
    statistics, even one regenerated from an older or hand-edited run file
    that predates that field.
    """

    def _action() -> Path:
        if format_ not in _REPORTERS:
            raise ManifestValidationError(
                message=(f"unknown report format {format_!r}; choose one of {sorted(_REPORTERS)}"),
                context={"errors": ({"path": "format", "message": f"unknown format {format_!r}"},)},
            )
        run = apply_redaction(load_run_result(source), DEFAULT_REDACTION_POLICY)
        default_suffix = _SUFFIXES.get(format_, f".{format_}")
        destination = output if output is not None else source.with_suffix(default_suffix)
        return _REPORTERS[format_].write(run, destination, aggregates=build_report_aggregates(run))

    destination = run_cli_command(_action, debug=debug)
    console.print(safe_text(f"report: {destination}"), soft_wrap=True)


#: Default output suffix per format when ``--output`` is omitted.
_SUFFIXES = {"jsonl": ".jsonl", "markdown": ".md", "html": ".html"}
