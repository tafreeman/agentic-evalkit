"""``agentic-evalkit calibrate``: measure a judge against labeled answers (ADR-0024).

This is the command that makes the release gate two-sided. Everything needed
to *reject* an unproven judge already shipped -- ``JudgeGrader`` refuses to
hard-gate without calibration, and ``judge_authority`` explains why -- but
nothing shipped that could produce the evidence to get through it, so the
only route to a gating judge was to hand-write four counts and invent a
fingerprint. This command runs the judge over examples somebody already knows
the answers to, writes the resulting
:class:`~agentic_evalkit.graders.calibration.CalibrationArtifact` as JSON, and
prints what that artifact entitles the judge to do.

Two behaviours here are decisions rather than conveniences:

* **A thin or failing calibration is still written out.** It exits nonzero
  and says plainly that the judge cannot gate, but the artifact is produced.
  Refusing to write one would leave a user unable to record the honest state
  of a judge they have only partially measured, and would introduce a third
  policy into a codebase that deliberately keeps one -- the two-tier rule
  (D-1 as amended 2026-07-04) already distinguishes thin evidence from bad
  evidence, and thin evidence is a fact worth recording (ADR-0024).
* **The verdict is not computed here.** It comes from
  :func:`~agentic_evalkit.integrations.base.judge_authority`, called on the
  artifact this command just produced -- the same function a host-platform
  export calls, reading the same thresholds off the artifact itself. A
  second implementation of "is this good enough?" that agreed today would be
  free to drift tomorrow, and the whole value of the control is that it
  cannot.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

import typer
from pydantic import ValidationError
from rich.table import Table
from rich.text import Text

from agentic_evalkit.cli.app import (
    ExitCode,
    app,
    console,
    print_output,
    run_cli_command,
    safe_text,
)
from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.errors import (
    DatasetNotFound,
    DatasetSchemaMismatch,
    ManifestValidationError,
)
from agentic_evalkit.examples.reference_judge import ReferenceJudgeClient
from agentic_evalkit.graders.calibration import (
    PROJECT_MIN_TPR,
    AuthorityLevel,
    judge_authority,
)
from agentic_evalkit.graders.measure import DEFAULT_PASS_SCORE_THRESHOLD, measure_calibration
from agentic_evalkit.models import DatasetRef, LabeledJudgeSample

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from agentic_evalkit.graders.calibration import CalibrationArtifact
    from agentic_evalkit.graders.judge import JudgeClient

__all__ = ["calibrate"]

T = TypeVar("T")

#: The name that resolves to the packaged stand-in judge. Spelled as a short
#: word rather than an import string because it is the one judge this package
#: ships, and because ``calibrate <set> --judge reference`` is what somebody
#: reaches for to see the command work before they have wired up their own.
#: Its calibration is real -- genuinely measured against whatever labels it
#: was given -- but it describes a substring matcher, not a model, so it is
#: evidence about this pipeline rather than about anyone's judge.
_REFERENCE_JUDGE_NAME = "reference"


def _run_async(coroutine_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run one coroutine to completion; the CLI is not itself async."""
    return asyncio.run(coroutine_factory())


def _load_judge(spec: str) -> JudgeClient:
    """Resolve ``--judge`` to a live judge client.

    Accepts either the packaged reference judge by name, or a
    ``module.path:attribute`` import string naming a zero-argument callable
    that returns a :class:`~agentic_evalkit.graders.judge.JudgeClient` --
    the same import-string shape ``runs.py`` already uses for a callable
    target, so a user who has written one has written the other.
    """
    if spec == _REFERENCE_JUDGE_NAME:
        return ReferenceJudgeClient()
    module_name, _, attr_name = spec.partition(":")
    if not module_name or not attr_name:
        raise ManifestValidationError(
            message=(
                f"judge {spec!r} must be either {_REFERENCE_JUDGE_NAME!r} or an import string "
                "of the form 'module.path:factory'"
            ),
            context={"judge": spec},
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ManifestValidationError(
            message=f"could not import module {module_name!r} for judge: {error}",
            context={"judge": spec},
        ) from error
    try:
        factory = getattr(module, attr_name)
    except AttributeError as error:
        raise ManifestValidationError(
            message=f"module {module_name!r} has no attribute {attr_name!r}",
            context={"judge": spec},
        ) from error
    judge: JudgeClient = factory()
    return judge


async def _read_labeled_set(path: Path) -> tuple[tuple[LabeledJudgeSample, ...], str]:
    """Read and validate a labeled set, returning its samples and the file's revision.

    Reading goes through :class:`~agentic_evalkit.datasets.local.LocalDatasetProvider`
    rather than opening the file directly, which is what makes this accept
    JSON, JSONL, YAML, and CSV without a second decoder, keeps YAML on
    ``yaml.safe_load``, and confines reads to the working directory. The
    ``revision`` it returns is a SHA-256 of the file's bytes, used below to
    name the calibration after the exact content it was measured from.

    ``LocalDatasetProvider`` signals a path it will not read with a plain
    ``ValueError``, which ``run_cli_command`` would let through as a
    traceback; it is converted here into a typed error the CLI knows the
    exit code for. The four cases it covers -- outside the allowed root, a
    directory, an unsupported suffix, missing -- are deliberately not
    distinguished in the message, so this never reports whether a path
    outside the working directory happens to exist.
    """
    provider = LocalDatasetProvider(allowed_roots=(Path.cwd(),))
    ref = DatasetRef(provider="local", dataset_id=str(path))
    try:
        resolved = await provider.resolve(ref)
    except ValueError as error:
        raise DatasetNotFound(
            message=f"labeled set {str(path)!r} could not be read: {error}",
            context={"path": str(path)},
        ) from error

    samples: list[LabeledJudgeSample] = []
    async for record in provider.iter_records(resolved):
        try:
            samples.append(LabeledJudgeSample.model_validate(record.data))
        except ValidationError as error:
            raise DatasetSchemaMismatch(
                message=(
                    f"row {record.row_id} of {str(path)!r} is not a labeled judge sample: "
                    f"{error.error_count()} validation error(s); expected the fields "
                    "sample_id, prompt, candidate_output, label ('good' or 'bad'), "
                    "and optionally reference"
                ),
                context={"path": str(path), "row_id": record.row_id},
            ) from error
    return tuple(samples), resolved.revision


def _artifact_payload(
    artifact: CalibrationArtifact, level: AuthorityLevel, reason: str
) -> dict[str, Any]:
    """Build the ``--format json`` payload: the artifact plus the verdict on it."""
    return {
        "artifact": artifact.model_dump(mode="json"),
        "authority": {"level": level.value, "reason": reason},
    }


def _print_verdict_table(
    artifact: CalibrationArtifact, level: AuthorityLevel, reason: str, output: Path
) -> None:
    """Render the counts, the rates, and the verdict as a Rich table.

    Every cell is dynamic -- a calibration ID a user chose, a fingerprint, a
    reason string assembled from them -- so every cell goes through
    ``safe_text``: a fingerprint or ID containing bracketed text would
    otherwise be read as Rich markup and silently deleted from exactly the
    value somebody is trying to copy.
    """
    table = Table(title=safe_text(f"Calibration {artifact.calibration_id}"))
    table.add_column("Field", no_wrap=True)
    table.add_column("Value", no_wrap=True)
    table.add_row("judge_fingerprint", safe_text(artifact.judge_fingerprint))
    table.add_row("calibrated_at", safe_text(artifact.calibrated_at))
    table.add_row("expires_at", safe_text(artifact.expires_at))
    table.add_row("total_labeled", safe_text(artifact.total_labeled))
    table.add_row(
        "true_positive / false_negative",
        safe_text(f"{artifact.true_positive} / {artifact.false_negative}"),
    )
    table.add_row(
        "true_negative / false_positive",
        safe_text(f"{artifact.true_negative} / {artifact.false_positive}"),
    )
    table.add_row(
        "abstained / errored",
        safe_text(f"{artifact.abstained_count} / {artifact.error_count}"),
    )
    table.add_row("true_positive_rate", safe_text(_format_rate(artifact.true_positive_rate)))
    table.add_row("true_negative_rate", safe_text(_format_rate(artifact.true_negative_rate)))
    console.print(table)
    console.print(safe_text(f"wrote {output}"), soft_wrap=True)

    style = "bold green" if level is AuthorityLevel.GATING else "bold yellow"
    console.print(
        Text.assemble(Text(level.value.upper(), style=style), safe_text(f": {reason}")),
        soft_wrap=True,
    )


def _format_rate(rate: float | None) -> str:
    """Render a rate, distinguishing "no samples in that class" from a real 0.0."""
    return "n/a (no samples)" if rate is None else f"{rate:.4f}"


@app.command()
def calibrate(
    labeled_set: Annotated[
        Path,
        typer.Argument(help="Labeled judge set: a JSON, JSONL, YAML, or CSV file."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to write the calibration artifact JSON."),
    ],
    judge: Annotated[
        str,
        typer.Option(
            "--judge",
            help="Judge to measure: 'reference', or an import string 'module.path:factory'.",
        ),
    ] = _REFERENCE_JUDGE_NAME,
    calibration_id: Annotated[
        str | None,
        typer.Option("--calibration-id", help="Name for this calibration. Default: derived."),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="The judge's own pass bar for the measured TPR/TNR."),
    ] = PROJECT_MIN_TPR,
    pass_score_threshold: Annotated[
        float,
        typer.Option(
            "--pass-score-threshold",
            help="Score at or above which a judge verdict counts as 'good'.",
        ),
    ] = DEFAULT_PASS_SCORE_THRESHOLD,
    format_: Annotated[str, typer.Option("--format", help="Output format: table or json.")] = (
        "table"
    ),
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Measure a judge against a labeled set and write its calibration artifact.

    Exits 0 when the measured judge has earned the right to gate a release,
    and 3 when it has not -- whether because the evidence is thin (advisory)
    or because it is present and bad (unavailable). The artifact is written
    either way; which of those two happened is in the printed verdict and in
    the JSON payload, where it can carry its reason.
    """

    def _action() -> tuple[tuple[LabeledJudgeSample, ...], str, JudgeClient]:
        judge_client = _load_judge(judge)
        samples, revision = _run_async(lambda: _read_labeled_set(labeled_set))
        return samples, revision, judge_client

    samples, revision, judge_client = run_cli_command(_action, debug=debug)

    def _measure() -> CalibrationArtifact:
        return _run_async(
            lambda: measure_calibration(
                judge_client,
                samples,
                # A derived ID names the calibration after the exact bytes it
                # was measured from, so two artifacts from two versions of a
                # labeled set can never quietly share an identity.
                calibration_id=calibration_id or f"{labeled_set.stem}-{revision[7:19]}",
                threshold=threshold,
                pass_score_threshold=pass_score_threshold,
            )
        )

    artifact = run_cli_command(_measure, debug=debug)

    # The verdict comes from judge_authority, never from re-reading the
    # counts here -- see the module docstring. The live fingerprint is passed
    # so the check that a calibration describes *this* judge is exercised on
    # the same path a real grader takes, even though it cannot fail here.
    authority = judge_authority(artifact, judge_fingerprint=judge_client.fingerprint)
    reason = authority.reason or "calibration clears every floor; this judge may gate a release"

    # Written before the verdict is acted on, so the artifact exists whether
    # or not the judge earned gating authority (ADR-0024).
    def _write() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    run_cli_command(_write, debug=debug)

    if format_ == "json":
        print_output(_artifact_payload(artifact, authority.level, reason), format_=format_)
    else:
        _print_verdict_table(artifact, authority.level, reason, output)

    if authority.level is not AuthorityLevel.GATING:
        raise typer.Exit(code=int(ExitCode.MISSING_CAPABILITY))
