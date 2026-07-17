"""Human-readable Markdown reporter (design doc section 11.3, plan Task 13).

Renders a run as Markdown: an identity section, provenance (which dataset
was used; which "adapter" -- the code that converts a raw dataset row into
this project's standard sample format -- prepared it; which grader judged
the results; and which target -- the system under test -- produced them),
"compatibility fingerprints" (short hash-like identifiers that capture
exactly which code version and which environment produced the run, so you
can tell whether two runs are even safe to compare), outcome counts, and a
table of per-sample evidence. All of it comes from the same fields the
JSON envelope carries -- this is just a friendlier rendering of the same
data -- so a reviewer can read a run's evidence in a text editor or browser
instead of needing JSON tooling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_evalkit.reporters.json import _atomic_write_text, _default_generated_at

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from agentic_evalkit.models import EvalRunResult, SampleResult


def _outcome_label(sample: SampleResult) -> str:
    if sample.grade is not None:
        return str(sample.grade.status.value)
    return str(sample.execution.status.value)


def _sample_row(sample: SampleResult) -> str:
    score = sample.grade.score if sample.grade is not None else None
    score_text = f"{score:.2f}" if score is not None else "-"
    return (
        f"| `{sample.sample.sample_id}` | {sample.execution.status.value} "
        f"| {_outcome_label(sample)} | {score_text} |"
    )


def _sample_table(samples: tuple[SampleResult, ...]) -> str:
    header = "| Sample | Execution | Outcome | Score |\n| --- | --- | --- | --- |"
    rows = "\n".join(_sample_row(sample) for sample in samples)
    return f"{header}\n{rows}"


def _outcome_counts_section(run: EvalRunResult) -> str:
    summary = run.summary
    return (
        f"- Passed: {summary.passed}/{summary.total}\n"
        f"- Failed: {summary.failed}/{summary.total}\n"
        f"- Partial: {summary.partial}/{summary.total}\n"
        f"- Errors: {summary.errors}/{summary.total}\n"
        f"- Timeouts: {summary.timeouts}/{summary.total}\n"
        f"- Cancelled: {summary.cancelled}/{summary.total}\n"
        f"- Abstained: {summary.abstained}/{summary.total}\n"
        f"- Unavailable: {summary.unavailable}/{summary.total}"
    )


def _mapping(aggregates: dict[str, JsonValue], key: str) -> dict[str, JsonValue]:
    """Return ``aggregates[key]`` when it is a nested object, else an empty dict."""
    value = aggregates.get(key)
    return value if isinstance(value, dict) else {}


def _number(value: JsonValue | None, *, digits: int = 4) -> str:
    """Format ``value`` as a fixed-precision decimal, or "-" if it's missing or not a real number.

    (Booleans are deliberately excluded even though Python treats them as a
    kind of ``int`` -- we don't want ``True``/``False`` rendered as
    "1.0000"/"0.0000" in a report.)
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{value:.{digits}f}"


def _integer(value: JsonValue | None) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "-"
    return str(value)


def _label(value: JsonValue | None) -> str:
    return value if isinstance(value, str) else "-"


def _ci_cell(mapping: dict[str, JsonValue]) -> str:
    lower = _number(mapping.get("lower_bound"))
    upper = _number(mapping.get("upper_bound"))
    if lower == "-" or upper == "-":
        return "-"
    return f"[{lower}, {upper}]"


def _pass_rate_row(aggregates: dict[str, JsonValue]) -> str:
    pass_rate = _mapping(aggregates, "pass_rate")
    fraction = f"{_integer(pass_rate.get('numerator'))}/{_integer(pass_rate.get('denominator'))}"
    value = _number(pass_rate.get("value"))
    value_cell = fraction if value == "-" else f"{fraction} = {value}"
    return (
        f"| Pass rate | {value_cell} | {_ci_cell(pass_rate)} "
        f"| {_label(pass_rate.get('interval_method'))} |"
    )


def _score_row(aggregates: dict[str, JsonValue]) -> str | None:
    score_mean = aggregates.get("score_mean")
    if isinstance(score_mean, bool) or not isinstance(score_mean, (int, float)):
        return None
    estimate = _mapping(aggregates, "score_estimate")
    return (
        f"| Score mean | {_number(score_mean)} | {_ci_cell(estimate)} "
        f"| {_label(estimate.get('interval_method'))} |"
    )


def _pass_at_k_row(aggregates: dict[str, JsonValue]) -> str | None:
    payload = _mapping(aggregates, "pass_at_k")
    if not payload:
        return None
    return f"| pass@{_integer(payload.get('k'))} | {_number(payload.get('mean'))} | - | - |"


def _aggregates_table(aggregates: dict[str, JsonValue]) -> list[str]:
    """Render the run's summary statistics as a Markdown table, not a raw Python dict.

    (Just inserting the raw dict into the report as text would look like
    ``{'pass_rate': {...}}`` -- technically correct but unreadable. This
    turns it into a proper table instead.)

    Reads the fixed set of keys that
    :func:`agentic_evalkit.stats.build_report_aggregates` always produces:
    the pass rate, together with a 95% confidence interval around it (a
    range expressing how much uncertainty the sample size leaves -- computed
    using either a "Wilson" interval, a standard technique that stays
    accurate even with small sample counts, or a "cluster-robust" one, which
    widens the range to account for samples that aren't fully independent of
    each other, such as repeated attempts on the same problem); the mean
    score and its own confidence interval (``score_estimate``); and, present
    only when the run attempted each sample more than once, ``pass_at_k``
    (the fraction of samples solved by at least one of ``k`` attempts -- a
    common way to measure how reliably a system succeeds when given
    multiple tries). Each of these becomes one table row. When a metric
    wasn't computed for this run, its row shows ``"-"`` (the same convention
    ``_sample_row`` uses elsewhere in this file) instead of inventing a
    value like ``0``, which could be misread as an actual measured result.
    """
    rows = [_pass_rate_row(aggregates), _score_row(aggregates), _pass_at_k_row(aggregates)]
    body = [row for row in rows if row is not None]
    return ["| Metric | Value | 95% CI | Method |", "| --- | --- | --- | --- |", *body]


def render_markdown(
    run: EvalRunResult,
    *,
    aggregates: dict[str, JsonValue] | None = None,
    generated_at: str,
) -> str:
    manifest = run.manifest
    resolved = run.resolved_dataset
    lines = [
        f"# Evaluation Run `{run.run_id}`",
        "",
        f"- Run name: {manifest.run_name}",
        f"- Generated at: {generated_at}",
        f"- Started at: {run.started_at.isoformat()}",
        f"- Finished at: {run.finished_at.isoformat() if run.finished_at else 'unfinished'}",
        "",
        "## Provenance",
        "",
        f"- Dataset: `{resolved.dataset_id}` @ `{resolved.revision}`",
        f"- Config / split: `{resolved.config}` / `{resolved.split}`",
        f"- Adapter: `{manifest.adapter}`",
        f"- Grader: `{manifest.grader}`",
        f"- Target: `{manifest.target_name}`",
        "",
        "## Compatibility",
        "",
        f"- Environment fingerprint: `{manifest.environment_fingerprint}`",
        f"- Code fingerprint: `{manifest.code_fingerprint}`",
        "",
        "## Outcome counts",
        "",
        _outcome_counts_section(run),
        "",
        "## Samples",
        "",
        _sample_table(run.samples),
        "",
    ]
    if aggregates is not None:
        lines.extend(["## Aggregates", "", *_aggregates_table(aggregates), ""])
    return "\n".join(lines)


class MarkdownReporter:
    """Renders identity, provenance, outcomes, and sample evidence as Markdown."""

    def write(
        self,
        run: EvalRunResult,
        destination: Path,
        *,
        aggregates: dict[str, JsonValue] | None = None,
        generated_at: str | None = None,
    ) -> Path:
        resolved_generated_at = (
            generated_at if generated_at is not None else _default_generated_at()
        )
        content = render_markdown(run, aggregates=aggregates, generated_at=resolved_generated_at)
        _atomic_write_text(destination, content)
        return destination


__all__ = ["MarkdownReporter", "render_markdown"]
