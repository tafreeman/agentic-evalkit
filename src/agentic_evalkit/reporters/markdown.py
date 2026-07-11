"""Human-readable Markdown reporter (design §11.3, plan Task 13).

Renders identity, provenance, compatibility fingerprints, outcome counts,
and a per-sample evidence table from the same fields the JSON envelope
carries, so a reviewer can read a run's evidence without JSON tooling.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import JsonValue

from agentic_evalkit.models import EvalRunResult, SampleResult
from agentic_evalkit.reporters.json import _atomic_write_text, _default_generated_at


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
    """Fixed-precision float, or ``"-"`` for an absent/non-numeric value."""
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
    """Render the aggregates payload as a Markdown table, never a raw dict repr.

    Reads the stable keys :func:`agentic_evalkit.stats.build_report_aggregates`
    produces -- the Wilson-or-cluster-robust pass rate, the score mean and its
    ``score_estimate`` interval, and (only when a repeated-attempt run produced
    one) ``pass_at_k`` -- formatting each as a row and reusing this file's
    ``"-"``-for-absent idiom (see ``_sample_row``) rather than fabricating a
    value for a metric the run did not define.
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
