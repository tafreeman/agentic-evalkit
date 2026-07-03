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
        lines.extend(["## Aggregates", "", f"```\n{aggregates}\n```", ""])
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
