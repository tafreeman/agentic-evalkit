"""Self-contained HTML reporter (design §11.3, plan Task 13).

Renders one HTML file with embedded CSS, embedded JSON, and embedded
JavaScript for outcome filtering. The file loads no remote scripts, fonts,
analytics, or stylesheets, and degrades to a readable static summary when
JavaScript is disabled (``<noscript>``).

Jinja2 autoescaping is on for the packaged ``report.html.j2`` template, so
any sample output containing HTML-significant characters (for example
``<script>``) is escaped in the rendered page rather than executed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from jinja2 import Environment, PackageLoader, select_autoescape

from agentic_evalkit.reporters.json import (
    _atomic_write_text,
    _default_generated_at,
    build_envelope,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from agentic_evalkit.models import EvalRunResult, SampleResult

_TEMPLATE_NAME = "report.html.j2"


def _outcome(sample: SampleResult) -> str:
    if sample.grade is not None:
        return str(sample.grade.status.value)
    return str(sample.execution.status.value)


def _evidence_json(sample: SampleResult) -> str:
    evidence: dict[str, JsonValue] = sample.grade.evidence if sample.grade is not None else {}
    return json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=False)


def _output_text(sample: SampleResult) -> str:
    output = sample.execution.output
    if output is None:
        return "-"
    return json.dumps(output, sort_keys=True, ensure_ascii=False)


def _sample_row(sample: SampleResult) -> dict[str, object]:
    return {
        "sample_id": sample.sample.sample_id,
        "execution_status": sample.execution.status.value,
        "outcome": _outcome(sample),
        "score": sample.grade.score if sample.grade is not None else None,
        "output_text": _output_text(sample),
        "evidence_json": _evidence_json(sample),
    }


def _embedded_run_json(
    run: EvalRunResult,
    *,
    aggregates: dict[str, JsonValue] | None,
    generated_at: str,
) -> str:
    envelope = build_envelope(run, aggregates=aggregates, generated_at=generated_at)
    # Escape "</" so the embedded JSON can never terminate the surrounding
    # <script> element early, without altering the JSON's semantic content.
    raw = json.dumps(envelope, sort_keys=True, ensure_ascii=False)
    return raw.replace("</", "<\\/")


def _build_environment() -> Environment:
    return Environment(
        loader=PackageLoader("agentic_evalkit.reporters", "templates"),
        autoescape=select_autoescape(enabled_extensions=("j2", "html", "xml")),
    )


class HtmlReporter:
    """Renders one self-contained HTML file with embedded CSS, JSON, and JS."""

    def __init__(self) -> None:
        self._environment = _build_environment()

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
        template = self._environment.get_template(_TEMPLATE_NAME)
        content = template.render(
            run=run,
            generated_at=resolved_generated_at,
            aggregates=aggregates,
            sample_rows=[_sample_row(sample) for sample in run.samples],
            run_json=_embedded_run_json(
                run, aggregates=aggregates, generated_at=resolved_generated_at
            ),
        )
        _atomic_write_text(destination, content)
        return destination


__all__ = ["HtmlReporter"]
