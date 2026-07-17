"""Self-contained HTML reporter (design doc section 11.3, plan Task 13).

Renders the entire report as a single HTML file with the CSS, the run data
(as JSON), and the JavaScript that powers outcome filtering all embedded
directly in the page. The file never reaches out to the internet for
scripts, fonts, analytics, or stylesheets, so it works the same whether
opened online or offline. If JavaScript is turned off in the browser, the
page still shows a readable static summary via a ``<noscript>`` block (the
standard HTML tag for "show this instead if JavaScript is disabled").

The HTML is built with the Jinja2 templating library, and its autoescaping
feature is turned on for the packaged ``report.html.j2`` template. That
means if a sample's output happens to contain text that looks like HTML
(for example, a literal ``<script>`` tag), Jinja2 converts the special
characters so the browser displays them as plain text instead of running
them as code -- otherwise a tested system's output could accidentally (or
deliberately) execute as a script when someone opens the report.
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
    # If any string value in the JSON contains "</" (for example, part of a
    # "</script>" tag that shows up inside a sample's output), a browser
    # reading the raw HTML would see it as the end of this <script> block
    # and cut the page off early. Escaping it to "<\/" prevents that, and is
    # safe: once JavaScript parses this text, "\/" reads back as an ordinary
    # "/", so the actual data is unchanged.
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
