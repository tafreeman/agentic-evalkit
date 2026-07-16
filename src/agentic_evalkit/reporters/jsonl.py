"""JSONL reporter: one header, one record per sample, one trailer.

The streaming-friendly counterpart to :class:`~agentic_evalkit.reporters.json.JsonReporter`.
Each line is independently parseable, so large runs can be processed without
loading the whole file, while the header and trailer still carry the same
provenance and summary fields as the canonical JSON envelope.
"""

from __future__ import annotations

import json as jsonlib
from typing import TYPE_CHECKING

from agentic_evalkit.reporters.json import _atomic_write_text, _default_generated_at, _provenance

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from agentic_evalkit.models import EvalRunResult


def _header_record(run: EvalRunResult) -> dict[str, JsonValue]:
    return {
        "record_type": "header",
        "schema_version": run.schema_version,
        "run_id": run.run_id,
        "provenance": _provenance(run),
        "manifest": run.manifest.model_dump(mode="json"),
        "resolved_dataset": run.resolved_dataset.model_dump(mode="json"),
        "summary": run.summary.model_dump(mode="json"),
        "started_at": run.started_at.isoformat(),
    }


def _sample_record(sample: JsonValue) -> dict[str, JsonValue]:
    payload = dict(sample) if isinstance(sample, dict) else {}
    payload["record_type"] = "sample"
    return payload


def _trailer_record(
    run: EvalRunResult, *, aggregates: dict[str, JsonValue] | None, generated_at: str
) -> dict[str, JsonValue]:
    trailer: dict[str, JsonValue] = {
        "record_type": "trailer",
        "run_id": run.run_id,
        "summary": run.summary.model_dump(mode="json"),
        "finished_at": run.finished_at.isoformat() if run.finished_at is not None else None,
        "generated_at": generated_at,
    }
    if aggregates is not None:
        trailer["aggregates"] = aggregates
    return trailer


class JsonlReporter:
    """Writes one header record, one record per sample, and one trailer record."""

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
        records: list[dict[str, JsonValue]] = [_header_record(run)]
        records.extend(_sample_record(sample.model_dump(mode="json")) for sample in run.samples)
        records.append(
            _trailer_record(run, aggregates=aggregates, generated_at=resolved_generated_at)
        )
        lines = (
            jsonlib.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for record in records
        )
        content = "\n".join(lines) + "\n"
        _atomic_write_text(destination, content)
        return destination


__all__ = ["JsonlReporter"]
