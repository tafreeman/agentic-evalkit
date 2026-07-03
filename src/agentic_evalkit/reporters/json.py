"""Canonical JSON reporter (design §11.3, plan Task 13).

The JSON envelope is the framework's single source of truth for a run: every
other reporter format (JSONL, Markdown, HTML) derives its content from the
same fields this module writes. Output is deterministic — sorted keys, fixed
indentation, and an atomic replace — so two renders of the same frozen run
with the same ``generated_at`` are byte-identical.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from agentic_evalkit.models import EvalRunResult


def _provenance(run: EvalRunResult) -> dict[str, JsonValue]:
    manifest = run.manifest
    resolved = run.resolved_dataset
    return {
        "dataset_id": resolved.dataset_id,
        "dataset_revision": resolved.revision,
        "config": resolved.config,
        "split": resolved.split,
        "adapter": manifest.adapter,
        "grader": manifest.grader,
        "target_name": manifest.target_name,
        "environment_fingerprint": manifest.environment_fingerprint,
        "code_fingerprint": manifest.code_fingerprint,
    }


def _default_generated_at() -> str:
    return datetime.now(UTC).isoformat()


def build_envelope(
    run: EvalRunResult,
    *,
    aggregates: dict[str, JsonValue] | None = None,
    generated_at: str | None = None,
) -> dict[str, JsonValue]:
    """Build the canonical JSON envelope dict for ``run``.

    Shared by :class:`JsonReporter` and any other reporter (JSONL, HTML)
    that needs the same provenance-carrying structure without duplicating
    field selection.
    """
    # model_dump(mode="json") always produces JSON-compatible data at
    # runtime; the cast bridges pydantic's broad `Any` dump type to the
    # precise recursive `JsonValue` alias this envelope is typed with.
    manifest_payload = cast("JsonValue", run.manifest.model_dump(mode="json"))
    resolved_dataset_payload = cast("JsonValue", run.resolved_dataset.model_dump(mode="json"))
    summary_payload = cast("JsonValue", run.summary.model_dump(mode="json"))
    samples_payload = cast("JsonValue", [sample.model_dump(mode="json") for sample in run.samples])
    envelope: dict[str, JsonValue] = {
        "schema_version": run.schema_version,
        "run_id": run.run_id,
        "provenance": _provenance(run),
        "manifest": manifest_payload,
        "resolved_dataset": resolved_dataset_payload,
        "summary": summary_payload,
        "samples": samples_payload,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at is not None else None,
        "generated_at": generated_at if generated_at is not None else _default_generated_at(),
    }
    if aggregates is not None:
        envelope["aggregates"] = aggregates
    return envelope


def _atomic_write_text(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


class JsonReporter:
    """Writes the complete versioned run as one canonical JSON file."""

    def write(
        self,
        run: EvalRunResult,
        destination: Path,
        *,
        aggregates: dict[str, JsonValue] | None = None,
        generated_at: str | None = None,
    ) -> Path:
        envelope = build_envelope(run, aggregates=aggregates, generated_at=generated_at)
        content = _dump_sorted_indented(envelope)
        _atomic_write_text(destination, content)
        return destination


def _dump_sorted_indented(payload: dict[str, JsonValue]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


__all__ = ["JsonReporter", "build_envelope"]
