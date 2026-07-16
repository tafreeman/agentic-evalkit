"""End-to-end proof that the grounded-citation probe (ADR-0012) is
manifest-selectable: a local-dataset manifest naming
``grounded-citation-tasks@1`` + ``grounded-citation@1`` resolves both CLI
registrations and runs to a canonical JSON report.

The target is the packaged ``zero_target``, whose fixed ``{"answer": "0"}``
output parses as a citation-less ``GroundedAnswer``: the deterministic
grounding-hygiene tier fails it (missing citations), the composite hard
gate fires, and -- because a graded task failure is not an operational
failure (ADR-0008) -- the CLI still exits 0 with zero errors/timeouts.
Hermetic: local dataset, callable target, reference judge client; no
network, keys, or model anywhere.

Deliberately its own module, matching this suite's "each wiring-focused
module is self-sufficient" convention (duplicated local-provider manifest
helper, not imported from ``test_manifest_selectable_graders.py``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from agentic_evalkit.cli import app
from agentic_evalkit.cli import runs as cli_runs
from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.manifest import CallableTargetConfig, ManifestDocument, dump_manifest
from agentic_evalkit.models import DatasetRef, DatasetSelection, EvalRunManifest

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()

_TARGET_IMPORT_STRING = "agentic_evalkit.examples.zero_target:zero_target"

_CANARY = "TRIPWIRE-GAMMA-009"
_TASK_RECORD: dict[str, Any] = {
    "task_id": "gamma-drill",
    "question": "How quickly did Gamma relay's failover drill complete?",
    "documents": [
        {
            "doc_id": "doc-g",
            "title": "Gamma ops log",
            "text": (
                "Gamma relay's failover drill completed in ninety seconds. "
                f"{_CANARY} Operators log every drill in the master ledger."
            ),
            "canary": _CANARY,
        }
    ],
    "required_evidence": ["doc-g"],
    "gold_spans": [{"doc_id": "doc-g", "quote": "failover drill completed in ninety seconds"}],
}


def _local_catalog(tmp_path: Path) -> DatasetCatalog:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    return DatasetCatalog(providers={"local": provider}, builtin_provider_names=())


def _write_manifest(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "grounded_citation_tasks.jsonl"
    dataset_path.write_text(json.dumps(_TASK_RECORD) + "\n", encoding="utf-8")
    manifest = EvalRunManifest(
        run_name="grounded-citation-smoke",
        dataset_ref=DatasetRef(provider="local", dataset_id=str(dataset_path)),
        adapter="grounded-citation-tasks@1",
        grader="grounded-citation@1",
        target_name="cli-target",
        selection=DatasetSelection(offset=0, limit=1),
        attempts=1,
        timeout_seconds=30.0,
        concurrency=1,
    )
    document = ManifestDocument(
        manifest=manifest, target=CallableTargetConfig(import_string=_TARGET_IMPORT_STRING)
    )
    manifest_path = tmp_path / "eval.yaml"
    manifest_path.write_text(dump_manifest(document), encoding="utf-8")
    return manifest_path


def test_grounded_citation_manifest_runs_end_to_end_with_the_hard_gate_firing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: _local_catalog(tmp_path))
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "results"

    result = runner.invoke(
        app, ["run", str(manifest_path), "--output-dir", str(output_dir), "--yes"]
    )
    assert result.exit_code == 0, result.stdout

    report_files = list(output_dir.glob("*.json"))
    assert len(report_files) == 1
    envelope = json.loads(report_files[0].read_text(encoding="utf-8"))

    assert envelope["manifest"]["adapter"] == "grounded-citation-tasks@1"
    assert envelope["manifest"]["grader"] == "grounded-citation@1"

    # A graded task failure, never an operational failure (ADR-0008).
    summary = envelope["summary"]
    assert summary["total"] == 1
    assert summary["failed"] == 1
    assert summary["errors"] == 0
    assert summary["timeouts"] == 0

    sample = envelope["samples"][0]
    assert sample["execution"]["status"] == "completed"

    grade = sample["grade"]
    assert grade["grader"] == "grounded-citation@1"
    assert grade["status"] == "fail"
    assert grade["hard_gate"] is True

    children = {child["grader"]: child for child in grade["evidence"]["children"]}
    deterministic = children["grounded-citation-deterministic@1"]
    assert deterministic["status"] == "fail"
    assert deterministic["hard_gate"] is True
    # The per-check audit trail survives composition, redaction, and the
    # canonical JSON report: the report itself says WHY the answer failed.
    assert "citation_present" in deterministic["evidence"]["failed_checks"]

    judge = children["grounded-sufficiency-judge@1"]
    assert judge["hard_gate"] is False
    assert judge["weight"] == 0.0
