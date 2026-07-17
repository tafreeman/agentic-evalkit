"""End-to-end proof that the grounded-citation check (ADR-0012) can be
selected by name from a run manifest: a local-dataset manifest naming
``grounded-citation-tasks@1`` (the adapter) and ``grounded-citation@1``
(the grader) is recognized by the CLI and actually runs, producing a
canonical (standard, on-disk) JSON report.

"Grounded citation" means checking that an AI's answer is actually backed
by a real quote from a source document, not just plausible-sounding text.
The target under test here is the packaged ``zero_target``, a fake target
used across this test suite that always answers with the fixed JSON
``{"answer": "0"}``. That fixed answer parses into a ``GroundedAnswer`` (a
structured answer object) that has no citations at all. Because of that:

- the deterministic (rule-based, not model-based) "grounding hygiene"
  check -- one layer of the overall grader -- fails it, since it is
  missing citations;
- that failure trips the composite grader's "hard gate": a rule that
  forces the overall grade to FAIL no matter what any other sub-check
  scored;
- but none of this counts as the CLI itself failing. A wrong or
  ungrounded *answer* is a normal graded outcome, not an "operational
  failure" like a crash or timeout (ADR-0008 draws exactly this line) --
  so the CLI process still exits 0, with zero errors or timeouts
  recorded, even though the sample's grade is FAIL.

Hermetic: everything here is local and fake -- a local dataset file, a
callable target, a reference (non-real) judge client -- so nothing
touches the network, needs an API key, or calls a real model.

This is deliberately its own module rather than sharing code with
``test_manifest_selectable_graders.py``, matching this test suite's
convention that each module focused on wiring multiple components
together should be understandable entirely on its own (its
local-provider manifest helper is duplicated here, not imported).
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

    # A graded task failure (a wrong/ungrounded answer) is not the same as
    # an operational failure (a crash, timeout, etc.) -- ADR-0008 keeps
    # those two kinds of "failure" separate, which is why summary.errors
    # and summary.timeouts are both 0 even though the grade itself is FAIL.
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
    # This proves the detailed, per-check record of *why* the grade failed
    # survives the whole pipeline intact: combining this check into the
    # composite grader's overall result, redacting sensitive data before
    # writing to disk, and serializing to the final JSON report all leave
    # this evidence in place. So the report itself explains why the answer
    # failed, not just that it failed.
    assert "citation_present" in deterministic["evidence"]["failed_checks"]

    judge = children["grounded-sufficiency-judge@1"]
    assert judge["hard_gate"] is False
    assert judge["weight"] == 0.0
