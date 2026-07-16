"""End-to-end proof that the SWE-bench harness pair is manifest-selectable and
degrades gracefully (ADR-0014).

A manifest naming adapter ``swebench-verified@1`` and grader
``swebench-harness@1`` resolves via ``cli/runs.py``'s known tables and runs
to a canonical report. Because the hermetic suite never installs the
``swebench`` extra, the registered ``SweBenchDockerHarnessExecutor``'s real
preflight reports the capability unavailable, so the grade is
``UNAVAILABLE`` -- a graded non-verdict, never an operational error or a
fabricated pass. The run still exits 0 (an unavailable capability is not a
task failure, ADR-0008).

Hermetic: local dataset + packaged ``zero_target``; no Docker, no swebench.
Its own module per this suite's self-sufficient-wiring-module convention.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

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

# A minimal but schema-valid SWE-bench Verified row (FAIL_TO_PASS/PASS_TO_PASS
# as JSON-encoded strings, which SweBenchVerifiedAdapter accepts).
_SWEBENCH_ROW = {
    "instance_id": "org__repo-1",
    "repo": "org/repo",
    "base_commit": "0" * 40,
    "problem_statement": "The widget crashes on empty input.",
    "test_patch": "diff --git a/t b/t\n",
    "FAIL_TO_PASS": '["tests/test_widget.py::test_empty"]',
    "PASS_TO_PASS": '["tests/test_widget.py::test_basic"]',
}


def _local_catalog(tmp_path: Path) -> DatasetCatalog:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    return DatasetCatalog(providers={"local": provider}, builtin_provider_names=())


def _write_manifest(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "swebench_local.jsonl"
    dataset_path.write_text(json.dumps(_SWEBENCH_ROW) + "\n", encoding="utf-8")
    manifest = EvalRunManifest(
        run_name="swebench-registration",
        dataset_ref=DatasetRef(provider="local", dataset_id=str(dataset_path)),
        adapter="swebench-verified@1",
        grader="swebench-harness@1",
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


def test_swebench_pair_is_registered_in_the_cli_tables() -> None:
    assert "swebench-verified@1" in cli_runs._KNOWN_ADAPTERS
    assert "swebench-harness@1" in cli_runs._KNOWN_GRADERS


def test_swebench_manifest_resolves_and_degrades_to_unavailable_without_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: _local_catalog(tmp_path))
    manifest_path = _write_manifest(tmp_path)
    output_dir = tmp_path / "results"

    result = runner.invoke(
        app, ["run", str(manifest_path), "--output-dir", str(output_dir), "--yes"]
    )
    # An unavailable authoritative capability is not an operational error:
    # the run resolves both names, completes, and exits 0.
    assert result.exit_code == 0, result.stdout

    envelope = json.loads(next(iter(output_dir.glob("*.json"))).read_text(encoding="utf-8"))
    assert envelope["manifest"]["adapter"] == "swebench-verified@1"
    assert envelope["manifest"]["grader"] == "swebench-harness@1"
    assert envelope["summary"]["unavailable"] == 1
    assert envelope["summary"]["errors"] == 0

    grade = envelope["samples"][0]["grade"]
    assert grade["grader"] == "swebench-harness@1"
    assert grade["status"] == "unavailable"
    assert grade["hard_gate"] is False
