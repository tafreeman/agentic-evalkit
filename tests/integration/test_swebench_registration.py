"""End-to-end test proving that the SWE-bench adapter-and-grader pair can be
selected by name from a manifest file, and that it fails safely (rather than
crashing, or silently lying about a result) when its real dependencies
aren't installed (ADR-0014).

SWE-bench is a benchmark where the AI system under test is asked to produce
a code patch that fixes a real bug, and the "harness" -- an official,
external tool -- actually applies that patch to the real project and runs
its real test suite to see whether the fix worked (see
``SweBenchDockerHarnessExecutor``). Running the real harness needs Docker
and the optional ``swebench`` package (an "extra": a group of dependencies
you can choose to install or skip), and this test suite -- which is meant to
run fully self-contained, with no external services -- installs neither.

This test writes a manifest that names adapter ``swebench-verified@1`` and
grader ``swebench-harness@1``, confirms the CLI (in ``cli/runs.py``) can
resolve both names from its table of known components, and runs it all the
way through to a finished report. Because the real harness's own upfront
check (a "preflight": checking whether it's actually able to run, before
attempting to) reports that Docker/swebench aren't available here, the
grade comes back as ``UNAVAILABLE``. That is a deliberate, valid outcome
meaning "we couldn't produce a real verdict" -- it is not the same thing as
an operational error (our own code breaking), and it must never be quietly
reported as a fabricated pass instead. The CLI run still exits with code 0
(success), because a capability being unavailable is not the same thing as
the evaluated task having failed (ADR-0008).

This test only uses a local, in-repo dataset file and the packaged
``zero_target`` example -- no Docker, no real ``swebench`` package. It
lives in its own file, following this test suite's convention that each
test module proving one piece of "wiring" between components is
self-contained.
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

# One fake row of data, shaped like a real SWE-bench Verified dataset entry,
# just valid enough to satisfy the schema. FAIL_TO_PASS and PASS_TO_PASS list
# which tests the harness expects to change from failing to passing (and
# which ones should simply keep passing) once a correct fix is applied; here
# they're encoded as JSON strings, the format ``SweBenchVerifiedAdapter``
# expects to parse.
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
    # A missing real/official grading capability is not the same thing as
    # our own code breaking (an operational error): the run still resolves
    # both component names, still completes normally, and still exits 0
    # (success).
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
