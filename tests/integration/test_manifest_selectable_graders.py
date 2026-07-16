"""End-to-end proof that judge/composite graders are selectable from a run
manifest (T2-A(c)), and that selecting one never puts the default hermetic
suite in a live-network or hard-gating position.

Before this module's coverage existed, ``agentic_evalkit.cli.runs._KNOWN_GRADERS``
had exactly one entry (``"normalized-exact@1"``): naming any other grader in
a manifest's ``grader`` field -- including ``JudgeGrader``, fully
implemented and unit-tested, but never reachable from the CLI -- failed
manifest validation with "unknown grader". Selecting ``judge-reference@1``/
``composite-reference@1`` here uses only the packaged, network-free
``ReferenceJudgeClient`` (``agentic_evalkit.examples.reference_judge``), so
this stays part of the default hermetic (``-m "not live"``) suite.

Deliberately its own module, matching this suite's "each wiring-focused
module is self-sufficient" convention (duplicated local-provider manifest
helper, not imported from ``test_cli.py``/``test_run_report_wiring.py``).
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


def _local_catalog(tmp_path: Path) -> DatasetCatalog:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    return DatasetCatalog(providers={"local": provider}, builtin_provider_names=())


def _write_manifest(tmp_path: Path, *, grader: str) -> Path:
    dataset_path = tmp_path / "gsm8k_local.jsonl"
    dataset_path.write_text('{"question":"2+2?","answer":"work\\n#### 4"}\n')
    manifest = EvalRunManifest(
        run_name="manifest-selectable-grader",
        dataset_ref=DatasetRef(provider="local", dataset_id=str(dataset_path)),
        adapter="gsm8k@1",
        grader=grader,
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


def _run_and_load_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, grader: str
) -> dict[str, object]:
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: _local_catalog(tmp_path))
    manifest_path = _write_manifest(tmp_path, grader=grader)
    output_dir = tmp_path / "results"
    result = runner.invoke(
        app, ["run", str(manifest_path), "--output-dir", str(output_dir), "--yes"]
    )
    assert result.exit_code == 0, result.stdout
    report_files = list(output_dir.glob("*.json"))
    assert len(report_files) == 1
    return json.loads(report_files[0].read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_judge_reference_grader_runs_end_to_end_and_never_hard_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope = _run_and_load_report(tmp_path, monkeypatch, grader="judge-reference@1")
    assert envelope["manifest"]["grader"] == "judge-reference@1"
    assert envelope["summary"]["total"] == 1
    sample = envelope["samples"][0]
    assert sample["execution"]["status"] == "completed"
    grade = sample["grade"]
    assert grade["grader"] == "judge-reference@1"
    # Wired in permanently uncalibrated (calibration=None): design §9 means
    # this can never hard-gate, regardless of the gate=True passed at
    # construction in cli/runs.py.
    assert grade["hard_gate"] is False
    assert grade["judge_calibration_ref"] is None


def test_composite_reference_grader_runs_end_to_end_with_noncompensable_objective_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The packaged ``zero_target`` always answers ``"0"``, which never
    matches this fixture's GSM8K reference ("4"), so the composite's
    objective (exact-match) component genuinely fails. Because that
    component is configured ``hard_gate=True``, the composite's overall
    result must be a noncompensable ``FAIL`` (design §9: "hard objective
    requirements cannot be averaged away by a judge score") regardless of
    what the advisory judge component scored -- proving the hard gate comes
    from the objective component, never from the permanently-uncalibrated
    judge component (whose own per-child ``hard_gate`` stays ``False``).
    """
    envelope = _run_and_load_report(tmp_path, monkeypatch, grader="composite-reference@1")
    assert envelope["manifest"]["grader"] == "composite-reference@1"
    grade = envelope["samples"][0]["grade"]
    assert grade["grader"] == "composite-reference@1"
    assert grade["status"] == "fail"
    assert grade["hard_gate"] is True

    children = grade["evidence"]["children"]
    child_by_grader = {child["grader"]: child for child in children}
    assert child_by_grader.keys() == {"normalized-exact@1", "judge-reference@1"}
    assert child_by_grader["normalized-exact@1"]["hard_gate"] is True
    assert child_by_grader["normalized-exact@1"]["status"] == "fail"
    assert child_by_grader["judge-reference@1"]["hard_gate"] is False


def test_default_preset_manifest_still_resolves_to_the_objective_grader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backward compatibility: selecting a judge/composite grader is always
    an explicit manifest choice -- the curated preset's own generated
    manifest is unaffected and keeps naming the plain objective grader."""
    envelope = _run_and_load_report(tmp_path, monkeypatch, grader="normalized-exact@1")
    assert envelope["manifest"]["grader"] == "normalized-exact@1"
    assert envelope["samples"][0]["grade"]["grader"] == "normalized-exact@1"
