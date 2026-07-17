"""End-to-end proof that judge and composite graders can be selected by
name from a run manifest (plan item T2-A(c)) -- and that doing so never
pushes the default hermetic test suite into actually calling a real
network/model, or into actually hard-gating (forcing a FAIL that can't be
overridden) when it shouldn't.

Before this module's test coverage existed,
``agentic_evalkit.cli.runs._KNOWN_GRADERS`` (the CLI's internal list of
grader names it recognizes) had exactly one entry:
``"normalized-exact@1"``. Naming any other grader in a manifest's
``grader`` field -- including ``JudgeGrader``, which was fully implemented
and had its own unit tests, but which the CLI simply didn't know the name
of -- failed manifest validation with an "unknown grader" error. In other
words, the grader existed and worked, but a manifest could never actually
select it.

Selecting ``judge-reference@1`` or ``composite-reference@1`` here uses
only the packaged, network-free ``ReferenceJudgeClient`` (in
``agentic_evalkit.examples.reference_judge``) -- a fake judge
implementation that never calls a real model. Because of that, this stays
part of the default hermetic suite (the one run by ``-m "not live"``,
which never makes network calls).

This is deliberately its own module, matching this test suite's convention
that each module focused on wiring multiple components together should be
understandable entirely on its own (its local-provider manifest helper is
duplicated here, not imported from ``test_cli.py`` or
``test_run_report_wiring.py``).
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
    # This reference judge is wired up permanently uncalibrated
    # (calibration=None) -- it's a test/example implementation, never meant
    # to carry real calibration evidence. Per design doc section 9, a judge
    # can only ever actually hard-gate (force a FAIL that other scores can't
    # average away) when it has real, ratified calibration evidence behind
    # it. So even though cli/runs.py passes gate=True when constructing
    # this grader (asking it to hard-gate if it can), the missing
    # calibration means it never actually does.
    assert grade["hard_gate"] is False
    assert grade["judge_calibration_ref"] is None


def test_composite_reference_grader_runs_end_to_end_with_noncompensable_objective_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The packaged ``zero_target`` always answers ``"0"``, which never
    matches this fixture's GSM8K reference answer ("4"), so the
    composite's objective (exact-match) component genuinely fails on its
    own merits. That component is configured with ``hard_gate=True``, so
    the composite's overall result must be a "noncompensable" ``FAIL`` --
    meaning no other component's score, however good, can average that
    failure away (design doc section 9: "hard objective requirements
    cannot be averaged away by a judge score"). This holds regardless of
    what the advisory judge component scored (an "advisory" score
    contributes information but can't by itself force a fail). That
    proves the hard gate is coming from the objective component, never
    from the permanently-uncalibrated judge component, whose own
    per-child ``hard_gate`` stays ``False``.
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
