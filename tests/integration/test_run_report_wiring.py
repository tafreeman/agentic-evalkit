"""End-to-end CLI proof that ``run``/``report`` carry real statistics and
provenance instead of leaving them ``null``/absent (items T2-A(b) and (d)).

Before this module's coverage existed:

- ``run``'s canonical JSON report never carried an ``"aggregates"`` key at
  all -- ``agentic_evalkit.stats.aggregate_run``/``pass_at_k`` existed and
  were unit-tested, but nothing in the CLI ever called them, even though
  every reporter (``agentic_evalkit.reporters.base.Reporter.write``) already
  accepted an ``aggregates`` parameter for exactly this purpose.
- ``EvalRunManifest.environment_fingerprint``/``code_fingerprint``/
  ``target_fingerprint`` were declared, versioned wire fields with their own
  pure computation helpers (``agentic_evalkit.provenance``), but no
  production code path ever called those helpers -- every real run's
  manifest carried ``None`` in all three, contradicting the README's
  reproducibility claims.

Deliberately its own module (not an addition to ``test_cli.py``), matching
this suite's "each offline/wiring-focused module is self-sufficient"
convention -- it duplicates the local-provider manifest-building helper
rather than importing it from ``test_cli.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from agentic_evalkit.cli import app
from agentic_evalkit.cli import runs as cli_runs
from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.datasets.presets import BUILTIN_PRESETS
from agentic_evalkit.manifest import CallableTargetConfig, ManifestDocument, dump_manifest
from agentic_evalkit.models import (
    ContaminationMetadata,
    ContaminationStatus,
    DatasetRef,
    DatasetSelection,
    EvalRunManifest,
    SamplingPolicy,
)
from agentic_evalkit.provenance import (
    compute_code_fingerprint,
    compute_environment_fingerprint,
    compute_target_fingerprint,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()

_TARGET_IMPORT_STRING = "agentic_evalkit.examples.zero_target:zero_target"


def _local_catalog(tmp_path: Path) -> DatasetCatalog:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    return DatasetCatalog(providers={"local": provider}, builtin_provider_names=())


def _write_local_manifest(tmp_path: Path, *, attempts: int = 1) -> Path:
    dataset_path = tmp_path / "gsm8k_local.jsonl"
    dataset_path.write_text(
        '{"question":"2+2?","answer":"work\\n#### 4"}\n'
        '{"question":"3+3?","answer":"work\\n#### 6"}\n'
    )
    manifest = EvalRunManifest(
        run_name="local-report-wiring",
        dataset_ref=DatasetRef(provider="local", dataset_id=str(dataset_path)),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="cli-target",
        selection=DatasetSelection(offset=0, limit=2),
        sampling=SamplingPolicy(attempts=attempts),
        attempts=attempts,
        timeout_seconds=30.0,
        concurrency=1,
    )
    document = ManifestDocument(
        manifest=manifest, target=CallableTargetConfig(import_string=_TARGET_IMPORT_STRING)
    )
    manifest_path = tmp_path / "eval.yaml"
    manifest_path.write_text(dump_manifest(document), encoding="utf-8")
    return manifest_path


def _run_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, attempts: int = 1) -> Path:
    """Invoke ``run`` and return the written canonical JSON report path."""
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: _local_catalog(tmp_path))
    manifest_path = _write_local_manifest(tmp_path, attempts=attempts)
    output_dir = tmp_path / "results"
    result = runner.invoke(
        app, ["run", str(manifest_path), "--output-dir", str(output_dir), "--yes"]
    )
    assert result.exit_code == 0, result.stdout
    report_files = list(output_dir.glob("*.json"))
    assert len(report_files) == 1
    return report_files[0]


# --- (b) aggregates wired into the canonical report and `report` command ----


def test_run_writes_canonical_report_with_aggregates_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = _run_cli(tmp_path, monkeypatch)
    envelope = json.loads(report_path.read_text(encoding="utf-8"))

    assert "aggregates" in envelope
    aggregates = envelope["aggregates"]
    assert aggregates["total"] == envelope["summary"]["total"]
    assert aggregates["passed"] == envelope["summary"]["passed"]
    assert "pass_rate" in aggregates
    assert aggregates["pass_rate"]["denominator"] == envelope["summary"]["total"]
    # A single-attempt run has no repeated-attempt pass@k to report.
    assert "pass_at_k" not in aggregates


def test_run_writes_canonical_report_with_pass_at_k_for_repeated_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = _run_cli(tmp_path, monkeypatch, attempts=2)
    envelope = json.loads(report_path.read_text(encoding="utf-8"))

    assert envelope["summary"]["total"] == 4  # 2 samples x 2 attempts each
    aggregates = envelope["aggregates"]
    assert "pass_at_k" in aggregates
    assert aggregates["pass_at_k"]["k"] == 2
    by_sample_id = aggregates["pass_at_k"]["by_sample_id"]
    assert isinstance(by_sample_id, dict)
    assert len(by_sample_id) == 2  # one estimate per distinct sample_id


def test_report_command_regenerates_markdown_with_aggregates_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = _run_cli(tmp_path, monkeypatch)
    result = runner.invoke(app, ["report", str(report_path), "--format", "markdown"])
    assert result.exit_code == 0, result.stdout

    markdown_path = report_path.with_suffix(".md")
    assert markdown_path.exists()
    content = markdown_path.read_text(encoding="utf-8")
    assert "## Aggregates" in content


def test_report_command_regenerates_from_a_pre_aggregates_run_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``report`` recomputes aggregates from the reconstructed run rather than
    only echoing a pre-existing ``"aggregates"`` key -- proven here by
    stripping that key from an already-written canonical JSON file (as an
    older tool, or a hand-edited file, would lack it) before regenerating."""
    report_path = _run_cli(tmp_path, monkeypatch)
    envelope = json.loads(report_path.read_text(encoding="utf-8"))
    assert "aggregates" in envelope  # sanity: run really did write one
    del envelope["aggregates"]
    report_path.write_text(json.dumps(envelope), encoding="utf-8")

    result = runner.invoke(app, ["report", str(report_path), "--format", "markdown"])
    assert result.exit_code == 0, result.stdout
    content = report_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Aggregates" in content


# --- (d) provenance fingerprints wired into the run manifest -----------------


def test_run_persists_environment_and_code_fingerprints_matching_pure_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact equality against independently, freshly recomputed values from
    the same pure functions (not a hardcoded digest literal) -- these
    fingerprints are deterministic per interpreter/package install, so a
    fresh call in the test process must match what the CLI persisted."""
    report_path = _run_cli(tmp_path, monkeypatch)
    envelope = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_payload = envelope["manifest"]

    assert manifest_payload["environment_fingerprint"] == compute_environment_fingerprint()
    assert manifest_payload["code_fingerprint"] == compute_code_fingerprint()
    # The top-level `provenance` summary echoes the same two fields.
    assert envelope["provenance"]["environment_fingerprint"] == compute_environment_fingerprint()
    assert envelope["provenance"]["code_fingerprint"] == compute_code_fingerprint()


def test_run_persists_target_fingerprint_matching_the_resolved_target_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = _run_cli(tmp_path, monkeypatch)
    envelope = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_payload = envelope["manifest"]

    expected = compute_target_fingerprint(
        CallableTargetConfig(import_string=_TARGET_IMPORT_STRING).model_dump(mode="json")
    )
    assert manifest_payload["target_fingerprint"] == expected


def test_run_fingerprints_are_non_none_and_never_the_pre_wiring_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before T2-A(d), every real run's manifest carried ``None`` in all
    three fingerprint fields; this pins that regression directly."""
    report_path = _run_cli(tmp_path, monkeypatch)
    manifest_payload = json.loads(report_path.read_text(encoding="utf-8"))["manifest"]

    for field_name in ("environment_fingerprint", "code_fingerprint", "target_fingerprint"):
        assert manifest_payload[field_name] is not None
        assert isinstance(manifest_payload[field_name], str)
        assert manifest_payload[field_name] != ""


# --- contamination label propagates into the run report (ADR-0013) ----------


def _write_local_manifest_with_contamination(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "gsm8k_local.jsonl"
    dataset_path.write_text('{"question":"2+2?","answer":"work\\n#### 4"}\n')
    manifest = EvalRunManifest(
        run_name="contamination-propagation",
        dataset_ref=DatasetRef(provider="local", dataset_id=str(dataset_path)),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="cli-target",
        selection=DatasetSelection(offset=0, limit=1),
        attempts=1,
        timeout_seconds=30.0,
        concurrency=1,
        contamination=ContaminationMetadata(status=ContaminationStatus.SUSPECT),
    )
    document = ManifestDocument(
        manifest=manifest, target=CallableTargetConfig(import_string=_TARGET_IMPORT_STRING)
    )
    manifest_path = tmp_path / "eval.yaml"
    manifest_path.write_text(dump_manifest(document), encoding="utf-8")
    return manifest_path


def test_run_stamps_manifest_contamination_onto_the_report_resolved_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Codex-review gap: a SUSPECT label on the manifest must reach the
    report's ``resolved_dataset`` so the score carries the prompt, not stay
    stranded on the preset."""
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: _local_catalog(tmp_path))
    manifest_path = _write_local_manifest_with_contamination(tmp_path)
    output_dir = tmp_path / "results"
    result = runner.invoke(
        app, ["run", str(manifest_path), "--output-dir", str(output_dir), "--yes"]
    )
    assert result.exit_code == 0, result.stdout

    envelope = json.loads(next(iter(output_dir.glob("*.json"))).read_text(encoding="utf-8"))
    assert envelope["resolved_dataset"]["contamination"]["status"] == "suspect"


def test_run_without_manifest_contamination_leaves_resolved_dataset_unlabeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No manifest label -> no fabricated label on the report (the stamp only
    fills a genuine gap; it never invents a status)."""
    report_path = _run_cli(tmp_path, monkeypatch)
    envelope = json.loads(report_path.read_text(encoding="utf-8"))
    assert envelope["resolved_dataset"]["contamination"] is None


def test_preset_generated_manifest_carries_the_preset_contamination_label() -> None:
    """``init --preset`` must not drop the label between the preset and the
    manifest it writes -- the first link in the propagation chain."""
    document = cli_runs._manifest_document_for_preset(BUILTIN_PRESETS["gsm8k"])
    assert document.manifest.contamination is not None
    assert document.manifest.contamination.status is ContaminationStatus.SUSPECT
