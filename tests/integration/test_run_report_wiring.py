"""End-to-end CLI test proving that ``run`` and ``report`` actually fill in
real statistics and provenance information (proof of exactly what ran),
instead of leaving those fields empty (``null``) or missing entirely
(tracked as items T2-A(b) and (d)).

Before the code this test file covers was written:

- The JSON report that ``run`` writes never had an ``"aggregates"`` section
  at all. The code that computes those aggregate statistics
  (``agentic_evalkit.stats.aggregate_run``/``pass_at_k``) already existed and
  had its own unit tests, but nothing in the CLI actually called it -- even
  though the reporter that writes the file
  (``agentic_evalkit.reporters.base.Reporter.write``) already had a spot (an
  ``aggregates`` parameter) ready and waiting to receive that data.
- ``EvalRunManifest`` has three fields --
  ``environment_fingerprint``/``code_fingerprint``/``target_fingerprint`` --
  meant to record proof of exactly what code, environment, and target were
  used for a run (see ``agentic_evalkit.provenance`` for how each one gets
  computed). Those fields existed in the schema, and the functions to
  compute their values existed too, but no real code path ever called those
  functions. So every real run's saved manifest just had ``None`` in all
  three fields, which contradicted what the project's README promised about
  being able to reproduce a run.

This lives in its own file rather than being added to ``test_cli.py``,
matching this test suite's convention that each file focused on proving
"the wiring between two pieces actually works" is self-contained -- so the
helper that builds a manifest using a local dataset provider is copied here
rather than imported from ``test_cli.py``.
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


# --- (b) aggregate stats now show up in the canonical report and the --------
# --- `report` command --------------------------------------------------------


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
    # "pass@k" measures the chance that at least one of k repeated attempts
    # at the same question succeeds. This run only makes one attempt per
    # question (not repeated attempts), so there's nothing to compute here.
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
    """Checks that ``report`` actually recalculates aggregate stats from the
    run data, instead of just copying forward whatever ``"aggregates"``
    value already happened to be sitting in the file. This is proven here by
    deleting the ``"aggregates"`` key from an already-written report file
    (simulating a file written by an older version of this tool, or edited
    by hand, that wouldn't have this key) and then checking that ``report``
    fills it back in correctly anyway."""
    report_path = _run_cli(tmp_path, monkeypatch)
    envelope = json.loads(report_path.read_text(encoding="utf-8"))
    assert "aggregates" in envelope  # sanity: run really did write one
    del envelope["aggregates"]
    report_path.write_text(json.dumps(envelope), encoding="utf-8")

    result = runner.invoke(app, ["report", str(report_path), "--format", "markdown"])
    assert result.exit_code == 0, result.stdout
    content = report_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "## Aggregates" in content


# --- (d) provenance fingerprints (proof of what code/env/target ran) now ----
# --- get saved onto the run manifest -----------------------------------------


def test_run_persists_environment_and_code_fingerprints_matching_pure_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks the saved fingerprint values by comparing them against a fresh,
    independent computation -- not against some fixed expected string typed
    into the test. This works because these fingerprint functions are "pure"
    (calling one twice with the same inputs always gives the same output,
    with no randomness or hidden state involved) and give the same result
    every time for a given Python interpreter and set of installed packages.
    So calling the same functions again right here, in the test itself, must
    produce the exact same value that the CLI saved earlier."""
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
    """Before the fix tracked as T2-A(d), every real run's saved manifest had
    ``None`` in all three fingerprint fields. This test locks in that fix by
    directly checking those fields are no longer empty."""
    report_path = _run_cli(tmp_path, monkeypatch)
    manifest_payload = json.loads(report_path.read_text(encoding="utf-8"))["manifest"]

    for field_name in ("environment_fingerprint", "code_fingerprint", "target_fingerprint"):
        assert manifest_payload[field_name] is not None
        assert isinstance(manifest_payload[field_name], str)
        assert manifest_payload[field_name] != ""


# --- dataset "contamination" label carries through into the run report -----
# --- (ADR-0013). "Contamination" here means: the model being tested may -----
# --- have already seen this benchmark's questions/answers during its own ---
# --- training, which would make a high score on it untrustworthy. -----------


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
    """Regression test for a gap a Codex code review caught: if the manifest
    marks this dataset as a contamination SUSPECT, that label must actually
    reach the report's ``resolved_dataset`` field -- not just stay on the
    input preset config and never appear next to the actual score."""
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
    """If the manifest has no contamination label at all, the report must
    not invent one either -- this fix only carries forward a label that's
    actually there; it never makes one up."""
    report_path = _run_cli(tmp_path, monkeypatch)
    envelope = json.loads(report_path.read_text(encoding="utf-8"))
    assert envelope["resolved_dataset"]["contamination"] is None


def test_preset_generated_manifest_carries_the_preset_contamination_label() -> None:
    """Checks the first step in the chain: when ``init --preset`` builds a
    manifest from a preset that already has a contamination label, that
    label must make it into the generated manifest, not get dropped along
    the way."""
    document = cli_runs._manifest_document_for_preset(BUILTIN_PRESETS["gsm8k"])
    assert document.manifest.contamination is not None
    assert document.manifest.contamination.status is ContaminationStatus.SUSPECT
