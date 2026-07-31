"""The ``run`` command must say out loud when a sample's output was lost.

A spill failure deliberately changes no count: the sample keeps the status
and the grade it genuinely earned, so it appears in the ``outcomes`` line as
an ordinary pass and ``summary.errors`` stays zero, and the process exits 0.
That accounting is correct -- a storage failure is neither a task failure nor
an operational one -- but it means the only thing standing between "this run
quietly lost some of its evidence" and "this run looks perfectly clean" is
the warning line itself.

``tests/unit/cli/test_run_spill_warning.py`` covers the counting helper. This
file covers the part a user actually sees: that the line is wired into the
command, reaches stdout, and does not disturb the exit code or the report.
Without it, deleting the ``if spilled:`` block leaves the entire suite green.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from agentic_evalkit import runner as runner_module
from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.cli import app
from agentic_evalkit.cli import runs as cli_runs
from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.manifest import CallableTargetConfig, ManifestDocument, dump_manifest
from agentic_evalkit.models import (
    DatasetRef,
    DatasetSelection,
    EvalRunManifest,
    SamplingPolicy,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()

_TARGET_IMPORT_STRING = "agentic_evalkit.examples.zero_target:zero_target"

#: Small enough that every output -- including ``zero_target``'s two-key
#: answer -- counts as oversized, so the spill is genuinely entered without
#: needing a target that manufactures kilobytes of padding.
_TINY_SPILL_THRESHOLD_BYTES = 4

#: A store that refuses everything above a single byte, so the spill it was
#: just handed fails. A real ``ArtifactStore``, not a double: the rejection
#: comes from the same ``max_bytes`` check production hits.
_REJECT_EVERYTHING_MAX_BYTES = 1


def _local_catalog(tmp_path: Path) -> DatasetCatalog:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    return DatasetCatalog(providers={"local": provider}, builtin_provider_names=())


def _write_local_manifest(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "gsm8k_local.jsonl"
    dataset_path.write_text(
        '{"question":"2+2?","answer":"work\\n#### 0"}\n'
        '{"question":"3+3?","answer":"work\\n#### 0"}\n'
    )
    manifest = EvalRunManifest(
        run_name="spill-warning-cli",
        dataset_ref=DatasetRef(provider="local", dataset_id=str(dataset_path)),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="cli-target",
        selection=DatasetSelection(offset=0, limit=2),
        sampling=SamplingPolicy(attempts=1),
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


def test_run_warns_on_stdout_when_sample_outputs_could_not_be_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both samples' outputs are refused by the artifact store. The run still
    succeeds -- that is the whole point of the isolation -- and exits 0, so
    the warning line is the only signal the operator gets that the report
    they are about to read is missing evidence. It must name how many samples
    lost their output, and point at where the per-sample detail lives.
    """
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: _local_catalog(tmp_path))
    monkeypatch.setattr(runner_module, "_LARGE_OUTPUT_THRESHOLD_BYTES", _TINY_SPILL_THRESHOLD_BYTES)
    monkeypatch.setattr(
        cli_runs,
        "ArtifactStore",
        lambda root: ArtifactStore(root, max_bytes=_REJECT_EVERYTHING_MAX_BYTES),
    )
    manifest_path = _write_local_manifest(tmp_path)
    output_dir = tmp_path / "results"

    result = runner.invoke(
        app, ["run", str(manifest_path), "--output-dir", str(output_dir), "--yes"]
    )

    assert result.exit_code == 0, result.stdout  # a storage failure is not a run failure
    assert "2 sample(s) lost their output" in result.stdout
    assert "output_spill_error" in result.stdout
    # The whole message must arrive on ONE line. It is 146 characters and the
    # console is pinned to width 120 whenever stdout is not a terminal
    # (``cli.app._MIN_CONSOLE_WIDTH``), so without ``soft_wrap=True`` Rich
    # word-wraps it and strands ``artifacts.output_spill_error`` on a line of
    # its own. The CLI guide tells scripts to check for this warning, so a
    # substring assertion alone would not notice that regression -- the two
    # above both survive the wrap.
    warning_lines = [line for line in result.stdout.splitlines() if "lost their output" in line]
    assert len(warning_lines) == 1
    assert warning_lines[0].endswith("artifacts.output_spill_error")
    # The report is still written -- the defect this whole change fixes was
    # that it was not.
    assert len(list(output_dir.glob("*.json"))) == 1


def test_run_stays_silent_about_spills_when_nothing_was_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The complement, so the assertion above cannot pass by the line always
    printing: an ordinary run whose outputs are small enough to stay inline
    must not mention dropped outputs at all.
    """
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: _local_catalog(tmp_path))
    manifest_path = _write_local_manifest(tmp_path)
    output_dir = tmp_path / "results"

    result = runner.invoke(
        app, ["run", str(manifest_path), "--output-dir", str(output_dir), "--yes"]
    )

    assert result.exit_code == 0, result.stdout
    assert "lost their output" not in result.stdout
