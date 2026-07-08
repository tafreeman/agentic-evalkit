"""Hermetic integration tests for the runnable objective-only CLI.

The default ``not live`` suite must never contact Hugging Face. Tests in this
module inject a canned provider whenever a CLI path needs a Hub-backed dataset;
an autouse guard rejects accidental construction of the real CLI catalog.
Real Hub and Dataset Viewer CLI coverage lives under ``tests/live``.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agentic_evalkit.cli import app
from agentic_evalkit.cli import datasets as cli_datasets
from agentic_evalkit.cli import runs as cli_runs
from agentic_evalkit.cli.runs import build_target_for_document, write_canonical_report
from agentic_evalkit.datasets.base import ProviderHealth
from agentic_evalkit.datasets.cache import DatasetCache
from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.errors import DatasetNotFound
from agentic_evalkit.manifest import (
    CallableTargetConfig,
    HttpTargetConfig,
    ManifestDocument,
    dump_manifest,
)
from agentic_evalkit.models import (
    DatasetRef,
    EvalRunManifest,
    EvalRunResult,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    RunSummary,
    SamplePage,
    SampleResult,
    SamplingPolicy,
    SearchHit,
    SearchPage,
    SourceRecord,
)
from agentic_evalkit.reporters.json import JsonReporter
from agentic_evalkit.targets import HttpTarget

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reject_real_provider_catalogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an accidental network-capable CLI path fail immediately."""

    def _fail(*, offline: bool) -> DatasetCatalog:
        del offline
        raise AssertionError("default CLI tests must inject a hermetic dataset catalog")

    monkeypatch.setattr(cli_datasets, "build_catalog", _fail)
    monkeypatch.setattr(cli_runs, "build_catalog", _fail)


# --- Deterministic run fixtures for compare/report (no network) -------------

_STARTED_AT = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 7, 3, 12, 5, 0, tzinfo=UTC)


def _sample_result(sample_id: str, *, passed: bool) -> SampleResult:
    grade_status = GradeStatus.PASS if passed else GradeStatus.FAIL
    return SampleResult(
        sample=EvalSample(
            sample_id=sample_id,
            input={"question": f"q-{sample_id}"},
            reference="42",
            source_digest=f"sha256:{sample_id}",
            adapter="gsm8k@1",
        ),
        execution=NormalizedExecutionResult(
            sample_id=sample_id,
            attempt=1,
            output={"answer": "42" if passed else "0"},
            status=ExecutionStatus.COMPLETED,
            started_at=_STARTED_AT,
            finished_at=_FINISHED_AT,
        ),
        grade=GradeResult(
            sample_id=sample_id,
            grader="normalized-exact@1",
            status=grade_status,
            score=1.0 if passed else 0.0,
            created_at=_FINISHED_AT,
        ),
    )


def _run_result(
    *,
    run_id: str,
    dataset_revision: str = "rev-abc",
    passed_flags: tuple[bool, ...] = (True, False),
) -> EvalRunResult:
    samples = tuple(
        _sample_result(f"gsm8k:{index}", passed=flag) for index, flag in enumerate(passed_flags)
    )
    manifest = EvalRunManifest(
        run_name=f"{run_id}-fixture",
        dataset_ref=DatasetRef(
            provider="huggingface", dataset_id="openai/gsm8k", config="main", split="test"
        ),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="cli-target",
        sampling=SamplingPolicy(seed=7, temperature=0.0, attempts=1),
        attempts=1,
    )
    resolved = ResolvedDataset(
        dataset_id="openai/gsm8k",
        revision=dataset_revision,
        config="main",
        split="test",
    )
    passed = sum(passed_flags)
    return EvalRunResult(
        run_id=run_id,
        manifest=manifest,
        resolved_dataset=resolved,
        samples=samples,
        summary=RunSummary(total=len(samples), passed=passed, failed=len(samples) - passed),
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _write_run(tmp_path: Path, name: str, run: EvalRunResult) -> Path:
    destination = tmp_path / f"{name}.json"
    JsonReporter().write(run, destination, generated_at="2026-07-03T12:05:00+00:00")
    return destination


def test_curated_and_init_work_without_manual_import(tmp_path) -> None:  # type: ignore[no-untyped-def]
    listed = runner.invoke(app, ["datasets", "curated", "--format", "json"])
    assert listed.exit_code == 0
    assert "swe-bench-verified" in listed.stdout
    destination = tmp_path / "eval.yaml"
    created = runner.invoke(app, ["init", "--preset", "gsm8k", "--output", str(destination)])
    assert created.exit_code == 0
    assert destination.exists()
    validated = runner.invoke(app, ["validate", str(destination)])
    assert validated.exit_code == 0
    assert "valid" in validated.stdout.lower()


# --- Hermetic Hugging Face CLI coverage -------------------------------------


class _CannedHubProvider:
    """Hermetic stand-in for the Hugging Face provider (no network, no cache).

    The provider supplies one GSM8K-shaped row so the CLI run path remains an
    end-to-end integration test without depending on the Dataset Viewer.
    """

    api_version = "1"

    async def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchPage:
        return SearchPage(
            hits=(SearchHit(dataset_id="openai/gsm8k", provider="huggingface"),),
            cursor=None,
            total_hits=1,
        )

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        return ResolvedDataset(
            dataset_id=ref.dataset_id,
            revision="canned-gsm8k-revision",
            config=ref.config,
            split=ref.split,
            row_count=1,
        )

    async def preview(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int = 10
    ) -> SamplePage:
        records = tuple(
            record async for record in self.iter_records(dataset, offset=offset, limit=limit)
        )
        return SamplePage(records=records, offset=offset, total_rows=1)

    async def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        del dataset
        records = (
            SourceRecord(
                row_id="0",
                data={"question": "What is zero plus zero?", "answer": "#### 0"},
                digest="sha256:canned-gsm8k-row",
            ),
        )
        stop = None if limit is None else offset + limit
        for record in records[offset:stop]:
            yield record

    async def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(status="ok")


class _MissingHubProvider(_CannedHubProvider):
    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        raise DatasetNotFound(
            message=f"dataset {ref.dataset_id!r} does not exist",
            context={"dataset_id": ref.dataset_id},
        )


def _canned_hub_catalog(*, missing: bool = False) -> DatasetCatalog:
    provider = _MissingHubProvider() if missing else _CannedHubProvider()
    return DatasetCatalog(providers={"huggingface": provider}, builtin_provider_names=())


def test_provider_failure_exit_code_contract_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _canned_hub_catalog(missing=True)
    monkeypatch.setattr(cli_datasets, "build_catalog", lambda *, offline: catalog)
    result = runner.invoke(app, ["datasets", "inspect", "hf:missing/not-found"])
    assert result.exit_code == 4
    assert "dataset_not_found" in result.stdout


# --- Additional coverage (plan Task 14, Steps 5-7) --------------------------


def test_root_app_shows_help_without_a_subcommand() -> None:
    # Typer/Click's no_args_is_help=True prints help and exits 2 (its
    # standard "no command given" usage-error convention) rather than 0;
    # this still surfaces every command name so a user immediately sees
    # what is available.
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "doctor" in result.stdout
    assert "run" in result.stdout


def test_version_flag_prints_the_installed_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_doctor_runs_offline_and_reports_checks() -> None:
    result = runner.invoke(app, ["doctor", "--offline", "--format", "json"])
    assert result.exit_code in (0, 3)
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert all("status" in entry for entry in payload)


def test_doctor_reserved_placeholder_extras_have_no_install_remediation() -> None:
    """AEK-02: ``parquet``/``swebench`` back empty extras (ADR-0009).

    ``pip install 'agentic-evalkit[parquet|swebench]'`` satisfies nothing
    today, so ``doctor`` must never print that install command for either
    -- only an informational, reserved-placeholder detail.
    """
    result = runner.invoke(app, ["doctor", "--offline", "--format", "json"])
    assert result.exit_code in (0, 3)
    payload = json.loads(result.stdout)
    placeholder_names = {"capability_parquet", "capability_swebench"}
    placeholder_checks = [entry for entry in payload if entry["name"] in placeholder_names]
    assert {entry["name"] for entry in placeholder_checks} == placeholder_names
    for entry in placeholder_checks:
        remediation = entry.get("remediation")
        assert remediation is None or "pip install" not in remediation
        if entry["status"] != "ok":
            assert "reserved placeholder" in entry["detail"]


def test_datasets_curated_table_format_lists_both_presets() -> None:
    result = runner.invoke(app, ["datasets", "curated"])
    assert result.exit_code == 0
    assert "gsm8k" in result.stdout
    assert "swe-bench-verified" in result.stdout


def test_init_without_preset_or_dataset_is_invalid_input() -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 2


def test_init_refuses_to_overwrite_existing_file_without_force(tmp_path: Path) -> None:
    destination = tmp_path / "eval.yaml"
    destination.write_text("existing: true\n")
    result = runner.invoke(app, ["init", "--preset", "gsm8k", "--output", str(destination)])
    assert result.exit_code == 2
    assert destination.read_text() == "existing: true\n"


def test_validate_rejects_malformed_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "bad.yaml"
    destination.write_text("run_name: only-one-field\n")
    result = runner.invoke(app, ["validate", str(destination)])
    assert result.exit_code == 2
    assert "manifest_validation_error" in result.stdout


def test_validate_rejects_python_tagged_yaml(tmp_path: Path) -> None:
    destination = tmp_path / "unsafe.yaml"
    destination.write_text("run_name: !!python/object/apply:os.system ['echo hi']\n")
    result = runner.invoke(app, ["validate", str(destination)])
    assert result.exit_code == 2


def test_run_executes_gsm8k_against_the_zero_target_and_writes_canonical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: _canned_hub_catalog())
    manifest_path = tmp_path / "eval.yaml"
    created = runner.invoke(app, ["init", "--preset", "gsm8k", "--output", str(manifest_path)])
    assert created.exit_code == 0

    output_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "run",
            str(manifest_path),
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.stdout

    report_files = list(output_dir.glob("*.json"))
    assert len(report_files) == 1
    envelope = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert envelope["summary"]["total"] == 1
    assert len(envelope["samples"]) == 1
    sample = envelope["samples"][0]
    assert sample["execution"]["status"] == "completed"
    assert sample["grade"]["status"] in ("pass", "fail")
    assert str(report_files[0]) in result.stdout


def test_run_missing_manifest_file_is_invalid_input(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", str(tmp_path / "does-not-exist.yaml"), "--yes"],
    )
    assert result.exit_code == 2


def test_datasets_search_json_shape_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _canned_hub_catalog()
    monkeypatch.setattr(cli_datasets, "build_catalog", lambda *, offline: catalog)
    result = runner.invoke(
        app,
        ["datasets", "search", "gsm8k", "--provider", "huggingface", "--format", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    assert payload["hits"][0]["dataset_id"] == "openai/gsm8k"
    assert payload["total_hits"] == 1


# --- --offline CLI coverage (ADR-0010, plan Task 2 Step 3) -------------------


def _local_only_catalog(tmp_path: Path, *, with_cache: bool = False) -> DatasetCatalog:
    """A real ``DatasetCatalog`` wired to only the genuine local provider.

    Deliberately registers no ``huggingface`` provider at all, so any code
    path that mistakenly tried to route to it would fail with a ``KeyError``
    rather than silently succeeding -- this catalog can only ever serve the
    real, network-free ``local`` provider.

    ``with_cache``: ``preview`` is deliberately NOT gated by a provider's
    ``requires_network`` declaration (ADR-0010) -- it is always
    cache-backed, for every provider, so an offline ``preview`` still needs
    a configured ``DatasetCache`` even when the underlying provider is
    ``local``. Callers exercising ``search``/``resolve``/``iter_records``
    offline against ``local`` do not need one (the exemption applies
    directly); callers exercising ``preview`` offline do, matching how the
    real CLI's ``build_catalog`` always configures one.
    """
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    cache = DatasetCache(tmp_path / ".cache") if with_cache else None
    return DatasetCatalog(providers={"local": provider}, cache=cache, builtin_provider_names=())


@pytest.mark.usefixtures("_forbid_outbound_network")
def test_run_offline_over_local_dataset_succeeds_with_zero_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run --offline`` over a local-provider dataset must reach the real
    ``DatasetCatalog``/``LocalDatasetProvider`` and complete successfully
    without ever attempting an outbound network connection (spec item 3a).
    """
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: _local_only_catalog(tmp_path))

    manifest_path = _local_gsm8k_style_manifest_path(tmp_path)
    output_dir = tmp_path / "results"
    result = runner.invoke(
        app,
        ["run", str(manifest_path), "--output-dir", str(output_dir), "--yes", "--offline"],
    )
    assert result.exit_code == 0, result.stdout

    report_files = list(output_dir.glob("*.json"))
    assert len(report_files) == 1
    envelope = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert envelope["summary"]["total"] == 1
    assert envelope["samples"][0]["execution"]["status"] == "completed"


def test_run_offline_over_uncached_hf_dataset_fails_with_typed_error_not_silence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run --offline`` against a network-requiring provider with nothing
    cached must fail loudly with the typed ``OfflineCacheMiss`` error
    surfaced through the CLI's normal error boundary (a nonzero, mapped
    exit code and the error's code/message in output) -- never silently
    proceed as if offline had no effect (spec item 3b; this is the exact
    "worse than absent" failure mode ADR-0010 exists to close).
    """
    catalog = _canned_hub_catalog()
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: catalog)

    manifest = EvalRunManifest(
        run_name="hf-offline-failure",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="cli-target",
        attempts=1,
    )
    document = ManifestDocument(
        manifest=manifest,
        target=CallableTargetConfig(
            import_string="agentic_evalkit.examples.zero_target:zero_target"
        ),
    )
    manifest_path = tmp_path / "eval.yaml"
    manifest_path.write_text(dump_manifest(document), encoding="utf-8")

    result = runner.invoke(
        app,
        ["run", str(manifest_path), "--yes", "--offline"],
    )
    # The runner's own catalog.resolve() failure is an infrastructure-level
    # abort of the run (agentic_evalkit.runner.EvalRunner.run re-raises the
    # original exception unchanged after emitting RunFailed), which the
    # CLI's error boundary maps like any other OfflineCacheMiss: exit code 4
    # (PROVIDER_ERROR), never exit 0 as if --offline had simply been
    # ignored.
    assert result.exit_code == 4, result.stdout
    assert "offline_cache_miss" in result.stdout


def test_datasets_search_offline_over_local_provider_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli_datasets, "build_catalog", lambda *, offline: _local_only_catalog(tmp_path)
    )
    result = runner.invoke(
        app,
        ["datasets", "search", "anything", "--provider", "local", "--offline", "--format", "json"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["total_hits"] == 0


def test_datasets_search_offline_over_network_provider_fails_with_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _canned_hub_catalog()
    monkeypatch.setattr(cli_datasets, "build_catalog", lambda *, offline: catalog)
    result = runner.invoke(
        app,
        ["datasets", "search", "gsm8k", "--provider", "huggingface", "--offline"],
    )
    assert result.exit_code == 4
    assert "offline_cache_miss" in result.stdout


def test_datasets_inspect_offline_over_local_provider_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "local.jsonl"
    dataset_path.write_text('{"question":"1+1?","answer":"#### 2"}\n')
    monkeypatch.setattr(
        cli_datasets, "build_catalog", lambda *, offline: _local_only_catalog(tmp_path)
    )
    result = runner.invoke(
        app,
        ["datasets", "inspect", f"local:{dataset_path}", "--offline", "--format", "json"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["revision"].startswith("sha256:")


def test_datasets_inspect_offline_over_network_provider_fails_with_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _canned_hub_catalog()
    monkeypatch.setattr(cli_datasets, "build_catalog", lambda *, offline: catalog)
    result = runner.invoke(
        app,
        ["datasets", "inspect", "hf:openai/gsm8k", "--offline"],
    )
    assert result.exit_code == 4
    assert "offline_cache_miss" in result.stdout


def test_datasets_preview_offline_over_local_provider_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers the previously-half-hermetic bug directly: before this task,
    ``preview --offline``'s inner ``resolve()`` call never received
    ``offline=True`` at all, so this exact scenario would have made a live
    (albeit locally-served) resolve even though ``local`` never needs one.

    ``preview`` is not exempted by provider (ADR-0010 -- it is always
    cache-backed for every provider), so this test first warms the cache
    with one online preview (the same "resolve once online, then offline
    calls succeed" story ADR-0004/ADR-0010 describe), then repeats the
    exact call with ``--offline`` and expects success.
    """
    dataset_path = tmp_path / "local.jsonl"
    dataset_path.write_text(
        '{"question":"1+1?","answer":"#### 2"}\n{"question":"2+2?","answer":"#### 4"}\n'
    )
    monkeypatch.setattr(
        cli_datasets,
        "build_catalog",
        lambda *, offline: _local_only_catalog(tmp_path, with_cache=True),
    )

    warmup = runner.invoke(
        app, ["datasets", "preview", f"local:{dataset_path}", "--format", "json"]
    )
    assert warmup.exit_code == 0, warmup.stdout

    result = runner.invoke(
        app,
        ["datasets", "preview", f"local:{dataset_path}", "--offline", "--format", "json"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["page"]["total_rows"] == 2


def test_datasets_preview_offline_over_network_provider_fails_with_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _canned_hub_catalog()
    monkeypatch.setattr(cli_datasets, "build_catalog", lambda *, offline: catalog)
    result = runner.invoke(
        app,
        ["datasets", "preview", "hf:openai/gsm8k", "--offline"],
    )
    assert result.exit_code == 4
    assert "offline_cache_miss" in result.stdout


# --- compare (plan Task 14 Step 10) -----------------------------------------


def test_compare_two_compatible_runs_reports_estimate_and_seed(tmp_path: Path) -> None:
    left = _write_run(tmp_path, "left", _run_result(run_id="run-a", passed_flags=(True, False)))
    right = _write_run(tmp_path, "right", _run_result(run_id="run-b", passed_flags=(True, True)))

    result = runner.invoke(
        app,
        ["compare", str(left), str(right), "--bootstrap-samples", "200", "--seed", "13"],
    )
    assert result.exit_code == 0, result.stdout
    assert "estimate" in result.stdout.lower()
    assert "seed" in result.stdout.lower()
    assert "13" in result.stdout


def test_compare_json_format_carries_percentiles_and_paired_count(tmp_path: Path) -> None:
    left = _write_run(tmp_path, "left", _run_result(run_id="run-a"))
    right = _write_run(tmp_path, "right", _run_result(run_id="run-b"))

    result = runner.invoke(
        app,
        [
            "compare",
            str(left),
            str(right),
            "--bootstrap-samples",
            "200",
            "--seed",
            "5",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["seed"] == 5
    assert payload["paired_count"] == 2
    assert "estimate" in payload
    assert "lower_percentile" in payload
    assert "upper_percentile" in payload


def test_compare_incompatible_runs_lists_every_mismatch_and_exits_two(tmp_path: Path) -> None:
    left = _write_run(tmp_path, "left", _run_result(run_id="run-a", dataset_revision="rev-abc"))
    right = _write_run(tmp_path, "right", _run_result(run_id="run-b", dataset_revision="rev-def"))

    result = runner.invoke(app, ["compare", str(left), str(right), "--seed", "1"])
    assert result.exit_code == 2
    assert "incompatible_runs" in result.stdout
    assert "dataset revision" in result.stdout
    assert "rev-abc" in result.stdout
    assert "rev-def" in result.stdout


def test_compare_rejects_bootstrap_samples_out_of_range(tmp_path: Path) -> None:
    left = _write_run(tmp_path, "left", _run_result(run_id="run-a"))
    right = _write_run(tmp_path, "right", _run_result(run_id="run-b"))

    too_low = runner.invoke(
        app, ["compare", str(left), str(right), "--bootstrap-samples", "50", "--seed", "1"]
    )
    assert too_low.exit_code == 2

    too_high = runner.invoke(
        app, ["compare", str(left), str(right), "--bootstrap-samples", "20000", "--seed", "1"]
    )
    assert too_high.exit_code == 2


def test_compare_missing_run_file_is_invalid_input(tmp_path: Path) -> None:
    right = _write_run(tmp_path, "right", _run_result(run_id="run-b"))
    result = runner.invoke(app, ["compare", str(tmp_path / "nope.json"), str(right), "--seed", "1"])
    assert result.exit_code == 2


# --- report (plan Task 14 Step 10) ------------------------------------------


def test_report_regenerates_jsonl(tmp_path: Path) -> None:
    source = _write_run(tmp_path, "run", _run_result(run_id="run-a"))
    destination = tmp_path / "out.jsonl"
    result = runner.invoke(
        app, ["report", str(source), "--format", "jsonl", "--output", str(destination)]
    )
    assert result.exit_code == 0, result.stdout
    assert destination.exists()
    lines = destination.read_text(encoding="utf-8").strip().splitlines()
    # header + one record per sample + trailer
    assert len(lines) == 4
    header = json.loads(lines[0])
    assert header["record_type"] == "header"
    assert header["run_id"] == "run-a"


def test_report_regenerates_markdown(tmp_path: Path) -> None:
    source = _write_run(tmp_path, "run", _run_result(run_id="run-a"))
    destination = tmp_path / "out.md"
    result = runner.invoke(
        app, ["report", str(source), "--format", "markdown", "--output", str(destination)]
    )
    assert result.exit_code == 0, result.stdout
    content = destination.read_text(encoding="utf-8")
    assert "# Evaluation Run" in content
    assert "run-a" in content


def test_report_regenerates_self_contained_html(tmp_path: Path) -> None:
    source = _write_run(tmp_path, "run", _run_result(run_id="run-a"))
    destination = tmp_path / "out.html"
    result = runner.invoke(
        app, ["report", str(source), "--format", "html", "--output", str(destination)]
    )
    assert result.exit_code == 0, result.stdout
    content = destination.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    # self-contained: no remote script/style/font references
    assert "http://" not in content
    assert "https://" not in content


def _with_leaky_evidence(run: EvalRunResult, value: str) -> EvalRunResult:
    """Copy ``run`` with a credential-shaped string planted in one grade's evidence."""
    first = run.samples[0]
    assert first.grade is not None
    grade = first.grade.model_copy(update={"evidence": {"note": value}})
    sample = first.model_copy(update={"grade": grade})
    return run.model_copy(update={"samples": (sample, *run.samples[1:])})


def test_report_applies_default_redaction_to_regenerated_output(tmp_path: Path) -> None:
    leaky = _with_leaky_evidence(
        _run_result(run_id="run-a"),
        "captured header Authorization: Bearer hf_AbCdEf0123456789XYZq",
    )
    source = _write_run(tmp_path, "run", leaky)
    assert "hf_AbCdEf0123456789XYZq" in source.read_text(encoding="utf-8")

    destination = tmp_path / "out.jsonl"
    result = runner.invoke(
        app, ["report", str(source), "--format", "jsonl", "--output", str(destination)]
    )
    assert result.exit_code == 0, result.stdout
    regenerated = destination.read_text(encoding="utf-8")
    assert "hf_AbCdEf0123456789XYZq" not in regenerated
    assert "[REDACTED]" in regenerated


def test_run_canonical_json_is_redacted_before_reaching_disk(tmp_path: Path) -> None:
    leaky = _with_leaky_evidence(_run_result(run_id="run-a"), "token=sk-live_abcDEF0123456789")
    written = write_canonical_report(leaky, tmp_path)
    assert written == tmp_path / "run-a.json"
    text = written.read_text(encoding="utf-8")
    assert "sk-live_abcDEF0123456789" not in text
    assert "[REDACTED]" in text


def test_report_missing_run_file_is_invalid_input(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "report",
            str(tmp_path / "nope.json"),
            "--format",
            "jsonl",
            "--output",
            str(tmp_path / "x.jsonl"),
        ],
    )
    assert result.exit_code == 2


# --- Story 2.3 (R-002): credential-hook runtime resolution never recorded ---
#
# The CLI-path half of Story 2.3: ``build_target_for_document`` is where the
# credential hook actually resolves (``_load_http_target`` reads
# ``os.environ.get(credential_hook)`` at build time). With the hook's env var
# set to a sentinel secret, building the target must not write the resolved
# value back into the document, and a canonical report of the run must never
# contain it. (The manifest-persistence half is covered in
# ``tests/unit/test_manifest.py``; recorded-evidence header redaction is
# covered in ``tests/unit/targets/test_http_target.py``.)

_CRED_HOOK_SECRET = "hook-secret-XYZZY-do-not-persist"
_CRED_HOOK_ENV_NAME = "AGENTIC_EVALKIT_TEST_CLI_CRED_HOOK"


def _http_hook_document() -> ManifestDocument:
    manifest = EvalRunManifest(
        run_name="http-hook-run",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="cli-target",
    )
    return ManifestDocument(
        manifest=manifest,
        target=HttpTargetConfig(
            url="https://example.test/eval", credential_hook=_CRED_HOOK_ENV_NAME
        ),
    )


def test_building_target_resolves_hook_without_persisting_the_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building the target through the CLI dispatch reads the hook's env var
    (the real resolution point) but leaves the source document carrying only
    the hook name -- the resolved secret never lands on the document/manifest.
    """
    monkeypatch.setenv(_CRED_HOOK_ENV_NAME, _CRED_HOOK_SECRET)
    document = _http_hook_document()

    target = build_target_for_document(document)
    try:
        assert isinstance(target, HttpTarget)
        # Resolution actually happened: the hook's env var was read at build
        # time and the resolved secret reached the target's header provider
        # (an Authorization: Bearer <token> header), which is the mechanism a
        # real request would carry. Without this the test could pass on a
        # target that silently never resolved the hook at all.
        header_provider = target._headers  # type: ignore[attr-defined]
        assert header_provider is not None
        resolved_headers = header_provider()
        assert _CRED_HOOK_SECRET in resolved_headers["Authorization"]
        # ...yet the document that produced the target is unchanged: it
        # references only the hook name and never the resolved secret value.
        assert isinstance(document.target, HttpTargetConfig)
        assert document.target.credential_hook == _CRED_HOOK_ENV_NAME
        serialized = document.model_dump_json()
        assert _CRED_HOOK_ENV_NAME in serialized
        assert _CRED_HOOK_SECRET not in serialized
    finally:
        # build_target_for_document constructs its own httpx client; close it
        # so no unclosed-client ResourceWarning leaks into the suite. Guard the
        # attribute access with getattr so a failed isinstance/assertion above
        # is never masked by an AttributeError here.
        client = getattr(target, "_client", None)
        if client is not None:
            asyncio.run(client.aclose())


# --- Offline CLI coverage (ADR-0010, plan Task 2 Step 3) --------------------
#
# Loopback-allowlisting socket guard, duplicated (not imported) from
# ``tests/unit/datasets/test_offline_socket_guard.py``'s more heavily
# commented version -- see that module for the full two-round Windows
# debugging rationale (a naive "block every socket call" guard breaks
# ``pytest-asyncio``'s own event-loop setup on this platform via
# ``ProactorEventLoop``'s ``socketpair()`` emulation). Kept local to this
# module rather than shared via a new ``conftest.py`` fixture, matching this
# test suite's existing convention of each module being self-sufficient (no
# cross-module fixture sharing exists elsewhere in ``tests/``).

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback_address(address: object) -> bool:
    if isinstance(address, tuple) and len(address) >= 2 and isinstance(address[0], str):
        return address[0] in _LOOPBACK_HOSTS
    return False


def _raise_unless_loopback(real: Any) -> Any:
    def _guarded(self: socket.socket, address: object, *args: Any, **kwargs: Any) -> Any:
        if _is_loopback_address(address):
            return real(self, address, *args, **kwargs)
        raise AssertionError(
            f"outbound network connection attempted during a --offline CLI "
            f"invocation (target={address!r})"
        )

    return _guarded


@pytest.fixture
def _forbid_outbound_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _raise_unless_loopback(socket.socket.connect))
    monkeypatch.setattr(
        socket.socket, "connect_ex", _raise_unless_loopback(socket.socket.connect_ex)
    )


def _local_gsm8k_style_manifest_path(tmp_path: Path) -> Path:
    """Write a local GSM8K-shaped dataset file plus a manifest pointing at it.

    Mirrors ``agentic-evalkit init --preset gsm8k``'s manifest shape but
    with ``dataset_ref.provider = "local"`` instead of ``"huggingface"``, so
    a real (non-fake) :class:`~agentic_evalkit.datasets.local.LocalDatasetProvider`
    -- exercised through the CLI's real ``build_catalog`` -- is the code path
    under test, not a canned double. Returns the manifest YAML path.
    """
    dataset_path = tmp_path / "gsm8k_local.jsonl"
    dataset_path.write_text('{"question":"2+2?","answer":"work\\n#### 4"}\n')
    manifest = EvalRunManifest(
        run_name="local-offline-quickstart",
        dataset_ref=DatasetRef(provider="local", dataset_id=str(dataset_path)),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="cli-target",
        attempts=1,
        timeout_seconds=30.0,
        concurrency=1,
    )
    document = ManifestDocument(
        manifest=manifest,
        target=CallableTargetConfig(
            import_string="agentic_evalkit.examples.zero_target:zero_target"
        ),
    )
    manifest_path = tmp_path / "eval.yaml"
    manifest_path.write_text(dump_manifest(document), encoding="utf-8")
    return manifest_path


def test_canonical_report_of_a_hook_run_contains_no_resolved_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A canonical report whose manifest *references the hook by name* (here in
    ``run_name``, a serialized free-text field) records that hook NAME but never
    the resolved secret VALUE, even though the hook's env var is set while the
    report is written.

    Anchoring the hook name into the serialized report makes the discrimination
    real rather than vacuous: the persisted bytes provably carry a hook
    reference, so if the pipeline ever resolved-and-persisted the secret it
    could plausibly appear -- and it must not. What this does *not* cover: the
    actual ``_load_http_target`` env-var resolution path (that is exercised by
    ``test_building_target_resolves_hook_without_persisting_the_secret`` above);
    here the run result is constructed directly, so no resolution runs.
    """
    monkeypatch.setenv(_CRED_HOOK_ENV_NAME, _CRED_HOOK_SECRET)
    base = _run_result(run_id="hook-run")
    hook_referencing_manifest = base.manifest.model_copy(
        update={"run_name": f"hook-run-using-{_CRED_HOOK_ENV_NAME}"}
    )
    run = base.model_copy(update={"manifest": hook_referencing_manifest})

    written = write_canonical_report(run, tmp_path)
    text = written.read_text(encoding="utf-8")
    # The report carries the hook NAME (proving it can carry hook-related
    # text)...
    assert _CRED_HOOK_ENV_NAME in text
    # ...but never the resolved secret VALUE.
    assert _CRED_HOOK_SECRET not in text
