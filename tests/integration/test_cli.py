"""Hermetic integration tests for the CLI's core commands (the plain,
non-judge "objective" grading path).

"Hermetic" means these tests never reach out over the network: the default
``not live`` test suite must never actually contact Hugging Face (the
service that hosts many real datasets). Whenever a CLI code path needs a
dataset that would normally come from the Hugging Face Hub, tests in this
module hand it a "canned" stand-in provider instead -- a fake
implementation that returns fixed, pre-scripted data rather than making a
real network call. An autouse fixture (one that runs automatically for
every test in this file, without being named as an argument) fails loudly
if any test accidentally tries to build the real, network-capable CLI
catalog instead of using the canned one. Coverage that does use the real
Hugging Face Hub and its Dataset Viewer API lives separately, under
``tests/live``.
"""

from __future__ import annotations

import asyncio
import json
import socket
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from agentic_evalkit.cli import app
from agentic_evalkit.cli import datasets as cli_datasets
from agentic_evalkit.cli import doctor as cli_doctor
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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reject_real_provider_catalogs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any CLI code path that would build the real, network-connected
    dataset catalog fail immediately, instead of quietly working.
    """

    def _fail(*, offline: bool) -> DatasetCatalog:
        del offline
        raise AssertionError("default CLI tests must inject a hermetic dataset catalog")

    monkeypatch.setattr(cli_datasets, "build_catalog", _fail)
    monkeypatch.setattr(cli_runs, "build_catalog", _fail)


# --- Fixed, repeatable run-result fixtures for compare/report (no network) --

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
    environment_fingerprint: str | None = None,
    code_fingerprint: str | None = None,
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
        environment_fingerprint=environment_fingerprint,
        code_fingerprint=code_fingerprint,
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


# --- CLI coverage for Hugging Face-backed commands, via a fake network-free provider ---


class _CannedHubProvider:
    """A fake, network-free stand-in for the real Hugging Face dataset
    provider (no network calls, no on-disk cache).

    It hands back exactly one GSM8K-shaped row of data, which is enough to
    let the CLI's ``run`` command exercise its full, real code path
    end-to-end -- without this test actually depending on Hugging Face's
    "Dataset Viewer" API (the real service normally used to preview a
    dataset's rows).
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
        records = [
            record async for record in self.iter_records(dataset, offset=offset, limit=limit)
        ]
        return SamplePage(records=tuple(records), offset=offset, total_rows=1)

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
    # This CLI is built with Typer (on top of Click), and both treat "no
    # command given" as a usage error by convention: with no_args_is_help=True
    # set, running the bare CLI with no subcommand prints the help text but
    # exits with code 2 (not 0). That help text still lists every available
    # command, so a user immediately sees what's available even though the
    # exit code signals an error.
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


def test_doctor_swebench_capability_reports_ok_when_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the optional ``swebench`` package is installed, the "doctor"
    diagnostic command reports that capability as ok, with no fix needed
    (ADR-0009 / ADR-0014).
    """
    monkeypatch.setattr(cli_doctor, "find_spec", lambda name: object())
    result = runner.invoke(app, ["doctor", "--offline", "--format", "json"])
    assert result.exit_code in (0, 3)
    payload = json.loads(result.stdout)
    entries = [entry for entry in payload if entry["name"] == "capability_swebench"]
    assert len(entries) == 1
    assert entries[0]["status"] == "ok"
    assert entries[0]["detail"] == "optional capability swebench is installed"
    assert entries[0]["remediation"] is None


def test_doctor_swebench_capability_warns_with_install_remediation_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the optional ``swebench`` package is missing, "doctor" warns and
    names both the fix (a pip "extra": an optional, named bundle of extra
    dependencies) and the Docker daemon (the background service Docker
    containers need running) that this capability actually depends on
    (ADR-0009 / ADR-0014).

    ``pip install 'agentic-evalkit[swebench]'`` is a real, actionable fix
    now that the extra is actually populated with dependencies (ADR-0014) --
    unlike the "reserved but empty" placeholder extras this repo removed on
    2026-07-11. Because it's now real, the warning message is allowed to
    name it.
    """
    monkeypatch.setattr(cli_doctor, "find_spec", lambda name: None)
    result = runner.invoke(app, ["doctor", "--offline", "--format", "json"])
    assert result.exit_code in (0, 3)
    payload = json.loads(result.stdout)
    entries = [entry for entry in payload if entry["name"] == "capability_swebench"]
    assert len(entries) == 1
    assert entries[0]["status"] == "warning"
    remediation = entries[0]["remediation"]
    assert remediation is not None
    assert "agentic-evalkit[swebench]" in remediation
    assert "docker" in remediation.lower()


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


# --- `--offline` flag CLI coverage (ADR-0010; plan Task 2 Step 3) -----------


def _local_only_catalog(tmp_path: Path, *, with_cache: bool = False) -> DatasetCatalog:
    """A real ``DatasetCatalog``, wired up with only the genuine local
    provider registered.

    This deliberately registers no ``huggingface`` provider at all. That
    way, if some code path ever mistakenly tried to route to
    "huggingface", it would fail loudly with a ``KeyError`` rather than
    quietly succeeding -- this catalog can only ever serve the real,
    network-free ``local`` provider.

    About the ``with_cache`` parameter: you might expect that since
    ``local`` never needs the network, none of its operations would need a
    cache either. That's true for ``search``/``resolve``/``iter_records``,
    but not for ``preview``. Per ADR-0010, ``preview`` is deliberately
    *not* exempted the way other operations are: it's always backed by the
    on-disk cache, for every provider, even ones like ``local`` that never
    make a network call. So calling ``preview`` while offline still needs
    a configured ``DatasetCache`` to read from, even though the provider
    underneath is ``local``. This matches how the real CLI's
    ``build_catalog`` always configures a cache in practice.
    """
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    cache = DatasetCache(tmp_path / ".cache") if with_cache else None
    return DatasetCatalog(providers={"local": provider}, cache=cache, builtin_provider_names=())


@pytest.mark.usefixtures("_forbid_outbound_network")
def test_run_offline_over_local_dataset_succeeds_with_zero_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running ``run --offline`` over a dataset served by the local provider
    must go through the real ``DatasetCatalog``/``LocalDatasetProvider``
    code path and complete successfully, without ever attempting an
    outbound network connection (spec item 3a).
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
    """Running ``run --offline`` against a provider that needs the network,
    when nothing is cached yet, must fail loudly with the typed
    ``OfflineCacheMiss`` error. That error must surface through the CLI's
    normal error-handling path (its "error boundary": the place where
    internal exceptions get turned into a specific nonzero exit code plus
    a message) -- a nonzero, mapped exit code with the error's code and
    message in the output. It must never silently proceed as if
    ``--offline`` had simply been ignored (spec item 3b). Silently
    ignoring it would be a "worse than absent" failure: a user who asked
    to run offline would believe they got that guarantee while a real
    network call happened anyway. Closing that exact failure mode is what
    ADR-0010 exists to do.
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
    # Here, catalog.resolve() itself fails inside the runner -- an
    # infrastructure-level problem that aborts the whole run, not a normal
    # graded failure. (Internally, agentic_evalkit.runner.EvalRunner.run
    # emits a RunFailed event and then re-raises the original exception
    # unchanged.) The CLI's error-handling boundary maps that the same way
    # it maps any other OfflineCacheMiss: exit code 4 (PROVIDER_ERROR),
    # never exit 0 as if --offline had simply been ignored.
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
    """A regression test for a bug where this code path was only *half*
    actually hermetic: before this task existed, ``preview --offline``'s
    inner ``resolve()`` call never actually received ``offline=True`` --
    so even though the ``local`` provider never needs the network, this
    exact scenario used to make what was technically a "live" resolve call
    (it just happened to be harmless because ``local`` never reaches the
    network anyway).

    ``preview`` is never exempted by provider (per ADR-0010, it's always
    backed by the on-disk cache, for every provider) -- so this test first
    "warms" the cache with one ordinary, online ``preview`` call (writing
    an entry into the cache so a later offline read has something to
    find). This is the same "resolve once online, then later offline calls
    succeed" pattern described by ADR-0004 and ADR-0010. It then repeats
    the exact same call with ``--offline`` added, and expects it to
    succeed.
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


def test_canned_hub_preview_consumes_async_records_directly() -> None:
    """Regression test for a real bug in ``_CannedHubProvider.preview()``: it
    built its output records using a plain, synchronous ``tuple(...)``
    around an ``async for`` generator expression. That doesn't work in
    Python -- you cannot collect the results of an ``async for`` loop
    synchronously like that -- so it raised ``TypeError: 'async_generator'
    object is not iterable`` the moment ``preview()`` was actually awaited
    (run as a coroutine).

    This bug stayed hidden ("latent") because every *other* canned-hub
    preview test in this file uses ``--offline``, which fails early with
    ``offline_cache_miss`` before ever reaching the buggy line. This test
    exists to actually run the real coroutine (via ``asyncio.run``) and
    prove the async records get consumed correctly and a page comes back.
    """
    provider = _CannedHubProvider()
    dataset = ResolvedDataset(
        dataset_id="openai/gsm8k",
        revision="canned-gsm8k-revision",
        config=None,
        split=None,
        row_count=1,
    )
    page = asyncio.run(provider.preview(dataset))
    assert page.total_rows == 1
    assert len(page.records) == 1
    assert page.records[0].data["answer"] == "#### 0"


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


# --- compare --allow-cross-environment: opt-in cross-environment override (ADR-0015) ---


def test_compare_environment_fingerprint_mismatch_is_gated_by_default(tmp_path: Path) -> None:
    left = _write_run(
        tmp_path,
        "left",
        _run_result(run_id="run-a", environment_fingerprint="sha256:env-aaaa"),
    )
    right = _write_run(
        tmp_path,
        "right",
        _run_result(run_id="run-b", environment_fingerprint="sha256:env-bbbb"),
    )

    result = runner.invoke(app, ["compare", str(left), str(right), "--seed", "1"])
    assert result.exit_code == 2
    assert "incompatible_runs" in result.stdout
    assert "environment fingerprint" in result.stdout


def test_compare_allow_cross_environment_waives_fingerprint_mismatch(tmp_path: Path) -> None:
    left = _write_run(
        tmp_path,
        "left",
        _run_result(run_id="run-a", environment_fingerprint="sha256:env-aaaa"),
    )
    right = _write_run(
        tmp_path,
        "right",
        _run_result(run_id="run-b", environment_fingerprint="sha256:env-bbbb"),
    )

    result = runner.invoke(
        app,
        ["compare", str(left), str(right), "--seed", "1", "--allow-cross-environment"],
    )
    assert result.exit_code == 0, result.stdout
    assert "waived" in result.stdout.lower()
    assert "environment_fingerprint" in result.stdout


def test_compare_allow_cross_environment_json_output_carries_waived_fields(
    tmp_path: Path,
) -> None:
    left = _write_run(
        tmp_path,
        "left",
        _run_result(run_id="run-a", environment_fingerprint="sha256:env-aaaa"),
    )
    right = _write_run(
        tmp_path,
        "right",
        _run_result(run_id="run-b", environment_fingerprint="sha256:env-bbbb"),
    )

    result = runner.invoke(
        app,
        [
            "compare",
            str(left),
            str(right),
            "--seed",
            "1",
            "--allow-cross-environment",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["waived_provenance_fields"] == ["environment_fingerprint"]


def test_compare_allow_cross_environment_still_gates_non_waivable_mismatch(
    tmp_path: Path,
) -> None:
    # This proves the override is scoped narrowly, right at the CLI
    # boundary: --allow-cross-environment only waives the one kind of
    # mismatch it's meant for (the "environment fingerprint" -- a hash
    # summarizing things like OS/dependency versions, which can validly
    # differ between machines without invalidating the comparison). A
    # *different* kind of mismatch -- the dataset revision itself being
    # different -- is never waivable, so the command still exits 2 (its
    # "incompatible runs" error) even with the flag set.
    left = _write_run(
        tmp_path,
        "left",
        _run_result(
            run_id="run-a",
            dataset_revision="rev-abc",
            environment_fingerprint="sha256:env-aaaa",
        ),
    )
    right = _write_run(
        tmp_path,
        "right",
        _run_result(
            run_id="run-b",
            dataset_revision="rev-def",
            environment_fingerprint="sha256:env-bbbb",
        ),
    )

    result = runner.invoke(
        app,
        ["compare", str(left), str(right), "--seed", "1", "--allow-cross-environment"],
    )
    assert result.exit_code == 2
    assert "dataset revision" in result.stdout
    assert "environment fingerprint" not in result.stdout


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
    # A JSONL report (one JSON object per line) always has this fixed shape:
    # one header line, then one line per sample, then one trailer line. This
    # fixture has 2 samples, so 1 (header) + 2 (samples) + 1 (trailer) = 4.
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
    # "Self-contained" means the HTML never references anything over the
    # network -- no remote <script>, <link rel="stylesheet">, or web font
    # URLs -- so checking that there's no http(s):// URL anywhere is a good
    # proxy for that: the whole report has to work with no internet access.
    assert "http://" not in content
    assert "https://" not in content


def _with_leaky_evidence(run: EvalRunResult, value: str) -> EvalRunResult:
    """Return a copy of ``run`` with a credential-shaped string (something
    that looks like a real secret/API key) planted inside one grade's
    evidence, so a test can check whether redaction catches it.
    """
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


# --- Story 2.3 (risk R-002): credential-hook secrets resolve but never persist ---
#
# This is the CLI-facing half of Story 2.3. A "credential hook" lets a
# manifest reference a secret (like an API key) by naming an environment
# variable, instead of writing the secret's actual value into the config
# file. ``build_target_for_document`` is where that hook actually gets
# resolved: internally, ``_load_http_target`` reads
# ``os.environ.get(credential_hook)`` at build time to fetch the real
# value. These tests set the hook's env var to a "sentinel" secret (an
# obviously-fake value used only so a test can check whether it leaks) and
# confirm two things: building the target must not write that resolved
# value back into the document, and the run's canonical report (the
# standard JSON output written to disk) must never contain it either. (The
# other half of Story 2.3 -- making sure the secret never gets persisted
# into a saved *manifest* -- is covered in ``tests/unit/test_manifest.py``;
# redacting the secret out of *recorded HTTP evidence* is covered in
# ``tests/unit/targets/test_http_target.py``.)

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
    """Building the target (through the same code path the CLI uses) reads
    the hook's environment variable -- this is the one real place the
    secret actually gets resolved -- but the source document is left
    carrying only the hook's *name*. The resolved secret value itself never
    lands on the document or the manifest.
    """
    monkeypatch.setenv(_CRED_HOOK_ENV_NAME, _CRED_HOOK_SECRET)
    document = _http_hook_document()

    target = build_target_for_document(document)
    try:
        assert isinstance(target, HttpTarget)
        # Confirm resolution actually happened (not just that the code didn't
        # crash): the hook's env var was read at build time, and the
        # resolved secret reached the target's header provider -- the piece
        # that would generate a real "Authorization: Bearer <token>" header
        # on an actual request. Without this check, the test could pass even
        # if the target silently never resolved the hook at all.
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
        # build_target_for_document creates its own httpx (HTTP client
        # library) client under the hood; this closes it afterward so Python
        # doesn't emit an "unclosed client" ResourceWarning into the test
        # suite's output. Using getattr() (instead of target._client
        # directly) means that if one of the assertions above already
        # failed, this cleanup step can't itself blow up with an unrelated
        # AttributeError and hide the real failure.
        client = getattr(target, "_client", None)
        if client is not None:
            asyncio.run(client.aclose())


# --- Offline CLI coverage (ADR-0010; plan Task 2 Step 3) --------------------
#
# Below is a "loopback-allowlisting socket guard": a test helper that blocks
# every outbound network connection except to loopback addresses
# (127.0.0.1, ::1, "localhost"), so a test can prove it made zero real
# network calls while still letting Python's own local plumbing work. This
# is duplicated here (not imported) from
# ``tests/unit/datasets/test_offline_socket_guard.py``, which has a more
# heavily commented version -- see that module for the full two-round
# Windows debugging story. In short: a naive "block every socket call"
# guard breaks ``pytest-asyncio``'s own event-loop setup on Windows,
# because Windows' ``ProactorEventLoop`` implements ``socketpair()`` (a
# pair of connected sockets asyncio uses internally) by making a real
# loopback connection under the hood -- blocking that call breaks
# pytest-asyncio itself, not just this test. This guard is kept local to
# this module instead of being shared through a new ``conftest.py``
# fixture, matching this test suite's existing convention that each module
# is self-sufficient (no cross-module fixture sharing exists anywhere else
# in ``tests/``).

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
    """Write a local, GSM8K-shaped dataset file plus a manifest that points at it.

    ("GSM8K" is a well-known dataset of grade-school math word problems,
    often used as a benchmark for whether a system produces the right
    numeric answer; "shaped like GSM8K" here just means the file has the
    same question/answer JSON structure.)

    This mirrors what ``agentic-evalkit init --preset gsm8k`` would
    generate, except with ``dataset_ref.provider = "local"`` instead of
    ``"huggingface"``. That swap matters: it means the code path under test
    is a real (non-fake)
    :class:`~agentic_evalkit.datasets.local.LocalDatasetProvider`, reached
    through the CLI's real ``build_catalog`` -- not a canned test double
    standing in for it. Returns the path to the written manifest YAML file.
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
    """A canonical (the run's standard, on-disk JSON) report whose manifest
    *references the hook by name* -- here, via ``run_name``, an ordinary
    free-text field -- records that hook's NAME but never the resolved
    secret VALUE, even though the hook's environment variable is set while
    the report gets written.

    Putting the hook's name into the serialized report is what makes this a
    meaningful test rather than a trivial one: the persisted bytes provably
    do carry a hook-related reference, so if the pipeline ever actually
    resolved the secret and wrote it to disk, it plausibly *could* show up
    right there -- and it must not. What this test does *not* cover: the
    actual environment-variable resolution path inside
    ``_load_http_target`` (that's exercised separately, by
    ``test_building_target_resolves_hook_without_persisting_the_secret``
    above). Here the run result is built directly in Python, so no hook
    resolution actually runs -- this test only checks what gets
    *persisted*.
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
