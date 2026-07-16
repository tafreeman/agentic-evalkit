"""End-to-end proof of AEK-01 / ADR-0011: ``datasets pull`` then ``run --offline``
against a network-requiring provider, with zero network calls on the offline
run.

Deliberately its own module rather than an addition to ``test_cli.py``,
matching this suite's own stated convention (see that module's "Offline CLI
coverage" section) of each offline-focused test module being self-sufficient
-- the fake provider and the loopback socket guard below are duplicated, not
imported, from ``test_cli.py``/``tests/unit/datasets/test_offline_socket_guard.py``.

Two independent properties are asserted together, not just one:

1. The fake ``huggingface`` provider's ``resolve``/``preview``/``iter_records``
   call counts are unchanged across the offline ``run`` -- proving the
   *framework code path* never reaches the provider a second time, which a
   real (non-fake) provider's own network-safety cannot prove by itself.
2. An OS-level loopback-allowlisting socket guard wraps the offline phase --
   proving no code path (not just the fake provider) opens a real outbound
   connection.
"""

from __future__ import annotations

import json
import socket
from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from agentic_evalkit.cli import app
from agentic_evalkit.cli import datasets as cli_datasets
from agentic_evalkit.cli import runs as cli_runs
from agentic_evalkit.datasets.base import ProviderHealth
from agentic_evalkit.datasets.cache import DatasetCache
from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.resolution_cache import ResolutionCache
from agentic_evalkit.manifest import CallableTargetConfig, ManifestDocument, dump_manifest
from agentic_evalkit.models import (
    DatasetRef,
    DatasetSelection,
    EvalRunManifest,
    ResolvedDataset,
    SamplePage,
    SearchPage,
    SourceRecord,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    import pytest

runner = CliRunner()

_DATASET_ID = "openai/gsm8k"
_REVISION = "canned-gsm8k-revision-for-offline-resolution-cache-test"


class _CallCountedHubProvider:
    """Hermetic, call-counted stand-in for the Hugging Face provider.

    Unlike ``tests/integration/test_cli.py``'s ``_CannedHubProvider``, every
    method here increments its own counter so a test can assert the exact
    number of times the provider was actually reached -- the precise
    property this module exists to prove stays at its pre-offline-run value.
    """

    api_version = "1"

    def __init__(self) -> None:
        self.resolve_calls = 0
        self.preview_calls = 0
        self.iter_records_calls = 0

    async def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchPage:
        return SearchPage(hits=(), cursor=None, total_hits=0)

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        self.resolve_calls += 1
        return ResolvedDataset(
            dataset_id=ref.dataset_id,
            revision=_REVISION,
            config=ref.config,
            split=ref.split,
            row_count=1,
        )

    async def preview(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int = 10
    ) -> SamplePage:
        self.preview_calls += 1
        # `tuple(x async for x in ...)` does NOT work here: passed as a bare
        # generator-expression argument, `async for` makes the whole
        # expression an async generator object, which the synchronous
        # `tuple()` builtin cannot iterate (`TypeError: 'async_generator'
        # object is not iterable`). An async list comprehension, awaited
        # implicitly by this `async def` body, is the correct shape.
        records = [
            record async for record in self.iter_records(dataset, offset=offset, limit=limit)
        ]
        return SamplePage(records=tuple(records), offset=offset, total_rows=1)

    async def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        self.iter_records_calls += 1
        del dataset
        records = (
            SourceRecord(
                row_id="0",
                data={"question": "What is zero plus zero?", "answer": "#### 0"},
                digest="sha256:canned-gsm8k-row-offline-resolution-cache-test",
            ),
        )
        stop = None if limit is None else offset + limit
        for record in records[offset:stop]:
            yield record

    async def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(status="ok")


def _build_catalog(tmp_path: Path, provider: _CallCountedHubProvider) -> DatasetCatalog:
    return DatasetCatalog(
        providers={"huggingface": provider},
        cache=DatasetCache(tmp_path / ".cache"),
        resolution_cache=ResolutionCache(tmp_path / ".cache" / "resolutions"),
        builtin_provider_names=(),
    )


def _manifest_path(tmp_path: Path) -> Path:
    manifest = EvalRunManifest(
        run_name="hf-offline-resolution-cache",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id=_DATASET_ID),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="cli-target",
        selection=DatasetSelection(offset=0, limit=1),
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
    path = tmp_path / "eval.yaml"
    path.write_text(dump_manifest(document), encoding="utf-8")
    return path


# --- loopback-allowlisting socket guard, duplicated per this suite's own -----
# --- established convention (see test_cli.py / test_offline_socket_guard.py) -

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
            f"outbound network connection attempted during an offline call "
            f"after 'datasets pull' (target={address!r})"
        )

    return _guarded


def _forbid_outbound_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``socket.socket.connect``/``connect_ex`` to fail on any non-loopback
    destination. A plain helper (not a fixture) so the caller controls exactly
    when the guard becomes active -- here, only for the offline phase, not the
    online ``pull`` phase that precedes it."""
    monkeypatch.setattr(socket.socket, "connect", _raise_unless_loopback(socket.socket.connect))
    monkeypatch.setattr(
        socket.socket, "connect_ex", _raise_unless_loopback(socket.socket.connect_ex)
    )


# --- the actual end-to-end proof ---------------------------------------------


def test_pull_then_run_offline_over_hf_provider_makes_zero_further_provider_or_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _CallCountedHubProvider()
    catalog = _build_catalog(tmp_path, provider)
    monkeypatch.setattr(cli_datasets, "build_catalog", lambda *, offline: catalog)
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: catalog)

    # --- online phase: `datasets pull` resolves and caches one page -------
    pulled = runner.invoke(
        app,
        ["datasets", "pull", f"hf:{_DATASET_ID}", "--limit", "1", "--format", "json"],
    )
    assert pulled.exit_code == 0, pulled.stdout
    pulled_payload = json.loads(pulled.stdout)
    assert pulled_payload["revision"] == _REVISION
    assert pulled_payload["cached_rows"] == 1
    assert provider.resolve_calls == 1
    assert provider.preview_calls == 1

    calls_after_pull = (
        provider.resolve_calls,
        provider.preview_calls,
        provider.iter_records_calls,
    )

    # --- offline phase: `run --offline` must never reach the provider -----
    manifest_path = _manifest_path(tmp_path)
    output_dir = tmp_path / "results"

    def _invoke_offline_run() -> Any:
        return runner.invoke(
            app,
            [
                "run",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--yes",
                "--offline",
            ],
        )

    # Apply the socket guard only around the offline call: the online pull
    # step above is expected to (and, via the fake, safely does not) use the
    # fake provider's own in-process "network" path.
    _forbid_outbound_network(monkeypatch)
    result = _invoke_offline_run()

    assert result.exit_code == 0, result.stdout
    report_files = list(output_dir.glob("*.json"))
    assert len(report_files) == 1
    envelope = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert envelope["summary"]["total"] == 1
    assert envelope["resolved_dataset"]["revision"] == _REVISION
    assert envelope["samples"][0]["execution"]["status"] == "completed"

    # The precise regression this test exists to catch: the offline run must
    # not have reached the provider's resolve/preview/iter_records again.
    assert (
        provider.resolve_calls,
        provider.preview_calls,
        provider.iter_records_calls,
    ) == calls_after_pull


def test_run_offline_without_a_prior_pull_still_fails_with_typed_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contrast case: skipping the online ``pull`` must still fail loudly
    (never silently succeed), never touching the provider. With a
    ``resolution_cache`` configured (ADR-0011), the error message also
    becomes more actionable -- it names ``datasets pull`` as the fix --
    instead of ADR-0010's original unconditional "resolution is never
    cached" wording. The underlying ``OfflineCacheMiss.retryable`` value
    this message reflects (``True`` here, vs. ``False`` with no
    ``resolution_cache`` configured at all) is asserted directly at the
    catalog level in ``tests/unit/datasets/test_catalog.py``; the CLI's
    error boundary only ever renders ``[code] message``, never the boolean
    itself.
    """
    provider = _CallCountedHubProvider()
    catalog = _build_catalog(tmp_path, provider)
    monkeypatch.setattr(cli_runs, "build_catalog", lambda *, offline: catalog)

    manifest_path = _manifest_path(tmp_path)
    result = runner.invoke(app, ["run", str(manifest_path), "--yes", "--offline"])

    assert result.exit_code == 4, result.stdout
    assert "offline_cache_miss" in result.stdout
    assert "datasets pull" in result.stdout
    assert provider.resolve_calls == 0
