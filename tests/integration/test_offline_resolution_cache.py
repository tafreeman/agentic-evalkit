"""End-to-end test proving AEK-01 / ADR-0011: once you've run ``datasets pull``
to cache a dataset locally, running ``run --offline`` against that same
dataset must make zero network calls, even though the underlying provider
(Hugging Face here) normally needs the network to work at all.

This test lives in its own file instead of being added to ``test_cli.py``,
following the same pattern this test suite already uses for offline tests
(see the "Offline CLI coverage" section in that file): each offline-focused
test file is self-contained, so the fake provider and the "block outbound
network calls" guard below are copied here rather than imported from
``test_cli.py`` / ``tests/unit/datasets/test_offline_socket_guard.py``.

This test checks two separate things, because either one alone wouldn't be
enough proof:

1. The fake ``huggingface`` provider counts how many times its ``resolve``,
   ``preview``, and ``iter_records`` methods get called. Those counts must
   not change during the offline run -- this proves our own code never calls
   the provider again once it's supposed to be running offline. (A real
   provider simply refusing network access wouldn't prove this -- it would
   only prove the real provider behaves well, not that our code never tried
   to reach it.)
2. A guard replaces Python's low-level socket-connect function so that any
   attempt to connect to somewhere other than localhost raises an error.
   This wraps the entire offline phase, proving that *no* code anywhere --
   not just the fake provider -- opens a real network connection.
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
    """A fake stand-in for the real Hugging Face provider, safe to use here
    because it never touches the real network.

    Unlike ``tests/integration/test_cli.py``'s ``_CannedHubProvider``, every
    method here increments its own counter, so a test can check the exact
    number of times the provider was actually called -- which is exactly the
    thing this file's test needs to prove stays unchanged after the offline
    run.
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
        # Why not just write `tuple(record async for record in ...)`? Because
        # putting `async for` inside a generator expression turns the whole
        # expression into an "async generator" object, and the plain,
        # synchronous `tuple()` function cannot loop over one of those --
        # Python raises `TypeError: 'async_generator' object is not
        # iterable`. Writing it as a list comprehension instead works, because
        # this function is itself `async def`, so Python knows to await each
        # step of an async list comprehension automatically.
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


# --- guard that blocks any network connection except to localhost ("loopback")
# --- copied here (not imported) to match this suite's convention that each --
# --- offline-focused test file is self-contained -- see test_cli.py / -------
# --- test_offline_socket_guard.py --------------------------------------------

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
    """Make any attempt to open a network connection to somewhere other than
    localhost raise an error, by replacing ``socket.socket.connect`` and
    ``connect_ex`` with versions that check the destination first.

    This is a plain function, not a pytest fixture, so the test controls
    exactly when the guard turns on -- in this file, only during the offline
    phase, not during the earlier online ``pull`` step, which is allowed to
    use the network."""
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

    # The socket guard is switched on only for the offline call below. The
    # online `pull` step above also talks to the provider, but since that
    # provider here is a fake running in-process (no real sockets involved),
    # there's nothing yet for the guard to catch, and it's safe to leave off.
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
    """The opposite case from the test above: what happens if you try
    ``run --offline`` WITHOUT ever running ``datasets pull`` first? This must
    still fail with a clear error -- it must never quietly "succeed" with no
    data, and it must never touch the real provider.

    Because this test's catalog is set up with a ``resolution_cache`` (see
    ADR-0011, the decision that added an on-disk cache of resolved dataset
    metadata), the error message here is more helpful than it used to be: it
    directly tells the user to run ``datasets pull`` to fix the problem,
    instead of ADR-0010's older, blanket message that just said dataset
    resolution is never cached at all.

    Under the hood, this better message exists because
    ``OfflineCacheMiss.retryable`` is ``True`` in this situation (there IS a
    resolution cache configured, it's just empty) -- versus ``False`` when no
    ``resolution_cache`` is configured at all. That ``True``/``False`` value
    itself is checked directly, at the catalog level, in
    ``tests/unit/datasets/test_catalog.py``. This test only checks what the
    CLI actually prints to the user, which is always in the form
    ``[code] message`` -- the underlying ``True``/``False`` value never shows
    up in the CLI's own output.
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
