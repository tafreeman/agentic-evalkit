"""Socket-guard proof that ``--offline`` over the ``local`` provider makes
zero network syscalls (ADR-0010, plan Task 2 Step 3a).

The other catalog tests prove *behavioral* correctness (the right value is
returned, the right error is raised) using fake providers. This module goes
one level deeper: it patches the socket methods that actually reach out to a
remote address (``socket.socket.connect``/``connect_ex``, plus the
``socket.create_connection`` convenience wrapper both ``httpx`` and
``huggingface_hub``'s HTTP stack funnel outbound connections through) to
raise on any invocation, then drives a real (non-fake)
``LocalDatasetProvider``-backed ``DatasetCatalog`` through every
offline-affected method. If any code path under test tried to open an
outbound network connection, the patched methods would raise immediately
and the test would fail loudly rather than the test simply "happening" to
use a hermetic double.

Guard shape note (important on Windows -- two things had to be ruled out
before landing on this shape):

1. Patching the ``socket.socket`` *constructor* itself is too broad:
   ``asyncio``'s ``ProactorEventLoop`` (Windows' default event loop)
   constructs an internal self-pipe socket pair as part of ordinary
   event-loop setup/teardown, entirely unrelated to any outbound network
   attempt this test cares about. That made every ``async def`` test fail
   at ``pytest-asyncio`` fixture setup before the test body ever ran.
2. Patching ``socket.socket.connect``/``connect_ex`` *unconditionally* is
   also too broad on Windows specifically: the platform has no native
   ``socketpair()`` at the OS level, so CPython's standard library emulates
   ``socket.socketpair()`` (which asyncio's self-pipe setup calls) by
   creating a loopback listener and a second socket that genuinely calls
   ``.connect()`` against ``127.0.0.1`` to reach it. Blocking every
   ``connect``/``connect_ex`` call regardless of target address still
   caught this internal loopback connection.

The guard below therefore inspects the *address argument* passed to
``connect``/``connect_ex``/``create_connection`` and only raises for a
non-loopback destination. Real dataset-provider network code (``httpx`` to
``huggingface.co``, ``huggingface_hub`` to the Hub API) never legitimately
connects to ``127.0.0.1``/``::1``/``localhost``, so allowing loopback
through while raising on everything else is still a complete, correct proof
that no real outbound network call happened -- it no longer collides with
asyncio's own Windows-specific self-pipe implementation detail. No new
dependency is introduced; ``socket`` is standard library.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Any

import pytest

from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.models import DatasetRef

if TYPE_CHECKING:
    from pathlib import Path

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _address_host(address: object) -> str | None:
    """Extract the host component from a socket address, if there is one.

    ``connect``/``connect_ex`` take either a ``(host, port)`` tuple (IPv4)
    or a longer tuple (IPv6, ``(host, port, flowinfo, scopeid)``) or, more
    rarely, a single AF_UNIX path string. Only the two-plus-element tuple
    shapes have a meaningful "host" to check against the loopback allowlist;
    anything else returns ``None`` and is treated as non-loopback (fails
    closed rather than open).
    """
    if isinstance(address, tuple) and len(address) >= 2 and isinstance(address[0], str):
        return address[0]
    return None


def _is_loopback_address(address: object) -> bool:
    host = _address_host(address)
    return host is not None and host in _LOOPBACK_HOSTS


def _raise_unless_loopback(real: Any) -> Any:
    """Wrap a real socket method so only a non-loopback call raises.

    ``real`` is the original bound method (captured before patching), so a
    loopback call -- asyncio's own Windows ``socketpair()`` emulation --
    still behaves exactly as it would unpatched, while any other
    destination raises immediately.
    """

    def _guarded(self: socket.socket, address: object, *args: Any, **kwargs: Any) -> Any:
        if _is_loopback_address(address):
            return real(self, address, *args, **kwargs)
        raise AssertionError(
            "outbound network connection attempted during an --offline call "
            f"over the local provider (target={address!r})"
        )

    return _guarded


def _raise_unless_loopback_create_connection(real: Any) -> Any:
    def _guarded(address: object, *args: Any, **kwargs: Any) -> Any:
        if _is_loopback_address(address):
            return real(address, *args, **kwargs)
        raise AssertionError(
            "outbound network connection attempted during an --offline call "
            f"over the local provider (target={address!r})"
        )

    return _guarded


@pytest.fixture
def _forbid_outbound_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any non-loopback socket connection attempt fail the test immediately.

    Patches ``socket.socket.connect``/``connect_ex`` (the instance methods
    that take a remote address and actually attempt to reach it) and the
    ``socket.create_connection`` module-level convenience wrapper (what
    ``httpx``'s sync transport and ``huggingface_hub``'s urllib3-based HTTP
    stack ultimately call for a plain TCP connection), each wrapped to allow
    a loopback destination through unchanged (see the module docstring for
    why: Windows' ``socketpair()`` emulation, used by asyncio's own event
    loop, genuinely connects to ``127.0.0.1`` as an implementation detail
    unrelated to outbound network access).
    """
    monkeypatch.setattr(socket.socket, "connect", _raise_unless_loopback(socket.socket.connect))
    monkeypatch.setattr(
        socket.socket, "connect_ex", _raise_unless_loopback(socket.socket.connect_ex)
    )
    monkeypatch.setattr(
        socket,
        "create_connection",
        _raise_unless_loopback_create_connection(socket.create_connection),
    )


@pytest.fixture
def _local_dataset_path(tmp_path: Path) -> Path:
    source = tmp_path / "offline_fixture.jsonl"
    source.write_text('{"question":"2+2?","answer":"4"}\n{"question":"3+3?","answer":"6"}\n')
    return source


def _local_catalog(tmp_path: Path) -> DatasetCatalog:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    # This test supplies the genuine built-in "local" provider itself (the
    # same real class the CLI wires up), not a plugin masquerading under a
    # reserved name -- so, exactly like `cli/datasets.py`'s own
    # `build_catalog`, the reserved-name collision guard does not apply
    # here and must be disabled by passing an empty reserved-name tuple.
    return DatasetCatalog(providers={"local": provider}, builtin_provider_names=())


@pytest.mark.usefixtures("_forbid_outbound_network")
@pytest.mark.asyncio
async def test_offline_resolve_over_local_provider_makes_zero_network_calls(
    tmp_path: Path, _local_dataset_path: Path
) -> None:
    catalog = _local_catalog(tmp_path)
    ref = DatasetRef(provider="local", dataset_id=str(_local_dataset_path))
    resolved = await catalog.resolve(ref, offline=True)
    assert resolved.revision.startswith("sha256:")
    assert resolved.row_count == 2


@pytest.mark.usefixtures("_forbid_outbound_network")
@pytest.mark.asyncio
async def test_offline_search_over_local_provider_makes_zero_network_calls(
    tmp_path: Path,
) -> None:
    catalog = _local_catalog(tmp_path)
    page = await catalog.search("anything", provider="local", offline=True)
    # Local roots are not recursively indexed (design), so an empty,
    # *successful* page is the correct, network-free result here -- the
    # point of this test is that no OfflineCacheMiss was raised and no
    # outbound connection was attempted, not what the search returns.
    assert page.total_hits == 0


@pytest.mark.usefixtures("_forbid_outbound_network")
@pytest.mark.asyncio
async def test_offline_iter_records_over_local_provider_makes_zero_network_calls(
    tmp_path: Path, _local_dataset_path: Path
) -> None:
    catalog = _local_catalog(tmp_path)
    ref = DatasetRef(provider="local", dataset_id=str(_local_dataset_path))
    resolved = await catalog.resolve(ref, offline=True)
    records = [record async for record in catalog.iter_records(ref, resolved, offline=True)]
    assert [record.data["question"] for record in records] == ["2+2?", "3+3?"]


@pytest.mark.usefixtures("_forbid_outbound_network")
@pytest.mark.asyncio
async def test_offline_preview_over_local_provider_makes_zero_network_calls_even_uncached(
    tmp_path: Path, _local_dataset_path: Path
) -> None:
    """The local provider's ``preview`` never needs the cache to be
    pre-warmed under ``offline=True`` -- it is never a network call in the
    first place. This test drives ``LocalDatasetProvider.preview`` directly
    (rather than through ``DatasetCatalog.preview``, whose own
    "no cache configured" branch is provider-agnostic and already covered by
    ``test_catalog.py::test_preview_offline_with_no_cache_configured_is_not_retryable``)
    to isolate proof that the provider-level call itself never touches the
    network, with no cache configured at all.
    """
    catalog = _local_catalog(tmp_path)
    ref = DatasetRef(provider="local", dataset_id=str(_local_dataset_path))
    resolved = await catalog.resolve(ref, offline=True)
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    page = await provider.preview(resolved, offset=0, limit=10)
    assert page.total_rows == 2
