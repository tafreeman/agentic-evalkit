"""Proof, at the network-socket level, that running with ``--offline`` against
the ``local`` provider makes zero real network system calls (ADR-0010, plan
Task 2 Step 3a).

The rest of the test suite for the dataset catalog already proves
*behavioral* correctness for offline mode -- the right value comes back, or
the right error gets raised -- but it does that using fake/stub providers.
This file goes one level deeper: it replaces (patches) the actual low-level
socket methods that any code has to go through to reach a remote address --
``socket.socket.connect``/``connect_ex``, plus the ``socket.create_connection``
helper function, which both the ``httpx`` and ``huggingface_hub`` libraries
ultimately call under the hood to open an outbound connection -- so that
calling any of them raises an error. It then runs a real
``LocalDatasetProvider``-backed ``DatasetCatalog`` (not a fake one) through
every method that offline mode affects. If any of that code had tried to
open a real outbound network connection, the patched socket method would
have raised immediately, and the test would fail loudly -- rather than the
test merely passing because it happened to use a stand-in that was never
going to touch the network anyway.

A note on the exact shape of this guard (important on Windows specifically --
two earlier approaches had to be ruled out before arriving at the one used
below):

1. Patching the ``socket.socket`` class constructor itself -- i.e., blocking
   the creation of *any* socket object -- is too broad. On Windows,
   asyncio's default event loop (``ProactorEventLoop``) creates an internal
   pair of connected sockets (a "self-pipe") as a normal part of starting up
   and shutting down, and this has nothing to do with the outbound network
   calls this test cares about. Blocking it broke event-loop startup itself,
   so every ``async def`` test failed while ``pytest-asyncio`` was still
   setting up its fixtures -- before the test body ever ran.
2. Patching ``socket.socket.connect``/``connect_ex`` to always raise, no
   matter the destination, is *also* too broad specifically on Windows:
   Windows has no built-in equivalent of Unix's ``socketpair()`` (a function
   that creates two sockets already connected to each other, without going
   over a real network). CPython's standard library fills that gap by
   opening a listening socket on the loopback address and a second socket
   that genuinely calls ``.connect()`` on ``127.0.0.1`` to reach it -- and
   asyncio's self-pipe setup relies on that emulation. So blocking every
   ``connect``/``connect_ex`` call regardless of destination still caught --
   and broke -- this internal loopback connection.

So the guard defined below instead looks at *which address* is being passed
to ``connect``/``connect_ex``/``create_connection``, and only raises for a
non-loopback destination. Real network code from a dataset provider
(``httpx`` talking to ``huggingface.co``, or ``huggingface_hub`` talking to
the Hub API) never has a legitimate reason to connect to
``127.0.0.1``/``::1``/``localhost``, so letting loopback connections through
while raising on everything else is still complete, correct proof that no
real outbound network call happened -- and it no longer conflicts with
asyncio's own Windows-specific internal plumbing. This doesn't add any new
dependency: ``socket`` is part of the Python standard library.
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
    """Pull the hostname out of a socket address, if it has one.

    ``connect``/``connect_ex`` are called with different shapes of address
    depending on the connection type: a ``(host, port)`` tuple for IPv4, a
    longer ``(host, port, flowinfo, scopeid)`` tuple for IPv6, or, rarely, a
    single file-path string for a Unix domain socket. Only the tuple shapes
    have a "host" worth checking against the loopback allowlist. For
    anything else, return ``None``, which the caller treats as "not
    loopback" -- i.e., when in doubt, block the connection rather than
    silently allow it.
    """
    if isinstance(address, tuple) and len(address) >= 2 and isinstance(address[0], str):
        return address[0]
    return None


def _is_loopback_address(address: object) -> bool:
    host = _address_host(address)
    return host is not None and host in _LOOPBACK_HOSTS


def _raise_unless_loopback(real: Any) -> Any:
    """Wrap a real socket method so it only raises for a non-loopback call.

    ``real`` is the original method, captured before we patch over it, so a
    loopback call -- which is just asyncio's own Windows ``socketpair()``
    workaround described above, not a real network attempt -- still behaves
    exactly as it would if nothing were patched, while any other
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
    """Make any attempt to open a real (non-loopback) network connection fail
    the test immediately.

    Patches ``socket.socket.connect``/``connect_ex`` (the methods that
    actually attempt to reach a remote address) and ``socket.create_connection``
    (a standard-library helper function that ``httpx``'s synchronous transport
    and ``huggingface_hub``'s HTTP stack both ultimately call to open a plain
    TCP connection). Each patched version still lets a loopback destination
    through unchanged -- see the module docstring above for why: Windows'
    emulation of ``socketpair()``, used internally by asyncio's own event
    loop, genuinely connects to ``127.0.0.1`` as an implementation detail
    that has nothing to do with real outbound network access.
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
    # We're registering the real, built-in "local" provider here (the same
    # class the CLI itself uses) under the name "local" -- this is not some
    # unrelated plugin impersonating a reserved built-in name. The catalog
    # normally guards against that kind of name collision, but the guard
    # doesn't apply to this legitimate case, so -- just like
    # `cli/datasets.py`'s own `build_catalog` does -- we pass an empty tuple
    # for `builtin_provider_names` to turn that guard off.
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
    # By design, the local provider doesn't scan its allowed directories for
    # searchable content, so getting back an empty (but successful) page is
    # the expected, network-free result here -- not a bug. The point of this
    # test isn't what search returns; it's that calling it didn't raise an
    # OfflineCacheMiss error and didn't attempt any outbound connection.
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
    """The local provider's ``preview`` method never needs data to already be
    cached, even with ``offline=True`` -- reading from the local filesystem
    was never a network call to begin with. This test calls
    ``LocalDatasetProvider.preview`` directly, rather than going through
    ``DatasetCatalog.preview`` (whose own "no cache configured" handling is
    shared by every provider and is already tested separately, in
    ``test_catalog.py::test_preview_offline_with_no_cache_configured_is_not_retryable``).
    Calling the provider directly isolates the proof that this specific
    method never touches the network -- even with no cache set up at all.
    """
    catalog = _local_catalog(tmp_path)
    ref = DatasetRef(provider="local", dataset_id=str(_local_dataset_path))
    resolved = await catalog.resolve(ref, offline=True)
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    page = await provider.preview(resolved, offset=0, limit=10)
    assert page.total_rows == 2
