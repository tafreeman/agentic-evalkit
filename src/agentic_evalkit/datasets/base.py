"""Defines the interface ("protocol") every dataset provider must implement,
plus a shared type describing a provider's health status.

Design §6.1: every provider must implement five methods -- ``search``,
``resolve``, ``preview``, ``iter_records``, and ``healthcheck``. This is
defined as a ``Protocol`` (Python's structural-typing mechanism), which
means a class counts as a valid provider just by having methods with
matching names and signatures -- it does not need to explicitly inherit
from any base/framework class. Providers make themselves discoverable by
registering under the ``agentic_evalkit.providers.v1`` entry-point group (a
standard Python packaging mechanism plugins use to advertise themselves).

Per ADR-0010, every provider must also declare a ``requires_network``
attribute -- a flag, alongside ``api_version``, that says whether this
provider can ever legitimately be called while ``offline=True`` is set
(i.e. while the caller has said "don't touch the network"). ``DatasetCatalog``
reads this attribute to decide whether it's safe to let an offline call
reach a given provider at all. If a provider doesn't declare this
attribute, it's treated as ``True`` (needs network) by default -- the
cautious assumption -- rather than the catalog trying to guess based on
whether the provider happens to be named ``"local"``.
"""

from collections.abc import AsyncIterator, Mapping
from typing import Literal, Protocol, runtime_checkable

from agentic_evalkit.models import DatasetRef, ResolvedDataset, SamplePage, SearchPage, SourceRecord
from agentic_evalkit.models.base import FrozenModel


class ProviderHealth(FrozenModel):
    """Result of a provider healthcheck."""

    status: Literal["ok", "degraded", "error"]
    latency_ms: float | None = None
    capabilities: tuple[str, ...] = ()
    error_code: str | None = None


@runtime_checkable
class DatasetProvider(Protocol):
    """The interface that separates evalkit's core from any specific dataset
    provider (design §6.1). Anything that implements these methods with
    these signatures counts as a valid provider."""

    api_version: str
    #: Whether this provider ever needs network access to do any of its
    #: jobs. ``False`` means this provider is safe to call even when
    #: ``offline=True`` is set (ADR-0010) -- for example, a provider that
    #: only reads from the local filesystem. ``True`` (or simply leaving
    #: this attribute off, on an older provider written before this flag
    #: existed) means ``DatasetCatalog`` will keep rejecting
    #: ``offline=True`` calls made to this provider.
    requires_network: bool

    async def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchPage: ...

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset: ...

    async def preview(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int = 10
    ) -> SamplePage: ...

    def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]: ...

    async def healthcheck(self) -> ProviderHealth: ...
