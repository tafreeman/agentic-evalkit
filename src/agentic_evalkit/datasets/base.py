"""Dataset provider protocol and shared provider health contract.

Design §6.1: every provider implements ``search``, ``resolve``, ``preview``,
``iter_records``, and ``healthcheck``. The protocol is structural, so host
adapters do not inherit framework classes; providers register through the
``agentic_evalkit.providers.v1`` entry-point group.

Per ADR-0010, every provider also declares ``requires_network`` -- a
structural marker parallel to ``api_version`` that says whether the provider
can ever legitimately be called while ``offline=True`` is in effect.
``DatasetCatalog`` reads this attribute (defaulting an unmarked provider to
``True``, the conservative assumption) to decide whether an offline call may
reach the provider at all, rather than special-casing the literal name
``"local"``.
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
    """The provider boundary (design §6.1)."""

    api_version: str
    #: Whether this provider ever needs network access to satisfy any of its
    #: methods. ``False`` marks a provider as safe to call under
    #: ``offline=True`` (ADR-0010) -- for example, a filesystem-only
    #: provider. ``True`` (or the attribute being absent on an older
    #: provider) means ``DatasetCatalog`` continues to reject
    #: ``offline=True`` calls to it.
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
