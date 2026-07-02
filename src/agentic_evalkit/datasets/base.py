"""Dataset provider protocol and shared provider health contract.

Design §6.1: every provider implements ``search``, ``resolve``, ``preview``,
``iter_records``, and ``healthcheck``. The protocol is structural, so host
adapters do not inherit framework classes; providers register through the
``agentic_evalkit.providers.v1`` entry-point group.
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
