"""Dataset catalog: provider routing, presets, and cache decoration.

Design §6.1-§6.3, plan Task 7 Steps 3-4. ``DatasetCatalog`` is the single
entry point callers use instead of talking to a ``DatasetProvider`` directly:
it dispatches ``search``/``resolve``/``preview``/``iter_records`` to the
provider named by ``DatasetRef.provider`` (never by config, split, or any
other field), exposes the built-in verified presets from
:mod:`agentic_evalkit.datasets.presets`, and decorates ``preview`` with the
content-addressed cache from :mod:`agentic_evalkit.datasets.cache` so a
repeated request for the exact same page never re-hits the provider.

Providers are supplied by the caller as a name-to-provider mapping (built-ins
plus any entry-point-discovered plugins already loaded via
``agentic_evalkit.plugins.load_plugins("agentic_evalkit.providers.v1", ...)``
before construction). This module does not perform entry-point discovery
itself; it only enforces that a caller-supplied provider can never silently
replace a name reserved for a built-in provider (``builtin_provider_names``),
raising :class:`~agentic_evalkit.errors.PluginCompatibilityError` instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

from agentic_evalkit.datasets.base import DatasetProvider
from agentic_evalkit.datasets.cache import CacheKey, DatasetCache
from agentic_evalkit.datasets.presets import BUILTIN_PRESETS, DatasetPreset
from agentic_evalkit.errors import OfflineCacheMiss, PluginCompatibilityError
from agentic_evalkit.models import DatasetRef, ResolvedDataset, SamplePage, SearchPage, SourceRecord

__all__ = ["DatasetCatalog"]

_BUILTIN_PROVIDER_NAMES: tuple[str, ...] = ("local", "huggingface")


class DatasetCatalog:
    """Routes dataset operations by provider name and decorates ``preview`` with a cache.

    Args:
        providers: Every available provider, keyed by the name that
            ``DatasetRef.provider`` must match to route to it. This mapping
            is the caller's responsibility to assemble (built-ins plus any
            already-loaded entry-point plugins); the catalog only validates
            it for built-in-name collisions.
        presets: The named preset catalog exposed via :meth:`list_presets`.
            Defaults to :data:`agentic_evalkit.datasets.presets.BUILTIN_PRESETS`.
        cache: The content-addressed cache :meth:`preview` reads from and
            writes to. If ``None``, :meth:`preview` always calls the
            provider directly and ``offline=True`` is rejected (there is
            nothing to serve offline).
        builtin_provider_names: Provider names reserved for built-in
            providers. A name in both this tuple and ``providers`` raises
            :class:`PluginCompatibilityError` at construction time rather
            than letting a plugin silently replace a built-in. Defaults to
            ``("local", "huggingface")`` (design §6.1's initial built-in
            providers).

    Raises:
        PluginCompatibilityError: A key in ``providers`` collides with a
            name in ``builtin_provider_names``.
    """

    def __init__(
        self,
        providers: Mapping[str, DatasetProvider],
        *,
        presets: Mapping[str, DatasetPreset] = BUILTIN_PRESETS,
        cache: DatasetCache | None = None,
        builtin_provider_names: tuple[str, ...] = _BUILTIN_PROVIDER_NAMES,
    ) -> None:
        for name in providers:
            if name in builtin_provider_names:
                raise PluginCompatibilityError(
                    message=(
                        f"provider name {name!r} is reserved for a built-in provider and "
                        "cannot be registered by a plugin"
                    ),
                    context={"provider": name},
                )
        self._providers: dict[str, DatasetProvider] = dict(providers)
        self._presets: Mapping[str, DatasetPreset] = presets
        self._cache = cache

    def _provider_for(self, name: str) -> DatasetProvider:
        try:
            return self._providers[name]
        except KeyError as error:
            raise KeyError(f"provider {name!r} is not registered with this catalog") from error

    def list_presets(self) -> tuple[DatasetPreset, ...]:
        """Return every registered preset, in insertion order."""
        return tuple(self._presets.values())

    async def search(
        self,
        query: str,
        *,
        provider: str,
        filters: Mapping[str, str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchPage:
        """Route a search to the named provider (design §6.1)."""
        return await self._provider_for(provider).search(
            query, filters=filters, limit=limit, cursor=cursor
        )

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        """Route resolution to ``ref.provider`` — the only field used for routing."""
        return await self._provider_for(ref.provider).resolve(ref)

    async def preview(
        self,
        ref: DatasetRef,
        dataset: ResolvedDataset,
        *,
        offset: int = 0,
        limit: int = 10,
        offline: bool = False,
    ) -> SamplePage:
        """Return one page of ``dataset``, serving from cache when possible.

        Builds an exact :class:`~agentic_evalkit.datasets.cache.CacheKey`
        from ``ref.provider``, ``dataset``, ``offset``, and ``limit`` (a
        page-record-type key, per design §6.3). A verified cache hit returns
        the decoded page without calling the provider. A miss calls
        ``ref.provider``'s ``preview`` and writes the exact returned page
        back to the cache before returning it.

        Args:
            offline: When ``True``, this method never calls the provider —
                it only ever returns an exact cached page, or propagates
                :class:`~agentic_evalkit.errors.OfflineCacheMiss` /
                :class:`~agentic_evalkit.errors.DatasetIntegrityError` from
                the cache. Requires ``cache`` to have been supplied at
                construction time.

        Raises:
            OfflineCacheMiss: ``offline=True`` and no exact cached page
                exists for this ``(provider, dataset, offset, limit)``.
        """
        if self._cache is None:
            if offline:
                raise OfflineCacheMiss(
                    message="offline preview requested but this catalog has no cache configured",
                    context={"provider": ref.provider, "dataset_id": dataset.dataset_id},
                )
            return await self._provider_for(ref.provider).preview(
                dataset, offset=offset, limit=limit
            )

        key = self._cache_key(ref, dataset, offset=offset, limit=limit)

        try:
            cached_payload = self._cache.read(key)
        except OfflineCacheMiss:
            if offline:
                raise
        else:
            return SamplePage.model_validate_json(cached_payload)

        page = await self._provider_for(ref.provider).preview(dataset, offset=offset, limit=limit)
        self._cache.write(key, page.model_dump_json().encode("utf-8"))
        return page

    def iter_records(
        self,
        ref: DatasetRef,
        dataset: ResolvedDataset,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> AsyncIterator[SourceRecord]:
        """Route bounded iteration to ``ref.provider`` (not cache-decorated).

        Unlike :meth:`preview`, iteration is not paginated by a single
        exact-match cache key, so this always delegates directly to the
        provider; callers wanting cached, resumable pagination should use
        repeated :meth:`preview` calls instead.
        """
        return self._provider_for(ref.provider).iter_records(dataset, offset=offset, limit=limit)

    @staticmethod
    def _cache_key(
        ref: DatasetRef, dataset: ResolvedDataset, *, offset: int, limit: int
    ) -> CacheKey:
        return CacheKey(
            provider=ref.provider,
            dataset_id=dataset.dataset_id,
            revision=dataset.revision,
            config=dataset.config,
            split=dataset.split,
            offset=offset,
            limit=limit,
            record_type="page",
        )
