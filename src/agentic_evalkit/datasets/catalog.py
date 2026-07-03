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

``offline`` is a per-call argument on every method, never construction-time
state on ``DatasetCatalog`` itself — the same per-call shape ``preview`` has
always had. Only ``preview`` can honor it (it is the one operation backed by
an exact-match cache key per design §6.3); ``search``, ``resolve``, and
``iter_records`` each inherently require the provider (a query has no stable
cache key, a resolution is what produces the revision a cache key would need,
and iteration is not cache-decorated at all), so ``offline=True`` on any of
them raises :class:`~agentic_evalkit.errors.OfflineCacheMiss` rather than
either contacting the provider or fabricating a stale result. The catalog
does not remember or propagate the flag across calls, so an offline caller
is limited to ``preview`` of exactly-cached pages; the other three
operations reject ``offline=True`` outright.
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
        offline: bool = False,
    ) -> SearchPage:
        """Route a search to the named provider (design §6.1).

        Search results are never cached (there is no stable, exact key for a
        free-text query the way there is for a resolved page — design
        §6.3's cache identity model is page/dataset-keyed, not query-keyed),
        so ``offline=True`` can never be honored honestly here: it always
        raises rather than either silently contacting ``provider`` or
        silently returning a stale/empty result.

        Raises:
            KeyError: ``provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed; search results
                are never cached, so a search can never be served offline.
        """
        provider_impl = self._provider_for(provider)
        if offline:
            raise OfflineCacheMiss(
                message=(
                    "offline search requested but search results are never cached; "
                    f"provider {provider!r}, query {query!r} require contacting the provider"
                ),
                context={"provider": provider, "query": query},
            )
        return await provider_impl.search(query, filters=filters, limit=limit, cursor=cursor)

    async def resolve(self, ref: DatasetRef, *, offline: bool = False) -> ResolvedDataset:
        """Route resolution to ``ref.provider`` — the only field used for routing.

        Resolution pins a revision (and often config/split/row-count/license
        metadata) by asking the provider, and the existing content-addressed
        cache (:mod:`agentic_evalkit.datasets.cache`) has no key for "the
        most recent resolution of this ref" — ``CacheKey.revision`` is
        required precisely because a resolution is what produces it, so
        there is no revision-independent key to resolve *from* cache without
        inventing a second, parallel keying scheme. Per design §6.3, this
        method does not add one: ``offline=True`` always raises rather than
        serving a possibly-stale resolution or silently contacting the
        provider.

        Raises:
            KeyError: ``ref.provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed; resolution always
                requires the provider.
        """
        provider_impl = self._provider_for(ref.provider)
        if offline:
            raise OfflineCacheMiss(
                message=(
                    "offline resolve requested but resolution is never cached; "
                    f"provider {ref.provider!r}, dataset {ref.dataset_id!r} require "
                    "contacting the provider to pin a revision"
                ),
                context={"provider": ref.provider, "dataset_id": ref.dataset_id},
            )
        return await provider_impl.resolve(ref)

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
        offline: bool = False,
    ) -> AsyncIterator[SourceRecord]:
        """Route bounded iteration to ``ref.provider`` (not cache-decorated).

        Unlike :meth:`preview`, iteration is not paginated by a single
        exact-match cache key, so this always delegates directly to the
        provider; callers wanting cached, resumable pagination should use
        repeated :meth:`preview` calls instead.

        Args:
            offline: When ``True``, this raises immediately rather than
                returning an iterator that would touch the provider on
                first iteration. Iteration is not backed by an exact-match
                cache key — providers (built-in or plugin) stream records
                from their own source, outside this catalog's cache
                decoration — so honoring ``offline=True`` honestly means
                rejecting it, never silently iterating from the provider.

        Raises:
            KeyError: ``ref.provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed; iteration always
                requires the provider.
        """
        provider_impl = self._provider_for(ref.provider)
        if offline:
            raise OfflineCacheMiss(
                message=(
                    "offline iteration requested but iter_records is not cache-backed; "
                    f"provider {ref.provider!r}, dataset {dataset.dataset_id!r} require "
                    "contacting the provider"
                ),
                context={"provider": ref.provider, "dataset_id": dataset.dataset_id},
            )
        return provider_impl.iter_records(dataset, offset=offset, limit=limit)

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
