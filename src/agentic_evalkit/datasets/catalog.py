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
always had. Per ADR-0010, whether ``offline=True`` is honored now turns on
the *provider's* declared :attr:`~agentic_evalkit.datasets.base.DatasetProvider.requires_network`,
not the operation alone:

- A provider that declares ``requires_network = False`` (the built-in
  ``local`` provider; any future provider with the same property) is safe to
  call under ``offline=True`` on every method, because "offline" means "do
  not use the network" and such a provider never touches the network in the
  first place. ``search``/``resolve``/``iter_records`` route straight
  through to it exactly as they would with ``offline=False``.
- A provider that declares ``requires_network = True`` (or omits the
  attribute entirely -- treated as ``True``, the conservative default so an
  older/third-party provider's behavior never silently changes) still cannot
  honor ``offline=True`` on ``search``, ``resolve``, or ``iter_records``:
  none of the three is backed by an exact-match cache key (a query has no
  stable cache key, a resolution is what produces the revision a cache key
  would need, and iteration is not cache-decorated at all), so
  ``offline=True`` on any of them raises
  :class:`~agentic_evalkit.errors.OfflineCacheMiss` with
  ``retryable=False`` (per ADR-0010, this is the "categorically
  uncacheable" case: no amount of prior or future warming makes an
  offline call to one of these three operations succeed against a
  network-requiring provider) rather than either contacting the provider
  or fabricating a stale result.
- ``preview`` is unaffected by ``requires_network`` and keeps its existing
  behavior regardless of provider: it is the one operation backed by an
  exact-match cache key (design §6.3), so it always serves from the cache
  when possible and only raises when no cache is configured or no exact
  entry exists.

The catalog does not remember or propagate the flag across calls.
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

#: Maximum characters of a free-text search query preserved verbatim in an
#: ``OfflineCacheMiss`` error's ``context["query"]``. A caller-supplied query
#: string is unbounded (design places no length limit on it), but error
#: context is surfaced in CLI stderr and, later, in structured logs/reports
#: -- an unbounded value would let an arbitrarily large query balloon a
#: single error message. Truncated values are suffixed with ``"...(truncated,
#: N chars total)"`` so the original length is never silently lost.
_MAX_QUERY_CONTEXT_CHARS = 200


def _truncate_query(query: str) -> str:
    """Bound ``query`` for safe inclusion in error context (ADR-0010).

    Returns ``query`` unchanged when it already fits within
    :data:`_MAX_QUERY_CONTEXT_CHARS`; otherwise returns the first
    ``_MAX_QUERY_CONTEXT_CHARS`` characters plus a suffix recording the
    original total length, so a truncated context value is always
    self-describing rather than silently lossy.
    """
    if len(query) <= _MAX_QUERY_CONTEXT_CHARS:
        return query
    return f"{query[:_MAX_QUERY_CONTEXT_CHARS]}...(truncated, {len(query)} chars total)"


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

    @staticmethod
    def _provider_requires_network(provider_impl: DatasetProvider) -> bool:
        """Read a provider's network-independence declaration (ADR-0010).

        Defaults to ``True`` (network-required) when the attribute is
        absent, so a provider implementation written before ADR-0010 --
        including every pre-existing test fake in this codebase's own suite
        -- keeps today's behavior (``offline=True`` is rejected for it)
        rather than silently becoming exempt. Only a provider that
        explicitly declares ``requires_network = False`` is ever routed
        through under ``offline=True``.
        """
        return bool(getattr(provider_impl, "requires_network", True))

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
        §6.3's cache identity model is page/dataset-keyed, not query-keyed).
        A provider that declares ``requires_network = False`` (ADR-0010) is
        still called normally under ``offline=True`` -- it never touches the
        network regardless -- but a network-requiring provider can never
        honor ``offline=True`` honestly here: it always raises rather than
        either silently contacting ``provider`` or silently returning a
        stale/empty result.

        Raises:
            KeyError: ``provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed and ``provider``
                requires network access; search results are never cached,
                so a search can never be served offline for such a
                provider. Raised with ``retryable=False`` (ADR-0010): no
                amount of warming makes a query-keyed search cacheable.
        """
        provider_impl = self._provider_for(provider)
        if offline and self._provider_requires_network(provider_impl):
            raise OfflineCacheMiss(
                message=(
                    "offline search requested but search results are never cached; "
                    f"provider {provider!r}, query {query!r} require contacting the provider"
                ),
                context={"provider": provider, "query": _truncate_query(query)},
                retryable=False,
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
        method does not add one. A provider that declares
        ``requires_network = False`` (ADR-0010, e.g. the built-in ``local``
        provider) is still resolved normally under ``offline=True`` — its
        "resolution" is reading a local file, not a network round trip — but
        a network-requiring provider's ``offline=True`` resolve always
        raises rather than serving a possibly-stale resolution or silently
        contacting the provider.

        Raises:
            KeyError: ``ref.provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed and ``ref.provider``
                requires network access; resolution always requires such a
                provider. Raised with ``retryable=False`` (ADR-0010): no
                amount of warming makes an offline resolve of a
                network-requiring provider succeed -- resolution is not
                itself cache-backed today.
        """
        provider_impl = self._provider_for(ref.provider)
        if offline and self._provider_requires_network(provider_impl):
            raise OfflineCacheMiss(
                message=(
                    "offline resolve requested but resolution is never cached; "
                    f"provider {ref.provider!r}, dataset {ref.dataset_id!r} require "
                    "contacting the provider to pin a revision"
                ),
                context={"provider": ref.provider, "dataset_id": ref.dataset_id},
                retryable=False,
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
                construction time. Unlike ``search``/``resolve``/
                ``iter_records``, this behavior does not depend on the
                provider's ``requires_network`` declaration (ADR-0010):
                ``preview`` is always cache-backed when a cache exists, for
                every provider.

        Raises:
            OfflineCacheMiss: ``offline=True`` and either no cache is
                configured on this catalog (``retryable=False`` -- no cache
                exists to ever warm) or no exact cached page exists yet for
                this ``(provider, dataset, offset, limit)`` (propagated from
                :meth:`agentic_evalkit.datasets.cache.DatasetCache.read`
                with its default ``retryable=True`` -- an online preview of
                this exact page would populate it).
        """
        if self._cache is None:
            if offline:
                raise OfflineCacheMiss(
                    message="offline preview requested but this catalog has no cache configured",
                    context={"provider": ref.provider, "dataset_id": dataset.dataset_id},
                    retryable=False,
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
        repeated :meth:`preview` calls instead. A provider that declares
        ``requires_network = False`` (ADR-0010) is still iterated normally
        under ``offline=True`` — it never touches the network regardless —
        but a network-requiring provider's ``offline=True`` iteration always
        raises.

        Args:
            offline: When ``True`` and ``ref.provider`` requires network
                access, this raises immediately rather than returning an
                iterator that would touch the provider on first iteration.
                Iteration is not backed by an exact-match cache key —
                providers stream records from their own source, outside
                this catalog's cache decoration — so honoring
                ``offline=True`` honestly for such a provider means
                rejecting it, never silently iterating from the provider.

        Raises:
            KeyError: ``ref.provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed and ``ref.provider``
                requires network access; iteration always requires such a
                provider. Raised with ``retryable=False`` (ADR-0010): no
                amount of warming makes an offline iteration of a
                network-requiring provider succeed -- iteration is never
                cache-backed at all.
        """
        provider_impl = self._provider_for(ref.provider)
        if offline and self._provider_requires_network(provider_impl):
            raise OfflineCacheMiss(
                message=(
                    "offline iteration requested but iter_records is not cache-backed; "
                    f"provider {ref.provider!r}, dataset {dataset.dataset_id!r} require "
                    "contacting the provider"
                ),
                context={"provider": ref.provider, "dataset_id": dataset.dataset_id},
                retryable=False,
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
