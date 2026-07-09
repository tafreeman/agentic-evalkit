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
not the operation alone. Per ADR-0011, ``resolve`` and ``iter_records`` each
gain one narrow, additive exception to ADR-0010's rejection below: when an
optional ``resolution_cache``/``cache`` has already been warmed by a prior
online call (typically ``datasets pull``) for the *exact same* request, the
warmed value is served instead of raising -- see each method's own
docstring for the precise condition.

- A provider that declares ``requires_network = False`` (the built-in
  ``local`` provider; any future provider with the same property) is safe to
  call under ``offline=True`` on every method, because "offline" means "do
  not use the network" and such a provider never touches the network in the
  first place. ``search``/``resolve``/``iter_records`` route straight
  through to it exactly as they would with ``offline=False``.
- A provider that declares ``requires_network = True`` (or omits the
  attribute entirely -- treated as ``True``, the conservative default so an
  older/third-party provider's behavior never silently changes) still cannot
  honor ``offline=True`` on ``search``: a free-text query has no stable cache
  key, so ``offline=True`` search always raises
  :class:`~agentic_evalkit.errors.OfflineCacheMiss` with ``retryable=False``
  (the "categorically uncacheable" case). ``resolve`` and ``iter_records``
  are *usually* the same -- a resolution is what produces the revision a page
  cache key would need, and iteration is not cache-decorated by default --
  **unless** ADR-0011's ``resolution_cache``/``cache`` has already been
  warmed for this exact request (see each method's docstring), in which case
  the warmed value is served and no provider call or raise happens at all.
  Absent that warm entry, both still raise exactly as before.
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
from agentic_evalkit.datasets.resolution_cache import ResolutionCache, ResolutionKey
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
            nothing to serve offline). Per ADR-0011, when set, this same
            cache is also consulted by :meth:`iter_records` under
            ``offline=True`` for a network-requiring provider, using the
            identical page key :meth:`preview` would use.
        resolution_cache: Persists the most recent successful resolution per
            ``(provider, dataset_id, config, split, revision)`` request
            identity (ADR-0011), where ``revision`` is the caller's
            *requested* pin -- so two distinct pinned revisions of the same
            dataset never collide. If
            ``None`` (the default), :meth:`resolve` behaves exactly as
            before ADR-0011: an ``offline=True`` resolve against a
            network-requiring provider always raises. If set, every
            successful :meth:`resolve` call writes the resolved identity
            here, and an ``offline=True`` resolve against a network-requiring
            provider first consults it before raising -- so a dataset
            resolved online once (e.g. via ``datasets pull``) can be
            resolved again offline without contacting the provider.
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
        resolution_cache: ResolutionCache | None = None,
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
        self._resolution_cache = resolution_cache

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
        page cache (:mod:`agentic_evalkit.datasets.cache`) has no key for
        "the most recent resolution of this ref" — ``CacheKey.revision`` is
        required precisely because a resolution is what produces it, so
        design §6.3's page cache cannot itself serve a resolution. Per
        ADR-0011, a *separate* :class:`~agentic_evalkit.datasets.resolution_cache.ResolutionCache`
        closes that gap when one is configured: every successful resolve
        (online, or from a network-free provider) writes the resolved
        identity there, and an ``offline=True`` resolve against a
        network-requiring provider consults it before raising. A provider
        that declares ``requires_network = False`` (ADR-0010, e.g. the
        built-in ``local`` provider) is always resolved normally under
        ``offline=True`` regardless — its "resolution" is reading a local
        file, not a network round trip.

        Raises:
            KeyError: ``ref.provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed, ``ref.provider``
                requires network access, and no cached resolution exists for
                this exact ``(provider, dataset_id, config, split)``.
                ``retryable`` is ``True`` when a ``resolution_cache`` is
                configured (an online resolve, or ``datasets pull``, for
                this exact request would populate it) and ``False`` when no
                ``resolution_cache`` is configured at all (ADR-0010's
                original, unconditional behavior).
        """
        provider_impl = self._provider_for(ref.provider)
        if offline and self._provider_requires_network(provider_impl):
            cached = self._cached_resolution(ref)
            if cached is not None:
                return cached
            raise self._offline_resolve_miss(ref)
        resolved = await provider_impl.resolve(ref)
        if self._resolution_cache is not None:
            self._resolution_cache.write(self._resolution_key(ref), resolved)
        return resolved

    def _cached_resolution(self, ref: DatasetRef) -> ResolvedDataset | None:
        """Return a warmed :class:`ResolutionCache` entry for ``ref``, or ``None``.

        ``None`` covers both "no resolution cache is configured" and "one is
        configured but has no exact entry for this request yet" -- both mean
        the caller must fall back to raising. A corrupt entry
        (:class:`~agentic_evalkit.errors.DatasetIntegrityError`) is
        deliberately NOT caught here: corruption must never be silently
        treated as a miss (ADR-0004's distinction, reused by ADR-0011).
        """
        if self._resolution_cache is None:
            return None
        try:
            return self._resolution_cache.read(self._resolution_key(ref))
        except OfflineCacheMiss:
            return None

    def _offline_resolve_miss(self, ref: DatasetRef) -> OfflineCacheMiss:
        if self._resolution_cache is None:
            message = (
                "offline resolve requested but resolution is never cached; "
                f"provider {ref.provider!r}, dataset {ref.dataset_id!r} require "
                "contacting the provider to pin a revision"
            )
        else:
            message = (
                "offline resolve requested but no cached resolution exists yet; "
                f"provider {ref.provider!r}, dataset {ref.dataset_id!r} require an online "
                "resolve (e.g. 'datasets pull') first to pin a revision"
            )
        return OfflineCacheMiss(
            message=message,
            context={"provider": ref.provider, "dataset_id": ref.dataset_id},
            retryable=self._resolution_cache is not None,
        )

    @staticmethod
    def _resolution_key(ref: DatasetRef) -> ResolutionKey:
        # ``ref.revision`` is the caller's *requested* pin (``None`` = "latest
        # at resolution time"), included so two distinct pinned revisions of
        # the same dataset never share a cache slot: without it, resolving
        # ``ref@revB`` online would overwrite ``ref@revA``'s entry and an
        # offline ``resolve(ref@revA)`` would be silently served ``revB``'s
        # resolution (ADR-0011, 2026-07-09 fix). ``DatasetRef.revision`` is
        # provider-honored (e.g. the Hugging Face provider forwards it to
        # ``dataset_info(..., revision=...)``), so different pins genuinely
        # resolve to different immutable revisions.
        return ResolutionKey(
            provider=ref.provider,
            dataset_id=ref.dataset_id,
            config=ref.config,
            split=ref.split,
            revision=ref.revision,
        )

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
        """Route bounded iteration to ``ref.provider``, or serve it from the page cache.

        Unlike :meth:`preview`, iteration is not itself paginated by a
        single exact-match cache key call -- a caller may request any
        ``offset``/``limit`` shape. But per ADR-0011, when this exact call's
        ``offset``/``limit`` (with ``limit`` not ``None``) already has a
        warmed :meth:`preview`-compatible page entry in ``cache`` (e.g. from
        an earlier ``datasets pull`` or ``preview`` of this identical page),
        that entry is exactly what ``EvalRunner`` needs (it calls this
        method with one fixed ``offset``/``limit`` per run), so it is served
        from there under ``offline=True`` instead of raising. Any other
        shape -- an unbounded (``limit=None``) iteration, or no matching
        cache entry -- still delegates directly to the provider when
        allowed, or raises when not; callers wanting guaranteed cached,
        resumable pagination should still prefer repeated :meth:`preview`
        calls. A provider that declares ``requires_network = False``
        (ADR-0010) is always iterated normally under ``offline=True``
        regardless -- it never touches the network in the first place.

        Args:
            offline: When ``True`` and ``ref.provider`` requires network
                access, this raises immediately (or returns a cache-backed
                iterator, per ADR-0011 above) rather than returning an
                iterator that would touch the provider on first iteration.

        Raises:
            KeyError: ``ref.provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed, ``ref.provider``
                requires network access, and no cached page covers this
                exact call. ``retryable`` is ``True`` when ``limit`` is not
                ``None`` and a page ``cache`` is configured (an online
                ``preview``/``pull`` of this exact page would populate it)
                and ``False`` otherwise (unbounded iteration, or no cache
                configured at all -- ADR-0010's original, unconditional
                behavior).
        """
        provider_impl = self._provider_for(ref.provider)
        if offline and self._provider_requires_network(provider_impl):
            cached_records = self._cached_page_records(ref, dataset, offset=offset, limit=limit)
            if cached_records is not None:
                return cached_records
            raise OfflineCacheMiss(
                message=(
                    "offline iteration requested but iter_records is not cache-backed for "
                    f"provider {ref.provider!r}, dataset {dataset.dataset_id!r} without an "
                    "exact prior page cached via 'datasets pull' or 'preview' at this "
                    f"(offset={offset}, limit={limit})"
                ),
                context={"provider": ref.provider, "dataset_id": dataset.dataset_id},
                retryable=limit is not None and self._cache is not None,
            )
        return provider_impl.iter_records(dataset, offset=offset, limit=limit)

    def _cached_page_records(
        self, ref: DatasetRef, dataset: ResolvedDataset, *, offset: int, limit: int | None
    ) -> AsyncIterator[SourceRecord] | None:
        """Return a warmed page's records as an async iterator, or ``None``.

        ``None`` covers "unbounded iteration" (``limit is None``, which has
        no single-page cache key), "no page cache configured", and "cache
        configured but no exact entry for this ``(offset, limit)`` yet" --
        every one of those means the caller must fall back to raising or
        delegating to the provider. A corrupt entry
        (:class:`~agentic_evalkit.errors.DatasetIntegrityError`) is
        deliberately NOT caught here, matching :meth:`_cached_resolution`.
        """
        if limit is None or self._cache is None:
            return None
        key = self._cache_key(ref, dataset, offset=offset, limit=limit)
        try:
            cached_payload = self._cache.read(key)
        except OfflineCacheMiss:
            return None
        page = SamplePage.model_validate_json(cached_payload)
        return _records_from_page(page)

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


async def _records_from_page(page: SamplePage) -> AsyncIterator[SourceRecord]:
    """Adapt an already-decoded :class:`SamplePage` to an async record iterator."""
    for record in page.records:
        yield record
