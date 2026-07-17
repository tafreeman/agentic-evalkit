"""Dataset catalog: picks which provider handles a request, exposes presets,
and adds caching around ``preview`` (design doc §6.1-§6.3, plan Task 7
Steps 3-4).

``DatasetCatalog`` is the single front door callers use instead of talking
to a specific ``DatasetProvider`` (e.g. the Hugging Face or local-file
backend) directly. It does three things:

1. Routes each ``search``/``resolve``/``preview``/``iter_records`` call to
   the provider named by ``DatasetRef.provider`` -- and only by that field;
   config, split, and everything else about the request are irrelevant to
   *which provider* handles it.
2. Exposes the built-in, verified preset datasets from
   :mod:`agentic_evalkit.datasets.presets`.
3. Wraps ``preview`` with the "content-addressed" cache from
   :mod:`agentic_evalkit.datasets.cache` -- a cache whose lookup key is
   built from the exact details of the request (which dataset, which
   revision, which page), so asking for the exact same page of the exact
   same dataset twice is served from the cache the second time instead of
   asking the provider again.

Providers themselves are supplied by the caller as a plain ``{name:
provider}`` mapping, built and handed to ``DatasetCatalog.__init__`` --
see :func:`agentic_evalkit.cli.datasets.build_catalog` for the reference
example, which constructs the built-in ``local`` and ``huggingface``
providers directly and passes them in. This module does not go looking for
providers on its own: an earlier version used Python's plugin/entry-point
mechanism (a way for separately-installed packages to register themselves
automatically, without this code needing to import them by name) to
auto-populate a mapping like this one, but ADR-0019 (an architecture
decision recorded in this project's docs) retracted that in favor of the
caller always supplying providers explicitly. The one thing this module
*does* still enforce on its own is that a caller-supplied provider can
never silently take over a name reserved for a built-in one
(``builtin_provider_names``) -- doing so raises
:class:`~agentic_evalkit.errors.PluginCompatibilityError` instead of
quietly letting the built-in provider be shadowed.

``offline`` is an argument passed on every individual call, never something
remembered on the ``DatasetCatalog`` object itself between calls -- the same
per-call shape ``preview`` has always had. "Offline" means "answer this
call without touching the network, using only what's already cached."
Per ADR-0010, whether ``offline=True`` actually works for a given call now
depends on the *provider's* own declaration of whether it needs the network
at all (:attr:`~agentic_evalkit.datasets.base.DatasetProvider.requires_network`),
not on which operation (search/resolve/etc.) is being called. Per ADR-0011,
``resolve`` and ``iter_records`` each get one narrow addition on top of
ADR-0010's rule: if an optional ``resolution_cache`` or page ``cache`` has
already been "warmed" -- i.e. already has a stored result -- by an earlier
*online* call (typically running ``datasets pull``) for that *exact same*
request, the stored (warmed) value is served instead of raising an error.
See each method's own docstring for the exact condition that has to hold.

Putting that together, here is what ``offline=True`` actually does for each
kind of provider and each method:

- A provider that declares ``requires_network = False`` (the built-in
  ``local`` provider, or any future provider with the same declaration) can
  always be called under ``offline=True`` for every method, because
  "offline" just means "don't use the network," and this kind of provider
  never uses the network regardless of the flag. ``search``, ``resolve``,
  and ``iter_records`` all route straight through to it exactly as they
  would if ``offline=False`` had been passed instead.
- A provider that declares ``requires_network = True`` (or simply doesn't
  declare the attribute at all -- which is treated as ``True``, the safe
  default that keeps an older or third-party provider's behavior from
  silently changing underneath it) still can never honor ``offline=True``
  on ``search``: a free-text search query has no stable, reusable cache key
  the way a specific dataset page does, so there's nothing to have cached
  in the first place. ``offline=True`` search against such a provider
  therefore always raises
  :class:`~agentic_evalkit.errors.OfflineCacheMiss` with
  ``retryable=False`` -- meaning this isn't a "try again once something
  warms the cache" situation, it is "categorically uncacheable": no amount
  of waiting or retrying will ever make it work. ``resolve`` and
  ``iter_records`` are *usually* in the same boat (a resolution is exactly
  what produces the revision a page-cache key would need in the first
  place, and iteration isn't cached by default) -- **except** when
  ADR-0011's ``resolution_cache``/``cache`` has already been warmed for
  this exact request (see each method's own docstring for details), in
  which case the warmed value is served with no provider call and no error
  at all. Without that warmed entry, both still raise exactly as they did
  before ADR-0011.
- ``preview`` ignores ``requires_network`` entirely and behaves the same
  way regardless of which provider is involved: it is the one operation
  backed by an exact-match cache key from the start (design doc §6.3) --
  every ``preview`` call always serves from the cache when the exact page
  is already stored there, and only raises when either no cache was
  configured at all, or the cache exists but doesn't have this exact page
  yet.

The catalog itself never remembers or carries the ``offline`` flag forward
from one call to the next -- it is re-decided fresh on every single call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_evalkit.datasets.cache import CacheKey, DatasetCache
from agentic_evalkit.datasets.presets import BUILTIN_PRESETS, DatasetPreset
from agentic_evalkit.datasets.resolution_cache import ResolutionCache, ResolutionKey
from agentic_evalkit.errors import OfflineCacheMiss, PluginCompatibilityError
from agentic_evalkit.models import DatasetRef, ResolvedDataset, SamplePage, SearchPage, SourceRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from agentic_evalkit.datasets.base import DatasetProvider

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
    """Cap the length of ``query`` so it's safe to include in error context (ADR-0010).

    Returns ``query`` unchanged if it already fits within
    :data:`_MAX_QUERY_CONTEXT_CHARS` characters. Otherwise, returns just the
    first ``_MAX_QUERY_CONTEXT_CHARS`` characters, with a suffix recording
    how long the original query really was -- so a truncated value always
    makes clear that it's been cut short (and by how much), instead of
    silently looking like the whole, unmodified query.
    """
    if len(query) <= _MAX_QUERY_CONTEXT_CHARS:
        return query
    return f"{query[:_MAX_QUERY_CONTEXT_CHARS]}...(truncated, {len(query)} chars total)"


class DatasetCatalog:
    """Picks which provider handles each dataset operation, and adds caching around ``preview``.

    Args:
        providers: Every available provider, keyed by the name that
            ``DatasetRef.provider`` must match for a request to be routed to
            it. Assembling and constructing this mapping is the caller's
            job (see :func:`agentic_evalkit.cli.datasets.build_catalog` for
            the reference example) -- the catalog itself only checks it for
            name collisions with the built-in providers.
        presets: The named collection of preset datasets exposed through
            :meth:`list_presets`. Defaults to
            :data:`agentic_evalkit.datasets.presets.BUILTIN_PRESETS`.
        cache: The "content-addressed" cache (one keyed by the exact
            details of the request, so identical requests always find the
            same entry) that :meth:`preview` reads from and writes to. If
            ``None``, :meth:`preview` always calls the provider directly,
            and passing ``offline=True`` is rejected -- there's no cache to
            serve a result from. Per ADR-0011, when a cache *is* set, it is
            also checked by :meth:`iter_records` when ``offline=True`` is
            passed for a provider that needs the network, using the exact
            same page-lookup key :meth:`preview` would use for that page.
        resolution_cache: Stores the most recent successful ``resolve()``
            result for each distinct ``(provider, dataset_id, config,
            split, revision)`` combination (ADR-0011). Here ``revision`` is
            whatever revision the *caller* originally asked for (their
            "pin") -- so two different pinned revisions of the same
            dataset are stored separately and never overwrite each other.
            If this is ``None`` (the default), :meth:`resolve` behaves
            exactly as it did before ADR-0011 introduced this cache: an
            ``offline=True`` resolve against a provider that needs the
            network always fails. If it is set, every successful
            :meth:`resolve` call saves its result here, and a later
            ``offline=True`` resolve against a network-requiring provider
            checks here first before giving up -- so a dataset that was
            already resolved once online (e.g. via running ``datasets
            pull``) can be resolved again offline, without contacting the
            provider a second time.
        builtin_provider_names: Provider names reserved for the built-in
            providers. If a name appears in both this tuple and in
            ``providers``, construction fails immediately with
            :class:`PluginCompatibilityError` rather than silently letting
            a caller-supplied provider (e.g. a third-party plugin) replace a
            trusted built-in one under the same name. Defaults to
            ``("local", "huggingface")`` -- the design doc's §6.1 initial
            set of built-in providers.

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
        """Read whether a provider says it needs the network to function (ADR-0010).

        If the provider doesn't declare a ``requires_network`` attribute at
        all, this defaults to ``True`` (assume it needs the network). That
        default matters for backward compatibility: a provider written
        before ADR-0010 introduced this attribute -- including every
        pre-existing test double ("fake") already in this codebase's test
        suite -- keeps today's behavior (``offline=True`` calls against it
        are rejected) instead of silently and unexpectedly starting to work
        offline just because it was never updated to declare the
        attribute. Only a provider that explicitly sets
        ``requires_network = False`` is ever allowed to run under
        ``offline=True``.
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
        """Send a search to the named provider (design doc §6.1).

        Search results are never cached at all: there's no stable, exact
        key for a free-text query the way there is for one specific,
        already-resolved page of a dataset -- the cache's whole identity
        scheme (design doc §6.3) is built around "this dataset, this
        page," not "this query string." A provider that declares
        ``requires_network = False`` (ADR-0010) is still called normally
        even when ``offline=True``, since it never touches the network
        regardless of that flag. But a provider that needs the network can
        never honestly honor ``offline=True`` here: rather than either
        silently making a real network call anyway, or silently handing
        back a stale or empty result, this always raises instead.

        Raises:
            KeyError: ``provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed and ``provider``
                needs network access. Because search results are never
                cached, a search can never be served offline for such a
                provider, no matter what. Raised with ``retryable=False``
                (ADR-0010) to signal that this isn't a "try again later"
                situation -- no amount of caching or warming will ever make
                a free-text search satisfiable offline.
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
        """Send resolution to ``ref.provider`` -- the only field this uses to pick a provider.

        "Resolving" a dataset means asking the provider to pin down an
        exact revision (and usually config, split, row count, license, and
        other metadata too) for a loose reference. The existing
        "content-addressed" page cache (:mod:`agentic_evalkit.datasets.cache`,
        keyed by the exact details of a request) has no way to represent
        "the most recent resolution of this reference": its cache key
        requires a revision to already be known (``CacheKey.revision``),
        but producing that revision is exactly what resolving does in the
        first place -- so design doc §6.3's page cache can never itself be
        used to answer a resolve call. Per ADR-0011, a *separate* cache,
        :class:`~agentic_evalkit.datasets.resolution_cache.ResolutionCache`,
        closes that gap when one is configured: every successful resolve
        (whether it happened online, or came from a provider that doesn't
        need the network at all) writes its result there, and a later
        ``offline=True`` resolve against a network-requiring provider
        checks there first before giving up and raising an error. A
        provider that declares ``requires_network = False`` (ADR-0010 --
        e.g. the built-in ``local`` provider) is always resolved normally
        under ``offline=True`` regardless of any of this, because its
        "resolution" just means reading a local file, never an actual
        round trip over the network.

        Raises:
            KeyError: ``ref.provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed, ``ref.provider``
                needs network access, and no cached resolution exists yet
                for this exact ``(provider, dataset_id, config, split)``
                combination. ``retryable`` is ``True`` when a
                ``resolution_cache`` is configured at all (because an
                online resolve, or running ``datasets pull``, for this
                exact request would then populate it and let a later
                offline call succeed) and ``False`` when no
                ``resolution_cache`` is configured (ADR-0010's original,
                unconditional "this can never work offline" behavior).
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
        """Return an already-cached (:class:`ResolutionCache`) resolution for ``ref``, or ``None``.

        A ``None`` return covers two different situations: "no resolution
        cache was configured on this catalog at all" and "one is
        configured, but doesn't have an entry for this exact request yet."
        Either way, the caller's only option is to fall back to raising an
        error. Note that a *corrupted* cache entry
        (:class:`~agentic_evalkit.errors.DatasetIntegrityError` -- meaning
        the cached data is damaged or doesn't match what it should) is
        deliberately **not** caught and hidden here: corruption is a real,
        serious problem and must never be quietly treated as if the cache
        simply had nothing stored (a "miss"). That distinction between
        "miss" and "corrupt" comes from ADR-0004, and ADR-0011 reuses the
        same rule here.
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
        # ``ref.revision`` is whatever exact revision the caller asked for
        # up front -- their "pin" (``None`` means "whatever is latest at
        # resolution time," rather than one specific fixed version). It's
        # included in the cache key so that two different pinned revisions
        # of the same dataset never share one cache slot. Without this: for
        # a dataset requested as both "revision A" and "revision B",
        # resolving "revision B" online would overwrite "revision A"'s
        # stored entry, and a later offline resolve asking for "revision A"
        # would silently be handed "revision B"'s resolution instead -- a
        # real bug that was fixed as ADR-0011 (2026-07-09). This only works
        # because ``DatasetRef.revision`` is actually honored by providers
        # (e.g. the Hugging Face provider passes it straight through to
        # ``dataset_info(..., revision=...)``), so two different requested
        # pins really do resolve to two different, unchanging revisions --
        # they're not just labels.
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
        """Return one page of ``dataset``, serving it from the cache when possible.

        Builds an exact :class:`~agentic_evalkit.datasets.cache.CacheKey`
        -- a lookup key built from ``ref.provider``, ``dataset``,
        ``offset``, and ``limit`` together (a key specifically for "one
        page of records," per design doc §6.3). If that exact key is
        already in the cache and passes its integrity check, the decoded
        page is returned straight away with no provider call at all. If
        it's not there, this calls ``ref.provider``'s own ``preview``
        method instead, and writes the exact page it returns back into the
        cache before handing it back to the caller -- so the *next*
        identical request will find it cached.

        Args:
            offline: When ``True``, this method will never call the
                provider under any circumstances -- it only ever returns an
                exact page already sitting in the cache, or lets
                :class:`~agentic_evalkit.errors.OfflineCacheMiss` /
                :class:`~agentic_evalkit.errors.DatasetIntegrityError` from
                the cache propagate up. This requires a ``cache`` to have
                been supplied when this catalog was constructed. Unlike
                ``search``/``resolve``/``iter_records``, this behavior does
                not depend at all on the provider's ``requires_network``
                declaration (ADR-0010): ``preview`` is always backed by the
                cache whenever one exists, no matter which provider is
                involved.

        Raises:
            OfflineCacheMiss: ``offline=True`` was passed, and either (a)
                no cache was configured on this catalog at all
                (``retryable=False`` -- there is no cache that could ever
                warm up and start working) or (b) a cache exists but
                doesn't have an exact cached page yet for this particular
                ``(provider, dataset, offset, limit)`` combination
                (propagated from
                :meth:`agentic_evalkit.datasets.cache.DatasetCache.read`,
                whose default is ``retryable=True`` -- meaning an online
                preview of this exact page later would populate the cache
                and let a retry succeed).
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
        """Send bounded iteration to ``ref.provider``, or serve it from the existing page cache.

        Unlike :meth:`preview`, iteration doesn't go through a single
        exact-match cache lookup by itself -- a caller can ask for any
        ``offset``/``limit`` shape, not just one page at a time. But per
        ADR-0011, if this exact call's ``offset``/``limit`` (as long as
        ``limit`` isn't ``None``) happens to already match a page that's
        already stored ("warmed") in ``cache`` in the same format
        :meth:`preview` would use -- for example because an earlier
        ``datasets pull`` or an earlier :meth:`preview` call already
        fetched this identical page -- then that cached entry is exactly
        what ``EvalRunner`` needs (it always calls this method with one
        fixed ``offset``/``limit`` per run), so it's served straight from
        there under ``offline=True`` instead of raising an error. Any
        other shape -- an iteration with no upper bound (``limit=None``),
        or one where no matching cache entry exists -- still goes directly
        to the provider when that's allowed, or raises when it isn't. A
        caller that wants guaranteed, resumable, cache-backed pagination
        should still prefer making repeated :meth:`preview` calls instead
        of relying on this method's caching. A provider that declares
        ``requires_network = False`` (ADR-0010) is always iterated
        normally under ``offline=True`` regardless of any of the above,
        since it never touches the network to begin with.

        Args:
            offline: When ``True`` and ``ref.provider`` needs network
                access, this either raises right away, or (per ADR-0011
                above) returns an iterator backed entirely by the cache --
                it never returns an iterator that would end up quietly
                calling the provider once you start consuming it.

        Raises:
            KeyError: ``ref.provider`` is not registered with this catalog.
            OfflineCacheMiss: ``offline=True`` was passed, ``ref.provider``
                needs network access, and no cached page covers this exact
                call. ``retryable`` is ``True`` when ``limit`` is not
                ``None`` and a page ``cache`` is configured at all (because
                an online ``preview``/``pull`` of this exact page would
                populate it and let a retry succeed) and ``False``
                otherwise -- i.e. for an unbounded iteration, or when no
                cache is configured at all (ADR-0010's original,
                unconditional "this can never work offline" behavior).
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
        """Return an already-cached page's records as an async iterator, or ``None``.

        A ``None`` return covers three different situations: iteration
        with no upper bound (``limit is None``, which has no single
        well-defined page to look up in the cache), no page cache
        configured on this catalog at all, and a page cache that exists
        but doesn't have an exact entry yet for this particular ``(offset,
        limit)``. In every one of those cases, the caller has to fall back
        to either raising an error or delegating to the provider directly.
        As in :meth:`_cached_resolution`, a corrupted cache entry
        (:class:`~agentic_evalkit.errors.DatasetIntegrityError`) is
        deliberately **not** caught here -- it's allowed to propagate
        rather than being silently treated as if nothing were cached.
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
