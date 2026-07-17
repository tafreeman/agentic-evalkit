"""Tests for the dataset catalog: the router that dispatches requests to
dataset providers, the built-in dataset presets, and the caching layer
wrapped around the whole thing.

Covers: built-in presets (like "gsm8k") have exactly the field values they're
supposed to -- their dataset id, config, split, grading adapter, and so on
are all "pinned" (locked to fixed, known-correct values); the catalog picks
the right provider by name and raises a clear ``KeyError`` for an
unrecognized one instead of failing silently or confusingly; calling
``preview`` twice with identical arguments is served from the cache the
second time (a "cache hit") rather than calling the provider again, while
different arguments correctly trigger a fresh provider call (a "cache
miss"); passing ``offline=True`` must never reach out to a provider, even
when the requested data isn't cached; and registering a plugin provider
under a name that collides with a built-in one is rejected, while a
genuinely new name is accepted. The first two tests below are copied,
unmodified, from a code snippet written out in full in the original
implementation plan (Task 7 Step 1 of
docs/plans/2026-07-02-agentic-evalkit-initial-release.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agentic_evalkit.datasets.base import ProviderHealth
from agentic_evalkit.datasets.cache import DatasetCache
from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.presets import BUILTIN_PRESETS
from agentic_evalkit.datasets.resolution_cache import ResolutionCache
from agentic_evalkit.errors import OfflineCacheMiss, PluginCompatibilityError
from agentic_evalkit.models import (
    ContaminationStatus,
    DatasetRef,
    ResolvedDataset,
    SamplePage,
    SearchPage,
    SourceRecord,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

# --- Step 1 (copied word-for-word from the plan doc): preset and provider-routing tests ---


def test_builtin_presets_pin_configs_splits_and_adapters() -> None:
    gsm = BUILTIN_PRESETS["gsm8k"]
    swe = BUILTIN_PRESETS["swe-bench-verified"]
    assert (gsm.ref.dataset_id, gsm.ref.config, gsm.ref.split) == (
        "openai/gsm8k",
        "main",
        "test",
    )
    assert gsm.adapter == "gsm8k@1"
    assert swe.ref.config == "default"
    assert swe.readiness == "prediction_export"


@pytest.mark.asyncio
async def test_unknown_provider_is_explicit() -> None:
    catalog = DatasetCatalog(providers={})
    with pytest.raises(KeyError, match="provider 'missing'"):
        await catalog.search("x", provider="missing", limit=10)


# --- Additional preset coverage ----------------------------------------------


def test_builtin_presets_full_field_set() -> None:
    gsm = BUILTIN_PRESETS["gsm8k"]
    swe = BUILTIN_PRESETS["swe-bench-verified"]

    assert gsm.name == "gsm8k"
    assert gsm.grader == "normalized-exact@1"
    assert gsm.readiness == "runnable"
    assert gsm.required_capabilities == ()
    assert gsm.ref.provider == "huggingface"
    assert gsm.ref.split == "test"

    assert swe.name == "swe-bench-verified"
    assert swe.ref.dataset_id == "princeton-nlp/SWE-bench_Verified"
    assert swe.ref.split == "test"
    assert swe.adapter == "swebench-verified@1"
    assert swe.grader == "swebench-harness@1"
    assert swe.required_capabilities == ("swebench",)
    assert swe.ref.provider == "huggingface"

    # Both gsm8k and SWE-bench-Verified are long-established, publicly
    # available benchmarks, so there's a real chance parts of them ended up
    # in some model's training data ("contamination") without anyone being
    # able to say for sure either way. ADR-0013 says we must not pretend
    # they're clean -- both are honestly labeled SUSPECT rather than left
    # unmarked.
    assert gsm.contamination is not None
    assert gsm.contamination.status is ContaminationStatus.SUSPECT
    assert swe.contamination is not None
    assert swe.contamination.status is ContaminationStatus.SUSPECT


def test_all_builtin_presets_declare_a_contamination_status() -> None:
    """If a preset is ever added later without a contamination label, this
    test fails immediately instead of letting it silently ship unlabeled
    (ADR-0013 requires every preset to carry one). This mirrors how
    ``_build_builtin_presets`` itself refuses to even start up if it finds a
    duplicate preset, rather than quietly ignoring the problem."""
    for preset in BUILTIN_PRESETS.values():
        assert preset.contamination is not None, preset.name


def test_builtin_presets_are_frozen_and_forbid_unknown_fields() -> None:
    gsm = BUILTIN_PRESETS["gsm8k"]
    with pytest.raises(Exception):  # noqa: B017 - pydantic.ValidationError, frozen instance
        gsm.name = "renamed"  # type: ignore[misc]


def test_builtin_presets_mapping_is_immutable() -> None:
    with pytest.raises(TypeError):
        BUILTIN_PRESETS["new-preset"] = BUILTIN_PRESETS["gsm8k"]  # type: ignore[index]


# --- Fake provider used across catalog tests ---------------------------------


class _CountingFakeProvider:
    """A minimal fake dataset provider that counts how many times
    ``preview`` is called.

    Used to prove the catalog's caching actually works: calling ``preview``
    again with the exact same arguments must NOT increase
    ``preview_calls`` (it should be served from the cache instead), while
    calling it with different pagination arguments (asking for a different
    page of results) must.
    """

    api_version = "1"

    def __init__(self) -> None:
        self.preview_calls = 0
        self.resolve_calls = 0

    async def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchPage:
        return SearchPage(hits=(), cursor=None, total_hits=0)

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        self.resolve_calls += 1
        return ResolvedDataset(
            dataset_id=ref.dataset_id,
            revision="abc123",
            config=ref.config,
            split=ref.split,
        )

    async def preview(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int = 10
    ) -> SamplePage:
        self.preview_calls += 1
        records = tuple(
            SourceRecord(row_id=str(offset + i), data={"value": offset + i}, digest=f"sha256:{i}")
            for i in range(limit)
        )
        return SamplePage(records=records, offset=offset, total_rows=1000)

    def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        raise NotImplementedError

    async def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(status="ok")


def _fake_ref() -> DatasetRef:
    return DatasetRef(provider="fake", dataset_id="fake/dataset", config="main", split="test")


# --- Step 5: cache hit, offline miss, plugin collision -----------------------


@pytest.mark.asyncio
async def test_second_identical_preview_uses_cache_not_provider(tmp_path: Path) -> None:
    provider = _CountingFakeProvider()
    cache = DatasetCache(tmp_path)
    catalog = DatasetCatalog(providers={"fake": provider}, cache=cache)
    ref = _fake_ref()
    dataset = await catalog.resolve(ref)

    first = await catalog.preview(ref, dataset, offset=0, limit=5)
    second = await catalog.preview(ref, dataset, offset=0, limit=5)

    assert provider.preview_calls == 1
    assert first == second


@pytest.mark.asyncio
async def test_different_offset_invokes_provider_again(tmp_path: Path) -> None:
    provider = _CountingFakeProvider()
    cache = DatasetCache(tmp_path)
    catalog = DatasetCatalog(providers={"fake": provider}, cache=cache)
    ref = _fake_ref()
    dataset = await catalog.resolve(ref)

    await catalog.preview(ref, dataset, offset=0, limit=5)
    await catalog.preview(ref, dataset, offset=5, limit=5)

    assert provider.preview_calls == 2


@pytest.mark.asyncio
async def test_offline_mode_returns_only_exact_cached_pages(tmp_path: Path) -> None:
    provider = _CountingFakeProvider()
    cache = DatasetCache(tmp_path)
    catalog = DatasetCatalog(providers={"fake": provider}, cache=cache)
    ref = _fake_ref()
    dataset = await catalog.resolve(ref)

    # Populate the cache for offset=0 first.
    warmed = await catalog.preview(ref, dataset, offset=0, limit=5)
    assert provider.preview_calls == 1

    # A cached, exact page is served offline without touching the provider.
    served_offline = await catalog.preview(ref, dataset, offset=0, limit=5, offline=True)
    assert served_offline == warmed
    assert provider.preview_calls == 1

    # A page with no cache entry raises rather than silently calling the
    # provider.
    with pytest.raises(Exception):  # noqa: B017 - agentic_evalkit.errors.OfflineCacheMiss
        await catalog.preview(ref, dataset, offset=5, limit=5, offline=True)
    assert provider.preview_calls == 1


@pytest.mark.asyncio
async def test_registering_plugin_with_builtin_name_raises_compatibility_error() -> None:
    plugin_provider = _CountingFakeProvider()
    with pytest.raises(PluginCompatibilityError, match="huggingface"):
        DatasetCatalog(
            providers={"huggingface": plugin_provider},
            builtin_provider_names=("local", "huggingface"),
        )


@pytest.mark.asyncio
async def test_registering_plugin_with_new_name_succeeds() -> None:
    plugin_provider = _CountingFakeProvider()
    catalog = DatasetCatalog(
        providers={"custom": plugin_provider},
        builtin_provider_names=("local", "huggingface"),
    )
    ref = DatasetRef(provider="custom", dataset_id="x", config="main", split="test")
    page = await catalog.search("query", provider="custom", limit=1)
    assert page.total_hits == 0
    resolved = await catalog.resolve(ref)
    assert resolved.dataset_id == "x"


# --- offline mode on operations that have no cache entry to fall back on ------


@pytest.mark.asyncio
async def test_search_offline_always_raises_without_calling_provider() -> None:
    catalog = DatasetCatalog(providers={"fake": _CountingFakeProvider()})
    with pytest.raises(OfflineCacheMiss, match="never cached"):
        await catalog.search("x", provider="fake", offline=True)


@pytest.mark.asyncio
async def test_resolve_offline_always_raises_without_calling_provider() -> None:
    catalog = DatasetCatalog(providers={"fake": _CountingFakeProvider()})
    with pytest.raises(OfflineCacheMiss, match="never cached"):
        await catalog.resolve(_fake_ref(), offline=True)


@pytest.mark.asyncio
async def test_iter_records_offline_raises_at_call_time() -> None:
    catalog = DatasetCatalog(providers={"fake": _CountingFakeProvider()})
    ref = _fake_ref()
    dataset = await catalog.resolve(ref)
    # ``iter_records`` is a regular (non-async) method that just hands back
    # an async iterator -- it isn't itself a coroutine that gets awaited. So
    # the offline error must be raised immediately when ``iter_records`` is
    # called, not later when the caller starts pulling items out of the
    # iterator (its first ``__anext__``). We can tell the check happens at
    # the right time because the fake provider's real ``iter_records`` body
    # raises ``NotImplementedError`` instead -- if the offline check were
    # accidentally skipped, we'd see that error instead of the one we
    # actually expect.
    with pytest.raises(OfflineCacheMiss, match="not cache-backed"):
        catalog.iter_records(ref, dataset, offline=True)


@pytest.mark.asyncio
async def test_offline_with_unknown_provider_reports_missing_provider() -> None:
    # Checking whether the provider name is valid happens before checking
    # the offline rules: a typo'd provider name must be reported as an
    # unknown provider, not confusingly blamed on the offline restriction.
    catalog = DatasetCatalog(providers={})
    with pytest.raises(KeyError, match="provider 'missing'"):
        await catalog.search("x", provider="missing", offline=True)


# --- list_presets --------------------------------------------------------------


def test_list_presets_returns_all_builtins() -> None:
    catalog = DatasetCatalog(providers={})
    names = {preset.name for preset in catalog.list_presets()}
    assert names == {"gsm8k", "swe-bench-verified"}


# --- providers that don't need the network are exempt from offline checks (ADR-0010) ---


class _NetworkFreeFakeProvider(_CountingFakeProvider):
    """A fake provider that declares it doesn't need the network, unlike its
    parent class.

    It reuses all of ``_CountingFakeProvider``'s behavior (which is already
    in-memory and doesn't touch the network in practice), but adds a
    ``requires_network = False`` flag. That flag tells ``DatasetCatalog`` to
    treat it the way it treats the real ``LocalDatasetProvider``: since it
    never needs the network anyway, none of its methods are blocked when
    ``offline=True``.
    """

    requires_network = False


@pytest.mark.asyncio
async def test_search_offline_succeeds_for_a_network_free_provider() -> None:
    provider = _NetworkFreeFakeProvider()
    catalog = DatasetCatalog(providers={"fake": provider})
    page = await catalog.search("x", provider="fake", offline=True)
    assert page.total_hits == 0


@pytest.mark.asyncio
async def test_resolve_offline_succeeds_for_a_network_free_provider() -> None:
    provider = _NetworkFreeFakeProvider()
    catalog = DatasetCatalog(providers={"fake": provider})
    resolved = await catalog.resolve(_fake_ref(), offline=True)
    assert resolved.dataset_id == "fake/dataset"


@pytest.mark.asyncio
async def test_iter_records_offline_does_not_raise_for_a_network_free_provider() -> None:
    """Confirms the offline rejection is skipped entirely for a
    network-free provider. The parent class's ``iter_records`` just raises
    ``NotImplementedError`` as a placeholder, which would make it impossible
    to tell "the call correctly reached the provider" apart from "the call
    was blocked" -- both would look like a failure. So this test defines a
    version of ``iter_records`` that actually returns data, to prove the
    call really does reach the provider and succeed, rather than merely
    failing to raise the offline error for some unrelated reason."""

    class _IterableNetworkFreeProvider(_NetworkFreeFakeProvider):
        def iter_records(
            self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
        ) -> AsyncIterator[SourceRecord]:
            async def _gen() -> AsyncIterator[SourceRecord]:
                yield SourceRecord(row_id="0", data={"value": 0}, digest="sha256:0")

            return _gen()

    provider = _IterableNetworkFreeProvider()
    catalog = DatasetCatalog(providers={"fake": provider})
    dataset = await catalog.resolve(_fake_ref(), offline=True)
    records = [record async for record in catalog.iter_records(_fake_ref(), dataset, offline=True)]
    assert len(records) == 1


@pytest.mark.asyncio
async def test_provider_without_requires_network_attribute_still_rejects_offline() -> None:
    """Providers written before ADR-0010 introduced this flag (like
    ``_CountingFakeProvider`` itself, which doesn't declare
    ``requires_network`` at all) must keep behaving exactly as they always
    did: when the catalog looks for the flag and doesn't find it, it must
    default to ``True`` (assume the network IS required) as the safe
    choice, so ``offline=True`` still correctly rejects the call instead of
    silently treating an unmarked provider as network-free."""
    assert not hasattr(_CountingFakeProvider(), "requires_network")
    catalog = DatasetCatalog(providers={"fake": _CountingFakeProvider()})
    with pytest.raises(OfflineCacheMiss):
        await catalog.search("x", provider="fake", offline=True)


# --- the "is this retryable?" flag on offline errors (ADR-0010) --------------


@pytest.mark.asyncio
async def test_search_offline_rejection_is_not_retryable() -> None:
    """A free-text search query has no fixed, reusable cache key at all (the
    same words could match different results later on), so no amount of
    "warming the cache" by running the search online first could ever make
    this exact same offline search succeed afterward."""
    catalog = DatasetCatalog(providers={"fake": _CountingFakeProvider()})
    with pytest.raises(OfflineCacheMiss) as excinfo:
        await catalog.search("x", provider="fake", offline=True)
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_resolve_offline_rejection_is_not_retryable() -> None:
    catalog = DatasetCatalog(providers={"fake": _CountingFakeProvider()})
    with pytest.raises(OfflineCacheMiss) as excinfo:
        await catalog.resolve(_fake_ref(), offline=True)
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_iter_records_offline_rejection_is_not_retryable() -> None:
    catalog = DatasetCatalog(providers={"fake": _CountingFakeProvider()})
    ref = _fake_ref()
    dataset = await catalog.resolve(ref)
    with pytest.raises(OfflineCacheMiss) as excinfo:
        catalog.iter_records(ref, dataset, offline=True)
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_preview_offline_with_no_cache_configured_is_not_retryable() -> None:
    """No cache exists on this catalog at all -- there is nothing to warm."""
    provider = _CountingFakeProvider()
    catalog = DatasetCatalog(providers={"fake": provider})  # no cache=...
    ref = _fake_ref()
    dataset = await catalog.resolve(ref)
    with pytest.raises(OfflineCacheMiss) as excinfo:
        await catalog.preview(ref, dataset, offline=True)
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_preview_offline_cache_miss_is_retryable(tmp_path: Path) -> None:
    """A genuine cache miss on an otherwise-cacheable key IS retryable: one
    online preview of this exact page would populate the cache and let the
    same offline call succeed afterward."""
    provider = _CountingFakeProvider()
    cache = DatasetCache(tmp_path)
    catalog = DatasetCatalog(providers={"fake": provider}, cache=cache)
    ref = _fake_ref()
    dataset = await catalog.resolve(ref)
    with pytest.raises(OfflineCacheMiss) as excinfo:
        await catalog.preview(ref, dataset, offset=5, limit=5, offline=True)
    assert excinfo.value.retryable is True


# --- long search queries: shortened in structured error details, not in the message (ADR-0010) ---


@pytest.mark.asyncio
async def test_search_offline_rejection_truncates_long_query_in_context_only() -> None:
    """The machine-readable ``context["query"]`` value is cut off at a
    length limit (so a huge query can't bloat structured logs or error
    data), but the plain-English error message keeps the query in full,
    since a person reading the message directly still needs to see all of
    it."""
    long_query = "q" * 5000
    catalog = DatasetCatalog(providers={"fake": _CountingFakeProvider()})
    with pytest.raises(OfflineCacheMiss) as excinfo:
        await catalog.search(long_query, provider="fake", offline=True)
    context_query = excinfo.value.context["query"]
    assert isinstance(context_query, str)
    assert len(context_query) < len(long_query)
    assert context_query.endswith(f"...(truncated, {len(long_query)} chars total)")
    # The message itself keeps the full, untruncated query text.
    assert long_query in excinfo.value.message


@pytest.mark.asyncio
async def test_search_offline_rejection_leaves_short_query_untouched() -> None:
    catalog = DatasetCatalog(providers={"fake": _CountingFakeProvider()})
    with pytest.raises(OfflineCacheMiss) as excinfo:
        await catalog.search("short query", provider="fake", offline=True)
    assert excinfo.value.context["query"] == "short query"


# --- resolution_cache: offline resolve after a prior online resolve (ADR-0011) --


@pytest.mark.asyncio
async def test_resolve_offline_succeeds_via_resolution_cache_after_online_resolve(
    tmp_path: Path,
) -> None:
    provider = _CountingFakeProvider()
    resolution_cache = ResolutionCache(tmp_path)
    catalog = DatasetCatalog(providers={"fake": provider}, resolution_cache=resolution_cache)
    ref = _fake_ref()

    online = await catalog.resolve(ref)
    assert provider.resolve_calls == 1

    offline = await catalog.resolve(ref, offline=True)
    assert offline == online
    # The offline resolve must be served entirely from the resolution cache:
    # the provider's own resolve() is never called a second time.
    assert provider.resolve_calls == 1


@pytest.mark.asyncio
async def test_resolve_offline_without_prior_online_resolve_is_retryable_when_configured(
    tmp_path: Path,
) -> None:
    """Unlike the case with no ``resolution_cache`` configured at all
    (``retryable=False``, covered by
    ``test_resolve_offline_rejection_is_not_retryable`` above), here a
    resolution cache IS configured, just not warmed up yet. That means
    running this exact same call online once first would make the offline
    call succeed afterward -- this is the genuine "just warm the cache"
    case, so ``retryable`` must be ``True``."""
    catalog = DatasetCatalog(
        providers={"fake": _CountingFakeProvider()}, resolution_cache=ResolutionCache(tmp_path)
    )
    with pytest.raises(OfflineCacheMiss) as excinfo:
        await catalog.resolve(_fake_ref(), offline=True)
    assert excinfo.value.retryable is True


@pytest.mark.asyncio
async def test_resolve_offline_via_resolution_cache_still_rejects_a_different_dataset(
    tmp_path: Path,
) -> None:
    """A resolution cache warmed for one dataset must not leak into serving
    an offline resolve for a different, never-resolved dataset."""
    provider = _CountingFakeProvider()
    resolution_cache = ResolutionCache(tmp_path)
    catalog = DatasetCatalog(providers={"fake": provider}, resolution_cache=resolution_cache)
    await catalog.resolve(_fake_ref())

    other_ref = DatasetRef(provider="fake", dataset_id="other/dataset", config="main", split="test")
    with pytest.raises(OfflineCacheMiss):
        await catalog.resolve(other_ref, offline=True)
    assert provider.resolve_calls == 1


class _RevisionEchoingFakeProvider(_CountingFakeProvider):
    """A fake whose ``resolve`` copies the *requested* ``ref.revision``
    straight into the resolved dataset it returns, so a test can check
    exactly which requested version ("pinned revision") ends up served
    back.

    ``ResolvedDataset.revision`` is required and can never be null: even a
    request that didn't ask for a specific revision (``ref.revision is
    None``) must still resolve to some concrete version -- in real life,
    "whatever the latest one happens to be right now". This fake models
    that case with a fixed placeholder value instead of a real "latest"
    lookup.
    """

    _LATEST_SENTINEL = "sha256:" + "0" * 64

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        self.resolve_calls += 1
        return ResolvedDataset(
            dataset_id=ref.dataset_id,
            revision=ref.revision if ref.revision is not None else self._LATEST_SENTINEL,
            config=ref.config,
            split=ref.split,
        )


@pytest.mark.asyncio
async def test_offline_resolve_of_a_pinned_revision_never_returns_a_different_pin(
    tmp_path: Path,
) -> None:
    """CRITICAL regression test (bug found and fixed 2026-07-09): resolving
    ``ref`` pinned to revision A online, and then the same ``ref`` pinned to
    revision B online (same provider/dataset/config/split -- just a
    different exact version requested), must cache BOTH resolutions
    separately. An offline resolve of revision A afterward must return
    revision A's data -- it must never be silently handed revision B's data
    instead.

    The bug: the old cache key didn't include the revision at all, so
    resolving two different pinned versions of the same dataset made the
    second online resolve silently overwrite the first one's cache entry.
    An offline resolve for revision A would then quietly come back with
    revision B's data instead -- wrong data, with no error raised to warn
    anyone."""
    provider = _RevisionEchoingFakeProvider()
    catalog = DatasetCatalog(
        providers={"fake": provider}, resolution_cache=ResolutionCache(tmp_path)
    )
    ref_a = DatasetRef(provider="fake", dataset_id="fake/dataset", revision="revA")
    ref_b = DatasetRef(provider="fake", dataset_id="fake/dataset", revision="revB")

    resolved_a = await catalog.resolve(ref_a)
    resolved_b = await catalog.resolve(ref_b)
    assert resolved_a.revision == "revA"
    assert resolved_b.revision == "revB"
    assert provider.resolve_calls == 2

    # The offline resolve of revA must return revA's resolution, served
    # entirely from cache (no third provider call), never revB's.
    offline_a = await catalog.resolve(ref_a, offline=True)
    assert offline_a.revision == "revA"
    assert offline_a == resolved_a
    assert provider.resolve_calls == 2

    # And revB's offline resolve independently returns revB's, not revA's.
    offline_b = await catalog.resolve(ref_b, offline=True)
    assert offline_b.revision == "revB"
    assert provider.resolve_calls == 2


@pytest.mark.asyncio
async def test_offline_resolve_of_an_uncached_pinned_revision_raises_not_wrong_data(
    tmp_path: Path,
) -> None:
    """Requesting a pinned revision that was never resolved online before
    must raise ``OfflineCacheMiss`` -- it must never instead be quietly
    served a *different* revision's cached data. Raising an error is the
    safe outcome here; silently returning the wrong data would be the
    defect."""
    provider = _RevisionEchoingFakeProvider()
    catalog = DatasetCatalog(
        providers={"fake": provider}, resolution_cache=ResolutionCache(tmp_path)
    )
    # Warm only revA online.
    await catalog.resolve(DatasetRef(provider="fake", dataset_id="fake/dataset", revision="revA"))

    # revB was never resolved: offline must miss, not return revA's data.
    ref_b = DatasetRef(provider="fake", dataset_id="fake/dataset", revision="revB")
    with pytest.raises(OfflineCacheMiss):
        await catalog.resolve(ref_b, offline=True)


@pytest.mark.asyncio
async def test_offline_resolve_of_the_unpinned_request_still_round_trips(
    tmp_path: Path,
) -> None:
    """The common case from before this fix -- a request with no specific
    revision pinned (``revision is None``) -- still works the same way as
    before: an online resolve with no pinned revision gets cached, and an
    offline resolve with no pinned revision returns that cached result."""
    provider = _RevisionEchoingFakeProvider()
    catalog = DatasetCatalog(
        providers={"fake": provider}, resolution_cache=ResolutionCache(tmp_path)
    )
    ref = DatasetRef(provider="fake", dataset_id="fake/dataset")  # revision=None
    online = await catalog.resolve(ref)
    offline = await catalog.resolve(ref, offline=True)
    assert offline == online
    assert provider.resolve_calls == 1


@pytest.mark.asyncio
async def test_resolve_offline_still_raises_not_retryable_without_resolution_cache(
    tmp_path: Path,
) -> None:
    """Backward compatibility check: a catalog configured with a page
    ``cache`` but no ``resolution_cache`` must keep behaving exactly as it
    did under ADR-0010, before the resolution cache existed -- the page
    cache and the resolution cache are two separate features that must each
    be turned on independently."""
    provider = _CountingFakeProvider()
    catalog = DatasetCatalog(providers={"fake": provider}, cache=DatasetCache(tmp_path))
    await catalog.resolve(_fake_ref())
    with pytest.raises(OfflineCacheMiss) as excinfo:
        await catalog.resolve(_fake_ref(), offline=True)
    assert excinfo.value.retryable is False


# --- page cache: offline iter_records() succeeds if the page was already
# previewed online (ADR-0011) ---


@pytest.mark.asyncio
async def test_iter_records_offline_succeeds_via_page_cache_after_online_preview(
    tmp_path: Path,
) -> None:
    provider = _CountingFakeProvider()
    cache = DatasetCache(tmp_path)
    catalog = DatasetCatalog(providers={"fake": provider}, cache=cache)
    ref = _fake_ref()
    dataset = await catalog.resolve(ref)

    warmed_page = await catalog.preview(ref, dataset, offset=0, limit=5)
    assert provider.preview_calls == 1

    # `_CountingFakeProvider.iter_records` unconditionally raises
    # NotImplementedError, so if the cache lookup below silently fell
    # through to the provider instead of serving the warmed page, this
    # would fail with that error rather than returning records.
    records = [
        record
        async for record in catalog.iter_records(ref, dataset, offset=0, limit=5, offline=True)
    ]
    assert [record.row_id for record in records] == [r.row_id for r in warmed_page.records]
    assert provider.preview_calls == 1


@pytest.mark.asyncio
async def test_iter_records_offline_page_cache_miss_is_retryable(tmp_path: Path) -> None:
    """A configured page cache with no matching (offset, limit) entry yet is
    the genuine "warm it once online" case, unlike the no-cache-at-all case."""
    provider = _CountingFakeProvider()
    cache = DatasetCache(tmp_path)
    catalog = DatasetCatalog(providers={"fake": provider}, cache=cache)
    ref = _fake_ref()
    dataset = await catalog.resolve(ref)
    await catalog.preview(ref, dataset, offset=0, limit=5)

    # A *different* (offset, limit) than what was warmed above. The
    # rejection must surface synchronously at the call (matching
    # ``preview``/``resolve``), not only once iteration begins.
    with pytest.raises(OfflineCacheMiss) as excinfo:
        catalog.iter_records(ref, dataset, offset=5, limit=5, offline=True)
    assert excinfo.value.retryable is True
    assert "not cache-backed" in str(excinfo.value)


@pytest.mark.asyncio
async def test_iter_records_offline_with_no_limit_is_not_retryable_even_with_cache(
    tmp_path: Path,
) -> None:
    """Unbounded iteration (``limit=None``) has no single-page cache key, so
    it must keep raising ``retryable=False`` even when a page cache is
    configured and warmed for *some* bounded page of the same dataset."""
    provider = _CountingFakeProvider()
    cache = DatasetCache(tmp_path)
    catalog = DatasetCatalog(providers={"fake": provider}, cache=cache)
    ref = _fake_ref()
    dataset = await catalog.resolve(ref)
    await catalog.preview(ref, dataset, offset=0, limit=5)

    with pytest.raises(OfflineCacheMiss) as excinfo:
        catalog.iter_records(ref, dataset, offline=True)
    assert excinfo.value.retryable is False
