"""Tests for the dataset catalog, built-in presets, and cache decoration.

Covers built-in preset field pinning, provider routing (including the
explicit-KeyError-on-unknown-provider requirement), cache-hit/cache-miss
decoration around ``preview``, ``offline=True`` never calling a provider, and
plugin-vs-built-in provider name collisions. The first two tests reproduce
the plan's verbatim Task 7 Step 1 snippet
(docs/plans/2026-07-02-agentic-evalkit-initial-release.md) unmodified.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest

from agentic_evalkit.datasets.base import ProviderHealth
from agentic_evalkit.datasets.cache import DatasetCache
from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.presets import BUILTIN_PRESETS
from agentic_evalkit.datasets.resolution_cache import ResolutionCache
from agentic_evalkit.errors import OfflineCacheMiss, PluginCompatibilityError
from agentic_evalkit.models import (
    DatasetRef,
    ResolvedDataset,
    SamplePage,
    SearchPage,
    SourceRecord,
)

# --- Step 1 (plan verbatim): preset and provider-routing tests --------------


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


def test_builtin_presets_are_frozen_and_forbid_unknown_fields() -> None:
    gsm = BUILTIN_PRESETS["gsm8k"]
    with pytest.raises(Exception):  # noqa: B017 - pydantic.ValidationError, frozen instance
        gsm.name = "renamed"  # type: ignore[misc]


def test_builtin_presets_mapping_is_immutable() -> None:
    with pytest.raises(TypeError):
        BUILTIN_PRESETS["new-preset"] = BUILTIN_PRESETS["gsm8k"]  # type: ignore[index]


# --- Fake provider used across catalog tests ---------------------------------


class _CountingFakeProvider:
    """A minimal ``DatasetProvider`` that counts ``preview`` invocations.

    Used to prove cache decoration: an identical ``preview`` call must not
    increase ``preview_calls``, while a call with different pagination
    parameters must.
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


# --- offline rejection on non-cacheable operations ----------------------------


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
    # ``iter_records`` is a plain method returning the provider's iterator, so
    # the offline rejection must surface synchronously at the call, not on the
    # first ``__anext__``. The fake provider's ``iter_records`` raises
    # ``NotImplementedError``, so reaching the provider would fail differently.
    with pytest.raises(OfflineCacheMiss, match="not cache-backed"):
        catalog.iter_records(ref, dataset, offline=True)


@pytest.mark.asyncio
async def test_offline_with_unknown_provider_reports_missing_provider() -> None:
    # Provider validation precedes the offline rejection: a typo'd provider
    # must not be misreported as an offline limitation.
    catalog = DatasetCatalog(providers={})
    with pytest.raises(KeyError, match="provider 'missing'"):
        await catalog.search("x", provider="missing", offline=True)


# --- list_presets --------------------------------------------------------------


def test_list_presets_returns_all_builtins() -> None:
    catalog = DatasetCatalog(providers={})
    names = {preset.name for preset in catalog.list_presets()}
    assert names == {"gsm8k", "swe-bench-verified"}


# --- requires_network provider exemption (ADR-0010) --------------------------


class _NetworkFreeFakeProvider(_CountingFakeProvider):
    """A fake that declares network independence, unlike its parent class.

    Reuses ``_CountingFakeProvider``'s bodies (in-memory, already network-free
    in practice) but adds the ``requires_network = False`` declaration so
    ``DatasetCatalog`` treats it the way it treats the real
    ``LocalDatasetProvider``: exempt from offline rejection on every method.
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
    """Confirms the offline rejection is skipped entirely for such a provider
    -- reaching the fake's real (non-``NotImplementedError``-raising)
    ``iter_records`` body would still fail if this test's fixture provider
    did not override it, so the parent's ``NotImplementedError`` stand-in is
    replaced here with a working body to prove the call actually proceeds."""

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
    """Pre-ADR-0010 fakes (like ``_CountingFakeProvider`` itself, which
    declares no ``requires_network`` at all) must keep today's behavior: the
    safe getattr-default is ``True`` (network-required), so offline is still
    rejected rather than silently becoming exempt."""
    assert not hasattr(_CountingFakeProvider(), "requires_network")
    catalog = DatasetCatalog(providers={"fake": _CountingFakeProvider()})
    with pytest.raises(OfflineCacheMiss):
        await catalog.search("x", provider="fake", offline=True)


# --- retryable discriminator values (ADR-0010) --------------------------------


@pytest.mark.asyncio
async def test_search_offline_rejection_is_not_retryable() -> None:
    """A query-keyed search has no stable cache key at all -- no amount of
    warming ever makes the exact same offline call succeed."""
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


# --- unbounded query truncation in error context (ADR-0010) ------------------


@pytest.mark.asyncio
async def test_search_offline_rejection_truncates_long_query_in_context_only() -> None:
    """The structured ``context["query"]`` value is bounded; the free-text
    error message is left untruncated for a human reading it directly."""
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
    """Unlike the no-``resolution_cache`` case (``retryable=False``, covered by
    ``test_resolve_offline_rejection_is_not_retryable`` above), a configured
    but not-yet-warmed resolution cache means the exact same offline call
    *would* succeed after one online resolve -- the genuine "warm the cache"
    case."""
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
    """A fake whose ``resolve`` echoes the *requested* ``ref.revision`` into
    the resolved dataset, so a test can tell which pinned revision was served.

    ``ResolvedDataset.revision`` is a required, non-null field: an unpinned
    (``ref.revision is None``) request still resolves to a concrete revision
    ("latest at resolution time"), modeled here by a fixed sentinel.
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
    """CRITICAL regression (2026-07-09): resolving ``ref@revA`` online then
    ``ref@revB`` online (same provider/dataset/config/split, different
    pinned revision) must cache BOTH; an offline ``resolve(ref@revA)`` must
    return ``revA``'s resolution, never be silently served ``revB``'s. The
    revision-blind key this fix replaced collapsed both pins into one slot,
    so the second online resolve overwrote the first and the offline resolve
    returned the wrong pinned revision with no error."""
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
    """A pinned revision that was never resolved online must raise
    ``OfflineCacheMiss`` rather than being served a *different* revision's
    cached resolution -- a miss is safe; wrong data is the defect."""
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
    """The pre-fix common path (``revision is None``) is unchanged: an
    unpinned online resolve is cached and an unpinned offline resolve
    returns it."""
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
    """Backward compatibility: a catalog that opts into a page ``cache`` but
    not a ``resolution_cache`` keeps ADR-0010's original resolve() behavior
    unchanged -- the two caches are independent opt-ins."""
    provider = _CountingFakeProvider()
    catalog = DatasetCatalog(providers={"fake": provider}, cache=DatasetCache(tmp_path))
    await catalog.resolve(_fake_ref())
    with pytest.raises(OfflineCacheMiss) as excinfo:
        await catalog.resolve(_fake_ref(), offline=True)
    assert excinfo.value.retryable is False


# --- cache: offline iter_records after a prior online preview (ADR-0011) --------


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
