"""Hugging Face discovery and Dataset Viewer provider (design §6.1-§6.2).

This provider never imports ``datasets`` or ``pyarrow`` and never sets
``trust_remote_code=True``. Discovery and immutable revision metadata come
from ``huggingface_hub.HfApi`` (injected, so sync Hub calls run through
``asyncio.to_thread``); row access, validity, schema, size, statistics, and
Parquet-file metadata come from the Dataset Viewer HTTP API
(``https://datasets-server.huggingface.co``) through an injected
``httpx.AsyncClient``.

``resolve()`` treats ``/is-valid``, ``dataset_info``, and ``/splits`` as
load-bearing: a failure on any of them fails the whole resolution with a
typed error. ``/size``, ``/statistics``, and ``/parquet`` are best-effort:
many valid datasets legitimately lack statistics or a Parquet conversion, so
a failure on any of them is recorded as an absence in
``ResolvedDataset.schema_metadata`` (``"<name>_available": False``) and
resolution continues.

See ``docs/plans/agent-prompts/task6-hf.md`` and design §6.2 for the
contract this module implements.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Protocol, cast, runtime_checkable

import httpx
from huggingface_hub import HfApi

from agentic_evalkit.datasets.base import ProviderHealth
from agentic_evalkit.errors import (
    AgenticEvalkitError,
    DatasetAccessDenied,
    DatasetConfigRequired,
    DatasetNotFound,
    DatasetProviderUnavailable,
    DatasetRateLimited,
    SecretValue,
)
from agentic_evalkit.errors import JsonValue as ErrorContextValue
from agentic_evalkit.models import (
    DatasetRef,
    ResolvedDataset,
    SamplePage,
    SearchHit,
    SearchPage,
    SourceRecord,
)

if TYPE_CHECKING:
    from types import TracebackType

    from pydantic import JsonValue as ModelJsonValue

# Error ``context=`` dicts use ``errors.JsonValue`` (the stdlib-only type the
# error hierarchy is typed against); model fields (e.g.
# ``ResolvedDataset.schema_metadata``) use pydantic's richer ``JsonValue``.
# The two are structurally similar but not the same type, so this module
# keeps them distinctly named rather than aliasing one to the other.
# ``AgenticEvalkitError.__init__`` types ``context`` as
# ``dict[str, JsonValue | SecretValue] | None``; matching that union exactly
# (rather than just ``JsonValue``) satisfies mypy's dict invariance check.
ErrorContext = dict[str, ErrorContextValue | SecretValue]

_VIEWER_BASE_URL: Final[str] = "https://datasets-server.huggingface.co"
_HEALTHCHECK_DATASET: Final[str] = "openai/gsm8k"
_MAX_PAGE_LIMIT: Final[int] = 100
_DEFAULT_MAX_RETRIES: Final[int] = 3
_DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
_HEALTHCHECK_TIMEOUT_SECONDS: Final[float] = 5.0
_RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 502, 503, 504})
_BACKOFF_BASE_SECONDS: Final[float] = 0.5

SleepFn = Callable[[float], Awaitable[None]]


async def _default_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _canonical_digest(row: dict[str, ModelJsonValue]) -> str:
    """SHA-256 of the canonical (sorted-key, compact) JSON of one row.

    Matches the convention used by ``agentic_evalkit.datasets.local`` so a
    given row's digest does not depend on which provider produced it.
    """
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _response_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@runtime_checkable
class _HubClient(Protocol):
    """The subset of ``huggingface_hub.HfApi`` this provider calls.

    A structural protocol so tests can inject a lightweight ``_FakeHub``
    instead of a real ``HfApi`` (and so this provider never assumes more of
    the Hub client than it actually uses).
    """

    def dataset_info(self, repo_id: str, *, revision: str | None = None, **kwargs: Any) -> Any: ...

    def list_datasets(self, **kwargs: Any) -> Any: ...


def _viewer_url(path: str) -> str:
    return f"{_VIEWER_BASE_URL}/{path}"


_KNOWN_VALIDITY_CAPABILITIES: Final[tuple[str, ...]] = (
    "preview",
    "viewer",
    "search",
    "filter",
    "statistics",
)


def _reports_usable_capability(payload: dict[str, Any]) -> bool:
    """Whether an ``/is-valid`` payload signals at least one usable capability.

    A successful (2xx) ``/is-valid`` response is treated as usable unless it
    explicitly reports every known capability as false; an unrecognized or
    empty payload shape is not itself proof of invalidity, only an explicit
    "nothing works" signal is.
    """
    reported = [payload[key] for key in _KNOWN_VALIDITY_CAPABILITIES if key in payload]
    if not reported:
        return True
    return any(bool(value) for value in reported)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        return None


def _jittered_backoff_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter, zero-indexed by retry attempt."""
    ceiling = _BACKOFF_BASE_SECONDS * (2**attempt)
    return random.uniform(0.0, ceiling)  # noqa: S311 -- backoff jitter, not security-sensitive


def _raise_for_load_bearing_status(
    response: httpx.Response, *, endpoint: str, dataset_id: str
) -> None:
    """Classify a final (non-retried, or retries-exhausted) response status.

    Only called for load-bearing endpoints (``/is-valid``, ``/splits``) and
    for the Dataset Viewer request path in general; best-effort endpoints use
    :func:`_is_success` and swallow non-2xx responses instead of calling
    this.
    """
    status = response.status_code
    context: ErrorContext = {
        "endpoint": endpoint,
        "dataset_id": dataset_id,
        "status_code": status,
    }
    if status in (401, 403):
        raise DatasetAccessDenied(
            message=f"access denied calling {endpoint} for {dataset_id} (HTTP {status})",
            context=context,
        )
    if status == 404:
        raise DatasetNotFound(
            message=f"{dataset_id} not found calling {endpoint} (HTTP 404)", context=context
        )
    if status == 422:
        raise DatasetConfigRequired(
            message=f"{endpoint} rejected the requested config/split for {dataset_id} (HTTP 422)",
            context=context,
        )
    if status == 429:
        retry_after = _retry_after_seconds(response)
        raise DatasetRateLimited(
            message=f"rate limited calling {endpoint} for {dataset_id} (HTTP 429)",
            context={**context, "retry_after_seconds": retry_after},
        )
    if not response.is_success:
        raise DatasetProviderUnavailable(
            message=f"{endpoint} returned HTTP {status} for {dataset_id}",
            context=context,
        )


@dataclass(frozen=True, slots=True)
class _DatasetInfoSummary:
    """Every field ``resolve()`` needs from one Hub ``dataset_info`` call."""

    revision: str
    license: str | None
    citation: str | None
    card_metadata: dict[str, ModelJsonValue]
    gated: bool


class HuggingFaceDatasetProvider:
    """Dataset provider backed by the Hugging Face Hub and Dataset Viewer.

    Construct directly with an injected ``client``/``hub`` for tests, or use
    :meth:`create` for an async context manager that owns its own
    ``httpx.AsyncClient`` (the normal way to use this provider outside
    tests).
    """

    api_version: Final[str] = "1"
    #: Every method here calls the Hub or the Dataset Viewer HTTP API
    #: (ADR-0010); this provider can never honestly satisfy ``offline=True``
    #: itself, so ``DatasetCatalog`` continues to reject offline calls to it
    #: (except ``preview``, already served from the content-addressed cache
    #: when a cache is configured).
    requires_network: Final[bool] = True

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        hub: _HubClient,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        sleep: SleepFn = _default_sleep,
    ) -> None:
        self._client = client
        self._hub = hub
        self._max_retries = max_retries
        self._sleep = sleep

    @classmethod
    def create(
        cls,
        *,
        hub: _HubClient | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        sleep: SleepFn = _default_sleep,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> _OwnedProviderContext:
        """Return an async context manager owning a fresh ``AsyncClient``.

        Usage: ``async with HuggingFaceDatasetProvider.create() as provider:``.
        The client is closed on ``__aexit__``. ``hub`` defaults to a fresh
        ``HfApi()`` using default Hugging Face authentication (public access
        needs no token; ``HF_TOKEN``/the standard credential store are
        honored automatically for private or gated data).
        """
        # HfApi's real methods use enumerated keyword-only parameters rather
        # than a **kwargs catch-all, so mypy cannot statically prove it
        # satisfies _HubClient's looser **kwargs signature even though every
        # call this provider makes only uses the specific keywords HfApi
        # actually accepts (see _HubClient's docstring). This is verified at
        # runtime via @runtime_checkable in the test suite
        # (test_create_returns_context_manager_owning_its_client and direct
        # isinstance checks against a real HfApi()).
        resolved_hub: _HubClient = hub if hub is not None else cast("_HubClient", HfApi())
        return _OwnedProviderContext(
            hub=resolved_hub,
            max_retries=max_retries,
            sleep=sleep,
            timeout_seconds=timeout_seconds,
        )

    # --- search() -------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchPage:
        """Map ``HfApi.list_datasets`` results to a ``SearchPage``.

        Hugging Face Hub search does not expose a stable opaque cursor for
        this endpoint, so this provider fetches at most ``limit`` hits per
        call and always returns ``cursor=None``; a caller wanting more
        results reissues ``search`` with a larger ``limit``.
        """
        kwargs: dict[str, Any] = {"search": query, "limit": limit}
        if filters:
            kwargs.update(filters)

        def _call() -> list[Any]:
            return list(self._hub.list_datasets(**kwargs))

        try:
            results = await asyncio.to_thread(_call)
        except Exception as error:
            raise DatasetProviderUnavailable(
                message=f"Hub search failed for query {query!r}: {type(error).__name__}: {error}",
                context={"query": query},
            ) from error

        hits = tuple(_search_hit_from_hub_result(result) for result in results)
        return SearchPage(hits=hits, cursor=None, total_hits=len(hits))

    # --- resolve() --------------------------------------------------------

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        """Immutably resolve ``ref`` per plan Step 3 / design §6.2.

        Load-bearing (failure fails resolution): ``/is-valid``,
        ``dataset_info`` (requiring a nonempty commit SHA), ``/splits``.
        Best-effort (failure is recorded in ``schema_metadata`` and
        resolution continues): ``/size``, ``/statistics``, ``/parquet``.
        """
        await self._check_is_valid(ref.dataset_id)
        info = await self._fetch_dataset_info(ref)
        splits = await self._fetch_splits(ref.dataset_id)
        config, split = _select_config_and_split(ref, splits)

        schema_metadata: dict[str, ModelJsonValue] = {}
        provider_response_digests: dict[str, str] = {}

        row_count = await self._best_effort_size(
            ref.dataset_id,
            config,
            split,
            schema_metadata=schema_metadata,
            digests=provider_response_digests,
        )
        await self._best_effort_statistics(
            ref.dataset_id,
            config,
            split,
            schema_metadata=schema_metadata,
            digests=provider_response_digests,
        )
        selected_files = await self._best_effort_parquet(
            ref.dataset_id,
            config,
            split,
            schema_metadata=schema_metadata,
            digests=provider_response_digests,
        )

        return ResolvedDataset(
            dataset_id=ref.dataset_id,
            revision=info.revision,
            config=config,
            split=split,
            selected_files=selected_files,
            schema_metadata=schema_metadata,
            row_count=row_count,
            license=info.license,
            citation=info.citation,
            gated=info.gated,
            card_metadata=info.card_metadata,
            retrieved_at=datetime.now(UTC),
            provider_response_digests=provider_response_digests,
        )

    async def _check_is_valid(self, dataset_id: str) -> None:
        response = await self._get(
            "is-valid", {"dataset": dataset_id}, endpoint="is-valid", dataset_id=dataset_id
        )
        payload = response.json()
        if not _reports_usable_capability(payload):
            raise DatasetProviderUnavailable(
                message=f"Dataset Viewer reports no usable capability for {dataset_id}",
                context={"dataset_id": dataset_id, "is_valid_response": payload},
            )

    async def _fetch_dataset_info(self, ref: DatasetRef) -> _DatasetInfoSummary:
        """Call Hub ``dataset_info`` exactly once and extract every field ``resolve()`` needs.

        ``dataset_info`` is a single synchronous Hub call run through
        ``asyncio.to_thread``; both the load-bearing commit SHA and the
        best-effort license/citation/card/gated metadata come from this one
        result rather than issuing the call twice.
        """

        def _call() -> Any:
            return self._hub.dataset_info(ref.dataset_id, revision=ref.revision)

        try:
            info = await asyncio.to_thread(_call)
        except AgenticEvalkitError:
            raise
        except Exception as error:
            raise DatasetProviderUnavailable(
                message=(
                    f"Hub dataset_info failed for {ref.dataset_id}: {type(error).__name__}: {error}"
                ),
                context={"dataset_id": ref.dataset_id},
            ) from error

        sha = getattr(info, "sha", None)
        if not sha:
            raise DatasetProviderUnavailable(
                message=f"Hub dataset_info returned an empty commit SHA for {ref.dataset_id}",
                context={"dataset_id": ref.dataset_id},
            )

        card_dict = _card_data_to_dict(getattr(info, "card_data", None))
        return _DatasetInfoSummary(
            revision=str(sha),
            license=_first_str(card_dict.get("license")),
            citation=_first_str(card_dict.get("citation")),
            card_metadata=card_dict,
            gated=bool(getattr(info, "gated", False)),
        )

    async def _fetch_splits(self, dataset_id: str) -> list[dict[str, Any]]:
        response = await self._get(
            "splits", {"dataset": dataset_id}, endpoint="splits", dataset_id=dataset_id
        )
        payload = response.json()
        splits = payload.get("splits")
        if not isinstance(splits, list):
            raise DatasetProviderUnavailable(
                message=f"/splits returned an unexpected shape for {dataset_id}",
                context={"dataset_id": dataset_id},
            )
        return splits

    async def _best_effort_size(
        self,
        dataset_id: str,
        config: str,
        split: str,
        *,
        schema_metadata: dict[str, ModelJsonValue],
        digests: dict[str, str],
    ) -> int | None:
        try:
            response = await self._get(
                "size",
                {"dataset": dataset_id, "config": config, "split": split},
                endpoint="size",
                dataset_id=dataset_id,
            )
        except AgenticEvalkitError:
            schema_metadata["size_available"] = False
            return None

        digests["size"] = _response_digest(response.content)
        payload = response.json()
        for split_entry in payload.get("size", {}).get("splits", []):
            if split_entry.get("config") == config and split_entry.get("split") == split:
                schema_metadata["size_available"] = True
                num_rows = split_entry.get("num_rows")
                return int(num_rows) if isinstance(num_rows, int) else None
        schema_metadata["size_available"] = False
        return None

    async def _best_effort_statistics(
        self,
        dataset_id: str,
        config: str,
        split: str,
        *,
        schema_metadata: dict[str, ModelJsonValue],
        digests: dict[str, str],
    ) -> None:
        try:
            response = await self._get(
                "statistics",
                {"dataset": dataset_id, "config": config, "split": split},
                endpoint="statistics",
                dataset_id=dataset_id,
            )
        except AgenticEvalkitError:
            schema_metadata["statistics_available"] = False
            return

        digests["statistics"] = _response_digest(response.content)
        payload = response.json()
        schema_metadata["statistics_available"] = True
        schema_metadata["statistics_num_examples"] = payload.get("num_examples")

    async def _best_effort_parquet(
        self,
        dataset_id: str,
        config: str,
        split: str,
        *,
        schema_metadata: dict[str, ModelJsonValue],
        digests: dict[str, str],
    ) -> tuple[str, ...]:
        try:
            response = await self._get(
                "parquet",
                {"dataset": dataset_id, "config": config, "split": split},
                endpoint="parquet",
                dataset_id=dataset_id,
            )
        except AgenticEvalkitError:
            schema_metadata["parquet_available"] = False
            return ()

        digests["parquet"] = _response_digest(response.content)
        payload = response.json()
        files = payload.get("parquet_files", [])
        selected = tuple(
            str(entry["url"])
            for entry in files
            if entry.get("config") == config and entry.get("split") == split and "url" in entry
        )
        schema_metadata["parquet_available"] = bool(selected)
        return selected

    # --- preview() / iter_records() ---------------------------------------

    async def preview(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int = 10
    ) -> SamplePage:
        """Return one page of rows via ``/rows``, capped at 100 per request.

        A partial page (fewer rows than requested because the upstream total
        was reached) is returned as-is; the caller sees ``total_rows`` and
        can tell a partial page from a full dataset.
        """
        bounded_limit = min(limit, _MAX_PAGE_LIMIT)
        config, split = _require_config_and_split(dataset)
        response = await self._get(
            "rows",
            {
                "dataset": dataset.dataset_id,
                "config": config,
                "split": split,
                "offset": offset,
                "length": bounded_limit,
            },
            endpoint="rows",
            dataset_id=dataset.dataset_id,
        )
        payload = response.json()
        records = tuple(_row_to_source_record(row) for row in payload.get("rows", []))
        total_rows = payload.get("num_rows_total")
        return SamplePage(
            records=records,
            offset=offset,
            total_rows=int(total_rows) if isinstance(total_rows, int) else None,
        )

    async def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        """Yield successive rows until ``limit``, upstream total, or an empty page."""
        remaining = limit
        current_offset = offset
        while remaining is None or remaining > 0:
            page_limit = _MAX_PAGE_LIMIT if remaining is None else min(remaining, _MAX_PAGE_LIMIT)
            page = await self.preview(dataset, offset=current_offset, limit=page_limit)
            if not page.records:
                return
            for record in page.records:
                yield record
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return
            current_offset += len(page.records)
            if page.total_rows is not None and current_offset >= page.total_rows:
                return

    # --- healthcheck() ------------------------------------------------------

    async def healthcheck(self) -> ProviderHealth:
        started = time.monotonic()
        try:
            response = await self._get(
                "is-valid",
                {"dataset": _HEALTHCHECK_DATASET},
                endpoint="is-valid",
                dataset_id=_HEALTHCHECK_DATASET,
                timeout_seconds=_HEALTHCHECK_TIMEOUT_SECONDS,
            )
        except DatasetRateLimited as error:
            return ProviderHealth(
                status="degraded",
                latency_ms=(time.monotonic() - started) * 1000,
                capabilities=("search", "resolve", "preview", "iter_records"),
                error_code=error.code,
            )
        except AgenticEvalkitError as error:
            return ProviderHealth(
                status="error",
                latency_ms=(time.monotonic() - started) * 1000,
                error_code=error.code,
            )

        latency_ms = (time.monotonic() - started) * 1000
        payload = response.json()
        if not _reports_usable_capability(payload):
            return ProviderHealth(
                status="degraded",
                latency_ms=latency_ms,
                capabilities=("search", "resolve", "preview", "iter_records"),
                error_code="dataset_provider_unavailable",
            )
        return ProviderHealth(
            status="ok",
            latency_ms=latency_ms,
            capabilities=("search", "resolve", "preview", "iter_records"),
        )

    # --- shared HTTP + retry machinery --------------------------------------

    async def _get(
        self,
        path: str,
        params: Mapping[str, str | int],
        *,
        endpoint: str,
        dataset_id: str,
        timeout_seconds: float | None = None,
    ) -> httpx.Response:
        """GET a Dataset Viewer endpoint with bounded retries.

        Retries only connection errors and HTTP 429/502/503/504, honoring
        ``Retry-After`` when present and otherwise using jittered exponential
        backoff, up to ``self._max_retries`` retries (so at most
        ``max_retries + 1`` total attempts). Any other status is classified
        immediately by :func:`_raise_for_load_bearing_status` without a
        retry.
        """
        url = _viewer_url(path)
        attempt = 0
        while True:
            try:
                kwargs: dict[str, Any] = {"params": params}
                if timeout_seconds is not None:
                    kwargs["timeout"] = timeout_seconds
                response = await self._client.get(url, **kwargs)
            except httpx.TransportError as error:
                if attempt >= self._max_retries:
                    raise DatasetProviderUnavailable(
                        message=(
                            f"transport error calling {endpoint} for {dataset_id}: "
                            f"{type(error).__name__}: {error}"
                        ),
                        context={"endpoint": endpoint, "dataset_id": dataset_id},
                    ) from error
                await self._sleep(_jittered_backoff_seconds(attempt))
                attempt += 1
                continue

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                delay = _retry_after_seconds(response)
                if delay is None:
                    delay = _jittered_backoff_seconds(attempt)
                await self._sleep(delay)
                attempt += 1
                continue

            if not response.is_success:
                _raise_for_load_bearing_status(response, endpoint=endpoint, dataset_id=dataset_id)

            return response


class _OwnedProviderContext:
    """Async context manager that owns and closes an ``httpx.AsyncClient``."""

    def __init__(
        self,
        *,
        hub: _HubClient,
        max_retries: int,
        sleep: SleepFn,
        timeout_seconds: float,
    ) -> None:
        self._hub = hub
        self._max_retries = max_retries
        self._sleep = sleep
        self._timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._provider: HuggingFaceDatasetProvider | None = None

    async def __aenter__(self) -> HuggingFaceDatasetProvider:
        self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        self._provider = HuggingFaceDatasetProvider(
            client=self._client,
            hub=self._hub,
            max_retries=self._max_retries,
            sleep=self._sleep,
        )
        return self._provider

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()


def _search_hit_from_hub_result(result: Any) -> SearchHit:
    card_data = _card_data_to_dict(getattr(result, "card_data", None))
    return SearchHit(
        dataset_id=str(result.id),
        provider="huggingface",
        revision=getattr(result, "sha", None),
        tags=tuple(getattr(result, "tags", None) or ()),
        gated=bool(getattr(result, "gated", False)),
        private=bool(getattr(result, "private", False)),
        downloads=getattr(result, "downloads", None),
        card_metadata=card_data,
    )


def _card_data_to_dict(card_data: Any) -> dict[str, ModelJsonValue]:
    if card_data is None:
        return {}
    if isinstance(card_data, dict):
        return dict(card_data)
    to_dict = getattr(card_data, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, dict):
            return dict(result)
    return {}


def _first_str(value: ModelJsonValue) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, str):
            return first
    return None


def _select_config_and_split(ref: DatasetRef, splits: list[dict[str, Any]]) -> tuple[str, str]:
    """Validate the ref's config/split against ``/splits``, or uniquely infer them.

    Raises ``DatasetConfigRequired`` when the config/split cannot be
    determined unambiguously (either an explicit value the viewer does not
    know about, or an omitted value with more than one candidate).
    """
    available = tuple((str(entry.get("config")), str(entry.get("split"))) for entry in splits)

    if ref.config is not None and ref.split is not None:
        if (ref.config, ref.split) in available:
            return ref.config, ref.split
        raise DatasetConfigRequired(
            message=(
                f"config={ref.config!r} split={ref.split!r} is not a valid combination for "
                f"{ref.dataset_id}"
            ),
            context={"dataset_id": ref.dataset_id, "available": available},
        )

    candidates: list[tuple[str, str]] = list(available)
    if ref.config is not None:
        candidates = [pair for pair in candidates if pair[0] == ref.config]
    if ref.split is not None:
        candidates = [pair for pair in candidates if pair[1] == ref.split]

    unique_configs = {pair[0] for pair in candidates}
    unique_splits = {pair[1] for pair in candidates}
    if len(candidates) == 1 or (len(unique_configs) == 1 and len(unique_splits) == 1):
        return candidates[0]

    raise DatasetConfigRequired(
        message=(
            f"{ref.dataset_id} has multiple config/split combinations and none could be "
            f"uniquely inferred from config={ref.config!r} split={ref.split!r}"
        ),
        context={"dataset_id": ref.dataset_id, "available": available},
    )


def _require_config_and_split(dataset: ResolvedDataset) -> tuple[str, str]:
    if dataset.config is None or dataset.split is None:
        raise DatasetProviderUnavailable(
            message=f"resolved dataset {dataset.dataset_id} is missing config/split",
            context={"dataset_id": dataset.dataset_id},
        )
    return dataset.config, dataset.split


def _row_to_source_record(row: dict[str, Any]) -> SourceRecord:
    row_idx = row.get("row_idx")
    data = row.get("row", {})
    return SourceRecord(row_id=str(row_idx), data=data, digest=_canonical_digest(data))
