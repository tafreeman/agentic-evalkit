"""Connects this codebase to Hugging Face's dataset ecosystem: dataset
search/discovery, plus the "Dataset Viewer" API (design doc §6.1-§6.2).

Hugging Face Hub is a hosting platform for ML datasets and models (some
public, some private or "gated" -- requiring the owner's approval). The
"Dataset Viewer" is a separate HTTP service Hugging Face runs alongside the
Hub that lets a caller inspect a dataset's rows, schema, size, and
statistics directly over HTTP, without downloading and loading the entire
dataset first. This module talks to both:

- Dataset discovery and *revision* metadata -- i.e. figuring out exactly
  which unchanging version of a dataset we mean, so it can't silently
  change later (see ``resolve()`` below) -- come from
  ``huggingface_hub.HfApi`` (injected as a dependency). ``HfApi``'s calls
  are ordinary blocking/synchronous Python calls, so every one used here
  runs through ``asyncio.to_thread`` (which hands the blocking call to a
  background thread) so it never freezes this module's own async code while
  waiting for a response.
- Row access, validity, schema, size, statistics, and Parquet-file metadata
  come from the Dataset Viewer's HTTP API
  (``https://datasets-server.huggingface.co``), called through an injected
  ``httpx.AsyncClient``.

This module deliberately never imports the heavier ``datasets`` or
``pyarrow`` libraries -- which would let it download and load an entire
dataset into memory -- and never sets Hugging Face's
``trust_remote_code=True`` option, which would let a dataset's own
(potentially untrusted) Python code run locally. This keeps the module a
thin, read-only HTTP client that can never execute code supplied by a
dataset.

``resolve()`` -- the method that pins down exactly which dataset, version,
config, and split to use -- treats three Dataset Viewer calls as
**load-bearing**: ``/is-valid``, ``dataset_info`` (which must return a
non-empty commit SHA -- Hugging Face's unique, Git-style identifier for one
exact version of the dataset), and ``/splits``. "Load-bearing" means if any
one of these three fails, the entire resolution fails with it, raising a
specific, purpose-built ("typed") exception describing what went wrong,
rather than quietly returning a result that might be wrong. Three other
calls -- ``/size``, ``/statistics``, and ``/parquet`` -- are **best-effort**:
plenty of valid, perfectly usable datasets simply don't have statistics or a
Parquet export available, so a failure on any one of these is not treated as
an error at all. Instead it is recorded as an absence in
``ResolvedDataset.schema_metadata`` (as ``"<name>_available": False``, e.g.
``"statistics_available": False``), and resolution continues normally.

See ``docs/plans/agent-prompts/task6-hf.md`` and the design doc's §6.2 for
the full contract this module implements.
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

# This codebase actually has two different "any JSON-shaped value" types,
# and this module has to be careful to use the right one in the right
# place. Error ``context=`` dicts (extra debugging details attached to an
# exception) use ``errors.JsonValue`` -- a type built only from plain
# Python types (str, int, dict, etc.), because the exception-class
# hierarchy itself doesn't depend on pydantic. Model fields, like
# ``ResolvedDataset.schema_metadata``, instead use pydantic's own richer
# ``JsonValue`` type. The two describe similar data but are technically
# different Python types, so this module keeps them distinctly named here
# (``ErrorContextValue`` for the former) rather than treating them as
# interchangeable.
#
# ``AgenticEvalkitError.__init__`` declares its ``context`` parameter as
# exactly ``dict[str, JsonValue | SecretValue] | None``. The type alias
# below matches that exact shape (rather than just reusing ``JsonValue`` on
# its own) because mypy checks generic containers like ``dict[str, X]``
# strictly: even though every plain ``JsonValue`` is also a valid
# ``JsonValue | SecretValue``, mypy will not automatically treat
# ``dict[str, JsonValue]`` as compatible with ``dict[str, JsonValue |
# SecretValue]``. Spelling out the exact same union here is what keeps
# mypy satisfied.
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
    """Compute a SHA-256 fingerprint (hash) of one row's data.

    To get the same fingerprint for the same data every time, we first turn
    the row into "canonical" JSON: keys sorted alphabetically and no extra
    whitespace, so two Python dicts with identical contents always produce
    the exact same JSON string (and therefore the exact same hash), no
    matter what order their keys happened to be in originally. This matches
    the convention used by ``agentic_evalkit.datasets.local``, so the same
    row's fingerprint comes out identical whether it was fetched from
    Hugging Face or read from a local file -- the digest depends only on
    the row's content, never on which provider produced it.
    """
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _response_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@runtime_checkable
class _HubClient(Protocol):
    """Declares only the two ``huggingface_hub.HfApi`` methods this file calls.

    ``Protocol`` is Python's structural-typing tool: instead of requiring a
    formal ``class Foo(HfApi)`` relationship, anything with matching
    methods -- ``dataset_info`` and ``list_datasets`` -- counts as a
    ``_HubClient``, whether or not it's actually related to ``HfApi`` at
    all. That lets tests inject a small, fast, hand-written stand-in
    (``_FakeHub``) instead of a real ``HfApi``, and it keeps this file
    honest about the fact that it only ever needs these two methods, not
    the dozens of others the real ``HfApi`` class provides.
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
    """Decide whether a Hugging Face ``/is-valid`` response says this dataset works at all.

    The Dataset Viewer's ``/is-valid`` endpoint reports true/false for each
    named feature it might support for a dataset -- e.g. ``preview``,
    ``viewer``, ``search``, ``filter``, ``statistics`` (see
    ``_KNOWN_VALIDITY_CAPABILITIES`` below). We treat a successful (2xx)
    response as "this dataset is usable" unless it explicitly marks *every*
    one of those features as false. If the response is missing these
    fields, or has some other shape we don't recognize, that is not itself
    treated as proof the dataset is broken -- only an explicit "nothing
    here works" answer counts as that.
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
    """Pick a random wait time before the next retry ("exponential backoff with full jitter").

    ``attempt`` counts retries from 0 (the first retry is attempt 0). The
    maximum possible wait doubles with each attempt -- this is the
    "exponential" part -- but instead of always waiting that maximum, we
    pick a uniformly random value between 0 and it (the "full jitter"
    part). Randomizing like this spreads retries out over time, so that
    many callers who all failed at the same moment don't all retry at
    exactly the same moment and overwhelm the server again.
    """
    ceiling = _BACKOFF_BASE_SECONDS * (2**attempt)
    return random.uniform(0.0, ceiling)  # noqa: S311 -- backoff jitter, not security-sensitive


def _raise_for_load_bearing_status(
    response: httpx.Response, *, endpoint: str, dataset_id: str
) -> None:
    """Turn a final failed HTTP response into a specific, typed exception.

    "Final" means either the response's status was not one of the
    retryable ones handled by :func:`_get`, or all of :func:`_get`'s
    retries have already been used up -- so this only runs once nothing
    more can be done about the request itself. Depending on the status
    code, this raises a different, purpose-built exception -- e.g.
    :class:`DatasetAccessDenied` for 401/403, :class:`DatasetNotFound` for
    404, :class:`DatasetRateLimited` for 429 -- so callers can tell these
    situations apart instead of catching one generic error.

    This function itself does not know or care whether the endpoint being
    called is "load-bearing" (a failure that must fail the whole
    operation) or "best-effort" (a failure that's fine to shrug off and
    record instead) -- see the module docstring for what those two words
    mean here. That distinction is enforced by each *caller* of
    :func:`_get` instead: load-bearing callers (like the ``/is-valid`` and
    ``/splits`` calls in ``resolve()``) let the exception this function
    raises propagate straight up and fail the whole request, while
    best-effort callers (the ``_best_effort_*`` methods below) catch that
    same exception and record the corresponding feature as unavailable
    rather than letting it fail anything.
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
    #: Declares to callers (see ADR-0010, an architecture decision recorded
    #: in this project's docs) that this provider always needs the
    #: network: every single method here calls either the Hub or the
    #: Dataset Viewer HTTP API, with no local fallback. Because of that,
    #: this provider can never honestly claim to work with
    #: ``offline=True`` (a caller asking "don't touch the network for this
    #: call") -- so ``DatasetCatalog``, the class that routes requests to
    #: providers, keeps rejecting offline calls made against this
    #: provider. The one exception is ``preview``, whose results
    #: ``DatasetCatalog`` can still serve from its "content-addressed"
    #: cache (a cache keyed by an exact fingerprint of the request,
    #: defined in ``agentic_evalkit.datasets.cache``) when one has been
    #: configured and already has this exact page stored.
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
        # The real HfApi class spells out its methods' keyword arguments
        # explicitly, rather than accepting a generic **kwargs catch-all
        # the way _HubClient's Protocol methods do. Because of that
        # mismatch in how the two are *declared*, mypy's static type
        # checker cannot prove on its own that a real HfApi instance
        # actually satisfies the _HubClient Protocol -- even though, in
        # practice, every call this provider makes only ever uses keyword
        # arguments that HfApi genuinely accepts (see _HubClient's
        # docstring above). Since mypy can't confirm this ahead of time,
        # it's instead checked while the tests actually run, via Python's
        # @runtime_checkable decorator on _HubClient (see
        # test_create_returns_context_manager_owning_its_client and the
        # direct isinstance() checks against a real HfApi() elsewhere in
        # the test suite).
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
        """Run a Hub search and package the results as a ``SearchPage``.

        Many paginated APIs hand back an opaque "cursor" token alongside a
        page of results -- a stand-in value you pass back on the next call
        meaning "continue after where I left off." Hugging Face Hub's
        search does not offer a stable cursor like that for this endpoint,
        so this method has no "next page" token to give back: it fetches
        at most ``limit`` hits in one call and always returns
        ``cursor=None``. A caller wanting more results simply calls
        ``search`` again with a larger ``limit``, rather than paging
        through a cursor.
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
        """Turn a loose dataset reference into one exact, pinned-down dataset snapshot.

        ``ref`` may be underspecified -- e.g. it might name a dataset
        without saying exactly which revision, config, or split to use.
        ``resolve()`` fills in every one of those and returns a
        ``ResolvedDataset``: an immutable (never modified after creation)
        record describing one exact, reproducible dataset snapshot, so that
        running the same evaluation again later uses the exact same data.
        See plan Step 3 / the design doc's §6.2 for the full contract this
        implements.

        Three of the Hugging Face calls this makes are **load-bearing**,
        meaning a failure on any one of them fails the entire resolution:
        ``/is-valid``, ``dataset_info`` (which must return a non-empty
        commit SHA -- Hugging Face's unique, Git-style ID for one exact
        version of the dataset; an empty one means we can't actually pin a
        version, so we cannot proceed), and ``/splits``. Three other calls
        are **best-effort**, meaning a failure on any one of them is simply
        recorded rather than treated as fatal: ``/size``, ``/statistics``,
        and ``/parquet``. Plenty of legitimate datasets lack statistics or
        a Parquet export, so those failures are recorded in
        ``schema_metadata`` and resolution continues normally rather than
        being aborted.
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
        """Call the Hub's ``dataset_info`` exactly once and pull out everything ``resolve()`` needs.

        ``dataset_info`` is an ordinary blocking/synchronous Hub call, so it
        is run through ``asyncio.to_thread`` (handing it off to a
        background thread) like every other Hub call in this module. Both
        the load-bearing commit SHA (the piece ``resolve()`` cannot proceed
        without -- see its docstring) and the best-effort license,
        citation, "card" metadata (the structured, README-like description
        a dataset publishes on the Hub), and gated flag (whether the
        dataset requires the owner's approval to access) all come from
        this single response, rather than calling the Hub twice to get
        them separately.
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
        """Return one page of rows via the Dataset Viewer's ``/rows`` endpoint (100 rows max).

        If fewer rows come back than were requested because the dataset's
        total row count was reached, that shorter page is returned as-is
        rather than treated as an error; the caller can compare the number
        of rows returned against ``total_rows`` to tell a "ran out of
        data" partial page apart from a request that simply asked for
        fewer rows to begin with.
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
        """Make one GET request to a Dataset Viewer endpoint, retrying a limited number of times.

        Only two kinds of failure are retried: connection-level errors (the
        request never reached the server at all) and HTTP responses with
        status 429 (rate limited -- too many requests too fast) or
        502/503/504 (server-side errors that are usually temporary, e.g.
        the upstream service was briefly overloaded or restarting). When
        the server sends a ``Retry-After`` header telling us how long to
        wait, we honor that; otherwise we wait using
        :func:`_jittered_backoff_seconds` (a randomized, increasingly long
        delay -- see that function's docstring). We retry at most
        ``self._max_retries`` times, so at most ``max_retries + 1``
        attempts happen in total. Any other failure status is not retried
        at all -- it is immediately turned into a specific exception by
        :func:`_raise_for_load_bearing_status` (see its docstring for what
        "immediately" means here and how it decides which exception to
        raise).
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
    """Pin down which config and split to use, validating or inferring them from ``/splits``.

    Many Hugging Face datasets bundle multiple variants under one dataset
    ID -- a "config" is the name of one such variant (e.g. different
    languages of the same dataset), and a "split" is a named partition
    within it (e.g. ``train``, ``validation``, ``test``). ``/splits``
    reports every config/split combination the Dataset Viewer actually
    knows about for this dataset. If the caller's ``ref`` already names
    both an exact config and split, this just checks that combination is
    really one of the ones ``/splits`` lists. If the caller left one or
    both unspecified, this tries to fill in the gap automatically -- but
    only when there is exactly one combination it could possibly mean.

    Raises ``DatasetConfigRequired`` when a single config/split combination
    cannot be pinned down unambiguously -- either because the caller gave
    an explicit combination that ``/splits`` doesn't recognize, or because
    they left it unspecified and more than one combination could match.
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
