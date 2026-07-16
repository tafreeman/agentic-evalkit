"""Unit tests for the Hugging Face discovery and Dataset Viewer provider.

These tests never touch the network: every Dataset Viewer HTTP call is
served by ``httpx.MockTransport`` from real, captured JSON fixtures under
``tests/fixtures/huggingface/``, and every Hub call goes through
``_FakeHub``, a minimal stand-in for the subset of ``huggingface_hub.HfApi``
this provider calls (``dataset_info`` and ``list_datasets``).

Fixture layout (see task brief: "you may nest per-dataset subdirs; document
your layout in the test module"):

- ``tests/fixtures/huggingface/*.json`` -- the seven plan-listed endpoint
  fixtures (``dataset_info``, ``is_valid``, ``splits``, ``rows``, ``size``,
  ``statistics``, ``parquet``) captured live for ``openai/gsm8k``
  (config ``main``, split ``test``).
- ``tests/fixtures/huggingface/swebench_verified/*.json`` -- the same seven
  endpoints captured live for ``princeton-nlp/SWE-bench_Verified``
  (config ``default``, split ``test``).

Every fixture file is a verbatim, unmodified capture of the real HTTP
response body (see the callback report for the exact URLs used).

Live network verification (both presets resolve and preview two real rows)
lives in ``tests/live/test_huggingface_live.py`` behind the ``live`` marker.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from agentic_evalkit.datasets.huggingface import HuggingFaceDatasetProvider
from agentic_evalkit.errors import (
    DatasetAccessDenied,
    DatasetConfigRequired,
    DatasetNotFound,
    DatasetProviderUnavailable,
    DatasetRateLimited,
)
from agentic_evalkit.models import DatasetRef

if TYPE_CHECKING:
    from collections.abc import Iterable

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "huggingface"


def _load(*parts: str) -> dict[str, Any]:
    path = _FIXTURES.joinpath(*parts)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


# --- Captured fixture payloads (loaded once at import time) -----------------

GSM8K_DATASET_INFO = _load("dataset_info.json")
GSM8K_IS_VALID = _load("is_valid.json")
GSM8K_SPLITS = _load("splits.json")
GSM8K_ROWS = _load("rows.json")
GSM8K_SIZE = _load("size.json")
GSM8K_STATISTICS = _load("statistics.json")
GSM8K_PARQUET = _load("parquet.json")

SWE_DATASET_INFO = _load("swebench_verified", "dataset_info.json")
SWE_IS_VALID = _load("swebench_verified", "is_valid.json")
SWE_SPLITS = _load("swebench_verified", "splits.json")
SWE_ROWS = _load("swebench_verified", "rows.json")
SWE_SIZE = _load("swebench_verified", "size.json")
SWE_STATISTICS = _load("swebench_verified", "statistics.json")
SWE_PARQUET = _load("swebench_verified", "parquet.json")


class _FakeDatasetInfo:
    """Minimal stand-in for ``huggingface_hub.hf_api.DatasetInfo``."""

    def __init__(
        self,
        *,
        id: str,  # noqa: A002 -- mirrors huggingface_hub.hf_api.DatasetInfo's real field name
        sha: str | None,
        private: bool = False,
        gated: bool | str = False,
        downloads: int | None = None,
        tags: Iterable[str] = (),
        card_data: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.sha = sha
        self.private = private
        self.gated = gated
        self.downloads = downloads
        self.tags = list(tags)
        self.card_data = card_data


def _dataset_info_from_fixture(
    fixture: dict[str, Any], *, sha: str | None = None
) -> _FakeDatasetInfo:
    return _FakeDatasetInfo(
        id=fixture["id"],
        sha=sha if sha is not None else fixture["sha"],
        private=fixture["private"],
        gated=fixture["gated"],
        downloads=fixture.get("downloads"),
        tags=fixture.get("tags", ()),
        card_data=fixture.get("cardData"),
    )


class _FakeHub:
    """Test double for the subset of ``HfApi`` the provider calls.

    Configured with a mapping of ``repo_id`` to a ``_FakeDatasetInfo`` (or an
    exception instance to raise) so tests can drive Hub responses without a
    real ``HfApi`` or network access.
    """

    def __init__(
        self,
        *,
        sha: str | None = None,
        datasets: dict[str, _FakeDatasetInfo | Exception] | None = None,
        search_results: tuple[_FakeDatasetInfo, ...] = (),
    ) -> None:
        self._sha = sha
        self._datasets = datasets or {}
        self._search_results = search_results
        self.dataset_info_calls: list[tuple[str, str | None]] = []
        self.list_datasets_calls: list[dict[str, Any]] = []

    def dataset_info(
        self, repo_id: str, *, revision: str | None = None, **_: Any
    ) -> _FakeDatasetInfo:
        self.dataset_info_calls.append((repo_id, revision))
        if repo_id in self._datasets:
            configured = self._datasets[repo_id]
            if isinstance(configured, Exception):
                raise configured
            return configured
        if self._sha is not None:
            return _FakeDatasetInfo(id=repo_id, sha=self._sha)
        raise AssertionError(f"_FakeHub not configured for dataset_info({repo_id!r})")

    def list_datasets(self, **kwargs: Any) -> tuple[_FakeDatasetInfo, ...]:
        self.list_datasets_calls.append(kwargs)
        return self._search_results


def _handler_for(fixtures: dict[str, dict[str, Any]]) -> Any:
    """Build a MockTransport handler serving Dataset Viewer fixture payloads.

    ``fixtures`` maps an endpoint suffix (``"is-valid"``, ``"splits"``,
    ``"rows"``, ``"size"``, ``"statistics"``, ``"parquet"``) to the JSON
    payload it should return for any request whose path ends with that
    suffix.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        for suffix, payload in fixtures.items():
            if request.url.path.endswith(f"/{suffix}"):
                return httpx.Response(200, json=payload)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    return handler


def _handler_with_override(
    overrides: dict[str, httpx.Response | Any] | None = None,
) -> Any:
    """Build the standard 5-endpoint GSM8K-fixture handler with per-test overrides.

    By default every endpoint (``is-valid``, ``splits``, ``size``,
    ``statistics``, ``parquet``) returns its captured 200 GSM8K fixture.
    ``overrides`` lets a test replace the response for just the endpoint(s)
    it cares about, keyed by endpoint suffix. Each override value is either
    an ``httpx.Response`` (returned as-is) or a callable
    ``(request: httpx.Request) -> httpx.Response`` for tests that need
    stateful/dynamic behavior (e.g. failing once then succeeding).

    Any request whose path doesn't match one of the five known endpoints
    (or an extra endpoint introduced via ``overrides``, e.g. ``"rows"``)
    still raises ``AssertionError``, matching the previous inline handlers.
    """

    defaults: dict[str, httpx.Response] = {
        "is-valid": httpx.Response(200, json=GSM8K_IS_VALID),
        "splits": httpx.Response(200, json=GSM8K_SPLITS),
        "size": httpx.Response(200, json=GSM8K_SIZE),
        "statistics": httpx.Response(200, json=GSM8K_STATISTICS),
        "parquet": httpx.Response(200, json=GSM8K_PARQUET),
    }
    resolved_overrides = overrides or {}
    known_suffixes = {**defaults, **resolved_overrides}

    def handler(request: httpx.Request) -> httpx.Response:
        for suffix in known_suffixes:
            if request.url.path.endswith(f"/{suffix}"):
                override = resolved_overrides.get(suffix)
                if override is None:
                    return defaults[suffix]
                if isinstance(override, httpx.Response):
                    return override
                return override(request)
        raise AssertionError(f"unexpected request {request.url}")

    return handler


def _gsm8k_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        _handler_for(
            {
                "is-valid": GSM8K_IS_VALID,
                "splits": GSM8K_SPLITS,
                "rows": GSM8K_ROWS,
                "size": GSM8K_SIZE,
                "statistics": GSM8K_STATISTICS,
                "parquet": GSM8K_PARQUET,
            }
        )
    )


def _swe_transport() -> httpx.MockTransport:
    return httpx.MockTransport(
        _handler_for(
            {
                "is-valid": SWE_IS_VALID,
                "splits": SWE_SPLITS,
                "rows": SWE_ROWS,
                "size": SWE_SIZE,
                "statistics": SWE_STATISTICS,
                "parquet": SWE_PARQUET,
            }
        )
    )


def _gsm8k_ref(**overrides: Any) -> DatasetRef:
    fields: dict[str, Any] = {
        "provider": "huggingface",
        "dataset_id": "openai/gsm8k",
        "config": "main",
        "split": "test",
    }
    fields.update(overrides)
    return DatasetRef(**fields)


def _swe_ref(**overrides: Any) -> DatasetRef:
    fields: dict[str, Any] = {
        "provider": "huggingface",
        "dataset_id": "princeton-nlp/SWE-bench_Verified",
        "config": "default",
        "split": "test",
    }
    fields.update(overrides)
    return DatasetRef(**fields)


# --- Step 1 (plan verbatim): request-shape and pagination -------------------


@pytest.mark.asyncio
async def test_preview_always_sends_resolved_config_and_split() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/splits"):
            return httpx.Response(200, json={"splits": [{"config": "default", "split": "test"}]})
        return httpx.Response(
            200,
            json={
                "rows": [{"row_idx": 0, "row": {"instance_id": "x"}}],
                "num_rows_total": 1,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HuggingFaceDatasetProvider(client=client, hub=_FakeHub(sha="abc123"))
    resolved = await provider.resolve(
        DatasetRef(
            provider="huggingface",
            dataset_id="princeton-nlp/SWE-bench_Verified",
            config="default",
            split="test",
        )
    )
    await provider.preview(resolved, offset=0, limit=1)
    row_request = seen[-1]
    assert row_request.url.params["config"] == "default"
    assert row_request.url.params["split"] == "test"
    await client.aclose()


# --- resolve(): captured fixtures for both verified presets ------------------


@pytest.mark.asyncio
async def test_resolve_gsm8k_from_captured_fixtures() -> None:
    client = httpx.AsyncClient(transport=_gsm8k_transport())
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)

    resolved = await provider.resolve(_gsm8k_ref())

    assert resolved.dataset_id == "openai/gsm8k"
    assert resolved.revision == GSM8K_DATASET_INFO["sha"]
    assert resolved.config == "main"
    assert resolved.split == "test"
    assert resolved.row_count == 1319
    assert resolved.license == "mit"
    assert resolved.gated is False
    # dataset_info supplies both the commit SHA and the card metadata, so
    # resolve() must call the Hub exactly once for it, not once per field.
    assert hub.dataset_info_calls == [("openai/gsm8k", None)]
    await client.aclose()


@pytest.mark.asyncio
async def test_resolve_swebench_verified_from_captured_fixtures() -> None:
    client = httpx.AsyncClient(transport=_swe_transport())
    hub = _FakeHub(
        datasets={"princeton-nlp/SWE-bench_Verified": _dataset_info_from_fixture(SWE_DATASET_INFO)}
    )
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)

    resolved = await provider.resolve(_swe_ref())

    assert resolved.dataset_id == "princeton-nlp/SWE-bench_Verified"
    assert resolved.revision == SWE_DATASET_INFO["sha"]
    assert resolved.config == "default"
    assert resolved.split == "test"
    assert resolved.row_count == 500
    await client.aclose()


@pytest.mark.asyncio
async def test_resolve_requires_nonempty_commit_sha() -> None:
    client = httpx.AsyncClient(transport=_gsm8k_transport())
    hub = _FakeHub(
        datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO, sha="")}
    )
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)

    with pytest.raises(DatasetProviderUnavailable):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


@pytest.mark.asyncio
async def test_resolve_ambiguous_config_raises_config_required() -> None:
    """gsm8k has two configs (main, socratic); omitting config is ambiguous."""
    client = httpx.AsyncClient(transport=_gsm8k_transport())
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)

    with pytest.raises(DatasetConfigRequired):
        await provider.resolve(_gsm8k_ref(config=None, split=None))
    await client.aclose()


@pytest.mark.asyncio
async def test_resolve_unique_split_can_be_inferred() -> None:
    """A dataset with exactly one config/split pair matching the ref infers it."""
    client = httpx.AsyncClient(transport=_swe_transport())
    hub = _FakeHub(
        datasets={"princeton-nlp/SWE-bench_Verified": _dataset_info_from_fixture(SWE_DATASET_INFO)}
    )
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)

    resolved = await provider.resolve(_swe_ref(config=None, split=None))

    assert resolved.config == "default"
    assert resolved.split == "test"
    await client.aclose()


# --- Best-effort endpoints: size/statistics/parquet failures don't fail resolve


@pytest.mark.asyncio
async def test_size_failure_is_recorded_not_fatal() -> None:
    handler = _handler_with_override(
        {"size": httpx.Response(500, json={"error": "internal error"})}
    )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    resolved = await provider.resolve(_gsm8k_ref())

    # resolve() succeeds despite /size failing; absence is recorded, not fatal.
    assert resolved.row_count is None
    assert resolved.schema_metadata.get("size_available") is False
    await client.aclose()


@pytest.mark.asyncio
async def test_statistics_and_parquet_failure_is_recorded_not_fatal() -> None:
    handler = _handler_with_override(
        {
            "statistics": httpx.Response(404, json={"error": "no statistics"}),
            "parquet": httpx.Response(404, json={"error": "no parquet"}),
        }
    )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    resolved = await provider.resolve(_gsm8k_ref())

    assert resolved.row_count == 1319  # /size succeeded
    assert resolved.schema_metadata.get("statistics_available") is False
    assert resolved.schema_metadata.get("parquet_available") is False
    await client.aclose()


# --- Load-bearing endpoint errors classify correctly -------------------------


@pytest.mark.asyncio
async def test_is_valid_404_raises_dataset_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/is-valid"):
            return httpx.Response(404, json={"error": "not found"})
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(sha="deadbeef")
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    with pytest.raises(DatasetNotFound):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


@pytest.mark.asyncio
async def test_is_valid_403_raises_dataset_access_denied() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/is-valid"):
            return httpx.Response(403, json={"error": "forbidden"})
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(sha="deadbeef")
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    with pytest.raises(DatasetAccessDenied):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


@pytest.mark.asyncio
async def test_is_valid_401_raises_dataset_access_denied() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/is-valid"):
            return httpx.Response(401, json={"error": "unauthorized"})
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(sha="deadbeef")
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    with pytest.raises(DatasetAccessDenied):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


@pytest.mark.asyncio
async def test_splits_422_raises_dataset_config_required() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/is-valid"):
            return httpx.Response(200, json=GSM8K_IS_VALID)
        if request.url.path.endswith("/splits"):
            return httpx.Response(422, json={"error": "invalid config"})
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    with pytest.raises(DatasetConfigRequired):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


@pytest.mark.asyncio
async def test_transport_error_on_is_valid_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(sha="deadbeef")
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    with pytest.raises(DatasetProviderUnavailable):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


@pytest.mark.asyncio
async def test_dataset_info_not_found_raises_dataset_not_found() -> None:
    from agentic_evalkit.errors import DatasetNotFound as ProviderDatasetNotFound

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/is-valid"):
            return httpx.Response(200, json=GSM8K_IS_VALID)
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": ProviderDatasetNotFound(message="missing")})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    with pytest.raises(DatasetNotFound):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


# --- Retry policy -------------------------------------------------------------


async def _no_op_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_429_then_200_succeeds() -> None:
    attempts = {"is-valid": 0}

    def is_valid_once_then_ok(request: httpx.Request) -> httpx.Response:
        attempts["is-valid"] += 1
        if attempts["is-valid"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "slow down"})
        return httpx.Response(200, json=GSM8K_IS_VALID)

    handler = _handler_with_override({"is-valid": is_valid_once_then_ok})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep)

    resolved = await provider.resolve(_gsm8k_ref())

    assert resolved.dataset_id == "openai/gsm8k"
    assert attempts["is-valid"] == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_repeated_429_raises_dataset_rate_limited() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/is-valid"):
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "slow down"})
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(sha="deadbeef")
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=3)

    with pytest.raises(DatasetRateLimited):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


@pytest.mark.asyncio
async def test_nonretryable_4xx_attempted_once() -> None:
    attempts = {"is-valid": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/is-valid"):
            attempts["is-valid"] += 1
            return httpx.Response(404, json={"error": "not found"})
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(sha="deadbeef")
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=3)

    with pytest.raises(DatasetNotFound):
        await provider.resolve(_gsm8k_ref())
    assert attempts["is-valid"] == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_502_is_retried_and_recovers() -> None:
    attempts = {"is-valid": 0}

    def is_valid_bad_gateway_then_ok(request: httpx.Request) -> httpx.Response:
        attempts["is-valid"] += 1
        if attempts["is-valid"] < 2:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(200, json=GSM8K_IS_VALID)

    handler = _handler_with_override({"is-valid": is_valid_bad_gateway_then_ok})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep)

    resolved = await provider.resolve(_gsm8k_ref())
    assert resolved.dataset_id == "openai/gsm8k"
    assert attempts["is-valid"] == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_connect_error_is_retried_then_raises_unavailable() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ConnectError("boom", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(sha="deadbeef")
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=2)

    with pytest.raises(DatasetProviderUnavailable):
        await provider.resolve(_gsm8k_ref())
    # 1 initial attempt + 2 retries = 3 total attempts.
    assert attempts["count"] == 3
    await client.aclose()


# --- preview() / iter_records() ----------------------------------------------


@pytest.mark.asyncio
async def test_preview_converts_rows_to_source_records() -> None:
    client = httpx.AsyncClient(transport=_gsm8k_transport())
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)
    resolved = await provider.resolve(_gsm8k_ref())

    page = await provider.preview(resolved, offset=0, limit=2)

    assert page.offset == 0
    assert page.total_rows == 1319
    assert len(page.records) == 2
    assert page.records[0].row_id == "0"
    first_question = page.records[0].data["question"]
    assert isinstance(first_question, str)
    assert first_question.startswith("Janet")
    assert page.records[0].digest.startswith("sha256:")
    await client.aclose()


@pytest.mark.asyncio
async def test_preview_page_capped_at_100() -> None:
    seen: list[httpx.Request] = []
    inner_handler = _handler_with_override({"rows": httpx.Response(200, json=GSM8K_ROWS)})

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return inner_handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)
    resolved = await provider.resolve(_gsm8k_ref())

    await provider.preview(resolved, offset=0, limit=500)

    rows_requests = [r for r in seen if r.url.path.endswith("/rows")]
    assert len(rows_requests) == 1
    assert rows_requests[0].url.params["length"] == "100"


@pytest.mark.asyncio
async def test_iter_records_stops_at_caller_limit() -> None:
    call_count = {"rows": 0}

    def rows_page(request: httpx.Request) -> httpx.Response:
        call_count["rows"] += 1
        offset = int(request.url.params["offset"])
        return httpx.Response(
            200,
            json={
                "rows": [
                    {"row_idx": offset, "row": {"question": "q", "answer": "a"}},
                    {"row_idx": offset + 1, "row": {"question": "q2", "answer": "a2"}},
                ],
                "num_rows_total": 1319,
            },
        )

    handler = _handler_with_override({"rows": rows_page})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)
    resolved = await provider.resolve(_gsm8k_ref())

    records = [record async for record in provider.iter_records(resolved, offset=0, limit=3)]

    assert len(records) == 3
    assert call_count["rows"] >= 1


@pytest.mark.asyncio
async def test_iter_records_stops_on_empty_page() -> None:
    handler = _handler_with_override(
        {"rows": httpx.Response(200, json={"rows": [], "num_rows_total": 0})}
    )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)
    resolved = await provider.resolve(_gsm8k_ref())

    records = [record async for record in provider.iter_records(resolved, offset=0, limit=None)]

    assert records == []


# --- search() -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_maps_hub_results_to_search_hits() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    hub = _FakeHub(
        search_results=(
            _FakeDatasetInfo(
                id="openai/gsm8k",
                sha="740312add88f781978c0658806c59bc2815b9866",
                private=False,
                gated=False,
                downloads=934833,
                tags=["language:en", "license:mit"],
                card_data={"license": ["mit"]},
            ),
        )
    )
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)

    page = await provider.search("gsm8k", limit=5)

    assert len(page.hits) == 1
    hit = page.hits[0]
    assert hit.dataset_id == "openai/gsm8k"
    assert hit.provider == "huggingface"
    assert hit.revision == "740312add88f781978c0658806c59bc2815b9866"
    assert hit.downloads == 934833
    assert hit.gated is False
    assert hit.private is False
    assert "license:mit" in hit.tags
    assert hub.list_datasets_calls[0]["search"] == "gsm8k"
    assert hub.list_datasets_calls[0]["limit"] == 5
    await client.aclose()


# --- healthcheck() --------------------------------------------------------------


@pytest.mark.asyncio
async def test_healthcheck_ok_reports_latency() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/is-valid")
        assert request.url.params["dataset"] == "openai/gsm8k"
        return httpx.Response(200, json=GSM8K_IS_VALID)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HuggingFaceDatasetProvider(client=client, hub=_FakeHub(sha="deadbeef"))

    health = await provider.healthcheck()

    assert health.status == "ok"
    assert health.latency_ms is not None
    assert health.latency_ms >= 0
    await client.aclose()


@pytest.mark.asyncio
async def test_healthcheck_error_reports_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HuggingFaceDatasetProvider(
        client=client, hub=_FakeHub(sha="deadbeef"), sleep=_no_op_sleep, max_retries=0
    )

    health = await provider.healthcheck()

    assert health.status == "error"
    assert health.error_code == "dataset_provider_unavailable"
    await client.aclose()


@pytest.mark.asyncio
async def test_healthcheck_rate_limited_reports_degraded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "slow down"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HuggingFaceDatasetProvider(
        client=client, hub=_FakeHub(sha="deadbeef"), sleep=_no_op_sleep, max_retries=0
    )

    health = await provider.healthcheck()

    assert health.status == "degraded"
    assert health.error_code == "dataset_rate_limited"
    await client.aclose()


@pytest.mark.asyncio
async def test_healthcheck_reports_degraded_when_no_capability_is_usable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"preview": False, "viewer": False, "search": False, "filter": False},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HuggingFaceDatasetProvider(client=client, hub=_FakeHub(sha="deadbeef"))

    health = await provider.healthcheck()

    assert health.status == "degraded"
    assert health.error_code == "dataset_provider_unavailable"
    await client.aclose()


# --- Additional edge cases: malformed responses and alternate shapes --------


@pytest.mark.asyncio
async def test_check_is_valid_rejects_all_capabilities_false() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/is-valid")
        return httpx.Response(200, json={"preview": False, "viewer": False})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(sha="deadbeef")
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    with pytest.raises(DatasetProviderUnavailable):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


@pytest.mark.asyncio
async def test_malformed_retry_after_header_falls_back_to_jittered_backoff() -> None:
    """A non-numeric Retry-After must not crash retry scheduling."""
    attempts = {"is-valid": 0}

    def is_valid_bad_retry_after_then_ok(request: httpx.Request) -> httpx.Response:
        attempts["is-valid"] += 1
        if attempts["is-valid"] == 1:
            return httpx.Response(
                429, headers={"Retry-After": "not-a-number"}, json={"error": "slow down"}
            )
        return httpx.Response(200, json=GSM8K_IS_VALID)

    handler = _handler_with_override({"is-valid": is_valid_bad_retry_after_then_ok})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep)

    resolved = await provider.resolve(_gsm8k_ref())

    assert resolved.dataset_id == "openai/gsm8k"
    assert attempts["is-valid"] == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_malformed_splits_shape_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/is-valid"):
            return httpx.Response(200, json=GSM8K_IS_VALID)
        if request.url.path.endswith("/splits"):
            return httpx.Response(200, json={"splits": "not-a-list"})
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    with pytest.raises(DatasetProviderUnavailable):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


@pytest.mark.asyncio
async def test_dataset_info_unexpected_exception_raises_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/is-valid"):
            return httpx.Response(200, json=GSM8K_IS_VALID)
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": RuntimeError("Hub is down")})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub, sleep=_no_op_sleep, max_retries=0)

    with pytest.raises(DatasetProviderUnavailable):
        await provider.resolve(_gsm8k_ref())
    await client.aclose()


@pytest.mark.asyncio
async def test_explicit_config_and_split_not_in_splits_raises_config_required() -> None:
    client = httpx.AsyncClient(transport=_gsm8k_transport())
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)

    with pytest.raises(DatasetConfigRequired):
        await provider.resolve(_gsm8k_ref(config="main", split="nonexistent"))
    await client.aclose()


@pytest.mark.asyncio
async def test_config_given_split_omitted_infers_unique_split() -> None:
    client = httpx.AsyncClient(transport=_swe_transport())
    hub = _FakeHub(
        datasets={"princeton-nlp/SWE-bench_Verified": _dataset_info_from_fixture(SWE_DATASET_INFO)}
    )
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)

    resolved = await provider.resolve(_swe_ref(split=None))

    assert resolved.config == "default"
    assert resolved.split == "test"
    await client.aclose()


@pytest.mark.asyncio
async def test_search_forwards_filters_and_maps_hub_failure() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    hub = _FakeHub(search_results=())
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)

    await provider.search("gsm8k", filters={"language": "en"}, limit=3)

    assert hub.list_datasets_calls[0]["language"] == "en"
    await client.aclose()


@pytest.mark.asyncio
async def test_search_hub_failure_raises_provider_unavailable() -> None:
    class _FailingHub(_FakeHub):
        def list_datasets(self, **kwargs: Any) -> tuple[_FakeDatasetInfo, ...]:
            raise RuntimeError("Hub search is down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    provider = HuggingFaceDatasetProvider(client=client, hub=_FailingHub())

    with pytest.raises(DatasetProviderUnavailable):
        await provider.search("gsm8k")
    await client.aclose()


class _RealisticCardData:
    """Mimics ``huggingface_hub.repocard_data.DatasetCardData``'s ``to_dict()`` shape."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


@pytest.mark.asyncio
async def test_card_data_object_with_to_dict_is_converted() -> None:
    client = httpx.AsyncClient(transport=_gsm8k_transport())
    info = _FakeDatasetInfo(
        id="openai/gsm8k",
        sha=GSM8K_DATASET_INFO["sha"],
        private=False,
        gated=False,
        card_data=_RealisticCardData({"license": ["mit"], "citation": "@misc{gsm8k}"}),  # type: ignore[arg-type]
    )
    hub = _FakeHub(datasets={"openai/gsm8k": info})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)

    resolved = await provider.resolve(_gsm8k_ref())

    assert resolved.license == "mit"
    assert resolved.citation == "@misc{gsm8k}"
    await client.aclose()


@pytest.mark.asyncio
async def test_iter_records_stops_exactly_at_limit_mid_page() -> None:
    """The caller limit falls inside a page rather than on a page boundary."""

    handler = _handler_with_override(
        {
            "rows": httpx.Response(
                200,
                json={
                    "rows": [
                        {"row_idx": i, "row": {"question": f"q{i}", "answer": f"a{i}"}}
                        for i in range(5)
                    ],
                    "num_rows_total": 1319,
                },
            )
        }
    )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)
    resolved = await provider.resolve(_gsm8k_ref())

    records = [record async for record in provider.iter_records(resolved, offset=0, limit=2)]

    assert [record.row_id for record in records] == ["0", "1"]


@pytest.mark.asyncio
async def test_iter_records_stops_exactly_at_upstream_total() -> None:
    """current_offset reaching num_rows_total exactly must stop, not overshoot."""

    handler = _handler_with_override(
        {
            "rows": httpx.Response(
                200,
                json={
                    "rows": [{"row_idx": 0, "row": {"question": "q", "answer": "a"}}],
                    "num_rows_total": 1,
                },
            )
        }
    )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = _FakeHub(datasets={"openai/gsm8k": _dataset_info_from_fixture(GSM8K_DATASET_INFO)})
    provider = HuggingFaceDatasetProvider(client=client, hub=hub)
    resolved = await provider.resolve(_gsm8k_ref())

    records = [record async for record in provider.iter_records(resolved, offset=0, limit=None)]

    assert len(records) == 1


# --- create() async context manager ------------------------------------------


@pytest.mark.asyncio
async def test_create_returns_context_manager_owning_its_client() -> None:
    async with HuggingFaceDatasetProvider.create() as provider:
        assert isinstance(provider, HuggingFaceDatasetProvider)
        assert provider.api_version == "1"
