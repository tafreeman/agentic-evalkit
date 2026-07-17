"""Tests for LocalDatasetProvider: the dataset provider that reads datasets
from the local filesystem (as opposed to a remote source like HuggingFace).
Checks both that it behaves correctly on its own (raises the right errors on
bad input, pages and iterates records correctly) and that the same dataset
stored in different file formats -- JSONL, CSV, YAML -- decodes to identical
data (plan Task 5)."""

from pathlib import Path

import pytest

from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.errors import DatasetSchemaMismatch
from agentic_evalkit.models import DatasetRef


@pytest.mark.asyncio
async def test_resolve_preview_and_iterate_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "items.jsonl"
    source.write_text('{"id":"a","prompt":"alpha"}\n{"id":"b","prompt":"beta"}\n')
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    resolved = await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
    page = await provider.preview(resolved, offset=1, limit=1)
    records = [record async for record in provider.iter_records(resolved, offset=0, limit=None)]
    assert resolved.revision.startswith("sha256:")
    assert page.total_rows == 2
    assert page.records[0].data["id"] == "b"
    assert [record.row_id for record in records] == ["0", "1"]


@pytest.mark.asyncio
async def test_rejects_path_outside_allowed_roots(tmp_path: Path) -> None:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path / "allowed",))
    with pytest.raises(ValueError, match="outside allowed roots"):
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(tmp_path / "x.json")))


_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "datasets"


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["items.jsonl", "items.csv", "items.yaml"])
async def test_fixture_formats_decode_two_rows(filename: str) -> None:
    provider = LocalDatasetProvider(allowed_roots=(_FIXTURES_DIR,))
    resolved = await provider.resolve(
        DatasetRef(provider="local", dataset_id=str(_FIXTURES_DIR / filename))
    )
    page = await provider.preview(resolved, offset=0, limit=10)
    assert page.total_rows == 2
    assert [record.data for record in page.records] == [
        {"id": "a", "prompt": "alpha"},
        {"id": "b", "prompt": "beta"},
    ]


@pytest.mark.asyncio
async def test_fixture_formats_have_identical_data_but_different_revisions() -> None:
    provider = LocalDatasetProvider(allowed_roots=(_FIXTURES_DIR,))
    resolved_by_format = {
        filename: await provider.resolve(
            DatasetRef(provider="local", dataset_id=str(_FIXTURES_DIR / filename))
        )
        for filename in ("items.jsonl", "items.csv", "items.yaml")
    }

    pages = {
        filename: await provider.preview(resolved, limit=10)
        for filename, resolved in resolved_by_format.items()
    }

    canonical_data = {
        filename: tuple(record.data for record in page.records) for filename, page in pages.items()
    }
    data_values = list(canonical_data.values())
    assert all(value == data_values[0] for value in data_values), canonical_data

    canonical_digests = {
        filename: tuple(record.digest for record in page.records)
        for filename, page in pages.items()
    }
    unique_digest_shapes = set(canonical_digests.values())
    assert len(unique_digest_shapes) == 1, canonical_digests

    revisions = {resolved.revision for resolved in resolved_by_format.values()}
    assert len(revisions) == 3, revisions


@pytest.mark.asyncio
async def test_json_object_with_records_key(tmp_path: Path) -> None:
    source = tmp_path / "items.json"
    source.write_text('{"records": [{"id": "a"}, {"id": "b"}]}')
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    resolved = await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
    page = await provider.preview(resolved, offset=0, limit=10)
    assert [record.data["id"] for record in page.records] == ["a", "b"]


@pytest.mark.asyncio
async def test_json_list_of_objects(tmp_path: Path) -> None:
    source = tmp_path / "items.json"
    source.write_text('[{"id": "a"}, {"id": "b"}]')
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    resolved = await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
    page = await provider.preview(resolved, offset=0, limit=10)
    assert [record.data["id"] for record in page.records] == ["a", "b"]


@pytest.mark.asyncio
async def test_malformed_jsonl_raises_schema_mismatch_not_empty(tmp_path: Path) -> None:
    source = tmp_path / "broken.jsonl"
    source.write_text('{"id":"a"}\nnot json at all\n')
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(DatasetSchemaMismatch):
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))


@pytest.mark.asyncio
async def test_scalar_yaml_raises_schema_mismatch_not_empty(tmp_path: Path) -> None:
    source = tmp_path / "scalar.yaml"
    source.write_text("just a string\n")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(DatasetSchemaMismatch):
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))


@pytest.mark.asyncio
async def test_search_returns_empty_successful_page(tmp_path: Path) -> None:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    page = await provider.search("anything")
    assert page.hits == ()
    assert page.total_hits == 0


@pytest.mark.asyncio
async def test_healthcheck_reports_ok_for_readable_roots(tmp_path: Path) -> None:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    health = await provider.healthcheck()
    assert health.status == "ok"


@pytest.mark.asyncio
async def test_healthcheck_reports_error_for_missing_root(tmp_path: Path) -> None:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path / "does-not-exist",))
    health = await provider.healthcheck()
    assert health.status == "error"


@pytest.mark.asyncio
async def test_rejects_directory_path(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(ValueError, match="directory"):
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(tmp_path / "subdir")))


@pytest.mark.asyncio
async def test_rejects_unsupported_suffix(tmp_path: Path) -> None:
    source = tmp_path / "items.txt"
    source.write_text("hello")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(ValueError, match="unsupported suffix"):
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
