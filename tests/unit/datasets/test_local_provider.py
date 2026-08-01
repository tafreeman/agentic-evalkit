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
from agentic_evalkit.models import DatasetRef, ResolvedDataset


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


# ---------------------------------------------------------------------------
# Malformed-input contract.
#
# Every decoder below reads a file the caller supplied, so every one of them
# is a boundary against untrusted bytes: a dataset file can be truncated,
# mis-encoded, or shaped like something other than a table, and none of those
# may escape as a raw ValueError, UnicodeDecodeError, or yaml.YAMLError. The
# contract is that malformed input always surfaces as DatasetSchemaMismatch
# carrying enough context to name the offending file and row -- the same
# typed-error contract the cache read path holds. These tests pin that
# contract per decoder and per failure mode rather than trusting one
# representative case to stand in for the rest.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_row_that_is_not_an_object_names_the_offending_row(tmp_path: Path) -> None:
    """A scalar sitting among otherwise valid rows is rejected, and the error
    says which row index and which type, so the file can be fixed without
    bisecting it by hand."""
    source = tmp_path / "items.json"
    source.write_text('[{"id": "a"}, 42]', encoding="utf-8")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(DatasetSchemaMismatch) as raised:
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
    assert raised.value.context["row_index"] == 1
    assert raised.value.context["row_type"] == "int"


@pytest.mark.asyncio
async def test_yaml_non_string_key_is_rejected(tmp_path: Path) -> None:
    """YAML -- unlike JSON -- can express a mapping keyed by an integer, so it
    is the only decoder that can produce a dict whose keys are not strings.
    That would break canonical digesting downstream (json.dumps refuses to
    sort mixed-type keys), so it is rejected at the boundary instead."""
    source = tmp_path / "items.yaml"
    source.write_text("- 1: alpha\n", encoding="utf-8")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(DatasetSchemaMismatch, match="non-string key"):
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))


@pytest.mark.asyncio
async def test_truncated_json_file_is_rejected(tmp_path: Path) -> None:
    """A half-written .json file must not decode to an empty dataset -- a run
    over zero samples that reports success is the failure mode this guards."""
    source = tmp_path / "items.json"
    source.write_text('[{"id": "a"}', encoding="utf-8")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(DatasetSchemaMismatch, match="not valid JSON"):
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))


@pytest.mark.asyncio
async def test_json_object_without_a_records_list_is_rejected(tmp_path: Path) -> None:
    """The object form of a .json dataset carries its rows under "records".
    A near-miss key must fail loudly rather than silently yielding no rows."""
    source = tmp_path / "items.json"
    source.write_text('{"rows": [{"id": "a"}]}', encoding="utf-8")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(DatasetSchemaMismatch, match="'records' list"):
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))


@pytest.mark.asyncio
async def test_json_scalar_payload_is_rejected(tmp_path: Path) -> None:
    """A bare JSON string is valid JSON but is not a table; the error records
    what the payload actually decoded to."""
    source = tmp_path / "items.json"
    source.write_text('"just a string"', encoding="utf-8")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(DatasetSchemaMismatch) as raised:
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
    assert raised.value.context["payload_type"] == "str"


@pytest.mark.asyncio
async def test_invalid_yaml_is_rejected(tmp_path: Path) -> None:
    """A YAML parse error is converted to the typed error rather than leaking
    yaml.YAMLError, which callers outside this package cannot be expected to
    catch."""
    source = tmp_path / "items.yaml"
    source.write_text("[unclosed\n", encoding="utf-8")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(DatasetSchemaMismatch, match="not valid YAML"):
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))


@pytest.mark.parametrize("filename", ["items.jsonl", "items.csv", "items.yaml"])
@pytest.mark.asyncio
async def test_non_utf8_bytes_are_rejected_by_every_text_decoder(
    tmp_path: Path, filename: str
) -> None:
    """A file saved in a non-UTF-8 encoding (or a binary file with the right
    extension) reaches a bytes-to-text decode in every text decoder. Each one
    must convert UnicodeDecodeError into the typed error -- including the CSV
    decoder, whose utf-8-sig codec differs from the others."""
    source = tmp_path / filename
    source.write_bytes(b"\xff\xfe{not text}")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(DatasetSchemaMismatch) as raised:
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
    assert raised.value.context["path"] == str(source)


@pytest.mark.asyncio
async def test_blank_lines_in_jsonl_are_skipped_without_shifting_row_ids(tmp_path: Path) -> None:
    """Blank and whitespace-only lines are separators, not rows. They must not
    become empty records, and the surviving rows must still be numbered
    contiguously from zero -- a gap in row_id would misalign a dataset against
    any per-row expectations keyed on position."""
    source = tmp_path / "items.jsonl"
    source.write_text('{"id":"a"}\n\n   \n{"id":"b"}\n', encoding="utf-8")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    resolved = await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
    page = await provider.preview(resolved, offset=0, limit=10)
    assert resolved.row_count == 2
    assert [record.row_id for record in page.records] == ["0", "1"]
    assert [record.data["id"] for record in page.records] == ["a", "b"]


@pytest.mark.asyncio
async def test_csv_row_with_more_fields_than_the_header_is_rejected(tmp_path: Path) -> None:
    """csv.DictReader parks surplus values under a None key instead of
    failing, which would produce a row this package cannot digest. The extra
    field is treated as corruption and named by row index."""
    source = tmp_path / "items.csv"
    source.write_text("id,prompt\na,alpha\nb,beta,surplus\n", encoding="utf-8")
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(DatasetSchemaMismatch) as raised:
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
    assert raised.value.context["row_index"] == 1


@pytest.mark.asyncio
async def test_missing_file_with_a_supported_suffix_is_rejected(tmp_path: Path) -> None:
    """A supported extension is not enough: the path check runs to the end and
    still rejects a file that is not there, so a typo in a dataset id fails at
    resolve time rather than as an OSError deeper in the read."""
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    with pytest.raises(ValueError, match="does not exist"):
        await provider.resolve(
            DatasetRef(provider="local", dataset_id=str(tmp_path / "absent.jsonl"))
        )


@pytest.mark.asyncio
async def test_resolved_dataset_with_no_selected_file_is_rejected(tmp_path: Path) -> None:
    """A ResolvedDataset can arrive from a cache or a manifest rather than from
    this provider's own resolve(), so the read path re-checks that it names a
    file instead of indexing an empty tuple."""
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    orphan = ResolvedDataset(dataset_id=str(tmp_path / "items.jsonl"), revision="sha256:0")
    with pytest.raises(DatasetSchemaMismatch, match="no selected file"):
        await provider.preview(orphan, offset=0, limit=10)


@pytest.mark.asyncio
async def test_healthcheck_reports_error_when_a_root_is_not_a_directory(tmp_path: Path) -> None:
    """An allowed root that exists but cannot be listed -- here a regular file
    passed where a directory belongs -- is unhealthy, not healthy. The
    readability probe catches the OSError rather than letting it escape a
    healthcheck, whose whole job is to answer instead of raise."""
    not_a_directory = tmp_path / "roots.txt"
    not_a_directory.write_text("not a directory", encoding="utf-8")
    provider = LocalDatasetProvider(allowed_roots=(not_a_directory,))
    health = await provider.healthcheck()
    assert health.status == "error"
    assert health.error_code == "dataset_provider_unavailable"
