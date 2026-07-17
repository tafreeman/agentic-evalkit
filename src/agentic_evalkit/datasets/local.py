"""Local filesystem dataset provider (design §6.1, plan Task 5).

Reads dataset files in JSON, JSONL, CSV, or YAML format, but only from a
fixed, pre-approved list of root directories (an "allow list") -- this
provider will never read arbitrary paths on disk. Every row read from a
file is checked to make sure it's a JSON object (``dict[str, JsonValue]``),
and each row gets two identifiers: a row number (as a string, starting at
0) and a SHA-256 hash of its own contents in a standardized ("canonical")
JSON form, so that identical row content always produces the identical
hash. The dataset's ``revision`` field is the SHA-256 hash of the raw
file's bytes -- so if even one byte of the source file changes, that
counts as a completely different, new "revision" of the dataset.

This provider does not scan its directories to build a search index, so
its ``search`` method always returns an empty (but successful) result --
figuring out which local files exist is left up to whoever is calling this
code.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

import yaml

from agentic_evalkit.datasets.base import ProviderHealth
from agentic_evalkit.errors import DatasetSchemaMismatch
from agentic_evalkit.models import (
    DatasetRef,
    ResolvedDataset,
    SamplePage,
    SearchPage,
    SourceRecord,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from pydantic import JsonValue

_SUPPORTED_SUFFIXES: Final[frozenset[str]] = frozenset({".json", ".jsonl", ".csv", ".yaml", ".yml"})


def _canonical_digest(row: dict[str, JsonValue]) -> str:
    """Compute the SHA-256 hash of one row, after converting it to a
    standardized ("canonical") JSON form -- keys sorted alphabetically, no
    extra whitespace -- so that two rows with identical data always hash to
    the same value."""
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_row(value: object, *, path: Path, index: int) -> dict[str, JsonValue]:
    """Check that a decoded row is a JSON object (a Python dict with string
    keys); raise DatasetSchemaMismatch if it isn't."""
    if not isinstance(value, dict):
        raise DatasetSchemaMismatch(
            message=f"row {index} in {path} is not a JSON object",
            context={"path": str(path), "row_index": index, "row_type": type(value).__name__},
        )
    for key in value:
        if not isinstance(key, str):
            raise DatasetSchemaMismatch(
                message=f"row {index} in {path} has a non-string key",
                context={"path": str(path), "row_index": index},
            )
    return value


def _decode_json(raw: bytes, *, path: Path) -> list[dict[str, JsonValue]]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetSchemaMismatch(
            message=f"{path} is not valid JSON", context={"path": str(path)}
        ) from exc
    if isinstance(payload, dict):
        records = payload.get("records")
        if not isinstance(records, list):
            raise DatasetSchemaMismatch(
                message=f"{path} object payload is missing a 'records' list",
                context={"path": str(path)},
            )
        rows = records
    elif isinstance(payload, list):
        rows = payload
    else:
        raise DatasetSchemaMismatch(
            message=f"{path} must decode to a list of objects or an object with 'records'",
            context={"path": str(path), "payload_type": type(payload).__name__},
        )
    return [_validate_row(row, path=path, index=i) for i, row in enumerate(rows)]


def _decode_jsonl(raw: bytes, *, path: Path) -> list[dict[str, JsonValue]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetSchemaMismatch(
            message=f"{path} is not valid UTF-8", context={"path": str(path)}
        ) from exc
    rows: list[dict[str, JsonValue]] = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetSchemaMismatch(
                message=f"line {index} in {path} is not valid JSON",
                context={"path": str(path), "line_index": index},
            ) from exc
        rows.append(_validate_row(payload, path=path, index=index))
    return rows


def _decode_csv(raw: bytes, *, path: Path) -> list[dict[str, JsonValue]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetSchemaMismatch(
            message=f"{path} is not valid UTF-8", context={"path": str(path)}
        ) from exc
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, JsonValue]] = []
    for index, row in enumerate(reader):
        if None in row:
            raise DatasetSchemaMismatch(
                message=f"row {index} in {path} has more fields than the CSV header",
                context={"path": str(path), "row_index": index},
            )
        rows.append(_validate_row(dict(row), path=path, index=index))
    return rows


def _decode_yaml(raw: bytes, *, path: Path) -> list[dict[str, JsonValue]]:
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DatasetSchemaMismatch(
            message=f"{path} is not valid YAML", context={"path": str(path)}
        ) from exc
    if not isinstance(payload, list):
        raise DatasetSchemaMismatch(
            message=f"{path} must decode to a list of objects",
            context={"path": str(path), "payload_type": type(payload).__name__},
        )
    return [_validate_row(row, path=path, index=i) for i, row in enumerate(payload)]


_DECODERS: Final[dict[str, object]] = {
    ".json": _decode_json,
    ".jsonl": _decode_jsonl,
    ".csv": _decode_csv,
    ".yaml": _decode_yaml,
    ".yml": _decode_yaml,
}


def _decode_rows(raw: bytes, *, path: Path) -> list[dict[str, JsonValue]]:
    decoder = _DECODERS[path.suffix.lower()]
    rows: list[dict[str, JsonValue]] = decoder(raw, path=path)  # type: ignore[operator]
    return rows


def _rows_to_records(rows: list[dict[str, JsonValue]]) -> tuple[SourceRecord, ...]:
    return tuple(
        SourceRecord(row_id=str(i), data=row, digest=_canonical_digest(row))
        for i, row in enumerate(rows)
    )


class LocalDatasetProvider:
    """Dataset provider for local JSON/JSONL/CSV/YAML files (design §6.1).

    Every method in this class only reads from the local filesystem (e.g.
    ``Path.read_bytes``, checking whether a directory exists) -- nothing in
    this class ever makes a network call. The class-level flag
    ``requires_network = False`` records that fact so other code can rely on
    it (ADR-0010 is the design decision that introduced this flag). Because
    of that flag, :class:`~agentic_evalkit.datasets.catalog.DatasetCatalog`
    knows it's safe to send this provider requests made with
    ``offline=True`` (i.e. "don't touch the network") instead of blocking
    them the way it would for a provider that might need the network.
    """

    api_version: Final[str] = "1"
    requires_network: Final[bool] = False

    def __init__(self, allowed_roots: tuple[Path, ...]) -> None:
        self._allowed_roots: tuple[Path, ...] = tuple(root.resolve() for root in allowed_roots)

    def _resolve_and_validate_path(self, dataset_id: str) -> Path:
        path = Path(dataset_id).resolve()
        if not any(path == root or path.is_relative_to(root) for root in self._allowed_roots):
            raise ValueError(f"path {path} is outside allowed roots {self._allowed_roots}")
        if path.is_dir():
            raise ValueError(f"path {path} is a directory, not a dataset file")
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"path {path} has an unsupported suffix {path.suffix!r}")
        if not path.is_file():
            raise ValueError(f"path {path} does not exist")
        return path

    async def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> SearchPage:
        """Always returns an empty result: this provider doesn't build a
        search index over its local root directories, so it has nothing to
        search through."""
        return SearchPage(hits=(), cursor=None, total_hits=0)

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        path = self._resolve_and_validate_path(ref.dataset_id)
        raw = path.read_bytes()
        revision = "sha256:" + hashlib.sha256(raw).hexdigest()
        rows = _decode_rows(raw, path=path)
        return ResolvedDataset(
            dataset_id=str(path),
            revision=revision,
            config=ref.config,
            split=ref.split,
            selected_files=(str(path),),
            row_count=len(rows),
        )

    async def preview(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int = 10
    ) -> SamplePage:
        rows = self._load_rows(dataset)
        records = _rows_to_records(rows)
        page = records[offset : offset + limit]
        return SamplePage(records=page, offset=offset, total_rows=len(records))

    async def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        rows = self._load_rows(dataset)
        records = _rows_to_records(rows)
        end = len(records) if limit is None else offset + limit
        for record in records[offset:end]:
            yield record

    async def healthcheck(self) -> ProviderHealth:
        for root in self._allowed_roots:
            if not root.exists() or not _is_readable(root):
                return ProviderHealth(
                    status="error",
                    capabilities=("search", "resolve", "preview", "iter_records"),
                    error_code="dataset_provider_unavailable",
                )
        return ProviderHealth(
            status="ok", capabilities=("search", "resolve", "preview", "iter_records")
        )

    def _load_rows(self, dataset: ResolvedDataset) -> list[dict[str, JsonValue]]:
        if not dataset.selected_files:
            raise DatasetSchemaMismatch(
                message="resolved local dataset has no selected file",
                context={"dataset_id": dataset.dataset_id},
            )
        path = Path(dataset.selected_files[0])
        raw = path.read_bytes()
        return _decode_rows(raw, path=path)


def _is_readable(root: Path) -> bool:
    try:
        next(root.iterdir(), None)
    except OSError:
        return False
    return True
