"""The shared local/Hugging Face dataset-row serialization contract."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agentic_evalkit._hashing import sha256_digest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic import JsonValue


def canonical_row_bytes(row: Mapping[str, JsonValue]) -> bytes:
    """Preserve sorted, compact, ASCII-escaped JSON encoded as UTF-8."""
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_row_digest(row: Mapping[str, JsonValue]) -> str:
    """Hash a row using the providers' existing byte representation."""
    return sha256_digest(canonical_row_bytes(row))
