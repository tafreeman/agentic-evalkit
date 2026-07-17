"""Tests for :class:`agentic_evalkit.artifacts.ArtifactStore` (plan Task 11, Step 1).

The first test below is copied word-for-word from the original project plan
(``docs/plans/2026-07-02-agentic-evalkit-initial-release.md``, Task 11, Step
1). Its expected digest -- the SHA-256 hash ``ArtifactStore`` computes for
the exact bytes ``b"same"`` -- is a fixed value baked directly into the
test. If that hash ever came out differently, it would mean something about
how content gets hashed had changed, which is exactly what this test exists
to catch.
"""

from pathlib import Path

import pytest

from agentic_evalkit.artifacts import ArtifactStore, ArtifactStoreLimitExceeded


def test_artifacts_are_content_addressed_and_deduplicated(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    first = store.put_bytes(b"same", media_type="text/plain")
    second = store.put_bytes(b"same", media_type="text/plain")
    assert first == second
    assert store.read(first) == b"same"
    assert first.digest == "sha256:0967115f2813a3541eaef77de9d9d5773f1c0c04314b0bbfe4ff3b3b1c55b5d5"


# --- Additional ArtifactStore coverage: sidecar metadata, size limits, and redaction ---


def test_different_content_produces_different_digests(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    first = store.put_bytes(b"alpha", media_type="text/plain")
    second = store.put_bytes(b"beta", media_type="text/plain")
    assert first.digest != second.digest
    assert store.read(first) == b"alpha"
    assert store.read(second) == b"beta"


def test_sidecar_records_media_type_byte_count_creation_time_and_redaction(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"secret payload", media_type="application/json", redacted=True)
    metadata = store.metadata(ref)
    assert metadata.media_type == "application/json"
    assert metadata.byte_count == len(b"secret payload")
    assert metadata.redacted is True
    assert metadata.created_at is not None


def test_default_redaction_status_is_false(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"public payload", media_type="text/plain")
    assert store.metadata(ref).redacted is False


def test_put_bytes_over_configured_maximum_raises_without_writing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_bytes=4)
    with pytest.raises(ArtifactStoreLimitExceeded):
        store.put_bytes(b"too-large", media_type="text/plain")
    assert list(tmp_path.iterdir()) == []


def test_put_bytes_at_exactly_the_configured_maximum_succeeds(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_bytes=4)
    ref = store.put_bytes(b"1234", media_type="text/plain")
    assert store.read(ref) == b"1234"


def test_writes_are_not_left_as_stray_temp_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.put_bytes(b"payload", media_type="text/plain")
    leftover_temp_files = [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]
    assert leftover_temp_files == []
