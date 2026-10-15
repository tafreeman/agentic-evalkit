"""Byte-only SHA-256 helpers; callers own serialization contracts."""

import hashlib


def sha256_hex(payload: bytes) -> str:
    """Return the full hexadecimal SHA-256 hash of payload."""
    return hashlib.sha256(payload).hexdigest()


def sha256_digest(payload: bytes) -> str:
    """Return the existing prefixed SHA-256 fingerprint."""
    return "sha256:" + sha256_hex(payload)
