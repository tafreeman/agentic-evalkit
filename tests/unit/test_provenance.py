"""Tests for deterministic fingerprint helpers (design §5.6).

Every fingerprint helper must be a pure function of its inputs: same
inputs -> same ``"sha256:" + 64 hex chars`` digest, every time, regardless
of dict key order for :func:`compute_target_fingerprint`. These tests pin
that contract rather than any specific digest value, since the environment
and code fingerprints legitimately vary across interpreters/installs.
"""

from __future__ import annotations

import re

from agentic_evalkit.provenance import (
    compute_code_fingerprint,
    compute_environment_fingerprint,
    compute_target_fingerprint,
)

_SHA256_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def test_environment_fingerprint_is_deterministic() -> None:
    first = compute_environment_fingerprint()
    second = compute_environment_fingerprint()
    assert first == second


def test_environment_fingerprint_has_sha256_prefix_and_64_hex_chars() -> None:
    fingerprint = compute_environment_fingerprint()
    assert _SHA256_FINGERPRINT_PATTERN.match(fingerprint)


def test_code_fingerprint_is_deterministic() -> None:
    first = compute_code_fingerprint()
    second = compute_code_fingerprint()
    assert first == second


def test_code_fingerprint_has_sha256_prefix_and_64_hex_chars() -> None:
    fingerprint = compute_code_fingerprint()
    assert _SHA256_FINGERPRINT_PATTERN.match(fingerprint)


def test_target_fingerprint_is_deterministic() -> None:
    config = {"kind": "callable", "import_string": "pkg.mod:fn"}
    first = compute_target_fingerprint(config)
    second = compute_target_fingerprint(config)
    assert first == second


def test_target_fingerprint_has_sha256_prefix_and_64_hex_chars() -> None:
    fingerprint = compute_target_fingerprint({"kind": "callable", "import_string": "pkg.mod:fn"})
    assert _SHA256_FINGERPRINT_PATTERN.match(fingerprint)


def test_target_fingerprint_changes_when_config_changes() -> None:
    original = compute_target_fingerprint({"kind": "callable", "import_string": "pkg.mod:fn"})
    changed = compute_target_fingerprint({"kind": "callable", "import_string": "pkg.mod:other"})
    assert original != changed


def test_target_fingerprint_is_order_insensitive_to_dict_key_order() -> None:
    config_a = {"kind": "http", "url": "https://example.test", "credential_hook": "API_KEY"}
    config_b = {"credential_hook": "API_KEY", "url": "https://example.test", "kind": "http"}
    assert compute_target_fingerprint(config_a) == compute_target_fingerprint(config_b)


def test_target_fingerprint_handles_empty_config() -> None:
    fingerprint = compute_target_fingerprint({})
    assert _SHA256_FINGERPRINT_PATTERN.match(fingerprint)
