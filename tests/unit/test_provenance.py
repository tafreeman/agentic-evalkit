"""Tests for deterministic fingerprint helpers (design §5.6).

A "fingerprint" here is a hash that proves what exact code, environment, or
target configuration produced a run -- see ``agentic_evalkit.provenance``'s
own module docstring for the full explanation. Every fingerprint function
must be "deterministic": call it twice with the exact same input and you
always get back the exact same output, in the form ``"sha256:"`` followed
by 64 hex characters. For :func:`compute_target_fingerprint` specifically,
that determinism holds no matter what order a dict's keys happen to be
listed in.

These tests check that *shape* of guarantee -- same input always gives the
same output, in the right format -- rather than pinning any one specific
fingerprint value. That's deliberate: the environment and code fingerprints
are allowed to come out differently across different interpreters or
installs. What must never change is that calling the same function twice,
on the same machine, always agrees with itself.
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
