"""The ``review-outcome.v1`` JSON Schema accepts a well-formed record and rejects bad ones.

The schema describes the append-only ledger the AI code reviewer writes. These
tests pin the contract that matters to a reader of that ledger: a record with
an unknown outcome, a missing fingerprint, a malformed head SHA or an extra
top-level key must never validate.
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "review-outcome.v1.schema.json"

VALID_RECORD: dict[str, Any] = {
    "schema_version": "review-outcome/v1",
    "record_id": "0b8c1f0e-6d2a-4a54-9a43-3f1d2f6c9a10",
    "recorded_at": "2026-10-20T12:00:00Z",
    "repo": "tafreeman/agentic-evalkit",
    "pr": 79,
    "head_sha": "0123456789abcdef0123456789abcdef01234567",
    "finding_id": "a" * 64,
    "comment_id": None,
    "tier": "high_risk",
    "file": "src/agentic_evalkit/graders/exact.py",
    "line": 42,
    "category": "correctness",
    "severity": "critical",
    "evidence_label": "SOURCE-VERIFIED",
    "source": "llm",
    "outcome": "posted",
    "outcome_source": "post",
    "reject_reason": None,
    "fingerprint": {
        "model": "claude-opus-5-5",
        "prompt_version": "b" * 64,
        "harness_version": "c" * 40,
        "router_version": "d" * 64,
    },
}


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _record(**overrides: Any) -> dict[str, Any]:
    record = copy.deepcopy(VALID_RECORD)
    record.update(overrides)
    return record


def test_a_well_formed_record_validates(validator: Draft202012Validator) -> None:
    assert list(validator.iter_errors(VALID_RECORD)) == []


def test_a_record_with_optional_fields_validates(validator: Draft202012Validator) -> None:
    record = _record(
        comment_id=123,
        line_end=45,
        outcome="rejected",
        outcome_source="reaction",
        reject_reason="false_positive",
        cost={"usd": 0.12, "input_tokens": 1000, "output_tokens": 200},
    )
    record["fingerprint"]["provider"] = "anthropic"
    record["fingerprint"]["effort"] = "high"
    assert list(validator.iter_errors(record)) == []


def test_an_unknown_outcome_is_rejected(validator: Draft202012Validator) -> None:
    assert list(validator.iter_errors(_record(outcome="approved")))


def test_a_missing_fingerprint_is_rejected(validator: Draft202012Validator) -> None:
    record = _record()
    del record["fingerprint"]
    errors = list(validator.iter_errors(record))
    assert any("fingerprint" in error.message for error in errors)


@pytest.mark.parametrize(
    "head_sha",
    [
        "0123456789ABCDEF0123456789ABCDEF01234567",  # upper-case hex
        "0123456789abcdef0123456789abcdef0123456",  # 39 characters
        "g123456789abcdef0123456789abcdef01234567",  # non-hex character
    ],
)
def test_a_non_hex_head_sha_is_rejected(validator: Draft202012Validator, head_sha: str) -> None:
    assert list(validator.iter_errors(_record(head_sha=head_sha)))


def test_an_extra_top_level_key_is_rejected(validator: Draft202012Validator) -> None:
    errors = list(validator.iter_errors(_record(unexpected="value")))
    assert any("unexpected" in error.message for error in errors)
