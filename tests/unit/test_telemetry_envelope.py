"""Tests for ``targets/telemetry.envelope_from_records``.

The envelope is the contract between an instrumented target and the run
report: totals appear only at full coverage, missing data stays missing, and
malformed records fail loudly.
"""

from __future__ import annotations

import json

import pytest

from agentic_evalkit.targets.telemetry import COVERAGE_KEY, envelope_from_records


def _record(cid: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "call_id": cid,
        "attempt_ids": (f"{cid}-a1",),
        "input_tokens": 3,
        "output_tokens": 5,
        "duration_ms": 12.5,
        "served_model": "m-served",
        "requested_model": "m-asked",
        "tool_call_names": (),
    }
    base.update(overrides)
    return base


def test_full_coverage_envelope_is_complete_and_json_safe() -> None:
    envelope = envelope_from_records(
        [_record("c1"), _record("c2", input_tokens=7, output_tokens=11)],
        environment_metadata={"build": "test"},
        input_rate=1e-6,
        output_rate=2e-6,
    )
    assert envelope["input_tokens"] == 10
    assert envelope["output_tokens"] == 16
    assert envelope["latency_ms"] == 25.0
    assert envelope["cost_usd"] == pytest.approx(10e-6 + 16 * 2e-6)
    assert envelope["model_name"] == "m-served"
    assert envelope["trace_refs"] == ["call:c1", "call:c2", "attempt:c1-a1", "attempt:c2-a1"]
    coverage = envelope["environment_metadata"][COVERAGE_KEY]
    assert coverage == {
        "records": 2,
        "with_input_tokens": 2,
        "with_output_tokens": 2,
        "with_duration_ms": 2,
        "models": ["m-served"],
        "unattributed": 0,
    }
    # Caller metadata survives; the whole envelope round-trips as JSON.
    assert envelope["environment_metadata"]["build"] == "test"
    json.dumps(envelope)


def test_partial_usage_omits_totals_but_keeps_coverage() -> None:
    envelope = envelope_from_records(
        [_record("c1"), _record("c2", input_tokens=None, output_tokens=None)]
    )
    # Never a partial sum that would read as the sample's full consumption.
    assert "input_tokens" not in envelope
    assert "output_tokens" not in envelope
    assert "cost" not in envelope
    assert envelope["latency_ms"] == 25.0  # both calls reported a duration
    coverage = envelope["environment_metadata"][COVERAGE_KEY]
    assert coverage["with_input_tokens"] == 1
    assert coverage["records"] == 2


def test_cost_requires_full_coverage_and_both_rates() -> None:
    with pytest.raises(ValueError, match="together"):
        envelope_from_records([_record("c1")], input_rate=1e-6)
    # No usable coverage: no cost key at all (never a fabricated 0.0).
    envelope = envelope_from_records(
        [_record("c1", input_tokens=None, output_tokens=None)],
        input_rate=1e-6,
        output_rate=1e-6,
    )
    assert "cost_usd" not in envelope


def test_multiple_models_are_not_flattened_into_one_name() -> None:
    envelope = envelope_from_records(
        [_record("c1", served_model="model-a"), _record("c2", served_model="model-b")]
    )
    assert "model_name" not in envelope
    assert envelope["environment_metadata"][COVERAGE_KEY]["models"] == ["model-a", "model-b"]


def test_requested_model_attributes_when_served_is_absent() -> None:
    envelope = envelope_from_records([_record("c1", served_model=None)])
    assert envelope["model_name"] == "m-asked"


def test_tool_calls_and_attempt_ids_flatten() -> None:
    envelope = envelope_from_records(
        [
            _record("c1", tool_call_names=("search", "calc")),
            _record("c2", attempt_ids=("c2-a1", "c2-a2")),
        ]
    )
    assert envelope["tool_calls"] == [{"name": "search"}, {"name": "calc"}]
    assert "attempt:c2-a2" in envelope["trace_refs"]


def test_empty_records_yield_only_coverage() -> None:
    envelope = envelope_from_records([])
    assert set(envelope) == {"environment_metadata"}
    assert envelope["environment_metadata"][COVERAGE_KEY]["records"] == 0


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"call_id": ""}, "call_id"),
        ({"input_tokens": -1}, "input_tokens"),
        ({"input_tokens": True}, "input_tokens"),
        ({"duration_ms": float("nan")}, "duration_ms"),
        ({"served_model": ""}, "served_model"),
        ({"attempt_ids": ["ok", ""]}, "attempt_ids"),
        ({"tool_call_names": ["ok", 7]}, "tool_call_names"),
    ],
)
def test_malformed_records_fail_loudly(override: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        envelope_from_records([_record("c1", **override)])
