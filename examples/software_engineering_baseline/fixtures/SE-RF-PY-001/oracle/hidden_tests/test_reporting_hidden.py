import inspect

import pytest

from src.reporting import summarize


def test_preserves_first_seen_kind_order_and_totals() -> None:
    events = [
        {"kind": "test", "ok": True, "duration_ms": 5},
        {"kind": "build", "ok": False, "duration_ms": 9},
        {"kind": "test", "ok": False, "duration_ms": 7},
    ]
    assert summarize(events) == [
        {"kind": "test", "count": 2, "failures": 1, "duration_ms": 12},
        {"kind": "build", "count": 1, "failures": 1, "duration_ms": 9},
    ]


def test_non_positive_durations_do_not_reduce_total() -> None:
    events = [
        {"kind": "test", "ok": True, "duration_ms": -3},
        {"kind": "test", "ok": True, "duration_ms": 0},
    ]
    assert summarize(events) == [
        {"kind": "test", "count": 2, "failures": 0, "duration_ms": 0}
    ]


@pytest.mark.parametrize("missing", ["kind", "ok", "duration_ms"])
def test_missing_required_key_remains_key_error(missing: str) -> None:
    event = {"kind": "test", "ok": True, "duration_ms": 1}
    del event[missing]
    with pytest.raises(KeyError):
        summarize([event])


def test_public_signature_is_unchanged() -> None:
    assert tuple(inspect.signature(summarize).parameters) == ("events",)
