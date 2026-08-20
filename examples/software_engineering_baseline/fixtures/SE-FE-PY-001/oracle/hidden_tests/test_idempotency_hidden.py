import pytest

from src.idempotency import CommandLedger, IdempotencyConflict


def test_first_call_executes_and_repeat_replays() -> None:
    ledger = CommandLedger()
    calls = 0

    def handler() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"sequence": calls}

    assert ledger.execute("k1", {"amount": 10}, handler) == {"sequence": 1}
    assert ledger.execute("k1", {"amount": 10}, handler) == {"sequence": 1}
    assert calls == 1


def test_mapping_order_does_not_create_a_false_conflict() -> None:
    ledger = CommandLedger()
    ledger.execute("k1", {"a": 1, "b": [2, 3]}, lambda: "stored")
    assert ledger.execute("k1", {"b": [2, 3], "a": 1}, lambda: "wrong") == "stored"


def test_different_payload_for_same_key_conflicts_without_calling_handler() -> None:
    ledger = CommandLedger()
    ledger.execute("k1", {"amount": 10}, lambda: "stored")
    called = False

    def handler() -> str:
        nonlocal called
        called = True
        return "new"

    with pytest.raises(IdempotencyConflict):
        ledger.execute("k1", {"amount": 11}, handler)
    assert called is False


def test_handler_failure_is_not_cached() -> None:
    ledger = CommandLedger()

    def fail() -> str:
        raise RuntimeError("transient")

    with pytest.raises(RuntimeError, match="transient"):
        ledger.execute("k1", {"amount": 10}, fail)
    assert ledger.execute("k1", {"amount": 10}, lambda: "recovered") == "recovered"


def test_none_is_a_replayable_result() -> None:
    ledger = CommandLedger()
    calls = 0

    def handler() -> None:
        nonlocal calls
        calls += 1

    assert ledger.execute("k1", {}, handler) is None
    assert ledger.execute("k1", {}, handler) is None
    assert calls == 1
