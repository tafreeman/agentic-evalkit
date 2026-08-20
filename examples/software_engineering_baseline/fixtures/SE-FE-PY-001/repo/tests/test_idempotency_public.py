from src.idempotency import CommandLedger, IdempotencyConflict


def test_ledger_starts_empty() -> None:
    ledger = CommandLedger()
    assert ledger._entries == {}


def test_conflict_is_a_value_error() -> None:
    assert issubclass(IdempotencyConflict, ValueError)
