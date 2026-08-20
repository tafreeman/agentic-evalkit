import json
from collections.abc import Callable
from typing import Any, TypeVar, cast


T = TypeVar("T")


class IdempotencyConflict(ValueError):
    pass


class CommandLedger:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, Any]] = {}

    def execute(self, key: str, payload: Any, handler: Callable[[], T]) -> T:
        fingerprint = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        existing = self._entries.get(key)
        if existing is not None:
            previous_fingerprint, result = existing
            if previous_fingerprint != fingerprint:
                raise IdempotencyConflict(f"key {key!r} was already used for another payload")
            return cast(T, result)
        result = handler()
        self._entries[key] = (fingerprint, result)
        return result
