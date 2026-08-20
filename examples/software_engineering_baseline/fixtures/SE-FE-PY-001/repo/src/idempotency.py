"""In-memory command ledger used by the feature eval."""

from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")


class IdempotencyConflict(ValueError):
    """Raised when one key is reused for a different request payload."""


class CommandLedger:
    """Store completed command outcomes by idempotency key."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, Any]] = {}

    def execute(self, key: str, payload: Any, handler: Callable[[], T]) -> T:
        """Execute or replay a command according to the module contract."""
        raise NotImplementedError
