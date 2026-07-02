"""Immutable contracts for execution requests and normalized results (design §5.4)."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue

from agentic_evalkit.models.base import FrozenModel


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ERROR = "error"


class ExecutionRequest(FrozenModel):
    """A single attempt request sent to an ``ExecutionTarget``."""

    sample_id: str
    attempt: int
    input: dict[str, JsonValue]
    timeout_seconds: float | None = None
    trace_id: str | None = None


class NormalizedExecutionResult(FrozenModel):
    """A target's output normalized to a host-neutral shape (design §5.4).

    Every optional field represents data a target may or may not supply;
    absence means "not reported by this target", not "empty".
    """

    sample_id: str
    attempt: int
    output: dict[str, JsonValue] | None = None
    structured_output: dict[str, JsonValue] | None = None
    artifacts: dict[str, JsonValue] = Field(default_factory=dict)
    tool_calls: tuple[dict[str, JsonValue], ...] = ()
    trace_refs: tuple[str, ...] = ()
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model_name: str | None = None
    status: ExecutionStatus
    error: dict[str, JsonValue] | None = None
    environment_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    target_fingerprint: str | None = None
    started_at: datetime
    finished_at: datetime
