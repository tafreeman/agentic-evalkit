"""Frozen progress events emitted by :class:`agentic_evalkit.runner.EvalRunner`.

Each event names the run (and, where applicable, the sample) it belongs to
and carries a timestamp, so a caller-supplied event sink can build a live
progress view, structured log, or audit trail without polling run state.
Events are wire-safe: every field is a plain identifier, enum value, count,
or timestamp. No event carries raw target/harness secrets, request headers,
or unredacted output payloads -- large or sensitive data is stored through
:class:`agentic_evalkit.artifacts.ArtifactStore` and referenced by digest
instead (plan Task 11, Step 4).
"""

from __future__ import annotations

from datetime import datetime

from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.execution import ExecutionStatus
from agentic_evalkit.models.grades import GradeStatus


class RunStarted(FrozenModel):
    """Emitted once, before any sample in the run begins."""

    run_id: str
    run_name: str
    total_samples: int | None = None
    started_at: datetime


class DatasetResolved(FrozenModel):
    """Emitted once the run's dataset has been resolved and pinned."""

    run_id: str
    dataset_id: str
    dataset_revision: str
    resolved_at: datetime


class SampleStarted(FrozenModel):
    """Emitted when a sample's attempt begins execution."""

    run_id: str
    sample_id: str
    attempt: int
    started_at: datetime


class ExecutionCompleted(FrozenModel):
    """Emitted after a sample's attempt finishes executing (any status)."""

    run_id: str
    sample_id: str
    attempt: int
    status: ExecutionStatus
    completed_at: datetime


class GradeCompleted(FrozenModel):
    """Emitted after a completed execution has been graded."""

    run_id: str
    sample_id: str
    attempt: int
    status: GradeStatus
    completed_at: datetime


class SampleCompleted(FrozenModel):
    """Emitted once a sample has no further pipeline work (executed, graded or not)."""

    run_id: str
    sample_id: str
    attempt: int
    completed_at: datetime


class RunCompleted(FrozenModel):
    """Emitted once, after every sample in the run has completed."""

    run_id: str
    total_samples: int
    finished_at: datetime


class RunFailed(FrozenModel):
    """Emitted once, when the run stops due to an infrastructure-level failure.

    This is distinct from a sample-level ``error`` execution status: it marks
    that the run itself could not continue (e.g. dataset resolution failed,
    or the caller cancelled the run before it finished).
    """

    run_id: str
    error_type: str
    message: str
    failed_at: datetime


#: The union of every event type ``EvalRunner.run`` may pass to an event sink.
RunEvent = (
    RunStarted
    | DatasetResolved
    | SampleStarted
    | ExecutionCompleted
    | GradeCompleted
    | SampleCompleted
    | RunCompleted
    | RunFailed
)

#: Enumerable, reflection-friendly counterpart to the :data:`RunEvent` union:
#: every concrete event type the runner may emit, in emission order. The
#: redaction-enumeration contract asserts this tuple equals the union's
#: members exactly, so adding an event type to one but not the other fails CI
#: -- that binding is what makes structural contracts over events (e.g.
#: "every field is wire-safe") impossible to bypass (Story 2.2).
ALL_EVENT_TYPES: tuple[type[FrozenModel], ...] = (
    RunStarted,
    DatasetResolved,
    SampleStarted,
    ExecutionCompleted,
    GradeCompleted,
    SampleCompleted,
    RunCompleted,
    RunFailed,
)

__all__ = [
    "ALL_EVENT_TYPES",
    "DatasetResolved",
    "ExecutionCompleted",
    "GradeCompleted",
    "RunCompleted",
    "RunEvent",
    "RunFailed",
    "RunStarted",
    "SampleCompleted",
    "SampleStarted",
]
