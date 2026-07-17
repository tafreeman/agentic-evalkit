"""The progress events :class:`agentic_evalkit.runner.EvalRunner` emits while a run is in progress.

Every event identifies which run (and, when relevant, which sample) it
belongs to, and carries a timestamp. This lets whatever is watching the run
-- a live progress bar, a structured log, an audit trail -- follow along as
it happens, instead of having to repeatedly ask "are we done yet?" by
checking run state.

Events are safe to send anywhere -- print them, log them, ship them over a
network -- because every field is a simple identifier, enum value, count, or
timestamp. No event ever carries a raw secret, an HTTP request header, or an
unredacted copy of a sample's output. If an output is large or might contain
sensitive data, it's stored separately in
:class:`agentic_evalkit.artifacts.ArtifactStore`, and the event only carries
a reference (a digest) pointing at it (plan Task 11, Step 4).
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
    """Emitted once, when the whole run has to stop because of a failure in our own infrastructure.

    This is different from a single sample getting an ``error`` execution
    status. This event means the *entire run* couldn't continue at all --
    for example, the dataset failed to load, or someone cancelled the run
    before it finished -- as opposed to the AI system under test failing on
    one particular sample while the rest of the run keeps going.
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

#: The same set of event types as the :data:`RunEvent` union above, but as a
#: plain tuple instead of a type union -- listed in the order the runner
#: emits them. A type union is great for static type-checking, but you can't
#: loop over it at runtime to check something about "every event type"; this
#: tuple exists so test code can do exactly that. A test (part of what we
#: call the redaction-enumeration contract) checks that this tuple always
#: contains exactly the same event types as the ``RunEvent`` union above --
#: so if someone adds a new event type to one but forgets the other, that
#: test fails in CI. That's what guarantees automated checks like "every
#: event field is safe to log or send over the wire" can never silently skip
#: a newly added event type (Story 2.2).
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
