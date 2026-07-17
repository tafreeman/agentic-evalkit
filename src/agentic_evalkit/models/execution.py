"""Data models for one request to run the system under test, and its result in a standard shape.

Design §5.4.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue

from agentic_evalkit.models.base import FrozenModel


class ExecutionStatus(StrEnum):
    """How the attempt to run the system went, separate from whether its answer was correct (§5.4).

    This describes the *plumbing* of running one attempt, not the quality
    of what came back: whether the answer was actually right or wrong is a
    completely separate question, answered later by ``GradeStatus``. It's
    a fixed set of named outcomes rather than a plain boolean or free-form
    string (ADR-0002), so callers can tell exactly what kind of
    non-success happened instead of everything collapsing into "it didn't
    work."

    - ``COMPLETED``: the attempt ran end-to-end and produced a result.
    - ``FAILED``: running the attempt itself broke down (for example, the
      system being evaluated crashed or reported its own internal error).
      This is treated as an operational problem, not a graded task
      failure, and it never reaches a grader.
    - ``TIMEOUT``: the attempt didn't finish within its allotted time.
    - ``CANCELLED``: the attempt was deliberately stopped before finishing
      (e.g. the whole run was cancelled).
    - ``ERROR``: something went wrong in this framework's own harness
      while running the attempt, as opposed to the system being evaluated
      failing on its own.

    ``FAILED`` and ``ERROR`` are both counted as operational problems in
    run statistics, rather than as the system being evaluated getting the
    task wrong -- keeping "our plumbing broke" separate from "the system
    tried and got it wrong" is what stops an infrastructure hiccup from
    silently deflating an accuracy score (ADR-0008).
    """

    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ERROR = "error"


class ExecutionRequest(FrozenModel):
    """A single attempt request sent to an ``ExecutionTarget``.

    Attributes:
        sample_id: Which sample this attempt is for.
        attempt: Which attempt number this is, for samples run more than
            once (see ``SamplingPolicy.attempts``).
        input: The actual question or task being sent to the system being
            evaluated -- copied from ``EvalSample.input``.
        timeout_seconds: How long to allow this attempt to run before
            giving up on it. Left unset to mean "no explicit timeout."
        trace_id: An ID for correlating this request with an external
            tracing/observability system, if one is in use.
    """

    sample_id: str
    attempt: int
    input: dict[str, JsonValue]
    timeout_seconds: float | None = None
    trace_id: str | None = None


class NormalizedExecutionResult(FrozenModel):
    """A system's output, converted into one standard shape every grader/report can rely on (§5.4).

    Every ``ExecutionTarget`` implementation (a plain callable, an HTTP
    endpoint, a subprocess, or something a caller wrote themselves) can
    differ wildly in what it's able to report back. This model is the
    common shape all of those get converted into, so nothing downstream
    needs to know which kind of target actually produced a given result.
    Every optional field represents data a particular target may or may
    not be able to supply; when a field is absent, it means "this target
    didn't report this," not "the value is empty."

    Attributes:
        sample_id: Which sample this is the result for.
        attempt: Which attempt number this result is for.
        output: The system's main answer, if it produced one.
        structured_output: A structured, schema-shaped version of the
            answer, for systems that support returning one, separate from
            the free-form ``output``.
        artifacts: Any extra files or data the system produced along the
            way (e.g. generated files), keyed by name.
        tool_calls: A record of any tools or functions the system invoked
            while producing its answer, if it reports them.
        trace_refs: Pointers (IDs or URLs) to a more detailed execution
            trace stored elsewhere -- not the trace itself, just
            references to it.
        latency_ms: How long this attempt took to run, in milliseconds.
        input_tokens: How many tokens the system consumed as input, if
            known.
        output_tokens: How many tokens the system generated as output, if
            known.
        cost_usd: The estimated dollar cost of this attempt, if known.
        model_name: Which underlying model the system used, if known and
            applicable.
        status: What happened when this attempt ran (see
            ``ExecutionStatus``) -- separate from whether the answer was
            correct.
        error: Structured details about what went wrong, when ``status``
            indicates something other than a clean completion.
        environment_metadata: Free-form details about the runtime
            environment this attempt executed in (e.g. HTTP
            request/response details for an HTTP target), for debugging
            and audit purposes.
        target_fingerprint: A hash identifying exactly which target
            configuration produced this result, so results can be traced
            back to precisely the setup that generated them.
        started_at: When this attempt began.
        finished_at: When this attempt ended.
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
