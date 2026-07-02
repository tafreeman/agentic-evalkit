"""Immutable contracts for typed evaluation samples (design §5.3)."""

from pydantic import Field, JsonValue

from agentic_evalkit.models.base import FrozenModel


class GraderSpec(FrozenModel):
    """Names the grader a sample should be evaluated with, and how."""

    name: str
    grader_type: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    hard_gate: bool = False


class EvalSample(FrozenModel):
    """A typed, benchmark-adapter-projected unit of evaluation (design §5.3).

    Provider-native records never flow directly into execution or grading;
    a ``BenchmarkAdapter`` always produces this contract from a
    ``SourceRecord`` first.
    """

    sample_id: str
    input: dict[str, JsonValue]
    reference: str | None = None
    expected_artifacts: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()
    source_row_id: str | None = None
    source_digest: str
    adapter: str
    allowed_execution_policy: dict[str, JsonValue] = Field(default_factory=dict)
    grader: GraderSpec | None = None
