"""Immutable contracts for grading outcomes (design §5.5)."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue

from agentic_evalkit.models.base import FrozenModel


class GradeStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"
    ABSTAIN = "abstain"
    UNAVAILABLE = "unavailable"


class GradeResult(FrozenModel):
    """The outcome of grading one sample's execution (design §5.5).

    ``status`` is a ``GradeStatus`` rather than a boolean so that abstention,
    partial credit, error, and unavailable-capability outcomes remain
    distinguishable from a definitive pass or fail (ADR-0002).
    """

    sample_id: str
    grader: str
    grader_type: str | None = None
    status: GradeStatus
    score: float | None = None
    hard_gate: bool = False
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    rubric_id: str | None = None
    oracle_provenance: dict[str, JsonValue] = Field(default_factory=dict)
    judge_calibration_ref: str | None = None
    created_at: datetime
