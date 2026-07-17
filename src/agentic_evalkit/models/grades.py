"""Data models describing the outcome of grading one sample's execution (design §5.5)."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue

from agentic_evalkit.models.base import FrozenModel


class GradeStatus(StrEnum):
    """How one sample's execution was judged, once grading actually ran (design §5.5).

    This is deliberately a fixed set of named outcomes, not a plain
    pass/fail boolean (ADR-0002) -- a boolean can't represent partial
    credit, a grader declining to answer, a grader breaking while trying
    to grade, or a grader whose evidence isn't trustworthy enough to use
    here, without quietly losing information about which of those actually
    happened.

    - ``PASS``: the execution met the bar for this grader.
    - ``FAIL``: the execution did not meet the bar.
    - ``PARTIAL``: the execution partly met the bar (partial credit).
    - ``ERROR``: the grader itself broke while trying to grade this
      sample -- a problem with grading, not proof the sample failed, so
      it's never treated as a ``FAIL``.
    - ``ABSTAIN``: the grader declined to render a verdict at all (for
      example, an AI judge that said "I'm not going to answer this one").
    - ``UNAVAILABLE``: the grader couldn't be trusted to produce a
      meaningful verdict here (for example, an AI judge whose proof of
      accuracy doesn't clear the bar required for this decision -- see
      ``graders/judge.py``).
    """

    PASS = "pass"  # noqa: S105 -- grade status enum value, not a credential
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"
    ABSTAIN = "abstain"
    UNAVAILABLE = "unavailable"


class GradeResult(FrozenModel):
    """The outcome of grading one sample's execution (design §5.5).

    ``status`` is a ``GradeStatus`` rather than a plain pass/fail boolean,
    so that abstention, partial credit, a grader breaking, and "this
    grader can't be trusted here" outcomes all stay distinguishable from a
    clean pass or fail instead of collapsing into one bit of information
    (ADR-0002).

    Attributes:
        sample_id: Which sample this grade is for.
        grader: The name of the grader that produced this result, e.g.
            ``"normalized-exact@1"``.
        grader_type: A free-text category label for the grader (e.g.
            ``"objective"``, ``"composite"``), mirroring
            ``GraderSpec.grader_type``.
        status: The verdict this grader reached (see ``GradeStatus``).
        score: A numeric score, for graders that produce one. Left unset
            for graders whose verdict is categorical only.
        hard_gate: Whether this specific result is actually allowed to
            block a release outright, as opposed to being purely
            informational. Defaults to ``False``; a grader only sets this
            to ``True`` once its own reliability checks (if it has any)
            have passed -- see ``graders/judge.py`` for the
            calibrated-judge example.
        evidence: Supporting details explaining the verdict (e.g. a reason
            string, or specifics about what was checked), meant for a
            human reviewing the result later.
        artifact_refs: Pointers (IDs or paths) to any artifacts produced
            while grading (e.g. logs or generated files) -- references
            only, not the content itself.
        rubric_id: Which rubric this grade was scored against, for graders
            that use one. Left unset otherwise.
        oracle_provenance: Details about an authoritative external checker
            used to produce this grade, if one was used (for example,
            which test harness ran, and how) -- so the evidence trail
            shows the grade came from a real, authoritative check, not
            just this grader's own say-so.
        judge_calibration_ref: Which calibration record backs this grade,
            for results produced by an AI judge that's proven its
            reliability (see ``CalibrationArtifact`` in
            ``graders/judge.py``). Left unset when no calibration applies,
            including whenever ``hard_gate`` is ``False``.
        created_at: When this grade was produced.
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
