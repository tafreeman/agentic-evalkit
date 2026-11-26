"""Wire contracts for the integrity harness (ADR-0027, ADR-0002).

Two files cross the boundary between EvalKit and whatever runs the agent:

- ``tool_calls.jsonl`` -- one :class:`ToolCallRecord` per line, written by the
  runner (``schemas/tool-call-record.v1.schema.json``).
- ``integrity_report.json`` -- one :class:`IntegrityReport`, written by EvalKit.

Neither adds a ``GradeStatus`` value: a flagged run is reported as
``UNAVAILABLE`` by the grader, never as a fail (ADR-0008).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue, model_validator

from agentic_evalkit.models.base import FrozenModel

__all__ = [
    "CheckStatus",
    "InstanceIntegrity",
    "IntegrityFlag",
    "IntegrityReport",
    "IntegrityStage",
    "PreflightCheck",
    "Severity",
    "ToolCallRecord",
    "severity_rank",
    "worst_severity",
]


class Severity(StrEnum):
    """How bad a flag is. Ordered ``CLEAN < SUSPECT < EXPLOIT`` (see :func:`severity_rank`)."""

    CLEAN = "clean"
    SUSPECT = "suspect"
    EXPLOIT = "exploit"


_RANK = {Severity.CLEAN: 0, Severity.SUSPECT: 1, Severity.EXPLOIT: 2}


def severity_rank(severity: Severity) -> int:
    """Numeric order of a severity; a higher number is worse."""
    return _RANK[severity]


def worst_severity(severities: list[Severity] | tuple[Severity, ...]) -> Severity:
    """The worst of ``severities``; ``CLEAN`` when there are none."""
    return max(severities, key=severity_rank, default=Severity.CLEAN)


class IntegrityStage(StrEnum):
    """Which of the three v1 stages raised a flag."""

    PREFLIGHT = "preflight"
    DIFF_GUARD = "diff_guard"
    TRAJECTORY = "trajectory"


class CheckStatus(StrEnum):
    """Outcome of one pre-flight check.

    ``ERROR`` means the check could not run, which is not the same as the
    environment failing it: it still keeps an instance out of the clean rate,
    because nothing was verified, but a reader can tell the two apart.
    """

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ToolCallRecord(FrozenModel):
    """One tool call from an agent run (``schemas/tool-call-record.v1.schema.json``).

    ``arguments`` and ``result`` are kept verbatim because the detectors need
    the command lines and fetched URLs; redaction happens when evidence is
    built (:func:`agentic_evalkit.integrity.flags.make_flag`).
    """

    run_id: str
    sample_id: str
    call_id: str
    step: str | None = None
    tool: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    result: JsonValue = None
    gate_decision: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class IntegrityFlag(FrozenModel):
    """One suspected integrity violation, with the line that triggered it."""

    instance_id: str
    stage: IntegrityStage
    rule: str
    severity: Severity
    evidence: str
    call_id: str | None = None

    @model_validator(mode="after")
    def _flag_is_not_clean(self) -> IntegrityFlag:
        if self.severity is Severity.CLEAN:
            raise ValueError("a flag is never clean; leave the instance unflagged instead")
        return self


class PreflightCheck(FrozenModel):
    """The result of one pre-flight check, recorded whether it passed or not."""

    name: str
    status: CheckStatus
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASSED


class InstanceIntegrity(FrozenModel):
    """Everything the audit found for one instance."""

    instance_id: str
    severity: Severity = Severity.CLEAN
    raw_passed: bool = False
    flags: tuple[IntegrityFlag, ...] = ()

    @model_validator(mode="after")
    def _severity_is_the_worst_flag(self) -> InstanceIntegrity:
        if self.severity is not worst_severity([flag.severity for flag in self.flags]):
            raise ValueError("severity must equal the worst flag severity")
        return self


class IntegrityReport(FrozenModel):
    """``integrity_report.json``: raw vs clean pass rate plus per-instance flags.

    ``total`` is the denominator of both rates and never changes with
    flagging, so ``clean_pass_rate <= raw_pass_rate`` always holds
    (spec lemma L2): flagging only moves instances from passed to not-clean.

    ``harness_version``, ``context_digest``, ``tool_calls_digest`` and
    ``rules`` record what the audit ran against, so a changed harness or input
    is visible in the file (a changed harness means a new run directory).
    """

    run_id: str
    created_at: datetime
    total: int = Field(ge=0)
    raw_passed: int = Field(ge=0)
    clean_passed: int = Field(ge=0)
    raw_pass_rate: float | None = None
    clean_pass_rate: float | None = None
    harness_version: str | None = None
    context_digest: str | None = None
    tool_calls_digest: str | None = None
    rules: tuple[str, ...] = ()
    preflight: tuple[PreflightCheck, ...] = ()
    instances: tuple[InstanceIntegrity, ...] = ()

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> IntegrityReport:
        if self.clean_passed > self.raw_passed or self.raw_passed > self.total:
            raise ValueError("require clean_passed <= raw_passed <= total")
        return self
