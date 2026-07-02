"""Public, immutable wire contracts for agentic-evalkit.

See design §5 (`docs/specs/2026-07-02-agentic-evalkit-design.md`) and
ADR-0002 (`docs/adr/0002-immutable-versioned-contracts.md`). Every model
here is frozen, forbids unknown fields, carries an explicit
``schema_version``, and performs no I/O.
"""

from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.datasets import (
    DatasetRef,
    ResolvedDataset,
    SamplePage,
    SearchHit,
    SearchPage,
    SourceRecord,
)
from agentic_evalkit.models.execution import (
    ExecutionRequest,
    ExecutionStatus,
    NormalizedExecutionResult,
)
from agentic_evalkit.models.grades import GradeResult, GradeStatus
from agentic_evalkit.models.runs import (
    DatasetSelection,
    EvalRunManifest,
    EvalRunResult,
    RunSummary,
    SampleResult,
    SamplingPolicy,
)
from agentic_evalkit.models.samples import EvalSample, GraderSpec

__all__ = [
    "DatasetRef",
    "DatasetSelection",
    "EvalRunManifest",
    "EvalRunResult",
    "EvalSample",
    "ExecutionRequest",
    "ExecutionStatus",
    "FrozenModel",
    "GradeResult",
    "GradeStatus",
    "GraderSpec",
    "NormalizedExecutionResult",
    "ResolvedDataset",
    "RunSummary",
    "SamplePage",
    "SampleResult",
    "SamplingPolicy",
    "SearchHit",
    "SearchPage",
    "SourceRecord",
]
