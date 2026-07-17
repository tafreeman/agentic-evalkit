"""The public data models different parts of agentic-evalkit use to pass information around.

See design §5 (`docs/specs/2026-07-02-agentic-evalkit-design.md`) and
ADR-0002 (`docs/adr/0002-immutable-versioned-contracts.md`) for the full
reasoning. Every model in this package is immutable once built, rejects
any field it doesn't recognize, carries an explicit ``schema_version`` so
old saved data stays readable as the package evolves, and never performs
file or network access itself.
"""

from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.datasets import (
    ContaminationMetadata,
    ContaminationStatus,
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
    "ContaminationMetadata",
    "ContaminationStatus",
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
