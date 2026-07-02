"""Immutable contracts for run manifests and run results (design §5.6)."""

from datetime import datetime

from pydantic import Field, JsonValue

from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.datasets import DatasetRef, ResolvedDataset
from agentic_evalkit.models.execution import NormalizedExecutionResult
from agentic_evalkit.models.grades import GradeResult
from agentic_evalkit.models.samples import EvalSample


class DatasetSelection(FrozenModel):
    """Which rows of a resolved dataset a run pulls from."""

    offset: int = 0
    limit: int | None = None
    filter: str | None = None


class SamplingPolicy(FrozenModel):
    """Seed, temperature, and repeated-attempt policy for a run."""

    seed: int | None = None
    temperature: float | None = None
    attempts: int = 1


class EvalRunManifest(FrozenModel):
    """Pins every input that must be reproducible for a run (design §5.6).

    A manifest is the unit of comparability: two runs are only comparable
    (design §10) when their dataset revision, adapter, grader, target
    policy, and sampling policy line up.
    """

    run_name: str
    dataset_ref: DatasetRef
    revision_policy: str | None = None
    adapter: str
    grader: str
    target_name: str
    target_fingerprint_policy: str | None = None
    selection: DatasetSelection = Field(default_factory=DatasetSelection)
    sampling: SamplingPolicy = Field(default_factory=SamplingPolicy)
    attempts: int = 1
    timeout_seconds: float | None = None
    concurrency: int = 1
    artifact_policy: dict[str, JsonValue] = Field(default_factory=dict)
    redaction_policy: dict[str, JsonValue] = Field(default_factory=dict)
    environment_fingerprint: str | None = None
    code_fingerprint: str | None = None
    baseline_compatibility_rules: dict[str, JsonValue] = Field(default_factory=dict)


class SampleResult(FrozenModel):
    """One sample's full pipeline outcome: sample, execution, and grade.

    ``grade`` is optional because execution can fail, time out, or be
    cancelled before grading ever runs.
    """

    sample: EvalSample
    execution: NormalizedExecutionResult
    grade: GradeResult | None = None


class RunSummary(FrozenModel):
    """Separated outcome counts for a run (design §10).

    Every outcome category is counted independently so operational
    failures (errors, timeouts, cancellations, unavailable capabilities)
    can never masquerade as task failures.
    """

    total: int = 0
    passed: int = 0
    failed: int = 0
    partial: int = 0
    errors: int = 0
    timeouts: int = 0
    cancelled: int = 0
    abstained: int = 0
    unavailable: int = 0


class EvalRunResult(FrozenModel):
    """The complete, provenance-carrying outcome of an evaluation run (design §5.6).

    ``samples`` is a tuple, so growing a run's results over time (e.g. a
    streaming or resumed run) is expressed by building a new
    ``EvalRunResult`` via ``model_copy(update=...)`` with an extended tuple
    and an updated ``summary``, not by mutating this instance. No field on
    this model hard-codes finality.
    """

    run_id: str
    manifest: EvalRunManifest
    resolved_dataset: ResolvedDataset
    samples: tuple[SampleResult, ...] = ()
    summary: RunSummary = Field(default_factory=RunSummary)
    started_at: datetime
    finished_at: datetime | None = None
