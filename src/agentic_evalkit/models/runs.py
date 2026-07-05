"""Immutable contracts for run manifests and run results (design §5.6)."""

from datetime import datetime

from pydantic import Field, JsonValue, model_validator

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
    """Seed, temperature, and repeated-attempt policy for a run.

    ``attempts`` here is a *policy declaration*, not the value the runner
    consumes: :attr:`EvalRunManifest.attempts` is canonical (it is what
    :class:`~agentic_evalkit.runner.EvalRunner` actually reads). The two
    must agree -- :class:`EvalRunManifest` enforces this with a validator --
    so this field can never silently diverge from the value that governs
    execution.
    """

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
    #: Pins the actual resolved target identity: a sha256 digest (see
    #: :func:`agentic_evalkit.provenance.compute_target_fingerprint`) over
    #: the canonical target config, computed once the target is resolved.
    #: ``target_fingerprint_policy`` states *how* this field is enforced
    #: (e.g. "required"); this field carries the enforced value itself.
    target_fingerprint: str | None = None
    selection: DatasetSelection = Field(default_factory=DatasetSelection)
    sampling: SamplingPolicy = Field(default_factory=SamplingPolicy)
    #: Canonical repeated-attempt count: :class:`~agentic_evalkit.runner.EvalRunner`
    #: reads this field, not ``sampling.attempts``. The two are validated
    #: equal below so ``sampling.attempts`` can never silently disagree
    #: with the value that actually governs execution.
    attempts: int = 1
    timeout_seconds: float | None = None
    concurrency: int = 1
    artifact_policy: dict[str, JsonValue] = Field(default_factory=dict)
    redaction_policy: dict[str, JsonValue] = Field(default_factory=dict)
    environment_fingerprint: str | None = None
    code_fingerprint: str | None = None
    baseline_compatibility_rules: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def provenance_field_names(cls) -> frozenset[str]:
        """The comparability-relevant manifest provenance fields (design section 10)."""
        return frozenset(
            {
                "adapter",
                "grader",
                "target_name",
                "target_fingerprint_policy",
                "target_fingerprint",
                "sampling.temperature",
                "sampling.seed",
                "attempts",
            }
        )

    @model_validator(mode="after")
    def _validate_attempts_agree(self) -> "EvalRunManifest":
        """Reject a manifest whose two attempt counts have silently diverged.

        ``sampling.attempts`` and ``attempts`` duplicate one concept; only
        ``attempts`` is canonical (the runner never reads
        ``sampling.attempts``). Both default to ``1``, so this only fires
        on an explicit, meaningful mismatch -- never on the common
        all-defaults case.
        """
        if self.sampling.attempts != self.attempts:
            raise ValueError(
                "sampling.attempts "
                f"({self.sampling.attempts}) and attempts ({self.attempts}) must be "
                "equal -- attempts is canonical (the runner reads it, not "
                "sampling.attempts); set both to the same value"
            )
        return self


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
