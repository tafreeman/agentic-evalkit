"""Data models describing a run's plan (the manifest) and its actual results (design §5.6)."""

from datetime import datetime

from pydantic import Field, JsonValue, model_validator

from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.datasets import ContaminationMetadata, DatasetRef, ResolvedDataset
from agentic_evalkit.models.execution import NormalizedExecutionResult
from agentic_evalkit.models.grades import GradeResult
from agentic_evalkit.models.samples import EvalSample


class DatasetSelection(FrozenModel):
    """Which rows of a resolved dataset a run actually pulls from.

    Attributes:
        offset: How many rows to skip from the start.
        limit: The maximum number of rows to use. Left unset to mean "no
            limit."
        filter: An optional filter expression narrowing which rows count.
    """

    offset: int = 0
    limit: int | None = None
    filter: str | None = None


class SamplingPolicy(FrozenModel):
    """Seed, temperature, and repeat-attempt policy for a run.

    Attributes:
        seed: The random seed to use, for reproducibility. Left unset to
            mean "no fixed seed."
        temperature: The sampling temperature to request from the system
            being evaluated, if it supports one.
        attempts: How many times to run each sample. This field is only a
            *declaration* of intent, not the value the runner actually
            uses -- :attr:`EvalRunManifest.attempts` is canonical (it's
            what :class:`~agentic_evalkit.runner.EvalRunner` actually
            reads). The two are required to agree --
            :class:`EvalRunManifest` enforces this with a validator -- so
            this field can never silently diverge from the value that
            governs execution.
    """

    seed: int | None = None
    temperature: float | None = None
    attempts: int = 1


class EvalRunManifest(FrozenModel):
    """Pins down every input that has to be reproducible for a run (design §5.6).

    A manifest is what makes two runs comparable at all: two runs can only
    be meaningfully compared (design §10) when their dataset revision,
    adapter, grader, target policy, and sampling policy all line up. This
    class is the record of exactly what those were for one particular run.

    Attributes:
        run_name: A human-readable name for this run.
        dataset_ref: Which dataset this run asks for (see ``DatasetRef``).
        revision_policy: A description of how the dataset's revision
            should be resolved (e.g. pin to an exact version, or allow
            latest).
        adapter: The name (and version) of the benchmark adapter used to
            turn raw dataset rows into ``EvalSample`` objects.
        grader: The name (and version) of the grader used to score this
            run.
        target_name: A human-readable name identifying the system being
            evaluated.
        target_fingerprint_policy: How strictly ``target_fingerprint``
            below is expected to be enforced, e.g. ``"required"``.
        target_fingerprint: A hash pinning the exact, resolved
            configuration of the system being evaluated, computed once
            that system is resolved (see the inline note below for how
            this relates to ``target_fingerprint_policy``).
        selection: Which rows of the dataset this run pulls from (see
            ``DatasetSelection``).
        sampling: The seed/temperature/attempts policy this run declares
            (see ``SamplingPolicy``).
        attempts: The real, canonical number of times each sample is run
            (see the inline note below for how this differs from
            ``sampling.attempts``).
        timeout_seconds: How long to allow each attempt to run before
            giving up on it. Left unset to mean "no explicit timeout."
        concurrency: How many samples to run at once.
        artifact_policy: Caller-declared configuration for how this run's
            artifacts (generated files, spilled output, etc.) should be
            handled. Carried on the manifest as a record of what was
            requested.
        redaction_policy: Caller-declared configuration for how secrets
            should be redacted for this run. Carried on the manifest as a
            record of what was requested.
        environment_fingerprint: A hash identifying the interpreter,
            platform, and installed package versions this run executed
            under (see
            ``agentic_evalkit.provenance.compute_environment_fingerprint``).
        code_fingerprint: A hash identifying which build of this framework
            produced this run (see
            ``agentic_evalkit.provenance.compute_code_fingerprint``).
        baseline_compatibility_rules: Caller-declared rules for what
            should count as "compatible" when comparing this run against
            a baseline later (e.g. "dataset revision must match exactly").
            This is a per-manifest declaration written ahead of time, not
            something this codebase currently reads or enforces on its
            own.
        contamination: What's known about this dataset's risk of having
            leaked into a model's training data (see ADR-0013 and
            ``ContaminationMetadata`` in ``models/datasets.py``), copied
            onto the manifest so it travels with the run's report. See
            the inline note below for why it's left out of
            ``provenance_field_names``.
    """

    run_name: str
    dataset_ref: DatasetRef
    revision_policy: str | None = None
    adapter: str
    grader: str
    target_name: str
    target_fingerprint_policy: str | None = None
    #: Pins down exactly which configuration of the system being evaluated
    #: produced this run: a sha256 hash (see
    #: :func:`agentic_evalkit.provenance.compute_target_fingerprint`)
    #: computed from that system's settings once it's actually resolved.
    #: Where ``target_fingerprint_policy`` above says *how strictly* this
    #: should be enforced (e.g. ``"required"``), this field carries the
    #: real, computed value.
    target_fingerprint: str | None = None
    selection: DatasetSelection = Field(default_factory=DatasetSelection)
    sampling: SamplingPolicy = Field(default_factory=SamplingPolicy)
    #: The real number of times each sample gets run:
    #: :class:`~agentic_evalkit.runner.EvalRunner` reads this field, not
    #: ``sampling.attempts`` above. A validator below double-checks the
    #: two always match, so ``sampling.attempts`` can never quietly say
    #: one thing while this field -- the one that actually controls
    #: execution -- says another.
    attempts: int = 1
    timeout_seconds: float | None = None
    concurrency: int = 1
    artifact_policy: dict[str, JsonValue] = Field(default_factory=dict)
    redaction_policy: dict[str, JsonValue] = Field(default_factory=dict)
    environment_fingerprint: str | None = None
    code_fingerprint: str | None = None
    baseline_compatibility_rules: dict[str, JsonValue] = Field(default_factory=dict)
    #: How much risk there is that the system being evaluated already saw
    #: this dataset during training (ADR-0013), carried on the manifest so
    #: that risk label travels through into the run report's
    #: ``resolved_dataset`` too. This is purely informative -- it's
    #: deliberately left out of ``provenance_field_names`` below, so it
    #: never affects whether two runs count as comparable.
    contamination: ContaminationMetadata | None = None

    @classmethod
    def provenance_field_names(cls) -> frozenset[str]:
        """Which manifest fields must match for two runs to count as comparable (design §10).

        ``environment_fingerprint`` and ``code_fingerprint`` were added to
        this set later, by ADR-0015. Both are still fully part of "must
        match by default" here -- ``compare_runs``'s keyword-only
        ``allow_cross_environment`` option lets a caller explicitly waive a
        mismatch on just these two fields for one specific comparison, but
        that's a deliberate, visible opt-out for that one call, not a
        reason these two fields are any less important the rest of the
        time.
        """
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
                "environment_fingerprint",
                "code_fingerprint",
            }
        )

    @model_validator(mode="after")
    def _validate_attempts_agree(self) -> "EvalRunManifest":
        """Reject a manifest where the two attempt-count fields disagree with each other.

        ``sampling.attempts`` and ``attempts`` both exist to say the same
        thing, but only ``attempts`` is the one the runner actually reads
        (it never looks at ``sampling.attempts``). Since both default to
        ``1``, this check only fires when someone has explicitly set them
        to two different values -- the common case, where neither is set
        and both stay at their default, never trips it.
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
    """One sample's full journey: the sample, what happened running it, and its grade (if graded).

    Attributes:
        sample: The sample that was run.
        execution: What happened when the system being evaluated ran this
            sample (see ``NormalizedExecutionResult``).
        grade: How this sample's execution was scored, if grading ran.
            Left unset when execution itself failed, timed out, or was
            cancelled before grading ever got a chance to run.
    """

    sample: EvalSample
    execution: NormalizedExecutionResult
    grade: GradeResult | None = None


class RunSummary(FrozenModel):
    """A run's outcomes, tallied into separate counters instead of one combined score (design §10).

    Every kind of outcome gets its own counter, counted independently, so
    that an operational problem (the harness erroring out, a timeout, a
    cancellation, a grader that couldn't be trusted here) can never get
    mixed in with, or mistaken for, the system being evaluated actually
    attempting a task and getting it wrong.

    Attributes:
        total: How many samples this run covered in total.
        passed: How many samples passed grading.
        failed: How many samples were graded and did not pass.
        partial: How many samples received partial credit.
        errors: How many samples hit an operational problem (in either
            execution or grading) rather than being cleanly graded.
        timeouts: How many samples timed out during execution.
        cancelled: How many samples were cancelled before finishing.
        abstained: How many samples a grader declined to render a verdict
            on.
        unavailable: How many samples a grader couldn't be trusted to
            grade at all (for example, an uncalibrated judge).
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
    """The complete record of one evaluation run, evidence and all (design §5.6).

    ``samples`` is a tuple (immutable), so a run whose results grow over
    time (for example, one that streams results in, or one that gets
    resumed) is handled by building a brand-new ``EvalRunResult`` --
    via ``model_copy(update=...)`` with a longer tuple and a refreshed
    ``summary`` -- rather than editing this instance's ``samples`` in
    place. Nothing on this model assumes or hard-codes that a run is
    "finished"; that's simply whatever ``finished_at`` and the caller's
    own bookkeeping say it is.

    Attributes:
        run_id: This run's unique ID.
        manifest: Every input that was pinned down for this run (see
            ``EvalRunManifest``).
        resolved_dataset: The exact dataset version this run actually used
            (see ``ResolvedDataset``).
        samples: Every sample's outcome so far (see ``SampleResult``).
        summary: The tallied outcome counts for ``samples`` so far (see
            ``RunSummary``).
        started_at: When this run began.
        finished_at: When this run finished. Left unset while the run is
            still in progress.
    """

    run_id: str
    manifest: EvalRunManifest
    resolved_dataset: ResolvedDataset
    samples: tuple[SampleResult, ...] = ()
    summary: RunSummary = Field(default_factory=RunSummary)
    started_at: datetime
    finished_at: datetime | None = None
