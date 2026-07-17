"""Data models describing one thing to evaluate, and how to grade it (design §5.3)."""

from pydantic import Field, JsonValue

from agentic_evalkit.models.base import FrozenModel


class GraderSpec(FrozenModel):
    """Declares which grader a sample should be evaluated with, and how it should be configured.

    Attributes:
        name: The grader's own name, e.g. ``"normalized-exact@1"`` --
            matches the name a grader records on the ``GradeResult`` it
            produces, so you can trace a sample's declared grader back to
            the result that actually graded it.
        grader_type: A free-text label for the *category* of grader this
            is (for example ``"objective"``, ``"authoritative"``, or
            ``"composite"``) -- useful for humans and reports skimming
            results, not a fixed, enforced vocabulary.
        parameters: Grader-specific configuration values, passed through
            as-is to whatever grader ``name`` refers to.
        hard_gate: Whether this sample's grade is meant to be able to
            block a release outright, rather than being purely
            informational. This is the sample's own *declaration* of
            intent; whether a given grader run actually honors it depends
            on how that grader is wired up.
    """

    name: str
    grader_type: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    hard_gate: bool = False


class EvalSample(FrozenModel):
    """One evaluation task, in its final, standardized shape (design §5.3).

    A raw row from a dataset provider is never handed straight to the
    system being evaluated or to a grader. Instead, a ``BenchmarkAdapter``
    always converts a ``SourceRecord`` into an ``EvalSample`` first -- this
    class is that converted, standardized shape every later pipeline stage
    actually works with.

    Attributes:
        sample_id: A unique ID for this sample across the whole dataset --
            no two samples share one.
        input: The actual question or task handed to the system being
            evaluated.
        reference: The known-correct answer, if there's a single simple
            one to compare against. Left unset when correctness instead
            depends on something more elaborate (running tests, checking a
            structured output) rather than a plain text match.
        expected_artifacts: Extra data describing what a correct answer
            needs to produce, beyond a simple text match -- for example, a
            test patch a code change is expected to pass. This is
            grading-time reference material, not something shown to the
            system being evaluated.
        metadata: Free-form, descriptive information about this sample
            (e.g. where it came from, difficulty tags) that isn't used to
            grade it directly.
        tags: Short labels for filtering or grouping samples, e.g. by
            topic or difficulty.
        source_row_id: The ID of the original raw row (see
            ``SourceRecord.row_id``) this sample was converted from, so you
            can trace it back to the source data if needed.
        source_digest: The integrity hash of that original raw row (see
            ``SourceRecord.digest``), carried forward so you can confirm
            this sample was built from exactly that row and nothing else.
        adapter: The name (and version) of the ``BenchmarkAdapter`` that
            produced this sample, e.g. ``"gsm8k@1"``.
        allowed_execution_policy: Constraints on how the system being
            evaluated is allowed to run this sample, e.g. a maximum number
            of attempts.
        grader: Which grader this sample should be evaluated with, and how
            (see ``GraderSpec``). Left unset when no specific grader is
            declared.
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
