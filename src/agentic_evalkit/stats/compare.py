"""Checking whether two runs are safe to compare, and comparing them with a
paired bootstrap (design §10, ADR-0008, ADR-0015).

"Provenance" here means all the facts about exactly what was run and how:
which dataset revision and split, which adapter, which grader, which
target (system under test) and which exact build ("fingerprint") of it,
what sampling settings were used, and so on. Two runs are only safe to
compare if all of that matches -- otherwise you might be comparing, say, a
harder dataset split against an easier one, and mistaking the difference
in difficulty for a real improvement. So if any of that provenance
doesn't line up, the comparison must fail loudly with an explanation,
rather than silently reporting a difference ("delta") that's actually
misleading (design §10).

``compare_runs`` checks every provenance field listed in Task 12 Step 6
(dataset ID/revision/config/split, adapter, grader, target policy, target
fingerprint, sampling temperature/seed policy, attempt count), plus the
environment and code fingerprints that :mod:`agentic_evalkit.provenance`
computes (ADR-0015) -- a "fingerprint" here is just a hash or identifier
that uniquely marks one exact version of something (e.g. one exact build
of the code, or one exact version of the target system). If anything
doesn't match, it raises :class:`~agentic_evalkit.errors.IncompatibleRuns`
listing *every* mismatch it finds, not just the first one it happens to
hit. A target identified by name but with no pinned/recorded fingerprint
(``None``) is never treated as if it matches a pinned fingerprint on the
other run -- "we don't know what exact version this was" must never
silently pass as "these two match."

A caller who knowingly compares runs captured under different Python
interpreters, operating systems, or ``agentic-evalkit`` builds may pass
``allow_cross_environment=True`` to waive *only* a mismatch in
``environment_fingerprint`` and/or ``code_fingerprint`` -- and even then,
which field(s) got waived is recorded on
:attr:`ComparisonResult.waived_provenance_fields` rather than silently
swept under the rug (ADR-0015). No other provenance field can ever be
waived through this flag.

Once two runs are confirmed compatible, this "pairs up" their individual
results by matching ``(sample_id, attempt)`` -- i.e. it lines up each run's
result for "question 5, attempt 2" with the other run's result for that
same question and attempt, and compares only those matched pairs. If one
side is missing an attempt the other has, that pair is simply left out of
the comparison rather than counted as a failure. If there turn out to be
*zero* matching pairs at all, there's nothing to compute a difference
from, so ``compare_runs`` raises
:class:`~agentic_evalkit.errors.IncompatibleRuns` rather than reporting a
confident-looking "zero difference."

Otherwise, it computes the observed difference in paired success rates
(the "delta") and then estimates a 95% confidence interval around that
delta using a technique called "bootstrapping": we repeatedly draw random
samples, *with replacement*, from the paired results we already have, and
recompute the delta each time. Doing this many times shows us how much the
delta would plausibly wobble if we'd happened to run a slightly different
-- but similarly sized -- set of paired comparisons, and that spread
becomes our confidence interval. This uses a local ``random.Random(seed)``
instance (not shared global randomness), so that using the same seed
always reproduces the exact same sequence of random draws and therefore
the exact same result (ADR-0008: the bootstrap is seeded and
deterministic, i.e. reproducible).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from agentic_evalkit.errors import IncompatibleRuns
from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.grades import GradeStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentic_evalkit.models.runs import EvalRunManifest, EvalRunResult, SampleResult

__all__ = ["PROVENANCE_FIELDS_CHECKED", "ComparisonResult", "compare_runs"]

#: The manifest provenance checks that ``_describe_mismatches`` actually
#: performs, one row per field, as ``(declared field name, human-readable
#: label, function to read the value off a manifest, whether it's
#: waivable)``.
#:
#: This table is the single source of truth: :data:`PROVENANCE_FIELDS_CHECKED`
#: and :data:`_WAIVABLE_UNDER_CROSS_ENVIRONMENT` below are both computed
#: from these rows, rather than being separately hand-maintained lists that
#: could quietly drift out of sync with what the checks actually do. A test
#: (``tests/contract/test_provenance_drift.py``) compares this table
#: against :meth:`EvalRunManifest.provenance_field_names` and against the
#: set of fields ADR-0015 actually approved as waivable -- so if a field
#: gets added as provenance but someone forgets to add a row here (or the
#: reverse), or a row here is marked waivable beyond what ADR-0015 allows,
#: that test fails CI (tracked as R-004 P0).
_PROVENANCE_CHECKS: tuple[tuple[str, str, Callable[[EvalRunManifest], object], bool], ...] = (
    ("adapter", "adapter", lambda m: m.adapter, False),
    ("grader", "grader", lambda m: m.grader, False),
    ("target_name", "target name", lambda m: m.target_name, False),
    (
        "target_fingerprint_policy",
        "target fingerprint policy",
        lambda m: m.target_fingerprint_policy,
        False,
    ),
    ("target_fingerprint", "target fingerprint", lambda m: m.target_fingerprint, False),
    ("sampling.temperature", "sampling temperature", lambda m: m.sampling.temperature, False),
    ("sampling.seed", "sampling seed", lambda m: m.sampling.seed, False),
    ("attempts", "attempt count", lambda m: m.attempts, False),
    (
        "environment_fingerprint",
        "environment fingerprint",
        lambda m: m.environment_fingerprint,
        True,
    ),
    ("code_fingerprint", "code fingerprint", lambda m: m.code_fingerprint, True),
)

#: Derived from :data:`_PROVENANCE_CHECKS` row names -- see the table's note.
PROVENANCE_FIELDS_CHECKED: frozenset[str] = frozenset(name for name, _, _, _ in _PROVENANCE_CHECKS)

#: The only provenance fields that ``compare_runs(...,
#: allow_cross_environment=True)`` is allowed to waive on a mismatch
#: (ADR-0015). This is computed from the table's "waivable" column above,
#: so it's automatically a subset of :data:`PROVENANCE_FIELDS_CHECKED`.
#: These are the exact declared field names (matching
#: :data:`_PROVENANCE_CHECKS` rows), never the human-readable labels, so
#: that :attr:`ComparisonResult.waived_provenance_fields` can reuse them
#: as-is. ``tests/contract/test_provenance_drift.py`` locks this set down
#: to exactly the two fields ADR-0015 approved.
_WAIVABLE_UNDER_CROSS_ENVIRONMENT: frozenset[str] = frozenset(
    name for name, _, _, waivable in _PROVENANCE_CHECKS if waivable
)

_MIN_BOOTSTRAP_SAMPLES = 100
_MAX_BOOTSTRAP_SAMPLES = 10_000
_DEFAULT_BOOTSTRAP_SAMPLES = 1000
_LOWER_PERCENTILE = 2.5
_UPPER_PERCENTILE = 97.5


class ComparisonResult(FrozenModel):
    """The result of comparing two compatible runs: the observed difference
    in their paired success rates, plus a bootstrap confidence interval
    around it.

    ``estimate`` is the observed difference in success rate between the
    two runs (``right_pass_rate - left_pass_rate``), computed only over
    the paired subset of matching observations.
    ``lower_percentile``/``upper_percentile`` are the 2.5th and 97.5th
    percentiles of the bootstrap's resampled distribution of that
    difference -- together they form the 95% confidence interval: the
    range the true difference plausibly falls within, given the limited
    data we have (see the module docstring above for what "bootstrap"
    means here).
    """

    estimate: float
    lower_percentile: float
    upper_percentile: float
    paired_count: int
    """How many paired ``(sample_id, attempt)`` observations the difference
    and the bootstrap were computed over. When ``attempts > 1`` (each
    question attempted more than once), this can be larger than
    ``sample_count``, since one distinct sample can contribute up to one
    paired observation per attempt."""
    sample_count: int
    """How many *distinct* ``sample_id`` values are represented in
    ``paired_count``, regardless of how many attempts each one
    contributed. Unlike ``paired_count``, this is never multiplied up by
    the attempt count."""
    seed: int
    waived_provenance_fields: tuple[str, ...] = ()
    """Which provenance field names, if any, were waived because the
    caller passed ``allow_cross_environment=True`` (e.g.
    ``("environment_fingerprint",)``), listed in :data:`_PROVENANCE_CHECKS`
    order. Empty when nothing actually differed, or when the flag was
    never set. This field is additive and optional, keeping the wire
    format at ``schema_version = "1"`` (ADR-0002); see ADR-0015 for the
    policy that allows waiving these two fields."""


def _describe_mismatches(
    left: EvalRunResult, right: EvalRunResult, *, allow_cross_environment: bool = False
) -> tuple[list[str], list[str]]:
    """Return a pair: (mismatch descriptions that still block the
    comparison, names of fields that were waived instead).

    This compares the *resolved* dataset identity -- i.e. the actual,
    already-looked-up dataset each run drew from (design §10: "dataset
    revision, split, adapter, harness, grader, target policy, and sampling
    policy") -- rather than the original request that asked for a dataset
    (a ``DatasetRef``), since it's the resolved, actual dataset that's
    immutable and comparable, not the request that produced it. ``adapter``
    also stands in for what the design doc calls the "harness" (design
    §7): there's no separate ``harness`` field on ``EvalRunManifest``, so
    ``adapter`` is used for that identity too.

    "Target policy" covers two things together: the declared
    ``target_fingerprint_policy`` (the *rule* for how the target's version
    should be pinned down) and the actual ``target_fingerprint`` each run
    recorded (the *specific version* that rule resolved to). Two runs can
    share the same ``target_name`` and the same policy, and still have
    provably run against different underlying targets -- so the
    fingerprints themselves have to match too, not just the name and
    policy. And if one run has no recorded fingerprint at all (``None``)
    while the other has a specific one pinned down, that counts as a
    mismatch, not something to silently let pass -- "we don't know what
    version this was" must never be treated as equal to "we've verified
    exactly what version this was."

    ``allow_cross_environment`` (ADR-0015) narrows which mismatches are
    allowed to block the comparison: when it's set, a field listed in
    :data:`_WAIVABLE_UNDER_CROSS_ENVIRONMENT` that differs gets added to
    the returned ``waived`` list instead of ``mismatches`` -- using its
    declared field name, not the human-readable label. Every other field
    -- both the dataset-identity checks above and the other eight rows in
    :data:`_PROVENANCE_CHECKS` -- always blocks the comparison regardless
    of this flag.
    """
    mismatches: list[str] = []
    waived: list[str] = []
    left_dataset, right_dataset = left.resolved_dataset, right.resolved_dataset
    left_manifest, right_manifest = left.manifest, right.manifest

    if left_dataset.dataset_id != right_dataset.dataset_id:
        mismatches.append(
            f"dataset id differs: {left_dataset.dataset_id!r} != {right_dataset.dataset_id!r}"
        )
    if left_dataset.revision != right_dataset.revision:
        mismatches.append(
            f"dataset revision differs: {left_dataset.revision!r} != {right_dataset.revision!r}"
        )
    if left_dataset.config != right_dataset.config:
        mismatches.append(
            f"dataset config differs: {left_dataset.config!r} != {right_dataset.config!r}"
        )
    if left_dataset.split != right_dataset.split:
        mismatches.append(
            f"dataset split differs: {left_dataset.split!r} != {right_dataset.split!r}"
        )
    # The manifest-provenance fields are compared by looping over the
    # checks table below, so the set of "fields we checked" is always
    # derived from the comparisons that actually ran (see
    # _PROVENANCE_CHECKS) -- the R-004 drift-detecting test depends on
    # this being true.
    for name, label, get_value, waivable in _PROVENANCE_CHECKS:
        left_value, right_value = get_value(left_manifest), get_value(right_manifest)
        if left_value == right_value:
            continue
        if allow_cross_environment and waivable:
            waived.append(name)
        else:
            mismatches.append(f"{label} differs: {left_value!r} != {right_value!r}")
    return mismatches, waived


def _is_pass(sample: SampleResult) -> bool:
    return sample.grade is not None and sample.grade.status is GradeStatus.PASS


def _index_by_sample_and_attempt(run: EvalRunResult) -> dict[tuple[str, int], bool]:
    return {
        (result.sample.sample_id, result.execution.attempt): _is_pass(result)
        for result in run.samples
    }


def compare_runs(
    left: EvalRunResult,
    right: EvalRunResult,
    *,
    bootstrap_samples: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int,
    allow_cross_environment: bool = False,
) -> ComparisonResult:
    """Compare two runs' paired success rates, with a 95% confidence
    interval built by seeded bootstrap resampling.

    Args:
        left: The baseline run.
        right: The candidate run being compared against ``left``.
        bootstrap_samples: Number of bootstrap resamples to draw (i.e. how
            many times to redraw and recompute -- see the module
            docstring for what this means). Must be in the inclusive
            range ``[100, 10000]``; defaults to 1000.
        seed: Seed for a local ``random.Random`` instance, so the same
            seed always reproduces the exact same sequence of random
            draws and therefore the exact same result. Required
            (keyword-only, no default) precisely so that a comparison is
            never silently different from one run of the code to the
            next.
        allow_cross_environment: When ``True`` (ADR-0015), a mismatch on
            *only* ``environment_fingerprint`` and/or ``code_fingerprint``
            is waived (allowed through) rather than raised as an error,
            and whichever field(s) got waived are recorded on the
            returned result's ``waived_provenance_fields``. The other
            eight provenance fields can never be waived through this
            flag. Defaults to ``False``, so comparisons reject any
            undisclosed difference in environment or code build unless a
            caller explicitly opts in to allowing it.

    Returns:
        A :class:`ComparisonResult` containing: the observed paired
        difference in pass rate (``right``'s pass rate minus ``left``'s,
        over the paired subset); its bootstrap-derived 2.5th/97.5th
        percentile bounds (the 95% confidence interval); the paired
        ``(sample_id, attempt)`` count and the distinct-sample count; the
        seed that was used; and any fields waived under
        ``allow_cross_environment``.

    Raises:
        ValueError: If ``bootstrap_samples`` is outside ``[100, 10000]``.
        IncompatibleRuns: If the two runs' resolved dataset identity,
            adapter, grader, target policy (including target fingerprint),
            sampling policy, or attempt count differ -- or if their
            environment/code fingerprint differs and that wasn't waived
            via ``allow_cross_environment``. Also raised if the two runs
            share zero paired ``(sample_id, attempt)`` observations, since
            you can't compute a meaningful difference from nothing to
            compare. The error message lists every mismatched field, or
            names both run IDs when the problem is zero overlap instead.
    """
    if not (_MIN_BOOTSTRAP_SAMPLES <= bootstrap_samples <= _MAX_BOOTSTRAP_SAMPLES):
        raise ValueError(
            "bootstrap_samples must satisfy "
            f"{_MIN_BOOTSTRAP_SAMPLES} <= bootstrap_samples <= {_MAX_BOOTSTRAP_SAMPLES} "
            f"(got {bootstrap_samples})"
        )

    mismatches, waived = _describe_mismatches(
        left, right, allow_cross_environment=allow_cross_environment
    )
    if mismatches:
        raise IncompatibleRuns(
            message=(
                f"runs {left.run_id!r} and {right.run_id!r} are not comparable: "
                + "; ".join(mismatches)
            ),
            context={"left_run_id": left.run_id, "right_run_id": right.run_id},
        )

    left_index = _index_by_sample_and_attempt(left)
    right_index = _index_by_sample_and_attempt(right)
    paired_keys = sorted(set(left_index) & set(right_index))

    if not paired_keys:
        raise IncompatibleRuns(
            message=(
                f"runs {left.run_id!r} and {right.run_id!r} are not comparable: "
                "they share zero paired (sample_id, attempt) observations"
            ),
            context={"left_run_id": left.run_id, "right_run_id": right.run_id},
        )

    # (right_pass - left_pass) per paired observation: +1 if only right
    # passed, -1 if only left passed, 0 if both agreed.
    deltas = [int(right_index[key]) - int(left_index[key]) for key in paired_keys]
    paired_count = len(deltas)
    estimate = sum(deltas) / paired_count
    sample_count = len({sample_id for sample_id, _attempt in paired_keys})

    lower, upper = _bootstrap_percentiles(deltas, bootstrap_samples=bootstrap_samples, seed=seed)

    return ComparisonResult(
        estimate=estimate,
        lower_percentile=lower,
        upper_percentile=upper,
        paired_count=paired_count,
        sample_count=sample_count,
        seed=seed,
        waived_provenance_fields=tuple(waived),
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Return the given percentile from an already-sorted, non-empty list,
    using linear interpolation.

    Unlike the "nearest rank" method used elsewhere in this codebase (see
    :func:`agentic_evalkit.stats.aggregate._percentile`), which always
    returns a value that was actually observed in the data, this version
    can return a value in between two observed data points: it looks at
    the two nearest sorted values and computes a proportional in-between
    point for the exact percentile requested.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    return sorted_values[lower_index] + fraction * (
        sorted_values[upper_index] - sorted_values[lower_index]
    )


def _bootstrap_percentiles(
    deltas: list[int], *, bootstrap_samples: int, seed: int
) -> tuple[float, float]:
    """Estimate the 2.5th/97.5th percentiles of the mean paired difference,
    using bootstrap resampling.

    "Resampling with replacement" means: repeatedly build a new list the
    same size as ``deltas`` by randomly picking values out of ``deltas``
    (the same original value can be picked more than once, and some might
    not get picked at all), and compute the mean of each such new list.
    Doing this ``bootstrap_samples`` times gives us a whole distribution of
    "what the mean might have been," which tells us how much it would
    plausibly vary -- and the 2.5th/97.5th percentiles of that
    distribution become our 95% confidence interval.

    This uses a local ``random.Random(seed)`` instance -- never the
    shared, module-level ``random`` state -- so that running a comparison
    never has side effects on unrelated code elsewhere, and the result is
    fully reproducible from the seed alone (run it twice with the same
    seed, get the exact same answer both times).

    ``compare_runs`` never actually calls this function with an empty
    ``deltas`` list: it raises
    :class:`~agentic_evalkit.errors.IncompatibleRuns` earlier, before
    reaching here, whenever the two runs share zero paired observations --
    rather than let a meaningless "estimate from nothing" masquerade as a
    real result. The empty-list guard below exists only as a defensive
    fallback, in case some future caller invokes this function directly
    and skips that check -- it avoids a ``ZeroDivisionError`` in the
    resample-mean computation below, and should never be read as
    something that normally happens in practice.
    """
    if not deltas:
        return (0.0, 0.0)

    rng = random.Random(seed)  # noqa: S311 -- seeded so this gives the same result every run, not random
    n = len(deltas)
    resample_means: list[float] = []
    for _ in range(bootstrap_samples):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        resample_means.append(sum(resample) / n)

    resample_means.sort()
    lower = _percentile(resample_means, _LOWER_PERCENTILE)
    upper = _percentile(resample_means, _UPPER_PERCENTILE)
    return (lower, upper)
