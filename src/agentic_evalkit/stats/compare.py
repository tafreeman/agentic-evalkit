"""Run compatibility checking and paired bootstrap comparison (design §10, ADR-0008).

Two runs are only comparable when their dataset revision, split, adapter,
grader, target policy, target fingerprint, and sampling policy are
compatible; an incompatible comparison must fail with an explanation rather
than silently producing a misleading delta (design §10). ``compare_runs``
checks every provenance field described in Task 12 Step 6 (dataset ID/
revision/config/split, adapter, grader, target policy, target fingerprint,
sampling temperature/seed policy, attempt count) and raises
:class:`~agentic_evalkit.errors.IncompatibleRuns` listing every mismatch it
finds, not just the first. A named target with an unpinned fingerprint
(``None``) is never treated as equal to a pinned fingerprint on the other
run: unknown provenance must not silently compare as equal to verified
provenance.

For compatible runs, it pairs observations by ``(sample_id, attempt)`` so
missing attempts on either side are simply excluded from the paired
comparison rather than treated as a failure. Zero paired observations means
there is nothing to estimate a delta from, so ``compare_runs`` raises
:class:`~agentic_evalkit.errors.IncompatibleRuns` rather than returning a
confident-looking zero delta. Otherwise it computes the observed paired
success-rate delta and bootstraps a 95% interval for that delta using a
local ``random.Random(seed)`` instance so the same seed always reproduces
the same bootstrap draw sequence (ADR-0008: deterministic seeded
bootstrap).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from agentic_evalkit.errors import IncompatibleRuns
from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.grades import GradeStatus
from agentic_evalkit.models.runs import EvalRunManifest

if TYPE_CHECKING:
    from agentic_evalkit.models.runs import EvalRunResult, SampleResult

__all__ = ["PROVENANCE_FIELDS_CHECKED", "ComparisonResult", "compare_runs"]

#: The manifest provenance fields ``_describe_mismatches`` actually compares.
#: Bound directly to :meth:`EvalRunManifest.provenance_field_names` so the
#: checked set and the manifest's declared provenance set can never drift: a
#: new declared field that ``_describe_mismatches`` fails to check is caught by
#: the ``tests/contract/test_provenance_drift.py`` contract (R-004 P0).
PROVENANCE_FIELDS_CHECKED: frozenset[str] = EvalRunManifest.provenance_field_names()

_MIN_BOOTSTRAP_SAMPLES = 100
_MAX_BOOTSTRAP_SAMPLES = 10_000
_DEFAULT_BOOTSTRAP_SAMPLES = 1000
_LOWER_PERCENTILE = 2.5
_UPPER_PERCENTILE = 97.5


class ComparisonResult(FrozenModel):
    """The outcome of a paired bootstrap comparison between two compatible runs.

    ``estimate`` is the observed paired success-rate delta
    (``right_pass_rate - left_pass_rate`` over the paired subset).
    ``lower_percentile``/``upper_percentile`` are the 2.5th/97.5th
    percentiles of the bootstrap resample distribution of that delta.
    """

    estimate: float
    lower_percentile: float
    upper_percentile: float
    paired_count: int
    """Number of paired ``(sample_id, attempt)`` observations the delta and
    bootstrap were computed over. With ``attempts > 1`` this can exceed
    ``sample_count``, since each distinct sample may contribute up to one
    paired observation per attempt."""
    sample_count: int
    """Number of *distinct* ``sample_id`` values represented in
    ``paired_count``, regardless of how many attempts each contributed.
    Never multiplies with the attempt count the way ``paired_count`` does."""
    seed: int


def _describe_mismatches(left: EvalRunResult, right: EvalRunResult) -> list[str]:
    """Return a human-readable description for every incompatible field.

    Compares the *resolved* dataset identity (design §10: "dataset
    revision, split, adapter, harness, grader, target policy, and sampling
    policy") rather than the requested ``DatasetRef``, since the resolved
    dataset is the actual immutable source each run drew from. ``adapter``
    doubles as the harness/benchmark-binding identity per design §7, since
    ``EvalRunManifest`` has no separate ``harness`` field. "Target policy"
    covers both the declared ``target_fingerprint_policy`` and the actual
    ``target_fingerprint`` each run recorded: two runs sharing a
    ``target_name`` under the same policy can still have drawn from
    provably different targets, so the fingerprints themselves must match
    too. An unset (``None``) fingerprint on one side and a pinned
    fingerprint on the other is a mismatch, not a pass-through -- unknown
    provenance is never treated as equal to verified provenance.
    """
    mismatches: list[str] = []
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
    if left_manifest.adapter != right_manifest.adapter:
        mismatches.append(
            f"adapter differs: {left_manifest.adapter!r} != {right_manifest.adapter!r}"
        )
    if left_manifest.grader != right_manifest.grader:
        mismatches.append(f"grader differs: {left_manifest.grader!r} != {right_manifest.grader!r}")
    if left_manifest.target_name != right_manifest.target_name:
        mismatches.append(
            f"target name differs: {left_manifest.target_name!r} != {right_manifest.target_name!r}"
        )
    if left_manifest.target_fingerprint_policy != right_manifest.target_fingerprint_policy:
        mismatches.append(
            "target fingerprint policy differs: "
            f"{left_manifest.target_fingerprint_policy!r} "
            f"!= {right_manifest.target_fingerprint_policy!r}"
        )
    if left_manifest.target_fingerprint != right_manifest.target_fingerprint:
        mismatches.append(
            "target fingerprint differs: "
            f"{left_manifest.target_fingerprint!r} != {right_manifest.target_fingerprint!r}"
        )
    if left_manifest.sampling.temperature != right_manifest.sampling.temperature:
        mismatches.append(
            "sampling temperature differs: "
            f"{left_manifest.sampling.temperature!r} != {right_manifest.sampling.temperature!r}"
        )
    if left_manifest.sampling.seed != right_manifest.sampling.seed:
        mismatches.append(
            f"sampling seed differs: {left_manifest.sampling.seed!r} "
            f"!= {right_manifest.sampling.seed!r}"
        )
    if left_manifest.attempts != right_manifest.attempts:
        mismatches.append(
            f"attempt count differs: {left_manifest.attempts!r} != {right_manifest.attempts!r}"
        )
    return mismatches


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
) -> ComparisonResult:
    """Compare two runs' paired success rates with a seeded bootstrap interval.

    Args:
        left: The baseline run.
        right: The candidate run being compared against ``left``.
        bootstrap_samples: Number of bootstrap resamples to draw. Must be
            in the inclusive range ``[100, 10000]``; defaults to 1000.
        seed: Seed for a local ``random.Random`` instance so the same seed
            always reproduces the same bootstrap draw sequence. Required
            (keyword-only, no default) so a comparison is never silently
            nondeterministic.

    Returns:
        A :class:`ComparisonResult` with the observed paired delta
        (``right`` pass rate minus ``left`` pass rate over the paired
        subset), its bootstrap 2.5/97.5 percentiles, the paired
        ``(sample_id, attempt)`` count and distinct sample count, and the
        seed used.

    Raises:
        ValueError: If ``bootstrap_samples`` is outside ``[100, 10000]``.
        IncompatibleRuns: If the two runs' resolved dataset identity,
            adapter, grader, target policy (including target fingerprint),
            sampling policy, or attempt count differ. Also raised if the
            two runs share zero paired ``(sample_id, attempt)``
            observations, since a delta computed over nothing is not a
            comparison. The error message lists every mismatched field, or
            names both run ids when the failure is zero overlap.
    """
    if not (_MIN_BOOTSTRAP_SAMPLES <= bootstrap_samples <= _MAX_BOOTSTRAP_SAMPLES):
        raise ValueError(
            "bootstrap_samples must satisfy "
            f"{_MIN_BOOTSTRAP_SAMPLES} <= bootstrap_samples <= {_MAX_BOOTSTRAP_SAMPLES} "
            f"(got {bootstrap_samples})"
        )

    mismatches = _describe_mismatches(left, right)
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
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Linear-interpolated percentile over an already-sorted, nonempty list."""
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
    """Bootstrap the 2.5/97.5 percentiles of the mean paired delta.

    Resamples ``deltas`` with replacement ``bootstrap_samples`` times using
    a local ``random.Random(seed)`` instance (never the shared module-level
    ``random`` state) so a comparison never has side effects on unrelated
    code and is fully reproducible from its seed alone.

    ``compare_runs`` never calls this with an empty ``deltas``: it raises
    :class:`~agentic_evalkit.errors.IncompatibleRuns` before reaching here
    when the two runs share zero paired observations, rather than let a
    zero-observation "estimate" masquerade as a real result. The guard
    below is defensive only -- it protects any other future caller from a
    ``ZeroDivisionError`` in the resample-mean computation -- and must
    never be read as this function's normal, expected return path.
    """
    if not deltas:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n = len(deltas)
    resample_means: list[float] = []
    for _ in range(bootstrap_samples):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        resample_means.append(sum(resample) / n)

    resample_means.sort()
    lower = _percentile(resample_means, _LOWER_PERCENTILE)
    upper = _percentile(resample_means, _UPPER_PERCENTILE)
    return (lower, upper)
