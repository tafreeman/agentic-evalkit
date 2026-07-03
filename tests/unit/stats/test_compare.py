"""Tests for run compatibility checking and paired bootstrap comparison
(design §10, ADR-0008).

``compare_runs`` first verifies that two runs share every field that makes
a delta between them meaningful: resolved dataset id/revision/config/
split, adapter, grader, target policy (including target fingerprint),
sampling policy, and attempt count. Any mismatch raises
:class:`~agentic_evalkit.errors.IncompatibleRuns` listing *every*
mismatched field, not just the first one found. A ``None`` fingerprint on
one side and a pinned fingerprint on the other is itself a mismatch --
unknown provenance is never treated as equal to verified provenance --
while ``None`` on both sides still compares fine, for backward
compatibility with runs recorded before fingerprints were captured.

Only for compatible runs does it pair observations by sample and attempt
id and bootstrap the paired success-rate delta with a local
``random.Random(seed)`` instance, so the same seed always reproduces the
same estimate. Zero paired observations is itself raised as
``IncompatibleRuns`` rather than returned as a confident-looking zero
delta. ``sample_count`` is the number of distinct sample ids represented,
which is not the same as ``paired_count`` once ``attempts > 1``.

Covers the plan's verbatim snippet (docs/plans/
2026-07-02-agentic-evalkit-initial-release.md, Task 12 Step 2) unmodified.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentic_evalkit.errors import IncompatibleRuns
from agentic_evalkit.models import (
    DatasetRef,
    EvalRunManifest,
    EvalRunResult,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    RunSummary,
    SampleResult,
    SamplingPolicy,
)
from agentic_evalkit.stats.compare import compare_runs

_STARTED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 7, 2, 12, 5, 0, tzinfo=UTC)


def _sample(sample_id: str) -> EvalSample:
    return EvalSample(
        sample_id=sample_id,
        input={"question": f"question for {sample_id}"},
        reference="42",
        source_digest=f"sha256:{sample_id}",
        adapter="gsm8k@1",
    )


def _execution(
    sample_id: str, *, attempt: int, status: ExecutionStatus
) -> NormalizedExecutionResult:
    return NormalizedExecutionResult(
        sample_id=sample_id,
        attempt=attempt,
        status=status,
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _grade(sample_id: str, *, status: GradeStatus) -> GradeResult:
    return GradeResult(
        sample_id=sample_id,
        grader="normalized-exact@1",
        status=status,
        score=1.0 if status is GradeStatus.PASS else 0.0,
        hard_gate=False,
        created_at=_FINISHED_AT,
    )


def _sample_result(sample_id: str, *, attempt: int, passed: bool) -> SampleResult:
    status = GradeStatus.PASS if passed else GradeStatus.FAIL
    return SampleResult(
        sample=_sample(sample_id),
        execution=_execution(sample_id, attempt=attempt, status=ExecutionStatus.COMPLETED),
        grade=_grade(sample_id, status=status),
    )


def _run(
    *,
    dataset_revision: str = "abc",
    dataset_id: str = "openai/gsm8k",
    config: str | None = "main",
    split: str | None = "test",
    adapter: str = "gsm8k@1",
    grader: str = "normalized-exact@1",
    target_name: str = "echo-target",
    target_fingerprint_policy: str | None = None,
    target_fingerprint: str | None = None,
    temperature: float | None = 0.0,
    seed: int | None = 7,
    attempts: int = 1,
    samples: tuple[SampleResult, ...] = (),
    run_id: str = "run-001",
) -> EvalRunResult:
    """A complete two-sample ``EvalRunResult`` fixture (Task 12 Step 2).

    Every provenance field is independently parameterized so tests can
    hold every field but one constant and assert only the intended field
    triggers a mismatch. When ``samples`` is left empty, two default
    passing samples at attempt 1 are used so a bootstrap comparison has
    something to pair.
    """
    if not samples:
        samples = (
            _sample_result("s0", attempt=1, passed=True),
            _sample_result("s1", attempt=1, passed=True),
        )
    manifest = EvalRunManifest(
        run_name="compare-fixture",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id=dataset_id),
        adapter=adapter,
        grader=grader,
        target_name=target_name,
        target_fingerprint_policy=target_fingerprint_policy,
        target_fingerprint=target_fingerprint,
        sampling=SamplingPolicy(seed=seed, temperature=temperature, attempts=attempts),
        attempts=attempts,
    )
    resolved_dataset = ResolvedDataset(
        dataset_id=dataset_id,
        revision=dataset_revision,
        config=config,
        split=split,
    )
    return EvalRunResult(
        run_id=run_id,
        manifest=manifest,
        resolved_dataset=resolved_dataset,
        samples=samples,
        summary=RunSummary(total=len(samples), passed=len(samples)),
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


# --- Step 2 (plan verbatim): dataset revision mismatch ----------------------


def test_rejects_different_dataset_revisions() -> None:
    left = _run(dataset_revision="abc")
    right = _run(dataset_revision="def")
    with pytest.raises(IncompatibleRuns, match="dataset revision"):
        compare_runs(left, right, bootstrap_samples=1000, seed=7)


# --- Compatibility: every other provenance field -----------------------------


def test_rejects_different_dataset_ids() -> None:
    left = _run(dataset_id="openai/gsm8k")
    right = _run(dataset_id="openai/other")
    with pytest.raises(IncompatibleRuns, match="dataset"):
        compare_runs(left, right, seed=1)


def test_rejects_different_configs() -> None:
    left = _run(config="main")
    right = _run(config="socratic")
    with pytest.raises(IncompatibleRuns, match="config"):
        compare_runs(left, right, seed=1)


def test_rejects_different_splits() -> None:
    left = _run(split="test")
    right = _run(split="train")
    with pytest.raises(IncompatibleRuns, match="split"):
        compare_runs(left, right, seed=1)


def test_rejects_different_adapters() -> None:
    left = _run(adapter="gsm8k@1")
    right = _run(adapter="gsm8k@2")
    with pytest.raises(IncompatibleRuns, match="adapter"):
        compare_runs(left, right, seed=1)


def test_rejects_different_graders() -> None:
    left = _run(grader="normalized-exact@1")
    right = _run(grader="normalized-exact@2")
    with pytest.raises(IncompatibleRuns, match="grader"):
        compare_runs(left, right, seed=1)


def test_rejects_different_target_names() -> None:
    left = _run(target_name="target-a")
    right = _run(target_name="target-b")
    with pytest.raises(IncompatibleRuns, match="target"):
        compare_runs(left, right, seed=1)


def test_rejects_different_target_fingerprint_policies() -> None:
    left = _run(target_fingerprint_policy="strict")
    right = _run(target_fingerprint_policy="loose")
    with pytest.raises(IncompatibleRuns, match="target"):
        compare_runs(left, right, seed=1)


# --- Actual target fingerprint comparison -------------------------------------
#
# Same target_name and target_fingerprint_policy is not enough: two runs can
# share both while having actually executed against provably different
# targets. compare_runs must compare the recorded fingerprints themselves.


def test_rejects_different_target_fingerprints() -> None:
    left = _run(target_fingerprint="sha256:aaaa")
    right = _run(target_fingerprint="sha256:bbbb")
    with pytest.raises(IncompatibleRuns, match="fingerprint"):
        compare_runs(left, right, seed=1)


def test_accepts_equal_non_none_target_fingerprints() -> None:
    left = _run(target_fingerprint="sha256:aaaa")
    right = _run(target_fingerprint="sha256:aaaa")
    result = compare_runs(left, right, seed=1)
    assert result.paired_count == 2


def test_rejects_none_fingerprint_against_a_pinned_fingerprint() -> None:
    # Unknown provenance (None) must never silently compare as equal to
    # verified provenance (a pinned fingerprint) -- regardless of which
    # side is which.
    left = _run(target_fingerprint=None)
    right = _run(target_fingerprint="sha256:aaaa")
    with pytest.raises(IncompatibleRuns, match="fingerprint"):
        compare_runs(left, right, seed=1)

    left = _run(target_fingerprint="sha256:aaaa")
    right = _run(target_fingerprint=None)
    with pytest.raises(IncompatibleRuns, match="fingerprint"):
        compare_runs(left, right, seed=1)


def test_accepts_none_target_fingerprint_on_both_sides() -> None:
    # Backward compatibility: runs recorded before target_fingerprint
    # capture existed both have None, and must still compare fine.
    left = _run(target_fingerprint=None)
    right = _run(target_fingerprint=None)
    result = compare_runs(left, right, seed=1)
    assert result.paired_count == 2


def test_rejects_different_sampling_temperatures() -> None:
    left = _run(temperature=0.0)
    right = _run(temperature=0.7)
    with pytest.raises(IncompatibleRuns, match="temperature"):
        compare_runs(left, right, seed=1)


def test_rejects_different_sampling_seeds() -> None:
    left = _run(seed=7)
    right = _run(seed=8)
    with pytest.raises(IncompatibleRuns, match="seed"):
        compare_runs(left, right, seed=1)


def test_rejects_different_attempt_counts() -> None:
    left = _run(attempts=1)
    right = _run(attempts=2)
    with pytest.raises(IncompatibleRuns, match="attempt"):
        compare_runs(left, right, seed=1)


def test_all_mismatches_are_listed_together_not_just_the_first() -> None:
    left = _run(dataset_revision="abc", adapter="gsm8k@1", grader="normalized-exact@1")
    right = _run(dataset_revision="def", adapter="gsm8k@2", grader="normalized-exact@2")
    with pytest.raises(IncompatibleRuns) as excinfo:
        compare_runs(left, right, seed=1)
    message = str(excinfo.value)
    assert "dataset revision" in message
    assert "adapter" in message
    assert "grader" in message


# --- bootstrap_samples validation --------------------------------------------


def test_rejects_bootstrap_samples_below_minimum() -> None:
    left = _run()
    right = _run()
    with pytest.raises(ValueError, match="bootstrap_samples"):
        compare_runs(left, right, bootstrap_samples=99, seed=1)


def test_rejects_bootstrap_samples_above_maximum() -> None:
    left = _run()
    right = _run()
    with pytest.raises(ValueError, match="bootstrap_samples"):
        compare_runs(left, right, bootstrap_samples=10_001, seed=1)


def test_accepts_bootstrap_samples_at_range_boundaries() -> None:
    left = _run()
    right = _run()
    low = compare_runs(left, right, bootstrap_samples=100, seed=1)
    high = compare_runs(left, right, bootstrap_samples=10_000, seed=1)
    assert low.paired_count == 2
    assert high.paired_count == 2


# --- Compatible runs: pairing and bootstrap estimate -------------------------


def test_compatible_runs_pair_by_sample_and_attempt_id() -> None:
    left = _run(
        run_id="left",
        samples=(
            _sample_result("s0", attempt=1, passed=True),
            _sample_result("s1", attempt=1, passed=False),
        ),
    )
    right = _run(
        run_id="right",
        samples=(
            _sample_result("s0", attempt=1, passed=True),
            _sample_result("s1", attempt=1, passed=True),
        ),
    )
    result = compare_runs(left, right, bootstrap_samples=500, seed=42)
    assert result.paired_count == 2
    assert result.sample_count == 2
    # right passes both, left passes only one: delta (right - left) is +0.5.
    assert result.estimate == pytest.approx(0.5)


def test_unmatched_attempts_are_excluded_from_pairing() -> None:
    left = _run(
        run_id="left",
        samples=(
            _sample_result("s0", attempt=1, passed=True),
            _sample_result("s1", attempt=1, passed=True),
        ),
    )
    right = _run(
        run_id="right",
        samples=(
            _sample_result("s0", attempt=1, passed=True),
            # s1 is missing from the right run entirely -- and s2 exists
            # only on the right, so neither one can be paired.
            _sample_result("s2", attempt=1, passed=True),
        ),
    )
    result = compare_runs(left, right, bootstrap_samples=200, seed=3)
    assert result.paired_count == 1
    assert result.sample_count == 1


def test_zero_paired_overlap_raises_incompatible_runs_naming_both_run_ids() -> None:
    # left and right are otherwise fully compatible (same manifest fields),
    # but share no (sample_id, attempt) keys at all -- there is nothing to
    # compute a delta from, so this must fail loudly rather than return a
    # plausible-looking "no difference" verdict from literally nothing.
    left = _run(
        run_id="left-run",
        samples=(_sample_result("s0", attempt=1, passed=True),),
    )
    right = _run(
        run_id="right-run",
        samples=(_sample_result("s1", attempt=1, passed=True),),
    )
    with pytest.raises(IncompatibleRuns) as excinfo:
        compare_runs(left, right, seed=1)
    message = str(excinfo.value)
    assert "left-run" in message
    assert "right-run" in message
    assert excinfo.value.context["left_run_id"] == "left-run"
    assert excinfo.value.context["right_run_id"] == "right-run"


def test_sample_count_counts_distinct_samples_not_attempt_pairs() -> None:
    # 2 samples x 3 attempts each, fully overlapping between left and
    # right: paired_count must be 6 (one pair per (sample_id, attempt)),
    # but sample_count must stay 2 -- it counts distinct sample ids, not
    # attempt-pairs, so it must never multiply with the attempt count the
    # way paired_count does.
    left_samples = tuple(
        _sample_result(sample_id, attempt=attempt, passed=True)
        for sample_id in ("s0", "s1")
        for attempt in (1, 2, 3)
    )
    right_samples = tuple(
        _sample_result(sample_id, attempt=attempt, passed=True)
        for sample_id in ("s0", "s1")
        for attempt in (1, 2, 3)
    )
    left = _run(run_id="left", attempts=3, samples=left_samples)
    right = _run(run_id="right", attempts=3, samples=right_samples)
    result = compare_runs(left, right, seed=1)
    assert result.paired_count == 6
    assert result.sample_count == 2


def test_bootstrap_percentiles_bracket_the_estimate_reasonably() -> None:
    left = _run(
        run_id="left",
        samples=tuple(_sample_result(f"s{i}", attempt=1, passed=(i % 2 == 0)) for i in range(20)),
    )
    right = _run(
        run_id="right",
        samples=tuple(_sample_result(f"s{i}", attempt=1, passed=(i % 3 != 0)) for i in range(20)),
    )
    result = compare_runs(left, right, bootstrap_samples=2000, seed=11)
    assert result.lower_percentile <= result.estimate <= result.upper_percentile
    assert -1.0 <= result.lower_percentile <= 1.0
    assert -1.0 <= result.upper_percentile <= 1.0


def test_seed_is_recorded_on_the_result() -> None:
    left = _run()
    right = _run()
    result = compare_runs(left, right, bootstrap_samples=100, seed=123)
    assert result.seed == 123


def test_same_seed_is_deterministic_across_calls() -> None:
    left = _run(
        run_id="left",
        samples=tuple(_sample_result(f"s{i}", attempt=1, passed=(i % 2 == 0)) for i in range(10)),
    )
    right = _run(
        run_id="right",
        samples=tuple(_sample_result(f"s{i}", attempt=1, passed=(i % 3 == 0)) for i in range(10)),
    )
    first = compare_runs(left, right, bootstrap_samples=500, seed=99)
    second = compare_runs(left, right, bootstrap_samples=500, seed=99)
    assert first.estimate == second.estimate
    assert first.lower_percentile == second.lower_percentile
    assert first.upper_percentile == second.upper_percentile


def test_different_seeds_do_not_change_the_point_estimate() -> None:
    # The point estimate (observed paired delta) is deterministic
    # regardless of seed; only the bootstrap percentile bounds depend on
    # the seed's resample draws.
    left = _run(
        run_id="left",
        samples=tuple(_sample_result(f"s{i}", attempt=1, passed=(i % 2 == 0)) for i in range(10)),
    )
    right = _run(
        run_id="right",
        samples=tuple(_sample_result(f"s{i}", attempt=1, passed=(i % 3 == 0)) for i in range(10)),
    )
    first = compare_runs(left, right, bootstrap_samples=500, seed=1)
    second = compare_runs(left, right, bootstrap_samples=500, seed=2)
    assert first.estimate == pytest.approx(second.estimate)


def test_default_bootstrap_samples_is_one_thousand() -> None:
    import inspect

    signature = inspect.signature(compare_runs)
    assert signature.parameters["bootstrap_samples"].default == 1000


def test_compare_runs_requires_keyword_arguments_for_bootstrap_and_seed() -> None:
    import inspect

    signature = inspect.signature(compare_runs)
    assert signature.parameters["bootstrap_samples"].kind == inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["seed"].kind == inspect.Parameter.KEYWORD_ONLY
