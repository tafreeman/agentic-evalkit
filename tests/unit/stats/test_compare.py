"""Tests for checking whether two runs are safe to compare, and for the
paired bootstrap comparison itself (design section 10, ADR-0008).

``compare_runs`` first checks that two runs share every fact about exactly
what was run and how -- the resolved dataset's id/revision/config/split,
the adapter, the grader, the target policy (including the target's exact
recorded "fingerprint": a hash that uniquely identifies one exact version
of something), the sampling policy, and the attempt count. Together, these
facts are what ``compare.py``'s module docstring calls the runs'
*provenance*; only if all of it matches is a difference between the two
runs actually meaningful to report -- otherwise you might just be
comparing, say, two different dataset splits, and mistaking that
difference in difficulty for a real improvement. Any mismatch raises
:class:`~agentic_evalkit.errors.IncompatibleRuns`, and the error lists
*every* mismatched field it found, not just the first one. A ``None``
fingerprint on one side paired with an actual, pinned fingerprint on the
other counts as a mismatch too -- "we don't know what exact version this
was" must never be silently treated as "these two match" -- while ``None``
on *both* sides compares fine, since that is just what runs recorded
before fingerprint capture existed look like.

Only once two runs are confirmed compatible does ``compare_runs`` pair up
their individual results by matching ``(sample_id, attempt)``, then
estimate a confidence interval around the paired difference in success
rate using bootstrap resampling (repeatedly resampling, with replacement,
from the paired results we already have -- see ``compare.py``'s module
docstring for the full explanation of why this works), using a local
``random.Random(seed)`` instance so that reusing the same seed always
reproduces the exact same estimate. If there are zero paired observations
at all, that itself raises ``IncompatibleRuns`` -- there is nothing to
compute a difference from, so it must fail loudly rather than return a
confident-looking "zero difference" that was actually computed from
nothing. ``sample_count`` counts distinct sample ids, which is a different
number from ``paired_count`` once ``attempts > 1`` (each sample can then
contribute more than one paired observation, one per attempt).

Covers the plan's verbatim snippet (docs/plans/
2026-07-02-agentic-evalkit-initial-release.md, Task 12 Step 2) unmodified.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from _stats_fixtures import _execution, _grade, _sample

from agentic_evalkit.errors import IncompatibleRuns
from agentic_evalkit.models import (
    DatasetRef,
    EvalRunManifest,
    EvalRunResult,
    ExecutionStatus,
    GradeStatus,
    ResolvedDataset,
    RunSummary,
    SampleResult,
    SamplingPolicy,
)
from agentic_evalkit.stats.compare import ComparisonResult, compare_runs

_STARTED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 7, 2, 12, 5, 0, tzinfo=UTC)


def _sample_result(sample_id: str, *, attempt: int, passed: bool) -> SampleResult:
    status = GradeStatus.PASS if passed else GradeStatus.FAIL
    score = 1.0 if passed else 0.0
    return SampleResult(
        sample=_sample(sample_id),
        execution=_execution(sample_id, attempt=attempt, status=ExecutionStatus.COMPLETED),
        grade=_grade(sample_id, status=status, score=score),
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
    environment_fingerprint: str | None = None,
    code_fingerprint: str | None = None,
    temperature: float | None = 0.0,
    seed: int | None = 7,
    attempts: int = 1,
    samples: tuple[SampleResult, ...] = (),
    run_id: str = "run-001",
) -> EvalRunResult:
    """A complete, two-sample ``EvalRunResult`` fixture (Task 12 Step 2).

    Every provenance field (every fact about exactly what was run and how
    -- see the module docstring above) can be set independently through
    this function's keyword arguments, so each test can hold every field
    but one constant and check that changing only that one field is what
    triggers a mismatch. When ``samples`` is left empty, this defaults to
    two samples that both passed on attempt 1, so there is always
    something for a bootstrap comparison to pair up and compare.
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
        environment_fingerprint=environment_fingerprint,
        code_fingerprint=code_fingerprint,
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


# --- Comparing the actual target fingerprint, not just its name and policy --
#
# Sharing the same target_name and target_fingerprint_policy is not enough
# proof that two runs tested the same thing: they could still have actually
# run against two different, specific versions of the target system.
# compare_runs must also compare the recorded fingerprints themselves -- the
# hash that identifies exactly which version was used.


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
    # An unknown/unrecorded fingerprint (None) must never silently be
    # treated as equal to a verified, pinned-down fingerprint -- no matter
    # which side of the comparison it is on.
    left = _run(target_fingerprint=None)
    right = _run(target_fingerprint="sha256:aaaa")
    with pytest.raises(IncompatibleRuns, match="fingerprint"):
        compare_runs(left, right, seed=1)

    left = _run(target_fingerprint="sha256:aaaa")
    right = _run(target_fingerprint=None)
    with pytest.raises(IncompatibleRuns, match="fingerprint"):
        compare_runs(left, right, seed=1)


def test_accepts_none_target_fingerprint_on_both_sides() -> None:
    # Backward compatibility: runs recorded before target-fingerprint
    # capture even existed will both have None here, and that must still
    # be treated as a fine, matching comparison.
    left = _run(target_fingerprint=None)
    right = _run(target_fingerprint=None)
    result = compare_runs(left, right, seed=1)
    assert result.paired_count == 2


# --- Environment and code fingerprint comparison (ADR-0015) -------------------
#
# environment_fingerprint and code_fingerprint get checked with the exact
# same rule already proven above for target_fingerprint: None on both sides
# is fine, but None on one side against a pinned value on the other side is
# a mismatch.


def test_rejects_different_environment_fingerprints() -> None:
    left = _run(environment_fingerprint="sha256:env-aaaa")
    right = _run(environment_fingerprint="sha256:env-bbbb")
    with pytest.raises(IncompatibleRuns, match="environment fingerprint"):
        compare_runs(left, right, seed=1)


def test_rejects_different_code_fingerprints() -> None:
    left = _run(code_fingerprint="sha256:code-aaaa")
    right = _run(code_fingerprint="sha256:code-bbbb")
    with pytest.raises(IncompatibleRuns, match="code fingerprint"):
        compare_runs(left, right, seed=1)


def test_accepts_none_environment_fingerprint_on_both_sides() -> None:
    # Backward compatibility: runs recorded before provenance.py even
    # existed will both have None here, and that must still compare fine.
    left = _run(environment_fingerprint=None)
    right = _run(environment_fingerprint=None)
    result = compare_runs(left, right, seed=1)
    assert result.paired_count == 2


def test_accepts_none_code_fingerprint_on_both_sides() -> None:
    left = _run(code_fingerprint=None)
    right = _run(code_fingerprint=None)
    result = compare_runs(left, right, seed=1)
    assert result.paired_count == 2


def test_rejects_none_environment_fingerprint_against_a_pinned_fingerprint() -> None:
    # An unknown/unrecorded fingerprint (None) must never silently be
    # treated as equal to a verified, pinned-down fingerprint -- no matter
    # which side of the comparison it is on.
    left = _run(environment_fingerprint=None)
    right = _run(environment_fingerprint="sha256:env-aaaa")
    with pytest.raises(IncompatibleRuns, match="environment fingerprint"):
        compare_runs(left, right, seed=1)

    left = _run(environment_fingerprint="sha256:env-aaaa")
    right = _run(environment_fingerprint=None)
    with pytest.raises(IncompatibleRuns, match="environment fingerprint"):
        compare_runs(left, right, seed=1)


def test_rejects_none_code_fingerprint_against_a_pinned_fingerprint() -> None:
    left = _run(code_fingerprint=None)
    right = _run(code_fingerprint="sha256:code-aaaa")
    with pytest.raises(IncompatibleRuns, match="code fingerprint"):
        compare_runs(left, right, seed=1)

    left = _run(code_fingerprint="sha256:code-aaaa")
    right = _run(code_fingerprint=None)
    with pytest.raises(IncompatibleRuns, match="code fingerprint"):
        compare_runs(left, right, seed=1)


# --- allow_cross_environment waiver (ADR-0015) --------------------------------


def test_allow_cross_environment_waives_environment_fingerprint_mismatch() -> None:
    left = _run(environment_fingerprint="sha256:env-aaaa")
    right = _run(environment_fingerprint="sha256:env-bbbb")
    result = compare_runs(left, right, seed=1, allow_cross_environment=True)
    assert result.waived_provenance_fields == ("environment_fingerprint",)
    assert result.paired_count == 2


def test_allow_cross_environment_waives_code_fingerprint_mismatch() -> None:
    left = _run(code_fingerprint="sha256:code-aaaa")
    right = _run(code_fingerprint="sha256:code-bbbb")
    result = compare_runs(left, right, seed=1, allow_cross_environment=True)
    assert result.waived_provenance_fields == ("code_fingerprint",)


def test_allow_cross_environment_waives_both_fingerprints_together() -> None:
    left = _run(environment_fingerprint="sha256:env-aaaa", code_fingerprint="sha256:code-aaaa")
    right = _run(environment_fingerprint="sha256:env-bbbb", code_fingerprint="sha256:code-bbbb")
    result = compare_runs(left, right, seed=1, allow_cross_environment=True)
    assert result.waived_provenance_fields == ("environment_fingerprint", "code_fingerprint")


def test_allow_cross_environment_waived_fields_empty_when_nothing_differs() -> None:
    # Setting allow_cross_environment=True does not waive anything by
    # itself if nothing actually differs -- the resulting tuple must be
    # empty, not a placeholder that just means "a waiver was available if
    # it had been needed."
    left = _run()
    right = _run()
    result = compare_runs(left, right, seed=1, allow_cross_environment=True)
    assert result.waived_provenance_fields == ()


def test_allow_cross_environment_does_not_waive_non_waivable_mismatch() -> None:
    # Proof that the waiver is properly scoped: mismatching a waivable
    # field (environment_fingerprint) at the same time as a non-waivable
    # field (adapter) must still raise IncompatibleRuns, and the error
    # message must name only the non-waivable mismatch -- even with
    # allow_cross_environment=True.
    left = _run(environment_fingerprint="sha256:env-aaaa", adapter="gsm8k@1")
    right = _run(environment_fingerprint="sha256:env-bbbb", adapter="gsm8k@2")
    with pytest.raises(IncompatibleRuns) as excinfo:
        compare_runs(left, right, seed=1, allow_cross_environment=True)
    message = str(excinfo.value)
    assert "adapter" in message
    assert "environment fingerprint" not in message


def test_allow_cross_environment_defaults_to_false() -> None:
    # Regression check: leaving the flag out entirely must still block a
    # mismatched environment_fingerprint, confirming this parameter
    # defaults to the strict/safe behavior rather than the permissive one.
    left = _run(environment_fingerprint="sha256:env-aaaa")
    right = _run(environment_fingerprint="sha256:env-bbbb")
    with pytest.raises(IncompatibleRuns, match="environment fingerprint"):
        compare_runs(left, right, seed=1)


def test_allow_cross_environment_is_keyword_only_and_defaults_to_false() -> None:
    import inspect

    signature = inspect.signature(compare_runs)
    parameter = signature.parameters["allow_cross_environment"]
    assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


# --- ComparisonResult.waived_provenance_fields round-trip ---------------------


def test_comparison_result_round_trips_with_waived_provenance_fields_populated() -> None:
    left = _run(environment_fingerprint="sha256:env-aaaa")
    right = _run(environment_fingerprint="sha256:env-bbbb")
    result = compare_runs(left, right, seed=1, allow_cross_environment=True)
    assert result.waived_provenance_fields == ("environment_fingerprint",)
    restored = ComparisonResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_comparison_result_round_trips_with_waived_provenance_fields_empty() -> None:
    left = _run()
    right = _run()
    result = compare_runs(left, right, seed=1)
    assert result.waived_provenance_fields == ()
    restored = ComparisonResult.model_validate_json(result.model_dump_json())
    assert restored == result


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
    # right passed both samples, left passed only one, so the difference
    # (right's pass rate minus left's) works out to +0.5.
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
            # s1 does not appear anywhere in the right run, and s2 only
            # exists in the right run -- so neither one has a matching
            # counterpart on both sides, and both get left out of the
            # comparison.
            _sample_result("s2", attempt=1, passed=True),
        ),
    )
    result = compare_runs(left, right, bootstrap_samples=200, seed=3)
    assert result.paired_count == 1
    assert result.sample_count == 1


def test_zero_paired_overlap_raises_incompatible_runs_naming_both_run_ids() -> None:
    # left and right otherwise match on every provenance field, but they
    # share no (sample_id, attempt) pairs at all -- there is nothing to
    # compute a difference from. This must fail loudly with an error
    # rather than quietly returning a plausible-looking "no difference"
    # answer that was actually computed from nothing.
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
    # 2 distinct questions (sample_ids), each attempted 3 times, and every
    # attempt overlaps between left and right. paired_count must be 6 (one
    # pair for each (sample_id, attempt) combination), but sample_count
    # must stay at 2 -- it only counts distinct sample_ids, so unlike
    # paired_count, it must never get multiplied up by the number of
    # attempts.
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
    # The point estimate (the actual observed difference between the two
    # runs' paired results) does not depend on randomness at all, so it
    # comes out identical no matter which seed is used. Only the
    # bootstrap's confidence-interval bounds depend on the seed, since
    # those come from the random resampling draws.
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
