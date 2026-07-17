"""Tests for the harness data shapes, and for how harness outcomes stay distinct (design §7.1).

("Harness" means an external, official tool that actually checks whether a
submitted fix works -- see :mod:`agentic_evalkit.benchmarks.harness` for the
full explanation.) This file covers two things required by this project's
written implementation plan (Step 4, function
``test_missing_harness_is_unavailable_not_failed`` below; and Step 7): that
a ``HarnessRequest``/``HarnessResult`` can be converted to JSON and read
back without losing any information, and that ``FakeHarnessExecutor`` (a
test-only stand-in harness) keeps four different outcomes -- "harness
unavailable," "harness broke with an infrastructure error," "harness ran
and confirmed the issue is still broken," and "harness ran and confirmed
the issue is fixed" -- distinct from each other, instead of collapsing them
all down into a single true/false.
"""

from datetime import UTC, datetime

import pytest
from _benchmark_fixtures import _harness_request
from pydantic import ValidationError

from agentic_evalkit.benchmarks.harness import (
    FakeHarnessExecutor,
    HarnessRequest,
    HarnessResult,
    HarnessStatus,
    UnavailableHarnessExecutor,
)
from agentic_evalkit.models import GradeResult, GradeStatus


def test_harness_request_round_trips_through_versioned_json() -> None:
    request = _harness_request()
    assert HarnessRequest.model_validate_json(request.model_dump_json()) == request


def test_harness_result_round_trips_through_versioned_json() -> None:
    result = HarnessResult(
        status=HarnessStatus.COMPLETED,
        resolved=True,
        message="issue resolved",
        evidence={"tests_passed": ["test_x"]},
        logs=("running tests...", "PASSED"),
        image_digests={"base": "sha256:abc"},
        error=None,
    )
    assert HarnessResult.model_validate_json(result.model_dump_json()) == result


def test_harness_result_is_frozen_and_forbids_unknown_fields() -> None:
    result = HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="ok")
    with pytest.raises(ValidationError):
        result.message = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        HarnessResult(  # type: ignore[call-arg]
            status=HarnessStatus.COMPLETED, resolved=True, message="ok", unknown=True
        )


@pytest.mark.asyncio
async def test_unavailable_harness_executor_reports_unavailable_with_install_hint() -> None:
    result = await UnavailableHarnessExecutor("install agentic-evalkit[swebench]").execute(
        _harness_request()
    )
    assert result.status == HarnessStatus.UNAVAILABLE
    assert result.resolved is None
    assert "agentic-evalkit[swebench]" in result.message


@pytest.mark.asyncio
async def test_fake_harness_executor_keeps_four_outcomes_distinct() -> None:
    """Four outcomes must never blur into each other: unavailable, infrastructure
    error, resolved=False (confirmed still broken), and resolved=True (confirmed fixed)."""
    unavailable = HarnessResult(status=HarnessStatus.UNAVAILABLE, resolved=None, message="n/a")
    infra_error = HarnessResult(
        status=HarnessStatus.ERROR,
        resolved=None,
        message="container failed to start",
        error={"code": "image_pull_failed"},
    )
    unresolved = HarnessResult(status=HarnessStatus.COMPLETED, resolved=False, message="failed")
    resolved = HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="passed")

    executor = FakeHarnessExecutor(
        results_by_sample_id={
            "unavailable-case": unavailable,
            "error-case": infra_error,
            "unresolved-case": unresolved,
            "resolved-case": resolved,
        }
    )

    outcomes = {
        sample_id: await executor.execute(_harness_request(sample_id))
        for sample_id in ("unavailable-case", "error-case", "unresolved-case", "resolved-case")
    }

    assert outcomes["unavailable-case"].status is HarnessStatus.UNAVAILABLE
    assert outcomes["error-case"].status is HarnessStatus.ERROR
    assert outcomes["unresolved-case"].status is HarnessStatus.COMPLETED
    assert outcomes["unresolved-case"].resolved is False
    assert outcomes["resolved-case"].status is HarnessStatus.COMPLETED
    assert outcomes["resolved-case"].resolved is True

    # Check that all four results are genuinely different from each other --
    # for example, that "the harness broke" never accidentally looks
    # identical to "the harness ran and said the issue isn't fixed."
    all_results = list(outcomes.values())
    assert len({result.model_dump_json() for result in all_results}) == 4


@pytest.mark.asyncio
async def test_fake_harness_executor_raises_for_unconfigured_sample() -> None:
    executor = FakeHarnessExecutor(
        results_by_sample_id={
            "known": HarnessResult(status=HarnessStatus.COMPLETED, resolved=True, message="ok")
        }
    )
    with pytest.raises(KeyError):
        await executor.execute(_harness_request("unknown-sample"))


def test_generic_grade_cannot_claim_resolved_without_a_harness_result() -> None:
    """A plain GradeResult has no way to claim a bug is "resolved" -- only a real HarnessResult can.

    This is guaranteed by the shape of the data itself, not by some check
    that runs at test time: `GradeResult` (design §5.5) simply has no
    `resolved` field defined on it anywhere, so there is no way for ordinary
    grading code to sneak a "yes, this is genuinely fixed" claim out through
    a plain grade. A real claim that something is fixed can only ever come
    from an actual `HarnessResult.resolved` -- the product of really running
    the harness, not an approximate grade standing in for one.
    """
    grade = GradeResult(
        sample_id="org__repo-1",
        grader="swebench-harness@1",
        status=GradeStatus.PASS,
        score=1.0,
        hard_gate=True,
        created_at=datetime.now(UTC),
    )
    assert not hasattr(grade, "resolved")
