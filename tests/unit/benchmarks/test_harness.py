"""Harness contract serialization and outcome-discrimination tests (design §7.1).

Verbatim plan Step 4 (test_missing_harness_is_unavailable_not_failed) and
Step 7 requirements live here: round-trip HarnessRequest/HarnessResult JSON,
and prove FakeHarnessExecutor keeps unavailable, infrastructure-error,
resolved=False, and resolved=True outcomes distinct rather than collapsing
them into a boolean.
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
    """unavailable, infrastructure error, resolved=False, and resolved=True never collapse."""
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

    # All four results are pairwise distinct: no outcome silently degrades
    # into another.
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
    """A GradeResult alone carries no `resolved` field: only a HarnessResult can assert it.

    This is a type-level guarantee, not a runtime check: GradeResult (design
    §5.5) has no `resolved` attribute at all, so grading code cannot smuggle
    an authoritative resolution verdict out of a plain grade. Authoritative
    resolution can only ever come from a real HarnessResult.resolved.
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
