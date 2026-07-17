"""Tests for :class:`agentic_evalkit.graders.exact.ExactMatchGrader`.

The grader is handed its "extractor" -- the function that pulls the actual
answer text out of the raw execution output -- as a parameter by the
caller, rather than importing one directly from
``agentic_evalkit.benchmarks``. This keeps a firm boundary in place (drawn
deliberately in Task 10): this grader only knows how to *compare* two
answers, never anything about any specific benchmark's output format.
Wiring a grader up to a real benchmark's own extractor happens later, in
Task 14, via ``EvalSample.grader`` / ``GraderSpec``.
"""

from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from agentic_evalkit.graders.exact import ExactMatchGrader
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeStatus,
    NormalizedExecutionResult,
)


def _sample(reference: str | None = "42") -> EvalSample:
    return EvalSample(
        sample_id="s1",
        input={"question": "What is 6 * 7?"},
        reference=reference,
        source_digest="sha256:row",
        adapter="identity@1",
    )


def _execution(output_text: str) -> NormalizedExecutionResult:
    now = datetime.now(UTC)
    return NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output={"answer": output_text},
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )


def _extract_answer_field(output: Mapping[str, object]) -> str:
    """A minimal example of the extractor function a caller might supply:
    it just reads one text field out of the output."""
    value = output["answer"]
    assert isinstance(value, str)
    return value


@pytest.mark.asyncio
async def test_exact_match_on_identical_text() -> None:
    grader = ExactMatchGrader(name="exact@1", extractor=_extract_answer_field)
    result = await grader.grade(_sample(reference="42"), _execution("42"))
    assert result.status is GradeStatus.PASS
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_exact_match_fails_on_different_text() -> None:
    grader = ExactMatchGrader(name="exact@1", extractor=_extract_answer_field)
    result = await grader.grade(_sample(reference="42"), _execution("41"))
    assert result.status is GradeStatus.FAIL
    assert result.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_whitespace_is_normalized() -> None:
    grader = ExactMatchGrader(name="exact@1", extractor=_extract_answer_field)
    result = await grader.grade(_sample(reference="hello world"), _execution("  hello   world  "))
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_unicode_is_normalized() -> None:
    # The word "café" can be represented two different ways in Unicode: as
    # one single character for "é" (called NFC form), or as a plain "e"
    # plus a separate combining accent mark that renders on top of it
    # (called NFD form). They look identical but are different bytes --
    # this test checks that the grader treats them as the same answer.
    grader = ExactMatchGrader(name="exact@1", extractor=_extract_answer_field)
    result = await grader.grade(_sample(reference="café"), _execution("café"))
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_case_folding_is_opt_in() -> None:
    strict = ExactMatchGrader(name="exact@1", extractor=_extract_answer_field, case_fold=False)
    result = await strict.grade(_sample(reference="Yes"), _execution("yes"))
    assert result.status is GradeStatus.FAIL

    lenient = ExactMatchGrader(name="exact@1", extractor=_extract_answer_field, case_fold=True)
    result = await lenient.grade(_sample(reference="Yes"), _execution("yes"))
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_numeric_canonicalization_treats_equivalent_numbers_as_equal() -> None:
    grader = ExactMatchGrader(name="exact@1", extractor=_extract_answer_field)
    result = await grader.grade(_sample(reference="5"), _execution("5.0"))
    assert result.status is GradeStatus.PASS

    result = await grader.grade(_sample(reference="1,234"), _execution("1234"))
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_missing_reference_abstains_rather_than_failing() -> None:
    grader = ExactMatchGrader(name="exact@1", extractor=_extract_answer_field)
    result = await grader.grade(_sample(reference=None), _execution("42"))
    assert result.status is GradeStatus.ABSTAIN


@pytest.mark.asyncio
async def test_non_completed_execution_is_not_gradable() -> None:
    grader = ExactMatchGrader(name="exact@1", extractor=_extract_answer_field)
    now = datetime.now(UTC)
    failed = NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output=None,
        status=ExecutionStatus.ERROR,
        started_at=now,
        finished_at=now,
    )
    result = await grader.grade(_sample(reference="42"), failed)
    assert result.status is GradeStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_grade_result_names_the_grader_and_sample() -> None:
    grader = ExactMatchGrader(name="exact@1", extractor=_extract_answer_field)
    result = await grader.grade(_sample(reference="42"), _execution("42"))
    assert result.sample_id == "s1"
    assert result.grader == "exact@1"
