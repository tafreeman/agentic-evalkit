"""Exact/normalized-match objective grader (design §9, plan Task 10 Step 4).

``ExactMatchGrader`` deliberately takes an *injected* extractor callable
instead of importing one from ``agentic_evalkit.benchmarks``: this package
owns grading policy only, not benchmark projection. A benchmark adapter
(e.g. GSM8K's ``extract_final_answer``) is wired in by the caller through
``EvalSample.grader`` / ``GraderSpec`` in a later task.
"""

import re
import unicodedata
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
)

# Matches a decimal number, optionally thousands-separated with commas, so
# "1,234" and "1234" and "5.0" canonicalize to the same numeric string.
_NUMERIC_PATTERN = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _canonicalize(text: str, *, case_fold: bool) -> str:
    """Normalize Unicode form, whitespace, casing, and numeric shape.

    Order matters: Unicode normalization first (so later regexes see a
    stable form), then whitespace collapsing, then optional case folding,
    then numeric canonicalization last (it only fires on the fully
    normalized string).
    """
    normalized = unicodedata.normalize("NFC", text)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    if case_fold:
        normalized = normalized.casefold()
    return _canonicalize_number(normalized)


def _canonicalize_number(text: str) -> str:
    if not _NUMERIC_PATTERN.match(text):
        return text
    without_separators = text.replace(",", "")
    try:
        as_float = float(without_separators)
    except ValueError:
        return text
    if as_float == int(as_float):
        return str(int(as_float))
    return repr(as_float)


class ExactMatchGrader:
    """Compares an extracted execution output against ``EvalSample.reference``.

    Args:
        name: Stable grader identifier reported on every ``GradeResult``.
        extractor: Injected callable that pulls the comparable text out of
            ``NormalizedExecutionResult.output``. Kept generic
            (``Mapping[str, object] -> str``) so this module never imports
            benchmark-specific extraction logic.
        case_fold: When ``True``, comparison is case-insensitive.
    """

    def __init__(
        self,
        *,
        name: str,
        extractor: Callable[[Mapping[str, object]], str],
        case_fold: bool = False,
    ) -> None:
        self._name = name
        self._extractor = extractor
        self._case_fold = case_fold

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        now = datetime.now(UTC)
        if execution.status is not ExecutionStatus.COMPLETED or execution.output is None:
            return GradeResult(
                sample_id=sample.sample_id,
                grader=self._name,
                grader_type="exact_match",
                status=GradeStatus.UNAVAILABLE,
                score=None,
                hard_gate=False,
                evidence={"reason": "execution did not complete"},
                created_at=now,
            )
        if sample.reference is None:
            return GradeResult(
                sample_id=sample.sample_id,
                grader=self._name,
                grader_type="exact_match",
                status=GradeStatus.ABSTAIN,
                score=None,
                hard_gate=False,
                evidence={"reason": "sample has no reference answer"},
                created_at=now,
            )

        extracted = self._extractor(execution.output)
        canonical_extracted = _canonicalize(extracted, case_fold=self._case_fold)
        canonical_reference = _canonicalize(sample.reference, case_fold=self._case_fold)
        is_match = canonical_extracted == canonical_reference

        return GradeResult(
            sample_id=sample.sample_id,
            grader=self._name,
            grader_type="exact_match",
            status=GradeStatus.PASS if is_match else GradeStatus.FAIL,
            score=1.0 if is_match else 0.0,
            hard_gate=False,
            evidence={
                "extracted": extracted,
                "reference": sample.reference,
                "canonical_extracted": canonical_extracted,
                "canonical_reference": canonical_reference,
            },
            created_at=now,
        )
