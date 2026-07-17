"""A grader checking whether the AI's answer matches the expected one, after normalizing formatting.

Design §9, plan Task 10 Step 4.

``ExactMatchGrader`` deliberately doesn't know how to pull "the actual
answer" out of a raw execution output by itself. Instead, the caller hands
it a small function (an "extractor") that does that extraction, and this
grader just calls it. This keeps a firm boundary: this package (``graders``)
only owns the policy of *how to compare* two answers and decide pass/fail;
it has no knowledge of any specific benchmark's output format. For example,
GSM8K (a benchmark of grade-school math word problems) has its own
``extract_final_answer`` function that knows how to pull the final numeric
answer out of a full written solution -- that function gets handed to this
grader by the caller (wired up via ``EvalSample.grader`` / ``GraderSpec`` in
a later task), instead of this module importing GSM8K-specific code itself.
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

# Matches a decimal number, optionally with commas every three digits (like
# "1,234"), so that "1,234" and "1234" and "5.0" all get rewritten to the
# same normalized numeric string (see `_canonicalize_number` below).
_NUMERIC_PATTERN = re.compile(r"^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _canonicalize(text: str, *, case_fold: bool) -> str:
    """Rewrite ``text`` into a normalized form, so equivalent, differently-formatted answers match.

    Four steps, applied in this order because each one depends on the
    previous one having already run: first normalize the Unicode encoding
    (so the pattern-matching below sees a predictable, stable form), then
    collapse any run of whitespace into a single space, then optionally
    lowercase everything (only when ``case_fold=True``), and finally --
    only once the text is already in this cleaned-up form -- try to rewrite
    it as a normalized number (see ``_canonicalize_number`` below), so that,
    for example, "5.0" and "5" end up identical.
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
    """Grades a sample by comparing the AI's extracted answer against the expected reference answer.

    Args:
        name: A stable label for this grader, recorded on every
            ``GradeResult`` so you can tell which grader produced it.
        extractor: A function, supplied by the caller, that pulls the
            comparable piece of text out of
            ``NormalizedExecutionResult.output`` (the AI's raw output).
            Kept as a generic function signature
            (``Mapping[str, object] -> str``) specifically so this module
            never has to import any benchmark-specific extraction code
            itself -- see the module docstring above.
        case_fold: When ``True``, the comparison ignores letter casing
            (e.g. "Paris" would match "paris").
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
