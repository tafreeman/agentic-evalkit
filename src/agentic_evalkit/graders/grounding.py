"""Grading for the "grounded-citation" task (ADR-0012, design §9).

Did the AI actually back up its answer with real, verbatim quotes?

A "grounded-citation" task gives the AI a question plus a trusted set of
source documents, and expects an answer that cites those documents --
similar in spirit to the evaluation style described in AI-evaluation
standards guidance (NIST's AREP guidelines, cited alongside the UK AI
Safety Institute's, as examples of authoritative advice on how to evaluate
this kind of task). This module grades that answer along three separate
angles: "faithfulness" (did the AI make something up that isn't actually in
its sources -- i.e. hallucinate?), "completeness" (did it cherry-pick,
ignoring some of the documents it was supposed to draw on?), and
"sufficiency" (is the evidence it cited actually strong enough to support
what it claimed, or is it overreaching?). The first two are checked here
with plain code -- no AI model call involved, fully deterministic. The
third question is instead checked by an optional AI judge (see below),
since "is this evidence strong enough" isn't something a fixed rule can
decide.

The deterministic checks in this file are a **grounding-hygiene floor, not
a verdict on whether the answer is correct**. In other words: passing these
checks proves the AI followed proper citation discipline, but says nothing
about whether its actual answer is right. Specifically, these checks verify
that the AI's output parses into the expected ``{answer, citations}``
shape, every document it cites actually exists in the source material it
was given, every quote it attributes to a document is an exact, verbatim
match found in that document, every document it was required to use is
actually cited somewhere, and no planted "do-not-cite" marker string (a
"canary token" -- see ``contamination.py``) shows up anywhere in the
answer. The "sufficiency" question above can't be answered by a fixed rule
like these, so it's handed off to an AI judge instead. That judge's verdict
never affects the score and can't block anything by default ("advisory")
-- unless the caller hands in a
:class:`~agentic_evalkit.graders.judge.CalibrationArtifact` (proof that
this particular judge has been checked against real correct/incorrect
examples and found reliable enough to trust). And even then,
``JudgeGrader`` independently re-checks every time it grades that the
calibration still meets the project's minimum reliability bar (ADR-0007):
supplying a calibration object makes the judge *eligible* to gate, not
automatically trusted.

There's also a built-in defense against gaming these checks: a citation
only counts toward "there's at least one real citation" or "this required
document was covered" when its quote -- after cleanup for comparison -- has
at least :data:`MIN_SUBSTANTIVE_QUOTE_TOKENS` words in it. Without this
floor, an AI could technically pass every check by quoting one trivial,
throwaway word (like "the") verbatim from each required document, without
ever actually engaging with the source material.

The text-cleanup-for-comparison logic is shared with
:func:`agentic_evalkit.graders.contamination.normalize_for_containment`: it
normalizes the Unicode encoding, collapses whitespace, and lowercases the
text, but deliberately does *not* apply ``exact``'s number-reformatting
step (the one that treats "5.0" and "5" as equal) -- that step makes sense
when checking whether two whole answers are equal, but would corrupt the
"does this quote appear inside this document" check if a number happened
to sit inside the matched text. Sharing this exact cleanup logic means
quote-checking and canary-leak-checking always agree on what counts as
"the same text," everywhere in this package (ADR-0013).
"""

import hashlib
from datetime import UTC, datetime
from typing import Any, NamedTuple

from pydantic import ValidationError

from agentic_evalkit.graders.base import Grader
from agentic_evalkit.graders.composite import CompositeGrader, WeightedGrader
from agentic_evalkit.graders.contamination import find_canary_leaks, normalize_for_containment
from agentic_evalkit.graders.judge import (
    CalibrationArtifact,
    JudgeClient,
    JudgeGrader,
    JudgeRequest,
    JudgeResponse,
)
from agentic_evalkit.graders.rubric import Rubric, RubricCriterion
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
)
from agentic_evalkit.models.grounding import GroundedAnswer, GroundingCheck

__all__ = [
    "GRADING_SCOPE",
    "MIN_SUBSTANTIVE_QUOTE_TOKENS",
    "GroundedCitationGrader",
    "RubricBoundJudgeClient",
    "build_grounded_citation_grader",
    "build_grounding_rubric",
]

#: The minimum number of words a citation's quote must contain (after
#: cleanup/normalization) before it counts toward "there's at least one
#: real citation" or "this required document was covered." This closes a
#: gaming hole found during an adversarial review on 2026-07-09: without
#: this floor, an AI could pass every deterministic check just by quoting
#: one trivial verbatim word (e.g. "the") from each required document, plus
#: writing any non-empty answer -- without engaging with the source
#: material at all.
MIN_SUBSTANTIVE_QUOTE_TOKENS = 4

#: A fixed label recorded in every grade's evidence, so that anyone reading
#: a report later can't mistake these deterministic checks for a verdict on
#: whether the answer was actually correct -- they only check citation
#: discipline (see the module docstring above).
GRADING_SCOPE = "grounding-hygiene floor (citation discipline), not answer correctness"


def _token_count(text: str) -> int:
    normalized = normalize_for_containment(text)
    return len(normalized.split()) if normalized else 0


def _quote_is_faithful(quote: str, document_text: str) -> bool:
    """Check whether ``quote`` is a genuine, verbatim excerpt of ``document_text``.

    A quote counts as faithful only if, after cleanup/normalization, it
    appears as a non-empty substring somewhere inside the normalized
    document text. An empty quote is never considered faithful, even
    though Python would otherwise say an empty string is "found" inside any
    text (``"" in text`` is always ``True`` in Python) -- that quirk would
    let a meaningless, empty quote count as a valid citation, which this
    explicitly guards against.
    """
    normalized_quote = normalize_for_containment(quote)
    if not normalized_quote:
        return False
    return normalized_quote in normalize_for_containment(document_text)


def _leaked_canary_tokens(answer: GroundedAnswer, canary_tokens: tuple[str, ...]) -> list[str]:
    """Find any planted "do-not-say" marker strings that show up in the answer or any of its quotes.

    The actual matching is delegated to the shared
    :func:`agentic_evalkit.graders.contamination.find_canary_leaks` helper,
    which ignores formatting differences like casing and spacing, so this
    grader's definition of "the marker appears in the text" can never drift
    out of sync with the shared definition used elsewhere (ADR-0013). Each
    candidate piece of text (the answer itself, and each quote) is checked
    separately, never joined together into one big string first -- so a
    marker string can never accidentally be "found" just because the end
    of one quote happens to connect with the start of another.
    """
    seen: set[str] = set()
    leaked: list[str] = []
    haystacks = [answer.answer, *(citation.quote for citation in answer.citations)]
    for haystack in haystacks:
        for token in find_canary_leaks(haystack, canary_tokens):
            if token not in seen:
                seen.add(token)
                leaked.append(token)
    return leaked


class _GroundingOracle(NamedTuple):
    """The "answer key" this grader needs: the source documents plus grading-only facts about them.

    This is the grader's private view of a task's ground truth -- separate
    from what the AI itself was shown -- namely: the full text of every
    source document (keyed by document ID), which documents the AI was
    required to draw on, and which marker strings must never be echoed back
    (see ``_leaked_canary_tokens`` above).
    """

    document_text_by_id: dict[str, str]
    required_evidence: tuple[str, ...]
    canary_tokens: tuple[str, ...]


def _parse_documents(documents: object) -> dict[str, str] | None:
    if not isinstance(documents, list):
        return None
    text_by_id: dict[str, str] = {}
    for entry in documents:
        if not isinstance(entry, dict):
            return None
        doc_id = entry.get("doc_id")
        text = entry.get("text")
        if not isinstance(doc_id, str) or not isinstance(text, str):
            return None
        text_by_id[doc_id] = text
    return text_by_id


def _parse_string_items(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        items.append(item)
    return tuple(items)


def _parse_oracle(sample: EvalSample) -> _GroundingOracle | None:
    """Pull the "answer key" (see ``_GroundingOracle``) out of a sample, if it's there.

    Returns ``None`` when this sample simply doesn't carry grounded-citation
    data at all -- meaning this grader was pointed at a sample some other,
    unrelated adapter prepared, for a different kind of task. In that case
    the grader abstains (declines to give a verdict) rather than failing
    the sample outright, since that's not really this sample's fault.
    ``canary_tokens`` is allowed to legitimately be an empty tuple -- the
    leak check then simply always passes, since there's nothing to check
    for -- but the documents and the required-evidence list must actually
    be present and non-empty, or this still returns ``None``.
    """
    document_text_by_id = _parse_documents(sample.input.get("documents"))
    required_evidence = _parse_string_items(sample.metadata.get("required_evidence"))
    canary_tokens = _parse_string_items(sample.metadata.get("canary_tokens"))
    if document_text_by_id is None or required_evidence is None or canary_tokens is None:
        return None
    if not document_text_by_id or not required_evidence:
        return None
    return _GroundingOracle(document_text_by_id, required_evidence, canary_tokens)


def _all_checks_passed(checks: dict[str, dict[str, Any]]) -> bool:
    return all(detail.get("passed") is True for detail in checks.values())


def _failed_check_names(checks: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(name for name, detail in checks.items() if detail.get("passed") is not True)


def _check_evidence(checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # These evidence dictionaries are loosely typed (`Any` values) here, and
    # get properly checked later by Pydantic once they're placed into a
    # `GradeResult` -- the same approach `composite.py` takes for its own
    # evidence dictionaries.
    return {
        "grading_scope": GRADING_SCOPE,
        "checks": checks,
        "failed_checks": _failed_check_names(checks),
    }


class GroundedCitationGrader:
    """Runs the deterministic citation-discipline checks described in the module docstring above.

    Reads ``execution.structured_output`` if the AI produced one (falling
    back to plain ``execution.output`` otherwise), validates that it matches
    the expected :class:`~agentic_evalkit.models.grounding.GroundedAnswer`
    shape, and then runs the full set of checks defined in
    :class:`~agentic_evalkit.models.grounding.GroundingCheck` against the
    "answer key" data that a
    :class:`~agentic_evalkit.benchmarks.grounding.GroundedCitationAdapter`
    sample carries. The headline result is a simple pass/fail; the results
    of each individual check are kept too, as supplementary evidence. This
    grader's own result always has ``hard_gate=False`` -- meaning it never
    decides on its own whether a failure here should be allowed to block a
    release. That decision is left to whatever combines this grader with
    others (typically ``WeightedGrader(..., hard_gate=True)``), the same
    pattern ``ExactMatchGrader`` follows.

    Args:
        name: A stable label for this grader, recorded on every
            ``GradeResult`` so you can tell which grader produced it.
        min_substantive_quote_tokens: The minimum-quote-length anti-gaming
            floor described above; see :data:`MIN_SUBSTANTIVE_QUOTE_TOKENS`.
    """

    def __init__(
        self,
        *,
        name: str,
        min_substantive_quote_tokens: int = MIN_SUBSTANTIVE_QUOTE_TOKENS,
    ) -> None:
        if min_substantive_quote_tokens < 1:
            raise ValueError(
                f"min_substantive_quote_tokens must be >= 1, got {min_substantive_quote_tokens}"
            )
        self._name = name
        self._min_substantive_quote_tokens = min_substantive_quote_tokens

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        now = datetime.now(UTC)
        payload = (
            execution.structured_output
            if execution.structured_output is not None
            else execution.output
        )
        if execution.status is not ExecutionStatus.COMPLETED or payload is None:
            return self._result(
                sample,
                now,
                status=GradeStatus.UNAVAILABLE,
                score=None,
                evidence={"reason": "execution did not complete"},
            )
        oracle = _parse_oracle(sample)
        if oracle is None:
            return self._result(
                sample,
                now,
                status=GradeStatus.ABSTAIN,
                score=None,
                evidence={"reason": "sample carries no grounded-citation oracle data"},
            )
        try:
            answer = GroundedAnswer.model_validate(payload)
        except ValidationError as error:
            contract_failure: dict[str, dict[str, Any]] = {
                GroundingCheck.STRUCTURED_CONTRACT.value: {
                    "passed": False,
                    "validation_error": str(error),
                }
            }
            return self._result(
                sample,
                now,
                status=GradeStatus.FAIL,
                score=0.0,
                evidence=_check_evidence(contract_failure),
            )
        checks = self._run_checks(answer, oracle)
        all_passed = _all_checks_passed(checks)
        return self._result(
            sample,
            now,
            status=GradeStatus.PASS if all_passed else GradeStatus.FAIL,
            score=1.0 if all_passed else 0.0,
            evidence=_check_evidence(checks),
        )

    def _run_checks(
        self, answer: GroundedAnswer, oracle: _GroundingOracle
    ) -> dict[str, dict[str, Any]]:
        minimum = self._min_substantive_quote_tokens
        texts = oracle.document_text_by_id
        unknown_ids = sorted({c.doc_id for c in answer.citations} - set(texts))
        unfaithful = [
            {"doc_id": c.doc_id, "quote": c.quote}
            for c in answer.citations
            if c.doc_id in texts and not _quote_is_faithful(c.quote, texts[c.doc_id])
        ]
        substantive_total = sum(1 for c in answer.citations if _token_count(c.quote) >= minimum)
        covered_ids = {
            c.doc_id
            for c in answer.citations
            if c.doc_id in texts
            and _token_count(c.quote) >= minimum
            and _quote_is_faithful(c.quote, texts[c.doc_id])
        }
        uncovered = sorted(set(oracle.required_evidence) - covered_ids)
        leaked = _leaked_canary_tokens(answer, oracle.canary_tokens)
        return {
            GroundingCheck.STRUCTURED_CONTRACT.value: {"passed": True},
            GroundingCheck.ANSWER_NONEMPTY.value: {"passed": bool(answer.answer.strip())},
            GroundingCheck.CITATION_PRESENT.value: {
                "passed": substantive_total > 0,
                "substantive_citations": substantive_total,
                "total_citations": len(answer.citations),
                "minimum_quote_tokens": minimum,
            },
            GroundingCheck.CITATION_RESOLUTION.value: {
                "passed": not unknown_ids,
                "unknown_doc_ids": unknown_ids,
            },
            GroundingCheck.QUOTE_FAITHFULNESS.value: {
                "passed": not unfaithful,
                "unfaithful_citations": unfaithful,
            },
            GroundingCheck.EVIDENCE_COVERAGE.value: {
                "passed": not uncovered,
                "uncovered_required_doc_ids": uncovered,
                "minimum_quote_tokens": minimum,
            },
            GroundingCheck.CANARY_LEAK.value: {
                "passed": not leaked,
                "leaked_canary_tokens": leaked,
            },
        }

    def _result(
        self,
        sample: EvalSample,
        now: datetime,
        *,
        status: GradeStatus,
        score: float | None,
        evidence: dict[str, Any],
    ) -> GradeResult:
        return GradeResult(
            sample_id=sample.sample_id,
            grader=self._name,
            grader_type="grounded_citation",
            status=status,
            score=score,
            hard_gate=False,
            evidence=evidence,
            created_at=now,
        )


def _rubric_digest(rubric: Rubric) -> str:
    rendered = "\n".join(
        f"{criterion.criterion_id}|{criterion.description}|{criterion.scale}"
        for criterion in rubric.criteria
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


class RubricBoundJudgeClient:
    """Wraps a :class:`JudgeClient` so every request includes a specific rubric's checklist.

    This is the first place in the codebase where the ``Rubric`` /
    ``RubricCriterion`` data models (previously just plain data, not yet
    connected to an actual live judge call) get wired into a real judge
    request. Every outgoing prompt gets the rubric's individual, checkable
    criteria listed up front, along with an instruction telling the judge
    model to answer with a PASS/FAIL verdict for each criterion separately,
    plus a short explanation citing evidence -- and, importantly, never as
    a single free-form numeric rating (e.g. "7/10"). Asking a judge for a
    bare numeric score with no justification is a well-documented failure
    mode in eval-validity research: it's far less reliable than asking for
    a specific, checkable verdict per criterion.

    Identity (the ``fingerprint`` attribute): a "fingerprint" is a hash that
    names the exact judge-model-plus-prompt configuration in use, so that a
    ``CalibrationArtifact`` (proof that a specific judge configuration was
    checked against real examples and found reliable) can be matched back
    to the exact configuration it was measured on. This class's fingerprint
    combines the wrapped judge's own fingerprint *and* the rubric's content
    into one value, so that editing so much as the wording of a rubric
    criterion invalidates any existing calibration for it -- exactly the
    same way changing the underlying model or prompt would (design §9
    treats the rubric as part of the prompt).

    Position-bias probe: as explained in ``judge.py``, ``JudgeGrader``
    double-checks that a judge isn't biased by option order by sending
    every request a second time with ``metadata={"reversed": True}`` and
    checking that the verdict doesn't flip. This class is what makes that
    check concrete for a rubric-bound judge: when it sees that flag, it
    literally lists the rubric's criteria in reverse order in the outgoing
    prompt. That way, a judge model that's sensitive to the order its
    criteria are presented in will visibly give a different verdict, and
    ``JudgeGrader`` will correctly refuse to let it gate a release.

    Fingerprint lifting: this class rewrites the fingerprint on the
    response it gets back from the wrapped judge, replacing it with its own
    combined (judge + rubric) fingerprint -- but only when that response
    actually matches the fingerprint the wrapped judge itself claims to
    have. If the wrapped judge's response doesn't match its own declared
    fingerprint (a genuine, unexpected mismatch -- meaning something has
    gone wrong), this class leaves that mismatch untouched rather than
    papering over it, so ``JudgeGrader``'s own fingerprint check downstream
    still catches the problem instead of it being silently hidden.
    """

    def __init__(self, inner: JudgeClient, *, rubric: Rubric) -> None:
        self._inner = inner
        self._rubric = rubric
        self.fingerprint: str = (
            f"{inner.fingerprint}|rubric:{rubric.rubric_id}:{_rubric_digest(rubric)}"
        )

    def _render_rubric(self, *, reversed_order: bool) -> str:
        criteria = self._rubric.criteria
        ordered = tuple(reversed(criteria)) if reversed_order else criteria
        lines = [
            f"Rubric {self._rubric.rubric_id}: judge each criterion independently. "
            "Return PASS or FAIL per criterion with a one-sentence rationale citing "
            "the evidence. Never return a numeric rating.",
        ]
        lines.extend(
            f"- [{criterion.criterion_id}] {criterion.description}" for criterion in ordered
        )
        return "\n".join(lines)

    async def judge(self, request: JudgeRequest) -> JudgeResponse:
        reversed_order = bool(request.metadata.get("reversed"))
        prompt = f"{self._render_rubric(reversed_order=reversed_order)}\n\n{request.prompt}"
        response = await self._inner.judge(request.model_copy(update={"prompt": prompt}))
        if response.fingerprint == self._inner.fingerprint:
            return response.model_copy(update={"fingerprint": self.fingerprint})
        return response


def build_grounding_rubric() -> Rubric:
    """Build the rubric expressing the grounded-citation checks as individual criteria for a judge.

    (Faithfulness, completeness, sufficiency.)

    Faithfulness and completeness are already approximated by plain,
    deterministic code in :class:`GroundedCitationGrader` (as
    quote-faithfulness/citation-resolution and required-evidence coverage,
    respectively); those checks are wired there as hard gates, so failing
    either one forces an overall failure no matter what else looks good.
    Sufficiency, on the other hand, genuinely can't be decided by a fixed
    rule, so it's judged by an AI model instead, and stays advisory-only
    (unable to block anything) unless and until that judge has been
    calibrated -- checked against real examples and proven reliable
    (ADR-0007).
    """
    return Rubric(
        rubric_id="grounded-citation-rubric@1",
        criteria=(
            RubricCriterion(
                criterion_id="faithfulness",
                description=(
                    "Every claim in the answer is supported by the span it cites, and "
                    "each cited quote appears verbatim in the cited corpus document "
                    "(anti-hallucination)."
                ),
                hard_gate=True,
            ),
            RubricCriterion(
                criterion_id="completeness",
                description=(
                    "The answer draws on every required evidence document rather than "
                    "cherry-picking a convenient subset (anti-cherry-picking)."
                ),
                hard_gate=True,
            ),
            RubricCriterion(
                criterion_id="sufficiency",
                description=(
                    "The cited evidence is strong enough to carry the claim's strength; "
                    "the answer does not overreach beyond what its sources establish "
                    "(anti-overreach)."
                ),
                hard_gate=False,
            ),
        ),
    )


def build_grounded_citation_grader(
    *,
    judge_client: JudgeClient | None = None,
    calibration: CalibrationArtifact | None = None,
    judge_weight: float = 0.0,
    name: str = "grounded-citation@1",
) -> Grader:
    """Assemble the full grounded-citation grader: the deterministic checks plus an optional judge.

    (ADR-0012.)

    The deterministic checks (:class:`GroundedCitationGrader`) are always
    the hard gate here (weight 1.0 -- a failure there always fails the
    whole thing). The optional judge tier -- ``judge_client`` wrapped with
    the rubric from :func:`build_grounding_rubric` via
    :class:`RubricBoundJudgeClient` -- defaults to weight 0.0, meaning its
    verdict is recorded as evidence for a human to read, but mathematically
    cannot move the combined score either way ("score-inert"). It's
    structurally impossible for the judge to gate (block a release) here
    unless the caller explicitly supplies a ``calibration`` artifact:
    without one, the judge component is always built with
    ``hard_gate=False`` and ``JudgeGrader(gate=False)``. And even when a
    ``calibration`` is supplied, ``JudgeGrader`` still independently
    re-checks at grading time that the calibration actually meets the
    project's minimum reliability bar (ADR-0007) -- supplying a calibration
    makes the judge *eligible* to gate, not automatically trusted.

    Raises:
        ValueError: ``calibration`` was supplied without a ``judge_client``
            (there would be nothing for that calibration to apply to), or
            ``judge_weight`` is negative (rejected by ``WeightedGrader``).
    """
    if calibration is not None and judge_client is None:
        raise ValueError("calibration requires a judge_client to calibrate")
    deterministic = GroundedCitationGrader(name="grounded-citation-deterministic@1")
    components = [WeightedGrader(deterministic, weight=1.0, hard_gate=True)]
    if judge_client is not None:
        bound = RubricBoundJudgeClient(judge_client, rubric=build_grounding_rubric())
        allow_gate = calibration is not None
        judge = JudgeGrader(
            bound,
            calibration=calibration,
            gate=allow_gate,
            name="grounded-sufficiency-judge@1",
        )
        components.append(WeightedGrader(judge, weight=judge_weight, hard_gate=allow_gate))
    return CompositeGrader(name=name, graders=tuple(components))
