"""Grounded-citation probe grading (ADR-0012, design §9).

Deterministic, LLM-free grading of a structured grounded answer against the
trusted corpus its task supplied, plus an optional rubric-bound advisory
judge tier -- the NIST-AREP-style grounded-citation probe: faithfulness
(anti-hallucination), completeness (anti-cherry-picking), and sufficiency
(anti-overreach).

The deterministic tier is a **grounding-hygiene floor, not an
answer-correctness verdict**: it checks the citation discipline any correct
grounded answer must clear -- the output parses into the documented
``{answer, citations}`` contract, every cited document exists in the task's
corpus, every quote is verbatim from its cited document, every required
document is actually cited, and no planted do-not-cite canary token is
echoed -- without asserting the answer itself is right. The sufficiency
axis ("is the cited evidence strong enough for the claim?") cannot be
decided deterministically; it belongs to the judge tier, which is
score-inert and advisory unless the caller supplies a
:class:`~agentic_evalkit.graders.judge.CalibrationArtifact` -- and even
then ``JudgeGrader`` re-verifies the ratified floor at grade time
(ADR-0007).

Anti-gaming floor: a citation counts toward citation-presence and
required-evidence coverage only when its normalized quote carries at least
:data:`MIN_SUBSTANTIVE_QUOTE_TOKENS` tokens, so a degenerate answer citing
one trivial verbatim word per required document cannot satisfy the gate.

Containment normalization is the shared
:func:`agentic_evalkit.graders.contamination.normalize_for_containment`
(Unicode NFC / whitespace collapse / case fold, deliberately without
``exact``'s numeric-shape rewrite -- an equality rule that would corrupt
substring containment), so quote-faithfulness and canary-leak matching
carry exactly one semantics across the package (ADR-0013).
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

#: Minimum token count (after containment normalization) a citation's quote
#: must carry before it counts toward citation-presence or required-evidence
#: coverage. Guards the degenerate-pass hole found in adversarial review
#: (2026-07-09): without it, one trivial verbatim word per required document
#: plus any nonempty answer would clear every deterministic check.
MIN_SUBSTANTIVE_QUOTE_TOKENS = 4

#: Stable scope label stamped into every grade's evidence so no report can
#: read the deterministic tier as an answer-correctness verdict.
GRADING_SCOPE = "grounding-hygiene floor (citation discipline), not answer correctness"


def _token_count(text: str) -> int:
    normalized = normalize_for_containment(text)
    return len(normalized.split()) if normalized else 0


def _quote_is_faithful(quote: str, document_text: str) -> bool:
    """A quote is faithful iff its normalized form is a nonempty substring
    of the normalized document text. An empty quote quotes nothing and is
    never faithful -- guarding Python's vacuous ``"" in text`` truth.
    """
    normalized_quote = normalize_for_containment(quote)
    if not normalized_quote:
        return False
    return normalized_quote in normalize_for_containment(document_text)


def _leaked_canary_tokens(answer: GroundedAnswer, canary_tokens: tuple[str, ...]) -> list[str]:
    """Canary tokens echoed in the answer or any quote.

    Delegates matching to the shared, normalization-insensitive
    :func:`agentic_evalkit.graders.contamination.find_canary_leaks` so this
    grader and the reusable helper can never drift apart in semantics
    (ADR-0013). Haystacks are scanned separately, never concatenated, so a
    token can never be assembled across an answer/quote boundary.
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
    """Grading-side view of one task's corpus and oracle metadata."""

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
    """Extract the grading oracle a grounded-citation sample carries, or ``None``.

    ``None`` means "this sample does not carry grounded-citation oracle
    data" (the grader was pointed at a sample some other adapter prepared);
    the grader abstains rather than failing the sample. ``canary_tokens``
    may legitimately be empty -- the leak check is then vacuously clean --
    but documents and required evidence must be present and nonempty.
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
    # Local evidence dicts use ``Any`` values and let pydantic validate at
    # the ``GradeResult`` boundary, matching ``composite.py``'s precedent.
    return {
        "grading_scope": GRADING_SCOPE,
        "checks": checks,
        "failed_checks": _failed_check_names(checks),
    }


class GroundedCitationGrader:
    """Deterministic grounding-hygiene grader for structured grounded answers.

    Consumes ``execution.structured_output`` when present (else
    ``execution.output``), validates it against
    :class:`~agentic_evalkit.models.grounding.GroundedAnswer`, and runs the
    :class:`~agentic_evalkit.models.grounding.GroundingCheck` battery
    against the oracle data a
    :class:`~agentic_evalkit.benchmarks.grounding.GroundedCitationAdapter`
    sample carries. The primary outcome is binary; the per-check breakdown
    is auxiliary evidence. ``hard_gate`` on the returned result is always
    ``False`` -- gate policy belongs to the composition layer
    (``WeightedGrader(..., hard_gate=True)``), matching ``ExactMatchGrader``.

    Args:
        name: Stable grader identifier reported on every ``GradeResult``.
        min_substantive_quote_tokens: Anti-gaming floor; see
            :data:`MIN_SUBSTANTIVE_QUOTE_TOKENS`.
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
    """Binds a :class:`Rubric` to any :class:`JudgeClient` (ADR-0012).

    The first production binding of the rubric module to the judge pipeline:
    every outgoing request's prompt is prefixed with the rubric's atomic
    criteria and an instruction to return per-criterion PASS/FAIL verdicts
    with evidence-citing rationales -- never a numeric rating (the
    free-form-numeric-rating anti-pattern the eval-validity literature
    warns against).

    Identity: ``fingerprint`` covers the inner judge's fingerprint AND the
    rubric content, so editing the rubric invalidates a
    ``CalibrationArtifact`` exactly like changing the model or prompt would
    (design §9: a fingerprint names the model+prompt configuration, and the
    rubric is part of the prompt).

    Position-bias probe: ``JudgeGrader`` re-sends every request with
    ``metadata={"reversed": True}``; this client makes that probe concrete
    by enumerating the rubric criteria in reverse order, so an
    order-sensitive judge visibly flips its verdict and can never gate.

    Fingerprint lifting: the wrapped response's ``fingerprint`` is rewritten
    to this client's composite fingerprint only when the inner response
    matched the inner client's own declared fingerprint; a genuine inner
    mismatch passes through unchanged so ``JudgeGrader``'s
    fingerprint-equality check still catches it.
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
    """The three NIST-AREP grounded-citation axes as atomic rubric criteria.

    Faithfulness and completeness are deterministically approximated by
    :class:`GroundedCitationGrader` (quote-faithfulness/citation-resolution
    and required-evidence coverage respectively) and hard-gate there;
    sufficiency cannot be decided deterministically and is judge-only,
    advisory until calibrated (ADR-0007).
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
    """Compose the grounded-citation probe (ADR-0012).

    The deterministic tier is the hard gate (weight 1.0); the judge tier --
    ``judge_client`` bound to :func:`build_grounding_rubric` via
    :class:`RubricBoundJudgeClient` -- defaults to weight 0.0 (score-inert:
    its verdict is recorded in child evidence but can never move the
    composite score). Judge hard-gating without a calibration artifact is
    structurally impossible here: the judge component is wired
    ``hard_gate=False`` and ``JudgeGrader(gate=False)`` unless
    ``calibration`` is supplied -- and even then ``JudgeGrader`` re-verifies
    the ratified floor at grade time (ADR-0007).

    Raises:
        ValueError: ``calibration`` was supplied without a ``judge_client``
            (nothing to calibrate), or ``judge_weight`` is negative
            (rejected by ``WeightedGrader``).
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
