"""Immutable wire contracts for the grounded-citation probe (ADR-0012).

These models carry the NIST-AREP-style grounded-citation evaluation shapes:
a trusted corpus document, a candidate citation, a structured grounded
answer, and a full task record. Every model follows ADR-0002 (frozen,
``extra="forbid"``, ``schema_version``, tuple collections) and performs no
I/O. Grading policy lives in :mod:`agentic_evalkit.graders.grounding`;
dataset projection lives in :mod:`agentic_evalkit.benchmarks.grounding`.

Gold data in this probe is deliberately **non-numeric**: document IDs and
verbatim source spans only. Construction-time validators enforce that every
gold span actually appears verbatim in its cited document and that every
distractor canary token is embedded in its document's text, so a task that
could silently grade against fabricated evidence is rejected before it can
ever reach a grader.
"""

from enum import StrEnum

from pydantic import model_validator

from agentic_evalkit.models.base import FrozenModel


class GroundingCheck(StrEnum):
    """The deterministic, LLM-free checks the grounded-citation grader runs.

    ``STRUCTURED_CONTRACT`` covers "the target's output parses into the
    documented ``{answer, citations}`` shape at all"; the remaining checks
    grade a parsed answer. The enum is the stable vocabulary reports use in
    grade evidence, so renaming a member is a breaking evidence change.
    """

    STRUCTURED_CONTRACT = "structured_contract"
    ANSWER_NONEMPTY = "answer_nonempty"
    CITATION_PRESENT = "citation_present"
    CITATION_RESOLUTION = "citation_resolution"
    QUOTE_FAITHFULNESS = "quote_faithfulness"
    EVIDENCE_COVERAGE = "evidence_coverage"
    CANARY_LEAK = "canary_leak"


class GroundingCorpusDoc(FrozenModel):
    """One trusted corpus document a task's answer must be grounded in.

    ``canary`` is the document's embedded distractor token (a synthetic
    "do not cite" marker used for contamination and leak detection). It is
    ``None`` on the system-under-test-facing projection of a document --
    the token itself still sits inside ``text``, but the labeled field is
    stripped so the target never receives a machine-readable canary list.
    """

    doc_id: str
    title: str = ""
    text: str
    canary: str | None = None

    @model_validator(mode="after")
    def _validate_document(self) -> "GroundingCorpusDoc":
        if not self.doc_id.strip():
            raise ValueError("doc_id must be a nonempty string")
        if not self.text.strip():
            raise ValueError(f"document {self.doc_id!r} has empty text")
        if self.canary is not None:
            if not self.canary.strip():
                raise ValueError(f"document {self.doc_id!r} has an empty canary token")
            if self.canary not in self.text:
                raise ValueError(
                    f"document {self.doc_id!r} declares canary {self.canary!r} "
                    "but the token is not embedded in the document text"
                )
        return self


class CitationRecord(FrozenModel):
    """One citation: a corpus document ID plus a verbatim quote from it."""

    doc_id: str
    quote: str


class GroundedAnswer(FrozenModel):
    """A target's structured grounded answer: free text plus its citations."""

    answer: str
    citations: tuple[CitationRecord, ...] = ()


class GroundedCitationTask(FrozenModel):
    """One full grounded-QA task record, validated fail-closed at load time.

    ``gold_spans`` are verbatim source spans (never computed values) that a
    correct answer is expected to be groundable in; they are reference
    material for the advisory judge tier and audit reports, **not** an
    exact-match target -- a system citing different-but-faithful spans still
    passes the deterministic tier (outcome-based scoring, not span matching).
    """

    task_id: str
    question: str
    documents: tuple[GroundingCorpusDoc, ...]
    required_evidence: tuple[str, ...]
    gold_spans: tuple[CitationRecord, ...]

    @model_validator(mode="after")
    def _validate_task(self) -> "GroundedCitationTask":
        if not self.task_id.strip():
            raise ValueError("task_id must be a nonempty string")
        if not self.question.strip():
            raise ValueError(f"task {self.task_id!r} has an empty question")
        if not self.documents:
            raise ValueError(f"task {self.task_id!r} has no documents")
        doc_ids = [document.doc_id for document in self.documents]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError(f"task {self.task_id!r} has duplicate document ids")
        self._validate_required_evidence(set(doc_ids))
        self._validate_gold_spans()
        self._validate_canaries()
        return self

    def _validate_required_evidence(self, known_ids: set[str]) -> None:
        if not self.required_evidence:
            raise ValueError(f"task {self.task_id!r} declares no required evidence")
        if len(self.required_evidence) != len(set(self.required_evidence)):
            raise ValueError(f"task {self.task_id!r} has duplicate required_evidence ids")
        unknown = sorted(set(self.required_evidence) - known_ids)
        if unknown:
            raise ValueError(
                f"task {self.task_id!r} requires evidence from unknown documents: {unknown}"
            )

    def _validate_gold_spans(self) -> None:
        if not self.gold_spans:
            raise ValueError(f"task {self.task_id!r} has no gold spans")
        text_by_id = {document.doc_id: document.text for document in self.documents}
        for span in self.gold_spans:
            text = text_by_id.get(span.doc_id)
            if text is None:
                raise ValueError(
                    f"task {self.task_id!r} gold span cites unknown document {span.doc_id!r}"
                )
            if not span.quote.strip() or span.quote not in text:
                raise ValueError(
                    f"task {self.task_id!r} gold span for {span.doc_id!r} is not a "
                    "verbatim substring of that document"
                )

    def _validate_canaries(self) -> None:
        canaries = [document.canary for document in self.documents]
        if any(canary is None for canary in canaries):
            raise ValueError(
                f"task {self.task_id!r} has documents without an embedded canary token"
            )
        if len(canaries) != len(set(canaries)):
            raise ValueError(f"task {self.task_id!r} has duplicate canary tokens")


__all__ = [
    "CitationRecord",
    "GroundedAnswer",
    "GroundedCitationTask",
    "GroundingCheck",
    "GroundingCorpusDoc",
]
