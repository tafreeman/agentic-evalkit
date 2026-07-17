"""Data models for the "grounded citation" evaluation (ADR-0012).

Does an answer actually cite and quote real source documents?

This evaluates a specific skill: given a question and a set of trusted
source documents, can the system being evaluated answer it while citing
real documents, quoting them accurately, and not accidentally citing
content it shouldn't? Concretely, this module defines the shapes for: one
trusted source document, one citation the system being evaluated
provides, a full structured answer (text plus its citations), and a
complete task record tying a question to its documents and the evidence a
good answer is expected to draw on. Every model here follows ADR-0002
(immutable, rejects unknown fields, carries a ``schema_version``, uses
tuples instead of mutable lists) and performs no file or network access
itself. The actual grading logic that scores an answer against these
shapes lives in :mod:`agentic_evalkit.graders.grounding`; turning a raw
dataset record into a ``GroundedCitationTask`` lives in
:mod:`agentic_evalkit.benchmarks.grounding`.

The "right answer" data in this probe is deliberately **not a number or a
short exact-match string**: it's document IDs and word-for-word quotes
only. Whenever a task is constructed, validators check that every
reference quote genuinely appears word-for-word in the document it's
attributed to, and that every planted decoy token (see
``GroundingCorpusDoc.canary`` below) is actually embedded in that
document's own text. This "fails closed": bad data is rejected immediately
at construction time, rather than being allowed through and graded against
later, so a task built from broken or fabricated reference data can never
silently produce a misleading score.
"""

from enum import StrEnum

from pydantic import model_validator

from agentic_evalkit.models.base import FrozenModel


class GroundingCheck(StrEnum):
    """The individual checks the grounded-citation grader runs -- all rule-based, no AI judgment.

    ``STRUCTURED_CONTRACT`` checks something more basic than the rest: did
    the output from the system being evaluated even parse into the
    expected ``{answer, citations}`` shape at all? The other checks all
    assume that much succeeded, and grade the parsed answer itself. This
    enum is the fixed, stable vocabulary that grading reports use to name
    which check they're talking about -- renaming a member here would be a
    breaking change for anyone reading older evidence.
    """

    STRUCTURED_CONTRACT = "structured_contract"
    ANSWER_NONEMPTY = "answer_nonempty"
    CITATION_PRESENT = "citation_present"
    CITATION_RESOLUTION = "citation_resolution"
    QUOTE_FAITHFULNESS = "quote_faithfulness"
    EVIDENCE_COVERAGE = "evidence_coverage"
    CANARY_LEAK = "canary_leak"


class GroundingCorpusDoc(FrozenModel):
    """One trusted source document that an answer is expected to cite and quote from.

    ``canary`` is a decoy token planted inside this document's text purely
    as a tripwire: it's a string a faithful answer should have no reason
    to ever repeat, so if it shows up in the answer from the system being
    evaluated anyway, that's a sign the system leaked or memorized content
    it wasn't supposed to reference directly (used for contamination and
    leak detection -- see ADR-0013). ``canary`` is ``None`` on the copy of
    this document actually shown to the system being evaluated -- the
    token itself is still sitting inside ``text`` where it was planted,
    but the labeled field naming it is stripped out, so that system never
    receives a ready-made list of "these are the tokens to watch for."

    Attributes:
        doc_id: This document's unique ID within its task.
        title: The document's title, if any.
        text: The document's full text.
        canary: The decoy tripwire token planted in ``text``, when this
            document has one recorded. ``None`` either when no canary was
            planted, or when this is the copy shown to the system being
            evaluated, with the label deliberately stripped (see above).
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
    """One citation: a source document's ID, plus a word-for-word quote taken from it."""

    doc_id: str
    quote: str


class GroundedAnswer(FrozenModel):
    """A system's free-text answer plus the citations it gave to back it up."""

    answer: str
    citations: tuple[CitationRecord, ...] = ()


class GroundedCitationTask(FrozenModel):
    """One full grounded-citation task record. Building one runs its validators immediately.

    If the data doesn't check out -- a quote that doesn't actually appear
    in its document, a reference to a document that doesn't exist -- then
    construction fails right away ("fail-closed") rather than quietly
    accepting bad reference data that could later produce a misleading
    grade.

    Attributes:
        task_id: This task's unique ID.
        question: The question being asked.
        documents: The trusted source documents available to answer from.
        required_evidence: The IDs of the documents a complete answer is
            expected to draw on.
        gold_spans: Reference, word-for-word quotes taken directly from
            the source documents (never computed or paraphrased) that a
            correct answer would be expected to be groundable in. These
            are reference material for the advisory judge tier and for
            audit reports -- they are **not** an exact-match target: a
            system that cites different, equally-faithful quotes still
            passes the deterministic checks, because those checks score
            whether the answer is well-grounded, not whether it happened
            to quote the exact same span as this reference.
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
