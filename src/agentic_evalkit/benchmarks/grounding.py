"""Grounded-citation benchmark adapter (ADR-0012, design §7).

This benchmark tests whether a system properly bases ("grounds") its answers
on the documents it's given, instead of making things up or citing the wrong
source. Each task record has: a question, a set of trusted source documents,
a list of which document IDs should be cited as evidence, and exact quotes
("gold spans") that count as correct supporting evidence. This module turns
one such record into a typed :class:`~agentic_evalkit.models.EvalSample`,
carefully keeping two things separate:

- ``EvalSample.input`` holds only what the system under test is actually
  allowed to see: the question and the source documents, but **with the
  labeled "canary" field removed**. A canary is a unique marker planted
  inside one document's text as bait -- the marker's raw text is still there
  as part of the document's prose (a well-behaved system will just read past
  it), but its explicit label is stripped out so the system can't tell it's
  a planted marker. If the system's answer ends up quoting that marker, that
  is evidence it wrongly cited a document it should have ignored.
- ``EvalSample.metadata`` holds the answer key -- data used only for
  grading and never shown to the system under test: which document IDs
  count as required evidence, the list of canary markers (so the grader can
  check whether any got cited), and the gold quote spans. None of this is
  part of what the system under test receives.

Keeping the answer key completely separate from what the system sees is
what makes this test meaningful -- if the answer key leaked into the input,
the test would no longer prove anything. Record validation "fails closed"
via :class:`~agentic_evalkit.models.grounding.GroundedCitationTask`: meaning
that if anything is inconsistent -- a gold quote that isn't an exact match
for text in its document, required evidence that points at a document ID
that doesn't exist, or canary markers that are missing or duplicated --
this raises :class:`~agentic_evalkit.errors.DatasetSchemaMismatch`
immediately and refuses to continue, rather than silently proceeding with
bad data.
"""

from pydantic import JsonValue, ValidationError

from agentic_evalkit.errors import DatasetSchemaMismatch
from agentic_evalkit.models import EvalSample, GraderSpec, SourceRecord
from agentic_evalkit.models.grounding import GroundedCitationTask

_API_VERSION = "1"
_ADAPTER_NAME = "grounded-citation-tasks@1"
_GRADER_NAME = "grounded-citation@1"

__all__ = ["GroundedCitationAdapter"]


class GroundedCitationAdapter:
    """Turns grounded-QA task records into samples that can be graded
    objectively (by checking citations and quotes, not by subjective
    judgment)."""

    api_version = _API_VERSION
    name = _ADAPTER_NAME

    def prepare(self, record: SourceRecord) -> EvalSample:
        """Turn one task record into an :class:`EvalSample`.

        Raises:
            DatasetSchemaMismatch: the record fails
                :class:`GroundedCitationTask` validation -- for example, a
                missing field, a gold quote that isn't an exact match for
                text in its document, required evidence pointing at a
                document ID that doesn't exist, or a canary marker that is
                missing or duplicated.
        """
        try:
            task = GroundedCitationTask.model_validate(record.data)
        except ValidationError as exc:
            raise DatasetSchemaMismatch(
                message=f"grounded-citation record failed validation: {exc}",
                context={"row_id": record.row_id},
            ) from exc

        target_visible_documents: list[JsonValue] = [
            {"doc_id": document.doc_id, "title": document.title, "text": document.text}
            for document in task.documents
        ]
        canary_tokens: list[JsonValue] = [
            document.canary for document in task.documents if document.canary is not None
        ]
        gold_spans: list[JsonValue] = [
            {"doc_id": span.doc_id, "quote": span.quote} for span in task.gold_spans
        ]
        return EvalSample(
            sample_id=f"grounded-citation:{task.task_id}",
            input={"question": task.question, "documents": target_visible_documents},
            reference=task.gold_spans[0].quote,
            metadata={
                "required_evidence": list(task.required_evidence),
                "canary_tokens": canary_tokens,
                "gold_spans": gold_spans,
            },
            tags=("grounded-citation",),
            source_row_id=record.row_id,
            source_digest=record.digest,
            adapter=_ADAPTER_NAME,
            grader=GraderSpec(name=_GRADER_NAME, grader_type="composite", hard_gate=True),
        )

    def validate_oracle(self, sample: EvalSample) -> bool:
        """This sample is ready to grade exactly when it still has its
        required-evidence list and its reference quote."""
        required = sample.metadata.get("required_evidence")
        has_required = isinstance(required, list) and len(required) > 0
        return has_required and bool(sample.reference)

    def aggregate_metadata(self) -> dict[str, JsonValue]:
        """Extra, benchmark-specific details recorded when summarizing a run
        (design §7)."""
        return {
            "benchmark": "grounded-citation",
            "adapter": _ADAPTER_NAME,
            "grader": _GRADER_NAME,
        }
