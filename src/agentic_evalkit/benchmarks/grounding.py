"""Grounded-citation benchmark adapter (ADR-0012, design §7).

Projects one grounded-QA task record -- question, trusted corpus documents,
required-evidence document IDs, and verbatim gold spans -- into a typed
:class:`~agentic_evalkit.models.EvalSample`. The projection enforces the
oracle/input separation the probe depends on:

- ``EvalSample.input`` carries only what the system under test may see: the
  question and the corpus documents **with the labeled canary field
  stripped** (the canary token itself stays embedded in each document's
  text, where a careful system should treat it as the do-not-cite
  distractor it is).
- ``EvalSample.metadata`` carries the grading-only oracle data: required
  evidence IDs, the canary token registry, and the gold spans. None of it
  is part of the execution request.

Record validation is fail-closed via
:class:`~agentic_evalkit.models.grounding.GroundedCitationTask`: a record
whose gold spans are not verbatim substrings of their documents, whose
required evidence names unknown documents, or whose canaries are missing or
duplicated raises :class:`~agentic_evalkit.errors.DatasetSchemaMismatch`
before any execution or grading can happen.
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
    """Projects grounded-QA task records into objectively gradable samples."""

    api_version = _API_VERSION
    name = _ADAPTER_NAME

    def prepare(self, record: SourceRecord) -> EvalSample:
        """Project one task record into an :class:`EvalSample`.

        Raises:
            DatasetSchemaMismatch: the record fails
                :class:`GroundedCitationTask` validation (missing fields,
                non-verbatim gold spans, unknown required evidence, or
                missing/duplicate canaries).
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
        """Oracle-valid iff the sample kept its required evidence and reference span."""
        required = sample.metadata.get("required_evidence")
        has_required = isinstance(required, list) and len(required) > 0
        return has_required and bool(sample.reference)

    def aggregate_metadata(self) -> dict[str, JsonValue]:
        """Benchmark-specific metadata recorded on run aggregation (design §7)."""
        return {
            "benchmark": "grounded-citation",
            "adapter": _ADAPTER_NAME,
            "grader": _GRADER_NAME,
        }
