"""GSM8K benchmark adapter (design §6.2, §7).

GSM8K is the runnable quickstart preset: ``openai/gsm8k``, config ``main``,
split ``test``, graded with ``normalized-exact@1`` (design §6.2). This
module projects raw GSM8K rows (``question``/``answer`` fields, where
``answer`` embeds step-by-step reasoning followed by a ``#### <number>``
final-answer marker) into typed :class:`~agentic_evalkit.models.EvalSample`
instances with a normalized numeric reference string.
"""

from pydantic import JsonValue

from agentic_evalkit.errors import DatasetSchemaMismatch
from agentic_evalkit.models import EvalSample, GraderSpec, SourceRecord

_FINAL_ANSWER_MARKER = "####"

_API_VERSION = "1"
_ADAPTER_NAME = "gsm8k@1"
_GRADER_NAME = "normalized-exact@1"


def extract_final_answer(answer_text: str) -> str:
    """Extract and normalize the final numeric answer from a GSM8K answer.

    GSM8K answers embed free-form reasoning followed by a ``####`` marker
    and the final numeric answer (for example, ``"work\\n#### 5.0"``). This
    takes the text after the *final* ``####`` occurrence (reasoning text may
    legitimately contain the literal characters elsewhere), then normalizes
    it by:

    - stripping surrounding whitespace;
    - removing thousands-separator commas (``"1,000"`` -> ``"1000"``);
    - collapsing an integer-equivalent trailing ``.0`` decimal
      (``"5.0"`` -> ``"5"``) so formatting differences do not cause a
      normalized-exact grading mismatch.

    Raises:
        ValueError: ``answer_text`` contains no ``####`` marker at all.
    """
    if _FINAL_ANSWER_MARKER not in answer_text:
        raise ValueError(f"no {_FINAL_ANSWER_MARKER!r} marker found in answer text")
    _, _, final_segment = answer_text.rpartition(_FINAL_ANSWER_MARKER)
    normalized = final_segment.strip().replace(",", "")
    if normalized.endswith(".0"):
        candidate = normalized[: -len(".0")]
        if candidate.lstrip("-").isdigit():
            normalized = candidate
    return normalized


class Gsm8kAdapter:
    """Projects raw GSM8K rows into typed, objectively gradable samples."""

    api_version = _API_VERSION
    name = _ADAPTER_NAME

    def prepare(self, record: SourceRecord) -> EvalSample:
        """Project one GSM8K source record into an :class:`EvalSample`.

        Raises:
            DatasetSchemaMismatch: the row is missing ``question`` or
                ``answer``, or ``answer`` has no ``####`` marker.
        """
        question = record.data.get("question")
        answer = record.data.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise DatasetSchemaMismatch(
                message="GSM8K row must have string 'question' and 'answer' fields",
                context={"row_id": record.row_id},
            )
        try:
            reference = extract_final_answer(answer)
        except ValueError as exc:
            raise DatasetSchemaMismatch(
                message=f"GSM8K row 'answer' field is malformed: {exc}",
                context={"row_id": record.row_id},
            ) from exc

        return EvalSample(
            sample_id=f"gsm8k:{record.row_id}",
            input={"question": question},
            reference=reference,
            source_row_id=record.row_id,
            source_digest=record.digest,
            adapter=_ADAPTER_NAME,
            grader=GraderSpec(name=_GRADER_NAME, grader_type="objective", hard_gate=True),
        )

    def validate_oracle(self, sample: EvalSample) -> bool:
        """A GSM8K sample is oracle-valid iff it carries a nonempty reference."""
        return bool(sample.reference)

    def aggregate_metadata(self) -> dict[str, JsonValue]:
        """Benchmark-specific metadata recorded on run aggregation (design §7)."""
        return {
            "benchmark": "gsm8k",
            "adapter": _ADAPTER_NAME,
            "grader": _GRADER_NAME,
        }
