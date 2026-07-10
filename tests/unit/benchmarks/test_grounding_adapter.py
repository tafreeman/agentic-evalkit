"""Tests for :mod:`agentic_evalkit.benchmarks.grounding` (ADR-0012).

The adapter's load-bearing property is the oracle/input split: the system
under test sees the question and canary-field-stripped documents only,
while required evidence, the canary registry, and gold spans stay in
grading-only metadata. Record validation is fail-closed: every malformed
record shape raises ``DatasetSchemaMismatch`` before execution or grading.
"""

from typing import Any

import pytest

from agentic_evalkit.benchmarks.grounding import GroundedCitationAdapter
from agentic_evalkit.errors import DatasetSchemaMismatch
from agentic_evalkit.models import EvalSample, SourceRecord

_CANARY_A = "TRIPWIRE-ALPHA-001"
_CANARY_B = "TRIPWIRE-BETA-002"
_DOC_A_TEXT = (
    "Alpha station's molten-salt loop runs at negative pressure. "
    f"{_CANARY_A} The loop is inspected every twelve hours by the night crew."
)
_DOC_B_TEXT = (
    "Beta station stores backup fuel cells in a shielded vault. "
    f"{_CANARY_B} Fuel cells are rotated monthly during scheduled maintenance windows."
)
_GOLD_QUOTE = "molten-salt loop runs at negative pressure"


def _task_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "task_id": "t1",
        "question": "How do Alpha and Beta stations maintain their equipment?",
        "documents": [
            {"doc_id": "doc-a", "title": "Alpha ops", "text": _DOC_A_TEXT, "canary": _CANARY_A},
            {"doc_id": "doc-b", "title": "Beta ops", "text": _DOC_B_TEXT, "canary": _CANARY_B},
        ],
        "required_evidence": ["doc-a", "doc-b"],
        "gold_spans": [{"doc_id": "doc-a", "quote": _GOLD_QUOTE}],
    }
    data.update(overrides)
    return data


def _record(data: dict[str, Any]) -> SourceRecord:
    return SourceRecord(row_id="0", data=data, digest="sha256:test-row")


def _prepare(data: dict[str, Any]) -> EvalSample:
    return GroundedCitationAdapter().prepare(_record(data))


def test_prepare_strips_the_canary_field_but_keeps_document_text() -> None:
    sample = _prepare(_task_data())
    documents = sample.input["documents"]
    assert isinstance(documents, list)
    for document in documents:
        assert isinstance(document, dict)
        # The machine-readable canary label never reaches the target...
        assert "canary" not in document
    # ...but the token itself stays embedded in the visible text, where a
    # careful system should treat it as the do-not-cite distractor it is.
    texts = [document["text"] for document in documents if isinstance(document, dict)]
    assert any(_CANARY_A in text for text in texts if isinstance(text, str))


def test_prepare_carries_the_grading_oracle_in_metadata() -> None:
    sample = _prepare(_task_data())
    assert sample.sample_id == "grounded-citation:t1"
    assert sample.adapter == "grounded-citation-tasks@1"
    assert sample.tags == ("grounded-citation",)
    assert sample.reference == _GOLD_QUOTE
    assert sample.metadata["required_evidence"] == ["doc-a", "doc-b"]
    assert sample.metadata["canary_tokens"] == [_CANARY_A, _CANARY_B]
    assert sample.metadata["gold_spans"] == [{"doc_id": "doc-a", "quote": _GOLD_QUOTE}]
    assert sample.grader is not None
    assert sample.grader.name == "grounded-citation@1"
    assert sample.grader.hard_gate is True


def test_prepare_rejects_a_non_verbatim_gold_span() -> None:
    data = _task_data(gold_spans=[{"doc_id": "doc-a", "quote": "words never in the document"}])
    with pytest.raises(DatasetSchemaMismatch, match="verbatim"):
        _prepare(data)


def test_prepare_rejects_unknown_required_evidence() -> None:
    data = _task_data(required_evidence=["doc-a", "doc-nope"])
    with pytest.raises(DatasetSchemaMismatch, match="unknown documents"):
        _prepare(data)


def test_prepare_rejects_a_document_without_a_canary() -> None:
    data = _task_data()
    data["documents"][1] = {"doc_id": "doc-b", "title": "Beta ops", "text": _DOC_B_TEXT}
    with pytest.raises(DatasetSchemaMismatch, match="canary"):
        _prepare(data)


def test_prepare_rejects_a_canary_not_embedded_in_its_document_text() -> None:
    data = _task_data()
    data["documents"][0]["canary"] = "TOKEN-NOT-IN-TEXT"
    with pytest.raises(DatasetSchemaMismatch, match="not embedded"):
        _prepare(data)


def test_prepare_rejects_duplicate_canary_tokens() -> None:
    data = _task_data()
    data["documents"][1]["text"] = f"{_DOC_B_TEXT} {_CANARY_A}"
    data["documents"][1]["canary"] = _CANARY_A
    with pytest.raises(DatasetSchemaMismatch, match="duplicate canary"):
        _prepare(data)


def test_prepare_rejects_duplicate_document_ids() -> None:
    data = _task_data()
    data["documents"][1]["doc_id"] = "doc-a"
    with pytest.raises(DatasetSchemaMismatch, match="duplicate document ids"):
        _prepare(data)


def test_prepare_rejects_empty_required_evidence() -> None:
    with pytest.raises(DatasetSchemaMismatch, match="no required evidence"):
        _prepare(_task_data(required_evidence=[]))


def test_prepare_rejects_a_gold_span_citing_an_unknown_document() -> None:
    data = _task_data(gold_spans=[{"doc_id": "doc-zz", "quote": _GOLD_QUOTE}])
    with pytest.raises(DatasetSchemaMismatch, match="unknown document"):
        _prepare(data)


def test_prepare_rejects_a_row_that_is_not_a_task_at_all() -> None:
    with pytest.raises(DatasetSchemaMismatch, match="failed validation"):
        _prepare({"question": "where is the rest of the record?"})


def test_validate_oracle_accepts_a_prepared_sample() -> None:
    adapter = GroundedCitationAdapter()
    assert adapter.validate_oracle(_prepare(_task_data())) is True


def test_validate_oracle_rejects_a_sample_without_oracle_metadata() -> None:
    adapter = GroundedCitationAdapter()
    stripped = EvalSample(
        sample_id="grounded-citation:t1",
        input={"question": "?"},
        reference=_GOLD_QUOTE,
        source_digest="sha256:test-row",
        adapter="grounded-citation-tasks@1",
    )
    assert adapter.validate_oracle(stripped) is False


def test_aggregate_metadata_names_the_benchmark_and_components() -> None:
    metadata = GroundedCitationAdapter().aggregate_metadata()
    assert metadata == {
        "benchmark": "grounded-citation",
        "adapter": "grounded-citation-tasks@1",
        "grader": "grounded-citation@1",
    }
