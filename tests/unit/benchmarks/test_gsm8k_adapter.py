import pytest

from agentic_evalkit.benchmarks.gsm8k import Gsm8kAdapter, extract_final_answer
from agentic_evalkit.errors import DatasetSchemaMismatch
from agentic_evalkit.models import GraderSpec, SourceRecord


def test_projects_question_and_normalized_reference() -> None:
    record = SourceRecord(
        row_id="0",
        data={"question": "What is 20 / 4?", "answer": "Reasoning #### 5"},
        digest="sha256:row",
    )
    sample = Gsm8kAdapter().prepare(record)
    assert sample.input == {"question": "What is 20 / 4?"}
    assert sample.reference == "5"
    assert extract_final_answer("work\n#### 5.0") == "5"


def test_extract_final_answer_strips_thousands_separator_commas() -> None:
    assert extract_final_answer("total is #### 1,000") == "1000"


def test_extract_final_answer_uses_the_final_marker_not_the_first() -> None:
    """Reasoning text may itself contain '####'; only the last one is authoritative."""
    assert extract_final_answer("intermediate #### 3\nfinal #### 42") == "42"


def test_extract_final_answer_preserves_non_integer_equivalent_decimals() -> None:
    """Only an exact `.0` suffix collapses; a genuine fraction like 5.5 must not be truncated."""
    assert extract_final_answer("#### 5.5") == "5.5"


def test_extract_final_answer_raises_without_marker() -> None:
    with pytest.raises(ValueError, match="####"):
        extract_final_answer("no marker here")


def test_prepare_raises_dataset_schema_mismatch_for_missing_fields() -> None:
    record = SourceRecord(row_id="1", data={"question": "only a question"}, digest="sha256:row")
    with pytest.raises(DatasetSchemaMismatch):
        Gsm8kAdapter().prepare(record)


def test_prepare_raises_dataset_schema_mismatch_for_answer_without_marker() -> None:
    record = SourceRecord(
        row_id="2",
        data={"question": "Q?", "answer": "reasoning with no final marker"},
        digest="sha256:row",
    )
    with pytest.raises(DatasetSchemaMismatch):
        Gsm8kAdapter().prepare(record)


def test_prepare_attaches_normalized_exact_grader_spec() -> None:
    record = SourceRecord(
        row_id="0", data={"question": "Q?", "answer": "#### 1"}, digest="sha256:row"
    )
    sample = Gsm8kAdapter().prepare(record)
    assert sample.grader == GraderSpec(
        name="normalized-exact@1", grader_type="objective", hard_gate=True
    )
    assert sample.adapter == "gsm8k@1"


def test_adapter_declares_api_version_and_name() -> None:
    adapter = Gsm8kAdapter()
    assert adapter.api_version == "1"
    assert adapter.name == "gsm8k@1"


def test_validate_oracle_requires_nonempty_reference() -> None:
    record = SourceRecord(
        row_id="0", data={"question": "Q?", "answer": "#### 1"}, digest="sha256:row"
    )
    sample = Gsm8kAdapter().prepare(record)
    assert Gsm8kAdapter().validate_oracle(sample) is True
    empty_reference_sample = sample.model_copy(update={"reference": None})
    assert Gsm8kAdapter().validate_oracle(empty_reference_sample) is False
