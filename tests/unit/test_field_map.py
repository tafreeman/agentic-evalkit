"""Field mapping preserves falsy values and never evaluates expressions."""

import pytest

from agentic_evalkit.benchmarks.field_map import FieldMapAdapter, FieldMapConfig
from agentic_evalkit.errors import DatasetSchemaMismatch
from agentic_evalkit.models import SourceRecord


def test_nested_mapping_preserves_false_and_zero() -> None:
    adapter = FieldMapAdapter(
        FieldMapConfig(
            sample_id=("id",),
            input={"question": ("payload", 0, "text"), "flag": ("expected",)},
            reference=("expected",),
        )
    )
    sample = adapter.prepare(
        SourceRecord(
            row_id="r",
            digest="sha256:r",
            data={
                "id": 0,
                "payload": [{"text": "test"}],
                "expected": False,
            },
        )
    )
    assert sample.sample_id == "0"
    assert sample.input == {"question": "test", "flag": False}
    assert sample.reference == "False"


def test_missing_mapped_path_is_an_error() -> None:
    adapter = FieldMapAdapter(
        FieldMapConfig(input={"question": ("missing",)}, reference=("answer",))
    )
    with pytest.raises(DatasetSchemaMismatch):
        adapter.prepare(SourceRecord(row_id="r", digest="sha256:r", data={"answer": "yes"}))
