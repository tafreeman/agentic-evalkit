"""Declarative projection of local or hosted rows into evaluation samples."""

from typing import Annotated

from pydantic import Field, JsonValue

from agentic_evalkit.errors import DatasetSchemaMismatch
from agentic_evalkit.models import EvalSample, GraderSpec, SourceRecord
from agentic_evalkit.models.base import FrozenModel

PathPart = str | Annotated[int, Field(ge=0)]
FieldPath = tuple[PathPart, ...]


class FieldMapConfig(FrozenModel):
    sample_id: FieldPath | None = None
    input: dict[str, FieldPath] = Field(min_length=1)
    reference: FieldPath


def select_path(value: JsonValue, path: FieldPath) -> JsonValue:
    """Follow literal object keys and nonnegative list indexes only."""
    for part in path:
        if isinstance(value, dict):
            if isinstance(part, str) and part in value:
                value = value[part]
                continue
        elif isinstance(value, list) and isinstance(part, int) and 0 <= part < len(value):
            value = value[part]
            continue
        raise DatasetSchemaMismatch(message=f"missing field-map path {path!r}")
    return value


class FieldMapAdapter:
    api_version = "1"
    name = "field-map@1"

    def __init__(self, config: FieldMapConfig, *, grader: str = "normalized-exact@1") -> None:
        self.config = config
        self.grader = grader

    def prepare(self, record: SourceRecord) -> EvalSample:
        identifier = (
            select_path(record.data, self.config.sample_id)
            if self.config.sample_id is not None
            else record.row_id
        )
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, (str, int))
            or str(identifier) == ""
        ):
            raise DatasetSchemaMismatch(
                message="mapped sample id must be a nonempty string or integer"
            )
        reference = select_path(record.data, self.config.reference)
        if reference is None or isinstance(reference, (dict, list)):
            raise DatasetSchemaMismatch(message="field-map reference must be a scalar answer")
        return EvalSample(
            sample_id=str(identifier),
            input={
                name: select_path(record.data, path) for name, path in self.config.input.items()
            },
            reference=str(reference),
            source_row_id=record.row_id,
            source_digest=record.digest,
            adapter=self.name,
            grader=GraderSpec(name=self.grader, grader_type="objective", hard_gate=True),
        )

    def validate_oracle(self, sample: EvalSample) -> bool:
        return sample.reference is not None

    def aggregate_metadata(self) -> dict[str, JsonValue]:
        return {"adapter": self.name, "field_map": self.config.model_dump(mode="json")}
