"""Immutable contracts for dataset identity, resolution, and paginated access.

See design §5.1-§5.2 (`docs/specs/2026-07-02-agentic-evalkit-design.md`) for
the field-level contract this module implements. Models perform no I/O.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue, model_validator

from agentic_evalkit.models.base import FrozenModel


class DatasetRef(FrozenModel):
    """Identifies a requested dataset source (design §5.1).

    A ``DatasetRef`` is a request, not a guarantee: ``revision`` may be left
    unset to mean "latest at resolution time", and ``config``/``split`` may
    be left unset when the provider can uniquely infer them.
    """

    provider: str
    dataset_id: str
    revision: str | None = None
    config: str | None = None
    split: str | None = None
    data_files: tuple[str, ...] = ()
    selection: str | None = None
    field_mapping: dict[str, str] = Field(default_factory=dict)
    allow_remote_code: bool = False


class ContaminationStatus(StrEnum):
    """Best-effort contamination/memorization-risk label for a dataset (ADR-0013).

    A ``StrEnum``, never a boolean, for the reason ADR-0002 already rejects
    boolean status fields: "never checked" (``UNKNOWN``, the honest default)
    must stay distinguishable from "checked and clean" (``VERIFIED_CLEAN``).
    """

    UNKNOWN = "unknown"
    SUSPECT = "suspect"
    VERIFIED_CLEAN = "verified_clean"
    CONFIRMED_CONTAMINATED = "confirmed_contaminated"


class ContaminationMetadata(FrozenModel):
    """Provenance signal for dataset contamination/memorization risk (ADR-0013).

    Not the ADR-0007 judge-calibration held-out corpus (``CalibrationArtifact``,
    ``graders/judge.py``) -- that is human-labeled evidence a judge is
    trustworthy; ``held_out`` here means this *eval* dataset itself was never
    published, so it cannot appear in any model's pretraining corpus by
    construction. Informative, not enforcing (like ``ResolvedDataset.gated``):
    no runtime refusal is derived from any of these fields.
    """

    status: ContaminationStatus = ContaminationStatus.UNKNOWN
    authored_after: datetime | None = None
    public_since: datetime | None = None
    canary_ids: tuple[str, ...] = ()
    held_out: bool = False

    @model_validator(mode="after")
    def _validate_held_out_consistency(self) -> "ContaminationMetadata":
        if self.held_out and self.public_since is not None:
            raise ValueError(
                "held_out=True is inconsistent with a non-null public_since "
                "(a dataset cannot be both withheld from publication and have "
                "a known public-release date)"
            )
        return self


class ResolvedDataset(FrozenModel):
    """Records the immutable source a run will actually use (design §5.2).

    Metadata fields sourced from best-effort provider endpoints (size,
    statistics, Parquet listing) are optional because many valid datasets
    legitimately lack them; a missing value means "unavailable", not "empty".
    """

    dataset_id: str
    revision: str
    config: str | None = None
    split: str | None = None
    selected_files: tuple[str, ...] = ()
    schema_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    row_count: int | None = None
    license: str | None = None
    citation: str | None = None
    gated: bool = False
    card_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    retrieved_at: datetime | None = None
    provider_response_digests: dict[str, str] = Field(default_factory=dict)
    cache_manifest_digest: str | None = None
    checksums: dict[str, str] = Field(default_factory=dict)
    contamination: ContaminationMetadata | None = None


class SourceRecord(FrozenModel):
    """One raw, provider-native row plus its identity and integrity digest.

    Provider-native records never flow directly into execution or grading
    (design §5.3); a ``BenchmarkAdapter`` projects a ``SourceRecord`` into an
    ``EvalSample`` first.
    """

    row_id: str
    data: dict[str, JsonValue]
    digest: str


class SearchHit(FrozenModel):
    """One dataset search result summary."""

    dataset_id: str
    provider: str
    revision: str | None = None
    tags: tuple[str, ...] = ()
    gated: bool = False
    private: bool = False
    downloads: int | None = None
    card_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SearchPage(FrozenModel):
    """A page of dataset search results with an opaque continuation cursor."""

    hits: tuple[SearchHit, ...] = ()
    cursor: str | None = None
    total_hits: int | None = None


class SamplePage(FrozenModel):
    """A page of raw source records returned by a provider preview/iteration."""

    records: tuple[SourceRecord, ...] = ()
    offset: int = 0
    total_rows: int | None = None
