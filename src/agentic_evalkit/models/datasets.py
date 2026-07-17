"""Data models for where a dataset comes from, what got resolved when fetched, and how to page it.

These are the shapes ("contracts") that other parts of this codebase agree
to pass back and forth when working with a dataset -- see design doc
§5.1-§5.2 (`docs/specs/2026-07-02-agentic-evalkit-design.md`) for the
authoritative, field-by-field description each class below implements.
Like every model in this package, these classes are immutable (you get a
new copy instead of editing one in place) and they never do any file or
network access themselves -- fetching the actual data is someone else's
job; these classes just describe its shape.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue, model_validator

from agentic_evalkit.models.base import FrozenModel


class DatasetRef(FrozenModel):
    """Describes which dataset you're asking for (design §5.1).

    A ``DatasetRef`` is a request, not a promise about exactly what you'll
    get back: leaving ``revision`` unset means "give me whatever is latest
    when you actually go fetch it," and ``config``/``split`` can be left
    unset whenever the data provider is able to figure out the right one on
    its own.

    Attributes:
        provider: Which dataset source to use, e.g. ``"huggingface"``.
        dataset_id: The dataset's canonical ID with that provider.
        revision: The exact version to fetch (e.g. a commit SHA or tag).
            Left unset to mean "whatever is latest when this gets
            resolved."
        config: Which named sub-configuration of the dataset to use, for
            datasets that offer more than one. Left unset when there's
            only one, or the provider can infer it.
        split: Which split to use (e.g. ``"train"``, ``"test"``). Same
            "leave it unset if it can be inferred" rule as ``config``.
        data_files: Specific files to pull, for datasets organized as
            loose files rather than one packaged whole.
        selection: An optional filter or subset expression narrowing which
            rows to use.
        field_mapping: Renames source columns to the names the rest of
            this codebase expects, for datasets whose own column names
            don't already match.
        allow_remote_code: Whether the provider is allowed to execute
            dataset-supplied code while loading this dataset. Defaults to
            ``False``; this codebase never turns on a provider's own
            "trust remote code" setting unless a caller explicitly opts in
            here, since running a dataset's bundled code is a real
            code-execution risk.
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
    """Risk that the system under test already saw this dataset during its own training (ADR-0013).

    This is a fixed set of named values (a ``StrEnum``), not a plain
    yes/no boolean -- deliberately, for the same reason ADR-0002 rules out
    booleans for status fields generally: a boolean can't distinguish "we
    never checked" from "we checked, and it's fine," since both would just
    show up as ``False``. Here, "never checked" is ``UNKNOWN`` (the safe,
    honest default), and it has to stay visibly different from "we
    checked, and this dataset looks clean" (``VERIFIED_CLEAN``).
    """

    UNKNOWN = "unknown"
    SUSPECT = "suspect"
    VERIFIED_CLEAN = "verified_clean"
    CONFIRMED_CONTAMINATED = "confirmed_contaminated"


class ContaminationMetadata(FrozenModel):
    """Records what's known about this dataset's risk of unfairly inflating scores (ADR-0013).

    "Contamination" here means the system being evaluated may have already
    seen this exact data during its own training -- which would let it
    "remember" the right answer instead of actually working it out, making
    the test unfairly easy and the resulting score misleading.

    Note this is a different "held out" from the one used to check whether
    an AI judge grader can be trusted (``CalibrationArtifact`` in
    ``graders/judge.py``, which is about human-labeled examples used to
    measure a judge's own accuracy). Here, ``held_out`` means something
    narrower: this particular evaluation dataset was never published
    anywhere, so it's structurally impossible for it to have leaked into
    any model's training data.

    Like ``ResolvedDataset.gated`` below, every field here is
    record-keeping only -- nothing in this codebase reads these fields to
    block or refuse a run; they exist so a human reviewing results later
    knows how much to trust them.

    Attributes:
        status: The current best-effort contamination label for this
            dataset (see ``ContaminationStatus``). Defaults to ``UNKNOWN``,
            the honest "we haven't checked" starting point.
        authored_after: When this dataset's content was written, if known
            -- useful for reasoning about whether it could have existed
            early enough to appear in a given model's training data.
        public_since: When this dataset was first made publicly available,
            if it ever was. Left unset for data that has never been
            published.
        canary_ids: IDs of any "canary" tripwire tokens planted in this
            dataset -- fake markers that shouldn't appear in a legitimate
            answer, used to catch a model regurgitating memorized content.
        held_out: Whether this dataset was authored specifically for
            evaluation and never published anywhere, meaning it cannot be
            in anyone's training data by construction. Defaults to
            ``False``.
    """

    status: ContaminationStatus = ContaminationStatus.UNKNOWN
    authored_after: datetime | None = None
    public_since: datetime | None = None
    canary_ids: tuple[str, ...] = ()
    held_out: bool = False

    @model_validator(mode="after")
    def _validate_consistency(self) -> "ContaminationMetadata":
        if self.held_out and self.public_since is not None:
            raise ValueError(
                "held_out=True is inconsistent with a non-null public_since "
                "(a dataset cannot be both withheld from publication and have "
                "a known public-release date)"
            )
        if (
            self.authored_after is not None
            and self.public_since is not None
            and self.authored_after > self.public_since
        ):
            raise ValueError(
                "authored_after must not be later than public_since "
                "(a dataset cannot be authored after its own public release date)"
            )
        return self


class ResolvedDataset(FrozenModel):
    """Records the exact, unchanging version of a dataset that a run actually used (design §5.2).

    Fields describing size, statistics, and file listings come from the
    dataset provider's own API, which doesn't always return all of them
    for every dataset -- so those fields are optional here, and a missing
    value means "the provider didn't tell us this," not "this dataset has
    none."

    Attributes:
        dataset_id: The dataset's canonical ID with its provider.
        revision: The exact version actually resolved and used (e.g. a
            commit SHA), so the run can always be traced back to precisely
            this data, even if "latest" later changes upstream.
        config: Which named sub-configuration was actually used.
        split: Which split was actually used.
        selected_files: The specific files actually pulled, for datasets
            organized as loose files.
        schema_metadata: What the provider reported about this dataset's
            column layout and types.
        row_count: How many rows this dataset has, if the provider reports
            it.
        license: The dataset's license, if the provider reports one.
        citation: The dataset's suggested citation text, if the provider
            reports one.
        gated: Whether the provider requires special access (e.g.
            accepting terms, or an access request) before this dataset can
            be downloaded. Recorded exactly as reported by the provider;
            not something this codebase enforces itself.
        card_metadata: The dataset's full "dataset card" -- descriptive
            metadata the provider publishes about it -- captured as-is.
        retrieved_at: When this dataset was actually fetched.
        provider_response_digests: Hashes of the provider's raw API
            responses used to resolve this dataset, kept so you could
            later prove exactly what the provider told us at the time.
        cache_manifest_digest: A hash identifying the local cache entry
            this resolution was read from or written to, if caching was
            used.
        checksums: Hashes of the actual downloaded data files, so you can
            later confirm none of them changed or got corrupted.
        contamination: What's known about this dataset's risk of having
            leaked into a model's training data (see
            ``ContaminationMetadata``). Left unset when nothing has been
            recorded.
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
    """One row exactly as the dataset provider returned it, tagged with an ID and an integrity hash.

    A row in this raw, as-received shape never flows straight into
    execution or grading (design §5.3). Instead, a ``BenchmarkAdapter``
    always converts ("projects") a ``SourceRecord`` into an ``EvalSample``
    first -- the cleaned-up shape the rest of the pipeline actually
    understands.

    Attributes:
        row_id: This row's identifier within its source dataset.
        data: The row's actual content, exactly as the provider returned
            it -- untouched, unconverted.
        digest: A hash of ``data``: a short fingerprint that lets later
            code confirm this row hasn't changed or been corrupted.
    """

    row_id: str
    data: dict[str, JsonValue]
    digest: str


class SearchHit(FrozenModel):
    """One dataset search result summary.

    Attributes:
        dataset_id: The matched dataset's canonical ID.
        provider: Which dataset source returned this result.
        revision: The specific version this result refers to, if the
            provider reports one at search time.
        tags: Labels the provider attaches to this dataset (e.g. topic or
            task-type tags).
        gated: Whether this dataset requires special access before it can
            be downloaded.
        private: Whether this dataset is private rather than publicly
            listed.
        downloads: The provider's own reported download count, if
            available -- a rough popularity signal, not verified by this
            codebase.
        card_metadata: The dataset's descriptive "card" metadata, captured
            as reported by the provider's search results.
    """

    dataset_id: str
    provider: str
    revision: str | None = None
    tags: tuple[str, ...] = ()
    gated: bool = False
    private: bool = False
    downloads: int | None = None
    card_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SearchPage(FrozenModel):
    """One page of dataset search results, plus a cursor for fetching the next page.

    Attributes:
        hits: The search results on this page.
        cursor: An opaque token to pass back to fetch the next page --
            treat it as a string to hand back, not something to parse or
            interpret yourself. Left unset when there is no next page.
        total_hits: The total number of matches across all pages, if the
            provider reports it.
    """

    hits: tuple[SearchHit, ...] = ()
    cursor: str | None = None
    total_hits: int | None = None


class SamplePage(FrozenModel):
    """One page of raw rows, returned while previewing or paging through a dataset provider's data.

    Attributes:
        records: The raw rows on this page.
        offset: How many rows come before this page, so pages can be
            requested in sequence.
        total_rows: The dataset's total row count, if the provider reports
            it.
    """

    records: tuple[SourceRecord, ...] = ()
    offset: int = 0
    total_rows: int | None = None
