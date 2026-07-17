"""Safely reads and writes manifest YAML files for the CLI (design §11.1, plan Task 14 Step 3).

A manifest *file* on disk (as opposed to the in-memory
:class:`~agentic_evalkit.models.EvalRunManifest` object) carries two extra
things: an explicit ``schema_version`` header at the top, and a ``target``
block describing which actual ``ExecutionTarget`` the CLI should build
before running -- for example, a Python import path to call directly, a
subprocess command line to invoke, or an HTTP URL plus the *name* of a
credential hook to use (never a literal secret -- see below).
:class:`ManifestDocument` bundles the parsed manifest together with that
target description. :func:`load_manifest` and :func:`dump_manifest` are the
only two functions that actually read or write files -- everything else in
this module is just data shaping in between.

``load_manifest`` always parses YAML using ``yaml.safe_load``, which is
important: unlike a plain ``yaml.load``, it never acts on special YAML tags
that could construct arbitrary Python objects, so a malicious manifest file
can't be used to run arbitrary code just by loading it. Every validation
problem found while loading is collected into one
:class:`~agentic_evalkit.errors.ManifestValidationError`, whose ``context``
holds a list of ``{"path": ..., "message": ...}`` entries -- so a user
looking at the error can immediately see which specific field in their YAML
file is wrong. This module also never does shell-style environment variable
substitution (expanding something like ``${VAR}`` into an environment
variable's value) in either direction. That's a deliberate rule: secrets
are only ever supplied through target/provider hooks at run time (design
§12), never written into or read from the manifest file itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, TypeAdapter, ValidationError
from pydantic import JsonValue as ModelJsonValue

from agentic_evalkit.errors import JsonValue as ErrorContextValue
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.models import (
    ContaminationMetadata,
    DatasetRef,
    DatasetSelection,
    EvalRunManifest,
    SamplingPolicy,
)
from agentic_evalkit.models.base import FrozenModel

# There are two different "JsonValue" types in this codebase:
# ``errors.JsonValue`` (a standard-library-only type that
# ``AgenticEvalkitError.context`` is declared to accept) and pydantic's own
# ``JsonValue``. They describe the same shape of data but aren't literally
# the same type, so we keep them named differently here -- following the
# same convention already used in ``datasets/huggingface.py``. This avoids a
# category of mypy error (it treats two different-but-similar dict types as
# incompatible even when their values are compatible) that would otherwise
# show up at every ``context={...}`` call below. ``ErrorContext`` is exactly
# the value type that ``AgenticEvalkitError.__init__`` expects for its
# ``context`` argument.
ErrorContext = dict[str, ErrorContextValue]

__all__ = [
    "CallableTargetConfig",
    "CliTarget",
    "HttpTargetConfig",
    "ManifestDocument",
    "SubprocessTargetConfig",
    "dump_manifest",
    "load_manifest",
]


class CallableTargetConfig(FrozenModel):
    """A Python import string of the form ``module.path:attribute``."""

    kind: Literal["callable"] = "callable"
    import_string: str


class SubprocessTargetConfig(FrozenModel):
    """An argv list invoked as a subprocess (design §8, plan Task 9)."""

    kind: Literal["subprocess"] = "subprocess"
    argv: tuple[str, ...] = Field(min_length=1)


class HttpTargetConfig(FrozenModel):
    """A URL to call, plus the *name* of a credential hook -- never an actual secret value.

    ``credential_hook`` names some mechanism outside the manifest file
    itself (for example, the name of an environment variable to read, or a
    header-provider callback the caller registered in code) that supplies
    the real header or token when the run actually happens. The manifest
    file itself never contains the credential's value (design §12).
    """

    kind: Literal["http"] = "http"
    url: str
    credential_hook: str | None = None


#: The type of every kind of target configuration the CLI knows how to
#: build, combined into one. Pydantic uses the ``kind`` field to decide
#: which of the three shapes (``callable``, ``subprocess``, or ``http``) a
#: given ``target`` block is supposed to be, *before* trying to validate it
#: against any of them (this technique is called "discriminating" on the
#: ``kind`` field). The alternative -- trying all three shapes in turn and
#: seeing which one fits -- would mean a mistake in, say, an ``http`` block
#: reports a confusing "none of these three shapes matched" error, instead
#: of pointing at the specific field that's actually wrong in the ``http``
#: shape the author clearly intended.
CliTarget = Annotated[
    CallableTargetConfig | SubprocessTargetConfig | HttpTargetConfig,
    Field(discriminator="kind"),
]

_CLI_TARGET_ADAPTER: TypeAdapter[
    CallableTargetConfig | SubprocessTargetConfig | HttpTargetConfig
] = TypeAdapter(CliTarget)


class _ManifestFile(FrozenModel):
    """A manifest's on-disk shape: an ``EvalRunManifest`` plus a CLI target block.

    Some field names here deliberately differ from ``EvalRunManifest``,
    because the on-disk manifest is meant to be easy for a person to write
    by hand, which isn't always the same as what's most convenient for the
    internal model used everywhere else in the package (for example, this
    uses ``dataset`` where ``EvalRunManifest`` uses ``dataset_ref``, and a
    nested ``target`` block instead of a single ``target_name`` string).
    :func:`load_manifest` is what translates between the two.
    """

    run_name: str
    dataset: DatasetRef
    adapter: str
    grader: str
    target: CallableTargetConfig | SubprocessTargetConfig | HttpTargetConfig = Field(
        discriminator="kind"
    )
    revision_policy: str | None = None
    selection: dict[str, ModelJsonValue] = Field(default_factory=dict)
    sampling: dict[str, ModelJsonValue] = Field(default_factory=dict)
    attempts: int = 1
    timeout_seconds: float | None = None
    concurrency: int = 1
    artifact_policy: dict[str, ModelJsonValue] = Field(default_factory=dict)
    redaction_policy: dict[str, ModelJsonValue] = Field(default_factory=dict)
    environment_fingerprint: str | None = None
    code_fingerprint: str | None = None
    target_fingerprint: str | None = None
    baseline_compatibility_rules: dict[str, ModelJsonValue] = Field(default_factory=dict)
    contamination: ContaminationMetadata | None = None


#: The fixed placeholder name used as ``EvalRunManifest.target_name`` for
#: every manifest loaded through the CLI. The CLI always builds exactly one
#: target per run (from the manifest's own ``target`` block) and registers
#: it under this single name -- so for CLI-driven runs,
#: ``EvalRunManifest.target_name`` is always this same constant; it never
#: needs to be anything else.
_CLI_TARGET_NAME = "cli-target"


class ManifestDocument(FrozenModel):
    """A fully validated manifest file: a ready-to-run manifest plus its CLI target description.

    ``manifest`` is a genuine :class:`~agentic_evalkit.models.EvalRunManifest`
    -- the exact same type :class:`~agentic_evalkit.runner.EvalRunner`
    consumes -- so nothing downstream of loading a manifest needs to know
    this ``ManifestDocument`` wrapper type even exists. ``target`` is the
    CLI-specific instruction describing *which* concrete
    :class:`~agentic_evalkit.targets.base.ExecutionTarget` to build and
    register under ``manifest.target_name`` before the run starts.
    """

    manifest: EvalRunManifest
    target: CallableTargetConfig | SubprocessTargetConfig | HttpTargetConfig = Field(
        discriminator="kind"
    )


def _field_errors(error: ValidationError) -> tuple[ErrorContextValue, ...]:
    """Turn a Pydantic ``ValidationError`` into a simple list of (field path, message) pairs.

    Returns a tuple rather than a list on purpose: the ``errors.JsonValue``
    type (which every ``AgenticEvalkitError.context`` value must match)
    represents a sequence as a ``tuple[JsonValue, ...]``, not as a mutable
    list. Likewise, each entry's type is exactly ``ErrorContextValue``
    (which is ``errors.JsonValue``) rather than a more specific type like
    ``dict[str, str]`` -- matching the exact expected type here means it can
    be passed straight into a ``context={...}`` argument without mypy
    complaining about a type mismatch.
    """
    entries: tuple[ErrorContextValue, ...] = tuple(
        {"path": ".".join(str(part) for part in item["loc"]) or "<root>", "message": item["msg"]}
        for item in error.errors()
    )
    return entries


def load_manifest(path: str | Path) -> ManifestDocument:
    """Read a manifest YAML file from disk, validate it, and return a :class:`ManifestDocument`.

    Always parses with ``yaml.safe_load``, so special YAML tags like
    ``!!python/...`` are never acted on and can never be used to execute
    arbitrary code. The file's top level must be a single YAML mapping
    (i.e. a set of key/value pairs) -- a list, a bare scalar value, or an
    empty file is rejected right away, before Pydantic even gets a chance to
    validate it. No ``${VAR}``-style environment variable substitution is
    ever performed, either on the raw file text or on the values after
    parsing.

    Raises:
        ManifestValidationError: Raised if the file can't be read, isn't
            valid YAML, doesn't parse down to a single mapping, or fails
            schema validation. In every case, ``context["errors"]`` is a
            list of ``{"path": ..., "message": ...}`` entries that pinpoint
            exactly which field(s) are at fault.
    """
    resolved_path = Path(path)
    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestValidationError(
            message=f"could not read manifest file {resolved_path}: {error}",
            context={
                "path": str(resolved_path),
                "errors": ({"path": "<file>", "message": str(error)},),
            },
        ) from error

    try:
        raw_document = yaml.safe_load(raw_text)
    except yaml.YAMLError as error:
        raise ManifestValidationError(
            message=f"manifest file {resolved_path} is not valid YAML: {error}",
            context={
                "path": str(resolved_path),
                "errors": ({"path": "<root>", "message": str(error)},),
            },
        ) from error

    if not isinstance(raw_document, dict):
        actual_type = type(raw_document).__name__
        raise ManifestValidationError(
            message=(
                f"manifest file {resolved_path} must decode to a single YAML mapping, "
                f"got {actual_type}"
            ),
            context={
                "path": str(resolved_path),
                "errors": (
                    {
                        "path": "<root>",
                        "message": f"expected a mapping, got {actual_type}",
                    },
                ),
            },
        )

    try:
        parsed = _ManifestFile.model_validate(raw_document)
    except ValidationError as error:
        raise ManifestValidationError(
            message=f"manifest file {resolved_path} failed validation",
            context={"path": str(resolved_path), "errors": _field_errors(error)},
        ) from error

    try:
        manifest = EvalRunManifest(
            run_name=parsed.run_name,
            dataset_ref=parsed.dataset,
            revision_policy=parsed.revision_policy,
            adapter=parsed.adapter,
            grader=parsed.grader,
            target_name=_CLI_TARGET_NAME,
            selection=DatasetSelection.model_validate(parsed.selection),
            sampling=SamplingPolicy.model_validate(parsed.sampling),
            attempts=parsed.attempts,
            timeout_seconds=parsed.timeout_seconds,
            concurrency=parsed.concurrency,
            artifact_policy=parsed.artifact_policy,
            redaction_policy=parsed.redaction_policy,
            environment_fingerprint=parsed.environment_fingerprint,
            code_fingerprint=parsed.code_fingerprint,
            target_fingerprint=parsed.target_fingerprint,
            baseline_compatibility_rules=parsed.baseline_compatibility_rules,
            contamination=parsed.contamination,
        )
    except ValidationError as error:
        raise ManifestValidationError(
            message=f"manifest file {resolved_path} failed validation",
            context={"path": str(resolved_path), "errors": _field_errors(error)},
        ) from error

    return ManifestDocument(manifest=manifest, target=parsed.target)


def dump_manifest(document: ManifestDocument) -> str:
    """Render ``document`` back out as YAML text that's stable and easy for a human to hand-edit.

    Always writes out an explicit ``schema_version``, the resolved
    dataset's provider/config/split, the adapter, target, grader, attempt
    count, timeout, concurrency, and artifact policy -- so the resulting
    YAML is a complete, self-contained input you could feed straight back
    into :func:`load_manifest`, with nothing left implicit or assumed.
    Dictionary keys are written in sorted order, so dumping two equivalent
    documents always produces byte-for-byte identical YAML text. This never
    writes out ``${VAR}``-style placeholder syntax -- every value is written
    out literally, exactly as it's stored in :class:`ManifestDocument`.
    """
    manifest = document.manifest
    payload: dict[str, object] = {
        "schema_version": manifest.schema_version,
        "run_name": manifest.run_name,
        "dataset": manifest.dataset_ref.model_dump(mode="json", exclude_none=True),
        "adapter": manifest.adapter,
        "grader": manifest.grader,
        "target": document.target.model_dump(mode="json", exclude_none=True),
        "selection": manifest.selection.model_dump(mode="json"),
        "sampling": manifest.sampling.model_dump(mode="json"),
        "attempts": manifest.attempts,
        "timeout_seconds": manifest.timeout_seconds,
        "concurrency": manifest.concurrency,
        "artifact_policy": manifest.artifact_policy,
    }
    if manifest.revision_policy is not None:
        payload["revision_policy"] = manifest.revision_policy
    if manifest.redaction_policy:
        payload["redaction_policy"] = manifest.redaction_policy
    if manifest.environment_fingerprint is not None:
        payload["environment_fingerprint"] = manifest.environment_fingerprint
    if manifest.code_fingerprint is not None:
        payload["code_fingerprint"] = manifest.code_fingerprint
    if manifest.target_fingerprint is not None:
        payload["target_fingerprint"] = manifest.target_fingerprint
    if manifest.baseline_compatibility_rules:
        payload["baseline_compatibility_rules"] = manifest.baseline_compatibility_rules
    if manifest.contamination is not None:
        payload["contamination"] = manifest.contamination.model_dump(mode="json", exclude_none=True)

    return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
