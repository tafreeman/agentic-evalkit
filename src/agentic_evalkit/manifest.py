"""Safe manifest loading/dumping for the CLI (design §11.1, plan Task 14 Step 3).

A manifest *file* on disk carries two things a bare
:class:`~agentic_evalkit.models.EvalRunManifest` does not: an explicit
``schema_version`` header, and a ``target`` block describing which concrete
``ExecutionTarget`` (callable import string, subprocess argv, or HTTP URL
plus named credential hook) the CLI should construct before running.
:class:`ManifestDocument` bundles the two; :func:`load_manifest` and
:func:`dump_manifest` are its only I/O boundary.

``load_manifest`` uses ``yaml.safe_load`` exclusively -- it never resolves
Python object tags -- and reports every validation failure as a single
:class:`~agentic_evalkit.errors.ManifestValidationError` whose ``context``
carries a list of ``{"path": ..., "message": ...}`` entries so a user can
find the exact field that is wrong. Environment interpolation is forbidden:
this module never expands ``${VAR}``-style syntax, in either direction;
secret values only ever enter through target/provider hooks (design §12),
never through the manifest file itself.
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

# ``errors.JsonValue`` (the stdlib-only type ``AgenticEvalkitError.context``
# is typed against) and pydantic's ``JsonValue`` are structurally similar
# but not the same type; keeping them distinctly named -- matching the
# convention already established in ``datasets/huggingface.py`` -- avoids
# mypy dict-invariance mismatches at every ``context={...}`` call site
# below. ``ErrorContext`` is the exact value type
# ``AgenticEvalkitError.__init__`` expects for its ``context`` mapping.
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
    """A URL plus the *name* of a credential hook (never a literal secret).

    ``credential_hook`` names an out-of-band mechanism (for example, an
    environment variable name or a caller-registered header-provider
    callback) that supplies the actual header/token at run time. The
    manifest file itself never carries a credential value (design §12).
    """

    kind: Literal["http"] = "http"
    url: str
    credential_hook: str | None = None


#: Discriminated union of every target configuration the CLI can construct.
#: Discriminating on ``kind`` (rather than trying each shape in turn) means a
#: malformed ``target`` block always reports the *intended* kind's field
#: errors, not a confusing "none of these three shapes matched" fan-out.
CliTarget = Annotated[
    CallableTargetConfig | SubprocessTargetConfig | HttpTargetConfig,
    Field(discriminator="kind"),
]

_CLI_TARGET_ADAPTER: TypeAdapter[
    CallableTargetConfig | SubprocessTargetConfig | HttpTargetConfig
] = TypeAdapter(CliTarget)


class _ManifestFile(FrozenModel):
    """The on-disk manifest shape: an ``EvalRunManifest`` plus a CLI target.

    Field names here intentionally differ from ``EvalRunManifest`` where the
    on-disk manifest is more ergonomic to hand-author than the wire model
    (``dataset`` instead of ``dataset_ref``, a nested ``target`` block
    instead of a single ``target_name`` string); :func:`load_manifest`
    reconciles the two.
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


#: The synthetic ``EvalRunManifest.target_name`` every CLI-loaded manifest
#: uses. The CLI always constructs exactly one target per run (from the
#: manifest's own ``target`` block) and registers it under this one name, so
#: ``EvalRunManifest.target_name`` never needs to vary for CLI-driven runs.
_CLI_TARGET_NAME = "cli-target"


class ManifestDocument(FrozenModel):
    """A validated manifest file: a wire-ready manifest plus its CLI target.

    ``manifest`` is a genuine :class:`~agentic_evalkit.models.EvalRunManifest`
    -- the same type :class:`~agentic_evalkit.runner.EvalRunner` consumes --
    so nothing downstream of manifest loading needs to know this document
    type exists. ``target`` is the CLI-specific instruction for *which*
    concrete :class:`~agentic_evalkit.targets.base.ExecutionTarget` to build
    and register under ``manifest.target_name`` before running.
    """

    manifest: EvalRunManifest
    target: CallableTargetConfig | SubprocessTargetConfig | HttpTargetConfig = Field(
        discriminator="kind"
    )


def _field_errors(error: ValidationError) -> tuple[ErrorContextValue, ...]:
    """Flatten a Pydantic ``ValidationError`` into field-path/message pairs.

    Returns a tuple (not a list): every ``AgenticEvalkitError.context`` value
    must be JSON-compatible per ``errors.JsonValue``, which represents
    sequences as ``tuple[JsonValue, ...]`` rather than mutable lists. Each
    entry's type is exactly ``ErrorContextValue`` (``errors.JsonValue``),
    not a narrower ``dict[str, str]``, so it slots into a ``context={...}``
    mapping without a dict-invariance mismatch.
    """
    entries: tuple[ErrorContextValue, ...] = tuple(
        {"path": ".".join(str(part) for part in item["loc"]) or "<root>", "message": item["msg"]}
        for item in error.errors()
    )
    return entries


def load_manifest(path: str | Path) -> ManifestDocument:
    """Load and validate a manifest YAML file into a :class:`ManifestDocument`.

    Uses ``yaml.safe_load`` exclusively, so ``!!python/...`` tags are never
    resolved and can never execute arbitrary code. The file must decode to
    exactly one YAML mapping; a list, scalar, or empty document is rejected
    before Pydantic ever sees it. No ``${VAR}``-style environment
    interpolation is performed on the raw text or the decoded values.

    Raises:
        ManifestValidationError: The file is not readable YAML, does not
            decode to a single mapping, or fails schema validation. In every
            case ``context["errors"]`` is a list of
            ``{"path": ..., "message": ...}`` entries identifying the exact
            field(s) at fault.
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
    """Render ``document`` as stable, hand-editable manifest YAML.

    Always emits an explicit ``schema_version``, the resolved dataset
    provider/config/split, adapter, target, grader, attempts, timeout,
    concurrency, and artifact policy, so a dumped manifest is a complete,
    reproducible input to :func:`load_manifest` with nothing implicit. Keys
    are sorted so two dumps of an equivalent document are byte-identical.
    Never emits ``${VAR}``-style interpolation syntax -- values are written
    literally, exactly as :class:`ManifestDocument` holds them.
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
