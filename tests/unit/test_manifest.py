"""Tests for safe manifest loading/dumping (plan Task 14, Steps 1 and 3).

``load_manifest`` must use ``yaml.safe_load`` only (never resolve Python
tags), require exactly one top-level mapping, validate the result against
``EvalRunManifest`` plus the CLI-specific ``target`` block, and report field
paths through ``ManifestValidationError`` rather than a raw Pydantic
traceback. ``dump_manifest`` is the inverse: stable YAML with an explicit
schema version and every field a run needs to be reproducible.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.manifest import (
    CliTarget,
    HttpTargetConfig,
    ManifestDocument,
    dump_manifest,
    load_manifest,
)
from agentic_evalkit.models import DatasetRef, EvalRunManifest, SamplingPolicy

#: A distinctive sentinel that must never be persisted anywhere: it is the
#: *value* an ``HttpTargetConfig.credential_hook`` env var holds at run time.
_HOOK_SECRET = "hook-secret-XYZZY-do-not-persist"
_HOOK_ENV_NAME = "AGENTIC_EVALKIT_TEST_CRED_HOOK"


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_the_gsm8k_fixture_manifest() -> None:
    fixture = Path("tests/fixtures/manifests/gsm8k.yaml")
    document = load_manifest(fixture)
    assert document.manifest.run_name == "gsm8k-smoke"
    assert document.manifest.dataset_ref.dataset_id == "openai/gsm8k"
    assert document.manifest.dataset_ref.config == "main"
    assert document.manifest.dataset_ref.split == "test"
    assert document.manifest.adapter == "gsm8k@1"
    assert document.manifest.grader == "normalized-exact@1"
    assert document.target.kind == "callable"
    assert document.target.import_string == "agentic_evalkit.examples.zero_target:zero_target"


def test_rejects_non_mapping_yaml_documents(tmp_path: Path) -> None:
    path = _write(tmp_path, "- one\n- two\n")
    with pytest.raises(ManifestValidationError, match="mapping"):
        load_manifest(path)


def test_rejects_empty_yaml_documents(tmp_path: Path) -> None:
    path = _write(tmp_path, "")
    with pytest.raises(ManifestValidationError, match="mapping"):
        load_manifest(path)


def test_never_resolves_python_tags(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "run_name: !!python/object/apply:os.system ['echo unsafe']\n",
    )
    # yaml.safe_load itself refuses the !!python/object/apply tag: this must
    # surface as our typed ManifestValidationError, never an executed side
    # effect and never an unhandled yaml.constructor.ConstructorError.
    with pytest.raises(ManifestValidationError):
        load_manifest(path)


def test_reports_field_paths_for_validation_errors(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
run_name: incomplete
dataset:
  provider: huggingface
  dataset_id: openai/gsm8k
adapter: gsm8k@1
grader: normalized-exact@1
target:
  kind: callable
""".strip()
        + "\n",
    )
    with pytest.raises(ManifestValidationError) as excinfo:
        load_manifest(path)
    # Whatever the exact set of missing/invalid fields, the error must name
    # at least one field path so a user can find the problem in their file.
    errors = excinfo.value.context.get("errors")
    assert errors
    assert any("import_string" in entry["path"] for entry in errors)  # type: ignore[union-attr]


def test_missing_target_block_is_a_validation_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
run_name: no-target
dataset:
  provider: huggingface
  dataset_id: openai/gsm8k
  config: main
  split: test
adapter: gsm8k@1
grader: normalized-exact@1
""".strip()
        + "\n",
    )
    with pytest.raises(ManifestValidationError):
        load_manifest(path)


def test_dump_manifest_emits_stable_yaml_with_explicit_fields() -> None:
    fixture = Path("tests/fixtures/manifests/gsm8k.yaml")
    document = load_manifest(fixture)

    dumped = dump_manifest(document)

    assert isinstance(dumped, str)
    parsed = yaml.safe_load(dumped)
    assert parsed["schema_version"] == "1"
    assert parsed["run_name"] == "gsm8k-smoke"
    assert parsed["dataset"]["config"] == "main"
    assert parsed["dataset"]["split"] == "test"
    assert parsed["adapter"] == "gsm8k@1"
    assert parsed["grader"] == "normalized-exact@1"
    assert parsed["target"]["kind"] == "callable"
    assert "attempts" in parsed
    assert "timeout_seconds" in parsed
    assert "concurrency" in parsed
    assert "artifact_policy" in parsed


def test_dump_manifest_round_trips_through_load_manifest(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/manifests/gsm8k.yaml")
    document = load_manifest(fixture)

    dumped = dump_manifest(document)
    round_tripped_path = tmp_path / "round-tripped.yaml"
    round_tripped_path.write_text(dumped, encoding="utf-8")
    round_tripped = load_manifest(round_tripped_path)

    assert round_tripped.manifest == document.manifest
    assert round_tripped.target == document.target


def test_dump_manifest_round_trips_target_fingerprint(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/manifests/gsm8k.yaml")
    document = load_manifest(fixture)
    with_fingerprint = ManifestDocument(
        manifest=document.manifest.model_copy(update={"target_fingerprint": "sha256:" + "a" * 64}),
        target=document.target,
    )

    dumped = dump_manifest(with_fingerprint)
    parsed = yaml.safe_load(dumped)
    assert parsed["target_fingerprint"] == "sha256:" + "a" * 64

    round_tripped_path = tmp_path / "round-tripped.yaml"
    round_tripped_path.write_text(dumped, encoding="utf-8")
    round_tripped = load_manifest(round_tripped_path)

    assert round_tripped.manifest == with_fingerprint.manifest
    assert round_tripped.manifest.target_fingerprint == "sha256:" + "a" * 64


def test_dump_manifest_omits_target_fingerprint_when_none() -> None:
    fixture = Path("tests/fixtures/manifests/gsm8k.yaml")
    document = load_manifest(fixture)
    assert document.manifest.target_fingerprint is None

    dumped = dump_manifest(document)
    parsed = yaml.safe_load(dumped)
    assert "target_fingerprint" not in parsed


def test_dump_manifest_never_interpolates_environment_variables() -> None:
    fixture = Path("tests/fixtures/manifests/gsm8k.yaml")
    document = load_manifest(fixture)
    dumped = dump_manifest(document)
    # Environment interpolation is forbidden in manifests (plan Task 14 Step
    # 3); the dumper must never emit shell/YAML interpolation syntax such as
    # "${VAR}" even if a value happened to contain literal "$" text.
    assert "${" not in dumped


def test_subprocess_target_requires_a_nonempty_argv() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        CliTarget(kind="subprocess", argv=())


def test_manifest_document_is_immutable() -> None:
    fixture = Path("tests/fixtures/manifests/gsm8k.yaml")
    document = load_manifest(fixture)
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError (frozen)
        document.target.kind = "subprocess"  # type: ignore[misc]


def _manifest_with_attempts(*, sampling_attempts: int, attempts: int) -> EvalRunManifest:
    return EvalRunManifest(
        run_name="attempts-check",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="echo",
        sampling=SamplingPolicy(attempts=sampling_attempts),
        attempts=attempts,
    )


def test_diverging_attempts_are_rejected_with_a_clear_message() -> None:
    with pytest.raises(ValidationError, match=r"sampling\.attempts.*attempts.*equal"):
        _manifest_with_attempts(sampling_attempts=2, attempts=1)


def test_equal_explicit_attempts_are_accepted() -> None:
    manifest = _manifest_with_attempts(sampling_attempts=3, attempts=3)
    assert manifest.sampling.attempts == 3
    assert manifest.attempts == 3


def test_default_attempts_on_both_fields_are_accepted() -> None:
    # Neither "sampling" nor "attempts" is overridden here, so both take
    # their class defaults of 1 -- the common case that must never fail.
    manifest = EvalRunManifest(
        run_name="attempts-defaults",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="echo",
    )
    assert manifest.sampling.attempts == 1
    assert manifest.attempts == 1


def test_load_manifest_surfaces_diverging_attempts_as_manifest_validation_error(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        """
run_name: diverging-attempts
dataset:
  provider: huggingface
  dataset_id: openai/gsm8k
  config: main
  split: test
adapter: gsm8k@1
grader: normalized-exact@1
target:
  kind: callable
  import_string: agentic_evalkit.examples.zero_target:zero_target
sampling:
  attempts: 2
attempts: 1
""".strip()
        + "\n",
    )
    with pytest.raises(ManifestValidationError) as excinfo:
        load_manifest(path)
    errors = excinfo.value.context.get("errors")
    assert errors
    assert any("equal" in entry["message"] for entry in errors)  # type: ignore[union-attr]


# --- Story 2.3 (R-002): credential-hook runtime resolution never recorded ---
#
# ``HttpTargetConfig.credential_hook`` stores the *name* of an env var the CLI
# reads at run time (``agentic_evalkit.cli.runs._load_http_target`` does
# ``os.environ.get(credential_hook)``); the resolved secret must never be
# persisted. These manifest-level guards prove the persistence half of the
# AC: with the env var actually set to the sentinel secret, neither the dumped
# YAML nor a JSON round-trip of the document carries the resolved value -- only
# the hook reference (the env-var name). The CLI-path (build + report)
# assertions live in ``tests/integration/test_cli.py``.


def _http_hook_document() -> ManifestDocument:
    manifest = EvalRunManifest(
        run_name="http-hook",
        dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
        adapter="gsm8k@1",
        grader="normalized-exact@1",
        target_name="cli-target",
    )
    return ManifestDocument(
        manifest=manifest,
        target=HttpTargetConfig(url="https://example.test/eval", credential_hook=_HOOK_ENV_NAME),
    )


def test_manifest_stores_only_the_credential_hook_name_never_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with the hook's env var set to a real secret, the manifest target
    carries only the hook *name*; the resolved value is nowhere on the model.
    """
    monkeypatch.setenv(_HOOK_ENV_NAME, _HOOK_SECRET)
    document = _http_hook_document()
    assert isinstance(document.target, HttpTargetConfig)
    assert document.target.credential_hook == _HOOK_ENV_NAME
    # The whole document, serialized, references the hook name but not the value.
    serialized = document.model_dump_json()
    assert _HOOK_ENV_NAME in serialized
    assert _HOOK_SECRET not in serialized


def test_dumped_manifest_yaml_never_contains_the_resolved_hook_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dump_manifest`` writes the hook name into the ``target`` block but
    never the resolved secret, so a dumped manifest committed to disk cannot
    leak the credential.
    """
    monkeypatch.setenv(_HOOK_ENV_NAME, _HOOK_SECRET)
    dumped = dump_manifest(_http_hook_document())
    assert _HOOK_ENV_NAME in dumped
    assert _HOOK_SECRET not in dumped
    # And the parsed YAML confirms the hook is stored as the credential_hook
    # reference, not an inlined token/value.
    parsed = yaml.safe_load(dumped)
    assert parsed["target"]["credential_hook"] == _HOOK_ENV_NAME
    assert "authorization" not in dumped.lower()


def test_manifest_round_trip_preserves_hook_name_and_drops_no_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dump -> load round-trip of an HTTP-hook manifest preserves the hook
    name exactly and introduces no secret value at any point.
    """
    monkeypatch.setenv(_HOOK_ENV_NAME, _HOOK_SECRET)
    document = _http_hook_document()
    dumped = dump_manifest(document)
    round_trip_path = tmp_path / "http-hook.yaml"
    round_trip_path.write_text(dumped, encoding="utf-8")

    reloaded = load_manifest(round_trip_path)
    assert isinstance(reloaded.target, HttpTargetConfig)
    assert reloaded.target.credential_hook == _HOOK_ENV_NAME
    # The on-disk file never held the secret.
    assert _HOOK_SECRET not in round_trip_path.read_text(encoding="utf-8")
