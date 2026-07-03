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

from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.manifest import CliTarget, dump_manifest, load_manifest


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
