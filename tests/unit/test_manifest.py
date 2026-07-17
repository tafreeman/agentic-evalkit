"""Tests for safe manifest loading/dumping (plan Task 14, Steps 1 and 3).

A manifest is the YAML file that describes one evaluation run. ``load_manifest``
must always parse it with ``yaml.safe_load`` -- never a plain ``yaml.load``,
which can be tricked into constructing arbitrary Python objects via special
YAML tags -- and must reject anything that doesn't decode to exactly one
YAML mapping (a single set of key/value pairs) at the top level. Once
parsed, it validates the result against ``EvalRunManifest`` plus the
CLI-specific ``target`` block (which describes which execution target to
build), and reports any problems it finds through
``ManifestValidationError`` with a clear list of exactly which fields are
wrong, instead of letting a raw Pydantic traceback leak through to the
user. ``dump_manifest`` does the reverse job: turning a manifest back into
stable YAML text that always includes an explicit schema version and every
field needed to run it again exactly as it ran before.
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

#: A distinctive, obviously-fake secret value used only in these tests, so
#: it's easy to search for in output and prove it was never written
#: anywhere. It represents the *actual value* an
#: ``HttpTargetConfig.credential_hook`` environment variable would hold at
#: run time -- the tests below check that this value never ends up
#: persisted to disk.
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
    # If this "!!python/object/apply" YAML tag were ever acted on, it would
    # call os.system("echo unsafe") -- i.e. run an arbitrary shell command --
    # just from loading this file. yaml.safe_load refuses to act on tags
    # like this at all, which is exactly why load_manifest is required to
    # use yaml.safe_load rather than the plain, unsafe yaml.load. What this
    # test checks is that the refusal always surfaces to the caller as our
    # own typed ManifestValidationError -- never as the command actually
    # running, and never as an unhandled yaml.constructor.ConstructorError
    # (PyYAML's own internal error type) leaking out instead.
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
    # No matter exactly which fields turn out to be missing or invalid, the
    # error must name at least one specific field path, so a user editing
    # their YAML file knows where to look.
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
    # Manifests are never allowed to do shell-style "${VAR}" substitution --
    # expanding a placeholder like ${VAR} into an environment variable's
    # actual value (plan Task 14 Step 3). This checks that dump_manifest
    # never even writes out that "${...}" syntax in the first place, even
    # if some value happened to literally contain a "$" character.
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


# --- Story 2.3 (R-002): a credential hook's real secret value must never be written to disk ---
#
# ``HttpTargetConfig.credential_hook`` doesn't store a secret at all -- it
# stores the *name* of an environment variable that the CLI reads at run
# time (see ``agentic_evalkit.cli.runs._load_http_target``, which calls
# ``os.environ.get(credential_hook)``). Whatever secret value that
# environment variable actually holds must never be written down anywhere.
# The tests below prove exactly that: with the environment variable set to
# a fake secret value, neither the dumped YAML nor a JSON round-trip of the
# document contains that value anywhere -- only the hook's name (which
# environment variable to read from) is ever recorded. These tests only
# cover the manifest/YAML side of that guarantee; the equivalent checks for
# the full CLI path (building the target and writing a report) live in
# ``tests/integration/test_cli.py``.


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
    """Even with the environment variable actually set to a real-looking
    secret, the manifest's target only ever stores the *name* of that
    environment variable -- the resolved secret value itself never appears
    anywhere on the model.
    """
    monkeypatch.setenv(_HOOK_ENV_NAME, _HOOK_SECRET)
    document = _http_hook_document()
    assert isinstance(document.target, HttpTargetConfig)
    assert document.target.credential_hook == _HOOK_ENV_NAME
    # Serializing the whole document to JSON includes the hook's name, but
    # never the secret value it points to.
    serialized = document.model_dump_json()
    assert _HOOK_ENV_NAME in serialized
    assert _HOOK_SECRET not in serialized


def test_dumped_manifest_yaml_never_contains_the_resolved_hook_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dump_manifest`` writes the environment variable's *name* into the
    ``target`` block, but never the secret value it resolves to at run time
    -- so if a dumped manifest is ever committed to a repo or shared with
    someone, it can't leak the credential.
    """
    monkeypatch.setenv(_HOOK_ENV_NAME, _HOOK_SECRET)
    dumped = dump_manifest(_http_hook_document())
    assert _HOOK_ENV_NAME in dumped
    assert _HOOK_SECRET not in dumped
    # Parsing the YAML back confirms the hook is stored as just the
    # environment-variable name (credential_hook) -- not as an actual token
    # or credential value written directly into the file.
    parsed = yaml.safe_load(dumped)
    assert parsed["target"]["credential_hook"] == _HOOK_ENV_NAME
    assert "authorization" not in dumped.lower()


def test_manifest_round_trip_preserves_hook_name_and_drops_no_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dump an HTTP-hook manifest to YAML and load it back again: the hook
    name comes back exactly as it was, and the secret value never appears
    anywhere along the way.
    """
    monkeypatch.setenv(_HOOK_ENV_NAME, _HOOK_SECRET)
    document = _http_hook_document()
    dumped = dump_manifest(document)
    round_trip_path = tmp_path / "http-hook.yaml"
    round_trip_path.write_text(dumped, encoding="utf-8")

    reloaded = load_manifest(round_trip_path)
    assert isinstance(reloaded.target, HttpTargetConfig)
    assert reloaded.target.credential_hook == _HOOK_ENV_NAME
    # The file actually written to disk never contained the secret either.
    assert _HOOK_SECRET not in round_trip_path.read_text(encoding="utf-8")
