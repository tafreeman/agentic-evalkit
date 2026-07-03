"""Tests for the canonical JSON reporter (design §11.3, plan Task 13)."""

import json
from pathlib import Path

from conftest import _run_with_pass_error_timeout_and_provenance

from agentic_evalkit.reporters import JsonReporter


def test_json_and_jsonl_retain_sample_evidence(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    json_path = JsonReporter().write(run, tmp_path / "run.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["dataset_revision"] == "abc"
    assert {item["execution"]["status"] for item in payload["samples"]} == {
        "completed",
        "error",
        "timeout",
    }


def test_envelope_has_required_top_level_keys(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    json_path = JsonReporter().write(run, tmp_path / "run.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema_version",
        "run_id",
        "provenance",
        "manifest",
        "resolved_dataset",
        "summary",
        "samples",
        "started_at",
        "finished_at",
        "generated_at",
    }
    assert payload["run_id"] == "run-001"
    assert payload["schema_version"] == "1"


def test_provenance_contains_all_required_fields(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    json_path = JsonReporter().write(run, tmp_path / "run.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    assert provenance == {
        "dataset_id": "openai/gsm8k",
        "dataset_revision": "abc",
        "config": "main",
        "split": "test",
        "adapter": "gsm8k@1",
        "grader": "normalized-exact@1",
        "target_name": "echo-target",
        "environment_fingerprint": "env:sha256:deadbeef",
        "code_fingerprint": "code:sha256:cafef00d",
    }


def test_manifest_and_resolved_dataset_are_fully_serialized(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    json_path = JsonReporter().write(run, tmp_path / "run.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["manifest"]["run_name"] == "gsm8k-smoke"
    assert payload["manifest"]["dataset_ref"]["dataset_id"] == "openai/gsm8k"
    assert payload["resolved_dataset"]["revision"] == "abc"
    assert payload["resolved_dataset"]["row_count"] == 3
    assert payload["summary"]["total"] == 3
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["errors"] == 1
    assert payload["summary"]["timeouts"] == 1


def test_output_uses_sorted_keys_and_two_space_indent(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    json_path = JsonReporter().write(
        run, tmp_path / "run.json", generated_at="2026-07-02T12:05:00+00:00"
    )
    content = json_path.read_text(encoding="utf-8")
    reparsed = json.dumps(json.loads(content), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    assert content == reparsed


def test_generated_at_is_used_verbatim_when_supplied(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    json_path = JsonReporter().write(
        run, tmp_path / "run.json", generated_at="2026-07-02T12:05:00+00:00"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["generated_at"] == "2026-07-02T12:05:00+00:00"


def test_write_is_atomic_and_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    destination = tmp_path / "run.json"
    JsonReporter().write(run, destination, generated_at="fixed")
    remaining = {path.name for path in tmp_path.iterdir()}
    assert remaining == {"run.json"}


def test_rewrite_replaces_prior_contents(tmp_path: Path) -> None:
    destination = tmp_path / "run.json"
    first_run = _run_with_pass_error_timeout_and_provenance()
    JsonReporter().write(first_run, destination, generated_at="first")
    second_run = first_run.model_copy(update={"run_id": "run-002"})
    JsonReporter().write(second_run, destination, generated_at="second")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-002"
    assert payload["generated_at"] == "second"


def test_two_renders_of_the_same_run_are_byte_identical_with_fixed_generated_at(
    tmp_path: Path,
) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    first_path = JsonReporter().write(run, tmp_path / "first.json", generated_at="fixed")
    second_path = JsonReporter().write(run, tmp_path / "second.json", generated_at="fixed")
    assert first_path.read_bytes() == second_path.read_bytes()
