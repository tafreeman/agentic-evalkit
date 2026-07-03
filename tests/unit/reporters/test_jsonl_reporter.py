"""Tests for the JSONL reporter (design §11.3, plan Task 13)."""

import json
from pathlib import Path

from conftest import _run_with_pass_error_timeout_and_provenance

from agentic_evalkit.reporters import JsonlReporter


def test_jsonl_writes_header_one_record_per_sample_and_trailer(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    jsonl_path = JsonlReporter().write(run, tmp_path / "run.jsonl", generated_at="fixed")
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + len(run.samples) + 1

    header = json.loads(lines[0])
    assert header["record_type"] == "header"
    assert header["manifest"]["run_name"] == "gsm8k-smoke"
    assert header["provenance"]["dataset_revision"] == "abc"
    assert header["summary"]["total"] == 3

    sample_records = [json.loads(line) for line in lines[1:-1]]
    assert {record["record_type"] for record in sample_records} == {"sample"}
    assert {record["execution"]["status"] for record in sample_records} == {
        "completed",
        "error",
        "timeout",
    }

    trailer = json.loads(lines[-1])
    assert trailer["record_type"] == "trailer"
    assert trailer["summary"]["total"] == 3
    assert trailer["generated_at"] == "fixed"


def test_jsonl_sample_records_preserve_sample_order(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    jsonl_path = JsonlReporter().write(run, tmp_path / "run.jsonl", generated_at="fixed")
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    sample_records = [json.loads(line) for line in lines[1:-1]]
    assert [record["sample"]["sample_id"] for record in sample_records] == [
        sample.sample.sample_id for sample in run.samples
    ]


def test_jsonl_each_line_is_compact_single_line_json(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    jsonl_path = JsonlReporter().write(run, tmp_path / "run.jsonl", generated_at="fixed")
    raw = jsonl_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    for line in raw.splitlines():
        assert line == line.strip()
        json.loads(line)  # each line parses independently


def test_two_renders_of_the_same_run_are_byte_identical_with_fixed_generated_at(
    tmp_path: Path,
) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    first = JsonlReporter().write(run, tmp_path / "first.jsonl", generated_at="fixed")
    second = JsonlReporter().write(run, tmp_path / "second.jsonl", generated_at="fixed")
    assert first.read_bytes() == second.read_bytes()
