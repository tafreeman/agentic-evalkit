"""Tests for the Markdown reporter (design §11.3, plan Task 13)."""

from pathlib import Path

from conftest import _run_with_pass_error_timeout_and_provenance

from agentic_evalkit.reporters import MarkdownReporter


def test_markdown_contains_identity_and_provenance(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    md_path = MarkdownReporter().write(run, tmp_path / "run.md", generated_at="fixed")
    content = md_path.read_text(encoding="utf-8")
    assert "run-001" in content
    assert "gsm8k-smoke" in content
    assert "openai/gsm8k" in content
    assert "abc" in content  # resolved dataset revision
    assert "main" in content  # config
    assert "test" in content  # split
    assert "gsm8k@1" in content  # adapter
    assert "normalized-exact@1" in content  # grader
    assert "echo-target" in content  # target_name


def test_markdown_contains_exact_numerator_and_denominator(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    md_path = MarkdownReporter().write(run, tmp_path / "run.md", generated_at="fixed")
    content = md_path.read_text(encoding="utf-8")
    # 1 passed out of 3 total is the exact numerator/denominator this run produces.
    assert "1/3" in content
    assert "3" in content  # total
    assert "1" in content  # errors
    # Distinct outcome categories must be visible, not merged.
    assert "error" in content.lower()
    assert "timeout" in content.lower()


def test_markdown_contains_compatibility_details(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    md_path = MarkdownReporter().write(run, tmp_path / "run.md", generated_at="fixed")
    content = md_path.read_text(encoding="utf-8")
    assert "env:sha256:deadbeef" in content
    assert "code:sha256:cafef00d" in content


def test_markdown_contains_sample_table_with_evidence(tmp_path: Path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    md_path = MarkdownReporter().write(run, tmp_path / "run.md", generated_at="fixed")
    content = md_path.read_text(encoding="utf-8")
    assert "gsm8k:main:test:0" in content
    assert "gsm8k:main:test:1" in content
    assert "gsm8k:main:test:2" in content
    assert "pass" in content.lower()


def test_two_renders_of_the_same_run_are_byte_identical_with_fixed_generated_at(
    tmp_path: Path,
) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    first = MarkdownReporter().write(run, tmp_path / "first.md", generated_at="fixed")
    second = MarkdownReporter().write(run, tmp_path / "second.md", generated_at="fixed")
    assert first.read_bytes() == second.read_bytes()
