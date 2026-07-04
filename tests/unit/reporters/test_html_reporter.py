"""Tests for the self-contained HTML reporter (design §11.3, plan Task 13)."""

import re
from pathlib import Path

from agentic_evalkit.models import EvalRunResult
from agentic_evalkit.reporters import HtmlReporter

_URL_PATTERN = re.compile(r"""(?:src|href)\s*=\s*["']https?://""", re.IGNORECASE)


def test_html_is_one_self_contained_file(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    html_path = HtmlReporter().write(run, tmp_path / "run.html", generated_at="fixed")
    content = html_path.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "<style" in content.lower()  # embedded CSS, not a linked stylesheet
    assert not _URL_PATTERN.search(content)
    assert "http://" not in content
    assert "https://" not in content


def test_html_has_a_readable_summary_without_javascript(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    html_path = HtmlReporter().write(run, tmp_path / "run.html", generated_at="fixed")
    content = html_path.read_text(encoding="utf-8")
    assert "run-001" in content
    assert "gsm8k-smoke" in content
    assert "openai/gsm8k" in content


def test_html_escapes_script_tags_in_model_output(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    malicious_execution = run.samples[0].execution.model_copy(
        update={"output": {"answer": "<script>alert('xss')</script>"}}
    )
    malicious_sample = run.samples[0].model_copy(update={"execution": malicious_execution})
    run = run.model_copy(update={"samples": (malicious_sample, *run.samples[1:])})

    html_path = HtmlReporter().write(run, tmp_path / "run.html", generated_at="fixed")
    content = html_path.read_text(encoding="utf-8")

    # The visible sample table (everything before the embedded JSON data
    # island) must never contain a literal, executable <script> tag sourced
    # from model output.
    visible_body, _, embedded_and_after = content.partition('<script id="embedded-run-data"')
    assert "<script>alert" not in visible_body
    assert "&lt;script&gt;alert" in visible_body

    # The embedded JSON data island may contain the malicious string as
    # inert JSON text (it is not executed as markup), but its closing
    # "</script>" sequence must be neutralized so it cannot terminate the
    # surrounding <script> element early.
    assert "<script>alert" in embedded_and_after
    assert "</script>alert" not in embedded_and_after
    assert "<\\/script>" in embedded_and_after


def test_html_contains_embedded_json_data(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    html_path = HtmlReporter().write(run, tmp_path / "run.html", generated_at="fixed")
    content = html_path.read_text(encoding="utf-8")
    assert '"run_id"' in content
    assert '"run-001"' in content


def test_html_has_filter_buttons_for_outcome_categories(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    html_path = HtmlReporter().write(run, tmp_path / "run.html", generated_at="fixed")
    content = html_path.read_text(encoding="utf-8")
    assert "data-filter" in content or "filter" in content.lower()


def test_two_renders_of_the_same_run_are_byte_identical_with_fixed_generated_at(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    first = HtmlReporter().write(run, tmp_path / "first.html", generated_at="fixed")
    second = HtmlReporter().write(run, tmp_path / "second.html", generated_at="fixed")
    assert first.read_bytes() == second.read_bytes()
