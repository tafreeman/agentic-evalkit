"""Tests for the self-contained HTML reporter (design §11.3, plan Task 13)."""

import re
from pathlib import Path

from agentic_evalkit.models import EvalRunResult
from agentic_evalkit.reporters import HtmlReporter
from agentic_evalkit.stats import build_report_aggregates, wilson_interval

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


# --- Uncertainty section (ADR-0016): visible, aggregates-gated ---------------


def test_html_uncertainty_section_shows_bounds_in_visible_body(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    aggregates = build_report_aggregates(run)
    html_path = HtmlReporter().write(
        run, tmp_path / "run.html", aggregates=aggregates, generated_at="fixed"
    )
    content = html_path.read_text(encoding="utf-8")

    # Assert on the visible body -- everything before the embedded JSON data
    # island -- so this proves the bounds reach the rendered page, not merely
    # the hidden JSON blob (which the reporter has always carried).
    visible_body, _, _embedded = content.partition('<script id="embedded-run-data"')
    assert "Uncertainty" in visible_body
    assert "wilson" in visible_body  # the interval_method label
    lower, upper = wilson_interval(successes=1, total=3)
    assert f"{lower:.4f}" in visible_body
    assert f"{upper:.4f}" in visible_body


def test_html_without_aggregates_has_no_uncertainty_section(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    html_path = HtmlReporter().write(run, tmp_path / "run.html", generated_at="fixed")
    content = html_path.read_text(encoding="utf-8")
    assert "Uncertainty" not in content


def test_html_uncertainty_shows_cluster_robust_score_ci_for_repeated_attempts(
    tmp_path: Path, repeated_attempts_run: EvalRunResult
) -> None:
    run = repeated_attempts_run
    aggregates = build_report_aggregates(run)
    html_path = HtmlReporter().write(
        run, tmp_path / "run.html", aggregates=aggregates, generated_at="fixed"
    )
    content = html_path.read_text(encoding="utf-8")

    # Everything asserted here must appear in the visible body, not merely the
    # embedded JSON island (which has always carried the aggregates): the
    # repeated-attempt run's cluster_robust label, the score-mean chip, and the
    # exact score CI the aggregates payload carries (its statistical
    # correctness is proven in tests/unit/stats/test_aggregate.py).
    visible_body, _, _embedded = content.partition('<script id="embedded-run-data"')
    assert "Uncertainty" in visible_body
    assert "cluster_robust" in visible_body
    assert "Score mean" in visible_body

    estimate = aggregates["score_estimate"]
    assert isinstance(estimate, dict)
    assert isinstance(estimate["lower_bound"], float)
    assert isinstance(estimate["upper_bound"], float)
    assert f"[{estimate['lower_bound']:.4f}, {estimate['upper_bound']:.4f}]" in visible_body
