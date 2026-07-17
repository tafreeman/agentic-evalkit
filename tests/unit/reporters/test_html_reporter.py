"""Tests for the HTML reporter, which produces one self-contained file --
all CSS and data embedded directly in the page, with no links out to other
files or the network (design doc §11.3, plan Task 13).
"""

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

    # The part of the page a human actually sees (everything before the
    # embedded JSON blob at the end of the file) must never contain a real,
    # runnable <script> tag built from the model's output text.
    visible_body, _, embedded_and_after = content.partition('<script id="embedded-run-data"')
    assert "<script>alert" not in visible_body
    assert "&lt;script&gt;alert" in visible_body

    # The embedded JSON blob is allowed to contain the malicious string as
    # plain JSON text -- browsers don't run JSON as code, so it's harmless
    # there. But its "</script>" sequence must still be altered so that it
    # can't accidentally close the surrounding <script> tag early and break
    # out into the rest of the page.
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


# --- Uncertainty section (ADR-0016): shown on the page, only when aggregates are given ---


def test_html_uncertainty_section_shows_bounds_in_visible_body(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    aggregates = build_report_aggregates(run)
    html_path = HtmlReporter().write(
        run, tmp_path / "run.html", aggregates=aggregates, generated_at="fixed"
    )
    content = html_path.read_text(encoding="utf-8")

    # Check the visible part of the page -- everything before the embedded
    # JSON blob -- so this proves the numbers actually show up on the
    # rendered page a person would look at, not just inside the hidden JSON
    # blob (which the reporter has always included).
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

    # Everything checked below must show up in the visible part of the page,
    # not just inside the hidden embedded JSON blob (which has always carried
    # this data): the repeated-attempt run's "cluster_robust" label, the
    # score-mean display, and the exact confidence interval (a range that's
    # likely to contain the true score) for the score. That interval's math is
    # verified separately in tests/unit/stats/test_aggregate.py -- this test
    # only checks that it's rendered correctly onto the page.
    visible_body, _, _embedded = content.partition('<script id="embedded-run-data"')
    assert "Uncertainty" in visible_body
    assert "cluster_robust" in visible_body
    assert "Score mean" in visible_body

    estimate = aggregates["score_estimate"]
    assert isinstance(estimate, dict)
    assert isinstance(estimate["lower_bound"], float)
    assert isinstance(estimate["upper_bound"], float)
    assert f"[{estimate['lower_bound']:.4f}, {estimate['upper_bound']:.4f}]" in visible_body
