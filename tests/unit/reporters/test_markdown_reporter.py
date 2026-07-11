"""Tests for the Markdown reporter (design §11.3, plan Task 13)."""

from pathlib import Path

from agentic_evalkit.models import EvalRunResult
from agentic_evalkit.reporters import MarkdownReporter
from agentic_evalkit.stats import build_report_aggregates, pass_at_k, wilson_interval


def test_markdown_contains_identity_and_provenance(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
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


def test_markdown_contains_exact_numerator_and_denominator(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    md_path = MarkdownReporter().write(run, tmp_path / "run.md", generated_at="fixed")
    content = md_path.read_text(encoding="utf-8")
    # 1 passed out of 3 total is the exact numerator/denominator this run produces.
    assert "1/3" in content
    assert "3" in content  # total
    assert "1" in content  # errors
    # Distinct outcome categories must be visible, not merged.
    assert "error" in content.lower()
    assert "timeout" in content.lower()


def test_markdown_contains_compatibility_details(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    md_path = MarkdownReporter().write(run, tmp_path / "run.md", generated_at="fixed")
    content = md_path.read_text(encoding="utf-8")
    assert "env:sha256:deadbeef" in content
    assert "code:sha256:cafef00d" in content


def test_markdown_contains_sample_table_with_evidence(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    md_path = MarkdownReporter().write(run, tmp_path / "run.md", generated_at="fixed")
    content = md_path.read_text(encoding="utf-8")
    assert "gsm8k:main:test:0" in content
    assert "gsm8k:main:test:1" in content
    assert "gsm8k:main:test:2" in content
    assert "pass" in content.lower()


def test_two_renders_of_the_same_run_are_byte_identical_with_fixed_generated_at(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    first = MarkdownReporter().write(run, tmp_path / "first.md", generated_at="fixed")
    second = MarkdownReporter().write(run, tmp_path / "second.md", generated_at="fixed")
    assert first.read_bytes() == second.read_bytes()


# --- aggregates rendering (ADR-0016): a real table, not str(dict) ------------


def test_markdown_renders_aggregates_as_table_not_dict_repr(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    aggregates = build_report_aggregates(run)
    md_path = MarkdownReporter().write(
        run, tmp_path / "run.md", aggregates=aggregates, generated_at="fixed"
    )
    content = md_path.read_text(encoding="utf-8")

    assert "## Aggregates" in content
    # A real Markdown table header, not the old ```{'pass_rate': ...}``` repr.
    assert "| Metric | Value | 95% CI | Method |" in content
    assert "'pass_rate'" not in content
    assert "| Pass rate |" in content
    # The interval-method label and the exact pass-rate bounds are visible.
    assert "wilson" in content
    lower, upper = wilson_interval(successes=1, total=3)
    assert f"{lower:.4f}" in content
    assert f"{upper:.4f}" in content


def test_markdown_aggregates_includes_pass_at_k_mean_when_present(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    aggregates = dict(build_report_aggregates(run))
    # The single-attempt fixture has no pass@k; inject a repeated-attempt block
    # with an independently computed mean so the row's rendering is exercised.
    estimate = pass_at_k(total_attempts=2, successful_attempts=1, k=2)
    aggregates["pass_at_k"] = {
        "k": 2,
        "mean": estimate,
        "by_sample_id": {"gsm8k:main:test:0": estimate},
    }
    md_path = MarkdownReporter().write(
        run, tmp_path / "run.md", aggregates=aggregates, generated_at="fixed"
    )
    content = md_path.read_text(encoding="utf-8")
    assert "pass@2" in content
    assert f"{estimate:.4f}" in content


def test_markdown_aggregates_omits_pass_at_k_row_when_absent(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    aggregates = build_report_aggregates(run)  # single attempt -> no pass_at_k
    md_path = MarkdownReporter().write(
        run, tmp_path / "run.md", aggregates=aggregates, generated_at="fixed"
    )
    content = md_path.read_text(encoding="utf-8")
    assert "pass@" not in content


def test_markdown_without_aggregates_has_no_aggregates_section(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    md_path = MarkdownReporter().write(run, tmp_path / "run.md", generated_at="fixed")
    content = md_path.read_text(encoding="utf-8")
    assert "## Aggregates" not in content


def test_markdown_aggregates_omits_score_row_when_score_mean_is_none(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    aggregates = dict(build_report_aggregates(run))
    # A run with no defined scores has score_mean=None: the score-mean row must
    # be omitted (not fabricated) while the pass-rate row still renders.
    aggregates["score_mean"] = None
    aggregates["score_estimate"] = None
    md_path = MarkdownReporter().write(
        run, tmp_path / "run.md", aggregates=aggregates, generated_at="fixed"
    )
    content = md_path.read_text(encoding="utf-8")
    assert "| Pass rate |" in content
    assert "Score mean" not in content


def test_markdown_renders_cluster_robust_aggregates_for_repeated_attempts(
    tmp_path: Path, repeated_attempts_run: EvalRunResult
) -> None:
    run = repeated_attempts_run
    aggregates = build_report_aggregates(run)
    md_path = MarkdownReporter().write(
        run, tmp_path / "run.md", aggregates=aggregates, generated_at="fixed"
    )
    content = md_path.read_text(encoding="utf-8")

    # The repeated-attempt regime is visibly labeled cluster_robust, and both
    # CI cells carry the exact bounds the aggregates payload does (the bounds'
    # statistical correctness is proven in tests/unit/stats/test_aggregate.py;
    # this pins rendering fidelity for a populated score_estimate row).
    assert "cluster_robust" in content
    pass_rate = aggregates["pass_rate"]
    assert isinstance(pass_rate, dict)
    assert pass_rate["interval_method"] == "cluster_robust"
    assert isinstance(pass_rate["lower_bound"], float)
    assert isinstance(pass_rate["upper_bound"], float)
    assert f"[{pass_rate['lower_bound']:.4f}, {pass_rate['upper_bound']:.4f}]" in content

    estimate = aggregates["score_estimate"]
    assert isinstance(estimate, dict)
    assert isinstance(estimate["lower_bound"], float)
    assert isinstance(estimate["upper_bound"], float)
    assert "| Score mean |" in content
    assert f"[{estimate['lower_bound']:.4f}, {estimate['upper_bound']:.4f}]" in content


def test_markdown_renders_placeholder_cells_for_minimal_aggregates(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    # A minimal aggregates mapping (as a hand-edited or older-tool report file
    # might carry) renders "-" placeholder cells rather than crashing or
    # fabricating numbers -- the same "-"-for-absent idiom the sample table uses.
    run = pass_error_timeout_and_provenance_run
    md_path = MarkdownReporter().write(
        run, tmp_path / "run.md", aggregates={}, generated_at="fixed"
    )
    content = md_path.read_text(encoding="utf-8")
    assert "## Aggregates" in content
    assert "| Pass rate | -/- | - | - |" in content
