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
    # This run has exactly 1 pass out of 3 samples total, so the report
    # should show that exact fraction, "1/3".
    assert "1/3" in content
    assert "3" in content  # total
    assert "1" in content  # errors
    # "error" and "timeout" are different kinds of failure and must each be
    # visible in the report on their own -- not collapsed into one generic
    # "failed" label.
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


# --- aggregates rendering (ADR-0016): a real Markdown table, not a raw dict printed as text ---


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
    # A real Markdown table header -- not the old behavior of just printing
    # the Python dict as text, e.g. ```{'pass_rate': ...}```.
    assert "| Metric | Value | 95% CI | Method |" in content
    assert "'pass_rate'" not in content
    assert "| Pass rate |" in content
    # The method used to compute the interval ("wilson") and the exact
    # lower/upper bounds of the pass-rate confidence interval both show up
    # in the rendered table.
    assert "wilson" in content
    lower, upper = wilson_interval(successes=1, total=3)
    assert f"{lower:.4f}" in content
    assert f"{upper:.4f}" in content


def test_markdown_aggregates_includes_pass_at_k_mean_when_present(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = pass_error_timeout_and_provenance_run
    aggregates = dict(build_report_aggregates(run))
    # The single-attempt fixture used here has no pass@k data (pass@k is the
    # chance that at least one of k attempts on the same sample succeeds --
    # it only makes sense once there's more than one attempt). We build a
    # small pass@k block by hand, with its mean computed the same way the
    # real stats module would, just so this test can check that the
    # Markdown table actually renders a pass@k row when one is present.
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
    aggregates = build_report_aggregates(run)  # one attempt each -> no pass_at_k data
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
    # If a run has no numeric scores at all, score_mean comes back as None.
    # In that case, the "Score mean" row must be left out of the table
    # entirely (never shown with a made-up value), even though the "Pass
    # rate" row still needs to render normally.
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

    # When samples have repeated attempts, the table visibly labels the
    # calculation "cluster_robust", and both confidence-interval cells show
    # the exact same lower/upper numbers that the aggregates data carries.
    # (Whether those numbers are statistically correct is checked separately,
    # in tests/unit/stats/test_aggregate.py -- this test only confirms they
    # render correctly into the table, including the "Score mean" row.)
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
    # If the "aggregates" data is nearly empty -- for example, a report file
    # that was hand-edited, or one produced by an older version of this tool
    # that didn't compute every field -- the table should show a "-"
    # placeholder in the missing cells instead of crashing or making up
    # numbers. This is the same convention the per-sample table already uses
    # for "no data here."
    run = pass_error_timeout_and_provenance_run
    md_path = MarkdownReporter().write(
        run, tmp_path / "run.md", aggregates={}, generated_at="fixed"
    )
    content = md_path.read_text(encoding="utf-8")
    assert "## Aggregates" in content
    assert "| Pass rate | -/- | - | - |" in content
