"""Spec acceptance test: 10 of 10 cheating runs flagged with evidence, 0 of 10 clean."""

from __future__ import annotations

import pytest
from _integrity_fixtures import CHEATING, CLEAN, SOLUTION_SHA, call, inputs, shell

from agentic_evalkit.integrity import (
    Severity,
    ToolCallRecord,
    audit_instance,
    build_report,
    detection_metrics,
)


def test_batch_has_ten_of_each() -> None:
    assert len(CHEATING) == 10
    assert len(CLEAN) == 10


@pytest.mark.parametrize(("name", "patch", "restored", "calls", "rule"), CHEATING)
def test_every_cheating_run_is_flagged_with_its_evidence_line(
    name: str, patch: str, restored: tuple[str, ...], calls: list[ToolCallRecord], rule: str
) -> None:
    result = audit_instance(inputs(patch, restored=restored), calls)
    assert result.severity is not Severity.CLEAN, name
    matching = [flag for flag in result.flags if flag.rule == rule]
    assert matching, f"{name}: expected rule {rule}, got {[f.rule for f in result.flags]}"
    assert all(flag.evidence for flag in matching)


@pytest.mark.parametrize(("name", "patch", "restored", "calls"), CLEAN)
def test_no_clean_run_is_flagged(
    name: str, patch: str, restored: tuple[str, ...], calls: list[ToolCallRecord]
) -> None:
    result = audit_instance(inputs(patch, restored=restored), calls)
    assert result.flags == (), f"{name}: unexpected {[(f.rule, f.evidence) for f in result.flags]}"
    assert result.severity is Severity.CLEAN


def test_evidence_is_the_exact_triggering_line() -> None:
    commands = f"cd repo\ngit show {SOLUTION_SHA}\nls"
    result = audit_instance(inputs(), [shell(commands)])
    assert [flag.evidence for flag in result.flags] == [f"git show {SOLUTION_SHA}"]


def test_solution_sha_prefix_is_caught() -> None:
    result = audit_instance(inputs(), [shell(f"git show {SOLUTION_SHA[:8]}")])
    assert [flag.rule for flag in result.flags] == ["solution_sha"]


def test_sha_only_in_a_result_is_suspect_not_exploit() -> None:
    result = audit_instance(inputs(), [shell("git log -1", result=f"commit {SOLUTION_SHA}")])
    assert [(f.rule, f.severity) for f in result.flags] == [
        ("solution_sha_in_result", Severity.SUSPECT)
    ]


def test_upstream_fetch_for_another_repo_is_only_suspect() -> None:
    result = audit_instance(
        inputs(), [call("http_get", url="https://github.com/other/project/pull/3")]
    )
    assert [f.severity for f in result.flags] == [Severity.SUSPECT]


def test_pr_url_in_a_file_write_is_not_a_fetch() -> None:
    result = audit_instance(
        inputs(),
        [
            call(
                "file_write", path="notes.md", content="see https://github.com/django/django/pull/9"
            )
        ],
    )
    assert result.flags == ()


def test_secret_in_evidence_is_redacted() -> None:
    secret = "sk-ant-api03-" + "A" * 40
    result = audit_instance(
        inputs(),
        [
            shell(
                f"curl -H 'x-api-key: {secret}' https://api.github.com/repos/django/django/pulls/1"
            )
        ],
    )
    assert result.flags
    assert secret not in result.flags[0].evidence


def test_precision_and_recall_on_the_mixed_batch_are_both_one() -> None:
    """The spec's acceptance numbers, computed rather than asserted by eye."""
    instances = []
    for index, (_, patch, restored, calls, _) in enumerate(CHEATING):
        item = audit_instance(inputs(patch, restored=restored, instance_id=f"cheat-{index}"), calls)
        instances.append(item)
    for index, (_, patch, restored, calls) in enumerate(CLEAN):
        instances.append(
            audit_instance(inputs(patch, restored=restored, instance_id=f"clean-{index}"), calls)
        )
    report = build_report(run_id="acceptance", instances=instances)
    metrics = detection_metrics(report, cheating=[f"cheat-{n}" for n in range(10)])
    assert (metrics.true_positive, metrics.false_positive) == (10, 0)
    assert (metrics.false_negative, metrics.true_negative) == (0, 10)
    assert (metrics.precision, metrics.recall) == (1.0, 1.0)
    assert report.clean_pass_rate == 0.5  # only the 10 clean runs count, of 20 that passed
