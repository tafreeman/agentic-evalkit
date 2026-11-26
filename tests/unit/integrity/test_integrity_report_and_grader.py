"""Report arithmetic (spec lemmas L2, L3), the integrity grader, and its use behind the gate."""

from __future__ import annotations

import itertools
import json
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import pytest
from _integrity_fixtures import (
    SOLUTION_SHA,
    TEST_EDIT_PATCH,
    run_result,
    sample_result,
)
from pydantic import ValidationError

from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.integrity import (
    InstanceIntegrity,
    IntegrityFlag,
    IntegrityGrader,
    IntegrityReport,
    IntegrityStage,
    Severity,
    ToolCallRecord,
    audit_run,
    build_report,
    integrity_gated,
    read_tool_calls,
    swebench_inputs,
)
from agentic_evalkit.integrity.models import severity_rank, worst_severity
from agentic_evalkit.models import GradeResult, GradeStatus

if TYPE_CHECKING:
    from agentic_evalkit.graders.composite import CompositeGrader


def _flag(severity: Severity) -> IntegrityFlag:
    return IntegrityFlag(
        instance_id="i", stage=IntegrityStage.TRAJECTORY, rule="r", severity=severity, evidence="e"
    )


def _instance(name: str, *, passed: bool, severity: Severity) -> InstanceIntegrity:
    flags = () if severity is Severity.CLEAN else (_flag(severity),)
    return InstanceIntegrity(instance_id=name, severity=severity, raw_passed=passed, flags=flags)


# --- models -----------------------------------------------------------------


def test_a_clean_flag_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _flag(Severity.CLEAN)


def test_instance_severity_must_match_its_worst_flag() -> None:
    with pytest.raises(ValidationError):
        InstanceIntegrity(
            instance_id="i", severity=Severity.CLEAN, flags=(_flag(Severity.EXPLOIT),)
        )


def test_models_are_frozen_and_versioned() -> None:
    flag = _flag(Severity.SUSPECT)
    assert flag.schema_version == "1"
    with pytest.raises(ValidationError):
        flag.rule = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        IntegrityFlag.model_validate({**flag.model_dump(), "unexpected": 1})


def test_severity_order_is_clean_suspect_exploit() -> None:
    order = [Severity.CLEAN, Severity.SUSPECT, Severity.EXPLOIT]
    assert [severity_rank(s) for s in order] == [0, 1, 2]
    assert worst_severity(()) is Severity.CLEAN


def test_adding_a_flag_never_lowers_severity() -> None:
    """Lemma L3 over every ordered pair of severities."""
    for first, second in itertools.product(Severity, repeat=2):
        flags = [s for s in (first, second) if s is not Severity.CLEAN]
        base = worst_severity([s for s in (first,) if s is not Severity.CLEAN])
        assert severity_rank(worst_severity(flags)) >= severity_rank(base)


# --- report arithmetic ------------------------------------------------------


def test_clean_rate_never_exceeds_raw_rate_and_denominator_is_fixed() -> None:
    """Lemma L2 across every pass/severity combination of three instances."""
    options = [(passed, severity) for passed in (True, False) for severity in Severity]
    for combo in itertools.product(options, repeat=3):
        report = build_report(
            run_id="r",
            instances=[
                _instance(f"i{n}", passed=passed, severity=severity)
                for n, (passed, severity) in enumerate(combo)
            ],
        )
        assert report.total == 3
        assert report.clean_passed <= report.raw_passed <= report.total
        assert report.clean_pass_rate is not None
        assert report.raw_pass_rate is not None
        assert report.clean_pass_rate <= report.raw_pass_rate


def test_flagging_moves_a_pass_out_of_clean_without_changing_the_denominator() -> None:
    clean = build_report(
        run_id="r",
        instances=[_instance("a", passed=True, severity=Severity.CLEAN)] * 2,
    )
    flagged = build_report(
        run_id="r",
        instances=[
            _instance("a", passed=True, severity=Severity.CLEAN),
            _instance("b", passed=True, severity=Severity.EXPLOIT),
        ],
    )
    assert (clean.raw_pass_rate, clean.clean_pass_rate) == (1.0, 1.0)
    assert (flagged.raw_pass_rate, flagged.clean_pass_rate) == (1.0, 0.5)
    assert flagged.total == clean.total == 2


def test_suspect_also_leaves_the_clean_rate() -> None:
    report = build_report(
        run_id="r", instances=[_instance("a", passed=True, severity=Severity.SUSPECT)]
    )
    assert (report.raw_passed, report.clean_passed) == (1, 0)


def test_an_empty_run_has_no_rates() -> None:
    report = build_report(run_id="r", instances=[])
    assert (report.total, report.raw_pass_rate, report.clean_pass_rate) == (0, None, None)


def test_report_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValidationError):
        IntegrityReport(
            run_id="r",
            created_at=run_result(()).started_at,
            total=1,
            raw_passed=1,
            clean_passed=2,
        )


def test_report_round_trips_through_json() -> None:
    report = build_report(
        run_id="r", instances=[_instance("a", passed=True, severity=Severity.EXPLOIT)]
    )
    assert IntegrityReport.model_validate_json(report.model_dump_json()) == report


# --- audit_run --------------------------------------------------------------


def _tool_calls(*commands: str) -> tuple[dict[str, object], ...]:
    return tuple({"tool": "shell_run", "arguments": {"command": c}} for c in commands)


def test_audit_run_uses_the_samples_own_tool_calls() -> None:
    result = run_result(
        (
            sample_result("a", tool_calls=_tool_calls("pytest -q")),
            sample_result("b", tool_calls=_tool_calls(f"git show {SOLUTION_SHA}")),
        )
    )
    report = audit_run(
        result,
        inputs_for=swebench_inputs,
        overrides={"b": {"solution_shas": [SOLUTION_SHA]}},
    )
    severities = {item.instance_id: item.severity for item in report.instances}
    assert severities == {"a": Severity.CLEAN, "b": Severity.EXPLOIT}
    assert (report.raw_pass_rate, report.clean_pass_rate) == (1.0, 0.5)


def test_audit_run_marks_failed_samples_as_not_passed() -> None:
    result = run_result((sample_result("a", passed=False),))
    report = audit_run(result, inputs_for=swebench_inputs)
    assert (report.raw_passed, report.clean_passed) == (0, 0)


def test_audit_run_prefers_the_jsonl_records_for_a_sample() -> None:
    result = run_result((sample_result("a", tool_calls=_tool_calls("pytest -q")),))
    from_file = [
        ToolCallRecord(
            run_id="r",
            sample_id="a",
            call_id="1",
            tool="shell_run",
            arguments={"command": f"git show {SOLUTION_SHA}"},
        )
    ]
    report = audit_run(
        result,
        inputs_for=swebench_inputs,
        calls=from_file,
        overrides={"a": {"solution_shas": [SOLUTION_SHA]}},
    )
    assert report.instances[0].severity is Severity.EXPLOIT


def test_audit_run_flags_a_test_edit_from_the_test_patch_files() -> None:
    result = run_result((sample_result("a", patch=TEST_EDIT_PATCH),))
    report = audit_run(result, inputs_for=swebench_inputs)
    assert [f.rule for f in report.instances[0].flags] == ["graded_file_edited"]


def test_audit_run_honours_restored_paths_override() -> None:
    result = run_result((sample_result("a", patch=TEST_EDIT_PATCH),))
    report = audit_run(
        result,
        inputs_for=swebench_inputs,
        overrides={"a": {"restored_paths": ["tests/validators/tests.py"]}},
    )
    assert report.instances[0].severity is Severity.CLEAN


def test_unknown_override_key_is_an_error_not_a_silent_skip() -> None:
    result = run_result((sample_result("a"),))
    with pytest.raises(ManifestValidationError, match="unknown integrity context"):
        audit_run(result, inputs_for=swebench_inputs, overrides={"a": {"solution_sha": []}})


def test_read_tool_calls_rejects_a_malformed_line(tmp_path: Path) -> None:
    good = ToolCallRecord(run_id="r", sample_id="a", call_id="1", tool="t").model_dump_json()
    path = tmp_path / "tool_calls.jsonl"
    path.write_text(f"{good}\n\n{{not json\n", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="line 3"):
        read_tool_calls(path)


def test_read_tool_calls_reads_records_and_skips_blank_lines(tmp_path: Path) -> None:
    lines = [
        ToolCallRecord(run_id="r", sample_id="a", call_id=str(n), tool="t").model_dump_json()
        for n in range(2)
    ]
    path = tmp_path / "tool_calls.jsonl"
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    assert [r.call_id for r in read_tool_calls(path)] == ["0", "1"]


def test_read_tool_calls_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError):
        read_tool_calls(tmp_path / "absent.jsonl")


# --- IntegrityGrader behind the gate ---------------------------------------


class _Oracle:
    def __init__(self, status: GradeStatus) -> None:
        self.status = status

    async def grade(self, sample, execution) -> GradeResult:  # type: ignore[no-untyped-def]
        return GradeResult(
            sample_id=sample.sample_id,
            grader="oracle",
            status=self.status,
            score=1.0 if self.status is GradeStatus.PASS else 0.0,
            created_at=execution.finished_at,
        )


def _gated(oracle: GradeStatus = GradeStatus.PASS) -> CompositeGrader:
    return integrity_gated(oracle=_Oracle(oracle), integrity=IntegrityGrader(), name="gated")


@pytest.mark.asyncio
async def test_clean_run_keeps_the_oracle_verdict() -> None:
    item = sample_result("a", tool_calls=_tool_calls("git log --oneline -3"))
    result = await _gated().grade(item.sample, item.execution)
    assert result.status is GradeStatus.PASS


@pytest.mark.asyncio
async def test_exploit_with_an_oracle_pass_is_unavailable_not_pass_or_fail() -> None:
    item = sample_result("a", patch=TEST_EDIT_PATCH)
    result = await _gated().grade(item.sample, item.execution)
    assert result.status is GradeStatus.UNAVAILABLE
    flags = result.evidence["children"][0]["evidence"]["flags"]  # type: ignore[index]
    assert flags[0]["rule"] == "graded_file_edited"


@pytest.mark.asyncio
async def test_a_flagged_run_that_the_oracle_failed_is_still_not_pass() -> None:
    item = sample_result("a", patch=TEST_EDIT_PATCH)
    result = await _gated(GradeStatus.FAIL).grade(item.sample, item.execution)
    assert result.status is not GradeStatus.PASS


@pytest.mark.asyncio
async def test_unparseable_tool_calls_are_an_error_not_a_silent_pass() -> None:
    item = sample_result("a", tool_calls=({"arguments": {"command": "ls"}},))  # no "tool"
    result = await IntegrityGrader().grade(item.sample, item.execution)
    assert result.status is GradeStatus.ERROR
    composite = await _gated().grade(item.sample, item.execution)
    assert composite.status is not GradeStatus.PASS


@pytest.mark.asyncio
async def test_integrity_grader_output_is_json_serialisable() -> None:
    item = sample_result("a", patch=TEST_EDIT_PATCH)
    result = await IntegrityGrader().grade(item.sample, item.execution)
    json.dumps(result.model_dump(mode="json"))
    assert result.hard_gate is True
