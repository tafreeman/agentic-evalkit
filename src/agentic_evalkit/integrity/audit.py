"""Combine the three stages into per-instance results and one report.

``raw_pass_rate`` counts every instance the oracle passed. ``clean_pass_rate``
counts only those that are also unflagged. Both divide by the same ``total``,
so a flagged run lowers ``clean_pass_rate`` instead of dropping out of the
denominator (the same rule ``aggregate_run`` follows).

The audit fails closed: anything it cannot read or match (a spilled patch, an
unparseable tool call, an orphan record, a context key that names no sample)
becomes a flag or an error, never a quiet skip, because a skip makes a run look
cleaner than it was.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import ValidationError

from agentic_evalkit import __version__
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.integrity.diff_guard import guard_patch
from agentic_evalkit.integrity.flags import make_flag, redact
from agentic_evalkit.integrity.models import (
    CheckStatus,
    InstanceIntegrity,
    IntegrityFlag,
    IntegrityReport,
    IntegrityStage,
    PreflightCheck,
    Severity,
    ToolCallRecord,
    worst_severity,
)
from agentic_evalkit.integrity.preflight import EvalScriptResets
from agentic_evalkit.integrity.trajectory import (
    RULES,
    AuditContext,
    audit_trajectory,
    scan_added_lines,
)
from agentic_evalkit.models import GradeStatus
from agentic_evalkit.models.base import FrozenModel

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from agentic_evalkit.models import (
        EvalRunResult,
        EvalSample,
        GradeResult,
        NormalizedExecutionResult,
        SampleResult,
    )

__all__ = [
    "DetectionMetrics",
    "InstanceInputs",
    "OutputLoader",
    "apply_override",
    "audit_instance",
    "audit_run",
    "build_report",
    "coerce_calls",
    "detection_metrics",
    "group_by_sample",
    "preflight_flags",
    "read_tool_calls",
]

#: Flag severities that keep an instance out of ``clean_pass_rate``.
NOT_CLEAN = frozenset({Severity.SUSPECT, Severity.EXPLOIT})

#: Reads the output an artifact store holds for a spilled execution, or ``None``.
OutputLoader = Callable[["NormalizedExecutionResult"], "dict[str, JsonValue] | None"]

#: Pre-flight checks whose failure means the agent could see or change the answer.
_EXPLOIT_CHECKS = frozenset({"git_history", "protected_paths"})

_CONTEXT_FIELDS = ("solution_shas", "protected_paths", "expected_outputs", "repo")
_INPUT_FIELDS = ("graded_paths", "restored_paths", "confine_to")
_SPILL_KEYS = ("output_ref", "output_spill_error")


class InstanceInputs(FrozenModel):
    """Everything the audit needs for one instance, as a ``--context`` file entry.

    ``context`` feeds the trajectory rules. ``graded_paths`` and
    ``restored_paths`` feed the diff guard. ``raw_passed`` is the oracle's
    verdict; the audit never recomputes it.
    """

    context: AuditContext
    patch: str = ""
    raw_passed: bool = False
    graded_paths: tuple[str, ...] = ()
    restored_paths: tuple[str, ...] = ()
    confine_to: tuple[str, ...] | None = None


def _instance_flag(
    instance_id: str, stage: IntegrityStage, rule: str, severity: Severity, evidence: str
) -> IntegrityFlag:
    return make_flag(
        instance_id=instance_id, stage=stage, rule=rule, severity=severity, evidence=evidence
    )


def preflight_flags(
    instance_id: str, checks: Iterable[PreflightCheck]
) -> tuple[IntegrityFlag, ...]:
    """A flag for every pre-flight check that did not pass.

    A failed history or mount check is ``exploit`` (the agent could see or
    change the answer); a failed egress check, or any check that could not run,
    is ``suspect``.
    """
    flags: list[IntegrityFlag] = []
    for check in checks:
        if check.status is CheckStatus.PASSED:
            continue
        exploit = check.status is CheckStatus.FAILED and check.name in _EXPLOIT_CHECKS
        flags.append(
            _instance_flag(
                instance_id,
                IntegrityStage.PREFLIGHT,
                "preflight_failed",
                Severity.EXPLOIT if exploit else Severity.SUSPECT,
                f"preflight {check.name} {check.status.value}: {check.detail}",
            )
        )
    return tuple(flags)


def _result(inputs: InstanceInputs, flags: Sequence[IntegrityFlag]) -> InstanceIntegrity:
    return InstanceIntegrity(
        instance_id=inputs.context.instance_id,
        severity=worst_severity([flag.severity for flag in flags]),
        raw_passed=inputs.raw_passed,
        flags=tuple(flags),
    )


def audit_instance(
    inputs: InstanceInputs,
    calls: Sequence[ToolCallRecord],
    *,
    extra_flags: Iterable[IntegrityFlag] = (),
) -> InstanceIntegrity:
    """Run the diff guard and trajectory audit for one instance.

    ``extra_flags`` carries flags the caller raised before auditing (a patch it
    could not load, tool calls it could not parse).
    """
    ctx = inputs.context
    flags: list[IntegrityFlag] = list(extra_flags)
    flags.extend(
        guard_patch(
            instance_id=ctx.instance_id,
            patch=inputs.patch,
            graded_paths=inputs.graded_paths,
            restored_paths=inputs.restored_paths,
            confine_to=inputs.confine_to,
        )
    )
    flags.extend(audit_trajectory(ctx, calls))
    flags.extend(scan_added_lines(ctx, inputs.patch))
    return _result(inputs, flags)


def build_report(
    *,
    run_id: str,
    instances: Iterable[InstanceIntegrity],
    preflight: Iterable[PreflightCheck] = (),
    now: datetime | None = None,
    context_digest: str | None = None,
    tool_calls_digest: str | None = None,
) -> IntegrityReport:
    """Assemble ``integrity_report.json`` from audited instances.

    Run-level ``preflight`` checks that did not pass flag every instance: if the
    agent could see the fix, no pass in the run can be trusted. Check details are
    redacted before they are stored.
    """
    checks = tuple(check.model_copy(update={"detail": redact(check.detail)}) for check in preflight)
    results = tuple(_with_preflight(item, checks) for item in instances)
    raw = sum(1 for item in results if item.raw_passed)
    clean = sum(1 for item in results if item.raw_passed and item.severity not in NOT_CLEAN)
    total = len(results)
    return IntegrityReport(
        run_id=run_id,
        created_at=now or datetime.now(UTC),
        total=total,
        raw_passed=raw,
        clean_passed=clean,
        raw_pass_rate=raw / total if total else None,
        clean_pass_rate=clean / total if total else None,
        harness_version=__version__,
        context_digest=context_digest,
        tool_calls_digest=tool_calls_digest,
        rules=RULES,
        preflight=checks,
        instances=results,
    )


def _with_preflight(item: InstanceIntegrity, checks: Sequence[PreflightCheck]) -> InstanceIntegrity:
    extra = preflight_flags(item.instance_id, checks)
    if not extra:
        return item
    flags = (*item.flags, *extra)
    return item.model_copy(
        update={"flags": flags, "severity": worst_severity([f.severity for f in flags])}
    )


def read_tool_calls(path: Path) -> tuple[ToolCallRecord, ...]:
    """Read ``tool_calls.jsonl``; a malformed line is an error, never skipped.

    Skipping a bad line would let a run look cleaner than it was. A file must
    describe one run: records with different ``run_id`` values are rejected.
    """
    records: list[ToolCallRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ManifestValidationError(message=f"cannot read tool calls: {path.name}") from error
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(ToolCallRecord.model_validate(json.loads(line)))
        except (ValueError, ValidationError, RecursionError) as error:
            raise ManifestValidationError(
                message=f"{path.name} line {number} is not a valid tool-call record"
            ) from error
    if len({record.run_id for record in records}) > 1:
        raise ManifestValidationError(message=f"{path.name} mixes records from different runs")
    return tuple(records)


def group_by_sample(calls: Iterable[ToolCallRecord]) -> dict[str, list[ToolCallRecord]]:
    """Group records by ``sample_id``, keeping call order."""
    grouped: dict[str, list[ToolCallRecord]] = defaultdict(list)
    for call in calls:
        grouped[call.sample_id].append(call)
    return dict(grouped)


_RECORD_FIELDS = frozenset(ToolCallRecord.model_fields)


def coerce_calls(
    raw_calls: Iterable[Mapping[str, JsonValue]], *, run_id: str, sample_id: str
) -> tuple[ToolCallRecord, ...]:
    """Turn the ``tool_calls`` a target put on an execution into records.

    Targets disagree on field names (``McpTarget`` records ``name``), so ``name``
    is read as ``tool`` and unknown keys are dropped. Identity fields a target
    did not stamp are filled in. Raises ``ValidationError`` for a call with no
    tool name.
    """
    records: list[ToolCallRecord] = []
    for index, raw in enumerate(raw_calls):
        data = {key: value for key, value in raw.items() if key in _RECORD_FIELDS}
        if "tool" not in data and isinstance(raw.get("name"), str):
            data["tool"] = raw["name"]
        records.append(
            ToolCallRecord.model_validate(
                {"run_id": run_id, "sample_id": sample_id, "call_id": f"call-{index}", **data}
            )
        )
    return tuple(records)


def apply_override(inputs: InstanceInputs, override: Mapping[str, JsonValue]) -> InstanceInputs:
    """Merge one ``integrity_context.json`` entry over inputs derived from the run.

    Only the documented keys are read, and the merged result is validated, so a
    typo or a wrong type (a SHA string where a list is needed) is an error and
    cannot silently switch a rule off.
    """
    unknown = set(override) - set(_CONTEXT_FIELDS) - set(_INPUT_FIELDS)
    if unknown:
        raise ManifestValidationError(
            message=f"unknown integrity context field(s): {', '.join(sorted(unknown))}"
        )
    try:
        context = {
            **inputs.context.model_dump(),
            **{key: override[key] for key in _CONTEXT_FIELDS if key in override},
        }
        merged = {
            **inputs.model_dump(),
            "context": context,
            **{key: override[key] for key in _INPUT_FIELDS if key in override},
        }
        return InstanceInputs.model_validate(merged)
    except ValidationError as error:
        raise ManifestValidationError(
            message=f"invalid integrity context for {inputs.context.instance_id}"
        ) from error


class DetectionMetrics(FrozenModel):
    """How well the flags match a labelled control set (precision and recall)."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float | None
    recall: float | None


def detection_metrics(report: IntegrityReport, *, cheating: Iterable[str]) -> DetectionMetrics:
    """Precision and recall of the report's flags against known-cheating instance IDs.

    An instance counts as flagged when its severity is not ``clean``.
    """
    known = set(cheating)
    tp = fp = fn = tn = 0
    for item in report.instances:
        flagged = item.severity is not Severity.CLEAN
        cheat = item.instance_id in known
        tp += flagged and cheat
        fp += flagged and not cheat
        fn += cheat and not flagged
        tn += not flagged and not cheat
    return DetectionMetrics(
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=tn,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
    )


def _harness_preflight(
    grade: GradeResult | None,
) -> tuple[EvalScriptResets | None, tuple[PreflightCheck, ...]]:
    """Pre-flight facts the SWE-bench executor stamped into the oracle's evidence."""
    if grade is None:
        return None, ()
    harness = grade.evidence.get("harness_evidence")
    stamped = harness.get("integrity_preflight") if isinstance(harness, dict) else None
    if not isinstance(stamped, dict):
        return None, ()
    if "error" in stamped:
        return None, (
            PreflightCheck(
                name="integrity_probe", status=CheckStatus.ERROR, detail="the probe did not run"
            ),
        )
    try:
        listed = stamped.get("checks")
        checks = tuple(
            PreflightCheck.model_validate(c) for c in (listed if isinstance(listed, list) else [])
        )
        script = stamped.get("eval_script")
        resets = EvalScriptResets.model_validate(script) if isinstance(script, dict) else None
    except ValidationError:
        return None, (
            PreflightCheck(
                name="integrity_probe", status=CheckStatus.ERROR, detail="unreadable probe output"
            ),
        )
    return resets, checks


def _resolve_execution(
    execution: NormalizedExecutionResult, loader: OutputLoader | None
) -> tuple[NormalizedExecutionResult, str | None]:
    """The execution with its spilled output restored, or a reason it cannot be."""
    if execution.output is not None or not any(key in execution.artifacts for key in _SPILL_KEYS):
        return execution, None
    loaded = loader(execution) if loader is not None else None
    if loaded is None:
        return execution, "output was spilled to the artifact store and could not be read"
    return execution.model_copy(update={"output": loaded}), None


def _calls_for(
    item: SampleResult, by_sample: Mapping[str, Sequence[ToolCallRecord]], run_id: str
) -> tuple[tuple[ToolCallRecord, ...], str | None]:
    recorded = by_sample.get(item.sample.sample_id)
    if recorded is not None:
        return tuple(recorded), None
    try:
        return (
            coerce_calls(item.execution.tool_calls, run_id=run_id, sample_id=item.sample.sample_id),
            None,
        )
    except ValidationError:
        return (), f"{len(item.execution.tool_calls)} tool call(s) could not be parsed"


def _audit_sample(
    item: SampleResult,
    *,
    run_id: str,
    inputs_for: Callable[[EvalSample, NormalizedExecutionResult], InstanceInputs],
    by_sample: Mapping[str, Sequence[ToolCallRecord]],
    override: Mapping[str, JsonValue] | None,
    loader: OutputLoader | None,
) -> InstanceIntegrity:
    execution, patch_problem = _resolve_execution(item.execution, loader)
    inputs = inputs_for(item.sample, execution)
    if override is not None:
        inputs = apply_override(inputs, override)
    resets, checks = _harness_preflight(item.grade)
    passed = item.grade is not None and item.grade.status is GradeStatus.PASS
    restored = tuple(
        dict.fromkeys((*inputs.restored_paths, *(resets.restored_paths if resets else ())))
    )
    inputs = inputs.model_copy(update={"raw_passed": passed, "restored_paths": restored})
    calls, calls_problem = _calls_for(item, by_sample, run_id)
    instance_id = inputs.context.instance_id
    extra = list(preflight_flags(instance_id, checks))
    for rule, problem, stage in (
        ("patch_unavailable", patch_problem, IntegrityStage.DIFF_GUARD),
        ("tool_calls_unparseable", calls_problem, IntegrityStage.TRAJECTORY),
    ):
        if problem is not None:
            extra.append(_instance_flag(instance_id, stage, rule, Severity.SUSPECT, problem))
    return audit_instance(inputs, calls, extra_flags=extra)


def _check_orphans(
    result: EvalRunResult,
    by_sample: Mapping[str, Sequence[ToolCallRecord]],
    overrides: Mapping[str, Mapping[str, JsonValue]] | None,
) -> None:
    known = {item.sample.sample_id for item in result.samples}
    orphans = sorted(set(by_sample) - known)
    if orphans:
        raise ManifestValidationError(
            message=f"tool calls name {len(orphans)} sample(s) that are not in this run"
        )
    unmatched = sorted(set(overrides or {}) - known)
    if unmatched:
        raise ManifestValidationError(
            message=f"integrity context names {len(unmatched)} sample(s) that are not in this run"
        )


def audit_run(
    result: EvalRunResult,
    *,
    inputs_for: Callable[[EvalSample, NormalizedExecutionResult], InstanceInputs],
    calls: Iterable[ToolCallRecord] = (),
    overrides: Mapping[str, Mapping[str, JsonValue]] | None = None,
    preflight: Iterable[PreflightCheck] = (),
    output_loader: OutputLoader | None = None,
    context_digest: str | None = None,
    tool_calls_digest: str | None = None,
) -> IntegrityReport:
    """Audit every sample of a finished run and return its integrity report.

    Tool calls come from ``calls`` (the run's ``tool_calls.jsonl``) when it has
    records for a sample, and otherwise from the sample's own
    ``execution.tool_calls``. ``overrides`` maps ``sample_id`` to extra context.
    ``output_loader`` reads an output the runner spilled to the artifact store;
    without it a spilled patch is flagged ``patch_unavailable`` instead of being
    audited as empty. Records or context keys that name no sample are an error.
    """
    by_sample = group_by_sample(calls)
    _check_orphans(result, by_sample, overrides)
    audited = [
        _audit_sample(
            item,
            run_id=result.run_id,
            inputs_for=inputs_for,
            by_sample=by_sample,
            override=(overrides or {}).get(item.sample.sample_id),
            loader=output_loader,
        )
        for item in result.samples
    ]
    return build_report(
        run_id=result.run_id,
        instances=audited,
        preflight=preflight,
        context_digest=context_digest,
        tool_calls_digest=tool_calls_digest,
    )
