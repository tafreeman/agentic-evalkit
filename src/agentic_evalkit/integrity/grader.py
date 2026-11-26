"""An integrity grader: PASS for a clean run, UNAVAILABLE for a flagged one.

Wrap it with :func:`agentic_evalkit.integrity.gate.integrity_gated` rather than
adding it to a ``CompositeGrader`` by hand; the gate module explains the two
ways a hand-wired composite lets a flagged run pass.

A flagged run is ``UNAVAILABLE``, never ``FAIL``: the agent may have solved the
task, but the score cannot be trusted (ADR-0008, ADR-0027).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from pydantic import ValidationError

from agentic_evalkit.integrity.audit import InstanceInputs, audit_instance, coerce_calls
from agentic_evalkit.integrity.diff_guard import changed_paths
from agentic_evalkit.integrity.models import Severity
from agentic_evalkit.integrity.preflight import parse_eval_script
from agentic_evalkit.integrity.trajectory import AuditContext
from agentic_evalkit.models import (
    EvalSample,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

__all__ = ["InputsFn", "IntegrityGrader", "swebench_inputs"]

InputsFn = Callable[[EvalSample, NormalizedExecutionResult], InstanceInputs]

#: Keys under which a target reports its final patch in ``NormalizedExecutionResult.output``.
PATCH_KEYS = ("patch", "model_patch")


def _string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _patch_of(execution: NormalizedExecutionResult) -> str:
    output = execution.output
    if isinstance(output, dict):
        for key in PATCH_KEYS:
            value = output.get(key)
            if isinstance(value, str):
                return value
    return ""


def swebench_inputs(sample: EvalSample, execution: NormalizedExecutionResult) -> InstanceInputs:
    """Build audit inputs from a SWE-bench sample (the adapter's metadata layout).

    The graded files are the ones the instance's ``test_patch`` touches, and the
    patch is read from ``output["patch"]`` or ``output["model_patch"]``.
    ``solution_shas``, ``protected_paths``, ``expected_outputs`` and
    ``restored_paths`` are read from sample metadata when a dataset supplies
    them; an ``eval_script`` string in metadata is parsed for what it resets. A
    field a dataset does not supply is empty, which switches that rule off
    rather than guessing.
    """
    meta = sample.metadata
    instance_id = meta.get("instance_id")
    test_patch = sample.expected_artifacts.get("test_patch")
    repo = meta.get("repo")
    script = meta.get("eval_script")
    restored = _string_list(meta.get("restored_paths"))
    if not restored and isinstance(script, str):
        restored = parse_eval_script(script).restored_paths
    return InstanceInputs(
        context=AuditContext(
            instance_id=instance_id if isinstance(instance_id, str) else sample.sample_id,
            repo=repo if isinstance(repo, str) else None,
            solution_shas=_string_list(meta.get("solution_shas")),
            protected_paths=_string_list(meta.get("protected_paths")),
            expected_outputs=_string_list(meta.get("expected_outputs")),
        ),
        patch=_patch_of(execution),
        graded_paths=changed_paths(test_patch) if isinstance(test_patch, str) else (),
        restored_paths=restored,
    )


class IntegrityGrader:
    """Audits one execution's patch and tool calls."""

    def __init__(
        self, *, name: str = "integrity@1", inputs_for: InputsFn = swebench_inputs
    ) -> None:
        self._name = name
        self._inputs_for = inputs_for

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        now = datetime.now(UTC)
        try:
            calls = coerce_calls(execution.tool_calls, run_id="", sample_id=sample.sample_id)
            result = audit_instance(self._inputs_for(sample, execution), calls)
        except (ValidationError, ValueError) as error:
            return GradeResult(
                sample_id=sample.sample_id,
                grader=self._name,
                grader_type="integrity",
                status=GradeStatus.ERROR,
                evidence={"error": type(error).__name__},
                hard_gate=False,
                created_at=now,
            )
        flagged = result.severity is not Severity.CLEAN
        return GradeResult(
            sample_id=sample.sample_id,
            grader=self._name,
            grader_type="integrity",
            status=GradeStatus.UNAVAILABLE if flagged else GradeStatus.PASS,
            score=None if flagged else 1.0,
            hard_gate=True,
            evidence={
                "severity": result.severity.value,
                "flags": cast(
                    "list[JsonValue]",
                    [flag.model_dump(mode="json") for flag in result.flags],
                ),
            },
            created_at=now,
        )
