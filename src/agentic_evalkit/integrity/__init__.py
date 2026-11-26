"""Integrity harness: pure detectors and the gate that stops a flagged run passing (ADR-0027)."""

from agentic_evalkit.integrity.audit import (
    DetectionMetrics,
    InstanceInputs,
    audit_instance,
    audit_run,
    build_report,
    coerce_calls,
    detection_metrics,
    group_by_sample,
    read_tool_calls,
)
from agentic_evalkit.integrity.gate import integrity_gated
from agentic_evalkit.integrity.grader import IntegrityGrader, swebench_inputs
from agentic_evalkit.integrity.models import (
    CheckStatus,
    InstanceIntegrity,
    IntegrityFlag,
    IntegrityReport,
    IntegrityStage,
    PreflightCheck,
    Severity,
    ToolCallRecord,
)
from agentic_evalkit.integrity.trajectory import AuditContext

__all__ = [
    "AuditContext",
    "CheckStatus",
    "DetectionMetrics",
    "InstanceInputs",
    "InstanceIntegrity",
    "IntegrityFlag",
    "IntegrityGrader",
    "IntegrityReport",
    "IntegrityStage",
    "PreflightCheck",
    "Severity",
    "ToolCallRecord",
    "audit_instance",
    "audit_run",
    "build_report",
    "coerce_calls",
    "detection_metrics",
    "group_by_sample",
    "integrity_gated",
    "read_tool_calls",
    "swebench_inputs",
]
