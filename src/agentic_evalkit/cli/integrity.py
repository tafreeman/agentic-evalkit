"""``agentic-evalkit integrity <run_dir>``: audit a finished run, write ``integrity_report.json``.

The command reads the run's canonical report (and ``tool_calls.jsonl`` when
the runner wrote one), runs the diff guard and trajectory audit, and writes
the report next to the run. It never calls a target and never changes the
run's own grades; a flagged instance is reported, not re-graded.

Inputs are never trusted to be complete: a named ``--context`` that does not
exist is an error, a spilled patch is read back from the run's artifact store
(and flagged ``patch_unavailable`` if it cannot be), and the digests of the
context and tool-call files are written into the report.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path  # noqa: TC003 -- Typer resolves annotations at runtime
from typing import TYPE_CHECKING, Annotated

import typer
from pydantic import ValidationError

from agentic_evalkit.artifacts import ArtifactRef, ArtifactStore
from agentic_evalkit.cli.app import app, console, run_cli_command, safe_text
from agentic_evalkit.cli.reports import load_run_result
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.experiment import atomic_json, file_digest
from agentic_evalkit.integrity import (
    IntegrityReport,
    PreflightCheck,
    Severity,
    audit_run,
    read_tool_calls,
    swebench_inputs,
)

if TYPE_CHECKING:
    from pydantic import JsonValue

    from agentic_evalkit.integrity.audit import OutputLoader
    from agentic_evalkit.models import NormalizedExecutionResult

__all__ = ["REPORT_NAME", "find_run_report", "run_integrity_audit", "summarize"]

REPORT_NAME = "integrity_report.json"
TOOL_CALLS_NAME = "tool_calls.jsonl"
CONTEXT_NAME = "integrity_context.json"
ARTIFACTS_DIR = "artifacts"
_NOT_RUN_REPORTS = frozenset({REPORT_NAME, CONTEXT_NAME, "state.json", "helm-result.json"})
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def find_run_report(run_dir: Path) -> Path:
    """The one canonical run report in ``run_dir``; zero or several is an error."""
    candidates = sorted(
        path for path in run_dir.glob("*.json") if path.name not in _NOT_RUN_REPORTS
    )
    if len(candidates) != 1:
        raise ManifestValidationError(
            message=f"expected exactly one run report in {run_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _load_context(
    path: Path | None,
) -> tuple[dict[str, dict[str, JsonValue]], tuple[PreflightCheck, ...]]:
    """Read ``integrity_context.json``: per-sample ``instances`` plus ``preflight`` checks."""
    if path is None:
        return {}, ()
    invalid = ManifestValidationError(message=f"invalid integrity context: {path.name}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        instances = document.get("instances") or {}
        checks = tuple(
            PreflightCheck.model_validate(item) for item in document.get("preflight") or []
        )
    except (
        OSError,
        ValueError,
        AttributeError,
        TypeError,
        ValidationError,
        RecursionError,
    ) as error:
        raise invalid from error
    if not isinstance(instances, dict) or not all(isinstance(v, dict) for v in instances.values()):
        raise invalid
    return instances, checks


def _artifact_loader(root: Path) -> OutputLoader | None:
    """Read a spilled output back from the run's artifact store.

    The runner stores ``str(output)`` (a dict literal), so it is read with
    ``ast.literal_eval``, which evaluates literals only. The digest is validated
    before it becomes a filename and the bytes are checked against it.
    """
    if not root.is_dir():
        return None
    store = ArtifactStore(root)

    def load(execution: NormalizedExecutionResult) -> dict[str, JsonValue] | None:
        digest = execution.artifacts.get("output_ref")
        if not isinstance(digest, str) or not _DIGEST.match(digest):
            return None
        try:
            data = store.read(
                ArtifactRef(digest=digest, media_type="application/json", byte_count=0)
            )
            if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
                return None
            value = ast.literal_eval(data.decode("utf-8"))
        except (OSError, ValueError, SyntaxError, MemoryError, RecursionError):
            return None
        return value if isinstance(value, dict) else None

    return load


def run_integrity_audit(
    report_path: Path,
    output_dir: Path,
    *,
    context_path: Path | None = None,
    default_context: bool = True,
) -> IntegrityReport:
    """Audit the run at ``report_path`` and write ``integrity_report.json`` to ``output_dir``.

    ``context_path`` is an explicit context file and must exist. With none given,
    ``integrity_context.json`` next to the report is used when present and
    ``default_context`` is true; an experiment passes false so an unhashed file in
    a run directory cannot change a verdict.
    """
    result = load_run_result(report_path)
    if context_path is not None and not context_path.is_file():
        raise ManifestValidationError(message=f"integrity context not found: {context_path.name}")
    if context_path is None and default_context:
        fallback = report_path.parent / CONTEXT_NAME
        context_path = fallback if fallback.is_file() else None
    calls_path = report_path.parent / TOOL_CALLS_NAME
    overrides, preflight = _load_context(context_path)
    report = audit_run(
        result,
        inputs_for=swebench_inputs,
        calls=read_tool_calls(calls_path) if calls_path.is_file() else (),
        overrides=overrides,
        preflight=preflight,
        output_loader=_artifact_loader(report_path.parent / ARTIFACTS_DIR),
        context_digest=file_digest(context_path) if context_path is not None else None,
        tool_calls_digest=file_digest(calls_path) if calls_path.is_file() else None,
    )
    atomic_json(output_dir / REPORT_NAME, report.model_dump(mode="json"))
    return report


def summarize(report: IntegrityReport) -> str:
    """One line: counts and both rates."""
    flagged = sum(1 for item in report.instances if item.severity is not Severity.CLEAN)
    raw = "n/a" if report.raw_pass_rate is None else f"{report.raw_pass_rate:.3f}"
    clean = "n/a" if report.clean_pass_rate is None else f"{report.clean_pass_rate:.3f}"
    return (
        f"{report.total} instance(s); {flagged} flagged; "
        f"raw_pass_rate={raw} clean_pass_rate={clean}"
    )


@app.command("integrity")
def integrity(
    run_dir: Annotated[Path, typer.Argument(help="A run directory holding one canonical report.")],
    context: Annotated[
        Path | None,
        typer.Option("--context", help=f"Per-instance context JSON (default: {CONTEXT_NAME})."),
    ] = None,
) -> None:
    """Audit a finished run for known-fix lookups, test edits and grader exploits."""

    def action() -> IntegrityReport:
        return run_integrity_audit(find_run_report(run_dir), run_dir, context_path=context)

    report = run_cli_command(action, debug=False)
    console.print(safe_text(summarize(report)), soft_wrap=True)
    console.print(safe_text(f"wrote {run_dir / REPORT_NAME}"), soft_wrap=True)
