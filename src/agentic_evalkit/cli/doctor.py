"""``agentic-evalkit doctor``: environment/capability preflight (design §11.1).

Checks Python version, cache read/write, Hugging Face provider health,
optional-capability availability, and judge calibration readiness. Each
check reports ``ok``, ``warning``, or ``error`` plus a short remediation
string; ``doctor`` never raises for an individual failed check (a broken
target or an unreachable provider is exactly the situation a user runs
``doctor`` to diagnose) -- it aggregates every check's status into one exit
code instead.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import httpx
import typer
from huggingface_hub import HfApi
from pydantic import BaseModel
from rich.table import Table

from agentic_evalkit.cli.app import ExitCode, app, console, print_output, safe_text
from agentic_evalkit.datasets.huggingface import HuggingFaceDatasetProvider

_MIN_PYTHON = (3, 11)


class DoctorCheck(BaseModel):
    """One ``doctor`` check result."""

    name: str
    status: Literal["ok", "warning", "error"]
    detail: str
    remediation: str | None = None


def _check_python_version() -> DoctorCheck:
    current = sys.version_info[:2]
    if current >= _MIN_PYTHON:
        return DoctorCheck(
            name="python_version",
            status="ok",
            detail=f"Python {sys.version.split()[0]}",
        )
    return DoctorCheck(
        name="python_version",
        status="error",
        detail=f"Python {sys.version.split()[0]} is below the minimum supported {_MIN_PYTHON}",
        remediation=f"Install Python {'.'.join(str(part) for part in _MIN_PYTHON)} or newer.",
    )


def _check_cache_read_write() -> DoctorCheck:
    try:
        cache_root = Path(tempfile.gettempdir()) / "agentic-evalkit-doctor-check"
        cache_root.mkdir(parents=True, exist_ok=True)
        probe = cache_root / "probe.txt"
        probe.write_text("ok", encoding="utf-8")
        probe.read_text(encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as error:
        return DoctorCheck(
            name="cache_read_write",
            status="error",
            detail=f"cache directory is not writable: {error}",
            remediation="Check filesystem permissions for the platform user-cache directory.",
        )
    return DoctorCheck(name="cache_read_write", status="ok", detail="cache directory is writable")


async def _check_huggingface_health() -> DoctorCheck:
    try:
        # HfApi structurally satisfies datasets.huggingface's private
        # _HubClient protocol at runtime (verified there via
        # @runtime_checkable isinstance checks) but mypy cannot prove it
        # statically; _HubClient itself is not exported, so this casts
        # through Any rather than importing a private symbol -- mirroring
        # the identical situation and fix in cli/datasets.py.
        async with HuggingFaceDatasetProvider.create(hub=cast("Any", HfApi())) as provider:
            health = await provider.healthcheck()
    except httpx.HTTPError as error:
        return DoctorCheck(
            name="huggingface_health",
            status="error",
            detail=f"Hugging Face Dataset Viewer unreachable: {error}",
            remediation="Check network connectivity, or run with --offline.",
        )
    if health.status == "ok":
        latency = "n/a" if health.latency_ms is None else f"{health.latency_ms:.1f}"
        return DoctorCheck(
            name="huggingface_health",
            status="ok",
            detail=f"Dataset Viewer reachable (latency={latency}ms)",
        )
    return DoctorCheck(
        name="huggingface_health",
        status="warning" if health.status == "degraded" else "error",
        detail=f"Dataset Viewer reports status={health.status} ({health.error_code})",
        remediation="Check https://status.huggingface.co and retry.",
    )


def _check_optional_capability(name: str, module: str) -> DoctorCheck:
    if find_spec(module) is not None:
        return DoctorCheck(
            name=f"capability_{name}",
            status="ok",
            detail=f"optional capability {name!r} is installed",
        )
    return DoctorCheck(
        name=f"capability_{name}",
        status="warning",
        detail=f"optional capability {name!r} is not installed",
        remediation=f"Install with: pip install 'agentic-evalkit[{name}]'",
    )


def _check_judge_calibration() -> DoctorCheck:
    # No judge is configured by default in the objective-only v0.1 CLI; an
    # uncalibrated judge must never gate a release (ADR-0007), so the
    # correct default state here is an explicit "not configured" warning
    # rather than a silently skipped check.
    return DoctorCheck(
        name="judge_calibration",
        status="warning",
        detail="no calibrated judge is configured",
        remediation="Objective graders gate this release; judges are configured separately.",
    )


def run_doctor_checks(*, offline: bool) -> list[DoctorCheck]:
    """Run every doctor check and return the results in a fixed order."""
    checks = [_check_python_version(), _check_cache_read_write()]
    if offline:
        checks.append(
            DoctorCheck(
                name="huggingface_health",
                status="warning",
                detail="skipped: --offline was requested",
            )
        )
    else:
        checks.append(asyncio.run(_check_huggingface_health()))
    checks.append(_check_optional_capability("parquet", "pyarrow"))
    checks.append(_check_optional_capability("swebench", "swebench"))
    checks.append(_check_judge_calibration())
    return checks


def _render_table(checks: list[DoctorCheck]) -> None:
    table = Table(title="agentic-evalkit doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_column("Remediation")
    status_styles = {"ok": "green", "warning": "yellow", "error": "red"}
    for check in checks:
        style = status_styles[check.status]
        # check.name is a hardcoded literal (safe as markup-free plain
        # text either way); check.detail/remediation are free-form dynamic
        # text (may legitimately contain "[...]", e.g. a pip extra name)
        # and must render literally via safe_text rather than be
        # re-parsed as Rich markup.
        table.add_row(
            check.name,
            f"[{style}]{check.status}[/{style}]",
            safe_text(check.detail),
            safe_text(check.remediation or ""),
        )
    console.print(table)


@app.command()
def doctor(
    format_: Annotated[
        str, typer.Option("--format", help="Output format: table or json.")
    ] = "table",
    offline: Annotated[
        bool, typer.Option("--offline", help="Skip checks that require network access.")
    ] = False,
) -> None:
    """Check provider access, cache permissions, and optional capabilities."""
    checks = run_doctor_checks(offline=offline)
    if format_ == "json":
        print_output([check.model_dump(mode="json") for check in checks], format_=format_)
    else:
        _render_table(checks)
    if any(check.status == "error" for check in checks):
        raise typer.Exit(code=int(ExitCode.MISSING_CAPABILITY))
