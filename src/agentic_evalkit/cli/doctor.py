"""``agentic-evalkit doctor``: checks the environment before you try to run an evaluation.

Design doc, section 11.1.

This command runs a handful of checks -- Python version, whether the cache
directory can actually be read from and written to, whether the Hugging
Face dataset provider is reachable, and whether optional add-on
capabilities are installed -- and reports each one as ``ok``, ``warning``,
or ``error``, plus a short suggestion for how to fix it when it isn't
``ok``. Note that ``doctor`` itself never raises an exception just because
one check failed: a broken target or an unreachable provider is exactly
the kind of problem a user runs ``doctor`` to find out about in the first
place, so a failed check is data to report, not an error to crash on.
Instead, after running every check, ``doctor`` looks at all of their
statuses together and picks one single exit code for the whole command
based on that.
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
        # This needs to pass an HfApi instance somewhere that expects the
        # private "_HubClient" protocol from datasets.huggingface. A
        # "protocol" here is just a structural interface: it lists the
        # methods an object must have, and anything with matching methods
        # counts as satisfying it, without needing to formally inherit from
        # it. HfApi does have matching methods -- and that really is
        # checked at runtime over there, via an isinstance check that
        # Python's @runtime_checkable decorator enables -- but mypy (the
        # static type checker) can't confirm that just by reading the code,
        # because HfApi's actual methods spell out specific keyword
        # parameters where the protocol's methods are declared more loosely
        # with **kwargs. On top of that, _HubClient is private (never
        # exported from its module), so this file cannot even import it to
        # spell out the match directly. So this casts through Any instead
        # -- telling mypy to stop type-checking this one value -- which is
        # the same workaround used for the same reason in cli/datasets.py.
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


def _check_swebench_capability() -> DoctorCheck:
    """Report whether the optional ``swebench`` add-on is usable.

    This project keeps its base install small and lets users opt into
    heavier, provider-specific features through "extras" -- optional
    groups of extra dependencies you install with, e.g., ``pip install
    'agentic-evalkit[swebench]'`` (that policy is ADR-0009). Two extras
    that were only ever placeholders, never actually wired up to anything,
    were removed from this project on 2026-07-11 -- but ``swebench`` is not
    one of them. It is a real, working extra: installing it pulls in the
    ``swebench`` and ``docker`` packages needed to run evaluations through
    the container-based SWE-bench test harness (that harness executor is
    described in ADR-0014). Finding the ``swebench`` module installed only
    tells you the extra itself is present, though -- it is not enough on
    its own to actually run that harness, since doing so also needs a
    Docker daemon this code can connect to. That's why the remediation
    message below, for when this check fails, mentions installing the
    extra *and* having Docker running.
    """
    if find_spec("swebench") is not None:
        return DoctorCheck(
            name="capability_swebench",
            status="ok",
            detail="optional capability swebench is installed",
        )
    return DoctorCheck(
        name="capability_swebench",
        status="warning",
        detail="optional capability swebench is not installed",
        remediation="Install agentic-evalkit[swebench] and ensure a Docker daemon is running.",
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
    checks.append(_check_swebench_capability())
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
        # check.name always comes from one of the hardcoded check names
        # defined above (e.g. "python_version"), so it's safe to print as a
        # plain string either way. check.detail and check.remediation,
        # though, are free-form text that can legitimately contain a
        # "[...]"-shaped chunk (a pip extra name, for instance) -- that
        # must not be misread as Rich style markup, so those two go
        # through safe_text (see cli/app.py) to render exactly as written.
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
