"""Run a pinned HELM installation through its own CLI and retain native evidence."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
from typing import TYPE_CHECKING

from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.experiment import atomic_json, file_digest
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY

if TYPE_CHECKING:
    from pathlib import Path

    from agentic_evalkit.experiment import HelmConfig


def helm_version(config: HelmConfig) -> str:
    try:
        result = subprocess.run(  # noqa: S603 - explicit executable from the user's experiment
            [
                config.python,
                "-c",
                "from importlib.metadata import version; print(version('crfm-helm'))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ManifestValidationError(message="HELM Python environment is unavailable") from error
    version = result.stdout.strip()
    if result.returncode or version != config.version:
        raise ManifestValidationError(
            message=(
                "HELM version mismatch: require "
                f"{config.version}, found {version or 'not installed'}"
            )
        )
    return version


def helm_command(config: HelmConfig, directory: Path, suite: str) -> list[str]:
    command = [
        config.python,
        "-X",
        "utf8",
        "-m",
        "helm.benchmark.run",
        "--run-entries",
        *config.run_entries,
        "--suite",
        suite,
        "--max-eval-instances",
        str(config.max_eval_instances),
        "--num-threads",
        str(config.num_threads),
        "--output-path",
        str(directory / "helm"),
        "--local-path",
        config.local_path,
        "--cache-instances",
        "--exit-on-error",
    ]
    if config.phase == "prepare":
        command.append("--cache-instances-only")
    else:
        command.append("--skip-completed-runs")
    return command


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if sys.platform == "win32":
        # Only the child we created and its descendants are targeted.
        await asyncio.to_thread(
            subprocess.run,
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def run_helm(config: HelmConfig, directory: Path, suite: str) -> Path:
    helm_version(config)
    directory.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    process = await asyncio.create_subprocess_exec(
        *helm_command(config, directory, suite),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=sys.platform != "win32",
    )
    try:
        async with asyncio.timeout(config.timeout_seconds):
            if process.stdout is None:
                raise RuntimeError("HELM process has no output stream")
            with (directory / "helm.log").open("a", encoding="utf-8") as log:
                while line := await process.stdout.readline():
                    text = line.decode("utf-8", errors="replace")
                    for pattern in DEFAULT_REDACTION_POLICY.secret_patterns:
                        text = re.sub(pattern, "[REDACTED]", text)
                    log.write(text)
                    log.flush()
            code = await process.wait()
    except BaseException:
        await _stop_process(process)
        raise
    if code != 0:
        raise ManifestValidationError(message=f"HELM exited {code}; see {directory / 'helm.log'}")
    output = directory / "helm"
    pattern = "input_instances.json" if config.phase == "prepare" else "stats.json"
    files = sorted(output.rglob(pattern))
    if not files:
        raise ManifestValidationError(
            message=f"HELM exited successfully but produced no {pattern} evidence"
        )
    # Native files are retained in their original schema; a successful process
    # is not converted into an EvalKit task pass or a comparable score.
    evidence = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, (dict, list)) or not data:
            raise ManifestValidationError(message=f"empty or malformed HELM evidence: {path.name}")
        evidence.append(
            {"path": path.relative_to(directory).as_posix(), "sha256": file_digest(path)}
        )
    report = directory / "helm-result.json"
    atomic_json(
        report,
        {
            "schema_version": "1",
            "backend": "helm",
            "version": config.version,
            "phase": config.phase,
            "suite": suite,
            "evidence": evidence,
            "score_interpretation": "native HELM metrics; no cross-backend comparison",
        },
    )
    return report
