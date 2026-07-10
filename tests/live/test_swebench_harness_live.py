"""Live, Docker-backed SWE-bench harness validation (ADR-0014, design §7.1).

Opt-in only: ``@pytest.mark.live``, excluded from the default hermetic suite
and run solely by ``.github/workflows/live-swebench.yml`` (which installs
``agentic-evalkit[swebench]`` and provides a Docker daemon). It skips cleanly
when the capability or the required fixtures are absent, so it never converts
a missing environment into a false pass.

The design §7.1 fidelity gate: a known-resolved (gold) patch and an
intentionally-corrupted patch must pass through the *identical* real
``execute()`` code path and yield ``resolved=True`` / ``resolved=False``
respectively. The specific instance and its gold patch are supplied via
environment variables so this file commits no multi-megabyte fixture:

- ``AGENTIC_EVALKIT_SWEBENCH_INSTANCE`` -- a SWE-bench Verified instance id.
- ``AGENTIC_EVALKIT_SWEBENCH_GOLD_PATCH`` -- path to that instance's gold patch.

Until those are set in the workflow, the fidelity assertions skip (they do
not pass vacuously); wiring a chosen instance is the remaining step to close
acceptance criterion 6.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentic_evalkit.benchmarks.harness import HarnessRequest, HarnessStatus
from agentic_evalkit.benchmarks.swebench_docker import (
    SweBenchDockerHarnessExecutor,
    _default_preflight,
)

pytestmark = pytest.mark.live


def _require_capability() -> None:
    reason = _default_preflight()
    if reason is not None:
        pytest.skip(f"SWE-bench harness capability unavailable: {reason}")


def _gold_fixture() -> tuple[str, str]:
    instance_id = os.environ.get("AGENTIC_EVALKIT_SWEBENCH_INSTANCE")
    gold_patch_path = os.environ.get("AGENTIC_EVALKIT_SWEBENCH_GOLD_PATCH")
    if not instance_id or not gold_patch_path:
        pytest.skip(
            "set AGENTIC_EVALKIT_SWEBENCH_INSTANCE and AGENTIC_EVALKIT_SWEBENCH_GOLD_PATCH "
            "to run the gold/invalid-patch fidelity check"
        )
    patch = Path(gold_patch_path).read_text(encoding="utf-8")
    return instance_id, patch


def _request(instance_id: str, patch: str) -> HarnessRequest:
    return HarnessRequest(
        benchmark="swebench-verified@1",
        sample_id=f"swebench-verified:{instance_id}",
        prediction={
            "instance_id": instance_id,
            "model_name_or_path": "agentic-evalkit-live-test",
            "model_patch": patch,
        },
        timeout_seconds=1800.0,
    )


@pytest.mark.asyncio
async def test_gold_patch_resolves_through_the_real_execute_path() -> None:
    _require_capability()
    instance_id, gold_patch = _gold_fixture()
    result = await SweBenchDockerHarnessExecutor().execute(_request(instance_id, gold_patch))
    assert result.status is HarnessStatus.COMPLETED
    assert result.resolved is True


@pytest.mark.asyncio
async def test_corrupted_patch_does_not_resolve_through_the_same_path() -> None:
    _require_capability()
    instance_id, gold_patch = _gold_fixture()
    corrupted = gold_patch + "\n@@ this hunk does not apply @@\n"
    result = await SweBenchDockerHarnessExecutor().execute(_request(instance_id, corrupted))
    # Either the patch fails to apply or the tests still fail -- both are an
    # authoritative non-resolution, never an ERROR masquerading as one.
    assert result.status is HarnessStatus.COMPLETED
    assert result.resolved is False
