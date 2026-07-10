"""Live, Docker-backed SWE-bench harness validation (ADR-0014, design §7.1).

Opt-in only: ``@pytest.mark.live``, excluded from the default hermetic suite
and run solely by ``.github/workflows/live-swebench.yml`` (which installs
``agentic-evalkit[swebench]`` and provides a Docker daemon). It skips only
when the capability is genuinely absent -- it does NOT skip for a missing
fixture, so a scheduled/dispatch run actually validates the harness rather
than passing vacuously.

The design §7.1 fidelity gate: a known-resolved (gold) patch and an
intentionally-corrupted patch pass through the *identical* real ``execute()``
code path and yield ``resolved=True`` / ``resolved=False`` respectively. The
gold patch is the dataset's own reference solution (the ``patch`` field of
each SWE-bench Verified row), so no external fixture file is needed:

- ``AGENTIC_EVALKIT_SWEBENCH_INSTANCE`` (optional) -- a specific instance id;
  when unset, the first dataset row is used, so the check always runs against
  a real instance under Docker.
"""

from __future__ import annotations

import os

import pytest

from agentic_evalkit.benchmarks.harness import HarnessRequest, HarnessStatus
from agentic_evalkit.benchmarks.swebench_docker import (
    SweBenchDockerHarnessExecutor,
    _default_preflight,
)

pytestmark = pytest.mark.live

_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"


def _require_capability() -> None:
    reason = _default_preflight()
    if reason is not None:
        pytest.skip(f"SWE-bench harness capability unavailable: {reason}")


def _gold_instance() -> tuple[str, str]:
    """Return ``(instance_id, gold_patch)`` from SWE-bench Verified.

    The dataset's own ``patch`` field IS the reference solution, so the
    fidelity check needs no committed multi-megabyte fixture. Honors
    ``AGENTIC_EVALKIT_SWEBENCH_INSTANCE`` when set, else the first row.
    """
    try:
        from datasets import load_dataset  # provided by the swebench extra
    except ImportError:  # pragma: no cover - live only
        pytest.skip("the 'datasets' package (swebench extra) is required")

    dataset = load_dataset(_DATASET_NAME, split="test")
    wanted = os.environ.get("AGENTIC_EVALKIT_SWEBENCH_INSTANCE")
    if wanted:
        matches = [row for row in dataset if row["instance_id"] == wanted]
        if not matches:
            pytest.skip(f"instance {wanted!r} not found in {_DATASET_NAME}")
        row = matches[0]
    else:
        row = dataset[0]
    return str(row["instance_id"]), str(row["patch"])


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
    instance_id, gold_patch = _gold_instance()
    result = await SweBenchDockerHarnessExecutor().execute(_request(instance_id, gold_patch))
    assert result.status is HarnessStatus.COMPLETED
    assert result.resolved is True


@pytest.mark.asyncio
async def test_corrupted_patch_does_not_resolve_through_the_same_path() -> None:
    _require_capability()
    instance_id, gold_patch = _gold_instance()
    corrupted = gold_patch + "\n@@ this hunk does not apply @@\n"
    result = await SweBenchDockerHarnessExecutor().execute(_request(instance_id, corrupted))
    # Either the patch fails to apply or the tests still fail -- both are an
    # authoritative non-resolution, never an ERROR masquerading as one.
    assert result.status is HarnessStatus.COMPLETED
    assert result.resolved is False
