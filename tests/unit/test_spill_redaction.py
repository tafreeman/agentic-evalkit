"""Regression guards for Story 2.1 -- spilled-artifact byte-path redaction
(R-002 P0, test-design gap #3).

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 2, Story 2.1) and
the TEA test design (R-002).

``EvalRunner._spill_large_output`` redacts spilled bytes using the runner's
``redaction_policy``. As of Story 2.1 that argument defaults to
``DEFAULT_REDACTION_POLICY``, so spill redaction is the shipped default: a
runner built with no explicit ``redaction_policy`` (as ``_default_runner``
below does, mirroring the shipped CLI) redacts secret-shaped substrings before
they land on disk. These tests exercise that default path and now pass as
regression guards: a real ``run`` whose target emits a large, secret-bearing
output no longer spills raw credentials to disk.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.models import ExecutionStatus, NormalizedExecutionResult
from agentic_evalkit.runner import EvalRunner

_SECRET = "sk-" + "A" * 40


def _default_runner(store: ArtifactStore) -> EvalRunner:
    """Construct a runner the way the shipped CLI does today: no explicit
    ``redaction_policy`` (i.e. the default). The spill path touches only the
    artifact store and the redaction policy, so the other collaborators are
    intentionally empty stand-ins.
    """
    return EvalRunner(
        catalog=cast(Any, None),
        adapters={},
        targets={},
        graders={},
        artifact_store=store,
    )


def _large_secret_execution() -> NormalizedExecutionResult:
    now = datetime.now(UTC)
    # > 8192 bytes even after redaction (the padding is not a secret pattern),
    # so it is always spilled to the artifact store.
    output: dict[str, JsonValue] = {"log": f"token={_SECRET} " + ("padding " * 2000)}
    return NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output=output,
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )


def _only_payload_text(root: Path) -> str:
    payloads = list(root.glob("*.bin"))
    assert len(payloads) == 1, f"expected exactly one spilled payload, found {payloads}"
    return payloads[0].read_bytes().decode("utf-8")


def _only_sidecar(root: Path) -> dict[str, Any]:
    sidecars = list(root.glob("*.json"))
    assert len(sidecars) == 1, f"expected exactly one sidecar, found {sidecars}"
    return cast("dict[str, Any]", json.loads(sidecars[0].read_text(encoding="utf-8")))


def test_default_run_path_redacts_spilled_secret_bytes(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    spilled = runner._spill_large_output(_large_secret_execution())
    # The output was spilled (replaced by a reference), not kept inline.
    assert "output_ref" in spilled.artifacts

    stored = _only_payload_text(root)
    assert _SECRET not in stored
    assert "[REDACTED]" in stored


def test_spilled_artifact_sidecar_marks_redacted(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    runner._spill_large_output(_large_secret_execution())

    assert _only_sidecar(root)["redacted"] is True


def test_redaction_applied_exactly_once(tmp_path: Path) -> None:
    # A single secret occurrence yields exactly one [REDACTED] marker: the
    # spill path redacts once and does not re-run redaction the report
    # boundary also applies (design: redaction applied exactly once).
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    runner._spill_large_output(_large_secret_execution())

    assert _only_payload_text(root).count("[REDACTED]") == 1
