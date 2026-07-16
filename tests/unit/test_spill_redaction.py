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
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.models import ExecutionStatus, NormalizedExecutionResult
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY
from agentic_evalkit.runner import _LARGE_OUTPUT_THRESHOLD_BYTES, EvalRunner

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

_SECRET = "sk-" + "A" * 40
# The three ratified credential shapes (design §12 / DEFAULT_REDACTION_POLICY):
# an OpenAI-style secret key, a Hugging Face user token, and an HTTP bearer
# value. Each is planted verbatim so the spill byte path is proven to redact
# every shape, not just the sk- case. Kept in sync with the patterns in
# ``agentic_evalkit.reporters.base.DEFAULT_REDACTION_POLICY``.
_SK_SECRET = _SECRET
_HF_SECRET = "hf_" + "B" * 40
_BEARER_SECRET = "Bearer " + "c" * 40
_ALL_SECRETS = (_SK_SECRET, _HF_SECRET, _BEARER_SECRET)


def _default_runner(store: ArtifactStore) -> EvalRunner:
    """Construct a runner the way the shipped CLI does today: no explicit
    ``redaction_policy`` (i.e. the default). The spill path touches only the
    artifact store and the redaction policy, so the other collaborators are
    intentionally empty stand-ins.
    """
    return EvalRunner(
        catalog=cast("Any", None),
        adapters={},
        targets={},
        graders={},
        artifact_store=store,
    )


def _execution_with_output(output: dict[str, JsonValue]) -> NormalizedExecutionResult:
    now = datetime.now(UTC)
    return NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output=output,
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )


def _large_secret_execution() -> NormalizedExecutionResult:
    # > 8192 bytes even after redaction (the padding is not a secret pattern),
    # so it is always spilled to the artifact store. Exactly one planted
    # secret, so a redacted spill carries exactly one [REDACTED] marker.
    return _execution_with_output({"log": f"token={_SECRET} " + ("padding " * 2000)})


def _large_multi_secret_execution() -> NormalizedExecutionResult:
    # > 8192 bytes and carrying all three ratified credential shapes at once,
    # so the spill byte path is exercised against every shape (sk-, hf_,
    # bearer), not just the sk- case.
    planted = f"sk={_SK_SECRET} hf={_HF_SECRET} auth={_BEARER_SECRET} "
    return _execution_with_output({"log": planted + ("padding " * 2000)})


def _output_serializing_to_exactly(n_bytes: int) -> dict[str, JsonValue]:
    """Return an output dict whose ``str(...)`` UTF-8-encodes to exactly
    ``n_bytes`` bytes and contains no secret-shaped substring.

    The spill path measures ``len(str(output).encode("utf-8"))``, so getting an
    exact serialized byte count means measuring the ``str(dict)`` overhead once
    (from an empty-valued dict) and padding the single string value to fill the
    remainder. The filler is a run of ``x`` (one byte in UTF-8, not part of any
    default pattern), so the encoded length is overhead + pad exactly.
    """
    overhead = len(str({"log": ""}).encode("utf-8"))
    pad = n_bytes - overhead
    assert pad >= 0, f"target {n_bytes} is below the {overhead}-byte serialization overhead"
    return {"log": "x" * pad}


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

    spilled = runner._spill_large_output(_large_multi_secret_execution())
    # The output was spilled (replaced by a reference), not kept inline.
    assert "output_ref" in spilled.artifacts

    stored = _only_payload_text(root)
    # Every ratified credential shape (sk-, hf_, bearer) is redacted from the
    # bytes that land on disk, proving the spill path covers all three, not
    # just the sk- case.
    for secret in _ALL_SECRETS:
        assert secret not in stored
    assert "[REDACTED]" in stored


def test_spilled_artifact_sidecar_marks_redacted(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    runner._spill_large_output(_large_secret_execution())

    assert _only_sidecar(root)["redacted"] is True


def test_output_exactly_at_threshold_stays_inline(tmp_path: Path) -> None:
    # The spill comparison is ``len(encoded) <= _LARGE_OUTPUT_THRESHOLD_BYTES``
    # (runner._spill_large_output), so a serialized output of *exactly* the
    # threshold is on the inline side, not spilled. Construct an output that
    # serializes to exactly the threshold and assert it is kept inline: no
    # payload is written and the output survives on the returned result.
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    output = _output_serializing_to_exactly(_LARGE_OUTPUT_THRESHOLD_BYTES)
    assert len(str(output).encode("utf-8")) == _LARGE_OUTPUT_THRESHOLD_BYTES

    result = runner._spill_large_output(_execution_with_output(output))

    # At exactly the threshold: inline, not spilled.
    assert "output_ref" not in result.artifacts
    assert result.output == output
    assert list(root.glob("*.bin")) == []


def test_one_byte_over_threshold_spills(tmp_path: Path) -> None:
    # One byte past the threshold is on the strictly-greater side and spills,
    # pinning the other side of the ``<=`` boundary tested above.
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    output = _output_serializing_to_exactly(_LARGE_OUTPUT_THRESHOLD_BYTES + 1)
    assert len(str(output).encode("utf-8")) == _LARGE_OUTPUT_THRESHOLD_BYTES + 1

    result = runner._spill_large_output(_execution_with_output(output))

    assert "output_ref" in result.artifacts
    assert result.output is None
    assert len(list(root.glob("*.bin"))) == 1


def test_spill_redaction_is_idempotent_at_the_boundary(tmp_path: Path) -> None:
    # An honest idempotence pin: re-running the SAME policy's patterns over the
    # already-redacted spill bytes changes nothing. (The old marker-count test
    # could not detect double application, because a second pass over redacted
    # output is a no-op regardless.) The marker-count == 1 assertion below still
    # holds for the single planted secret: exactly one [REDACTED] marker.
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    runner._spill_large_output(_large_secret_execution())

    persisted = _only_payload_text(root)
    # One planted secret -> exactly one redaction marker on disk.
    assert persisted.count("[REDACTED]") == 1

    # Apply the same default policy's patterns a second time, manually, exactly
    # as the runner does (compile each pattern, substitute "[REDACTED]"). An
    # idempotent redaction leaves the persisted bytes byte-for-byte unchanged.
    reapplied = persisted
    for pattern in DEFAULT_REDACTION_POLICY.secret_patterns:
        reapplied = re.sub(pattern, "[REDACTED]", reapplied)
    assert reapplied == persisted
