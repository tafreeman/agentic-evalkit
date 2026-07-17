"""Tests for :class:`agentic_evalkit.benchmarks.swebench_docker` (ADR-0014).

These tests are hermetic, meaning they never touch the network or a real
Docker daemon. That's possible because the two places this executor would
normally reach out to the real world -- ``preflight`` (checks whether
Docker and the ``swebench`` package are ready to use) and ``evaluator``
(actually runs the official harness) -- are passed into it as plain
functions, which tests can swap out for fakes. That lets every possible
outcome -- UNAVAILABLE, ERROR, resolved=True, resolved=False, and a
malformed report -- be tested directly, without needing a Docker daemon or
the ``swebench`` package installed. The real versions of those two
functions (``_run_official_harness`` and ``_default_preflight``) only run
for real inside ``tests/live/test_swebench_harness_live.py``, which does
use a real Docker daemon.
"""

from datetime import UTC, datetime

import pytest
from _benchmark_fixtures import _harness_request
from pydantic import JsonValue

from agentic_evalkit.benchmarks.harness import HarnessExecutor, HarnessRequest, HarnessStatus
from agentic_evalkit.benchmarks.swebench_docker import (
    SweBenchDockerHarnessExecutor,
    docker_safe_run_id,
    swebench_prediction,
)
from agentic_evalkit.models import EvalSample, ExecutionStatus, NormalizedExecutionResult


def _executor(
    *,
    preflight_reason: str | None,
    report: dict[str, JsonValue] | None = None,
    evaluator_error: Exception | None = None,
) -> SweBenchDockerHarnessExecutor:
    def _preflight() -> str | None:
        return preflight_reason

    def _evaluator(request: HarnessRequest) -> dict[str, JsonValue]:
        if evaluator_error is not None:
            raise evaluator_error
        assert report is not None
        return report

    return SweBenchDockerHarnessExecutor(preflight=_preflight, evaluator=_evaluator)


def test_executor_satisfies_the_harness_protocol() -> None:
    executor = _executor(preflight_reason="x")
    assert isinstance(executor, HarnessExecutor)


@pytest.mark.asyncio
async def test_missing_capability_is_unavailable_with_an_actionable_hint() -> None:
    executor = _executor(preflight_reason="the 'swebench' extra is not installed")
    result = await executor.execute(_harness_request())
    assert result.status is HarnessStatus.UNAVAILABLE
    assert result.resolved is None
    assert "swebench" in result.message
    assert "Docker" in result.message  # comes from the default install-hint text


@pytest.mark.asyncio
async def test_evaluator_exception_is_a_typed_error_never_a_verdict() -> None:
    executor = _executor(preflight_reason=None, evaluator_error=RuntimeError("image pull failed"))
    result = await executor.execute(_harness_request())
    assert result.status is HarnessStatus.ERROR
    assert result.resolved is None
    assert result.error is not None
    assert result.error["type"] == "RuntimeError"
    assert "image pull failed" in result.message


@pytest.mark.asyncio
async def test_resolved_report_maps_to_completed_true_with_evidence() -> None:
    report: dict[str, JsonValue] = {
        "instance_id": "org__repo-1",
        "resolved": True,
        "patch_exists": True,
        "patch_successfully_applied": True,
        "tests_status": {"FAIL_TO_PASS": {"success": ["test_x"], "failure": []}},
        "image_digests": {"base": "sha256:abc"},
    }
    result = await _executor(preflight_reason=None, report=report).execute(_harness_request())
    assert result.status is HarnessStatus.COMPLETED
    assert result.resolved is True
    assert result.evidence["patch_successfully_applied"] is True
    assert result.evidence["tests_status"] == report["tests_status"]
    # The pass/fail verdict lives in `result.resolved` -- it is not also
    # copied into `result.evidence`.
    assert "resolved" not in result.evidence
    assert result.image_digests == {"base": "sha256:abc"}


@pytest.mark.asyncio
async def test_unresolved_report_maps_to_completed_false() -> None:
    report: dict[str, JsonValue] = {
        "instance_id": "org__repo-1",
        "resolved": False,
        "patch_successfully_applied": True,
    }
    result = await _executor(preflight_reason=None, report=report).execute(_harness_request())
    assert result.status is HarnessStatus.COMPLETED
    assert result.resolved is False


@pytest.mark.asyncio
async def test_report_without_a_resolved_field_is_an_error_not_a_fabricated_verdict() -> None:
    report: dict[str, JsonValue] = {"instance_id": "org__repo-1", "patch_exists": False}
    result = await _executor(preflight_reason=None, report=report).execute(_harness_request())
    assert result.status is HarnessStatus.ERROR
    assert result.resolved is None
    assert result.error is not None


@pytest.mark.asyncio
async def test_non_mapping_image_digests_degrade_to_empty() -> None:
    report: dict[str, JsonValue] = {
        "instance_id": "org__repo-1",
        "resolved": True,
        "image_digests": "not-a-mapping",
    }
    result = await _executor(preflight_reason=None, report=report).execute(_harness_request())
    assert result.status is HarnessStatus.COMPLETED
    assert result.image_digests == {}


# --- Tests for the swebench_prediction() helper function -------------------


def _swebench_sample() -> EvalSample:
    return EvalSample(
        sample_id="swebench-verified:org__repo-1",
        input={"problem_statement": "fix it", "repo": "org/repo"},
        metadata={"instance_id": "org__repo-1"},
        source_digest="sha256:row",
        adapter="swebench-verified@1",
    )


def _completed(output: dict[str, JsonValue] | None) -> NormalizedExecutionResult:
    now = datetime.now(UTC)
    return NormalizedExecutionResult(
        sample_id="swebench-verified:org__repo-1",
        attempt=1,
        output=output,
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )


def test_swebench_prediction_exports_the_official_three_keys_from_model_patch() -> None:
    prediction = swebench_prediction(_swebench_sample(), _completed({"model_patch": "diff --git"}))
    assert prediction == {
        "instance_id": "org__repo-1",
        "model_name_or_path": "agentic-evalkit-target",
        "model_patch": "diff --git",
    }


def test_swebench_prediction_falls_back_to_a_patch_key() -> None:
    prediction = swebench_prediction(_swebench_sample(), _completed({"patch": "diff --git b"}))
    assert prediction["model_patch"] == "diff --git b"


def test_swebench_prediction_defaults_to_an_empty_patch_when_absent() -> None:
    prediction = swebench_prediction(_swebench_sample(), _completed({"answer": "no patch here"}))
    assert prediction["model_patch"] == ""


# --- docker_safe_run_id: fixes a bug where a ':' in run_id broke Docker's --
# --- container naming (flagged as a top-priority fix in a Codex code review) --


def test_docker_safe_run_id_strips_characters_docker_rejects() -> None:
    # This project's own sample ids look like "swebench-verified:<instance>".
    # But Docker doesn't allow a ':' character in container/image names, so
    # that colon has to be replaced before the id reaches Docker.
    run_id = docker_safe_run_id("swebench-verified:astropy__astropy-1")
    assert ":" not in run_id
    assert run_id == "agentic-evalkit-swebench-verified-astropy__astropy-1"


def test_docker_safe_run_id_preserves_a_clean_instance_id() -> None:
    assert docker_safe_run_id("astropy__astropy-1") == "agentic-evalkit-astropy__astropy-1"
