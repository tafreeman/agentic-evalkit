"""Container-backed SWE-bench harness executor (ADR-0014, design §7.1).

``SweBenchDockerHarnessExecutor`` is the concrete
:class:`~agentic_evalkit.benchmarks.harness.HarnessExecutor` that turns the
prediction-export-only SWE-bench Verified preset into an authoritatively
gradable one: it drives the official ``swebench`` package in Docker and maps
its per-instance report onto a :class:`HarnessResult`.

Two design rules keep it safe and testable:

- **Importable with zero extras.** Nothing from ``docker`` or ``swebench`` is
  imported at module load; the real integrations are lazy, inside the default
  preflight/evaluator callables, so ``benchmarks`` and ``cli.runs`` import
  cleanly on a base install and the grader simply reports ``UNAVAILABLE`` at
  run time.
- **Injected seams.** ``preflight`` (capability probe) and ``evaluator``
  (the official-harness call) are constructor-injected, so hermetic unit
  tests drive every UNAVAILABLE / ERROR / resolved-True / resolved-False
  branch with fakes and never touch a Docker daemon. Only the two default
  callables talk to Docker/``swebench``, and only the opt-in live workflow
  (``.github/workflows/live-swebench.yml``) exercises them.

Fidelity discipline (design §7.1): an infrastructure failure -- image pull,
timeout, OOM, a malformed report -- becomes ``HarnessStatus.ERROR`` with
``resolved=None``, never a guessed verdict; capability absence becomes
``HarnessStatus.UNAVAILABLE``. Only a report that actually carries a
``resolved`` field yields ``COMPLETED``.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import JsonValue

from agentic_evalkit.benchmarks.harness import HarnessRequest, HarnessResult, HarnessStatus
from agentic_evalkit.benchmarks.swebench import SweBenchVerifiedAdapter
from agentic_evalkit.models import EvalSample, NormalizedExecutionResult

__all__ = ["DEFAULT_INSTALL_HINT", "SweBenchDockerHarnessExecutor", "swebench_prediction"]

DEFAULT_INSTALL_HINT = "install agentic-evalkit[swebench] and start a Docker daemon"

_DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"

#: Keys of the official ``get_eval_report`` per-instance report we surface as
#: grading evidence (everything except the ``resolved`` verdict itself, which
#: maps to ``HarnessResult.resolved``).
_REPORT_EVIDENCE_KEYS = (
    "patch_is_None",
    "patch_exists",
    "patch_successfully_applied",
    "tests_status",
)

#: Callable that reports why the harness capability is unavailable, or
#: ``None`` when it is ready.
PreflightProbe = Callable[[], "str | None"]
#: Callable that runs the official harness for one request and returns its
#: per-instance ``get_eval_report``-shaped report.
Evaluator = Callable[[HarnessRequest], Mapping[str, JsonValue]]


class SweBenchDockerHarnessExecutor:
    """Runs the official SWE-bench harness in Docker (ADR-0014).

    Args:
        install_hint: Actionable message surfaced on ``UNAVAILABLE``.
        dataset_name: Hugging Face dataset the official harness resolves
            instances against (the default evaluator passes it through).
        preflight: Capability probe; returns a reason string when the harness
            cannot run, else ``None``. Defaults to the real docker+swebench
            probe. Injected in tests.
        evaluator: Runs the official harness for one request and returns its
            per-instance report. Defaults to the real subprocess invocation.
            Injected in tests.
    """

    def __init__(
        self,
        *,
        install_hint: str = DEFAULT_INSTALL_HINT,
        dataset_name: str = _DEFAULT_DATASET_NAME,
        preflight: PreflightProbe | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        self._install_hint = install_hint
        self._dataset_name = dataset_name
        self._preflight = preflight or _default_preflight
        self._evaluator = evaluator or self._run_official_harness

    async def execute(self, request: HarnessRequest) -> HarnessResult:
        unavailable_reason = self._preflight()
        if unavailable_reason is not None:
            return HarnessResult(
                status=HarnessStatus.UNAVAILABLE,
                resolved=None,
                message=(
                    f"Authoritative harness for {request.benchmark!r} is unavailable: "
                    f"{unavailable_reason} ({self._install_hint})"
                ),
            )
        try:
            report = await asyncio.to_thread(self._evaluator, request)
        except Exception as error:
            # Infrastructure failure is never a task failure (ADR-0008): no
            # verdict is invented, the cause is preserved for the report.
            return HarnessResult(
                status=HarnessStatus.ERROR,
                resolved=None,
                message=f"harness execution failed for {request.sample_id!r}: {error}",
                error={"type": type(error).__name__, "detail": str(error)},
            )
        return _result_from_report(report, request)

    def _run_official_harness(  # pragma: no cover - live Docker path, opt-in workflow only
        self, request: HarnessRequest
    ) -> Mapping[str, JsonValue]:
        """Drive ``swebench.harness.run_evaluation`` for one instance.

        Live-only: exercised by ``tests/live/test_swebench_harness_live.py``
        under the opt-in Docker workflow, never by the hermetic suite (which
        injects ``evaluator``). Delegates to the official package rather than
        reimplementing patch application or test execution.
        """
        instance_id = str(request.prediction["instance_id"])
        with tempfile.TemporaryDirectory(prefix="agentic-evalkit-swebench-") as tmp:
            work_dir = Path(tmp)
            predictions_path = work_dir / "predictions.json"
            predictions_path.write_text(json.dumps([dict(request.prediction)]), encoding="utf-8")
            run_id = f"agentic-evalkit-{request.sample_id}"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "swebench.harness.run_evaluation",
                    "--dataset_name",
                    self._dataset_name,
                    "--predictions_path",
                    str(predictions_path),
                    "--instance_ids",
                    instance_id,
                    "--run_id",
                    run_id,
                ],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                check=True,
            )
            return _read_instance_report(work_dir, instance_id, stdout=completed.stdout)


def swebench_prediction(
    sample: EvalSample, execution: NormalizedExecutionResult
) -> dict[str, JsonValue]:
    """Build the official SWE-bench prediction from an executed sample.

    The system under test emits its unified diff as ``model_patch`` (or
    ``patch``) in the normalized output; this exports the official three-key
    prediction through the adapter. A benchmark-neutral ``HarnessPredictor``
    (``graders.harness.HarnessPredictor``), so grading policy stays generic.
    """
    output = execution.output or {}
    raw_patch = output.get("model_patch")
    if raw_patch is None:
        raw_patch = output.get("patch", "")
    exported = SweBenchVerifiedAdapter().export_prediction(sample, str(raw_patch))
    prediction: dict[str, JsonValue] = {key: value for key, value in exported.items()}
    return prediction


def _result_from_report(report: Mapping[str, JsonValue], request: HarnessRequest) -> HarnessResult:
    """Map an official per-instance report onto a ``HarnessResult``.

    A report lacking a ``resolved`` field is not an authoritative verdict and
    surfaces as ``ERROR`` rather than a fabricated pass/fail.
    """
    if "resolved" not in report:
        return HarnessResult(
            status=HarnessStatus.ERROR,
            resolved=None,
            message=f"harness report for {request.sample_id!r} has no 'resolved' field",
            error={"report_keys": [key for key in sorted(report)]},
        )
    resolved = bool(report["resolved"])
    evidence: dict[str, JsonValue] = {
        key: report[key] for key in _REPORT_EVIDENCE_KEYS if key in report
    }
    raw_digests = report.get("image_digests")
    image_digests = (
        {str(k): str(v) for k, v in raw_digests.items()} if isinstance(raw_digests, Mapping) else {}
    )
    return HarnessResult(
        status=HarnessStatus.COMPLETED,
        resolved=resolved,
        message=(
            f"instance {request.sample_id!r} "
            f"{'resolved' if resolved else 'not resolved'} by the official harness"
        ),
        evidence=evidence,
        image_digests=image_digests,
    )


def _default_preflight() -> str | None:  # pragma: no cover - probes real Docker/swebench
    """Report why the SWE-bench Docker harness cannot run, or ``None`` if ready.

    Lazy imports keep this module importable on a base install; each failure
    yields an actionable reason string rather than raising.
    """
    try:
        import docker  # type: ignore
    except ImportError:
        return "the 'swebench' extra is not installed (missing docker SDK)"
    try:
        import swebench.harness.run_evaluation  # type: ignore  # noqa: F401
    except ImportError:
        return "the 'swebench' extra is not installed (missing swebench package)"
    try:
        docker.from_env().ping()
    except Exception as error:
        return f"Docker daemon is not reachable: {error}"
    return None


def _read_instance_report(  # pragma: no cover - live Docker path only
    work_dir: Path, instance_id: str, *, stdout: str
) -> Mapping[str, JsonValue]:
    """Locate and parse the official harness's per-instance report JSON.

    The harness writes an evaluation report under the working directory; this
    finds the single instance's entry (``get_eval_report`` shape). Raising
    here surfaces as ``HarnessStatus.ERROR`` in :meth:`execute`.
    """
    candidates = sorted(work_dir.rglob("*.json"))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        report = _extract_instance_report(payload, instance_id)
        if report is not None:
            return report
    raise RuntimeError(
        f"no per-instance report for {instance_id!r} found under {work_dir}; "
        f"harness stdout tail: {stdout[-500:]!r}"
    )


def _extract_instance_report(  # pragma: no cover - live Docker path only
    payload: object, instance_id: str
) -> Mapping[str, JsonValue] | None:
    """Pull one instance's report out of a harness JSON payload if present."""
    if not isinstance(payload, Mapping):
        return None
    if "resolved" in payload and payload.get("instance_id") in (instance_id, None):
        return payload
    nested = payload.get(instance_id)
    if isinstance(nested, Mapping):
        return nested
    return None
