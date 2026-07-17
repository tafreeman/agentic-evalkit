"""Container-backed SWE-bench harness executor (ADR-0014, design §7.1).

``SweBenchDockerHarnessExecutor`` is the actual, working implementation of
:class:`~agentic_evalkit.benchmarks.harness.HarnessExecutor` for SWE-bench.
On its own, the SWE-bench Verified adapter can only export a prediction file
in the right format -- it can't tell you whether a fix actually worked. This
class is what adds that real, authoritative grading: it runs the official
``swebench`` Python package inside Docker, then takes the report that
package produces for one task and converts it into this project's own
:class:`HarnessResult` format.

Two design rules keep this class safe to import and easy to test:

- **You can import this file even without Docker or ``swebench`` installed.**
  Nothing from the ``docker`` or ``swebench`` packages is imported when this
  module is first loaded -- those imports only happen lazily, inside the
  default "preflight" and "evaluator" functions described below, and only
  once someone actually calls them. That means the rest of the codebase
  (``benchmarks``, ``cli.runs``) can import this module cleanly even on a
  bare install that skips the optional SWE-bench extra, and only discovers
  the capability is missing at the moment it's actually used -- at which
  point it just reports ``UNAVAILABLE``.
- **The real Docker/network calls are swappable.** The two things this class
  needs to do its job -- checking whether the harness is even able to run
  (``preflight``), and actually invoking the official harness
  (``evaluator``) -- are passed in through the constructor instead of being
  hard-coded. That means ordinary, fully offline unit tests can supply fake
  versions of both and walk through every possible outcome (unavailable,
  error, resolved-true, resolved-false) without ever needing a real Docker
  daemon. Only the two real, default versions of these functions actually
  talk to Docker and ``swebench``, and those only ever run under the
  separate, opt-in live workflow (``.github/workflows/live-swebench.yml``),
  never in the normal test suite.

Being strictly honest about what actually happened (design §7.1): an
infrastructure problem -- failing to download a Docker image, a timeout,
running out of memory, a report that doesn't parse -- always becomes
``HarnessStatus.ERROR`` with ``resolved=None``. It is never turned into a
guessed pass/fail verdict. Similarly, the harness simply not being available
becomes ``HarnessStatus.UNAVAILABLE``. Only an official report that actually
contains a ``resolved`` field is allowed to produce
``HarnessStatus.COMPLETED``.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import JsonValue

from agentic_evalkit.benchmarks.harness import HarnessRequest, HarnessResult, HarnessStatus
from agentic_evalkit.benchmarks.swebench import SweBenchVerifiedAdapter

if TYPE_CHECKING:
    from agentic_evalkit.models import EvalSample, NormalizedExecutionResult

__all__ = [
    "DEFAULT_INSTALL_HINT",
    "SweBenchDockerHarnessExecutor",
    "docker_safe_run_id",
    "swebench_prediction",
]

DEFAULT_INSTALL_HINT = "install agentic-evalkit[swebench] and start a Docker daemon"

_DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"

#: The fields from the official ``get_eval_report`` per-instance report that
#: we copy into our own result as supporting evidence -- everything except
#: the ``resolved`` verdict itself, which instead becomes
#: ``HarnessResult.resolved``.
_REPORT_EVIDENCE_KEYS = (
    "patch_is_None",
    "patch_exists",
    "patch_successfully_applied",
    "tests_status",
)

#: A function that checks whether the harness is able to run at all. Returns
#: a reason string explaining why it can't, or ``None`` if everything's ready.
PreflightProbe = Callable[[], "str | None"]
#: A function that actually runs the official harness for one request and
#: returns its report, shaped like the official ``get_eval_report`` output
#: for a single instance.
Evaluator = Callable[[HarnessRequest], Mapping[str, JsonValue]]


class SweBenchDockerHarnessExecutor:
    """Runs the official SWE-bench harness inside Docker (ADR-0014).

    Args:
        install_hint: The actionable message shown to users when this
            reports ``UNAVAILABLE``, telling them what to install/start.
        dataset_name: The Hugging Face dataset name the official harness
            looks up each instance's details in (the default evaluator just
            passes this straight through to it).
        preflight: A function that checks whether the harness can run at
            all; it returns a reason string if not, or ``None`` if it's
            ready to go. Defaults to the real check against Docker and
            ``swebench``. Tests substitute a fake version of this.
        evaluator: A function that runs the official harness for one
            request and returns its per-instance report. Defaults to the
            real subprocess call. Tests substitute a fake version of this.
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
        # The default preflight check pings the Docker daemon, which is a
        # blocking network/IO call. We run it in a background thread (rather
        # than directly on the async event loop) so that a slow or stuck
        # Docker daemon can never freeze other runs happening concurrently.
        unavailable_reason = await asyncio.to_thread(self._preflight)
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
            # An infrastructure failure (the tooling itself breaking) is a
            # completely different thing from the task genuinely failing
            # (ADR-0008) -- so we don't invent a pass/fail verdict here. We
            # just record what went wrong so it's visible in the report.
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
        """Actually invoke ``swebench.harness.run_evaluation`` for one instance.

        This only ever runs for real inside
        ``tests/live/test_swebench_harness_live.py``, as part of the opt-in
        Docker workflow -- never as part of the normal, fully offline test
        suite (which instead injects a fake ``evaluator``). It calls out to
        the official ``swebench`` package to do the actual work, rather than
        reimplementing patch-applying or test-running logic itself.
        """
        instance_id = str(request.prediction["instance_id"])
        with tempfile.TemporaryDirectory(prefix="agentic-evalkit-swebench-") as tmp:
            work_dir = Path(tmp)
            predictions_path = work_dir / "predictions.json"
            predictions_path.write_text(json.dumps([dict(request.prediction)]), encoding="utf-8")
            run_id = docker_safe_run_id(request.sample_id)
            # This subprocess call is safe even though ruff's security linter
            # flags subprocess calls by default (rule S603): the command is
            # passed as a list, not a shell string, so there's no
            # shell-injection risk, and every argument below is built by
            # this code itself, never taken from raw, unvalidated user input.
            completed = subprocess.run(  # noqa: S603
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
                # Harness and Docker logs can contain bytes that aren't valid
                # UTF-8, or ANSI color-escape codes. We'd rather show
                # slightly garbled text than have decoding a log line crash
                # the run.
                errors="replace",
                timeout=request.timeout_seconds,
                check=True,
            )
            return _read_instance_report(work_dir, instance_id, stdout=completed.stdout)


def docker_safe_run_id(sample_id: str) -> str:
    """Turn a sample id into a ``run_id`` that Docker will actually accept.

    The official harness uses ``run_id`` as part of the name it gives
    containers and images, but Docker names don't allow characters like
    ``:`` -- and this project's own sample ids look like
    ``swebench-verified:<instance_id>``, colon included. Left as-is, every
    real Docker-backed run would fail right at container creation. So here,
    every character that isn't a letter, digit, underscore, dot, or hyphen
    gets replaced with a hyphen instead (flagged as a priority-1 fix in a
    Codex code review).
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", sample_id)
    return f"agentic-evalkit-{safe}"


def swebench_prediction(
    sample: EvalSample, execution: NormalizedExecutionResult
) -> dict[str, JsonValue]:
    """Build an official-format SWE-bench prediction from a completed run.

    The system under test is expected to emit its code fix as a unified
    diff, under either the ``model_patch`` or ``patch`` key in its
    normalized output. This function pulls that patch out and exports it,
    through the adapter, as the official three-field SWE-bench prediction.
    Its shape matches the generic ``HarnessPredictor`` interface
    (``graders.harness.HarnessPredictor``), which is what lets the general
    grading logic stay benchmark-neutral instead of hard-coding SWE-bench
    specifics.
    """
    output = execution.output or {}
    raw_patch = output.get("model_patch")
    if raw_patch is None:
        raw_patch = output.get("patch", "")
    exported = SweBenchVerifiedAdapter().export_prediction(sample, str(raw_patch))
    prediction: dict[str, JsonValue] = dict(exported)
    return prediction


def _result_from_report(report: Mapping[str, JsonValue], request: HarnessRequest) -> HarnessResult:
    """Convert one official per-instance report into a ``HarnessResult``.

    If the report doesn't even contain a ``resolved`` field, then it isn't
    a real, authoritative verdict at all -- so this reports ``ERROR``
    instead of inventing a pass or fail.
    """
    if "resolved" not in report:
        return HarnessResult(
            status=HarnessStatus.ERROR,
            resolved=None,
            message=f"harness report for {request.sample_id!r} has no 'resolved' field",
            # This looks like it could be simplified to plain `sorted(report)`,
            # and a linter would normally flag the comprehension below as
            # unnecessary -- but keeping it as a comprehension here is
            # deliberate, not a mistake. `report` is typed as
            # `Mapping[str, JsonValue]`, so `sorted(report)` on its own would
            # be typed as `list[str]`. But the dict field we're building here
            # (`error: dict[str, JsonValue] | None`) needs a `list[JsonValue]`
            # in that slot, and mypy treats `list[str]` and `list[JsonValue]`
            # as incompatible with each other -- even though every individual
            # `str` is a valid `JsonValue`, Python's type system won't let a
            # `list[str]` stand in for a `list[JsonValue]`, because lists are
            # mutable ("invariance"). Writing it as a comprehension instead
            # lets mypy check each element against the `JsonValue` type the
            # surrounding dict already expects, instead of rejecting a
            # `list[str]` outright.
            error={"report_keys": [key for key in sorted(report)]},  # noqa: C416
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
    """Explain why the SWE-bench Docker harness can't run, or return ``None`` if it's ready.

    The imports of ``docker`` and ``swebench`` happen here, inside the
    function, rather than at the top of the module -- that is exactly what
    keeps this module importable even on a base install that doesn't have
    those packages. Each way this check can fail returns a clear, actionable
    reason string instead of letting an exception escape.
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
    """Find and parse the official harness's report for this one instance.

    The harness writes its evaluation report as JSON files somewhere under
    the working directory; this searches for them and pulls out the single
    entry for this instance (in the same shape the official
    ``get_eval_report`` function produces). If nothing is found, this raises
    an exception, which :meth:`execute` turns into ``HarnessStatus.ERROR``.
    """
    candidates = sorted(work_dir.rglob("*.json"))
    for candidate in candidates:
        try:
            # We pass errors="replace" here because a report file that
            # happens to contain a stray non-UTF-8 byte must not blow up
            # with a UnicodeDecodeError. That error is a subclass of
            # ValueError, which the `except` clause below does NOT catch (it
            # only catches OSError and JSONDecodeError) -- so left
            # unhandled, a decoding error here would crash the whole harness
            # run instead of just being skipped like a bad JSON file is.
            payload = json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
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
