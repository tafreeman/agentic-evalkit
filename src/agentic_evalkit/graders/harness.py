"""Harness-backed grading: turn a HarnessResult into a GradeResult (ADR-0014).

``HarnessGrader`` is the previously-missing bridge between the framework's
grading boundary (``Grader.grade(sample, execution) -> GradeResult``) and an
authoritative :class:`~agentic_evalkit.benchmarks.harness.HarnessExecutor`.
Without it, an authoritative benchmark verdict (e.g. SWE-bench "resolved")
had no path into a ``GradeResult`` at all.

It follows ``ExactMatchGrader``'s injected-callable pattern: a benchmark-
neutral ``predictor`` extracts the harness prediction from the executed
sample, so this module owns grading policy only, not benchmark projection.

The outcome mapping is the load-bearing discipline (ADR-0005/0008): an
authoritative ``resolved`` verdict is the ONLY thing that hard-gates, and an
operational failure (capability unavailable, infrastructure error) can never
become a task ``FAIL``:

- executor ``UNAVAILABLE`` -> ``GradeStatus.UNAVAILABLE``, ``hard_gate=False``
- executor ``ERROR`` -> ``GradeStatus.ERROR``, ``hard_gate=False``
- ``COMPLETED`` + ``resolved=True`` -> ``GradeStatus.PASS``, ``hard_gate=True``
- ``COMPLETED`` + ``resolved=False`` -> ``GradeStatus.FAIL``, ``hard_gate=True``
- an un-executed sample or a predictor failure -> ``UNAVAILABLE``/``ERROR``,
  never a fabricated verdict.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import JsonValue

from agentic_evalkit.benchmarks.harness import (
    HarnessExecutor,
    HarnessRequest,
    HarnessResult,
    HarnessStatus,
)
from agentic_evalkit.models import (
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
)

__all__ = ["HarnessGrader", "HarnessPredictor"]

#: Extracts the harness prediction payload from an executed sample. For
#: SWE-bench, a thin closure over ``SweBenchVerifiedAdapter.export_prediction``.
HarnessPredictor = Callable[[EvalSample, NormalizedExecutionResult], dict[str, JsonValue]]

#: Default authoritative-verification timeout (seconds). SWE-bench instances
#: build an image and run a test suite; 30 minutes is a safe per-instance cap.
_DEFAULT_TIMEOUT_SECONDS = 1800.0


class HarnessGrader:
    """Grades an executed sample by routing it through a ``HarnessExecutor``.

    Args:
        executor: The authoritative-verification boundary to query.
        predictor: Builds the harness prediction from ``(sample, execution)``.
        benchmark: Benchmark identifier stamped on every ``HarnessRequest``.
        name: Stable grader identifier reported on every ``GradeResult``.
        timeout_seconds: Per-request authoritative-verification timeout.
    """

    def __init__(
        self,
        *,
        executor: HarnessExecutor,
        predictor: HarnessPredictor,
        benchmark: str,
        name: str,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._executor = executor
        self._predictor = predictor
        self._benchmark = benchmark
        self._name = name
        self._timeout_seconds = timeout_seconds

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        now = datetime.now(UTC)
        if execution.status is not ExecutionStatus.COMPLETED:
            return self._result(
                sample,
                now,
                status=GradeStatus.UNAVAILABLE,
                score=None,
                hard_gate=False,
                evidence={"reason": "execution did not complete; nothing to verify"},
            )
        if execution.output is None:
            # A None output carrying an ``output_ref`` artifact was SPILLED by
            # the runner (a patch larger than the spill threshold), not empty.
            # That is a real, gradable output the harness grader cannot yet
            # recover, so surface an explicit ERROR rather than silently
            # counting a valid large patch as "capability unavailable" (Codex
            # review, P2). Genuinely-empty output stays UNAVAILABLE.
            if "output_ref" in execution.artifacts:
                return self._result(
                    sample,
                    now,
                    status=GradeStatus.ERROR,
                    score=None,
                    hard_gate=False,
                    evidence={
                        "reason": (
                            "execution output was spilled to artifact "
                            f"{execution.artifacts['output_ref']!r}; the harness grader "
                            "needs the inline patch. Raise the run's spill threshold or "
                            "keep the patch inline until spill-aware recovery lands."
                        )
                    },
                )
            return self._result(
                sample,
                now,
                status=GradeStatus.UNAVAILABLE,
                score=None,
                hard_gate=False,
                evidence={"reason": "execution produced no output; nothing to verify"},
            )
        try:
            prediction = self._predictor(sample, execution)
        except Exception as error:
            return self._result(
                sample,
                now,
                status=GradeStatus.ERROR,
                score=None,
                hard_gate=False,
                evidence={"reason": f"could not build harness prediction: {error}"},
            )

        request = HarnessRequest(
            benchmark=self._benchmark,
            sample_id=sample.sample_id,
            prediction=prediction,
            timeout_seconds=self._timeout_seconds,
        )
        harness_result = await self._executor.execute(request)
        return self._map_harness_result(sample, now, harness_result)

    def _map_harness_result(
        self, sample: EvalSample, now: datetime, harness_result: HarnessResult
    ) -> GradeResult:
        evidence: dict[str, JsonValue] = {
            "harness_status": harness_result.status.value,
            "harness_message": harness_result.message,
            "harness_evidence": harness_result.evidence,
        }
        if harness_result.status is HarnessStatus.UNAVAILABLE:
            return self._result(
                sample,
                now,
                status=GradeStatus.UNAVAILABLE,
                score=None,
                hard_gate=False,
                evidence=evidence,
            )
        if harness_result.status is HarnessStatus.ERROR:
            if harness_result.error is not None:
                evidence["harness_error"] = harness_result.error
            return self._result(
                sample,
                now,
                status=GradeStatus.ERROR,
                score=None,
                hard_gate=False,
                evidence=evidence,
            )
        # COMPLETED: a real, earned verdict -- the only branch that hard-gates.
        if harness_result.resolved is None:
            evidence["reason"] = "harness completed without a resolution verdict"
            return self._result(
                sample,
                now,
                status=GradeStatus.UNAVAILABLE,
                score=None,
                hard_gate=False,
                evidence=evidence,
            )
        resolved = harness_result.resolved
        return self._result(
            sample,
            now,
            status=GradeStatus.PASS if resolved else GradeStatus.FAIL,
            score=1.0 if resolved else 0.0,
            hard_gate=True,
            evidence=evidence,
        )

    def _result(
        self,
        sample: EvalSample,
        now: datetime,
        *,
        status: GradeStatus,
        score: float | None,
        hard_gate: bool,
        evidence: dict[str, JsonValue],
    ) -> GradeResult:
        return GradeResult(
            sample_id=sample.sample_id,
            grader=self._name,
            grader_type="harness",
            status=status,
            score=score,
            hard_gate=hard_gate,
            evidence=evidence,
            created_at=now,
        )
