"""Turns the verdict from an external test harness into a GradeResult (ADR-0014).

A "harness" here means an outside, official tool that actually runs code and
tests to produce a real, authoritative pass/fail verdict -- as opposed to an
AI's opinion about whether something looks correct. For example, SWE-bench
(a benchmark built from real GitHub issues) has an official harness that
applies the AI's proposed code patch, actually runs the project's test
suite, and reports whether the issue was genuinely "resolved" or not.

Before ``HarnessGrader`` existed, there was no way to plug that kind of
outside, authoritative verdict into this framework's grading boundary
(every grader implements ``Grader.grade(sample, execution) -> GradeResult``
-- see ``base.py``) and get a proper ``GradeResult`` back out of it. This
class is that missing bridge, between the grading boundary and an
authoritative :class:`~agentic_evalkit.benchmarks.harness.HarnessExecutor`
(the object that actually knows how to talk to a specific harness).

It follows the same pattern ``ExactMatchGrader`` uses: rather than
importing benchmark-specific code directly, this module accepts an injected
``predictor`` function from the caller that knows how to build the
benchmark's expected input format (its "prediction") from what the AI
produced. That keeps this module focused purely on grading policy -- how to
turn a harness's verdict into a ``GradeResult`` -- without needing to know
anything about any specific benchmark's format.

The mapping from the harness's outcome to a grading result below is one of
the most important rules in this file (ADR-0005/0008): only a genuine,
authoritative "resolved" verdict from the harness is allowed to hard-gate
(force a failure that can't be averaged away by other good scores). An
operational failure -- meaning the harness itself couldn't run, or hit an
infrastructure problem, as opposed to the AI's actual work being judged
wrong -- can never be turned into a task failure. Concretely:

- harness reports ``UNAVAILABLE`` (it couldn't run at all) ->
  ``GradeStatus.UNAVAILABLE``, ``hard_gate=False``
- harness reports ``ERROR`` (it hit an infrastructure problem) ->
  ``GradeStatus.ERROR``, ``hard_gate=False``
- harness ``COMPLETED`` and found the issue ``resolved=True`` ->
  ``GradeStatus.PASS``, ``hard_gate=True``
- harness ``COMPLETED`` and found the issue ``resolved=False`` ->
  ``GradeStatus.FAIL``, ``hard_gate=True``
- the sample was never executed, or building the harness's expected input
  failed -> ``UNAVAILABLE``/``ERROR``, never a made-up verdict.
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

#: A function that builds the harness's expected input (its "prediction")
#: from an executed sample. For SWE-bench specifically, this is a small
#: wrapper function around ``SweBenchVerifiedAdapter.export_prediction``.
HarnessPredictor = Callable[[EvalSample, NormalizedExecutionResult], dict[str, JsonValue]]

#: The default timeout (in seconds) to wait for the harness to produce a
#: real, authoritative verdict. SWE-bench's harness has to build a container
#: image and then run a full test suite for each instance, so 30 minutes is
#: a safe per-instance limit that gives it enough time to finish.
_DEFAULT_TIMEOUT_SECONDS = 1800.0


class HarnessGrader:
    """Grades an executed sample by sending it to an external harness and trusting its verdict.

    Args:
        executor: The object that actually knows how to talk to the
            external, authoritative harness (build the request, wait for
            the result).
        predictor: The function that builds the harness's expected input
            (its "prediction") from ``(sample, execution)`` -- see
            ``HarnessPredictor`` above.
        benchmark: The benchmark's name, attached to every
            ``HarnessRequest`` so the harness knows which benchmark's rules
            to apply.
        name: A stable label for this grader, recorded on every
            ``GradeResult`` so you can tell which grader produced it.
        timeout_seconds: How long to wait for the harness to produce its
            real, authoritative verdict before giving up.
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
            # A `None` output that still has an `output_ref` entry in
            # `execution.artifacts` doesn't mean "there's no output" -- it
            # means the real output was too large to keep inline and got
            # "spilled": written out to separate storage, with only a
            # reference left behind here. That's a real, gradable output
            # (e.g. a large code patch) that this grader simply can't reach
            # directly from here, so we report an explicit ERROR instead of
            # silently treating a perfectly valid large answer as "nothing
            # we're able to check" (a gap flagged in a code review by a tool
            # named Codex, priority P2). A genuinely empty output (no patch
            # was ever produced) still gets reported as UNAVAILABLE below.
            #
            # In the normal pipeline, ``EvalRunner`` always grades a sample
            # using its full inline output *before* it ever spills that
            # output to separate storage (ADR-0017) -- spilling only
            # happens afterwards, to the copy that gets saved to disk. So
            # this branch should never actually trigger for anything that
            # went through ``EvalRunner.run`` normally. It exists to
            # protect a different caller: something that calls ``grade()``
            # directly on a ``NormalizedExecutionResult`` that was already
            # spilled and loaded back from a previous, saved run (for
            # example, a tool that re-grades old runs read off disk). That
            # caller can't recover the original inline patch from here, and
            # needs to re-grade using the original, unspilled execution
            # data instead.
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
                            "needs the inline patch. This should not happen inside "
                            "EvalRunner.run, which grades before spilling -- re-grade "
                            "from the original, unspilled execution instead."
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
        # COMPLETED means the harness actually finished running and produced
        # a genuine, earned verdict -- this is the only branch below that's
        # allowed to hard-gate (force a failure that other good scores
        # can't average away).
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
