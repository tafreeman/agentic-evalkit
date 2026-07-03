"""Grader protocol (design §9).

Every grader implements a single async ``grade`` method that consumes a
completed sample/execution pair and returns a :class:`GradeResult`. The
protocol is structural (``Protocol`` + ``runtime_checkable``), matching
``agentic_evalkit.datasets.base.DatasetProvider``, so host code and test
doubles never need to inherit a framework base class.

Graders never execute targets themselves (design §9); they consume an
already-normalized :class:`NormalizedExecutionResult`.
"""

from typing import Protocol, runtime_checkable

from agentic_evalkit.models import EvalSample, GradeResult, NormalizedExecutionResult


@runtime_checkable
class Grader(Protocol):
    """The grading boundary (design §9)."""

    async def grade(
        self, sample: EvalSample, execution: NormalizedExecutionResult
    ) -> GradeResult: ...
