"""The shared interface every grader must implement (design §9).

A "grader" is the piece of code that looks at what the AI produced and
decides pass/fail/score. Every grader implements a single async ``grade``
method: it takes a completed sample/execution pair and returns a
:class:`GradeResult`. The interface is "structural" (Python's ``Protocol``
+ ``runtime_checkable``) -- meaning any object with a matching ``grade``
method automatically counts as a ``Grader``, without needing to inherit
from a shared base class or explicitly declare that it implements this
interface. This matches how ``agentic_evalkit.datasets.base.DatasetProvider``
works, so neither this framework's own code nor a test's stand-in objects
ever need to inherit from a framework base class just to qualify.

Graders never run the AI system being evaluated themselves (design §9). By
the time a grader is called, something else has already run the AI and
packaged up what happened into a :class:`NormalizedExecutionResult`; the
grader's only job is to look at that result (and the original sample) and
decide the outcome.
"""

from typing import Protocol, runtime_checkable

from agentic_evalkit.models import EvalSample, GradeResult, NormalizedExecutionResult


@runtime_checkable
class Grader(Protocol):
    """The grading boundary: anything with an async ``grade(sample, execution)`` method (§9)."""

    async def grade(
        self, sample: EvalSample, execution: NormalizedExecutionResult
    ) -> GradeResult: ...
