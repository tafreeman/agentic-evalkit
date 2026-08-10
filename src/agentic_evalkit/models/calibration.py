"""The hand-labeled evidence a judge's calibration is measured from (ADR-0024).

A :class:`~agentic_evalkit.graders.calibration.CalibrationArtifact` records
how often a judge got the right answer. This module describes the input side
of that measurement: the examples where somebody already knows the right
answer, against which the judge is scored.

The distinction that matters here is *who decided*. ``candidate_output`` is
what some system under test produced, and ``label`` is what a human
concluded about it -- and the whole point of a calibration run is to find
out how often the judge's opinion matches that human's. So the label can
never come from a model, and it is deliberately not a field a judge or an
adapter can write.

This is a separate model from :class:`~agentic_evalkit.models.EvalSample`
rather than a reuse of it, for two reasons. ``EvalSample`` carries no
candidate output at all -- an output is produced later, by the target, and
arrives as a ``NormalizedExecutionResult`` -- and it carries no label,
because nothing in the ordinary evaluation path has one. It also requires
``source_digest`` and ``adapter``, which record which adapter converted
which dataset row; a row somebody labeled by hand came from neither, so
reusing the model would mean asking a human to invent two provenance values
that would then be false. See ADR-0024.
"""

from enum import StrEnum

from agentic_evalkit.models.base import FrozenModel


class CalibrationLabel(StrEnum):
    """What a human decided about one candidate output -- the ground truth.

    A named pair rather than a boolean, per ADR-0002's rule that a status
    crossing a wire is never a bare ``bool``. The practical difference is
    at the edge: ``{"label": true}`` in a hand-written file is ambiguous
    about what ``true`` is asserting (that the answer is good? that the
    judge should pass it? that the row is enabled?), and a typo silently
    flips the meaning of a calibration. ``"good"`` and ``"bad"`` say which,
    and anything else is rejected at parse time rather than quietly
    counted as the wrong class.

    - ``GOOD``: the candidate output really is a correct answer. A judge
      that says "pass" here is right (a true positive).
    - ``BAD``: the candidate output really is a wrong answer. A judge that
      says "fail" here is right (a true negative).
    """

    GOOD = "good"
    BAD = "bad"


class LabeledJudgeSample(FrozenModel):
    """One example with a known-correct answer, used to score a judge.

    Attributes:
        sample_id: A stable name for this example, so a count can be traced
            back to the rows that produced it.
        prompt: The original question or task the candidate output was an
            answer to. Passed to the judge as
            :attr:`~agentic_evalkit.graders.judge.JudgeRequest.prompt`.
        candidate_output: The answer being judged -- what a system under
            test produced (or what a human wrote to stand in for one).
        label: What a human concluded about ``candidate_output``: the
            ground truth the judge's verdict is scored against.
        reference: The known-correct answer, when there is a single simple
            one. Optional, and passed through to the judge unchanged --
            some judges compare against it, others ignore it. Matching
            :attr:`~agentic_evalkit.models.EvalSample.reference`, this is
            left unset when correctness depends on something more
            elaborate than a text match.
    """

    sample_id: str
    prompt: str
    candidate_output: str
    label: CalibrationLabel
    reference: str | None = None
