"""A working, deterministic ``JudgeClient`` you can use for demos and to get
the quickstart running.

A "judge" here is a component that grades an AI system's output by giving a
verdict on it -- typically by asking another AI model, "is this answer
good?". ``agentic_evalkit.graders.judge`` fully defines what a judge must
look like and how it earns the right to block a release: the
``JudgeClient`` protocol (the interface a judge must implement), the
``JudgeGrader`` that calls a judge and applies the rules, and
``CalibrationArtifact`` (the record of how accurate a judge has been
measured to be). But before this module existed, nothing in the package
actually implemented that interface for real use -- the only
implementations were fake stand-ins used solely inside the test suite
(``tests/unit/graders/``), not meant to be reused elsewhere. That meant the
README's headline feature, "calibrated judges" (judges whose accuracy has
been measured and shown to be trustworthy), wasn't actually reachable from
the command line: there was no judge a user could name in their manifest
(the config file describing an eval run) that would actually work.

``ReferenceJudgeClient`` fixes that, the same way
``agentic_evalkit.examples.zero_target`` fixes the equivalent problem for
systems under test: it is deliberately *not* a real, AI-model-backed judge.
It never calls out to an AI provider, so it lets you run the entire
judge-grading pipeline -- picking a judge from the manifest,
``JudgeGrader``'s retry logic and its check for order-of-answer bias, and
writing the judge's findings into the final report in the library's
standard format -- start to finish, with no API key and no network
connection needed. Its verdicts say nothing about how good a real AI judge
would be, and must never be treated as proof that a system passed a
"calibrated judge" check: this class is permanently wired up with no
calibration data at all (``JudgeGrader(calibration=None, ...)``). Both the
project's design spec (design §9) and ``JudgeGrader``'s own code already
guarantee that a judge with no calibration data can only ever produce an
informational ("advisory") result, never one that can block a release, no
matter what the caller passes for the ``gate`` argument.
"""

from __future__ import annotations

import hashlib

from agentic_evalkit.graders.judge import JudgeRequest, JudgeResponse

__all__ = ["ReferenceJudgeClient"]

#: This fingerprint is fixed once, ahead of time, and reused for every call --
#: it is never computed per-request. Elsewhere in this project (design §9), a
#: judge's ``fingerprint`` identifies exactly which judge is running: which
#: model, with which prompt. That matters because ``JudgeGrader`` checks the
#: fingerprint against any calibration data, to make sure that data was
#: actually measured on *this* judge and not some other one. A stand-in judge
#: like this one has no model and no prompt to configure -- it only ever
#: behaves one way -- so it only ever needs the one fingerprint value.
_FINGERPRINT = (
    "sha256:"
    + hashlib.sha256(
        b"agentic_evalkit.examples.reference_judge.ReferenceJudgeClient:v1"
    ).hexdigest()
)


def _normalize(text: str) -> str:
    """Collapse whitespace and casing so comparison ignores incidental formatting."""
    return " ".join(text.split()).casefold()


class ReferenceJudgeClient:
    """Grades a candidate answer by checking whether the reference text turns
    up inside it, once whitespace and letter case are normalized away.

    "Candidate output" is the answer produced by the system being evaluated;
    "reference" is the expected correct answer supplied by the dataset. The
    verdict is ``"pass"`` when the (whitespace-collapsed, case-folded)
    reference text appears anywhere inside the (whitespace-collapsed,
    case-folded) candidate output, and ``"fail"`` otherwise. A sample with no
    reference answer at all can't be checked this way, so this judge abstains
    (declines to give a verdict) instead of guessing.

    This judge completely ignores ``JudgeRequest.metadata``, including the
    ``{"reversed": True}`` flag ``JudgeGrader`` sends as a probe for
    "position bias" -- a known failure mode where a judge's verdict changes
    depending on which order two things being compared are shown in.
    Because this judge's verdict depends only on ``candidate_output`` and
    ``reference``, there is no "which one came first" for a plain substring
    check to be sensitive to, so it is automatically, correctly immune to
    that kind of bias.
    """

    fingerprint = _FINGERPRINT

    async def judge(self, request: JudgeRequest) -> JudgeResponse:
        if request.reference is None:
            return JudgeResponse(
                fingerprint=self.fingerprint,
                verdict="abstain",
                score=None,
                parse_ok=True,
                abstained=True,
            )
        reference = _normalize(request.reference)
        candidate = _normalize(request.candidate_output)
        matched = reference != "" and reference in candidate
        return JudgeResponse(
            fingerprint=self.fingerprint,
            verdict="pass" if matched else "fail",
            score=1.0 if matched else 0.0,
            parse_ok=True,
            abstained=False,
        )
