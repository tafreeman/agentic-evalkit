"""A packaged, deterministic ``JudgeClient`` for demos and quickstart wiring.

``agentic_evalkit.graders.judge`` defines the calibrated-judge contract
(``JudgeClient`` protocol, ``JudgeGrader``, ``CalibrationArtifact``) in full,
but before this module existed there was no concrete, importable
``JudgeClient`` implementation anywhere in the package -- only test doubles
private to ``tests/unit/graders/``. That made the README's "calibrated
judges" headline feature unreachable from the CLI/quickstart: there was
nothing a manifest's ``grader`` field could name that would actually
construct a working judge.

``ReferenceJudgeClient`` closes that gap the same way
``agentic_evalkit.examples.zero_target`` closes the analogous gap for
execution targets: it is deliberately *not* a real model judge. It never
calls an LLM provider, so it makes the judge-grading pipeline -- manifest
selection, ``JudgeGrader``'s retry/position-bias machinery, and canonical
reporting of judge evidence -- runnable end to end without an API key or
network access. Its verdicts carry no information about real judge quality
and must never be read as evidence a system passed a "calibrated judge"
check: it is wired in permanently uncalibrated (``JudgeGrader(calibration=None,
...)``), which design §9 and ``JudgeGrader`` itself already enforce as
advisory-only, never hard-gating, regardless of the ``gate`` argument a
caller passes.
"""

from __future__ import annotations

import hashlib

from agentic_evalkit.graders.judge import JudgeRequest, JudgeResponse

__all__ = ["ReferenceJudgeClient"]

#: A fixed, content-derived fingerprint (never per-request): design §9 treats
#: a judge's ``fingerprint`` as the *live judge identity* (model + prompt
#: configuration), which a stand-in, config-free client has exactly one of.
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
    """Scores a candidate output by normalized substring containment of the reference.

    A verdict of ``"pass"`` means the (whitespace-collapsed, case-folded)
    reference text appears as a substring of the candidate output; anything
    else is ``"fail"``. A sample with no ``reference`` at all cannot be
    judged this way and abstains rather than guessing.

    Ignores ``JudgeRequest.metadata`` entirely (including the
    ``{"reversed": True}`` position-bias probe ``JudgeGrader`` sends): its
    verdict depends only on ``candidate_output``/``reference``, so it is
    trivially, correctly immune to option-order position bias -- there is no
    "option order" in a containment check to be biased by.
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
