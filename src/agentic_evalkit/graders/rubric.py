"""Data models for a rubric -- the checklist of specific things a judge or human should check.

Design §9, plan Task 10 Step 4.

Design §9 lays out the policy this module encodes: "Rubrics use atomic
criteria with stable IDs, evidence requirements, weights, hard-gate flags,
and explicit handling of missing evidence. Broad holistic scores are
advisory only." In plain terms: a rubric should be built out of small,
specific, individually checkable items (e.g. "does the answer cite its
source"), each with a stable ID, a weight, and a flag saying whether
failing it alone should fail the whole rubric -- rather than one vague "is
this a good answer overall?" question that no one can check objectively.
These classes only validate that a rubric is well-formed *when it's
constructed*; they don't do any grading themselves -- actually scoring a
rubric against something an AI produced is the job of a judge (or a
human), wired in elsewhere.
"""

import re
from typing import Literal

from pydantic import Field, model_validator

from agentic_evalkit.models.base import FrozenModel

# Phrases that suggest a criterion is really asking for a vague, big-picture
# opinion (e.g. "is this a good response overall?") rather than checking one
# specific, checkable fact. A criterion like that isn't allowed to skip the
# evidence requirement or single-handedly fail the whole rubric (design §9:
# "Broad holistic scores are advisory only").
_BROAD_JUDGMENT_PATTERN = re.compile(
    r"\b(overall|in general|holistic|good response|bad response|quality of the response)\b",
    re.IGNORECASE,
)


class RubricCriterion(FrozenModel):
    """One single, specific, checkable item within a larger rubric.

    Attributes:
        criterion_id: A stable name for this criterion, unique within its
            parent ``Rubric`` (e.g. ``"cites_source"``).
        description: The specific, evidence-backed fact this criterion
            checks -- written so it's clear exactly what would make it pass
            or fail.
        scale: ``"binary"`` (simply met / not met) or ``"bounded"`` (scored
            somewhere within a numeric range given by
            ``scale_min``/``scale_max``).
        requires_evidence: Whether a grade against this criterion has to
            point to specific supporting evidence, rather than just stating
            a bare verdict. Any criterion whose ``description`` reads like a
            vague, big-picture judgment (see ``_BROAD_JUDGMENT_PATTERN``
            above) is required to set this to ``True`` -- enforced
            automatically by :meth:`_validate_policy` below.
        weight: How much this criterion contributes to the parent rubric's
            overall score, relative to the other criteria. Must be zero or
            positive.
        hard_gate: If ``True``, failing this one criterion fails the whole
            rubric, no matter how well everything else scores.
    """

    criterion_id: str
    description: str
    scale: Literal["binary", "bounded"] = "binary"
    scale_min: float | None = None
    scale_max: float | None = None
    requires_evidence: bool = True
    weight: float = 1.0
    hard_gate: bool = False

    @model_validator(mode="after")
    def _validate_policy(self) -> "RubricCriterion":
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")
        if self.scale == "bounded" and (self.scale_min is None or self.scale_max is None):
            raise ValueError("bounded scale requires scale_min and scale_max")
        if (
            self.scale == "bounded"
            and self.scale_min is not None
            and self.scale_max is not None
            and self.scale_min >= self.scale_max
        ):
            raise ValueError("scale_min must be less than scale_max")
        if not self.requires_evidence and _BROAD_JUDGMENT_PATTERN.search(self.description):
            raise ValueError(
                "broad holistic criteria (matched a holistic-judgment phrase in "
                f"description={self.description!r}) must set requires_evidence=True; "
                "design §9 treats broad holistic scores as advisory only"
            )
        return self


class Rubric(FrozenModel):
    """An ordered collection of :class:`RubricCriterion` with a stable ID."""

    rubric_id: str
    criteria: tuple[RubricCriterion, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_criteria(self) -> "Rubric":
        seen_ids: set[str] = set()
        for criterion in self.criteria:
            if criterion.criterion_id in seen_ids:
                raise ValueError(f"duplicate criterion_id: {criterion.criterion_id!r}")
            seen_ids.add(criterion.criterion_id)
        if self.criteria and sum(criterion.weight for criterion in self.criteria) == 0.0:
            raise ValueError("rubric criteria weights must not sum to zero")
        return self
