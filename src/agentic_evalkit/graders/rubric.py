"""Immutable rubric contracts and validation rules (design §9, plan Task 10 Step 4).

"Rubrics use atomic criteria with stable IDs, evidence requirements,
weights, hard-gate flags, and explicit handling of missing evidence. Broad
holistic scores are advisory only" (design §9). These models validate that
policy at construction time; they do not themselves grade anything (grading
a rubric against an execution is a judge/human concern, wired in later).
"""

import re
from typing import Literal

from pydantic import Field, model_validator

from agentic_evalkit.models.base import FrozenModel

# Phrases that signal a criterion is asking for a broad, holistic judgment
# rather than an atomic, checkable fact. Such criteria cannot be evidence-free
# hard gates (design §9: "Broad holistic scores are advisory only").
_BROAD_JUDGMENT_PATTERN = re.compile(
    r"\b(overall|in general|holistic|good response|bad response|quality of the response)\b",
    re.IGNORECASE,
)


class RubricCriterion(FrozenModel):
    """One atomic, independently checkable rubric criterion.

    Attributes:
        criterion_id: Stable identifier, unique within its ``Rubric``.
        description: What evidence-backed fact this criterion checks.
        scale: ``"binary"`` (met / not met) or ``"bounded"`` (a numeric
            range given by ``scale_min``/``scale_max``).
        requires_evidence: Whether a grade against this criterion must cite
            evidence. Criteria whose ``description`` reads as a broad
            holistic judgment must set this ``True`` (enforced by
            :meth:`_validate_policy`).
        weight: Non-negative contribution weight within the parent rubric.
        hard_gate: Whether failing this criterion alone fails the rubric.
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
        if self.scale == "bounded" and self.scale_min is not None and self.scale_max is not None:
            if self.scale_min >= self.scale_max:
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
