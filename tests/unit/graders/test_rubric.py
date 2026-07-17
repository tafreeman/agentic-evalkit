"""Tests for :mod:`agentic_evalkit.graders.rubric` (plan Task 10 Step 4).

``Rubric`` and ``RubricCriterion`` are just data models: they check that a
rubric is well-formed when it's built, but they don't do any grading
themselves. Design §9 states the policy they enforce: "atomic criteria with
stable IDs, evidence requirements, weights, hard-gate flags, and explicit
handling of missing evidence. Broad holistic scores are advisory only." In
plain terms: a rubric should be made of small, specific, individually
checkable items (not one vague "is this good overall?" question) -- and a
criterion that IS just a vague, big-picture judgment call is never allowed
to skip its evidence requirement or single-handedly fail the whole rubric.
"""

import pytest
from pydantic import ValidationError

from agentic_evalkit.graders.rubric import Rubric, RubricCriterion


def _criterion(**overrides: object) -> RubricCriterion:
    defaults: dict[str, object] = {
        "criterion_id": "factual-accuracy",
        "description": "The answer states the correct numeric result.",
        "scale": "binary",
        "requires_evidence": True,
        "weight": 1.0,
        "hard_gate": False,
    }
    defaults.update(overrides)
    return RubricCriterion.model_validate(defaults)


def test_criterion_accepts_binary_scale() -> None:
    criterion = _criterion(scale="binary")
    assert criterion.scale == "binary"


def test_criterion_accepts_bounded_scale() -> None:
    criterion = _criterion(scale="bounded", scale_min=0.0, scale_max=5.0)
    assert criterion.scale == "bounded"
    assert criterion.scale_min == 0.0
    assert criterion.scale_max == 5.0


def test_criterion_is_frozen() -> None:
    criterion = _criterion()
    with pytest.raises(ValidationError):
        criterion.weight = 2.0  # type: ignore[misc]


def test_criterion_rejects_negative_weight() -> None:
    with pytest.raises(ValidationError, match="weight"):
        _criterion(weight=-0.5)


def test_broad_criterion_without_evidence_requirement_is_rejected() -> None:
    """A criterion whose description reads like a vague, big-picture opinion
    (design §9: "Broad holistic scores are advisory only") is required to
    set requires_evidence=True. In other words: if a criterion is really
    just asking "overall, is this good?", it must still require evidence --
    it can't be graded on a bare, unbacked "yes" or "no".
    """
    with pytest.raises(ValidationError, match="evidence"):
        _criterion(
            criterion_id="overall-quality",
            description="Overall, is this a good response?",
            requires_evidence=False,
        )


def test_rubric_rejects_duplicate_criterion_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        Rubric(
            rubric_id="quality@1",
            criteria=(
                _criterion(criterion_id="dup"),
                _criterion(criterion_id="dup", description="A second, distinct criterion."),
            ),
        )


def test_rubric_rejects_weights_summing_to_zero() -> None:
    with pytest.raises(ValidationError, match="weight"):
        Rubric(
            rubric_id="quality@1",
            criteria=(
                _criterion(criterion_id="a", weight=0.0),
                _criterion(criterion_id="b", weight=0.0),
            ),
        )


def test_rubric_accepts_valid_criteria() -> None:
    rubric = Rubric(
        rubric_id="quality@1",
        criteria=(
            _criterion(criterion_id="a", weight=0.5),
            _criterion(criterion_id="b", weight=0.5, hard_gate=True),
        ),
    )
    assert len(rubric.criteria) == 2
    assert rubric.criteria[1].hard_gate is True
