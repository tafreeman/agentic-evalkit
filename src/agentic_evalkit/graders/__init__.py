"""This package's graders: the checks that decide whether an AI's answer passes.

Includes deterministic ("objective") checks like exact-match, a rubric
system for scoring named criteria, AI-judge-based grading, a "composite"
grader that combines several of these into one verdict, and the
grounded-citation grader that checks an answer's citations are real and
faithful to their source.
"""

from agentic_evalkit.graders.base import Grader
from agentic_evalkit.graders.composite import CompositeGrader, SchemaGrader, WeightedGrader
from agentic_evalkit.graders.contamination import (
    canary_leak_evidence,
    find_canary_leaks,
    normalize_for_containment,
)
from agentic_evalkit.graders.exact import ExactMatchGrader
from agentic_evalkit.graders.grounding import (
    GRADING_SCOPE,
    MIN_SUBSTANTIVE_QUOTE_TOKENS,
    GroundedCitationGrader,
    RubricBoundJudgeClient,
    build_grounded_citation_grader,
    build_grounding_rubric,
)
from agentic_evalkit.graders.harness import HarnessGrader, HarnessPredictor
from agentic_evalkit.graders.judge import (
    CalibrationArtifact,
    JudgeClient,
    JudgeGrader,
    JudgeRequest,
    JudgeResponse,
    JudgeResponseStatus,
)
from agentic_evalkit.graders.rubric import Rubric, RubricCriterion

__all__ = [
    "GRADING_SCOPE",
    "MIN_SUBSTANTIVE_QUOTE_TOKENS",
    "CalibrationArtifact",
    "CompositeGrader",
    "ExactMatchGrader",
    "Grader",
    "GroundedCitationGrader",
    "HarnessGrader",
    "HarnessPredictor",
    "JudgeClient",
    "JudgeGrader",
    "JudgeRequest",
    "JudgeResponse",
    "JudgeResponseStatus",
    "Rubric",
    "RubricBoundJudgeClient",
    "RubricCriterion",
    "SchemaGrader",
    "WeightedGrader",
    "build_grounded_citation_grader",
    "build_grounding_rubric",
    "canary_leak_evidence",
    "find_canary_leaks",
    "normalize_for_containment",
]
