"""Packaged demo targets/graders used by CLI quickstart flows, not benchmark baselines."""

from agentic_evalkit.examples.reference_judge import ReferenceJudgeClient
from agentic_evalkit.examples.zero_target import zero_target

__all__ = ["ReferenceJudgeClient", "zero_target"]
