"""Example "target" (system under test) and "grader" (scoring) implementations
packaged with the library so the CLI's quickstart flow has something runnable
out of the box. They exist purely for demos and onboarding -- not as real
benchmark baselines -- so their scores should never be read as evidence of how
good any actual system is.
"""

from agentic_evalkit.examples.reference_judge import ReferenceJudgeClient
from agentic_evalkit.examples.zero_target import zero_target

__all__ = ["ReferenceJudgeClient", "zero_target"]
