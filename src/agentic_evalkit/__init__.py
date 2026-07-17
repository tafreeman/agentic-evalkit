"""agentic-evalkit: reproducible, evidence-first grading for agentic systems.

This top-level module re-exports only a small, hand-picked set of names --
the minimum you need to wrap an AI system you want to test (``CallableTarget``
/ ``ExecutionTarget``), describe what a run should do (``DatasetRef`` /
``EvalRunManifest``), and hand it off to ``EvalRunner`` to actually execute
it. Everything else this package offers -- other kinds of targets, graders,
report renderers, dataset providers, benchmark adapters, statistics helpers
-- is still just one import away, but lives in its own subpackage instead
(for example ``agentic_evalkit.graders``, ``agentic_evalkit.reporters``),
each with its own curated list of exported names. See the "Python API"
section of README.md for a short usage example.
"""

from importlib.metadata import version

from agentic_evalkit.models import DatasetRef, EvalRunManifest
from agentic_evalkit.runner import EvalRunner
from agentic_evalkit.targets import CallableTarget, ExecutionTarget

__version__ = version("agentic-evalkit")

__all__ = [
    "CallableTarget",
    "DatasetRef",
    "EvalRunManifest",
    "EvalRunner",
    "ExecutionTarget",
    "__version__",
]
