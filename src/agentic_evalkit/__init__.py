"""agentic-evalkit: reproducible, evidence-first grading for agentic systems.

Curated top-level re-exports: the smallest set of objects a Python
integration needs to wrap a system under test (``CallableTarget`` /
``ExecutionTarget``) and describe a run (``DatasetRef`` /
``EvalRunManifest``) before handing it to ``EvalRunner``. Everything else
-- additional targets, graders, reporters, dataset providers, benchmark
adapters, statistics -- stays one import away under its own subpackage
(for example ``agentic_evalkit.graders``, ``agentic_evalkit.reporters``),
each with its own curated ``__all__``. See README.md's "Python API"
section for a short example.
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
