"""Bridges that carry this package's validity controls into a host platform (ADR-0022).

Importing this module is free: it names the sinks and re-exports the shared
plumbing, but pulls in neither MLflow nor Langfuse. The exporters themselves
live in :mod:`agentic_evalkit.integrations.mlflow` and
:mod:`agentic_evalkit.integrations.langfuse`, and each imports its host
library only when actually called.

The two registries below are the enforcement mechanism for the rule that
matters most here, and they are built the same way the reporters' pair is,
for the same reason. ``EXTERNAL_SINKS`` names every function in this package
that transmits run data off the machine. ``REDACTION_ROUTED_SINKS`` is a
hand-written list of those that scrub secrets first. Neither is derived from
the other -- that is the entire point. A test comparing them
(``tests/contract/test_integration_redaction.py``) is only a real tripwire
while both sides are maintained independently: derive one from the other and
the comparison passes by construction and can never catch the mistake it
exists to catch, which is somebody adding a third host platform and
forgetting that an export leaves the building.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING

from agentic_evalkit.integrations.base import (
    AuthorityLevel,
    JudgeAuthority,
    judge_authority,
    redact_for_export,
    require_dependency,
    run_blocking,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Every export entry point in this package, as ``"<host>.<function>"``.
#: Values are the dotted module path plus attribute name rather than the
#: imported callables themselves, deliberately: importing them here would
#: undo the lazy-import property the whole subpackage is built on, since
#: resolving the attribute is harmless but importing the module that defines
#: it is what must stay cheap.
EXTERNAL_SINKS: "Mapping[str, str]" = MappingProxyType(
    {
        "mlflow.log_eval_run": "agentic_evalkit.integrations.mlflow:log_eval_run",
        "langfuse.log_eval_run": "agentic_evalkit.integrations.langfuse:log_eval_run",
    }
)

#: The sinks whose implementation calls
#: :func:`~agentic_evalkit.integrations.base.redact_for_export` before
#: transmitting anything. Maintained BY HAND -- see this module's docstring
#: for why it must never be computed from :data:`EXTERNAL_SINKS`.
REDACTION_ROUTED_SINKS: frozenset[str] = frozenset(
    {
        "mlflow.log_eval_run",
        "langfuse.log_eval_run",
    }
)

__all__ = [
    "EXTERNAL_SINKS",
    "REDACTION_ROUTED_SINKS",
    "AuthorityLevel",
    "JudgeAuthority",
    "judge_authority",
    "redact_for_export",
    "require_dependency",
    "run_blocking",
]
