"""Bridges that carry this package's validity controls into a host platform (ADR-0022).

Importing this module is free: it names the sinks and re-exports the shared
plumbing, but pulls in neither MLflow nor Langfuse. The exporters themselves
live in :mod:`agentic_evalkit.integrations.mlflow` and
:mod:`agentic_evalkit.integrations.langfuse`, and each imports its host
library only when actually called.

The registries below are the enforcement mechanism for the rule that matters
most here, and they are built the same way the reporters' pair is, for the
same reason. ``EXTERNAL_SINKS`` names every function in this package that
transmits data off the machine. The two sets beneath it partition that list
by *how* each one scrubs. None is derived from another -- that is the entire
point. A test comparing them (``tests/contract/test_integration_redaction.py``)
is only a real tripwire while every side is maintained independently: derive
one from another and the comparison passes by construction and can never
catch the mistake it exists to catch, which is somebody adding a third host
platform and forgetting that an export leaves the building.

The partition exists because "redact everything through one function" is not
achievable on every path, and pretending otherwise is how the rule gets
quietly broken. A function handed a whole :class:`~agentic_evalkit.models.EvalRunResult`
can and must route it through :func:`~agentic_evalkit.integrations.base.redact_for_export`.
A function the *host* calls per row -- a scorer, a gate -- never sees a run
at all, so there is nothing for that pass to operate on; it scrubs the
strings it is about to transmit with
:func:`~agentic_evalkit.reporters.redact_text` instead. Both are real
redaction and neither is optional. Listing a transmitting function in
neither set, or in both, fails CI: the first is the leak, the second means
nobody knows which guarantee actually holds.
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

#: Every entry point in this package that transmits data to a host platform,
#: as ``"<host>.<function>"``. Not only the whole-run exporters: a scorer
#: that attaches a rationale to a feedback object, and a gate that writes a
#: comment beside a score, are transmitting too, and leaving them off this
#: list is what let them go unscrubbed.
#:
#: Values are the dotted module path plus attribute name rather than the
#: imported callables themselves, deliberately: importing them here would
#: undo the lazy-import property the whole subpackage is built on, since
#: resolving the attribute is harmless but importing the module that defines
#: it is what must stay cheap.
EXTERNAL_SINKS: "Mapping[str, str]" = MappingProxyType(
    {
        "mlflow.log_eval_run": "agentic_evalkit.integrations.mlflow:log_eval_run",
        "mlflow.as_mlflow_scorer": "agentic_evalkit.integrations.mlflow:as_mlflow_scorer",
        "mlflow.calibration_gate": "agentic_evalkit.integrations.mlflow:calibration_gate",
        "langfuse.log_eval_run": "agentic_evalkit.integrations.langfuse:log_eval_run",
        "langfuse.score_with_calibration_gate": (
            "agentic_evalkit.integrations.langfuse:score_with_calibration_gate"
        ),
    }
)

#: The sinks handed a whole run, which call
#: :func:`~agentic_evalkit.integrations.base.redact_for_export` before
#: transmitting anything. Maintained BY HAND -- see this module's docstring
#: for why it must never be computed from :data:`EXTERNAL_SINKS`.
REDACTION_ROUTED_SINKS: frozenset[str] = frozenset(
    {
        "mlflow.log_eval_run",
        "langfuse.log_eval_run",
    }
)

#: The sinks the host calls per row, which never receive a run to redact and
#: so scrub the strings they transmit with
#: :func:`~agentic_evalkit.reporters.redact_text` instead. Also maintained by
#: hand, and deliberately disjoint from :data:`REDACTION_ROUTED_SINKS`: a
#: function belongs in exactly one, because which guarantee applies to it is
#: something a reader must be able to look up rather than infer.
TEXT_REDACTED_SINKS: frozenset[str] = frozenset(
    {
        "mlflow.as_mlflow_scorer",
        "mlflow.calibration_gate",
        "langfuse.score_with_calibration_gate",
    }
)

__all__ = [
    "EXTERNAL_SINKS",
    "REDACTION_ROUTED_SINKS",
    "TEXT_REDACTED_SINKS",
    "AuthorityLevel",
    "JudgeAuthority",
    "judge_authority",
    "redact_for_export",
    "require_dependency",
    "run_blocking",
]
