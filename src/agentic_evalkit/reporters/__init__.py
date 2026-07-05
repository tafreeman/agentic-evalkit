"""Portable JSON, JSONL, Markdown, and HTML reporters."""

from collections.abc import Mapping
from types import MappingProxyType

from agentic_evalkit.reporters.base import (
    DEFAULT_REDACTION_POLICY,
    RedactionPolicy,
    Reporter,
    apply_redaction,
)
from agentic_evalkit.reporters.html import HtmlReporter
from agentic_evalkit.reporters.json import JsonReporter
from agentic_evalkit.reporters.jsonl import JsonlReporter
from agentic_evalkit.reporters.markdown import MarkdownReporter

#: Canonical name -> reporter-type registry. Every CLI write boundary
#: (``cli/runs.py:write_canonical_report`` and the ``cli/reports.py:report``
#: command) selects reporters only through this immutable table, so every
#: persisted format is enumerable and a reporter that is not registered here
#: is neither selectable nor written (Story 2.2 / R-002).
REPORTER_FORMATS: Mapping[str, type[Reporter]] = MappingProxyType(
    {
        "json": JsonReporter,
        "jsonl": JsonlReporter,
        "markdown": MarkdownReporter,
        "html": HtmlReporter,
    }
)

#: The formats whose write boundaries apply :func:`apply_redaction` before
#: writing. Maintained BY HAND -- deliberately NOT derived from
#: :data:`REPORTER_FORMATS`, so the redaction-enumeration contract comparing
#: the two sets is falsifiable: registering a new format without consciously
#: adding it here (together with redaction routing at its write boundary)
#: fails CI instead of passing by construction.
REDACTION_ROUTED_FORMATS: frozenset[str] = frozenset({"json", "jsonl", "markdown", "html"})

__all__ = [
    "DEFAULT_REDACTION_POLICY",
    "REDACTION_ROUTED_FORMATS",
    "REPORTER_FORMATS",
    "HtmlReporter",
    "JsonReporter",
    "JsonlReporter",
    "MarkdownReporter",
    "RedactionPolicy",
    "Reporter",
    "apply_redaction",
]
