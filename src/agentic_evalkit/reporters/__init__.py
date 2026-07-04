"""Portable JSON, JSONL, Markdown, and HTML reporters."""

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

#: Canonical name -> reporter-type registry. The write boundary selects a
#: reporter only through this table, so every persisted format is enumerable
#: and a newly added reporter that is not registered here is neither selectable
#: nor written (Story 2.2 / R-002).
REPORTER_FORMATS: dict[str, type[Reporter]] = {
    "json": JsonReporter,
    "jsonl": JsonlReporter,
    "markdown": MarkdownReporter,
    "html": HtmlReporter,
}

#: The formats the canonical write boundary routes through
#: :func:`apply_redaction`. Derived from :data:`REPORTER_FORMATS` so the two can
#: never drift: a format that is registered but not redaction-routed would fail
#: the redaction-enumeration contract.
REDACTION_ROUTED_FORMATS: frozenset[str] = frozenset(REPORTER_FORMATS)

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
