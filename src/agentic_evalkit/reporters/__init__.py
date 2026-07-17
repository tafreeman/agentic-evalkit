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

#: Maps each format's official name (like "json" or "html") to the reporter
#: class that implements it. Every place in the CLI that writes a report --
#: both ``cli/runs.py:write_canonical_report`` and the
#: ``cli/reports.py:report`` command -- looks up which reporter to use
#: through this one table, instead of importing or hard-coding a specific
#: reporter class. That means the list of supported formats can always be
#: read off this table (nothing is hidden elsewhere), and a reporter that
#: isn't registered here simply can't be chosen or written to disk -- there's
#: no way around it (tracked as Story 2.2 / requirement R-002).
REPORTER_FORMATS: Mapping[str, type[Reporter]] = MappingProxyType(
    {
        "json": JsonReporter,
        "jsonl": JsonlReporter,
        "markdown": MarkdownReporter,
        "html": HtmlReporter,
    }
)

#: The set of format names whose write path calls :func:`apply_redaction`
#: (the secret-scrubbing pass) before writing the file to disk.
#:
#: This list is written out by hand here, deliberately NOT computed from
#: :data:`REPORTER_FORMATS` above (e.g. by just taking its keys). That's on
#: purpose: an automated test compares this set against
#: ``REPORTER_FORMATS`` to make sure every registered format is accounted
#: for. If this set were instead auto-derived from ``REPORTER_FORMATS``,
#: that comparison would always trivially pass -- it could never catch
#: anything, because both sides would always be identical by definition.
#: Keeping the two lists independent means that if someone adds a new
#: report format and forgets to also wire up redaction for it, the test
#: suite (CI) catches the mismatch and fails, instead of silently shipping
#: a report format that never gets its secrets scrubbed.
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
