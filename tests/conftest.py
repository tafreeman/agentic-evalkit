"""Session-wide test setup.

``FORCE_COLOR`` is removed before any test module imports the CLI. Rich reads
it once, when ``agentic_evalkit.cli.app`` builds its module-level console, and
any value (including ``0``) makes tables, messages and warnings carry ANSI
codes. Tests that assert on plain text then fail for a reason unrelated to the
code under test: the 2026-09-22 audit's harness set it and saw 19 spurious
failures. JSON output no longer depends on this (see
``agentic_evalkit.cli.app.print_output``); human-readable output
legitimately does.
"""

from __future__ import annotations

import os

os.environ.pop("FORCE_COLOR", None)
