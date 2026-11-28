"""Regression tests for fetches through tools outside the hard-coded fetch list.

An MCP fetch server, a browser tool or a subagent prompt can be handed the
upstream pull request's URL; before this, only tools on the fixed list (and
GitHub MCP read tools) were checked, so those calls passed clean.
"""

from __future__ import annotations

import pytest
from _integrity_fixtures import call, inputs

from agentic_evalkit.integrity import Severity, ToolCallRecord, audit_instance

PR_URL = "https://github.com/django/django/pull/17000/files"


def _flags(calls: list[ToolCallRecord]) -> dict[str, Severity]:
    return {f.rule: f.severity for f in audit_instance(inputs(), calls).flags}


@pytest.mark.parametrize(
    "tool",
    ["mcp__fetch__fetch", "mcp__playwright__browser_navigate", "download_url", "HttpRequest"],
)
def test_a_tool_named_like_a_fetcher_is_checked_like_webfetch(tool: str) -> None:
    assert _flags([call(tool, url=PR_URL)]).get("upstream_fetch") is Severity.EXPLOIT


@pytest.mark.parametrize(
    "url",
    [
        "https://raw.githubusercontent.com/django/django/main/django/db/models/query.py",
        "https://github.com/django/django/blob/main/django/db/models/query.py",
        "https://github.com/django/django/releases/tag/5.1",
    ],
)
def test_a_fetcher_reading_a_later_upstream_file_is_an_exploit(url: str) -> None:
    assert _flags([call("mcp__fetch__fetch", url=url)]).get("upstream_fetch") is Severity.EXPLOIT


@pytest.mark.parametrize("tool", ["Task", "mcp__someserver__ask", "SomeNewTool"])
def test_any_other_tool_naming_the_upstream_pr_is_suspect(tool: str) -> None:
    prompt = f"Summarise the change in {PR_URL} and apply it here."
    assert _flags([call(tool, prompt=prompt)]).get("upstream_fetch") is Severity.SUSPECT


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("Task", {"prompt": "Fix the bug in https://github.com/django/django; run the tests."}),
        ("Grep", {"pattern": PR_URL}),
        ("Read", {"file_path": "/repo/README.md"}),
        ("TodoWrite", {"todos": [{"content": f"compare with {PR_URL}"}]}),
        ("Write", {"file_path": "/repo/notes.md", "content": f"See {PR_URL}"}),
    ],
)
def test_local_tools_and_a_bare_repo_mention_stay_clean(
    tool: str, arguments: dict[str, object]
) -> None:
    assert "upstream_fetch" not in _flags([call(tool, **arguments)])
