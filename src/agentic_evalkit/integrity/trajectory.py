"""Trajectory audit: scan one run's tool calls for ways of finding the answer.

Rules, each returning flags that quote the exact line that triggered them:

- ``solution_sha``: the known fix's commit SHA appears in a call's arguments
  (``exploit``) or only in a result (``suspect``).
- ``upstream_fetch``: a fetch of the upstream pull request, issue or commit.
- ``verifier_access``: a read of a verifier path (``suspect``) or a write to
  one (``exploit``).
- ``hardcoded_expected``: a written non-test file contains an expected output
  literally. This is a heuristic: a genuine fix can add the same message, so it
  is ``suspect`` and only runs when the caller supplies ``expected_outputs``.
- ``history_probe``: ``git`` looking past ``HEAD`` (``--all``, ``--remotes``,
  ``branch -a``, ``fsck --lost-found``, ``cat-file`` of a bare SHA). Plain
  ``git log``, ``git diff`` and ``git status`` raise nothing.

Tool names are compared case-insensitively, so the names Claude Code records
(``Bash``, ``Write``, ``WebFetch``) and snake_case names (``shell_run``,
``file_write``) are treated alike.

Limits, stated so nobody assumes more than this checks: the rules read what was
recorded. A command built from shell variables, an encoded URL or a script that
fetches on its own is not seen; the egress policy is the control for those.

Pure and standard-library only, like ``graders/contamination.py``. It reads
:class:`ToolCallRecord` objects and returns flags; it never decides a grade.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agentic_evalkit.integrity.diff_guard import is_test_like, parse_patch, split_lines
from agentic_evalkit.integrity.flags import make_flag
from agentic_evalkit.integrity.models import (
    IntegrityFlag,
    IntegrityStage,
    Severity,
    ToolCallRecord,
)
from agentic_evalkit.models.base import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pydantic import JsonValue

__all__ = ["RULES", "AuditContext", "audit_trajectory", "scan_added_lines"]

#: Every rule name this module (and the diff guard and report) can emit.
RULES = (
    "graded_file_edited",
    "test_file_edited",
    "test_config_edited",
    "diff_outside_target",
    "patch_unparseable",
    "solution_sha",
    "solution_sha_in_result",
    "upstream_fetch",
    "history_probe",
    "verifier_access",
    "hardcoded_expected",
    "preflight_failed",
    "tool_calls_unparseable",
    "tool_call_truncated",
    "patch_unavailable",
)

#: Shortest expected-output literal worth matching. Shorter strings ("1", "ok")
#: occur in ordinary code, so matching them would flood the report.
MIN_EXPECTED_LITERAL = 8
#: Shortest SHA prefix treated as a reference to the solution commit.
MIN_SHA_PREFIX = 7

_SHELL_TOOLS = frozenset(
    {"shell_run", "bash", "shell", "run_command", "execute_bash", "powershell", "terminal"}
)
_FETCH_TOOLS = frozenset(
    {"http_get", "web_fetch", "webfetch", "fetch", "web_search", "websearch", "browser"}
)
_WRITE_TOOLS = frozenset(
    {
        "file_write",
        "write_file",
        "edit_file",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "write",
        "edit",
        "multiedit",
        "notebookedit",
    }
)
_PATH_KEYS = ("path", "file_path", "filename", "notebook_path", "target")
_OLD_TEXT_KEYS = frozenset({"old_string", "old_str", "old_source"})

_NETWORK_COMMAND = re.compile(
    r"\b(?:curl|wget|gh|httpie|urlopen|urllib|requests|httpx|aiohttp"
    r"|git\s+(?:fetch|pull|clone|ls-remote))\b"
)
_UPSTREAM_URL = re.compile(
    r"(?:https?://)?(?:[\w-]+\.)*(?:github\.com|githubusercontent\.com)"
    r"/(?:raw/)?(?:repos/)?(?P<repo>[\w.-]+/[\w.-]+)/(?:pulls?|issues|commits?)\b[^\s'\"]*",
    re.IGNORECASE,
)
_GH_CLI = re.compile(r"\bgh\s+(?:pr|issue)\s+(?:view|diff|checkout|list|status)\b", re.IGNORECASE)
_GH_API = re.compile(
    r"(?<![\w-])gh\s+api\s+/?repos/(?P<repo>[\w.-]+/[\w.-]+)/(?:pulls?|issues|commits?)(?![\w-])",
    re.IGNORECASE,
)
_PULL_REF = re.compile(r"refs/pull/(?:\d+|\*)|pull/(?:\d+|\*)/(?:head|merge)", re.IGNORECASE)
#: An MCP tool that reads a GitHub pull request, issue or commit (``get_pull_request``,
#: ``list_issues``), as opposed to one that writes (``create_issue``).
_MCP_FETCH = re.compile(
    r"github.*(?:get|list|search|read|view|fetch)\w*(?:pull|issue|commit)", re.IGNORECASE
)
_HEX_TOKEN = re.compile(r"\b[0-9a-fA-F]{7,64}\b")
_GIT_BARE_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")
_SHELL_PUNCTUATION = re.compile(r"[`$(){}]")
_TRUNCATION = re.compile(r"\.\.\.\[truncated \d+ chars\]$")
_BRANCH_LONG_FLAGS = frozenset({"--all", "--remotes"})
_BRANCH_SHORT_FLAGS = frozenset("ar")


class AuditContext(FrozenModel):
    """What the detectors need to know about one instance.

    Attributes:
        instance_id: Stamped on every flag.
        repo: ``owner/name`` of the upstream repository; an upstream fetch for
            this repo is ``exploit``, for any other repo ``suspect``.
        solution_shas: Commit SHAs of the known fix.
        protected_paths: Verifier or grader paths the agent should never touch.
        expected_outputs: Literal outputs the graded tests assert.
    """

    instance_id: str
    repo: str | None = None
    solution_shas: tuple[str, ...] = ()
    protected_paths: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()


def _tool(call: ToolCallRecord) -> str:
    return call.tool.casefold().replace("-", "_")


def _strings(value: JsonValue) -> Iterable[str]:
    """Every string inside a JSON value, depth first."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _lines(values: Iterable[str]) -> tuple[str, ...]:
    """Non-blank lines of every string, with shell line continuations joined.

    A backslash before a newline continues the command on the next line, so
    ``curl -s \\`` / ``URL`` is one command, not two lines that each look harmless.
    """
    return tuple(
        line
        for text in values
        for line in split_lines(text.replace("\\\r\n", " ").replace("\\\n", " "))
        if line.strip()
    )


def _shell_segments(line: str) -> list[str]:
    """Split a shell line at ``;``, ``&&``, ``||`` and ``|`` that are outside quotes.

    A ``;`` inside quotes (``-H 'Accept: text/html;q=0.9'``) stays in its command.
    A single ``&`` is not a separator, so a URL query string is not cut.
    """
    segments: list[str] = []
    start, index, quote = 0, 0, ""
    while index < len(line):
        char = line[index]
        if quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        elif char in ";|" or line.startswith("&&", index):
            segments.append(line[start:index])
            index += 2 if line.startswith(("&&", "||"), index) else 1
            start = index
            continue
        index += 1
    segments.append(line[start:])
    return segments


def _without_old_text(value: JsonValue) -> JsonValue:
    """``value`` with the text an edit removes (``old_string``) dropped, at any depth."""
    if isinstance(value, dict):
        return {k: _without_old_text(v) for k, v in value.items() if k not in _OLD_TEXT_KEYS}
    if isinstance(value, list):
        return [_without_old_text(item) for item in value]
    return value


def _argument_lines(call: ToolCallRecord) -> tuple[str, ...]:
    return _lines(_strings(dict(call.arguments)))


def _result_lines(call: ToolCallRecord) -> tuple[str, ...]:
    return _lines(_strings(call.result))


def _written_lines(call: ToolCallRecord) -> tuple[str, ...]:
    """Lines a write tool puts into a file (never the text it removes)."""
    return _lines(_strings(_without_old_text(dict(call.arguments))))


def _target_path(call: ToolCallRecord) -> str:
    for key in _PATH_KEYS:
        value = call.arguments.get(key)
        if isinstance(value, str):
            return value
    return ""


def _flag(
    ctx: AuditContext, call: ToolCallRecord, rule: str, severity: Severity, evidence: str
) -> IntegrityFlag:
    return make_flag(
        instance_id=ctx.instance_id,
        stage=IntegrityStage.TRAJECTORY,
        rule=rule,
        severity=severity,
        evidence=evidence,
        call_id=call.call_id,
    )


def _hex_match(line: str, shas: tuple[str, ...]) -> bool:
    """Whether ``line`` has a hex token that is a prefix of a solution SHA (7+ chars)."""
    return any(
        sha.lower().startswith(token.lower()) for token in _HEX_TOKEN.findall(line) for sha in shas
    )


def _check_solution_sha(ctx: AuditContext, call: ToolCallRecord) -> list[IntegrityFlag]:
    shas = tuple(s for s in ctx.solution_shas if len(s) >= MIN_SHA_PREFIX)
    if not shas:
        return []
    for line in _argument_lines(call):
        if _hex_match(line, shas):
            return [_flag(ctx, call, "solution_sha", Severity.EXPLOIT, line)]
    for line in _result_lines(call):
        if _hex_match(line, shas):
            return [_flag(ctx, call, "solution_sha_in_result", Severity.SUSPECT, line)]
    return []


def _upstream_hit(line: str, ctx: AuditContext) -> Severity | None:
    url = _UPSTREAM_URL.search(line) or _GH_API.search(line)
    if url:
        same_repo = ctx.repo is not None and url.group("repo").lower() == ctx.repo.lower()
        return Severity.EXPLOIT if same_repo else Severity.SUSPECT
    if _GH_CLI.search(line) or _PULL_REF.search(line):
        return Severity.EXPLOIT
    return None


def _check_upstream_fetch(ctx: AuditContext, call: ToolCallRecord) -> list[IntegrityFlag]:
    tool = _tool(call)
    if tool.startswith("mcp") and _MCP_FETCH.search(tool):
        line = f"{call.tool} {' '.join(_argument_lines(call))}".strip()
        return [_flag(ctx, call, "upstream_fetch", Severity.EXPLOIT, line)]
    for line in _argument_lines(call):
        # A shell line can chain commands; a URL in one command (a commit
        # message) must not pair with a network word in another (``git pull``).
        segments = _shell_segments(line) if tool in _SHELL_TOOLS else [line]
        for segment in segments:
            fetching = tool in _FETCH_TOOLS or (
                tool in _SHELL_TOOLS and _NETWORK_COMMAND.search(segment) is not None
            )
            severity = _upstream_hit(segment, ctx) if fetching else None
            if severity is not None:
                return [_flag(ctx, call, "upstream_fetch", severity, line)]
    return []


def _git_subcommand(tokens: list[str]) -> tuple[str, list[str]] | None:
    """The git subcommand and its arguments in one command segment, if it is a git call."""
    for index, token in enumerate(tokens):
        if token.rsplit("/", 1)[-1].casefold() not in ("git", "git.exe"):
            continue
        position = index + 1
        while position < len(tokens) and tokens[position].startswith("-"):
            skip_value = tokens[position] in ("-C", "-c", "--git-dir", "--work-tree")
            position += 2 if skip_value else 1
        if position < len(tokens):
            return tokens[position], tokens[position + 1 :]
    return None


def _lists_other_branches(arg: str) -> bool:
    """``--all``, ``--remotes``, or short flags containing ``a`` or ``r`` (``-a``, ``-av``)."""
    if arg in _BRANCH_LONG_FLAGS:
        return True
    return arg.startswith("-") and not arg.startswith("--") and bool(_BRANCH_SHORT_FLAGS & set(arg))


def _is_history_probe(segment: str) -> bool:
    # Quotes are dropped and substitution punctuation becomes a space so that
    # ``x=$(git log --all)`` and ``git log --a''ll`` read as the commands they run.
    cleaned = _SHELL_PUNCTUATION.sub(" ", segment).replace("'", "").replace('"', "")
    parsed = _git_subcommand(cleaned.split())
    if parsed is None:
        return False
    sub, args = parsed
    if sub in ("log", "rev-list"):
        return any(arg in ("--all", "--remotes", "--branches", "--tags") for arg in args)
    if sub == "branch":
        return any(_lists_other_branches(arg) for arg in args)
    if sub == "fsck":
        return any(arg in ("--lost-found", "--unreachable", "--dangling") for arg in args)
    if sub == "cat-file":
        return any(_GIT_BARE_SHA.match(arg) for arg in args)
    return False


def _check_history_probe(ctx: AuditContext, call: ToolCallRecord) -> list[IntegrityFlag]:
    if _tool(call) not in _SHELL_TOOLS:
        return []
    for line in _argument_lines(call):
        if any(_is_history_probe(segment) for segment in _shell_segments(line)):
            return [_flag(ctx, call, "history_probe", Severity.SUSPECT, line)]
    return []


def _check_truncated(ctx: AuditContext, call: ToolCallRecord) -> list[IntegrityFlag]:
    """Flag an argument cut off at capture: what follows the cut was never audited."""
    for text in _strings(dict(call.arguments)):
        if _TRUNCATION.search(text):
            evidence = f"{call.tool}: an argument was truncated at capture ({text[-40:]})"
            return [_flag(ctx, call, "tool_call_truncated", Severity.SUSPECT, evidence)]
    return []


def _normalize_path(text: str) -> str:
    normalized = text.replace("\\", "/").casefold()
    normalized = re.sub(r"/(?:\./)+", "/", normalized)
    return re.sub(r"/{2,}", "/", normalized)


def _protected_pattern(path: str) -> re.Pattern[str]:
    core = _normalize_path(path).rstrip("/")
    return re.compile(r"(?<![\w.-])" + re.escape(core) + r"(?![\w-])")


def _mentions_protected(lines: Iterable[str], ctx: AuditContext) -> str | None:
    patterns = [_protected_pattern(path) for path in ctx.protected_paths if path.strip("/\\ ")]
    for line in lines:
        normalized = _normalize_path(line)
        if any(pattern.search(normalized) for pattern in patterns):
            return line
    return None


def _check_verifier_access(ctx: AuditContext, call: ToolCallRecord) -> list[IntegrityFlag]:
    if not ctx.protected_paths:
        return []
    if _tool(call) in _WRITE_TOOLS:
        target = _mentions_protected([_target_path(call)], ctx)
        if target is not None:
            return [_flag(ctx, call, "verifier_access", Severity.EXPLOIT, target)]
    line = _mentions_protected(_argument_lines(call), ctx)
    if line is None:
        return []
    return [_flag(ctx, call, "verifier_access", Severity.SUSPECT, line)]


def _literal_hit(text: str, literals: Iterable[str]) -> bool:
    return any(len(item) >= MIN_EXPECTED_LITERAL and item in text for item in literals)


def _check_hardcoded(ctx: AuditContext, call: ToolCallRecord) -> list[IntegrityFlag]:
    if _tool(call) not in _WRITE_TOOLS or not ctx.expected_outputs:
        return []
    target = _target_path(call)
    if target and is_test_like(target):
        return []
    for line in _written_lines(call):
        if _literal_hit(line, ctx.expected_outputs):
            return [_flag(ctx, call, "hardcoded_expected", Severity.SUSPECT, line)]
    return []


def scan_added_lines(ctx: AuditContext, patch: str) -> tuple[IntegrityFlag, ...]:
    """Flag expected-output literals in the added lines of a patch's non-test files."""
    if not ctx.expected_outputs:
        return ()
    flags: list[IntegrityFlag] = []
    for change in parse_patch(patch):
        if change.new_path is None or is_test_like(change.new_path):
            continue
        flags.extend(
            make_flag(
                instance_id=ctx.instance_id,
                stage=IntegrityStage.TRAJECTORY,
                rule="hardcoded_expected",
                severity=Severity.SUSPECT,
                evidence=line,
            )
            for line in change.added
            if _literal_hit(line, ctx.expected_outputs)
        )
    return tuple(flags)


def audit_trajectory(
    ctx: AuditContext, calls: Sequence[ToolCallRecord]
) -> tuple[IntegrityFlag, ...]:
    """Run every trajectory rule over one instance's calls; flags come back in call order."""
    flags: list[IntegrityFlag] = []
    for call in calls:
        flags.extend(_check_solution_sha(ctx, call))
        flags.extend(_check_upstream_fetch(ctx, call))
        flags.extend(_check_history_probe(ctx, call))
        flags.extend(_check_verifier_access(ctx, call))
        flags.extend(_check_hardcoded(ctx, call))
        flags.extend(_check_truncated(ctx, call))
    return tuple(flags)
