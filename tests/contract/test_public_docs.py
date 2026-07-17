"""User-facing documentation hygiene: no internal codenames leak out.

A "codename" here is an internal-only project or package name -- one that
should never appear anywhere an actual user of this library reads, because
it would be confusing (nobody outside this repo knows what it refers to)
or would leak information that is only meant for internal use. Plan Task
15 Step 1: "Add a public-document hygiene test that scans README, guides,
examples, CLI help snapshots, and error-message fixtures for internal
codenames `agentic_v2`, `agentic-v2-eval`, `tools.agents`, and
`executionkit`. The dependency-boundary test may contain forbidden names
as test data; public user-facing artifacts may not."

This module scans exactly the user-facing artifact set the plan names:
README.md, docs/index.md, docs/guides/*.md, examples/**, and the CLI
``--help`` output captured live via ``typer.testing.CliRunner`` (the root
command plus every registered subcommand and subcommand group, so a
codename hidden in a docstring that only surfaces through a subcommand's
own ``--help`` is still caught). It deliberately does NOT scan
docs/plans, docs/specs, or docs/adr -- those documents legitimately name
the forbidden identifiers as things this package must never import (see
ADR-0001, ADR-0006, and the plan's own dependency-boundary snippet), so
scanning them would produce permanent false positives (failures that flag
something as broken when it is actually fine and expected) that could
never be fixed, since that usage is correct by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_evalkit.cli import app

#: Internal, machine-readable codenames that must never appear in anything
#: a user reads: Python import roots and draft-name spellings, matched
#: case-sensitively. "agentic-v2-eval" (hyphenated) is the ARP-local draft
#: name ADR-0001 retires; "agentic_v2" (its Python-import-style spelling)
#: is scanned separately since the two do not share a substring.
#: "tools.agents" is the dotted-import spelling the dependency-boundary
#: test's own docstring uses for the agentic-tools companion package;
#: "executionkit" (all-lowercase) is EK's *import root*
#: (``import executionkit`` / ``executionkit.evals``) -- deliberately
#: case-sensitive so it does not also flag "ExecutionKit", the ordinary
#: English product name design §1-2 and ADR-0001/0006 themselves use in
#: sentences describing what this package does *not* import. A prose
#: mention of the companion product's name is not a leaked codename; a
#: lowercase import-path spelling is.
FORBIDDEN_CODENAMES = (
    "agentic_v2",
    "agentic-v2-eval",
    "tools.agents",
    "executionkit",
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The exact user-facing artifact set the plan names. Each entry is either
#: one file or a glob (relative to the repository root) whose *matching*
#: files must all be clean.
USER_FACING_FILES = (REPO_ROOT / "README.md",)
USER_FACING_GLOBS = (
    "docs/index.md",
    "docs/guides/*.md",
    "examples/**/*.md",
    "examples/**/*.py",
    "examples/**/*.yaml",
    "examples/**/*.yml",
    "examples/**/*.json",
    "examples/**/*.jsonl",
)


def _resolve_user_facing_paths() -> tuple[Path, ...]:
    paths: list[Path] = [path for path in USER_FACING_FILES if path.is_file()]
    for pattern in USER_FACING_GLOBS:
        paths.extend(sorted(REPO_ROOT.glob(pattern)))
    # De-duplicate while preserving order (a file could theoretically match
    # more than one glob pattern above).
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    return tuple(unique_paths)


def _find_codenames(text: str) -> list[str]:
    """Case-sensitive substring scan for every forbidden codename.

    Case-sensitive deliberately: "executionkit" (the lowercase Python
    import root) must be flagged, but "ExecutionKit" (the capitalized
    English product name used in legitimate prose describing what this
    package does not import) must not be. None of the other three
    codenames has a comparable legitimate capitalized-prose form, so one
    case-sensitive rule serves all four correctly.
    """
    return [codename for codename in FORBIDDEN_CODENAMES if codename in text]


def test_user_facing_file_set_is_nonempty() -> None:
    """Sanity check: the glob patterns above actually resolve to real files.

    Guards against this test silently passing because every pattern
    matched zero files (for example, after a future directory rename).
    """
    paths = _resolve_user_facing_paths()
    assert len(paths) >= 10, (
        f"expected at least 10 user-facing files (README + docs/index.md + guides + "
        f"examples/http_agent content), found {len(paths)}: {paths}"
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "README.md",
        "docs/index.md",
        "docs/guides/quickstart.md",
        "docs/guides/cli-reference.md",
        "docs/guides/providers.md",
        "docs/guides/graders.md",
        "docs/guides/targets.md",
        "docs/guides/swebench.md",
        "docs/guides/http-agent-example.md",
        "examples/http_agent/README.md",
        "examples/http_agent/run_example.py",
        "examples/http_agent/stub_agent_server.py",
        "examples/http_agent/eval.yaml",
        "examples/http_agent/questions.jsonl",
    ],
)
def test_named_user_facing_file_has_no_internal_codenames(relative_path: str) -> None:
    """Every specific user-facing file this task creates/modifies is clean.

    Parametrized per-file (rather than one aggregate assertion) so a
    failure names the exact offending file instead of an opaque combined
    violation list.
    """
    path = REPO_ROOT / relative_path
    assert path.is_file(), f"expected user-facing file {relative_path} to exist"
    found = _find_codenames(path.read_text(encoding="utf-8"))
    assert found == [], f"{relative_path} contains forbidden internal codenames: {found}"


def test_every_resolved_user_facing_file_has_no_internal_codenames() -> None:
    """Aggregate sweep over every file the glob patterns resolve to.

    Catches any user-facing file added later under docs/guides/ or
    examples/ that the explicit parametrized list above does not yet name.
    """
    violations: dict[str, list[str]] = {}
    for path in _resolve_user_facing_paths():
        found = _find_codenames(path.read_text(encoding="utf-8"))
        if found:
            violations[str(path.relative_to(REPO_ROOT))] = found
    assert violations == {}


#: Every command path this CLI registers (root plus every subcommand and,
#: for the ``datasets`` group, every sub-subcommand), each exercised with
#: ``--help`` so a codename hidden in a docstring surfaced only through a
#: nested ``--help`` cannot slip past a root-only check. The empty tuple is
#: the root app itself (``agentic-evalkit --help``).
_CLI_COMMAND_PATHS = (
    (),
    ("doctor",),
    ("init",),
    ("validate",),
    ("run",),
    ("compare",),
    ("report",),
    ("datasets",),
    ("datasets", "curated"),
    ("datasets", "search"),
    ("datasets", "inspect"),
    ("datasets", "preview"),
    ("datasets", "pull"),
)


@pytest.mark.parametrize(
    "command_path", _CLI_COMMAND_PATHS, ids=lambda path: " ".join(path) or "root"
)
def test_cli_help_output_has_no_internal_codenames(command_path: tuple[str, ...]) -> None:
    runner = CliRunner()
    args = [*command_path, "--help"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, f"agentic-evalkit {' '.join(args)} exited {result.exit_code}"
    found = _find_codenames(result.stdout)
    assert found == [], (
        f"agentic-evalkit {' '.join(args)} output contains forbidden codenames: {found}"
    )
