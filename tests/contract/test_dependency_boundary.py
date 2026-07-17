"""Static proof that the package never imports ARP, agentic-tools, or EK.

ARP, agentic-tools, and EK (ExecutionKit) are three separate, sibling
systems. ADR-0001 (the "dependency boundary" decision) says this package
must never import from any of them; this requirement is copied verbatim
from plan Task 15 Step 1
(`docs/plans/2026-07-02-agentic-evalkit-initial-release.md`) and from
ADR-0001 itself.

This check is an AST scan, not an import-time check: it parses every
source file into a Python AST (Abstract Syntax Tree -- a structured
representation of the code's structure, produced without running any of
it) and walks that tree looking for import statements. Because nothing is
actually executed, this catches a forbidden import even inside a code path
that the rest of the test suite never runs.
"""

import ast
from pathlib import Path

# These are Python import roots -- the first dotted segment of a module
# path, e.g. the "tools" in "tools.agents.something" -- not the
# human-readable product names used in prose elsewhere. "agentic_v2" is
# ARP's import root, "tools" is the import root of the agentic-tools
# companion package, and "executionkit" is EK's import root.
FORBIDDEN_ROOTS = {"agentic_v2", "tools", "executionkit"}


def test_package_does_not_import_arp_tools_or_executionkit() -> None:
    violations: list[str] = []
    for path in Path("src/agentic_evalkit").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in FORBIDDEN_ROOTS:
                    violations.append(f"{path}:{node.lineno}:{name}")
    assert violations == []
