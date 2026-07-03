"""Static proof that the package never imports ARP, agentic-tools, or EK.

Verbatim per plan Task 15 Step 1 (`docs/plans/2026-07-02-agentic-evalkit-initial-release.md`)
and ADR-0001. This is an AST scan, not an import-time check, so it catches a
forbidden import even if the module that contains it is never executed by
the rest of the test suite.
"""

import ast
from pathlib import Path

# "tools" is the import root of the agentic-tools companion package.
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
