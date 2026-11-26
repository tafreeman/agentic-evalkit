"""Executable architecture for the integrity harness (ADR-0027).

Three rules, each of which a later edit could break without any behavioural
test noticing:

- every ``IntegrityFlag`` is built through ``make_flag``, so evidence is always
  redacted;
- the detectors stay pure: standard library, pydantic and package contracts only;
- the committed ``tool-call-record.v1`` schema matches ``ToolCallRecord``.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from agentic_evalkit.integrity import ToolCallRecord

PACKAGE = Path("src/agentic_evalkit/integrity")
SCHEMA = Path("schemas/tool-call-record.v1.schema.json")

#: Modules allowed to build an ``IntegrityFlag`` directly: the factory itself,
#: and the model module that defines the class.
FLAG_BUILDERS = {"flags.py", "models.py"}

#: Detector modules that must stay pure (no I/O frameworks, no CLI, no runner).
PURE_DETECTORS = ("diff_guard.py", "trajectory.py", "preflight.py")
ALLOWED_PACKAGE_IMPORTS = ("pydantic", "agentic_evalkit.integrity", "agentic_evalkit.models")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_flags_are_only_built_through_make_flag(path: Path) -> None:
    if path.name in FLAG_BUILDERS:
        return
    tree = ast.parse(path.read_text(encoding="utf-8"))
    direct = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "IntegrityFlag"
    ]
    assert direct == [], f"{path.name} builds IntegrityFlag directly at lines {direct}"


def test_make_flag_redacts_and_bounds_evidence() -> None:
    from agentic_evalkit.integrity.flags import EVIDENCE_LIMIT, make_flag
    from agentic_evalkit.integrity.models import IntegrityStage, Severity

    secret = "sk-ant-api03-" + "Z" * 40
    flag = make_flag(
        instance_id="i",
        stage=IntegrityStage.TRAJECTORY,
        rule="r",
        severity=Severity.SUSPECT,
        evidence=f"curl -H 'x-api-key: {secret}' " + "x" * 1000,
    )
    assert secret not in flag.evidence
    assert len(flag.evidence) <= EVIDENCE_LIMIT


@pytest.mark.parametrize("name", PURE_DETECTORS)
def test_detectors_stay_pure(name: str) -> None:
    stdlib = set(sys.stdlib_module_names)
    for module in _imports(PACKAGE / name):
        root = module.split(".")[0]
        assert root in stdlib or module.startswith(ALLOWED_PACKAGE_IMPORTS), (
            f"{name} imports {module}; detectors may use the standard library, "
            "pydantic and package contracts only"
        )


def test_the_integrity_package_does_not_import_the_cli_or_runner() -> None:
    for path in PACKAGE.glob("*.py"):
        for module in _imports(path):
            assert not module.startswith(("agentic_evalkit.cli", "agentic_evalkit.runner"))


def test_tool_call_schema_matches_the_model() -> None:
    committed = json.loads(SCHEMA.read_text(encoding="utf-8"))
    generated = ToolCallRecord.model_json_schema()
    generated["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    generated["$id"] = "tool-call-record.v1"
    assert committed == generated, (
        "schemas/tool-call-record.v1.schema.json is out of date; regenerate it from "
        "ToolCallRecord.model_json_schema()"
    )


def test_tool_call_schema_pins_the_version() -> None:
    committed = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert committed["properties"]["schema_version"]["const"] == "1"
    assert committed["additionalProperties"] is False
