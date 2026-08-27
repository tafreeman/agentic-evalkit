"""Validate the synthetic software-engineering baseline catalog and isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "cases.jsonl"
SUITE = ROOT / "suite.json"
REQUIRED_CASE_KEYS = {
    "schema_version",
    "case_id",
    "track",
    "language",
    "priority",
    "difficulty",
    "title",
    "repo_path",
    "prompt",
    "allowed_changes",
    "forbidden_changes",
    "execution_policy",
    "output_contract",
}
REQUIRED_ORACLE_KEYS = {
    "schema_version",
    "case_id",
    "oracle_type",
    "hard_gates",
    "controls",
    "canary",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def main() -> int:
    errors: list[str] = []
    suite = _load_json(SUITE)
    if suite.get("suite_id") != "software-engineering-baseline-proposal":
        errors.append("suite.json: unexpected suite_id")
    if suite.get("status") != "proposal":
        errors.append("suite.json: status must remain proposal before promotion")
    if suite.get("relationship_to_established_evalkit") != "separate":
        errors.append("suite.json: relationship to established EvalKit must be separate")
    validation = suite.get("validation")
    if not isinstance(validation, dict):
        errors.append("suite.json: validation must be an object")
    elif (
        validation.get("agent_workflows") != "not_run"
        or validation.get("evalkit_integration") != "not_run"
    ):
        errors.append("suite.json: unexecuted validation statuses changed without promotion")
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(CATALOG.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as parse_error:
            errors.append(f"cases.jsonl:{line_number}: {parse_error}")
            continue
        if not isinstance(value, dict):
            errors.append(f"cases.jsonl:{line_number}: expected object")
            continue
        missing = REQUIRED_CASE_KEYS - value.keys()
        if missing:
            errors.append(
                f"{value.get('case_id', line_number)}: missing case keys {sorted(missing)}"
            )
        cases.append(value)

    ids = [str(case.get("case_id")) for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("case IDs are not unique")

    catalog_text = CATALOG.read_text(encoding="utf-8")
    for case in cases:
        case_id = str(case.get("case_id"))
        repo_rel = Path(str(case.get("repo_path", "")))
        if repo_rel.is_absolute() or ".." in repo_rel.parts:
            errors.append(f"{case_id}: repo_path must be confined and relative")
            continue
        repo = ROOT / repo_rel
        case_root = repo.parent
        oracle_path = case_root / "oracle" / "oracle.json"
        if not repo.is_dir():
            errors.append(f"{case_id}: missing target-visible repo {repo_rel}")
            continue
        if not oracle_path.is_file():
            errors.append(f"{case_id}: missing oracle/oracle.json")
            continue
        try:
            oracle = _load_json(oracle_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
            continue
        missing_oracle = REQUIRED_ORACLE_KEYS - oracle.keys()
        if missing_oracle:
            errors.append(f"{case_id}: missing oracle keys {sorted(missing_oracle)}")
        if oracle.get("case_id") != case_id:
            errors.append(f"{case_id}: oracle case_id mismatch")
        if not oracle.get("hard_gates"):
            errors.append(f"{case_id}: oracle has no hard gates")
        controls = oracle.get("controls")
        if (
            not isinstance(controls, dict)
            or not {
                "gold_passes",
                "gold_artifact",
                "noop_fails",
            }
            <= controls.keys()
        ):
            errors.append(
                f"{case_id}: oracle controls must declare gold_passes, "
                "gold_artifact, and noop_fails"
            )
        elif not (oracle_path.parent / str(controls["gold_artifact"])).is_file():
            errors.append(f"{case_id}: declared gold artifact does not exist")
        canary = oracle.get("canary")
        if not isinstance(canary, str) or not canary:
            errors.append(f"{case_id}: oracle canary must be a non-empty string")
        elif canary in catalog_text:
            errors.append(f"{case_id}: oracle canary leaked into cases.jsonl")
        else:
            for path in repo.rglob("*"):
                if not path.is_file() or path.suffix in {".pyc", ".pyo"}:
                    continue
                try:
                    target_text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                if canary in target_text:
                    errors.append(f"{case_id}: oracle canary leaked into {path.relative_to(ROOT)}")

    if errors:
        for diagnostic in errors:
            print(f"ERROR: {diagnostic}")
        return 1
    print(f"validated {len(cases)} isolated software-engineering eval cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
