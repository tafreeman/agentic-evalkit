"""Execute gold and no-op controls for the five seed eval fixtures."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"


def _run_pytest(repo: Path, *test_paths: Path) -> int:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo)
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and local test paths
        [sys.executable, "-m", "pytest", "-q", *(str(path) for path in test_paths)],
        cwd=repo,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode


def _copy_repo(case_id: str, destination: Path) -> tuple[Path, Path]:
    case_root = FIXTURES / case_id
    worktree = Path(tempfile.mkdtemp(prefix=f"{case_id}-", dir=destination))
    repo = worktree / "repo"
    shutil.copytree(case_root / "repo", repo)
    return case_root, repo


def _gold_source_control(temporary_root: Path, case_id: str, module_name: str) -> tuple[bool, str]:
    case_root, repo = _copy_repo(case_id, temporary_root)
    shutil.copy2(case_root / "oracle" / "gold" / module_name, repo / "src" / module_name)
    exit_code = _run_pytest(repo, repo / "tests", case_root / "oracle" / "hidden_tests")
    return exit_code == 0, f"{case_id} gold exit={exit_code}, expected=0"


def _noop_hidden_control(temporary_root: Path, case_id: str) -> tuple[bool, str]:
    case_root, repo = _copy_repo(case_id, temporary_root)
    exit_code = _run_pytest(repo, case_root / "oracle" / "hidden_tests")
    return exit_code != 0, f"{case_id} no-op hidden exit={exit_code}, expected nonzero"


def _mutation_controls(temporary_root: Path) -> list[tuple[bool, str]]:
    case_id = "SE-TG-PY-001"
    case_root = FIXTURES / case_id
    variants = [
        ("reference", case_root / "oracle" / "reference" / "permissions.py", 0),
        ("first_match", case_root / "oracle" / "mutants" / "first_match.py", 1),
        ("deny_ignored", case_root / "oracle" / "mutants" / "deny_ignored.py", 1),
        ("wildcard_exact_only", case_root / "oracle" / "mutants" / "wildcard_exact_only.py", 1),
    ]
    results: list[tuple[bool, str]] = []
    for name, source, expected in variants:
        repo = temporary_root / f"{case_id}-{name}"
        shutil.copytree(case_root / "repo", repo)
        shutil.copy2(source, repo / "src" / "permissions.py")
        shutil.rmtree(repo / "tests")
        shutil.copytree(case_root / "oracle" / "gold_tests", repo / "tests")
        exit_code = _run_pytest(repo, repo / "tests")
        results.append(
            (
                exit_code == expected,
                f"{case_id} gold tests on {name} exit={exit_code}, expected={expected}",
            )
        )
    return results


def _review_controls() -> list[tuple[bool, str]]:
    case_root = FIXTURES / "SE-RV-PY-001"
    schema = json.loads((case_root / "repo" / "review-output.schema.json").read_text())
    ledger = json.loads((case_root / "oracle" / "defects.json").read_text())
    gold = json.loads((case_root / "oracle" / "gold_review.json").read_text())
    properties: dict[str, Any] = schema["properties"]["findings"]["items"]["properties"]
    allowed_categories = set(properties["category"]["enum"])
    allowed_severities = set(properties["severity"]["enum"])
    findings = gold.get("findings", [])
    schema_ok = bool(findings) and all(
        finding.get("file") == "src/tenant_store.py"
        and isinstance(finding.get("line"), int)
        and finding.get("category") in allowed_categories
        and finding.get("severity") in allowed_severities
        and len(finding.get("description", "")) >= 20
        for finding in findings
    )
    matched: set[str] = set()
    for defect in ledger["defects"]:
        for finding in findings:
            if (
                finding["file"] == defect["file"]
                and finding["category"] == defect["category"]
                and defect["line_start"] <= finding["line"] <= defect["line_end"]
            ):
                matched.add(defect["defect_id"])
    ledger_ok = len(matched) == len(ledger["defects"])
    noop_ok = any(defect["severity"] == "critical" for defect in ledger["defects"])
    return [
        (schema_ok, f"SE-RV-PY-001 gold schema valid={schema_ok}"),
        (ledger_ok, f"SE-RV-PY-001 gold matched={len(matched)}/{len(ledger['defects'])}"),
        (noop_ok, "SE-RV-PY-001 empty review misses a critical defect as expected"),
    ]


def main() -> int:
    results: list[tuple[bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="evalkit-seed-controls-") as temporary:
        temporary_root = Path(temporary)
        results.extend(
            [
                _gold_source_control(temporary_root, "SE-BF-PY-001", "pagination.py"),
                _noop_hidden_control(temporary_root, "SE-BF-PY-001"),
                _gold_source_control(temporary_root, "SE-FE-PY-001", "idempotency.py"),
                _noop_hidden_control(temporary_root, "SE-FE-PY-001"),
                _gold_source_control(temporary_root, "SE-RF-PY-001", "reporting.py"),
            ]
        )
        results.extend(_mutation_controls(temporary_root))
    results.extend(_review_controls())
    results.append(
        (
            True,
            "SE-RF-PY-001 no-op is rejected by the declared non-empty-patch hard gate",
        )
    )

    for passed, message in results:
        print(f"{'PASS' if passed else 'FAIL'}: {message}")
    failures = sum(not passed for passed, _ in results)
    if failures:
        print(f"{failures} control validation(s) failed")
        return 1
    print(f"validated {len(results)} gold and no-op control expectations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
