"""Diff guard and pre-flight rules, including restoration and a real git repository."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path  # noqa: TC003

import pytest
from _integrity_fixtures import CONFTEST_PATCH, FIX_PATCH, TEST_EDIT_PATCH

from agentic_evalkit.integrity.diff_guard import changed_paths, guard_patch, is_test_like
from agentic_evalkit.integrity.models import Severity
from agentic_evalkit.integrity.preflight import (
    check_egress,
    check_history,
    check_mounts,
    parse_eval_script,
    preflight_evidence,
    subprocess_git,
)

# --- diff guard -------------------------------------------------------------


def test_changed_paths_reads_git_headers_and_dedupes() -> None:
    assert changed_paths(FIX_PATCH + TEST_EDIT_PATCH) == (
        "django/core/validators.py",
        "tests/validators/tests.py",
    )


def test_changed_paths_reports_a_deleted_file_under_its_old_path() -> None:
    patch = "--- a/tests/old_test.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    assert changed_paths(patch) == ("tests/old_test.py",)


def test_changed_paths_handles_plain_unified_diff() -> None:
    patch = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert changed_paths(patch) == ("pkg/mod.py",)


@pytest.mark.parametrize(
    "path",
    ["conftest.py", "tests/conftest.py", "pkg/tests/x.py", "test_a.py", "a_test.py", "pytest.ini"],
)
def test_test_like_paths(path: str) -> None:
    assert is_test_like(path)


@pytest.mark.parametrize("path", ["django/core/validators.py", ".github/workflows/ci.yml"])
def test_source_paths_are_not_test_like(path: str) -> None:
    assert not is_test_like(path)


def test_dot_directory_is_not_mangled_by_normalization() -> None:
    patch = "diff --git a/.github/x.py b/.github/x.py\n--- a/.github/x.py\n+++ b/.github/x.py\n"
    assert guard_patch(instance_id="i", patch=patch, confine_to=(".github",)) == ()


def test_unrestored_test_edit_is_an_exploit() -> None:
    flags = guard_patch(
        instance_id="i", patch=TEST_EDIT_PATCH, graded_paths=("tests/validators/tests.py",)
    )
    assert [(f.rule, f.severity) for f in flags] == [("graded_file_edited", Severity.EXPLOIT)]


def test_restored_test_edit_is_not_flagged() -> None:
    """The documented false-positive trap: an edit the grader resets is harmless."""
    flags = guard_patch(
        instance_id="i",
        patch=TEST_EDIT_PATCH,
        graded_paths=("tests/validators/tests.py",),
        restored_paths=("./tests/validators/tests.py",),
    )
    assert flags == ()


def test_restoring_one_file_does_not_excuse_another() -> None:
    flags = guard_patch(
        instance_id="i",
        patch=TEST_EDIT_PATCH + CONFTEST_PATCH,
        restored_paths=("tests/validators/tests.py",),
    )
    assert [f.evidence.endswith("tests/conftest.py") for f in flags] == [True]


def test_confine_to_flags_files_outside_the_target() -> None:
    flags = guard_patch(instance_id="i", patch=FIX_PATCH, confine_to=("docs",))
    assert [(f.rule, f.severity) for f in flags] == [("diff_outside_target", Severity.SUSPECT)]
    assert guard_patch(instance_id="i", patch=FIX_PATCH, confine_to=("django/core",)) == ()


def test_confine_to_matches_whole_path_segments() -> None:
    flags = guard_patch(instance_id="i", patch=FIX_PATCH, confine_to=("django/co",))
    assert [f.rule for f in flags] == ["diff_outside_target"]


def test_empty_patch_raises_nothing() -> None:
    assert guard_patch(instance_id="i", patch="") == ()


# --- eval.sh parsing --------------------------------------------------------

EVAL_SH = """#!/bin/bash
set -uxo pipefail
cd /testbed
git checkout abc123 tests/validators/tests.py tests/conftest.py
git apply -v - <<'EOF_1143'
diff --git a/tests/validators/tests.py b/tests/validators/tests.py
git checkout nothing/inside/heredoc.py
EOF_1143
pytest tests/validators/tests.py
git checkout abc123 tests/validators/tests.py
"""


def test_eval_script_records_restored_paths_and_ordering() -> None:
    resets = parse_eval_script(EVAL_SH)
    assert resets.restored_paths == ("tests/validators/tests.py", "tests/conftest.py")
    assert resets.resets_before_patch is True
    assert resets.applies_patch is True


def test_eval_script_ignores_checkout_inside_a_heredoc() -> None:
    assert "nothing/inside/heredoc.py" not in parse_eval_script(EVAL_SH).restored_paths


def test_eval_script_reset_only_after_patch_is_not_before() -> None:
    script = "git apply p.diff\ngit checkout abc tests/a.py\n"
    resets = parse_eval_script(script)
    assert resets.restored_paths == ()
    assert resets.late_reset_paths == ("tests/a.py",)
    assert resets.resets_before_patch is False


def test_branch_checkout_resets_nothing() -> None:
    assert parse_eval_script("git checkout main\n").restored_paths == ()


def test_checkout_with_double_dash_and_no_commit() -> None:
    assert parse_eval_script("git checkout -- tests/a.py\n").restored_paths == ("tests/a.py",)


# --- mounts, egress, evidence ----------------------------------------------


def test_mounts_pass_when_protected_paths_are_outside_writable_roots() -> None:
    check = check_mounts(writable=["/work/repo"], protected=["/eval/eval.sh", "/verifier"])
    assert check.passed


@pytest.mark.parametrize("protected", ["/work/repo/tests", "/work", "/work/repo"])
def test_mounts_fail_on_overlap_in_either_direction(protected: str) -> None:
    check = check_mounts(writable=["/work/repo"], protected=[protected])
    assert not check.passed
    assert protected in check.detail


def test_mounts_do_not_confuse_sibling_prefixes() -> None:
    assert check_mounts(writable=["/work/repo"], protected=["/work/repo2/tests"]).passed


def test_egress_modes() -> None:
    assert check_egress("none").passed
    assert check_egress("model_api_only").passed
    assert not check_egress("open").passed
    unknown = check_egress("wide-open")
    assert not unknown.passed
    assert "unknown" in unknown.detail


def test_preflight_evidence_lists_failures() -> None:
    evidence = preflight_evidence(
        [check_egress("none"), check_egress("open")], parse_eval_script(EVAL_SH)
    )
    assert evidence["failed"] == ["egress"]
    assert isinstance(evidence["eval_script"], dict)
    assert preflight_evidence([])["eval_script"] is None


# --- git history, against a real repository --------------------------------

pytestmark_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603
        [
            shutil.which("git") or "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    return root, _git(root, "rev-parse", "HEAD")


@pytestmark_git
def test_history_passes_when_head_is_the_base_commit(repo: tuple[Path, str]) -> None:
    root, base = repo
    assert check_history(subprocess_git(str(root)), base).passed


@pytestmark_git
def test_history_fails_when_a_branch_holds_a_later_commit(repo: tuple[Path, str]) -> None:
    root, base = repo
    _git(root, "checkout", "-q", "-b", "fix")
    (root / "a.txt").write_text("two\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "the fix")
    _git(root, "checkout", "-q", "main")
    check = check_history(subprocess_git(str(root)), base)
    assert not check.passed
    assert "1 commit" in check.detail


@pytestmark_git
def test_history_fails_on_an_unreachable_commit(repo: tuple[Path, str]) -> None:
    root, base = repo
    (root / "a.txt").write_text("two\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "the fix")
    _git(root, "reset", "-q", "--hard", base)
    _git(root, "reflog", "expire", "--expire=now", "--all")
    check = check_history(subprocess_git(str(root)), base)
    assert not check.passed
    assert "recoverable" in check.detail


@pytestmark_git
def test_history_fails_closed_when_git_cannot_run(tmp_path: Path) -> None:
    check = check_history(subprocess_git(str(tmp_path / "missing")), "abcdef1")
    assert not check.passed
    assert "could not inspect" in check.detail


def test_history_uses_the_injected_runner() -> None:
    outputs = {"rev-list": "", "fsck": ""}

    def run(args: tuple[str, ...]) -> str:
        return outputs[args[0]]

    assert check_history(run, "abcdef1").passed
