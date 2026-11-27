"""Pre-flight: facts about the environment that must hold before an agent runs.

Three checks, each recorded whether it passes or not:

1. **History:** the repo has no commit, ref or unreachable commit after
   ``base_commit``, so ``git log --all`` cannot show the fix.
2. **Mounts:** test and verifier paths are outside every directory the agent
   can write.
3. **Egress:** the network policy is recorded, and an open one fails.

A check that cannot run reports ``CheckStatus.ERROR``, never a pass. Plus
:func:`parse_eval_script`, which records what an instance's ``eval.sh`` resets
from the base commit so the diff guard knows which test edits are harmless.

The checks take their inputs as arguments. Only :func:`subprocess_git` talks to
``git``, through an injectable :data:`GitRunner`, so every rule is testable
without a repo. The runner runs git with the audited repo's own config and
hooks switched off, because a repository under audit is untrusted input.
"""

from __future__ import annotations

import os
import posixpath
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from agentic_evalkit.integrity.models import CheckStatus, PreflightCheck
from agentic_evalkit.models.base import FrozenModel

if TYPE_CHECKING:
    from pydantic import JsonValue

__all__ = [
    "EGRESS_MODES",
    "EvalScriptResets",
    "GitRunner",
    "check_egress",
    "check_history",
    "check_mounts",
    "parse_eval_script",
    "preflight_evidence",
    "run_preflight",
    "subprocess_git",
]

#: ``none`` and ``model_api_only`` are safe; ``open`` lets the agent fetch the fix.
EGRESS_MODES = ("none", "model_api_only", "open")

#: Runs ``git <args>`` in a repo and returns stdout. Raises ``RuntimeError`` on a git failure.
GitRunner = Callable[[tuple[str, ...]], str]

_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")
#: Commits passed to one ``git rev-list`` call, to stay under argument-length limits.
_BATCH = 200
_GIT_TIMEOUT_SECONDS = 120


def subprocess_git(repo: str) -> GitRunner:
    """A :data:`GitRunner` that shells out to ``git -C <repo>`` with the repo's config off.

    ``core.fsmonitor`` and ``core.hooksPath`` are overridden on the command
    line and global and system config are ignored, so a hostile ``.git/config``
    cannot run a program during the audit. Every call has a timeout.
    """
    git = shutil.which("git") or "git"
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }

    def run(args: tuple[str, ...]) -> str:
        command = (
            git,
            "-C",
            repo,
            "-c",
            "core.fsmonitor=false",
            "-c",
            f"core.hooksPath={os.devnull}",
            *args,
        )
        try:
            completed = subprocess.run(  # noqa: S603 -- fixed argv, no shell
                command,
                capture_output=True,
                text=True,
                check=False,
                env=environment,
                timeout=_GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(type(error).__name__) from error
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"git {args[0]} failed")
        return completed.stdout

    return run


def _error(name: str, detail: str) -> PreflightCheck:
    return PreflightCheck(name=name, status=CheckStatus.ERROR, detail=detail)


def check_history(run_git: GitRunner, base_commit: str) -> PreflightCheck:
    """Pass only if nothing is reachable or recoverable beyond ``base_commit``.

    ``git rev-list --all --not <base>`` lists commits reachable from any ref
    but not from the base. ``git fsck --unreachable`` lists commits no ref
    reaches, which still resolve by hash. Dangling commits that are ancestors of
    the base (an amended base, say) are fine; any other is a leak. ``base_commit``
    must be a hex revision so it can never be read as a git option.
    """
    if not _REVISION.match(base_commit):
        return _error("git_history", "base_commit must be a 7-64 character hex revision")
    try:
        beyond = run_git(("rev-list", "--all", "--not", base_commit)).split()
        dangling = [
            line.split()[-1]
            for line in run_git(("fsck", "--unreachable", "--no-reflogs")).splitlines()
            if line.startswith("unreachable commit")
        ]
        for start in range(0, len(dangling), _BATCH):
            batch = dangling[start : start + _BATCH]
            beyond.extend(run_git(("rev-list", *batch, "--not", base_commit)).split())
    except RuntimeError as error:
        return _error("git_history", f"could not inspect: {error}")
    if beyond:
        return PreflightCheck(
            name="git_history",
            status=CheckStatus.FAILED,
            detail=f"{len(set(beyond))} commit(s) reachable or recoverable after base_commit",
        )
    return PreflightCheck(
        name="git_history", status=CheckStatus.PASSED, detail="no history after base_commit"
    )


_DRIVE = re.compile(r"^[a-z]:/")


def _canonical(path: str) -> str:
    return posixpath.normpath(path.replace("\\", "/").casefold())


def _is_absolute(path: str) -> bool:
    return path.startswith("/") or bool(_DRIVE.match(path))


def _inside(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip("/") + "/")


def check_mounts(*, writable: Iterable[str], protected: Iterable[str]) -> PreflightCheck:
    """Pass only if no protected path lies inside, or contains, a writable mount.

    Both lists must be non-empty and use absolute paths: an empty list or a
    relative path cannot be compared, so it is an ``ERROR`` rather than a pass.
    Paths are normalized (``..``, ``//``, slashes, case) before comparing.
    """
    writable_paths = [_canonical(path) for path in writable]
    protected_paths = [_canonical(path) for path in protected]
    if not writable_paths or not protected_paths:
        return _error("protected_paths", "writable and protected paths must both be declared")
    if not all(_is_absolute(path) for path in (*writable_paths, *protected_paths)):
        return _error("protected_paths", "paths must be absolute to be compared")
    overlaps = sorted(
        f"{path} (writable {root})"
        for path in protected_paths
        for root in writable_paths
        if _inside(path, root) or _inside(root, path)
    )
    if overlaps:
        return PreflightCheck(
            name="protected_paths", status=CheckStatus.FAILED, detail="; ".join(overlaps[:5])
        )
    return PreflightCheck(
        name="protected_paths",
        status=CheckStatus.PASSED,
        detail="test and verifier paths are not writable",
    )


def check_egress(mode: str) -> PreflightCheck:
    """Record the egress policy; only ``open`` fails, and an unknown one cannot be judged."""
    if mode not in EGRESS_MODES:
        return _error("egress", f"unknown policy {mode!r}")
    status = CheckStatus.FAILED if mode == "open" else CheckStatus.PASSED
    return PreflightCheck(name="egress", status=status, detail=f"egress policy: {mode}")


def run_preflight(
    *,
    run_git: GitRunner,
    base_commit: str,
    writable: Iterable[str],
    protected: Iterable[str],
    egress: str,
) -> tuple[PreflightCheck, ...]:
    """Run all three checks for one environment."""
    return (
        check_history(run_git, base_commit),
        check_mounts(writable=writable, protected=protected),
        check_egress(egress),
    )


class EvalScriptResets(FrozenModel):
    """What one instance's ``eval.sh`` restores from the base commit.

    Attributes:
        restored_paths: Files the script checks out *before* it applies a patch
            or runs the tests, so an agent's edit to them never reaches grading.
        late_reset_paths: Files it checks out only afterwards. That is cleanup,
            not protection: the agent's edit is live while the tests run.
        resets_before_patch: Whether any protective reset exists.
        applies_patch: Whether the script applies a patch at all.
    """

    restored_paths: tuple[str, ...] = ()
    late_reset_paths: tuple[str, ...] = ()
    resets_before_patch: bool = False
    applies_patch: bool = False


_CHECKOUT = re.compile(r"^\s*git\s+checkout\s+(?P<args>.+)$")
_APPLY = re.compile(r"^\s*git\s+apply\b")
#: Where the tests start running. A bare ``python`` line is not one: SWE-bench scripts run
#: ``python -m pip install -e .`` before they reset the test files, and that is setup.
_RUN_TESTS = re.compile(
    r"\b(?:pytest|py\.test|tox|unittest|runtests|nosetests)\b|Start Test Output"
)
_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?\s*$")


def _checkout_paths(args: str) -> list[str]:
    tokens = args.split()
    if "--" in tokens:
        return tokens[tokens.index("--") + 1 :]
    positional = [token for token in tokens if not token.startswith("-")]
    return positional[1:]


def parse_eval_script(script: str) -> EvalScriptResets:
    """Read an ``eval.sh`` and record which files it resets, and when.

    Recognizes ``git checkout <commit> <paths...>`` (a bare ``git checkout
    <branch>`` has no path and resets nothing). A reset counts as protection
    only if it comes before the first ``git apply`` or test command. A heredoc
    body holding a patch is skipped so a ``git checkout`` inside diff text is not
    counted.
    """
    early: list[str] = []
    late: list[str] = []
    started = False
    applies = False
    in_heredoc: str | None = None
    for raw in script.replace(chr(13), "").split(chr(10)):
        if in_heredoc is not None:
            in_heredoc = None if raw.strip() == in_heredoc else in_heredoc
            continue
        heredoc = _HEREDOC.search(raw)
        checkout = _CHECKOUT.match(raw)
        if _APPLY.match(raw):
            applies = started = True
        elif checkout:
            (late if started else early).extend(_checkout_paths(checkout.group("args")))
        elif _RUN_TESTS.search(raw):
            started = True
        if heredoc:
            in_heredoc = heredoc.group(1)
    restored = tuple(dict.fromkeys(early))
    return EvalScriptResets(
        restored_paths=restored,
        late_reset_paths=tuple(p for p in dict.fromkeys(late) if p not in restored),
        resets_before_patch=bool(restored),
        applies_patch=applies,
    )


def preflight_evidence(
    checks: Iterable[PreflightCheck], resets: EvalScriptResets | None = None
) -> dict[str, JsonValue]:
    """Pre-flight facts as the JSON an executor stamps into its result evidence.

    ``failed`` lists the checks that did not pass, so a reader can test one key.
    """
    recorded = tuple(checks)
    return {
        "checks": [check.model_dump(mode="json") for check in recorded],
        "failed": [check.name for check in recorded if not check.passed],
        "eval_script": resets.model_dump(mode="json") if resets is not None else None,
    }
