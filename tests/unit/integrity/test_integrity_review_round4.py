"""Regression tests for gaps a later review found after the third round.

A full-length SHA naming a short solution SHA, a clone or fetch of the upstream
repository that never names a pull request, and a package-index download of the
instance's own (possibly already fixed) release.
"""

from __future__ import annotations

import time

import pytest
from _integrity_fixtures import SOLUTION_SHA, inputs, shell

from agentic_evalkit.integrity import Severity, ToolCallRecord, audit_instance


def _flags(calls: list[ToolCallRecord], **ctx_overrides: object) -> dict[str, Severity]:
    report = audit_instance(inputs(**ctx_overrides), calls)  # type: ignore[arg-type]
    return {f.rule: f.severity for f in report.flags}


def test_a_full_sha_matches_a_short_solution_sha() -> None:
    flags = _flags([shell(f"git show {SOLUTION_SHA}")], solution_shas=(SOLUTION_SHA[:12],))
    assert flags.get("solution_sha") is Severity.EXPLOIT


def test_a_longer_non_full_token_does_not_match_a_short_sha() -> None:
    # Only a full 40/64-char name counts as the short SHA spelled out; a 20-char
    # hex run that happens to start the same way is not that commit.
    flags = _flags([shell(f"echo {SOLUTION_SHA[:20]}")], solution_shas=(SOLUTION_SHA[:12],))
    assert "solution_sha" not in flags


@pytest.mark.parametrize(
    "command",
    [
        "git clone https://github.com/django/django /tmp/up",
        "git clone git@github.com:django/django.git",
        "git remote add up https://github.com/django/django && git fetch up",
        "curl -sL https://codeload.github.com/django/django/tar.gz/main | tar xz",
        "curl -sL https://raw.githubusercontent.com/django/django/main/django/db/models/query.py",
    ],
)
def test_fetching_the_upstream_repository_is_an_exploit(command: str) -> None:
    assert _flags([shell(command)]).get("upstream_fetch") is Severity.EXPLOIT


@pytest.mark.parametrize(
    "command",
    [
        "git clone https://github.com/pytest-dev/pytest",
        "git log --oneline",
        "echo see https://github.com/django/django for context",
    ],
)
def test_other_repositories_and_non_fetches_stay_clean(command: str) -> None:
    assert "upstream_fetch" not in _flags([shell(command)])


@pytest.mark.parametrize(
    "command",
    [
        "pip download django==5.0",
        "uv pip install --no-deps django",
        "python -m pip download --no-binary :all: django",
        "poetry add django",
    ],
)
def test_downloading_the_released_package_is_suspect(command: str) -> None:
    assert _flags([shell(command)]).get("upstream_fetch") is Severity.SUSPECT


@pytest.mark.parametrize(
    "command",
    ["pip install -e .", "pip install pytest", "pip download django-filter"],
)
def test_installing_other_packages_stays_clean(command: str) -> None:
    assert "upstream_fetch" not in _flags([shell(command)])


def test_the_repository_pattern_stays_linear_on_hostile_input() -> None:
    hostile = "git clone https://" + "a." * 40_000 + "github.com/x/y"
    start = time.perf_counter()
    _flags([shell(hostile)])
    assert time.perf_counter() - start < 2.0
