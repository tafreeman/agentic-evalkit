"""Enforce the repository's offline/default versus opt-in/live test boundary.

Most of this project's tests are hermetic: they never make a real network
call or depend on any external service being up, and they run by default.
A small number of tests ("live" tests) deliberately do call a real external
provider, so they must be opted into explicitly and must never run as part
of the default, always-on test suite. This module checks that the
separation between the two is actually enforced: every test marked
``@pytest.mark.live`` lives under ``tests/live/``, every module under
``tests/live/`` is marked live as a whole (not just some tests within it),
and both the default and the live-only CI workflows select tests using the
exact filters that keep the two suites apart.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"
LIVE_ROOT = TEST_ROOT / "live"


def test_live_markers_are_confined_to_the_live_test_tree() -> None:
    offenders = []
    for path in TEST_ROOT.rglob("test_*.py"):
        if path == Path(__file__) or LIVE_ROOT in path.parents:
            continue
        if "pytest.mark.live" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_every_live_test_module_is_marked_live_as_a_whole() -> None:
    live_modules = sorted(LIVE_ROOT.glob("test_*.py"))
    assert live_modules
    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in live_modules
        if "pytestmark = pytest.mark.live" not in path.read_text(encoding="utf-8")
    ]
    assert missing == []


def test_ci_workflows_select_the_offline_and_live_suites_explicitly() -> None:
    pytest_config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    default_workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    live_workflow = (REPO_ROOT / ".github" / "workflows" / "live-provider.yml").read_text(
        encoding="utf-8"
    )

    assert "-m 'not live'" in pytest_config
    assert 'pytest -m "not live"' in default_workflow
    assert "pytest tests/live -m live" in live_workflow
