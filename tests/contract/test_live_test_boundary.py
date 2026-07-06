"""Enforce the repository's offline/default versus opt-in/live test boundary."""

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
