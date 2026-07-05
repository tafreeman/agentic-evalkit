"""Clean-wheel verification: the built wheel is self-contained (plan Task 15 Step 4).

Builds the wheel with ``python -m build --wheel``, creates a temporary
virtual environment *outside* this repository, installs *only* the wheel
into it (no editable install, no dev dependencies, no access to this
checkout's ``src/`` on ``sys.path``), sets the working directory to that
temporary directory, and runs the verification commands the plan
specifies. This is the automated, repeatable form of the manual
clean-wheel verification already recorded in
``docs/release/v0.1-checkpoint.md``.

Marked ``@pytest.mark.integration`` per the plan: it is slow (a real wheel
build plus a real venv creation and package install). The project's default
pytest filter only excludes ``live``-marked tests (``-m 'not live'``, see
``pyproject.toml`` ``addopts`` and ``.github/workflows/ci.yml``), so this
test *does* run in a plain local ``pytest`` invocation and in CI today --
it is not excluded from the default suite. It does not require network
access (the wheel build, venv creation, and install all resolve from the
local checkout and uv's package cache), so running it by default is safe.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Import roots that must be completely absent from the installed wheel's
#: dependency closure -- the same set the dependency-boundary AST scan
#: forbids inside src/agentic_evalkit, verified here from the *installed
#: package's* perspective instead of by reading source.
_FORBIDDEN_IMPORT_ROOTS = ("agentic_v2", "tools.agents", "executionkit")

#: The isolated-verification Python script run inside the temp venv.
#: Written as a standalone string (not imported from this test module)
#: because it must execute with *only* the wheel's dependencies on
#: sys.path -- it cannot import pytest, this test file, or anything else
#: from the outer environment.
_VERIFICATION_SCRIPT = '''
import importlib.util
import sys


def _find_spec_or_none(module_name: str):
    """importlib.util.find_spec(), tolerating a missing parent package.

    find_spec() raises ModuleNotFoundError (not just returning None) when
    an intermediate parent package in a dotted name does not exist at all
    (e.g. "tools.agents" when "tools" itself is not installed) rather than
    when only the leaf submodule is missing. This helper normalizes both
    cases to a plain None so every forbidden root is checked the same way.
    """
    try:
        return importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return None


def main() -> None:
    import agentic_evalkit

    print("VERSION:" + agentic_evalkit.__version__)

    for module_name in ("agentic_v2", "tools.agents", "executionkit"):
        spec = _find_spec_or_none(module_name)
        status = "ABSENT" if spec is None else "PRESENT"
        print(f"FORBIDDEN:{module_name}:{status}")


if __name__ == "__main__":
    main()
'''


def _run(
    command: list[str], *, cwd: Path, timeout: float = 300.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _build_wheel(dist_dir: Path) -> Path:
    """Build the wheel with ``python -m build --wheel`` into ``dist_dir``.

    Uses the currently-running interpreter (this project's own venv, which
    has the ``build`` dev dependency installed) purely as the *builder*;
    the resulting wheel is what gets installed into the isolated verification
    venv below, not this interpreter's own site-packages.
    """
    result = _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"python -m build --wheel failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    wheels = sorted(dist_dir.glob("agentic_evalkit-*.whl"))
    assert len(wheels) == 1, f"expected exactly one built wheel in {dist_dir}, found {wheels}"
    return wheels[0]


def _create_isolated_venv(venv_dir: Path) -> Path:
    """Create a venv outside the repo using ``uv venv`` and return its Python executable."""
    result = _run(["uv", "venv", str(venv_dir), "--python", sys.executable], cwd=venv_dir.parent)
    assert result.returncode == 0, (
        f"uv venv failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    venv_python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    assert venv_python.is_file(), f"expected venv interpreter at {venv_python}"
    return venv_python


def _install_only_the_wheel(venv_python: Path, wheel_path: Path, *, cwd: Path) -> None:
    """Install *only* ``wheel_path`` into the venv -- no dev/test dependencies.

    Uses ``uv pip install`` with the venv's own interpreter targeted
    explicitly (``--python``), so the install is pinned to the isolated
    venv regardless of ``VIRTUAL_ENV``/activation state in the test
    process itself.
    """
    result = _run(
        ["uv", "pip", "install", "--python", str(venv_python), str(wheel_path)],
        cwd=cwd,
    )
    assert result.returncode == 0, (
        f"uv pip install of the wheel failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_clean_wheel_installs_and_runs_outside_the_repository() -> None:
    """Build the wheel, install it alone in an isolated venv, and verify it works.

    Per plan Task 15 Step 4, from a working directory *outside* the
    repository with *only* the wheel installed:

    - ``python -c "import agentic_evalkit; print(agentic_evalkit.__version__)"``
    - ``agentic-evalkit --help``
    - ``agentic-evalkit datasets curated --format json``

    all exit 0, and the curated-presets output contains both ``gsm8k`` and
    ``swe-bench-verified``. A helper built on
    ``importlib.util.find_spec()`` (catching ``ModuleNotFoundError`` for
    missing parent packages) confirms ``agentic_v2``, ``tools.agents``, and
    ``executionkit`` all resolve to no spec inside the installed
    environment.
    """
    with tempfile.TemporaryDirectory(prefix="agentic-evalkit-wheel-build-") as build_tmp:
        dist_dir = Path(build_tmp) / "dist"
        dist_dir.mkdir()
        wheel_path = _build_wheel(dist_dir)

        with tempfile.TemporaryDirectory(prefix="agentic-evalkit-clean-venv-") as venv_parent:
            # tempfile.mkdtemp()-equivalent directories, guaranteed outside
            # REPO_ROOT (system temp is never inside the repo checkout).
            venv_dir = Path(venv_parent) / "venv"
            work_dir = Path(venv_parent) / "work"
            work_dir.mkdir()
            assert not str(venv_dir).startswith(str(REPO_ROOT))
            assert not str(work_dir).startswith(str(REPO_ROOT))

            venv_python = _create_isolated_venv(venv_dir)
            _install_only_the_wheel(venv_python, wheel_path, cwd=work_dir)

            script_path = work_dir / "_verify_clean_wheel.py"
            script_path.write_text(_VERIFICATION_SCRIPT, encoding="utf-8")

            import_result = _run([str(venv_python), str(script_path)], cwd=work_dir)
            assert import_result.returncode == 0, (
                f"isolated verification script failed (exit {import_result.returncode}):\n"
                f"stdout:\n{import_result.stdout}\nstderr:\n{import_result.stderr}"
            )
            assert "VERSION:" in import_result.stdout
            for module_name in _FORBIDDEN_IMPORT_ROOTS:
                assert f"FORBIDDEN:{module_name}:ABSENT" in import_result.stdout, (
                    f"expected {module_name} to be ABSENT from the installed wheel's "
                    f"environment; script output:\n{import_result.stdout}"
                )

            # The console-script entry point (agentic-evalkit) is the
            # documented way to invoke the CLI; running it directly (rather
            # than importing agentic_evalkit.cli) also proves the entry
            # point itself was installed correctly by the wheel.
            cli_executable = venv_python.parent / (
                "agentic-evalkit.exe" if sys.platform == "win32" else "agentic-evalkit"
            )
            assert cli_executable.is_file(), (
                f"expected the agentic-evalkit console script at {cli_executable} "
                f"after installing the wheel"
            )
            help_result = _run([str(cli_executable), "--help"], cwd=work_dir)
            assert help_result.returncode == 0, (
                f"agentic-evalkit --help failed (exit {help_result.returncode}):\n"
                f"stdout:\n{help_result.stdout}\nstderr:\n{help_result.stderr}"
            )

            curated_result = _run(
                [str(cli_executable), "datasets", "curated", "--format", "json"], cwd=work_dir
            )
            assert curated_result.returncode == 0, (
                f"agentic-evalkit datasets curated --format json failed "
                f"(exit {curated_result.returncode}):\n"
                f"stdout:\n{curated_result.stdout}\nstderr:\n{curated_result.stderr}"
            )
            assert "gsm8k" in curated_result.stdout
            assert "swe-bench-verified" in curated_result.stdout
            # The output must also be valid JSON (not just contain the
            # substrings incidentally), confirming --format json actually
            # produced a parseable curated-preset listing.
            curated_payload = json.loads(curated_result.stdout)
            preset_names = {preset["name"] for preset in curated_payload}
            assert {"gsm8k", "swe-bench-verified"} <= preset_names
