"""Clean-wheel verification: proves the built package is truly self-contained
(plan Task 15 Step 4).

A "wheel" is Python's standard built-package format (a ``.whl`` file) --
the thing users actually ``pip install``. This test builds one with
``python -m build --wheel``, then creates a temporary virtual environment
*outside* this repository and installs *only* that wheel into it: no
"editable install" (the dev-mode install that just points back at this
source tree instead of copying code into the environment), no dev
dependencies, and no way for this checkout's ``src/`` directory to sneak
onto Python's import search path (``sys.path``) and paper over a packaging
mistake. It then changes into that temporary directory and runs the
verification commands the release plan specifies. This is the automated,
repeatable version of the manual clean-wheel check already recorded in
``docs/release/v0.1-checkpoint.md``.

Marked ``@pytest.mark.integration`` per the plan because it's slow (it does
a real wheel build plus a real virtual-environment creation and package
install). The project's default pytest filter only excludes tests marked
``live`` (``-m 'not live'`` -- see ``pyproject.toml``'s ``addopts`` and
``.github/workflows/ci.yml``), so this test *does* run in a plain local
``pytest`` invocation and in CI today -- it is not skipped by default. It
never needs network access (the wheel build, venv creation, and install
all come from the local checkout and uv's local package cache), so leaving
it in the default suite is safe.
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

#: Import roots (top-level package names) that must be completely missing
#: from everything the installed wheel could possibly import, directly or
#: indirectly. This is the same list a different contract test enforces by
#: statically scanning the source code (parsing it into an AST, an
#: "abstract syntax tree", to check import statements without running
#: them) inside ``src/agentic_evalkit``. Here we check the same rule from
#: the opposite direction: not by reading source, but by asking the
#: actually-installed package at runtime whether these modules exist.
_FORBIDDEN_IMPORT_ROOTS = ("agentic_v2", "tools.agents", "executionkit")

#: The verification script that actually runs inside the temporary,
#: wheel-only virtual environment. It's written out here as a plain string
#: (rather than living in its own file and being imported normally) because
#: it has to run using *only* the wheel's own dependencies on Python's
#: import path -- it must not be able to import pytest, this test file, or
#: anything else from the outer test environment that wouldn't actually
#: ship with the package.
_VERIFICATION_SCRIPT = '''
import importlib.util
import sys


def _find_spec_or_none(module_name: str):
    """Look up a module like importlib.util.find_spec() does, but never let
    a missing parent package raise an error.

    Normally, find_spec() just returns None when a module doesn't exist.
    But for a dotted name like "tools.agents", it instead *raises*
    ModuleNotFoundError if the parent package ("tools") doesn't exist at
    all -- unlike the plain None you get back when only the last part
    ("agents") is missing while the parent is present. That inconsistency
    would force callers to handle two different "not found" cases
    differently. This helper flattens both cases down to a plain None, so
    every forbidden import root can be checked the exact same way no
    matter which part of the dotted name is missing.
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
    # This silences a linter warning (S603) about subprocess calls that might
    # run untrusted input. That risk doesn't apply here: `command` is always
    # a plain list of arguments (never a shell string), and it's always one
    # of this test's own hardcoded build/check commands -- never anything
    # derived from user input.
    return subprocess.run(  # noqa: S603
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _build_wheel(dist_dir: Path) -> Path:
    """Build the wheel with ``python -m build --wheel`` into ``dist_dir``.

    This uses the interpreter currently running this test (this project's
    own dev environment, which has the ``build`` package installed) purely
    as the *builder* -- a tool that produces the ``.whl`` file, nothing
    more. The resulting wheel is what actually gets installed into the
    separate, isolated verification environment created below; the code
    under test never runs from this interpreter's own installed packages.
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
    """Install *only* ``wheel_path`` into the venv -- no dev or test dependencies.

    This runs ``uv pip install`` and explicitly points it at the target
    venv's own Python interpreter (via ``--python``), rather than relying on
    activation. That guarantees the install goes into the isolated venv no
    matter what ``VIRTUAL_ENV`` is set to, or whether any venv is
    "activated," in the process actually running this test.
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
    repository, with *only* the wheel installed, each of these commands
    must exit 0:

    - ``python -c "import agentic_evalkit; print(agentic_evalkit.__version__)"``
    - ``agentic-evalkit --help``
    - ``agentic-evalkit datasets curated --format json``

    and the curated-presets output must contain both ``gsm8k`` and
    ``swe-bench-verified``. A helper built on ``importlib.util.find_spec()``
    (which treats a ``ModuleNotFoundError`` from a missing parent package the
    same as an ordinary "not found") confirms that ``agentic_v2``,
    ``tools.agents``, and ``executionkit`` are all completely absent from the
    installed environment -- ``find_spec()`` finds nothing for any of them.
    """
    with tempfile.TemporaryDirectory(prefix="agentic-evalkit-wheel-build-") as build_tmp:
        dist_dir = Path(build_tmp) / "dist"
        dist_dir.mkdir()
        wheel_path = _build_wheel(dist_dir)

        with tempfile.TemporaryDirectory(prefix="agentic-evalkit-clean-venv-") as venv_parent:
            # These directories live under the system's temp folder (created
            # via tempfile.TemporaryDirectory() above), so they're
            # guaranteed to be outside REPO_ROOT -- the OS's temp location
            # is never inside this repo checkout.
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

            # "agentic-evalkit" is a console-script entry point: a small
            # standalone executable that pip/uv generates during install so
            # users can just type `agentic-evalkit` in a shell, instead of
            # `python -m ...`. That's the documented way to invoke the CLI.
            # Running this executable directly (rather than importing
            # agentic_evalkit.cli from Python) also proves the entry point
            # itself was installed correctly by the wheel, not just that the
            # underlying package imports.
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
