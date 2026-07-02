from importlib.metadata import version

from typer.testing import CliRunner

import agentic_evalkit
from agentic_evalkit.cli import app


def test_package_version_matches_distribution() -> None:
    assert agentic_evalkit.__version__ == version("agentic-evalkit")


def test_cli_version_flag_prints_installed_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == version("agentic-evalkit")
