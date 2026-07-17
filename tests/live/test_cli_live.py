"""Tests that exercise the CLI against the real Hugging Face services.

These only run when explicitly requested (see the ``live`` marker below)
rather than as part of the normal test suite, because they depend on a real
network connection to a service this project doesn't control.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_evalkit.cli import app

pytestmark = pytest.mark.live

runner = CliRunner()


def test_provider_failure_has_nonzero_exit_and_error_code(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["datasets", "inspect", "hf:missing/not-found"],
        env={"AGENTIC_EVALKIT_CACHE_DIR": str(tmp_path)},
    )
    assert result.exit_code == 4
    # Hugging Face's Dataset Viewer service can respond to a bad dataset
    # reference like this one in three different ways: 401 (refusing to even
    # confirm whether a private dataset exists), 404 (the dataset genuinely
    # doesn't exist), or 503 (the service is temporarily down). This project
    # turns all three into its own stable, well-known error codes. This test
    # isn't checking which of the three Hugging Face happens to return today
    # -- it only checks that a bad dataset reference always produces one of
    # these clean, recognizable error codes, and never crashes the CLI.
    assert (
        ("dataset_access_denied" in result.stdout)
        or ("dataset_not_found" in result.stdout)
        or ("dataset_provider_unavailable" in result.stdout)
    )


def test_datasets_search_supports_json_format(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["datasets", "search", "gsm8k", "--provider", "huggingface", "--format", "json"],
        env={"AGENTIC_EVALKIT_CACHE_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    assert "hits" in payload
