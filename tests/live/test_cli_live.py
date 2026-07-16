"""Opt-in CLI coverage against the real Hugging Face services."""

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
    # The Viewer may return 401 to prevent private-dataset enumeration, 404 for
    # a missing public dataset, or 503 during transient upstream unavailability.
    # All three map to stable provider errors -- this test asserts a bad dataset
    # reference never crashes the CLI, not which specific HTTP status HF returns.
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
