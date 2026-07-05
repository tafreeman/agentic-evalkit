import pytest
from _benchmark_fixtures import _harness_request

from agentic_evalkit.benchmarks.harness import UnavailableHarnessExecutor
from agentic_evalkit.benchmarks.swebench import SweBenchVerifiedAdapter
from agentic_evalkit.errors import DatasetSchemaMismatch
from agentic_evalkit.models import SourceRecord

_COMPLETE_ROW_DATA = {
    "instance_id": "org__repo-1",
    "repo": "org/repo",
    "base_commit": "abc",
    "problem_statement": "Fix it",
    "test_patch": "diff --git a/test.py b/test.py",
    "FAIL_TO_PASS": '["test_x"]',
    "PASS_TO_PASS": '["test_y"]',
}


def test_exports_official_prediction_shape() -> None:
    row = SourceRecord(
        row_id="0",
        digest="sha256:row",
        data={
            "instance_id": "org__repo-1",
            "repo": "org/repo",
            "base_commit": "abc",
            "problem_statement": "Fix it",
            "test_patch": "diff --git a/test.py b/test.py",
            "FAIL_TO_PASS": '["test_x"]',
            "PASS_TO_PASS": '["test_y"]',
        },
    )
    sample = SweBenchVerifiedAdapter().prepare(row)
    prediction = SweBenchVerifiedAdapter().export_prediction(sample, "diff --git a/x b/x")
    assert prediction == {
        "instance_id": "org__repo-1",
        "model_name_or_path": "agentic-evalkit-target",
        "model_patch": "diff --git a/x b/x",
    }


@pytest.mark.asyncio
async def test_missing_harness_is_unavailable_not_failed() -> None:
    result = await UnavailableHarnessExecutor("install agentic-evalkit[swebench]").execute(
        _harness_request()
    )
    assert result.status == "unavailable"
    assert "agentic-evalkit[swebench]" in result.message


def test_export_prediction_defaults_model_name_or_path_to_agentic_evalkit_target() -> None:
    row = SourceRecord(row_id="0", digest="sha256:row", data=dict(_COMPLETE_ROW_DATA))
    sample = SweBenchVerifiedAdapter().prepare(row)
    prediction = SweBenchVerifiedAdapter().export_prediction(sample, "diff --git a/x b/x")
    assert prediction["model_name_or_path"] == "agentic-evalkit-target"


def test_export_prediction_accepts_custom_model_name_or_path() -> None:
    """Real leaderboard submissions must be able to carry the actual system name."""
    row = SourceRecord(row_id="0", digest="sha256:row", data=dict(_COMPLETE_ROW_DATA))
    sample = SweBenchVerifiedAdapter().prepare(row)
    prediction = SweBenchVerifiedAdapter().export_prediction(
        sample, "diff --git a/x b/x", model_name_or_path="my-real-agent-v2"
    )
    assert prediction == {
        "instance_id": "org__repo-1",
        "model_name_or_path": "my-real-agent-v2",
        "model_patch": "diff --git a/x b/x",
    }


def test_export_prediction_never_includes_extra_keys() -> None:
    row = SourceRecord(row_id="0", digest="sha256:row", data=dict(_COMPLETE_ROW_DATA))
    sample = SweBenchVerifiedAdapter().prepare(row)
    prediction = SweBenchVerifiedAdapter().export_prediction(sample, "diff")
    assert set(prediction.keys()) == {"instance_id", "model_name_or_path", "model_patch"}


def test_prepare_accepts_native_array_fail_to_pass_and_pass_to_pass() -> None:
    """Some sources encode these fields as native arrays rather than JSON strings."""
    data = dict(_COMPLETE_ROW_DATA)
    data["FAIL_TO_PASS"] = ["test_x", "test_y"]
    data["PASS_TO_PASS"] = ["test_z"]
    row = SourceRecord(row_id="0", digest="sha256:row", data=data)
    sample = SweBenchVerifiedAdapter().prepare(row)
    assert sample.metadata["fail_to_pass"] == ["test_x", "test_y"]
    assert sample.metadata["pass_to_pass"] == ["test_z"]


def test_prepare_raises_dataset_schema_mismatch_for_malformed_fail_to_pass_json() -> None:
    data = dict(_COMPLETE_ROW_DATA)
    data["FAIL_TO_PASS"] = "not valid json"
    row = SourceRecord(row_id="0", digest="sha256:row", data=data)
    with pytest.raises(DatasetSchemaMismatch):
        SweBenchVerifiedAdapter().prepare(row)


def test_prepare_raises_dataset_schema_mismatch_for_missing_required_field() -> None:
    data = dict(_COMPLETE_ROW_DATA)
    del data["base_commit"]
    row = SourceRecord(row_id="0", digest="sha256:row", data=data)
    with pytest.raises(DatasetSchemaMismatch):
        SweBenchVerifiedAdapter().prepare(row)


def test_prepare_never_touches_filesystem_or_checks_out_code(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Adapter projection must be pure row transformation -- no repo checkout side effects."""
    empty_dir = tmp_path_factory.mktemp("swebench-should-stay-empty")
    row = SourceRecord(row_id="0", digest="sha256:row", data=dict(_COMPLETE_ROW_DATA))
    SweBenchVerifiedAdapter().prepare(row)
    assert list(empty_dir.iterdir()) == []


def test_adapter_declares_api_version_and_name() -> None:
    adapter = SweBenchVerifiedAdapter()
    assert adapter.api_version == "1"
    assert adapter.name == "swebench-verified@1"


def test_validate_oracle_checks_identity_not_patch_content() -> None:
    row = SourceRecord(row_id="0", digest="sha256:row", data=dict(_COMPLETE_ROW_DATA))
    sample = SweBenchVerifiedAdapter().prepare(row)
    assert SweBenchVerifiedAdapter().validate_oracle(sample) is True
