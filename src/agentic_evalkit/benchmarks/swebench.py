"""SWE-bench Verified benchmark adapter (design §6.2, §7.1).

``princeton-nlp/SWE-bench_Verified`` (config ``default``, split ``test``) is
discoverable, previewable, and projectable, and this adapter can export the
official prediction schema without Docker or a code checkout; authoritative
resolution grading requires the optional ``swebench`` harness capability
(``benchmarks.harness``). This module never checks out code, never executes
a patch, and never labels anything "resolved" — that verdict can only come
from a real :class:`~agentic_evalkit.benchmarks.harness.HarnessResult`
(design §7.1: "Generic rubric or similarity scoring must never be labeled
`SWE-bench resolved`.").
"""

import json

from agentic_evalkit.errors import DatasetSchemaMismatch
from agentic_evalkit.models import EvalSample, GraderSpec, SourceRecord

_API_VERSION = "1"
_ADAPTER_NAME = "swebench-verified@1"
_GRADER_NAME = "swebench-harness@1"
_DEFAULT_MODEL_NAME_OR_PATH = "agentic-evalkit-target"

_REQUIRED_STRING_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "test_patch",
)


def _parse_test_name_list(value: object, *, field_name: str, row_id: str) -> tuple[str, ...]:
    """Parse a SWE-bench fail/pass-to-pass field from a JSON string or array.

    Upstream SWE-bench rows encode ``FAIL_TO_PASS``/``PASS_TO_PASS`` as a
    JSON-encoded string (``'["test_x"]'``) in some sources and as a native
    array in others; this accepts either and always returns a tuple of
    plain strings.
    """
    parsed = value
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError as exc:
            raise DatasetSchemaMismatch(
                message=f"SWE-bench row field {field_name!r} is not valid JSON",
                context={"row_id": row_id},
            ) from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise DatasetSchemaMismatch(
            message=f"SWE-bench row field {field_name!r} must be a list of strings",
            context={"row_id": row_id},
        )
    return tuple(parsed)


class SweBenchVerifiedAdapter:
    """Projects raw SWE-bench Verified rows and exports official predictions."""

    api_version = _API_VERSION
    name = _ADAPTER_NAME

    def prepare(self, record: SourceRecord) -> EvalSample:
        """Project one SWE-bench Verified source record into an ``EvalSample``.

        Preserves issue, repository, base-commit, and test metadata as
        sample metadata/artifacts. This never checks out the repository or
        applies any patch — it is pure, offline row projection.

        Raises:
            DatasetSchemaMismatch: a required field is missing, not a
                string, or ``FAIL_TO_PASS``/``PASS_TO_PASS`` cannot be
                parsed into a list of test names.
        """
        missing = [
            field
            for field in _REQUIRED_STRING_FIELDS
            if not isinstance(record.data.get(field), str)
        ]
        if missing:
            raise DatasetSchemaMismatch(
                message=f"SWE-bench row is missing required string fields: {missing}",
                context={"row_id": record.row_id},
            )

        instance_id = record.data["instance_id"]
        repo = record.data["repo"]
        base_commit = record.data["base_commit"]
        problem_statement = record.data["problem_statement"]
        test_patch = record.data["test_patch"]
        assert isinstance(instance_id, str)  # narrowed by the `missing` check above
        assert isinstance(repo, str)
        assert isinstance(base_commit, str)
        assert isinstance(problem_statement, str)
        assert isinstance(test_patch, str)

        fail_to_pass = _parse_test_name_list(
            record.data.get("FAIL_TO_PASS"), field_name="FAIL_TO_PASS", row_id=record.row_id
        )
        pass_to_pass = _parse_test_name_list(
            record.data.get("PASS_TO_PASS"), field_name="PASS_TO_PASS", row_id=record.row_id
        )

        return EvalSample(
            sample_id=f"swebench-verified:{instance_id}",
            input={"problem_statement": problem_statement, "repo": repo},
            reference=None,
            expected_artifacts={"test_patch": test_patch},
            metadata={
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": base_commit,
                "fail_to_pass": list(fail_to_pass),
                "pass_to_pass": list(pass_to_pass),
            },
            source_row_id=record.row_id,
            source_digest=record.digest,
            adapter=_ADAPTER_NAME,
            grader=GraderSpec(name=_GRADER_NAME, grader_type="authoritative", hard_gate=True),
        )

    def export_prediction(
        self,
        sample: EvalSample,
        patch: str,
        model_name_or_path: str = _DEFAULT_MODEL_NAME_OR_PATH,
    ) -> dict[str, str]:
        """Export the official SWE-bench prediction shape for ``sample``.

        Returns exactly the three official SWE-bench prediction keys
        (``instance_id``, ``model_name_or_path``, ``model_patch``) — no
        adapter or framework metadata is ever mixed into the export, so the
        result is directly consumable by the real SWE-bench harness or a
        leaderboard submission. ``model_name_or_path`` defaults to
        ``"agentic-evalkit-target"`` but callers should pass the actual
        system name for real submissions.
        """
        instance_id = sample.metadata.get("instance_id")
        if not isinstance(instance_id, str):
            raise DatasetSchemaMismatch(
                message="sample.metadata['instance_id'] must be a string to export a prediction",
                context={"sample_id": sample.sample_id},
            )
        return {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": patch,
        }

    def validate_oracle(self, sample: EvalSample) -> bool:
        """Validate row completeness and prediction identity, not patch correctness.

        This confirms the sample carries everything needed to export a
        prediction (an ``instance_id``) and to eventually route it through
        an authoritative harness — it never inspects or judges patch
        content, since only a real ``HarnessResult`` may determine
        resolution (design §7.1).
        """
        instance_id = sample.metadata.get("instance_id")
        return isinstance(instance_id, str) and bool(instance_id)

    def aggregate_metadata(self) -> dict[str, object]:
        """Benchmark-specific metadata recorded on run aggregation (design §7)."""
        return {
            "benchmark": "swebench-verified",
            "adapter": _ADAPTER_NAME,
            "grader": _GRADER_NAME,
        }
