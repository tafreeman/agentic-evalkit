"""SWE-bench Verified benchmark adapter (design §6.2, §7.1).

The ``princeton-nlp/SWE-bench_Verified`` dataset (config ``default``, split
``test``) works with this project's normal dataset tooling: it can be
looked up, previewed a few rows at a time, and turned into typed samples.
This adapter can also export a prediction in the exact format the official
SWE-bench tooling expects, and it can do all of that without Docker
installed and without actually checking out the target repository's code.
Getting a real, authoritative pass/fail verdict on whether a fix actually
works, though, requires the optional ``swebench`` harness feature (see
``benchmarks.harness``) -- this module by itself never checks out code,
never applies a patch, and never labels anything "resolved" itself. That
verdict can only ever come from a real
:class:`~agentic_evalkit.benchmarks.harness.HarnessResult` (design §7.1:
"Generic rubric or similarity scoring must never be labeled
`SWE-bench resolved`.").
"""

import json
from typing import cast

from pydantic import JsonValue

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
    """Parse one of SWE-bench's test-name list fields, in whichever format it comes in.

    ``FAIL_TO_PASS`` and ``PASS_TO_PASS`` are lists of test names -- the
    tests that should go from failing to passing (or stay passing) once a
    correct fix is applied. Depending on where the data came from, the
    original SWE-bench rows store these either as a JSON string that still
    needs parsing (for example ``'["test_x"]'``) or as an already-native
    array. This function accepts either form and always hands back a plain
    tuple of strings.
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
    """Turns raw SWE-bench Verified rows into samples, and exports
    predictions in the official format the real SWE-bench tooling expects."""

    api_version = _API_VERSION
    name = _ADAPTER_NAME

    def prepare(self, record: SourceRecord) -> EvalSample:
        """Turn one raw SWE-bench Verified row into an ``EvalSample``.

        Keeps the issue description, repository name, base commit, and test
        information as sample metadata/artifacts. This never actually
        downloads or checks out the repository, and never applies any patch
        -- it's a pure, offline transformation of one row's fields into the
        typed sample format.

        Raises:
            DatasetSchemaMismatch: a required field is missing or isn't a
                string, or ``FAIL_TO_PASS``/``PASS_TO_PASS`` can't be parsed
                into a list of test names.
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

        # `cast` here just tells the type checker "trust me, this is a str"
        # -- it does nothing at runtime. We can safely say that because the
        # `missing` check just above already confirmed every field in
        # `_REQUIRED_STRING_FIELDS` really is a string. We use `cast`
        # instead of an `assert` because Python strips `assert` statements
        # out entirely when run in optimized mode (`python -O`), so an
        # `assert` here would silently stop protecting us in that mode.
        instance_id = cast("str", record.data["instance_id"])
        repo = cast("str", record.data["repo"])
        base_commit = cast("str", record.data["base_commit"])
        problem_statement = cast("str", record.data["problem_statement"])
        test_patch = cast("str", record.data["test_patch"])

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
        """Export ``sample``'s prediction in the official SWE-bench format.

        Returns exactly the three fields the official SWE-bench tooling
        expects (``instance_id``, ``model_name_or_path``, ``model_patch``)
        -- nothing else from this project's own data model gets mixed in,
        so the result can be handed directly to the real SWE-bench harness
        or submitted to a leaderboard as-is. ``model_name_or_path`` defaults
        to the placeholder ``"agentic-evalkit-target"``, but callers making
        a real submission should pass the actual system's name instead.
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
        """Check the row is complete enough to identify and route -- this
        does not judge whether any actual patch is correct.

        This only confirms the sample has what it needs to export a
        prediction (an ``instance_id``) and eventually be sent through an
        authoritative harness for real grading. It never looks at or judges
        any patch's content -- only a real ``HarnessResult`` is allowed to
        decide whether an issue is actually resolved (design §7.1).
        """
        instance_id = sample.metadata.get("instance_id")
        return isinstance(instance_id, str) and bool(instance_id)

    def aggregate_metadata(self) -> dict[str, JsonValue]:
        """Extra, benchmark-specific details recorded when summarizing a run
        (design §7)."""
        return {
            "benchmark": "swebench-verified",
            "adapter": _ADAPTER_NAME,
            "grader": _GRADER_NAME,
        }
