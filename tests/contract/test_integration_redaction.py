"""Integration-sink redaction contract (ADR-0022).

An "external sink" is any function in this package that transmits run data
off the machine that produced it -- today, the MLflow and Langfuse exports.
They are the highest-consequence surface in the codebase for exactly one
reason: a report file stays where it was written and can be deleted, while
an export lands on a shared server, gets indexed, and is read by people who
were not in the room. A secret that reaches a reporter is a local mistake. A
secret that reaches a tracking server is an incident.

So this module does for exports what
``tests/contract/test_redaction_enumeration.py`` does for report formats,
and deliberately in the same shape:

  * ``EXTERNAL_SINKS`` names every export entry point that exists.
  * ``REDACTION_ROUTED_SINKS`` is a separate, hand-maintained list of those
    that scrub secrets first.
  * Neither is computed from the other, which is what makes the equality
    check below a real tripwire rather than a tautology. Adding a third host
    platform and forgetting that an export leaves the building fails CI.

The equality check alone would still be satisfiable by lying -- adding a
name to the routed set without wiring anything up -- so the behavioural test
below actually calls each sink and counts, so a sink can only claim to
redact by really doing it.
"""

from __future__ import annotations

import importlib
import json
import typing
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from agentic_evalkit.integrations import (
    EXTERNAL_SINKS,
    REDACTION_ROUTED_SINKS,
    TEXT_REDACTED_SINKS,
)
from agentic_evalkit.models import (
    ContaminationMetadata,
    DatasetRef,
    EvalRunManifest,
    EvalRunResult,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    SampleResult,
)
from agentic_evalkit.models.samples import GraderSpec
from agentic_evalkit.reporters import DEFAULT_REDACTION_POLICY, apply_redaction

if TYPE_CHECKING:
    from collections.abc import Iterator

_AT = datetime(2026, 8, 4, 12, 0, 0, tzinfo=UTC)
_PLANTED_TOKEN = "hf_abcdefghijklmnopqrstuvwxyz012345"  # a fake token, planted to be redacted


def _resolve(target: str) -> Any:
    """Turn a ``"module:attr"`` sink reference into the real callable."""
    module_name, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


def _run_with_a_planted_secret() -> EvalRunResult:
    sample_id = "s0"
    return EvalRunResult(
        run_id="run-001",
        manifest=EvalRunManifest(
            run_name="redaction-contract",
            dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
            adapter="gsm8k@1",
            grader="normalized-exact@1",
            target_name="echo-target",
        ),
        resolved_dataset=ResolvedDataset(dataset_id="openai/gsm8k", revision="abc123"),
        samples=(
            SampleResult(
                sample=EvalSample(
                    sample_id=sample_id,
                    input={"question": "q"},
                    source_digest=f"sha256:{sample_id}",
                    adapter="gsm8k@1",
                ),
                execution=NormalizedExecutionResult(
                    sample_id=sample_id,
                    attempt=1,
                    output={"answer": "42", "trace": f"token {_PLANTED_TOKEN}"},
                    status=ExecutionStatus.COMPLETED,
                    started_at=_AT,
                    finished_at=_AT,
                ),
                grade=GradeResult(
                    sample_id=sample_id,
                    grader="normalized-exact@1",
                    status=GradeStatus.PASS,
                    created_at=_AT,
                ),
            ),
        ),
        started_at=_AT,
    )


def test_every_external_sink_is_redaction_routed() -> None:
    # The tripwire. Because the two redaction sets are maintained by hand
    # next to EXTERNAL_SINKS rather than derived from it, adding a new host
    # platform forces whoever added it to consciously pair the export with
    # redaction instead of shipping a sink that transmits raw output.
    #
    # The partition is checked rather than a single equality because the two
    # kinds of sink cannot redact the same way: one is handed a run and
    # routes it through redact_for_export, the other is called per row by
    # the host and scrubs with redact_text. What must hold is that every
    # transmitting function is in exactly one of them.
    assert set(EXTERNAL_SINKS) == REDACTION_ROUTED_SINKS | TEXT_REDACTED_SINKS
    assert not (REDACTION_ROUTED_SINKS & TEXT_REDACTED_SINKS), (
        "a sink claiming both redaction guarantees leaves it ambiguous which one holds"
    )


def test_the_sink_registry_is_not_empty() -> None:
    """Guards against the equality above passing because both sets are empty."""
    assert EXTERNAL_SINKS


#: Every field that holds free-form payload -- text written by a caller, an
#: adapter, a dataset provider, or the system under test -- on any model
#: reachable from ``EvalRunResult``, mapped to whether the redaction sweep
#: covers it.
#:
#: This table is the second half of the tripwire, and it exists because
#: counting calls to ``redact_for_export`` proves only that the sweep *ran*,
#: never that it *reached* anything. The sweep works field by field, so the
#: way it fails is a field being added to a model and nobody extending the
#: list -- silent, because every existing test still passes.
#:
#: ``_SWEPT`` fields must be scrubbed. ``_EXEMPT`` fields are deliberately
#: left alone and each needs a reason that survives being read aloud. The
#: reflection test below fails when a model grows a free-form field that is
#: in neither, which forces the decision to be made rather than defaulted.
#:
#: The table once listed only the four per-sample models, and the reflection
#: test iterated the table, so the run-level manifest and resolved dataset
#: were never inspected at all: their policy dictionaries and dataset card
#: went into every report unswept while this test stayed green. The models
#: are now discovered by walking ``EvalRunResult``, so a model can no longer
#: be missed by not being listed.
_SWEPT: dict[type, frozenset[str]] = {
    EvalRunManifest: frozenset(
        {"artifact_policy", "redaction_policy", "baseline_compatibility_rules"}
    ),
    DatasetRef: frozenset({"data_files"}),
    ResolvedDataset: frozenset({"card_metadata", "schema_metadata", "selected_files"}),
    EvalSample: frozenset(
        {"input", "reference", "metadata", "expected_artifacts", "allowed_execution_policy"}
    ),
    GraderSpec: frozenset({"parameters"}),
    NormalizedExecutionResult: frozenset(
        {
            "output",
            "structured_output",
            "error",
            "artifacts",
            "environment_metadata",
            "tool_calls",
            "trace_refs",
        }
    ),
    GradeResult: frozenset({"evidence", "oracle_provenance", "artifact_refs"}),
}

#: Free-form-typed fields the sweep deliberately skips, and why.
_EXEMPT: dict[type, frozenset[str]] = {
    #: ``tags`` holds short structural labels an adapter assigns for
    #: filtering, never payload; rewriting one breaks selection while
    #: protecting nothing.
    EvalSample: frozenset({"tags"}),
    #: ``canary_ids`` holds marker strings that are published on purpose so
    #: a model regurgitating them can be caught; the exact value is the
    #: evidence a leak check matches on.
    ContaminationMetadata: frozenset({"canary_ids"}),
}


def _field_annotations(annotation: object) -> Iterator[object]:
    """``annotation`` and every type nested inside it (Optional, tuple items, ...)."""
    yield annotation
    for argument in typing.get_args(annotation):
        yield from _field_annotations(argument)


def _models_reachable_from(root: type[BaseModel]) -> set[type[BaseModel]]:
    """Every model class a ``root`` instance can contain, found by walking field types."""
    reachable: set[type[BaseModel]] = set()
    pending: list[type[BaseModel]] = [root]
    while pending:
        model = pending.pop()
        if model in reachable:
            continue
        reachable.add(model)
        for field in model.model_fields.values():
            for annotation in _field_annotations(field.annotation):
                if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                    pending.append(annotation)
    return reachable


_RUN_MODELS: frozenset[type[BaseModel]] = frozenset(_models_reachable_from(EvalRunResult))


def _free_form_fields(model: type) -> set[str]:
    """Fields whose declared type can hold arbitrary caller or target text.

    Keyed off the annotation rather than a hand-listed set, so the check
    cannot be satisfied by forgetting to update it: a new
    ``dict[str, JsonValue]`` or ``tuple[str, ...]`` field is detected the
    moment it is declared.
    """
    detected = set()
    for name, field in model.model_fields.items():
        annotation = str(field.annotation)
        if "JsonValue" in annotation or annotation == "tuple[str, ...]":
            detected.add(name)
    return detected


@pytest.mark.parametrize("model", sorted(_RUN_MODELS, key=lambda m: m.__name__))
def test_every_free_form_field_is_either_swept_or_deliberately_exempt(model: type) -> None:
    """Adding a free-form field to a wire model must not silently escape redaction.

    This is the check that would have caught ``allowed_execution_policy``,
    ``trace_refs``, ``artifact_refs`` and ``GraderSpec.parameters`` sitting
    outside the sweep while every other redaction test stayed green: each
    was declared on a model whose neighbouring fields were being scrubbed.
    It runs over every model reachable from a run, not over the table, so a
    model nobody thought to list is inspected too.
    """
    swept = _SWEPT.get(model, frozenset())
    exempt = _EXEMPT.get(model, frozenset())
    uncategorized = _free_form_fields(model) - swept - exempt
    assert not uncategorized, (
        f"{model.__name__} has free-form field(s) {sorted(uncategorized)} that are neither "
        f"swept by apply_redaction nor listed as deliberately exempt. Decide which, and "
        f"if swept, add the field to the sweep in reporters/base.py."
    )


def test_the_model_walk_reaches_every_categorized_model() -> None:
    """Guards the walk itself: one that stopped at the root would make the test above vacuous.

    Every model the tables mention must be one the walk found, which proves it
    descends through the manifest, the resolved dataset, and each sample's
    grader spec, execution and grade. A table entry for a model a run cannot
    contain would be dead weight, so that fails too.
    """
    categorized = set(_SWEPT) | set(_EXEMPT)
    assert categorized <= _RUN_MODELS, sorted(m.__name__ for m in categorized - _RUN_MODELS)


def _model_instances(value: object) -> Iterator[BaseModel]:
    """Every model instance inside ``value``, depth first."""
    if isinstance(value, BaseModel):
        yield value
        for name in type(value).model_fields:
            yield from _model_instances(getattr(value, name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _model_instances(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _model_instances(item)


def test_the_fixture_plants_a_secret_in_every_swept_field() -> None:
    """A swept field the fixture leaves empty is a field the behavioural test never checks.

    ``test_no_swept_field_carries_a_secret_into_the_redacted_body`` is only as
    wide as the fixture it redacts, so this pins the fixture to the table:
    every swept field, on every model that has one, carries the token.
    """
    planted: dict[type, set[str]] = {}
    for instance in _model_instances(_run_with_secrets_in_every_swept_field()):
        for name in _SWEPT.get(type(instance), frozenset()):
            if _PLANTED_TOKEN in str(getattr(instance, name)):
                planted.setdefault(type(instance), set()).add(name)
    missing = {
        model.__name__: sorted(fields - planted.get(model, set()))
        for model, fields in _SWEPT.items()
        if fields - planted.get(model, set())
    }
    assert not missing, f"swept fields the fixture never plants a secret in: {missing}"


def _run_with_secrets_in_every_swept_field() -> EvalRunResult:
    """A run carrying the planted token in every field the sweep claims to cover."""
    sample_id = "s0"
    signed_file = f"https://files.example.com/train.parquet?token={_PLANTED_TOKEN}"
    return EvalRunResult(
        run_id="run-002",
        manifest=EvalRunManifest(
            run_name="redaction-coverage",
            dataset_ref=DatasetRef(
                provider="huggingface", dataset_id="openai/gsm8k", data_files=(signed_file,)
            ),
            adapter="gsm8k@1",
            grader="normalized-exact@1",
            target_name="echo-target",
            artifact_policy={"store": f"s3://bucket/run?sig={_PLANTED_TOKEN}"},
            # A caller scrubbing one known leaked key lists the literal key.
            redaction_policy={"secret_patterns": [_PLANTED_TOKEN]},
            baseline_compatibility_rules={"tracking_token": _PLANTED_TOKEN},
        ),
        resolved_dataset=ResolvedDataset(
            dataset_id="openai/gsm8k",
            revision="abc123",
            selected_files=(signed_file,),
            schema_metadata={"note": _PLANTED_TOKEN},
            card_metadata={"description": f"access with {_PLANTED_TOKEN}"},
        ),
        samples=(
            SampleResult(
                sample=EvalSample(
                    sample_id=sample_id,
                    input={"question": f"use {_PLANTED_TOKEN}"},
                    reference=f"ref {_PLANTED_TOKEN}",
                    metadata={"note": _PLANTED_TOKEN},
                    expected_artifacts={"file": _PLANTED_TOKEN},
                    allowed_execution_policy={"tool_token": _PLANTED_TOKEN},
                    grader=GraderSpec(
                        name="external-oracle@1",
                        parameters={"api_key": _PLANTED_TOKEN},
                    ),
                    source_digest=f"sha256:{sample_id}",
                    adapter="gsm8k@1",
                ),
                execution=NormalizedExecutionResult(
                    sample_id=sample_id,
                    attempt=1,
                    output={"answer": "42", "trace": f"token {_PLANTED_TOKEN}"},
                    structured_output={"raw": _PLANTED_TOKEN},
                    error={"message": _PLANTED_TOKEN},
                    artifacts={"log": _PLANTED_TOKEN},
                    environment_metadata={"AUTH": _PLANTED_TOKEN},
                    tool_calls=({"name": "fetch", "arguments": {"key": _PLANTED_TOKEN}},),
                    trace_refs=(f"https://tracing.example.com/t/1?key={_PLANTED_TOKEN}",),
                    status=ExecutionStatus.COMPLETED,
                    started_at=_AT,
                    finished_at=_AT,
                ),
                grade=GradeResult(
                    sample_id=sample_id,
                    grader="normalized-exact@1",
                    status=GradeStatus.PASS,
                    evidence={"detail": _PLANTED_TOKEN},
                    oracle_provenance={"command": _PLANTED_TOKEN},
                    artifact_refs=(f"https://s3.example.com/o?sig={_PLANTED_TOKEN}",),
                    created_at=_AT,
                ),
            ),
        ),
        started_at=_AT,
    )


def test_no_swept_field_carries_a_secret_into_the_redacted_body() -> None:
    """The behavioural half of the table above: inspect the payload, not the call count.

    ``test_each_routed_sink_really_calls_redaction_exactly_once`` proves the
    sweep runs. It cannot prove the sweep reached anything, because it
    asserts on a counter rather than on what came out. This serialises the
    redacted run exactly as an exporter transmits it and looks for the
    planted token anywhere in it.
    """
    redacted = apply_redaction(_run_with_secrets_in_every_swept_field(), DEFAULT_REDACTION_POLICY)
    body = json.dumps(redacted.model_dump(mode="json"))

    assert _PLANTED_TOKEN not in body, (
        "a planted credential survived redaction and would have been transmitted; "
        "compare the fields set in the fixture against the sweep in reporters/base.py"
    )


@pytest.mark.parametrize("name", sorted(EXTERNAL_SINKS))
def test_every_registered_sink_resolves_to_a_real_callable(name: str) -> None:
    """A registry entry naming a function that does not exist proves nothing.

    Sinks are recorded as dotted strings rather than imported objects so the
    registry stays free of MLflow and Langfuse imports; the cost of that
    choice is that a typo would go unnoticed, which is what this closes.
    """
    assert callable(_resolve(EXTERNAL_SINKS[name]))


def test_importing_the_registry_does_not_import_any_host_library() -> None:
    """``agentic_evalkit.integrations`` must stay free to import.

    If naming the sinks pulled in MLflow, the optional extra would not be
    optional: anyone touching this package would pay for a dependency they
    may never use.
    """
    # A subprocess is the only honest way to ask this: by the time this test
    # runs, the parent interpreter has already imported mlflow for the
    # bridge tests, so checking sys.modules here would always find it.
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import agentic_evalkit.integrations; "
                "assert 'mlflow' not in sys.modules, 'importing the registry imported mlflow'; "
                "assert 'langfuse' not in sys.modules, 'importing the registry imported langfuse'"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", sorted(REDACTION_ROUTED_SINKS))
def test_each_routed_sink_really_calls_redaction_exactly_once(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """The behavioural half: claiming to redact is not the same as redacting.

    ``redact_for_export`` is patched in the sink's own module namespace and
    counted. Exactly once matters in both directions -- zero means the sink
    transmits raw output, and more than once means redaction is being
    applied to already-redacted data, which is the shape a refactor takes
    just before someone removes the "duplicate" call that was the real one.
    """
    module_name, _, attribute = EXTERNAL_SINKS[name].partition(":")
    module = importlib.import_module(module_name)
    calls: list[EvalRunResult] = []

    real = module.redact_for_export

    def counting(run: EvalRunResult, policy: object) -> EvalRunResult:
        calls.append(run)
        return real(run, policy)

    monkeypatch.setattr(module, "redact_for_export", counting)
    getattr(module, attribute)(_run_with_a_planted_secret(), **_sink_kwargs(name, tmp_path))

    assert len(calls) == 1


def _sink_kwargs(name: str, tmp_path: object) -> dict[str, Any]:
    """The minimum each sink needs to run offline, keyed by sink name.

    Each host platform needs a different kind of nowhere to write to: MLflow
    a private tracking directory, Langfuse a stand-in client, since it has
    no offline mode at all.
    """
    if name.startswith("mlflow."):
        import os
        from pathlib import Path

        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        assert isinstance(tmp_path, Path)
        return {"tracking_uri": tmp_path.joinpath("mlruns").as_uri()}
    if name.startswith("langfuse."):
        return {"client": _NullLangfuseClient()}
    raise AssertionError(f"no offline setup registered for sink {name!r}")


class _NullLangfuseClient:
    """Accepts everything and keeps nothing: this test only counts redaction calls."""

    def start_observation(self, **kwargs: Any) -> _NullLangfuseClient:
        return self

    def create_score(self, **kwargs: Any) -> None:
        return None

    def flush(self) -> None:
        return None

    def end(self) -> None:
        return None

    @property
    def id(self) -> str:
        return "obs-null"

    @property
    def trace_id(self) -> str:
        return "trace-null"
