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
from datetime import UTC, datetime
from typing import Any

import pytest

from agentic_evalkit.integrations import EXTERNAL_SINKS, REDACTION_ROUTED_SINKS
from agentic_evalkit.models import (
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
    # The tripwire. Because REDACTION_ROUTED_SINKS is maintained by hand
    # next to EXTERNAL_SINKS rather than derived from it, adding a new host
    # platform forces whoever added it to consciously pair the export with
    # redaction instead of shipping a sink that transmits raw output.
    assert set(EXTERNAL_SINKS) == set(REDACTION_ROUTED_SINKS)


def test_the_sink_registry_is_not_empty() -> None:
    """Guards against the equality above passing because both sets are empty."""
    assert EXTERNAL_SINKS


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
