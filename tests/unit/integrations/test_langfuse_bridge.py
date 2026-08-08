"""The Langfuse bridge, exercised against a recording stand-in client.

Unlike the MLflow bridge -- which a test can point at a real tracking store
on local disk -- Langfuse has no offline or local mode: every write goes to a
server. So these tests drive a recorder that implements the module's own
``LangfuseClient`` protocol. That is a weaker guarantee than the MLflow
suite's and it is worth being honest about which half it covers: it proves
what this bridge *sends* (names, values, score types, nesting, redaction),
and it cannot prove the server accepts it. The protocol declaration is what
carries the other half -- ``LangfuseClient`` is written against the real
installed signatures, so mypy fails here if a Langfuse release moves one.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from agentic_evalkit.graders.calibration import CalibrationArtifact
from agentic_evalkit.integrations.base import AuthorityLevel
from agentic_evalkit.integrations.langfuse import (
    LangfuseClient,
    log_eval_run,
    score_with_calibration_gate,
)
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
    RunSummary,
    SampleResult,
    SamplingPolicy,
)
from agentic_evalkit.reporters import RedactionPolicy

_STARTED_AT = datetime(2026, 8, 4, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 8, 4, 12, 5, 0, tzinfo=UTC)
_PLANTED_TOKEN = "hf_abcdefghijklmnopqrstuvwxyz012345"  # a fake token, planted to be redacted


class _RecordingObservation:
    """A stand-in span that records its children instead of transmitting them."""

    def __init__(self, name: str, trace_id: str, recorder: _RecordingClient) -> None:
        self.name = name
        self.trace_id = trace_id
        self.id = f"obs-{name}"
        self.ended = False
        self._recorder = recorder

    def start_observation(self, **kwargs: Any) -> _RecordingObservation:
        child = _RecordingObservation(str(kwargs["name"]), self.trace_id, self._recorder)
        self._recorder.observations.append({"parent": self.name, **kwargs})
        return child

    def end(self) -> None:
        self.ended = True
        self._recorder.ended.append(self.name)


class _RecordingClient:
    """Implements ``LangfuseClient`` and keeps everything it was asked to send."""

    def __init__(self) -> None:
        self.observations: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.ended: list[str] = []
        self.flushes = 0

    def start_observation(self, **kwargs: Any) -> _RecordingObservation:
        self.observations.append({"parent": None, **kwargs})
        return _RecordingObservation(str(kwargs["name"]), "trace-abc", self)

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushes += 1


def _run(
    *,
    statuses: tuple[GradeStatus | None, ...] = (GradeStatus.PASS, GradeStatus.FAIL),
    planted_secret: str | None = None,
) -> EvalRunResult:
    samples = []
    for index, status in enumerate(statuses):
        sample_id = f"gsm8k:main:test:{index}"
        errored = status is None
        output: dict[str, object] | None = None
        if not errored:
            output = {"answer": "42"}
            if planted_secret is not None and index == 0:
                output = {"answer": "42", "trace": f"token {planted_secret}"}
        samples.append(
            SampleResult(
                sample=EvalSample(
                    sample_id=sample_id,
                    input={"question": f"q{index}"},
                    reference="42",
                    source_digest=f"sha256:{sample_id}",
                    adapter="gsm8k@1",
                ),
                execution=NormalizedExecutionResult(
                    sample_id=sample_id,
                    attempt=1,
                    output=output,
                    status=ExecutionStatus.ERROR if errored else ExecutionStatus.COMPLETED,
                    started_at=_STARTED_AT,
                    finished_at=_FINISHED_AT,
                ),
                grade=(
                    None
                    if errored
                    else GradeResult(
                        sample_id=sample_id,
                        grader="normalized-exact@1",
                        status=status,
                        score=1.0 if status is GradeStatus.PASS else 0.0,
                        created_at=_FINISHED_AT,
                    )
                ),
            )
        )
    return EvalRunResult(
        run_id="run-001",
        manifest=EvalRunManifest(
            run_name="gsm8k-smoke",
            dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
            adapter="gsm8k@1",
            grader="normalized-exact@1",
            target_name="echo-target",
            sampling=SamplingPolicy(seed=7, attempts=1),
            attempts=1,
        ),
        resolved_dataset=ResolvedDataset(
            dataset_id="openai/gsm8k", revision="abc123", config="main", split="test"
        ),
        samples=tuple(samples),
        summary=RunSummary(total=len(statuses)),
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _calibration(
    *,
    calibrated_at: datetime | None = _STARTED_AT - timedelta(days=1),
    expires_at: datetime = _STARTED_AT + timedelta(days=30),
    true_negative: int = 400,
    false_positive: int = 2,
) -> CalibrationArtifact:
    """Sized to clear the Wilson lower bound, not merely the raw rate."""
    return CalibrationArtifact(
        calibration_id="cal-001",
        judge_fingerprint="sha256:judge-a",
        expires_at=expires_at,
        calibrated_at=calibrated_at,
        true_positive=200,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=5,
        threshold=0.8,
    )


def test_the_recorder_really_satisfies_the_declared_client_protocol() -> None:
    """Otherwise every test below is passing against a shape Langfuse never had."""
    assert isinstance(_RecordingClient(), LangfuseClient)


def test_the_real_langfuse_client_satisfies_the_declared_protocol() -> None:
    """The half the recorder cannot prove: that the protocol matches Langfuse.

    Every other test here drives a stand-in, so they would all keep passing
    if ``LangfuseClient`` described a client Langfuse never shipped. This is
    the only assertion that touches the real class. It builds one against an
    unroutable host with tracing disabled -- construction alone contacts
    nothing, and no export is performed.

    Note the limit of what this proves: ``runtime_checkable`` protocols check
    method *names*, not signatures. Argument-level drift is caught by mypy
    over ``integrations/langfuse.py``, not here.
    """
    langfuse = pytest.importorskip("langfuse", reason="the langfuse extra is not installed")

    client = langfuse.Langfuse(
        public_key="pk-lf-not-a-real-key",
        secret_key="sk-lf-not-a-real-key",  # noqa: S106
        host="http://127.0.0.1:1",
        tracing_enabled=False,
    )
    assert isinstance(client, LangfuseClient)


def test_export_creates_a_root_observation_with_one_child_per_sample() -> None:
    client = _RecordingClient()
    log_eval_run(_run(), client=client)

    roots = [obs for obs in client.observations if obs["parent"] is None]
    children = [obs for obs in client.observations if obs["parent"] is not None]
    assert len(roots) == 1
    assert roots[0]["name"] == "gsm8k-smoke"
    assert len(children) == 2
    assert {child["name"] for child in children} == {
        "sample:gsm8k:main:test:0",
        "sample:gsm8k:main:test:1",
    }


def test_root_metadata_carries_provenance_and_the_full_run_body() -> None:
    """Langfuse has no artifact store, so the body has to ride in metadata.

    Without it the export is a summary rather than evidence: a reader could
    see the pass rate but could not re-grade or reproduce it.
    """
    client = _RecordingClient()
    log_eval_run(_run(), client=client)
    metadata = client.observations[0]["metadata"]

    assert metadata["provenance"]["manifest.adapter"] == "gsm8k@1"
    assert metadata["provenance"]["dataset.revision"] == "abc123"
    assert metadata["run"]["run_id"] == "run-001"
    assert metadata["summary"]["total"] == 2


def test_every_observation_is_closed_even_though_scoring_runs_between() -> None:
    client = _RecordingClient()
    log_eval_run(_run(), client=client)

    assert "gsm8k-smoke" in client.ended
    assert len([name for name in client.ended if name.startswith("sample:")]) == 2


def test_graded_outcomes_become_numeric_scores() -> None:
    client = _RecordingClient()
    log_eval_run(_run(statuses=(GradeStatus.PASS, GradeStatus.FAIL)), client=client)

    numeric = [score for score in client.scores if score["name"] == "evalkit.grade"]
    assert sorted(score["value"] for score in numeric) == [0.0, 1.0]
    assert {score["data_type"] for score in numeric} == {"NUMERIC"}


def test_an_ungraded_sample_never_becomes_a_zero_numeric_score() -> None:
    """The failure mode this bridge could most easily introduce.

    Langfuse averages numeric scores by name across a project. A sample that
    errored before grading, recorded as ``evalkit.grade = 0.0``, would drag
    down every dashboard built on that score while looking like a genuine
    wrong answer. It has to land on a separate categorical score instead,
    where nothing can average it.
    """
    client = _RecordingClient()
    log_eval_run(_run(statuses=(GradeStatus.PASS, None)), client=client)

    numeric = [score for score in client.scores if score["name"] == "evalkit.grade"]
    categorical = [score for score in client.scores if score["name"] == "evalkit.grade_status"]

    assert [score["value"] for score in numeric] == [1.0]
    assert len(categorical) == 1
    assert categorical[0]["value"] == "error"
    assert categorical[0]["data_type"] == "CATEGORICAL"


def test_a_grader_that_abstained_is_also_kept_out_of_the_numeric_score() -> None:
    client = _RecordingClient()
    log_eval_run(_run(statuses=(GradeStatus.ABSTAIN,)), client=client)

    assert [score for score in client.scores if score["name"] == "evalkit.grade"] == []
    categorical = [score for score in client.scores if score["name"] == "evalkit.grade_status"]
    assert categorical[0]["value"] == "abstain"


def test_secrets_are_scrubbed_before_anything_is_handed_to_the_client() -> None:
    client = _RecordingClient()
    log_eval_run(_run(planted_secret=_PLANTED_TOKEN), client=client)

    transmitted = repr(client.observations)
    assert _PLANTED_TOKEN not in transmitted
    assert "[REDACTED]" in transmitted


def test_redaction_can_be_opted_out_of_deliberately() -> None:
    client = _RecordingClient()
    log_eval_run(
        _run(planted_secret=_PLANTED_TOKEN),
        client=client,
        redaction_policy=RedactionPolicy(),
    )

    assert _PLANTED_TOKEN in repr(client.observations)


def test_export_flushes_by_default() -> None:
    """The client batches in the background, so a CI job would otherwise exit first.

    Losing an export because the process ended before the batch was sent is
    a silent failure: the command succeeds and nothing arrives.
    """
    client = _RecordingClient()
    log_eval_run(_run(), client=client)
    assert client.flushes == 1


def test_flush_can_be_left_to_a_caller_managing_the_client_lifecycle() -> None:
    client = _RecordingClient()
    log_eval_run(_run(), client=client, flush=False)
    assert client.flushes == 0


def test_calibration_summary_travels_with_the_export() -> None:
    client = _RecordingClient()
    log_eval_run(_run(), client=client, calibration=_calibration(), now=_STARTED_AT)

    judge = client.observations[0]["metadata"]["judge"]
    assert judge["authority"] == "gating"
    assert judge["can_gate"] is True
    assert judge["calibration_id"] == "cal-001"


# --- score_with_calibration_gate -------------------------------------------


def test_a_calibrated_judge_scores_under_the_ungated_name() -> None:
    client = _RecordingClient()
    level = score_with_calibration_gate(
        client, name="faithfulness", value=0.9, calibration=_calibration()
    )

    assert level is AuthorityLevel.GATING
    assert client.scores[0]["name"] == "faithfulness"
    assert client.scores[0]["value"] == 0.9


def test_an_uncalibrated_judge_scores_under_a_suffixed_name() -> None:
    """Renaming is the demotion. Metadata alone would not be enough.

    Langfuse aggregates by score name, so an advisory value written under
    the gating name has already moved the aggregate by the time anyone reads
    the metadata explaining that it should not have.
    """
    client = _RecordingClient()
    level = score_with_calibration_gate(client, name="faithfulness", value=0.9, calibration=None)

    assert level is AuthorityLevel.ADVISORY
    assert client.scores[0]["name"] == "faithfulness.advisory"
    assert client.scores[0]["value"] == 0.9
    assert client.scores[0]["metadata"]["evalkit_can_gate"] is False


def test_a_judge_proven_unreliable_writes_no_numeric_score_at_all() -> None:
    """Asserting only on ``scores[0]`` would let the withheld value through.

    Deleting the early return would leave the categorical marker first and a
    numeric 0.9 second, and a test reading only the first entry would still
    pass while the verdict it claims was withheld sat in the aggregate. So
    this asserts over the whole list.
    """
    client = _RecordingClient()
    level = score_with_calibration_gate(
        client,
        name="faithfulness",
        value=0.9,
        calibration=_calibration(true_negative=30, false_positive=40),
    )

    assert level is AuthorityLevel.UNAVAILABLE
    assert len(client.scores) == 1
    assert client.scores[0]["name"] == "faithfulness.unavailable"
    assert client.scores[0]["data_type"] == "CATEGORICAL"
    assert client.scores[0]["value"] == "unavailable"
    assert not [s for s in client.scores if s["data_type"] == "NUMERIC"]
    assert 0.9 not in [s["value"] for s in client.scores]


def test_every_sample_score_names_the_trace_it_belongs_to() -> None:
    """An observation ID alone does not say what a score is attached to.

    An observation is a span *within* a trace, so a score carrying only
    ``observation_id`` is under-specified for the API and risks being filed
    against nothing -- silently losing the outcome the export exists to
    record.
    """
    client = _RecordingClient()
    log_eval_run(_run(statuses=(GradeStatus.PASS, None)), client=client)

    assert client.scores
    for score in client.scores:
        assert score["trace_id"], f"score {score['name']} carries no trace_id"
        assert score["observation_id"]


def test_the_reason_for_a_demotion_is_recorded_where_a_human_will_see_it() -> None:
    client = _RecordingClient()
    score_with_calibration_gate(
        client,
        name="faithfulness",
        value=0.9,
        calibration=None,
        comment="nightly regression",
    )

    comment = client.scores[0]["comment"]
    assert "nightly regression" in comment
    assert "advisory-only" in comment


def test_a_string_verdict_is_written_as_a_categorical_score() -> None:
    client = _RecordingClient()
    score_with_calibration_gate(
        client, name="verdict", value="grounded", calibration=_calibration(), now=_STARTED_AT
    )

    assert client.scores[0]["data_type"] == "CATEGORICAL"
    # Pinned to the gating name as well as the data type: the UNAVAILABLE
    # path also writes CATEGORICAL, so asserting the type alone would pass
    # even if this judge had been withheld for the wrong reason.
    assert client.scores[0]["name"] == "verdict"


def test_the_comment_is_scrubbed_before_it_reaches_langfuse() -> None:
    """``comment`` is caller free text going to a shared server.

    This function is handed one score, never a run, so ``redact_for_export``
    has nothing to operate on and this is the only sweep available -- the
    same position ``as_mlflow_scorer`` is in with its rationale. The obvious
    way to build a comment is from target output or an exception message,
    both well-trodden routes for a credential to reach a string.
    """
    client = _RecordingClient()
    score_with_calibration_gate(
        client,
        name="faithfulness",
        value=0.9,
        calibration=_calibration(),
        comment=f"replayed with {_PLANTED_TOKEN}",
        now=_STARTED_AT,
    )

    comment = client.scores[0]["comment"]
    assert _PLANTED_TOKEN not in comment
    assert "[REDACTED]" in comment


def test_the_gate_can_be_pinned_to_an_instant_like_every_other_calibration_call() -> None:
    """Without ``now`` this function reads the wall clock and cannot be tested.

    Two consequences, and the second is the one that bites: expiry and the
    90-day age limit are unreachable in a hermetic test, and any test using
    a fixture whose ``expires_at`` is a fixed date silently becomes a time
    bomb that goes red the day it passes.
    """
    calibration = _calibration(
        calibrated_at=_STARTED_AT - timedelta(days=1),
        expires_at=_STARTED_AT + timedelta(days=30),
    )

    before = _RecordingClient()
    assert (
        score_with_calibration_gate(
            before, name="f", value=0.9, calibration=calibration, now=_STARTED_AT
        )
        is AuthorityLevel.GATING
    )

    after = _RecordingClient()
    assert (
        score_with_calibration_gate(
            after,
            name="f",
            value=0.9,
            calibration=calibration,
            now=_STARTED_AT + timedelta(days=31),
        )
        is AuthorityLevel.UNAVAILABLE
    )
    assert after.scores[0]["name"] == "f.unavailable"


def test_a_failure_partway_through_still_flushes_what_was_already_built() -> None:
    """A buffered client plus an exception equals a silently lost export.

    Langfuse batches in a background thread, so an exception escaping with
    the buffer unflushed discards every span and score recorded before the
    failure. A caller treating the error as fatal then exits and takes the
    evidence of what went wrong with it. Langfuse has no equivalent of
    MLflow's terminal run status, so the partial trace is the only thing a
    failed export leaves behind.
    """

    class _FailingClient(_RecordingClient):
        def create_score(self, **kwargs: Any) -> None:
            super().create_score(**kwargs)
            raise RuntimeError("langfuse went away")

    client = _FailingClient()
    with pytest.raises(RuntimeError, match="langfuse went away"):
        log_eval_run(_run(), client=client)

    assert client.flushes == 1, "the partial export was never flushed and is lost"


def test_missing_langfuse_is_only_an_error_when_no_client_was_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing a client must not require the package to be installed at all.

    This is what keeps the extra genuinely optional: a caller who already
    holds a configured client -- or a test harness like this one -- never
    triggers the import.
    """
    import builtins

    real_import = builtins.__import__

    def refuse_langfuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse":
            raise ImportError("no langfuse here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_langfuse)

    client = _RecordingClient()
    assert log_eval_run(_run(), client=client) == "trace-abc"
