"""The MLflow bridge, exercised against a real MLflow store on local disk.

These are not mock tests. Every assertion below goes through the actual
MLflow client into an actual tracking store in a temporary directory, and
reads back what MLflow really persisted -- no network, no server, and no
stand-in for the library's behaviour. That matters here more than usual,
because the failures this bridge can have are precisely the ones a mock
cannot show you: MLflow coercing every param to a string, silently dropping
a metric it will not accept, or storing a tag under a name that search
cannot find.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agentic_evalkit.errors import IncompatibleRuns
from agentic_evalkit.graders.calibration import CalibrationArtifact
from agentic_evalkit.integrations.mlflow import (
    CALIBRATION_ARTIFACT_PATH,
    RUN_ARTIFACT_PATH,
    as_mlflow_scorer,
    calibration_gate,
    compare_mlflow_runs,
    log_eval_run,
)
from agentic_evalkit.models import (
    DatasetRef,
    DatasetSelection,
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

mlflow = pytest.importorskip("mlflow", reason="the mlflow extra is not installed")

_STARTED_AT = datetime(2026, 8, 4, 12, 0, 0, tzinfo=UTC)
_FINISHED_AT = datetime(2026, 8, 4, 12, 5, 0, tzinfo=UTC)

#: A string shaped exactly like a Hugging Face token, which
#: DEFAULT_REDACTION_POLICY matches. Planted in target output so the
#: redaction assertion below is testing a real pattern rather than a
#: hand-picked one that happens to work.
_PLANTED_TOKEN = "hf_abcdefghijklmnopqrstuvwxyz012345"  # a fake token, planted to be redacted


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A private, file-backed MLflow store for one test, with no server involved.

    MLflow 3.x puts its filesystem backend in maintenance mode and refuses
    it unless ``MLFLOW_ALLOW_FILE_STORE`` is set. That opt-out is exactly
    what makes a hermetic test possible: the alternative backends are a
    SQL database or a running tracking server, and neither belongs in a unit
    suite. ``monkeypatch`` scopes the variable to the test.
    """
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return tmp_path.joinpath("mlruns").as_uri()


def _run(
    *,
    run_id: str = "run-001",
    adapter: str = "gsm8k@1",
    seed: int | None = 7,
    statuses: tuple[GradeStatus | None, ...] = (
        GradeStatus.PASS,
        GradeStatus.PASS,
        GradeStatus.FAIL,
        None,
    ),
    planted_secret: str | None = None,
) -> EvalRunResult:
    """Build a run whose sample outcomes the caller chooses.

    ``None`` in ``statuses`` means the sample errored during execution and
    was never graded -- the case that makes the ADR-0008 assertions below
    meaningful, since an operational failure must not become a task failure.
    """
    samples = []
    for index, status in enumerate(statuses):
        sample_id = f"gsm8k:main:test:{index}"
        errored = status is None
        output: dict[str, object] | None = None
        if not errored:
            output = {"answer": "42"}
            if planted_secret is not None and index == 0:
                output = {"answer": "42", "trace": f"called api with {planted_secret}"}
        samples.append(
            SampleResult(
                sample=EvalSample(
                    sample_id=sample_id,
                    input={"question": f"q{index}"},
                    reference="42",
                    source_digest=f"sha256:{sample_id}",
                    adapter=adapter,
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
                        evidence={"reason": "compared against the reference"},
                        created_at=_FINISHED_AT,
                    )
                ),
            )
        )
    return EvalRunResult(
        run_id=run_id,
        manifest=EvalRunManifest(
            run_name="gsm8k-smoke",
            dataset_ref=DatasetRef(provider="huggingface", dataset_id="openai/gsm8k"),
            adapter=adapter,
            grader="normalized-exact@1",
            target_name="echo-target",
            target_fingerprint="sha256:target-a",
            selection=DatasetSelection(offset=0, limit=4),
            sampling=SamplingPolicy(seed=seed, attempts=1),
            attempts=1,
            environment_fingerprint="sha256:env-a",
            code_fingerprint="sha256:code-a",
        ),
        resolved_dataset=ResolvedDataset(
            dataset_id="openai/gsm8k",
            revision="abc123",
            config="main",
            split="test",
            row_count=4,
        ),
        samples=tuple(samples),
        summary=RunSummary(total=4, passed=2, failed=1, errors=1),
        started_at=_STARTED_AT,
        finished_at=_FINISHED_AT,
    )


def _fetch(tracking_uri: str, run_id: str) -> mlflow.entities.Run:
    return mlflow.client.MlflowClient(tracking_uri=tracking_uri).get_run(run_id)


# --- Phase 1: exporting a run ----------------------------------------------


def test_export_writes_manifest_params_and_recounted_metrics(tracking_uri: str) -> None:
    mlflow_run_id = log_eval_run(_run(), tracking_uri=tracking_uri, experiment="evk")
    data = _fetch(tracking_uri, mlflow_run_id).data

    assert data.params["evalkit.adapter"] == "gsm8k@1"
    assert data.params["evalkit.grader"] == "normalized-exact@1"
    assert data.params["evalkit.dataset.revision"] == "abc123"
    # MLflow stores every param as a string, including numbers. Asserting the
    # rendered form pins what a consumer actually reads back.
    assert data.params["evalkit.sampling.seed"] == "7"

    assert data.metrics["evalkit.summary.total"] == 4.0
    assert data.metrics["evalkit.summary.passed"] == 2.0
    assert data.metrics["evalkit.summary.errors"] == 1.0


def test_operational_failure_is_exported_as_its_own_outcome_not_as_a_task_failure(
    tracking_uri: str,
) -> None:
    """ADR-0008, checked where an export is most likely to lose it.

    The fixture has four samples: two pass, one is graded FAIL, and one
    errors during execution and is never graded. The property that must
    survive the trip into MLflow is that those last two stay distinct --
    ``failed`` counts exactly the one sample a grader judged wrong, and the
    crashed sample lands in ``errors``. Collapsing them would report a
    harness that broke as a system that got the answer wrong, which is the
    misreading the whole outcome taxonomy exists to prevent.

    The pass rate itself keeps the package's own definition from
    ``aggregate_run`` -- ``passed / total``, with the errored sample in the
    denominator -- rather than a second definition invented here. That is
    the conservative direction: an unreliable harness drags the rate down
    instead of quietly vanishing from it, and a caller who wants the
    graded-only view has the separate counters to compute it from.
    """
    mlflow_run_id = log_eval_run(_run(), tracking_uri=tracking_uri)
    metrics = _fetch(tracking_uri, mlflow_run_id).data.metrics

    assert metrics["evalkit.summary.failed"] == 1.0
    assert metrics["evalkit.summary.errors"] == 1.0
    assert metrics["evalkit.summary.passed"] == 2.0
    assert metrics["evalkit.summary.total"] == 4.0

    assert metrics["evalkit.pass_rate.numerator"] == 2.0
    assert metrics["evalkit.pass_rate.denominator"] == 4.0
    assert metrics["evalkit.pass_rate"] == pytest.approx(0.5)


def test_pass_rate_is_exported_with_its_confidence_interval(tracking_uri: str) -> None:
    """A rate without an interval invites 0.67-from-3 to be read as a finding."""
    mlflow_run_id = log_eval_run(_run(), tracking_uri=tracking_uri)
    metrics = _fetch(tracking_uri, mlflow_run_id).data.metrics

    assert "evalkit.pass_rate.lower_bound" in metrics
    assert "evalkit.pass_rate.upper_bound" in metrics
    assert metrics["evalkit.pass_rate.lower_bound"] < metrics["evalkit.pass_rate"]
    assert metrics["evalkit.pass_rate.upper_bound"] > metrics["evalkit.pass_rate"]


def test_absent_measurements_are_omitted_rather_than_logged_as_zero(tracking_uri: str) -> None:
    """A metric MLflow never received is honest; one defaulted to 0.0 is invented.

    No sample in this fixture records latency or cost, so those metrics must
    simply not exist on the run -- writing 0.0 would put a fabricated
    observation in a chart.
    """
    mlflow_run_id = log_eval_run(_run(), tracking_uri=tracking_uri)
    metrics = _fetch(tracking_uri, mlflow_run_id).data.metrics

    assert not [key for key in metrics if "latency" in key or "cost" in key]


def test_secrets_in_target_output_are_scrubbed_before_transmission(tracking_uri: str) -> None:
    """The whole export must be redacted, not just the parts we remembered.

    This asserts against the serialized artifact body rather than against a
    field, because the failure being guarded is a secret surviving anywhere
    in what left the machine.
    """
    mlflow_run_id = log_eval_run(
        _run(planted_secret=_PLANTED_TOKEN),
        tracking_uri=tracking_uri,
    )
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    local = client.download_artifacts(mlflow_run_id, RUN_ARTIFACT_PATH)
    body = Path(local).read_text(encoding="utf-8")

    assert _PLANTED_TOKEN not in body
    assert "[REDACTED]" in body


def test_redaction_can_be_opted_out_of_deliberately(tracking_uri: str) -> None:
    """``RedactionPolicy()`` is the supported opt-out, and it must really opt out.

    Without this, a caller who genuinely needs raw output could not tell
    whether their policy was being honoured or silently overridden.
    """
    mlflow_run_id = log_eval_run(
        _run(planted_secret=_PLANTED_TOKEN),
        tracking_uri=tracking_uri,
        redaction_policy=RedactionPolicy(),
    )
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    local = client.download_artifacts(mlflow_run_id, RUN_ARTIFACT_PATH)

    assert _PLANTED_TOKEN in Path(local).read_text(encoding="utf-8")


def test_exported_run_body_round_trips_back_into_the_model(tracking_uri: str) -> None:
    """The artifact has to be re-parseable, or comparison later is impossible."""
    original = _run()
    mlflow_run_id = log_eval_run(original, tracking_uri=tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    local = client.download_artifacts(mlflow_run_id, RUN_ARTIFACT_PATH)

    restored = EvalRunResult.model_validate_json(Path(local).read_text(encoding="utf-8"))
    assert restored.run_id == original.run_id
    assert restored.manifest.adapter == original.manifest.adapter
    assert len(restored.samples) == len(original.samples)


def test_export_does_not_disturb_the_callers_mlflow_configuration(tracking_uri: str) -> None:
    """A library that reconfigures your process as a side effect is a trap.

    ``mlflow.set_tracking_uri`` and ``set_experiment`` both mutate
    process-global state that outlives the call, so an export built on the
    fluent API would silently redirect the caller's own logging afterwards.
    """
    before = mlflow.get_tracking_uri()
    log_eval_run(_run(), tracking_uri=tracking_uri, experiment="evk")

    assert mlflow.get_tracking_uri() == before
    assert mlflow.active_run() is None


def test_export_works_inside_the_callers_own_active_run(tracking_uri: str) -> None:
    """Exporting must not require the caller to have no run open.

    ``mlflow.start_run`` refuses to start while another run is active unless
    told to nest, so a fluent-API export would raise here. Going through the
    client sidesteps that entirely: this creates a sibling run and leaves
    the caller's own run active and untouched.
    """
    mlflow.set_tracking_uri(tracking_uri)
    try:
        with mlflow.start_run(run_name="callers-own") as caller_run:
            exported = log_eval_run(_run(), tracking_uri=tracking_uri)
            assert exported != caller_run.info.run_id
            assert mlflow.active_run() is not None
            assert mlflow.active_run().info.run_id == caller_run.info.run_id
    finally:
        mlflow.set_tracking_uri(None)


def test_calibration_artifact_travels_with_the_result_it_backed(tracking_uri: str) -> None:
    calibration = _good_calibration()
    mlflow_run_id = log_eval_run(_run(), tracking_uri=tracking_uri, calibration=calibration)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)

    local = client.download_artifacts(mlflow_run_id, CALIBRATION_ARTIFACT_PATH)
    restored = CalibrationArtifact.model_validate_json(Path(local).read_text(encoding="utf-8"))
    assert restored.calibration_id == calibration.calibration_id

    tags = client.get_run(mlflow_run_id).data.tags
    assert tags["evalkit.judge.authority"] == "gating"


def test_uncalibrated_judge_is_tagged_advisory_on_the_exported_run(tracking_uri: str) -> None:
    """The authority a judge claimed must be readable next to its results."""
    mlflow_run_id = log_eval_run(
        _run(),
        tracking_uri=tracking_uri,
        calibration=_good_calibration(calibrated_at=None),
    )
    tags = _fetch(tracking_uri, mlflow_run_id).data.tags

    assert tags["evalkit.judge.authority"] == "advisory"
    assert "calibrated_at" in tags["evalkit.judge.authority_reason"]


# --- Phase 3: provenance tags and comparison -------------------------------


def test_every_field_compare_runs_checks_is_exported_as_a_tag(tracking_uri: str) -> None:
    """The exported provenance surface must not be narrower than the enforced one.

    Derived from ``comparability_snapshot`` rather than a hand-written list
    precisely so this stays true when a provenance field is added.
    """
    from agentic_evalkit.stats import DATASET_IDENTITY_FIELDS_CHECKED, PROVENANCE_FIELDS_CHECKED

    mlflow_run_id = log_eval_run(_run(), tracking_uri=tracking_uri)
    tags = _fetch(tracking_uri, mlflow_run_id).data.tags

    for field in PROVENANCE_FIELDS_CHECKED:
        assert f"evalkit.provenance.manifest.{field}" in tags
    for field in DATASET_IDENTITY_FIELDS_CHECKED:
        assert f"evalkit.provenance.dataset.{field}" in tags


def test_two_identical_runs_compare_and_report_a_delta(tracking_uri: str) -> None:
    left = log_eval_run(_run(run_id="run-a"), tracking_uri=tracking_uri)
    right = log_eval_run(_run(run_id="run-b"), tracking_uri=tracking_uri)

    result = compare_mlflow_runs(left, right, seed=1234, tracking_uri=tracking_uri)

    assert result.estimate == 0.0
    assert result.paired_count == 4
    assert result.seed == 1234


def test_a_real_improvement_shows_up_as_a_positive_delta(tracking_uri: str) -> None:
    baseline = log_eval_run(
        _run(run_id="run-a", statuses=(GradeStatus.FAIL, GradeStatus.FAIL)),
        tracking_uri=tracking_uri,
    )
    candidate = log_eval_run(
        _run(run_id="run-b", statuses=(GradeStatus.PASS, GradeStatus.PASS)),
        tracking_uri=tracking_uri,
    )

    result = compare_mlflow_runs(baseline, candidate, seed=1234, tracking_uri=tracking_uri)
    assert result.estimate == 1.0


def test_comparison_is_refused_when_the_adapter_differs(tracking_uri: str) -> None:
    """The refusal is the product. A delta across two adapters is meaningless."""
    left = log_eval_run(_run(run_id="run-a", adapter="gsm8k@1"), tracking_uri=tracking_uri)
    right = log_eval_run(_run(run_id="run-b", adapter="gsm8k@2"), tracking_uri=tracking_uri)

    with pytest.raises(IncompatibleRuns, match="adapter"):
        compare_mlflow_runs(left, right, seed=1234, tracking_uri=tracking_uri)


def test_comparison_is_refused_when_the_sampling_seed_differs(tracking_uri: str) -> None:
    """Also pins that the cheap tag pre-check is what refuses this pair.

    The message names the provenance *field* (``manifest.sampling.seed``)
    rather than the human label ``compare_runs`` would use ("sampling seed
    differs"), which is the observable difference between the two refusal
    paths. That ordering is the intended one: a mismatched pair should fail
    after one metadata read, not after downloading two full run bodies.
    """
    left = log_eval_run(_run(run_id="run-a", seed=7), tracking_uri=tracking_uri)
    right = log_eval_run(_run(run_id="run-b", seed=99), tracking_uri=tracking_uri)

    with pytest.raises(IncompatibleRuns, match=re.escape("manifest.sampling.seed differs")):
        compare_mlflow_runs(left, right, seed=1234, tracking_uri=tracking_uri)


def test_comparison_is_refused_against_a_run_this_package_never_exported(
    tracking_uri: str,
) -> None:
    """A tracking server is full of other people's runs, and they are not comparable.

    There is no manifest and no provenance on a foreign run, so no claim
    about comparability can be supported -- refusing is the only honest
    answer available.
    """
    mlflow.set_tracking_uri(tracking_uri)
    try:
        with mlflow.start_run(run_name="somebody-elses-run") as foreign:
            foreign_id = foreign.info.run_id
    finally:
        mlflow.set_tracking_uri(None)

    ours = log_eval_run(_run(), tracking_uri=tracking_uri)

    with pytest.raises(IncompatibleRuns, match="not exported by agentic-evalkit"):
        compare_mlflow_runs(ours, foreign_id, seed=1234, tracking_uri=tracking_uri)


def test_comparison_requires_a_seed() -> None:
    """Keyword-only and no default, exactly as ``compare_runs`` requires it.

    A comparison read off a shared tracking server is the *most* likely one
    to be quoted later, so it is the last place a silently irreproducible
    number should be possible.
    """
    with pytest.raises(TypeError, match="seed"):
        compare_mlflow_runs("a", "b")  # type: ignore[call-arg]


# --- Phase 2: graders and judges as scorers --------------------------------


def _good_calibration(
    *,
    calibrated_at: datetime | None = _STARTED_AT - timedelta(days=1),
    expires_at: datetime = _STARTED_AT + timedelta(days=30),
    true_negative: int = 400,
    false_positive: int = 2,
) -> CalibrationArtifact:
    """Calibration large enough to clear the Wilson lower bound, not just the raw rate.

    See the equivalent fixture in test_integration_base.py: raw rates well
    above the floors still fail to gate on small samples, because the
    conservative lower bound is what the project floor is applied to.
    """
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


def test_calibrated_judge_verdict_passes_through_and_is_marked_gating() -> None:
    gated = calibration_gate(
        lambda **_: True,
        calibration=_good_calibration(),
        name="my_judge",
        now=_STARTED_AT,
    )
    feedback = gated(inputs={"q": "x"}, outputs={"a": "y"})

    assert feedback.value is True
    assert feedback.metadata["evalkit_authority"] == "gating"
    assert feedback.metadata["evalkit_can_gate"] == "true"


def test_uncalibrated_judge_still_answers_but_cannot_gate() -> None:
    """Demotion, not suppression. The verdict may well be right.

    Silencing an uncalibrated judge would cost a team all their existing
    signal the day they adopt the gate; reporting it while marking it
    unable to gate costs them nothing and tells the truth.
    """
    gated = calibration_gate(lambda **_: True, calibration=None, name="my_judge")
    feedback = gated(inputs={"q": "x"}, outputs={"a": "y"})

    assert feedback.value is True
    assert feedback.metadata["evalkit_authority"] == "advisory"
    assert feedback.metadata["evalkit_can_gate"] == "false"
    assert "advisory-only" in feedback.metadata["evalkit_authority_reason"]


def test_judge_with_expired_calibration_has_its_verdict_withheld() -> None:
    """Evidence present and bad: no value at all, only an error.

    MLflow keeps a feedback error out of any aggregate it computes, which is
    the point -- a verdict from a judge proven untrustworthy must not be
    quietly averaged into a dashboard.
    """
    gated = calibration_gate(
        lambda **_: True,
        calibration=_good_calibration(
            calibrated_at=_STARTED_AT - timedelta(days=200),
            expires_at=_STARTED_AT - timedelta(days=1),
        ),
        name="my_judge",
        now=_STARTED_AT,
    )
    feedback = gated(inputs={"q": "x"}, outputs={"a": "y"})

    assert feedback.value is None
    assert feedback.error is not None
    assert feedback.metadata["evalkit_authority"] == "unavailable"


def test_a_judge_proven_unreliable_is_never_even_called() -> None:
    """Asking a judge whose answer cannot be used bills the caller for nothing.

    With an LLM judge that call is a real API charge and real latency, so
    skipping it is a correctness property worth pinning, not an
    optimization.
    """
    calls: list[object] = []

    def expensive_judge(**kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    gated = calibration_gate(
        expensive_judge,
        calibration=_good_calibration(true_negative=30, false_positive=40),
        name="my_judge",
        now=_STARTED_AT,
    )
    gated(inputs={"q": "x"}, outputs={"a": "y"})

    assert calls == []


def test_gating_preserves_metadata_the_wrapped_judge_already_reported() -> None:
    """Wrapping must add authority, never discard what the judge said."""
    from mlflow.entities import Feedback

    gated = calibration_gate(
        lambda **_: Feedback(name="inner", value=0.9, metadata={"judge_model": "gpt-x"}),
        calibration=_good_calibration(),
        name="my_judge",
        now=_STARTED_AT,
    )
    feedback = gated(inputs={"q": "x"}, outputs={"a": "y"})

    assert feedback.metadata["judge_model"] == "gpt-x"
    assert feedback.metadata["evalkit_authority"] == "gating"


def test_evalkit_grader_becomes_a_working_mlflow_scorer() -> None:
    from agentic_evalkit.graders.exact import ExactMatchGrader

    scorer = as_mlflow_scorer(
        ExactMatchGrader(
            name="normalized-exact@1",
            extractor=lambda output: str(output.get("answer", "")),
        ),
        name="evk_exact",
    )
    feedback = scorer(
        inputs={"question": "2+2?"},
        outputs={"answer": "42"},
        expectations={"expected_response": "42"},
    )

    assert feedback.metadata["evalkit_status"] == "pass"
    assert feedback.error is None


def test_a_non_verdict_grade_becomes_an_error_not_a_failing_score() -> None:
    """ADR-0008 again, at the scorer boundary.

    ABSTAIN says the grader declined; it does not say the system under test
    got the answer wrong. Rendering it as ``False`` would fold a grading
    outcome into a task outcome, and MLflow would then average it into the
    score as though the system had failed.
    """

    class _AbstainingGrader:
        async def grade(
            self, sample: EvalSample, execution: NormalizedExecutionResult
        ) -> GradeResult:
            return GradeResult(
                sample_id=sample.sample_id,
                grader="abstaining@1",
                status=GradeStatus.ABSTAIN,
                created_at=_FINISHED_AT,
            )

    scorer = as_mlflow_scorer(_AbstainingGrader(), name="evk_abstain")
    feedback = scorer(inputs={"q": "x"}, outputs={"a": "y"})

    assert feedback.value is None
    assert feedback.error is not None
    assert feedback.metadata["evalkit_status"] == "abstain"
