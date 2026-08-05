"""Export runs, graders, and provenance into an MLflow tracking server (ADR-0022).

MLflow already owns the things this package deliberately does not have: a
tracking UI, an experiment store, a prompt registry, sixty framework
integrations, and a user base four orders of magnitude larger than this
one's. It also ships judge *alignment* -- an optimizer that takes a judge
disagreeing with humans and makes it agree better. What it does not ship is
judge *authority gating*: a rule that a judge with no proof of agreement may
not block a release. Those two controls solve different problems, and a team
holding both is strictly better off than a team holding either.

So this module does not ask anyone to leave MLflow. It carries three things
into it:

* :func:`log_eval_run` writes a finished run into an MLflow experiment --
  outcome counts and the pass rate with its confidence interval as metrics,
  the manifest as params, the full redacted run body and any calibration
  artifact as artifacts.
* :func:`calibration_gate` wraps a scorer the user already has -- including
  one built by ``mlflow.genai.judges.make_judge`` -- so that its verdict is
  demoted to advisory, or withheld entirely, when its calibration evidence
  does not earn the authority to gate. This is the piece with no equivalent
  in the host platform.
* :func:`compare_mlflow_runs` reads two MLflow run IDs and refuses to report
  a delta unless the runs are provably comparable.

Nothing here imports ``mlflow`` at module scope: this module must import
cleanly on a machine that has never installed it (see
:func:`~agentic_evalkit.integrations.base.require_dependency`). Note also
that this module is named ``mlflow.py`` while importing a package of the
same name -- that is safe, and not an accident. Python 3 resolves ``import
mlflow`` absolutely (PEP 328), so it always finds the installed
distribution, never this sibling module; and in any case the import here
goes through :func:`importlib.import_module` by an explicit name.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from agentic_evalkit.errors import IncompatibleRuns
from agentic_evalkit.integrations.base import (
    AuthorityLevel,
    judge_authority,
    redact_for_export,
    require_dependency,
    run_blocking,
)
from agentic_evalkit.models import (
    EvalRunResult,
    EvalSample,
    ExecutionStatus,
    GradeStatus,
    NormalizedExecutionResult,
)
from agentic_evalkit.reporters import DEFAULT_REDACTION_POLICY
from agentic_evalkit.stats import aggregate_run, comparability_snapshot, compare_runs

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pydantic import JsonValue

    from agentic_evalkit.graders.base import Grader
    from agentic_evalkit.graders.calibration import CalibrationArtifact
    from agentic_evalkit.models import GradeResult
    from agentic_evalkit.reporters import RedactionPolicy
    from agentic_evalkit.stats import ComparisonResult

__all__ = [
    "CALIBRATION_ARTIFACT_PATH",
    "RUN_ARTIFACT_PATH",
    "as_mlflow_scorer",
    "calibration_gate",
    "compare_mlflow_runs",
    "log_eval_run",
]

#: Where the full redacted run body is written inside an MLflow run's
#: artifacts. :func:`compare_mlflow_runs` reads exactly this path back, so
#: the two constants below are the wire contract between the two halves of
#: this module -- changing either breaks comparison against every run
#: already exported by an older version.
RUN_ARTIFACT_PATH = "evalkit/run.json"

#: Where the calibration artifact backing a gating judge is written, when
#: one is supplied. Kept beside the run body so the evidence for a gate
#: travels with the result the gate produced, rather than living in a wiki
#: nobody opens.
CALIBRATION_ARTIFACT_PATH = "evalkit/calibration.json"

#: Prefix on every key this module writes, so an exported run never collides
#: with a param, metric, or tag the user's own code already logs into the
#: same run.
_NAMESPACE = "evalkit"

#: MLflow rejects a param value over 6000 characters and a tag value over
#: 8000 (``mlflow.utils.validation``). Both are far beyond anything a
#: manifest field should hold, so hitting one means something unexpected got
#: in -- but a hard rejection mid-export would abandon a run halfway
#: written, having already logged some of it. Truncating with a visible
#: marker keeps the export atomic in practice and makes the overflow
#: obvious in the UI instead of silent.
_MAX_PARAM_LENGTH = 6000
_MAX_TAG_LENGTH = 8000
_TRUNCATION_MARKER = "...[truncated]"


class _MlflowRunData(Protocol):
    """The parts of ``mlflow.entities.RunData`` this module reads.

    Declared here rather than imported so that reading a run back stays
    type-checked without making ``mlflow`` an import-time dependency of this
    module -- the same reason ``EvalRunner`` declares its own
    ``_CatalogProtocol`` instead of importing the catalog.
    """

    @property
    def tags(self) -> dict[str, str]: ...


class _MlflowRun(Protocol):
    @property
    def data(self) -> _MlflowRunData: ...


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _manifest_params(run: EvalRunResult) -> dict[str, str]:
    """Flatten the manifest into MLflow params: the run's declared inputs.

    Params are MLflow's "what was this configured to do" surface and are
    immutable once written, which matches a manifest exactly. Everything
    here is a scalar the manifest already pins; nothing is derived and
    nothing is computed, so a reader comparing two runs' params is comparing
    two declarations rather than two summaries.
    """
    manifest = run.manifest
    dataset = run.resolved_dataset
    params: dict[str, JsonValue] = {
        "run_name": manifest.run_name,
        "adapter": manifest.adapter,
        "grader": manifest.grader,
        "target_name": manifest.target_name,
        "target_fingerprint_policy": manifest.target_fingerprint_policy,
        "attempts": manifest.attempts,
        "concurrency": manifest.concurrency,
        "timeout_seconds": manifest.timeout_seconds,
        "sampling.seed": manifest.sampling.seed,
        "sampling.temperature": manifest.sampling.temperature,
        "selection.offset": manifest.selection.offset,
        "selection.limit": manifest.selection.limit,
        "selection.filter": manifest.selection.filter,
        "dataset.provider": manifest.dataset_ref.provider,
        "dataset.id": dataset.dataset_id,
        "dataset.revision": dataset.revision,
        "dataset.config": dataset.config,
        "dataset.split": dataset.split,
        "dataset.row_count": dataset.row_count,
        "dataset.license": dataset.license,
    }
    if dataset.contamination is not None:
        # ADR-0013: this label never affects comparability, but a reader
        # deciding whether to believe a number needs to see it next to the
        # number rather than three clicks away.
        params["dataset.contamination_status"] = dataset.contamination.status
        params["dataset.contamination_held_out"] = dataset.contamination.held_out
    return {
        f"{_NAMESPACE}.{key}": _truncate(str(value), _MAX_PARAM_LENGTH)
        for key, value in params.items()
    }


def _outcome_metrics(run: EvalRunResult) -> dict[str, float]:
    """Recount outcomes and render them as MLflow metrics.

    The counts come from :func:`~agentic_evalkit.stats.aggregate_run`, which
    walks the samples itself rather than trusting ``run.summary``, and which
    keeps every operational outcome in its own counter (ADR-0008). That
    separation is the reason there is no single "score" metric here: an
    ``errors`` count folded into ``failed`` would produce a tidier dashboard
    and a dishonest one, since a harness that crashed on ten samples would
    become a system that got ten answers wrong.

    ``pass_rate`` is exported with its Wilson bounds beside it, never alone.
    A rate without an interval invites a reader to treat 0.80 from 10
    samples and 0.80 from 10,000 as the same finding.
    """
    stats = aggregate_run(run)
    metrics: dict[str, float | None] = {
        "summary.total": stats.total,
        "summary.passed": stats.passed,
        "summary.failed": stats.failed,
        "summary.partial": stats.partial,
        "summary.errors": stats.errors,
        "summary.timeouts": stats.timeouts,
        "summary.cancelled": stats.cancelled,
        "summary.abstained": stats.abstained,
        "summary.unavailable": stats.unavailable,
        "pass_rate": stats.pass_rate.value,
        "pass_rate.lower_bound": stats.pass_rate.lower_bound,
        "pass_rate.upper_bound": stats.pass_rate.upper_bound,
        "pass_rate.numerator": stats.pass_rate.numerator,
        "pass_rate.denominator": stats.pass_rate.denominator,
        "score_mean": stats.score_mean,
        "score_count": stats.score_count,
    }
    # A metric MLflow never received is honestly absent from the run; a
    # metric logged as 0.0 because the value was None is a fabricated
    # observation. Drop rather than default.
    return {
        f"{_NAMESPACE}.{key}": float(value) for key, value in metrics.items() if value is not None
    }


def _provenance_tags(run: EvalRunResult) -> dict[str, str]:
    """Every field ``compare_runs`` checks, as searchable MLflow tags.

    Tags are chosen over params because MLflow's run search filters on them,
    which is what makes this useful rather than decorative: with these
    written, finding every run that shares a target fingerprint is a query
    instead of a download.

    The field list is not written here. It comes from
    :func:`~agentic_evalkit.stats.comparability_snapshot`, which derives it
    from the same two tables ``compare_runs`` loops over -- so a provenance
    field added to the comparison automatically starts being exported, and
    this module cannot advertise a provenance surface narrower than the one
    actually enforced.
    """
    tags = {
        f"{_NAMESPACE}.provenance.{key}": _truncate(value, _MAX_TAG_LENGTH)
        for key, value in comparability_snapshot(run).items()
    }
    tags[f"{_NAMESPACE}.schema_version"] = run.schema_version
    tags[f"{_NAMESPACE}.run_id"] = run.run_id
    return tags


def _resolve_experiment_id(client: Any, experiment: str | None) -> str:
    """Find or create the experiment to log into, without disturbing the process.

    ``mlflow.set_experiment`` would be the short way to do this and is the
    wrong one for a library: it mutates process-global state that outlives
    the call, so exporting a run would silently redirect every subsequent
    ``mlflow.log_metric`` the caller's own code makes. Resolving the ID here
    and passing it explicitly leaves the caller's configuration exactly as
    it was found.

    ``"0"`` is MLflow's Default experiment, used when no name is given --
    the same place an unconfigured ``mlflow.start_run()`` would land.
    """
    if experiment is None:
        return "0"
    existing = client.get_experiment_by_name(experiment)
    if existing is not None:
        experiment_id: str = existing.experiment_id
        return experiment_id
    created: str = client.create_experiment(experiment)
    return created


def log_eval_run(
    run: EvalRunResult,
    *,
    tracking_uri: str | None = None,
    experiment: str | None = None,
    run_name: str | None = None,
    calibration: CalibrationArtifact | None = None,
    redaction_policy: RedactionPolicy = DEFAULT_REDACTION_POLICY,
    extra_tags: Mapping[str, str] | None = None,
) -> str:
    """Write a finished run into MLflow, evidence and all, and return its MLflow run ID.

    What lands in the tracking server is a complete, self-describing record:
    the manifest as params, recounted outcomes and the pass rate with its
    confidence interval as metrics, every provenance field ``compare_runs``
    checks as searchable tags, and the full redacted run body as a JSON
    artifact. That last part is what makes :func:`compare_mlflow_runs`
    possible later -- a tag tells you two runs *look* comparable, but only
    the run body lets the real comparison run.

    Redaction is applied once, here, before anything is transmitted. Unlike
    the reporters -- which write to a path the caller already controls and
    so default to whatever policy the caller passes -- this defaults to
    :data:`~agentic_evalkit.reporters.DEFAULT_REDACTION_POLICY`, because the
    destination is a server other people can read. Pass
    ``RedactionPolicy()`` to opt out deliberately.

    This deliberately goes through ``MlflowClient`` rather than the fluent
    ``mlflow.start_run`` / ``mlflow.log_metric`` API, and the difference is
    behavioural, not stylistic. The fluent functions read and write
    process-global state: they would refuse to run if the caller already had
    an active MLflow run open (unless told to nest), and
    ``set_tracking_uri`` / ``set_experiment`` would outlive this call and
    quietly redirect the caller's own logging afterwards. Exporting an
    evaluation result should not reconfigure the program that asked for it,
    so every write here names its run ID explicitly and nothing global is
    touched. Calling this from inside your own ``with mlflow.start_run()``
    block is therefore safe, and creates a sibling run rather than
    hijacking yours.

    Args:
        run: The finished run to export.
        tracking_uri: The MLflow tracking server to write to. Left unset to
            use whatever MLflow is already configured with.
        experiment: Experiment name to log under, created if it does not
            exist. Left unset to use MLflow's Default experiment.
        run_name: Name for the MLflow run. Defaults to the manifest's
            ``run_name``.
        calibration: The calibration artifact backing this run's judge, if
            there was one. Written as its own artifact and summarized into
            tags, so the authority a judge claimed is auditable next to the
            result it produced.
        redaction_policy: What to scrub before transmitting.
        extra_tags: Additional tags to set, for the caller's own bookkeeping
            (a git SHA, a CI job ID). Written verbatim, without this
            module's namespace prefix.

    Returns:
        The MLflow run ID that now holds this run.

    Raises:
        IntegrationUnavailable: If MLflow is not installed.
    """
    mlflow = require_dependency("mlflow", extra="mlflow")
    from mlflow.entities import Metric, Param

    redacted = redact_for_export(run, redaction_policy)

    tags = _provenance_tags(redacted)
    if calibration is not None:
        authority = judge_authority(calibration)
        tags[f"{_NAMESPACE}.judge.authority"] = str(authority.level)
        tags[f"{_NAMESPACE}.judge.calibration_id"] = calibration.calibration_id
        if authority.reason is not None:
            tags[f"{_NAMESPACE}.judge.authority_reason"] = _truncate(
                authority.reason, _MAX_TAG_LENGTH
            )
    if extra_tags:
        tags.update(extra_tags)

    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    mlflow_run = client.create_run(
        experiment_id=_resolve_experiment_id(client, experiment),
        run_name=run_name or redacted.manifest.run_name,
        tags=tags,
    )
    mlflow_run_id: str = mlflow_run.info.run_id

    # One timestamp for every metric in the batch: these are not a time
    # series, they are one measurement of one finished run, and giving them
    # drifting timestamps would invite a chart that implies otherwise.
    logged_at = int(time.time() * 1000)
    try:
        # Tags are not repeated here: create_run above already wrote them,
        # and logging them twice would be two round trips to set the same
        # values. Param is untyped upstream (mlflow.entities.Param takes
        # bare `key, value`), which strict mypy reports as an untyped call
        # into typed code -- a gap in MLflow's annotations, not a real type
        # problem, so it is silenced narrowly rather than by loosening the
        # checker for this module.
        client.log_batch(
            mlflow_run_id,
            metrics=[
                Metric(key, value, logged_at, 0)
                for key, value in _outcome_metrics(redacted).items()
            ],
            params=[
                Param(key, value)  # type: ignore[no-untyped-call]
                for key, value in _manifest_params(redacted).items()
            ],
        )
        # mode="json" so datetimes and enums land as strings the artifact
        # can be re-parsed from, rather than as Python objects json.dumps
        # would reject.
        client.log_dict(mlflow_run_id, redacted.model_dump(mode="json"), RUN_ARTIFACT_PATH)
        if calibration is not None:
            client.log_dict(
                mlflow_run_id, calibration.model_dump(mode="json"), CALIBRATION_ARTIFACT_PATH
            )
    except Exception:
        # A run left RUNNING forever is worse than one marked FAILED: the
        # UI shows it as still in progress, and nothing downstream can tell
        # a crashed export from a slow one.
        client.set_terminated(mlflow_run_id, status="FAILED")
        raise
    client.set_terminated(mlflow_run_id, status="FINISHED")

    return mlflow_run_id


def _load_exported_run(client: Any, run_id: str, dst_path: str) -> EvalRunResult:
    """Read one exported run body back out of MLflow, or refuse to guess.

    A tracking server is full of runs this package never wrote. Asked to
    compare one of those, the only honest answer is to refuse: there is no
    manifest, no resolved dataset, and no provenance, so no claim about
    comparability can be supported. Raising
    :class:`~agentic_evalkit.errors.IncompatibleRuns` here is the same
    refusal ``compare_runs`` makes for the same reason.

    This downloads through the client rather than resolving a ``runs:/``
    URI, because ``mlflow.artifacts.load_dict`` accepts no tracking URI and
    would therefore force a global ``set_tracking_uri`` -- reconfiguring the
    caller's process as a side effect of reading two runs.
    """
    try:
        local_path = client.download_artifacts(run_id, RUN_ARTIFACT_PATH, dst_path)
    except Exception as exc:
        raise IncompatibleRuns(
            message=(
                f"MLflow run {run_id!r} has no {RUN_ARTIFACT_PATH!r} artifact, so it was not "
                "exported by agentic-evalkit and its comparability cannot be established"
            ),
            context={"run_id": run_id, "artifact_path": RUN_ARTIFACT_PATH},
        ) from exc
    payload = json.loads(Path(local_path).read_text(encoding="utf-8"))
    return EvalRunResult.model_validate(payload)


def _snapshot_mismatches(left: _MlflowRun, right: _MlflowRun) -> list[str]:
    """Compare two runs' provenance tags, reporting only differences both sides declared.

    This is a fast pre-check, not the decision. It exists so that a
    mismatched pair fails in one cheap metadata read instead of after
    downloading two full run bodies, and so the error names the offending
    field even when a body is large or slow to fetch.

    A field is only reported when *both* runs carry the tag. A tag missing
    on one side means that run was exported by a version that did not write
    it, which is a gap in knowledge, not evidence of a difference -- and
    inventing a mismatch from it would refuse a comparison that is actually
    fine. The real :func:`~agentic_evalkit.stats.compare_runs` runs
    afterwards regardless and remains the authority, so anything this misses
    is still caught; nothing this reports can be a false alarm.
    """
    prefix = f"{_NAMESPACE}.provenance."
    left_tags = {k: v for k, v in left.data.tags.items() if k.startswith(prefix)}
    right_tags = {k: v for k, v in right.data.tags.items() if k.startswith(prefix)}
    return [
        f"{key.removeprefix(prefix)} differs: {left_tags[key]!r} != {right_tags[key]!r}"
        for key in sorted(set(left_tags) & set(right_tags))
        if left_tags[key] != right_tags[key]
    ]


def compare_mlflow_runs(
    left_run_id: str,
    right_run_id: str,
    *,
    seed: int,
    tracking_uri: str | None = None,
    bootstrap_samples: int = 1000,
    allow_cross_environment: bool = False,
) -> ComparisonResult:
    """Compare two MLflow runs, refusing the pair outright unless they are comparable.

    This is the recipe the bridge exists to make possible: two run IDs a
    colleague pasted from the MLflow UI go in, and either a paired-bootstrap
    delta with a confidence interval comes out, or an error naming every
    field that makes the pair meaningless. There is no third outcome where a
    number appears with a caveat attached, because a caveat next to a
    number does not survive being copied into a slide.

    The comparison itself is not reimplemented here. Both run bodies are
    downloaded and handed to :func:`~agentic_evalkit.stats.compare_runs`,
    which owns the provenance rules and the seeded bootstrap. This function
    only adds the MLflow-shaped parts: resolving IDs to bodies, and failing
    early and cheaply on a tag mismatch before paying for the download.

    Args:
        left_run_id: MLflow run ID of the baseline.
        right_run_id: MLflow run ID of the candidate.
        seed: Bootstrap seed. Required and keyword-only, exactly as
            ``compare_runs`` requires it, so a comparison read off a
            tracking server is never silently irreproducible.
        tracking_uri: Tracking server to read from. Left unset to use
            MLflow's current configuration.
        bootstrap_samples: Number of bootstrap resamples, in [100, 10000].
        allow_cross_environment: Waive a mismatch on *only* the environment
            and code fingerprints (ADR-0015), recording which were waived on
            the result. No other field can be waived.

    Returns:
        The :class:`~agentic_evalkit.stats.ComparisonResult` for the pair.

    Raises:
        IntegrationUnavailable: If MLflow is not installed.
        IncompatibleRuns: If either run was not exported by this package, or
            the two are not provably comparable.
    """
    mlflow = require_dependency("mlflow", extra="mlflow")
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)

    left_meta = client.get_run(left_run_id)
    right_meta = client.get_run(right_run_id)
    mismatches = _snapshot_mismatches(left_meta, right_meta)
    if mismatches:
        raise IncompatibleRuns(
            message=(
                f"MLflow runs {left_run_id!r} and {right_run_id!r} are not comparable: "
                + "; ".join(mismatches)
            ),
            context={"left_run_id": left_run_id, "right_run_id": right_run_id},
        )

    # The downloaded bodies are read once and thrown away: they are a
    # transport detail, not a cache, and leaving them behind would scatter
    # copies of possibly-sensitive run output around the caller's disk.
    with tempfile.TemporaryDirectory(prefix="evalkit-mlflow-compare-") as staging:
        left = _load_exported_run(client, left_run_id, f"{staging}/left")
        right = _load_exported_run(client, right_run_id, f"{staging}/right")
        return compare_runs(
            left,
            right,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            allow_cross_environment=allow_cross_environment,
        )


#: How each grade outcome maps onto a boolean MLflow feedback value. The two
#: statuses deliberately absent -- ``ABSTAIN`` and ``UNAVAILABLE`` -- are not
#: an oversight: neither is a verdict about the system under test, so
#: neither may be rendered as ``False``. They surface as a feedback *error*
#: instead, which keeps them out of any aggregate MLflow computes, exactly
#: as ``CompositeGrader`` excludes them from its weighted mean rather than
#: scoring them zero.
_GRADE_TO_FEEDBACK_VALUE: dict[GradeStatus, bool] = {
    GradeStatus.PASS: True,
    GradeStatus.FAIL: False,
    GradeStatus.PARTIAL: False,
}


def calibration_gate(
    scorer: Callable[..., Any],
    *,
    calibration: CalibrationArtifact | None,
    name: str | None = None,
    judge_fingerprint: str | None = None,
    now: datetime | None = None,
) -> Any:  # mlflow.genai.scorers.Scorer, unimportable at module scope
    """Wrap an MLflow scorer so it can only gate a release when its evidence says it may.

    This is the control MLflow does not have, offered in the shape MLflow
    already uses. Point it at a judge built with
    ``mlflow.genai.judges.make_judge``, or at any scorer, and hand it the
    calibration evidence for that judge. What comes back is a scorer with
    identical behaviour in the case that matters least -- fully calibrated,
    verdict passes through -- and materially different behaviour in the two
    cases that matter most:

    * **Evidence present and bad** (expired, or a measured TNR/TPR below the
      project floor on a sufficient sample): the verdict is *withheld*. The
      returned feedback carries an error rather than a value, so MLflow
      records that this judge was asked and could not be trusted, and no
      aggregate silently absorbs a number nobody should have used.
    * **Evidence absent or too thin** (no artifact, no ``calibrated_at``,
      too few held-out labels, a Wilson lower bound short of the floor, or a
      fingerprint that does not match the live judge): the verdict is
      *reported but demoted*. It appears in MLflow with
      ``evalkit_authority=advisory`` in its metadata and the reason spelled
      out, so a human can read it and a gate can be written not to trust it.

    Note what this does not claim. It does not make a judge more accurate --
    that is what MLflow's own alignment does, and the two compose: align the
    judge to raise agreement, gate it so it cannot act beyond what its
    agreement has been shown to be. A team running both has a judge that is
    better *and* one that cannot overreach.

    Args:
        scorer: The scorer or judge to wrap. Called with whatever keyword
            arguments MLflow passes it, forwarded unchanged.
        calibration: The evidence backing ``scorer``. ``None`` is the
            ordinary case for an ungated judge and produces advisory
            output -- it is not an error.
        name: Name for the wrapped scorer in MLflow. Defaults to the
            wrapped scorer's own name, or ``"evalkit_gated_judge"``.
        judge_fingerprint: The live judge's model+prompt fingerprint, if the
            host exposes one, so calibration measured against a *different*
            judge cannot gate this one.
        now: The moment to evaluate expiry and age against. Defaults to the
            current UTC time; tests pass a fixed value.

    Returns:
        An ``mlflow.genai.scorers.Scorer`` ready to pass to
        ``mlflow.genai.evaluate(scorers=[...])``.

    Raises:
        IntegrationUnavailable: If MLflow is not installed.
    """
    require_dependency("mlflow", extra="mlflow")
    from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
    from mlflow.genai.scorers import scorer as scorer_decorator

    authority = judge_authority(calibration, judge_fingerprint=judge_fingerprint, now=now)
    scorer_name = name or getattr(scorer, "name", None) or "evalkit_gated_judge"
    # Every value in Feedback.metadata must be a string (mlflow.entities
    # types it dict[str, str]), so the level enum and the reason are
    # rendered rather than passed through.
    base_metadata = {
        "evalkit_authority": str(authority.level),
        "evalkit_can_gate": str(authority.level is AuthorityLevel.GATING).lower(),
    }
    if authority.reason is not None:
        base_metadata["evalkit_authority_reason"] = authority.reason
    if authority.calibration_id is not None:
        base_metadata["evalkit_calibration_id"] = authority.calibration_id

    source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE,
        source_id=scorer_name,
    )

    @scorer_decorator(name=scorer_name)
    def gated(**kwargs: Any) -> Feedback:  # MLflow chooses the kwargs
        if authority.level is AuthorityLevel.UNAVAILABLE:
            # Withheld, not failed. The wrapped scorer is never called: its
            # answer would be unusable, and calling a judge whose verdict
            # cannot be used bills the caller for nothing.
            return Feedback(
                name=scorer_name,
                error=authority.reason,
                source=source,
                metadata=base_metadata,
            )
        result = scorer(**kwargs)
        # _attach_authority is typed to return Any because it cannot name
        # Feedback at module scope; the cast restores the real type here,
        # where the class is in scope.
        return cast(
            "Feedback", _attach_authority(result, scorer_name, source, base_metadata, Feedback)
        )

    return gated


def _attach_authority(
    result: object,
    scorer_name: str,
    source: object,
    metadata: dict[str, str],
    feedback_cls: type[Any],
) -> Any:  # mlflow.entities.Feedback
    """Carry the authority verdict onto whatever the wrapped scorer returned.

    A scorer is allowed to return a bare value (``True``, ``0.8``,
    ``"good"``) or a full ``Feedback``. Both are normalized to a
    ``Feedback`` here so the authority metadata has somewhere to live --
    without it, a gate downstream would have the verdict and no way to know
    whether it was entitled to act on it, which is the exact failure this
    whole module exists to prevent.

    An existing ``Feedback``'s own metadata is preserved and the authority
    keys are merged over it, so wrapping never discards what the judge
    already reported.

    The copy is made with :func:`copy.copy` rather than the more obvious
    :func:`dataclasses.replace`, because ``replace`` does not work on this
    class: ``Feedback`` declares dataclass fields (``run_id``,
    ``assessment_id``, ``expectation``, ``feedback``, ``issue``) that its
    ``__init__`` does not accept, so ``replace`` -- which rebuilds the
    object by calling the constructor with every field -- raises
    ``TypeError``. A shallow copy sidesteps the constructor entirely. The
    original is still never mutated: only the copy's ``metadata`` is
    rebound, and to a freshly built dict rather than to one shared with the
    input.
    """
    if isinstance(result, feedback_cls):
        merged = {**(result.metadata or {}), **metadata}
        copied = copy.copy(result)
        copied.metadata = merged
        return copied
    return feedback_cls(
        name=scorer_name,
        value=result,
        source=source,
        metadata=metadata,
    )


def _sample_id_for(inputs: object) -> str:
    """Derive a stable sample ID from a scorer's inputs.

    MLflow hands a scorer the row's inputs but no row identifier, while
    every grader here needs a ``sample_id`` -- it is the join key each grade
    is filed under. Hashing the inputs gives an ID that is stable across
    runs of the same dataset, which is what makes two exported runs pair up
    in :func:`compare_mlflow_runs` at all. ``default=str`` keeps the hash
    total over inputs holding datetimes or other non-JSON objects rather
    than raising mid-scorer.
    """
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def as_mlflow_scorer(
    grader: Grader,
    *,
    name: str,
    adapter: str = "mlflow-genai",
) -> Any:  # mlflow.genai.scorers.Scorer, unimportable at module scope
    """Expose an ``agentic-evalkit`` grader as an MLflow custom scorer.

    This is the other direction from :func:`calibration_gate`: rather than
    constraining a judge MLflow already has, it makes a grader from this
    package -- the exact-match grader, the grounded-citation grader, a
    composite, a calibrated judge -- usable inside
    ``mlflow.genai.evaluate`` alongside whatever scorers a team already
    runs.

    The outcome mapping is the part worth reading carefully, because it is
    where a lesser bridge would quietly lie. ``PASS`` becomes ``True`` and
    ``FAIL``/``PARTIAL`` become ``False``, but ``ABSTAIN``, ``ERROR`` and
    ``UNAVAILABLE`` become a feedback *error* rather than ``False``. None of
    those three is a statement that the system under test got the answer
    wrong -- they say the grader declined, broke, or could not be trusted --
    and rendering them as ``False`` would fold operational failure into task
    failure, which is precisely what ADR-0008 forbids. As an error, MLflow
    keeps them out of any aggregate it computes, mirroring how
    ``CompositeGrader`` excludes them from its weighted mean instead of
    scoring them zero.

    ``hard_gate`` and the backing calibration reference travel into the
    feedback's metadata, so a grade that is not entitled to block a release
    says so where an MLflow-side gate can read it.

    Args:
        grader: Any object satisfying the
            :class:`~agentic_evalkit.graders.base.Grader` protocol.
        name: The scorer's name in MLflow.
        adapter: Value recorded as the synthesized sample's ``adapter``.
            Only identifies where the sample came from; it does not select
            any behaviour.

    Returns:
        An ``mlflow.genai.scorers.Scorer`` ready to pass to
        ``mlflow.genai.evaluate(scorers=[...])``.

    Raises:
        IntegrationUnavailable: If MLflow is not installed.
    """
    require_dependency("mlflow", extra="mlflow")
    from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
    from mlflow.genai.scorers import scorer as scorer_decorator

    source = AssessmentSource(source_type=AssessmentSourceType.CODE, source_id=name)

    @scorer_decorator(name=name)
    def wrapped(
        inputs: Any = None,  # MLflow passes arbitrary row data
        outputs: Any = None,
        expectations: Any = None,
    ) -> Feedback:
        sample_id = _sample_id_for(inputs)
        expected = expectations or {}
        sample = EvalSample(
            sample_id=sample_id,
            input=inputs if isinstance(inputs, dict) else {"input": inputs},
            reference=(
                expected.get("expected_response") if isinstance(expected, dict) else str(expected)
            ),
            source_digest=f"sha256:{sample_id}",
            adapter=adapter,
        )
        now = datetime.now(UTC)
        execution = NormalizedExecutionResult(
            sample_id=sample_id,
            attempt=1,
            output=outputs if isinstance(outputs, dict) else {"response": outputs},
            status=ExecutionStatus.COMPLETED,
            started_at=now,
            finished_at=now,
        )
        grade = run_blocking(grader.grade(sample, execution))

        metadata = {
            "evalkit_status": str(grade.status),
            "evalkit_grader": grade.grader,
            "evalkit_hard_gate": str(grade.hard_gate).lower(),
        }
        if grade.judge_calibration_ref is not None:
            metadata["evalkit_calibration_id"] = grade.judge_calibration_ref

        if grade.status not in _GRADE_TO_FEEDBACK_VALUE:
            # Declined, broke, or untrustworthy -- never rendered as a
            # failing verdict about the system under test.
            return Feedback(
                name=name,
                error=f"grade status {grade.status!r} is not a verdict on the system under test",
                source=source,
                metadata=metadata,
            )
        return Feedback(
            name=name,
            value=(
                grade.score if grade.score is not None else _GRADE_TO_FEEDBACK_VALUE[grade.status]
            ),
            rationale=_rationale_of(grade),
            source=source,
            metadata=metadata,
        )

    return wrapped


def _rationale_of(grade: GradeResult) -> str | None:
    """Pull a human-readable justification out of a grade's evidence, if it has one.

    Graders here record their justification under a ``reason`` key by
    convention rather than by contract, so this reads that key when it holds
    a string and otherwise returns ``None`` -- an absent rationale is better
    than one invented by stringifying a dict of internals.
    """
    reason = grade.evidence.get("reason")
    return reason if isinstance(reason, str) else None
