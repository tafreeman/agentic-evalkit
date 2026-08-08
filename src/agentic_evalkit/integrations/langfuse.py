"""Export runs and calibration-gated scores into Langfuse (ADR-0022).

This mirrors :mod:`agentic_evalkit.integrations.mlflow` onto the other host
platform worth reaching: Langfuse is self-hostable, which matters for teams
who cannot send evaluation output to somebody else's cloud, and it is the
larger of the two by download volume.

The mirror is deliberately not a copy, because the two data models are not
the same shape. MLflow has a first-class *run* with params, metrics, tags
and artifacts, so a finished evaluation maps onto one MLflow run almost
field for field. Langfuse has no such object: it has observations (spans)
carrying arbitrary metadata, and *scores* attached to them. So a run here
becomes one root observation with the manifest and provenance in its
metadata, one child observation per sample, and scores for the outcomes --
with the run body carried in the root observation's metadata rather than as
an artifact, since Langfuse has no artifact store to put it in.

One consequence is worth stating plainly rather than discovering later:
because Langfuse has no artifact store, there is no Langfuse equivalent of
:func:`~agentic_evalkit.integrations.mlflow.compare_mlflow_runs`. Provenance
is exported and is visible, so an operator can *see* that two runs differ,
but this module will not offer a function that reads two Langfuse traces and
emits a delta -- reconstructing a full run body out of trace metadata is
exactly the kind of best-effort inference that produces a number nobody can
defend. Export both runs, and compare them with
:func:`~agentic_evalkit.stats.compare_runs` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agentic_evalkit.integrations.base import (
    AuthorityLevel,
    judge_authority,
    redact_for_export,
    require_dependency,
)
from agentic_evalkit.models import GradeStatus
from agentic_evalkit.reporters import DEFAULT_REDACTION_POLICY, redact_text
from agentic_evalkit.stats import aggregate_run, comparability_snapshot

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from agentic_evalkit.graders.calibration import CalibrationArtifact
    from agentic_evalkit.models import EvalRunResult, SampleResult
    from agentic_evalkit.reporters import RedactionPolicy

__all__ = [
    "LangfuseClient",
    "log_eval_run",
    "score_with_calibration_gate",
]

#: Prefix on every score and metadata key this module writes, so an exported
#: run never collides with the scores a team's own instrumentation records
#: against the same project.
_NAMESPACE = "evalkit"

#: Which grade statuses are a verdict about the system under test, and what
#: numeric score each becomes. The three deliberately absent -- ``ABSTAIN``,
#: ``ERROR`` and ``UNAVAILABLE`` -- are not verdicts, so they are never
#: written as ``0.0``. Langfuse averages numeric scores by name across a
#: project, so a withheld grade recorded as zero would silently drag down
#: every dashboard built on that score; those statuses are recorded as a
#: separate categorical score instead, where they cannot be averaged into
#: anything.
_GRADE_TO_SCORE: dict[GradeStatus, float] = {
    GradeStatus.PASS: 1.0,
    GradeStatus.FAIL: 0.0,
    GradeStatus.PARTIAL: 0.5,
}


@runtime_checkable
class LangfuseClient(Protocol):
    """The slice of ``langfuse.Langfuse`` this module actually calls.

    Declared as a Protocol rather than imported for two reasons. The first
    is the same one that governs the rest of this subpackage: the real class
    must not be needed at import time. The second is testability, and it is
    the more important of the two -- Langfuse has no local or offline mode,
    so unlike the MLflow bridge (which a test can point at a local tracking
    directory and exercise for real), the only way to test this module
    hermetically is against a stand-in. Naming the surface here makes that
    stand-in an implementation of a declared contract instead of a mock that
    happens to have the right method names, and it means a Langfuse release
    that changes one of these signatures fails type-checking here rather
    than at a customer's first export.

    This mirrors the pattern ``EvalRunner`` already uses for the dataset
    catalog: declare the narrow protocol locally, accept anything that
    satisfies it.

    ``start_observation`` is what sets the ``langfuse`` extra's lower bound
    of 3.3.1 in ``pyproject.toml``: earlier 3.x releases expose only
    ``start_span``/``start_generation``, so a client from one would satisfy
    neither this protocol nor the first export. Keep the two in step --
    widening this surface may raise that floor.
    """

    def start_observation(
        self,
        *,
        name: str,
        as_type: Any = ...,  # Langfuse's own observation-type literal union
        input: Any = ...,  # noqa: A002 -- Langfuse names this parameter `input`
        output: Any = ...,
        metadata: Any = ...,
    ) -> Any: ...  # LangfuseSpan and its eight siblings

    def create_score(
        self,
        *,
        name: str,
        value: float | str,
        trace_id: str | None = ...,
        observation_id: str | None = ...,
        data_type: Any = ...,  # Langfuse's ScoreDataType literal
        comment: str | None = ...,
        metadata: Any = ...,
    ) -> None: ...

    def flush(self) -> None: ...


def _resolve_client(client: LangfuseClient | None) -> LangfuseClient:
    """Return the caller's client, or build one from the ambient configuration.

    Accepting a client is the primary path and the one the docs show:
    it keeps credential handling in the caller's hands, which is where this
    package's redaction and secret rules say it belongs (never read a
    credential out of the environment inside library code that also
    transmits data). The zero-argument fallback exists only because
    ``Langfuse()`` reading ``LANGFUSE_PUBLIC_KEY``/``LANGFUSE_SECRET_KEY``
    from the environment is the idiom every Langfuse user already knows, and
    refusing to support it would make the bridge feel broken rather than
    careful.
    """
    if client is not None:
        return client
    langfuse = require_dependency("langfuse", extra="langfuse")
    constructed: LangfuseClient = langfuse.Langfuse()
    return constructed


def _run_metadata(
    run: EvalRunResult,
    calibration: CalibrationArtifact | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the root observation's metadata: manifest, provenance, and the run body.

    The full run body goes in under ``run`` because Langfuse has nowhere
    else to put it, and without it the export is a summary rather than
    evidence -- someone reading it later could see the pass rate but not
    reproduce or re-grade it.
    """
    stats = aggregate_run(run)
    metadata: dict[str, Any] = {
        "provenance": comparability_snapshot(run),
        "summary": {
            "total": stats.total,
            "passed": stats.passed,
            "failed": stats.failed,
            "partial": stats.partial,
            "errors": stats.errors,
            "timeouts": stats.timeouts,
            "cancelled": stats.cancelled,
            "abstained": stats.abstained,
            "unavailable": stats.unavailable,
        },
        "pass_rate": stats.pass_rate.model_dump(mode="json"),
        "run": run.model_dump(mode="json"),
    }
    if calibration is not None:
        authority = judge_authority(calibration, now=now)
        metadata["judge"] = {
            "authority": str(authority.level),
            "can_gate": authority.level is AuthorityLevel.GATING,
            "reason": authority.reason,
            "calibration_id": calibration.calibration_id,
            "calibration": calibration.model_dump(mode="json"),
        }
    return metadata


def _sample_metadata(result: SampleResult) -> dict[str, Any]:
    grade = result.grade
    return {
        "sample_id": result.sample.sample_id,
        "attempt": result.execution.attempt,
        "execution_status": str(result.execution.status),
        "grade_status": str(grade.status) if grade is not None else None,
        "grader": grade.grader if grade is not None else None,
        "hard_gate": grade.hard_gate if grade is not None else None,
        "calibration_id": grade.judge_calibration_ref if grade is not None else None,
    }


def _score_sample(
    client: LangfuseClient,
    result: SampleResult,
    *,
    trace_id: str | None,
    observation_id: str,
) -> None:
    """Record one sample's outcome, keeping non-verdicts out of the numeric score.

    A graded verdict becomes a numeric score under ``evalkit.grade``. A
    status that is *not* a verdict -- the grader abstained, broke, or could
    not be trusted -- becomes a categorical score under
    ``evalkit.grade_status`` instead, so it is visible in Langfuse without
    ever being averaged into the numeric one (ADR-0008).

    ``trace_id`` is passed alongside ``observation_id`` rather than left to
    be inferred. An observation ID identifies a span *within* a trace, so a
    score carrying only the former does not say what it is attached to; the
    Langfuse API expects the trace as the anchor, and omitting it risks the
    score being dropped or filed against nothing -- which would silently
    lose the outcome this whole export exists to record.
    """
    grade = result.grade
    if grade is None:
        client.create_score(
            name=f"{_NAMESPACE}.grade_status",
            value=str(result.execution.status),
            trace_id=trace_id,
            observation_id=observation_id,
            data_type="CATEGORICAL",
            comment="execution did not complete, so grading never ran",
        )
        return
    if grade.status in _GRADE_TO_SCORE:
        client.create_score(
            name=f"{_NAMESPACE}.grade",
            value=_GRADE_TO_SCORE[grade.status],
            trace_id=trace_id,
            observation_id=observation_id,
            data_type="NUMERIC",
            comment=grade.grader,
        )
    else:
        client.create_score(
            name=f"{_NAMESPACE}.grade_status",
            value=str(grade.status),
            trace_id=trace_id,
            observation_id=observation_id,
            data_type="CATEGORICAL",
            comment=f"{grade.grader} produced no verdict on the system under test",
        )


def log_eval_run(
    run: EvalRunResult,
    *,
    client: LangfuseClient | None = None,
    calibration: CalibrationArtifact | None = None,
    redaction_policy: RedactionPolicy = DEFAULT_REDACTION_POLICY,
    trace_name: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    flush: bool = True,
    now: datetime | None = None,
) -> str | None:
    """Write a finished run into Langfuse as one trace, and return its trace ID.

    The shape is one root observation named for the run, carrying the
    manifest, the provenance snapshot, the recounted summary, the pass rate
    with its interval, and the full redacted run body in its metadata; then
    one child observation per sample, each with its own score.

    Redaction is applied once, here, before anything is transmitted, and
    defaults to :data:`~agentic_evalkit.reporters.DEFAULT_REDACTION_POLICY`
    rather than to no redaction -- same reasoning as the MLflow bridge: the
    destination is a server other people can read. Pass ``RedactionPolicy()``
    to opt out deliberately.

    Args:
        run: The finished run to export.
        client: An authenticated Langfuse client. Left unset to construct
            one from the ambient ``LANGFUSE_*`` environment configuration.
        calibration: The calibration artifact backing this run's judge, if
            any, summarized into the root observation's metadata so the
            authority a judge claimed is auditable beside its results.
        redaction_policy: What to scrub before transmitting.
        trace_name: Name for the root observation. Defaults to the
            manifest's ``run_name``.
        extra_metadata: Additional metadata merged into the root
            observation, for the caller's own bookkeeping.
        now: The moment to evaluate ``calibration``'s expiry and age
            against. Defaults to the current UTC time; a fixed value makes
            the exported judge-authority metadata deterministic.
        flush: Whether to flush before returning. Defaults to ``True``,
            because the Langfuse client batches in the background and a
            short-lived process -- a CI job, a script -- routinely exits
            before the batch is sent, losing the export silently. Pass
            ``False`` only when the caller manages the client's lifecycle
            itself.

    Returns:
        The Langfuse trace ID holding the export, or ``None`` if the client
        did not expose one.

    Raises:
        IntegrationUnavailable: If ``client`` is omitted and Langfuse is not
            installed.
    """
    resolved = _resolve_client(client)
    redacted = redact_for_export(run, redaction_policy)

    metadata = _run_metadata(redacted, calibration, now)
    if extra_metadata:
        metadata.update(extra_metadata)

    root = resolved.start_observation(
        name=trace_name or redacted.manifest.run_name,
        as_type="evaluator",
        metadata=metadata,
    )
    trace_id: str | None = getattr(root, "trace_id", None)
    try:
        for result in redacted.samples:
            child = root.start_observation(
                name=f"sample:{result.sample.sample_id}",
                as_type="span",
                input=result.sample.input,
                output=result.execution.output,
                metadata=_sample_metadata(result),
            )
            try:
                _score_sample(
                    resolved,
                    result,
                    trace_id=trace_id or getattr(child, "trace_id", None),
                    observation_id=child.id,
                )
            finally:
                child.end()
    finally:
        root.end()
        # Flushed in the finally, not after it, so a failure partway through
        # still ships what was already built. The Langfuse client buffers in
        # a background thread, so an exception escaping this function with
        # the buffer unflushed loses every span and score recorded before the
        # failure -- and a caller that treats the error as fatal and exits
        # takes the evidence of what went wrong down with it. A partial trace
        # ending early is diagnosable; nothing at all is not. Langfuse has no
        # equivalent of MLflow's terminal run status, so this is the only
        # server-side trace a failed export leaves.
        if flush:
            resolved.flush()
    return trace_id


def score_with_calibration_gate(
    client: LangfuseClient,
    *,
    name: str,
    value: float | str,
    calibration: CalibrationArtifact | None,
    trace_id: str | None = None,
    observation_id: str | None = None,
    judge_fingerprint: str | None = None,
    comment: str | None = None,
    redaction_policy: RedactionPolicy = DEFAULT_REDACTION_POLICY,
    now: datetime | None = None,
) -> AuthorityLevel:
    """Record a judge's score in Langfuse under the authority its evidence earns.

    This is the Langfuse counterpart to
    :func:`~agentic_evalkit.integrations.mlflow.calibration_gate`, and it
    behaves the same way at the boundary that matters. A judge with full,
    unexpired calibration clearing the project floors gets its score written
    as-is. A judge whose evidence is merely *thin* gets its score written
    too, under a name suffixed ``.advisory`` so no dashboard or alert built
    on the ungated name can ever be moved by it. And a judge whose evidence
    is present and *bad* gets no numeric score at all -- only a categorical
    marker recording that it was asked and could not be trusted.

    The name suffix is doing real work rather than decorating: Langfuse
    aggregates numeric scores by name, so demotion has to change the name to
    mean anything. Writing an advisory score under the gating name and
    relying on a metadata field to be noticed would leave the aggregate
    already moved by the time anyone read it.

    Args:
        client: An authenticated Langfuse client.
        name: The score name a fully calibrated judge would write under.
        value: The judge's score.
        calibration: The evidence backing the judge. ``None`` is the
            ordinary ungated case and yields advisory output.
        trace_id: Trace to attach the score to.
        observation_id: Observation to attach the score to.
        judge_fingerprint: The live judge's fingerprint, so calibration
            measured against a different judge cannot gate this one.
        comment: Free-text comment stored alongside the score. The
            authority reason is appended to it.
        redaction_policy: Patterns scrubbed from the comment before it is
            transmitted. This function never sees an ``EvalRunResult``, so
            :func:`~agentic_evalkit.integrations.base.redact_for_export` has
            nothing to operate on and this is the only redaction available
            on this path -- the same position ``as_mlflow_scorer`` is in
            with its rationale. ``comment`` is wholly caller-supplied, and
            the obvious thing to build one from is target output or an
            exception message, both well-trodden routes for a credential to
            reach a string. Pass ``RedactionPolicy()`` to opt out.
        now: The moment to evaluate ``calibration``'s expiry and age
            against. Defaults to the current UTC time, which is what a live
            caller wants; tests and reproducible pipelines pass a fixed
            value. Mirrors
            :func:`~agentic_evalkit.integrations.mlflow.calibration_gate`,
            whose authority must be re-resolved per call for the same
            reason: a calibration that expires must stop gating.

    Returns:
        The :class:`~agentic_evalkit.integrations.base.AuthorityLevel` the
        score was written under, so a caller can branch on it.
    """
    authority = judge_authority(calibration, judge_fingerprint=judge_fingerprint, now=now)
    parts = [part for part in (comment, authority.reason) if part]
    full_comment = redact_text(" | ".join(parts), redaction_policy) if parts else None
    metadata = {
        "evalkit_authority": str(authority.level),
        "evalkit_can_gate": authority.level is AuthorityLevel.GATING,
        "evalkit_calibration_id": authority.calibration_id,
    }

    if authority.level is AuthorityLevel.UNAVAILABLE:
        client.create_score(
            name=f"{name}.unavailable",
            value=str(AuthorityLevel.UNAVAILABLE),
            trace_id=trace_id,
            observation_id=observation_id,
            data_type="CATEGORICAL",
            comment=full_comment,
            metadata=metadata,
        )
        return authority.level

    score_name = name if authority.level is AuthorityLevel.GATING else f"{name}.advisory"
    client.create_score(
        name=score_name,
        value=value,
        trace_id=trace_id,
        observation_id=observation_id,
        data_type="NUMERIC" if isinstance(value, (int, float)) else "CATEGORICAL",
        comment=full_comment,
        metadata=metadata,
    )
    return authority.level
