"""Runs one evaluation from start to finish: dataset -> adapter -> target -> grader (plan Task 11).

``EvalRunner`` is the one place in this package that actually drives an
evaluation run. Given a manifest (the config describing what to run) and a
set of already-built components -- adapters, execution targets (wrappers
around the AI system being tested), and graders (things that judge whether
an output was correct) -- it walks through the whole pipeline: pull records
from the dataset, adapt each one into a sample, run it against the target,
grade the result, and assemble everything into one complete result object.
``EvalRunner`` itself never chooses, imports, or constructs any of those
components -- the caller (usually the CLI, or a higher-level registry added
in a later task) is responsible for building them and handing them to the
runner by name.

The runner deliberately doesn't depend on the real dataset catalog class.
Instead, it only requires that whatever "catalog" it's given matches a
small, locally defined shape (``_CatalogProtocol`` below): something with
an async ``resolve`` method and an async ``iter_records`` iterator. This
keeps the runner lightweight to import and easy to test in isolation,
without dragging in all the dataset-provider and caching machinery -- and
it means any object with the right two methods (a real catalog, a
lightweight test double, a filtered view over a catalog) can stand in for
it.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import uuid4

from agentic_evalkit.errors import (
    GraderError,
    JsonValue,
    ManifestValidationError,
    OutputSpillFailed,
    TargetFailure,
    TargetTimeout,
)
from agentic_evalkit.events import (
    DatasetResolved,
    ExecutionCompleted,
    GradeCompleted,
    RunCompleted,
    RunEvent,
    RunFailed,
    RunStarted,
    SampleCompleted,
    SampleStarted,
)
from agentic_evalkit.models import (
    OUTPUT_REF_KEY,
    OUTPUT_SPILL_ERROR_KEY,
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
    SourceRecord,
    is_output_spill_error_record,
)
from agentic_evalkit.reporters.base import (
    DEFAULT_REDACTION_POLICY,
    RedactionPolicy,
    _resolve_redaction_policy,
)

if TYPE_CHECKING:
    from agentic_evalkit.artifacts import ArtifactStore
    from agentic_evalkit.graders.base import Grader
    from agentic_evalkit.targets.base import ExecutionTarget

#: If a sample's output, once serialized, is bigger than this many bytes, it
#: gets moved ("spilled") out to the artifact store and replaced in the run
#: result with just a reference pointing at it (plan Task 11, Step 5,
#: requirement 8), rather than being kept directly inline. This keeps large
#: tool outputs, logs, or generated files from bloating the in-memory or
#: JSON-serialized ``EvalRunResult``, while still letting you go fetch the
#: full content later if you need it.
_LARGE_OUTPUT_THRESHOLD_BYTES = 8192

#: The most characters of a raising target's/grader's exception message the
#: runner keeps on a per-sample error result. An exception message can echo
#: target- or grader-controlled text, so the message is first stripped of
#: secret-shaped substrings (``self._redaction_policy``) and then capped at
#: this length -- mirroring the redact-then-bound treatment ADR-0018 applies
#: to judge candidate output, so one raising sample can neither leak a secret
#: nor bloat the stored result with an unbounded message.
_MAX_ERROR_MESSAGE_CHARS = 8192

EventSink = Callable[[RunEvent], None]

#: The one method the runner actually needs from a ``BenchmarkAdapter``
#: (design §7): something callable that turns one raw ``SourceRecord`` into
#: one ``EvalSample``, matching ``BenchmarkAdapter.prepare``'s signature.
#: The runner only ever calls this ``prepare`` step -- it never checks
#: correct answers ("oracles") or gathers benchmark-wide statistics itself;
#: those are the adapter's own responsibility, not the runner's.
Adapter = Callable[[SourceRecord], EvalSample]


@runtime_checkable
class _CatalogProtocol(Protocol):
    """The minimal shape of "a dataset catalog" that the runner actually needs.

    This is defined here, locally, instead of importing the real catalog
    class from ``agentic_evalkit.datasets.catalog`` -- because the runner
    doesn't care about any particular catalog implementation. All it needs
    is something that can resolve a ``DatasetRef`` once, and then let it
    iterate over the records in that resolved dataset. Any object that has
    these two methods -- a real ``DatasetCatalog``, a single dataset
    provider, or a lightweight fake used in tests -- satisfies this
    protocol and can be passed to the runner.
    """

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset: ...

    def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]: ...


class _PrepareAdapter(Protocol):
    """The one piece of ``BenchmarkAdapter`` the runner actually calls: its ``prepare`` method."""

    def prepare(self, record: SourceRecord) -> EvalSample: ...


ClockFactory = Callable[[], datetime]
IdFactory = Callable[[], str]


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _default_id_factory() -> str:
    return uuid4().hex


class EvalRunner:
    """Runs one manifest through resolve -> prepare -> execute -> grade, end to end.

    Args:
        catalog: Anything matching :class:`_CatalogProtocol` -- it resolves
            a ``DatasetRef`` once per run, then lets the runner iterate over
            the ``SourceRecord`` values in that resolved dataset.
        adapters: A lookup of ``BenchmarkAdapter``-like objects, keyed by
            name. The runner only ever calls their ``prepare`` method. The
            name used to look one up comes from the manifest's ``adapter``
            field.
        targets: A lookup of
            :class:`~agentic_evalkit.targets.base.ExecutionTarget` instances
            (each one a wrapper around some AI system being tested), keyed
            by name. The name comes from the manifest's ``target_name``
            field.
        graders: A lookup of :class:`~agentic_evalkit.graders.base.Grader`
            instances (things that judge whether an output was correct),
            keyed by name. The name comes from the manifest's ``grader``
            field.
        artifact_store: Where outputs that are too large to keep inline get
            saved instead (see ``_LARGE_OUTPUT_THRESHOLD_BYTES``).
        clock: Where the runner gets the current time from. Defaults to the
            real ``datetime.now(UTC)``, but tests can substitute a fake
            clock that returns fixed, predictable timestamps.
        id_factory: Where the runner gets a new run ID from. Defaults to
            generating a random UUID, but tests can substitute something
            that returns predictable IDs instead.
        redaction_policy: The rules used to strip out anything that looks
            like a secret from output bytes before they're spilled to the
            artifact store (see ``_spill_large_output``). Defaults to
            :data:`~agentic_evalkit.reporters.base.DEFAULT_REDACTION_POLICY`,
            so spilled artifacts are redacted automatically and a real run
            never accidentally writes a raw secret to disk. A caller can
            supply a custom ``RedactionPolicy`` to change which patterns
            count as secrets, or pass ``RedactionPolicy()`` with no patterns
            at all -- the single supported way to deliberately turn this
            protection off. Passing ``None`` is still accepted for backward
            compatibility and behaves identically (redaction fully
            disabled), but is deprecated: it emits a ``DeprecationWarning``
            and is normalized internally to ``RedactionPolicy()``. It ships
            deprecated in 0.4.0; support for it will be removed in 0.5.0.
    """

    def __init__(
        self,
        *,
        catalog: _CatalogProtocol,
        adapters: Mapping[str, _PrepareAdapter],
        targets: Mapping[str, ExecutionTarget],
        graders: Mapping[str, Grader],
        artifact_store: ArtifactStore,
        clock: ClockFactory = _default_clock,
        id_factory: IdFactory = _default_id_factory,
        redaction_policy: RedactionPolicy | None = DEFAULT_REDACTION_POLICY,
    ) -> None:
        redaction_policy = _resolve_redaction_policy(redaction_policy, caller="EvalRunner")
        self._catalog = catalog
        self._adapters = dict(adapters)
        self._targets = dict(targets)
        self._graders = dict(graders)
        self._artifact_store = artifact_store
        self._clock = clock
        self._id_factory = id_factory
        self._redaction_policy = redaction_policy

    async def run(
        self,
        manifest: EvalRunManifest,
        event_sink: EventSink | None = None,
    ) -> EvalRunResult:
        """Run everything ``manifest`` describes and return the full, provenance-carrying result.

        ``manifest`` is treated as read-only: this method only ever reads
        values from it and never writes anything back onto it (requirement
        12). If the code awaiting this coroutine gets cancelled (for
        example via ``task.cancel()``), that cancellation is honored --
        Python raises ``asyncio.CancelledError`` here -- but only after any
        attempts that were already in flight get a chance to finish or
        notice the cancellation themselves first (requirement 11). Nothing
        is left running unsupervised in the background.

        The manifest is validated before a ``run_id`` is even generated.
        That ordering matters: if validation fails (for example, because of
        a typo in a component name), that's treated as "the run never
        actually started" rather than "a run started and then broke" -- a
        ``ManifestValidationError`` is raised directly here, and none of the
        failure-handling logic described below applies to it.

        From the point the dataset starts resolving onward, a ``run_id``
        exists and the ``RunStarted`` event has already been sent. If
        anything goes wrong from this point on -- the dataset provider
        raises an error, the run gets cancelled, or any other unexpected
        exception occurs -- that counts as our own infrastructure breaking
        (as opposed to the AI system under test just giving a wrong
        answer). In that case, exactly one
        :class:`~agentic_evalkit.events.RunFailed` event is sent for the
        already-known ``run_id``, and then the original exception is
        re-raised exactly as it was -- never swallowed, never replaced with
        something else. That preserves existing behavior for callers like
        the CLI, which decides its exit code based on the exception type --
        this method just also emits the extra event before re-raising.
        Because of this, ``RunCompleted`` is only ever sent when everything
        succeeded, and is never sent together with ``RunFailed`` for the
        same run.
        """
        sink: EventSink = event_sink if event_sink is not None else _noop_sink
        self._validate_manifest(manifest)

        run_id = self._id_factory()
        started_at = self._clock()
        sink(
            RunStarted(
                run_id=run_id,
                run_name=manifest.run_name,
                total_samples=manifest.selection.limit,
                started_at=started_at,
            )
        )

        try:
            resolved_dataset = await self._catalog.resolve(manifest.dataset_ref)
            sink(
                DatasetResolved(
                    run_id=run_id,
                    dataset_id=resolved_dataset.dataset_id,
                    dataset_revision=resolved_dataset.revision,
                    resolved_at=self._clock(),
                )
            )

            samples = await self._prepare_samples(manifest, resolved_dataset)
            sample_results = await self._execute_all(run_id, manifest, samples, sink)

            summary = _summarize(sample_results)
            finished_at = self._clock()
            sink(RunCompleted(run_id=run_id, total_samples=summary.total, finished_at=finished_at))
        except BaseException as error:
            self._emit_run_failed(sink, run_id=run_id, error=error)
            raise

        return EvalRunResult(
            run_id=run_id,
            manifest=manifest,
            resolved_dataset=resolved_dataset,
            samples=tuple(sample_results),
            summary=summary,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _emit_run_failed(self, sink: EventSink, *, run_id: str, error: BaseException) -> None:
        """Send one ``RunFailed`` event: ``run_id`` was aborted by an infrastructure failure.

        Whatever happens inside this method, the caller always re-raises
        the original ``error`` afterward. So if the event sink itself
        raises an exception while handling this notification, that
        secondary problem is simply thrown away rather than being allowed
        to hide or replace the real reason the run failed.
        """
        # If the event sink itself is broken and raises here, that must
        # never hide or replace the real failure (`error`) -- which the
        # caller is going to re-raise regardless of what happens in this
        # method.
        with contextlib.suppress(Exception):
            sink(
                RunFailed(
                    run_id=run_id,
                    error_type=type(error).__name__,
                    message=self._safe_error_message(error),
                    failed_at=self._clock(),
                )
            )

    def _validate_manifest(self, manifest: EvalRunManifest) -> None:
        """Check the manifest's component names and settings before doing anything else.

        (Requirement 1.)

        Checking this upfront -- before resolving the dataset or running
        anything -- means a typo in a component name is caught
        immediately, instead of producing a confusing, partially completed
        run.
        """
        missing: dict[str, JsonValue] = {}
        if manifest.adapter not in self._adapters:
            missing["adapter"] = manifest.adapter
        if manifest.grader not in self._graders:
            missing["grader"] = manifest.grader
        if manifest.target_name not in self._targets:
            missing["target_name"] = manifest.target_name
        if missing:
            raise ManifestValidationError(
                message=f"manifest references unknown component(s): {missing}",
                context={"missing": missing},
            )
        if manifest.concurrency < 1:
            raise ManifestValidationError(
                message=f"manifest.concurrency must be >= 1, got {manifest.concurrency}",
                context={"concurrency": manifest.concurrency},
            )
        if manifest.attempts < 1:
            raise ManifestValidationError(
                message=f"manifest.attempts must be >= 1, got {manifest.attempts}",
                context={"attempts": manifest.attempts},
            )

    async def _prepare_samples(
        self, manifest: EvalRunManifest, resolved_dataset: ResolvedDataset
    ) -> tuple[EvalSample, ...]:
        """Pull the selected records from the dataset and adapt each into a sample.

        (Requirements 3-4.)

        The order records come back from ``catalog.iter_records`` is kept
        exactly as-is. That's what makes requirement 10 possible (the final
        result always lists samples/attempts in the same, predictable
        order): the order is locked in right here, before any concurrent
        execution starts and could otherwise finish samples in a different
        order than they started in.
        """
        adapter = self._adapters[manifest.adapter]
        selection = manifest.selection
        records = [
            record
            async for record in self._catalog.iter_records(
                resolved_dataset, offset=selection.offset, limit=selection.limit
            )
        ]
        return tuple(adapter.prepare(record) for record in records)

    async def _execute_all(
        self,
        run_id: str,
        manifest: EvalRunManifest,
        samples: tuple[EvalSample, ...],
        sink: EventSink,
    ) -> list[SampleResult]:
        """Run every sample/attempt with a cap on how many run at once, in a predictable order.

        (Requirements 5, 9, 10, 11.)

        Every ``(sample, attempt)`` combination becomes its own task inside
        an ``asyncio.TaskGroup`` (a way of running several async tasks
        together and waiting for all of them to finish). A semaphore -- a
        simple counter that only lets a limited number of tasks proceed at
        once -- caps how many of these run at the same time, based on
        ``manifest.concurrency``. Results are written into a list that was
        already sized and laid out in advance, with each task writing to
        its own fixed slot (based on its position in the plan, not on when
        it happens to finish). That guarantees the returned list is always
        in the same sample/attempt order, no matter which task actually
        completes first. If the caller cancels the surrounding task,
        ``TaskGroup`` automatically cancels every child task that's still
        running, and only re-raises ``CancelledError`` once all of them
        have actually stopped -- so no child task is ever left running
        unsupervised in the background.
        """
        target = self._targets[manifest.target_name]
        grader = self._graders[manifest.grader]
        semaphore = asyncio.Semaphore(manifest.concurrency)

        attempt_plan = [
            (sample, attempt) for sample in samples for attempt in range(1, manifest.attempts + 1)
        ]
        results: list[SampleResult | None] = [None] * len(attempt_plan)

        async def _run_one(index: int, sample: EvalSample, attempt: int) -> None:
            async with semaphore:
                results[index] = await self._execute_and_grade(
                    run_id=run_id,
                    sample=sample,
                    attempt=attempt,
                    target=target,
                    grader=grader,
                    grader_name=manifest.grader,
                    timeout_seconds=manifest.timeout_seconds,
                    sink=sink,
                )

        async with asyncio.TaskGroup() as group:
            for index, (sample, attempt) in enumerate(attempt_plan):
                group.create_task(_run_one(index, sample, attempt))

        return [result for result in results if result is not None]

    async def _execute_and_grade(
        self,
        *,
        run_id: str,
        sample: EvalSample,
        attempt: int,
        target: ExecutionTarget,
        grader: Grader,
        grader_name: str,
        timeout_seconds: float | None,
        sink: EventSink,
    ) -> SampleResult:
        """Run one (sample, attempt) through execute -> grade -> move large output out of memory.

        Grading only ever happens if the execution's status came back as
        ``COMPLETED`` (requirement 6). Every other possible status --
        failed, timed out, cancelled, or errored -- is left exactly as it
        is on the returned ``NormalizedExecutionResult``, with ``grade``
        set to ``None`` (requirement 7). The runner deliberately never
        turns one of these into a grader-style "abstain" or "fail" verdict,
        because doing so would blur together two very different
        situations: "the AI system under test broke or didn't run" versus
        "the AI system under test ran fine but gave a wrong answer."
        Keeping them separate means a caller can always tell which one
        actually happened.

        When grading does run, it always sees the execution result exactly
        as the target originally returned it. That's because
        ``_spill_large_output`` (which moves oversized output out to the
        artifact store) deliberately doesn't run until *after* grading has
        already finished (ADR-0017) -- so a grader is never handed a
        stripped-out ``output=None`` placeholder just because the real
        output happened to be large. Spilling only affects how the final
        result gets stored, never what the grader is allowed to see.

        All three components this coroutine drives -- the target, the
        grader, and the artifact store it spills to -- are fault-isolated
        per sample
        (``_execute_isolated``/``_grade_isolated``/``_spill_isolated``): if
        the target, the grader, or the artifact store raises while working
        on *this* sample, that raise is converted into this sample's own
        error record (``ExecutionStatus.ERROR``/``GradeStatus.ERROR``, or an
        ``output_spill_failed`` record on the execution) rather than being
        allowed to escape. Because this coroutine therefore never raises for
        an ordinary target/grader/store failure, the surrounding
        ``TaskGroup`` (see ``_execute_all``) never cancels the other
        in-flight samples, no already-completed result is discarded, and
        ``RunCompleted`` still fires. This makes all three boundaries
        symmetric with the judge-transport isolation ADR-0020 already
        applied inside ``JudgeGrader``. ``asyncio.CancelledError`` is
        deliberately *not* isolated (see the helpers), so cancelling a run
        still cancels it. Note the scope: it is those three *components*
        that are isolated, not every call this coroutine makes. The
        ``sink(...)`` and ``self._clock()`` calls interleaved between them
        are still unguarded, so a caller-supplied event sink that raises
        does abort the run.

        The spill boundary differs from the other two in one deliberate
        way: a failed spill leaves ``execution.status`` -- and therefore
        this sample's already-computed ``grade`` -- exactly as they were.
        The store refusing the bytes says nothing about how the attempt
        went, and by ADR-0017 the grader had already seen the full inline
        output before the spill was even attempted. Marking the sample
        ``ERROR`` would throw away a genuinely earned verdict over a
        storage problem, which is precisely the kind of evidence loss this
        isolation exists to prevent.
        """
        sink(
            SampleStarted(
                run_id=run_id,
                sample_id=sample.sample_id,
                attempt=attempt,
                started_at=self._clock(),
            )
        )

        execution = await self._execute_isolated(
            sample=sample, attempt=attempt, target=target, timeout_seconds=timeout_seconds
        )
        sink(
            ExecutionCompleted(
                run_id=run_id,
                sample_id=sample.sample_id,
                attempt=attempt,
                status=execution.status,
                completed_at=self._clock(),
            )
        )

        grade: GradeResult | None = None
        if execution.status is ExecutionStatus.COMPLETED:
            grade = await self._grade_isolated(
                sample=sample, execution=execution, grader=grader, grader_name=grader_name
            )
            sink(
                GradeCompleted(
                    run_id=run_id,
                    sample_id=sample.sample_id,
                    attempt=attempt,
                    status=grade.status,
                    completed_at=self._clock(),
                )
            )

        # This spill step always runs, and deliberately happens after
        # grading (ADR-0017): even an execution that was never gradable
        # (for example, one that timed out) can still be carrying a huge
        # output that needs to be moved out to the artifact store before we
        # store the final result. It is fault-isolated like the two steps
        # above: an artifact store that refuses or fails to write the bytes
        # degrades this one sample's stored output instead of tearing down
        # the run around it.
        execution = self._spill_isolated(execution)
        sink(
            SampleCompleted(
                run_id=run_id,
                sample_id=sample.sample_id,
                attempt=attempt,
                completed_at=self._clock(),
            )
        )
        return SampleResult(sample=sample, execution=execution, grade=grade)

    async def _execute_isolated(
        self,
        *,
        sample: EvalSample,
        attempt: int,
        target: ExecutionTarget,
        timeout_seconds: float | None,
    ) -> NormalizedExecutionResult:
        """Run one attempt, converting a raising ``target.execute`` into an error result.

        Per-sample fault isolation, symmetric with the judge-transport
        isolation of ADR-0020: if the execution target raises, only this one
        sample is affected -- it is recorded as an ``ExecutionStatus.TIMEOUT``
        result for a ``TimeoutError`` (``asyncio.TimeoutError`` is the same
        builtin on 3.11+), or an ``ExecutionStatus.ERROR`` result otherwise
        -- instead of the raise escaping to cancel every other in-flight
        sample. Only ``Exception`` is caught: ``asyncio.CancelledError`` is a
        ``BaseException``, not an ``Exception``, so cancelling a run still
        actually cancels it (mirroring ``_judge_with_bounded_retries``).
        """
        started_at = self._clock()
        try:
            return await target.execute(sample, attempt=attempt, timeout_seconds=timeout_seconds)
        except Exception as error:
            return self._target_error_result(
                sample=sample,
                attempt=attempt,
                error=error,
                started_at=started_at,
                finished_at=self._clock(),
            )

    def _target_error_result(
        self,
        *,
        sample: EvalSample,
        attempt: int,
        error: Exception,
        started_at: datetime,
        finished_at: datetime,
    ) -> NormalizedExecutionResult:
        """Build the ``ERROR``/``TIMEOUT`` execution result for a target that raised.

        The exported error taxonomy is wired in here: a ``TimeoutError``
        becomes a :class:`~agentic_evalkit.errors.TargetTimeout`, anything
        else a :class:`~agentic_evalkit.errors.TargetFailure`. That wrapper's
        stable ``code`` (``"target_timeout"``/``"target_failure"``) and its
        redacted, bounded message are what get recorded on the result's
        ``error`` field, alongside the original exception's type name.
        """
        message = self._safe_error_message(error)
        wrapped: TargetTimeout | TargetFailure
        if isinstance(error, TimeoutError):
            wrapped = TargetTimeout(message=message)
            status = ExecutionStatus.TIMEOUT
        else:
            wrapped = TargetFailure(message=message)
            status = ExecutionStatus.ERROR
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output=None,
            status=status,
            error={
                "type": type(error).__name__,
                "code": wrapped.code,
                "message": wrapped.message,
            },
            started_at=started_at,
            finished_at=finished_at,
        )

    async def _grade_isolated(
        self,
        *,
        sample: EvalSample,
        execution: NormalizedExecutionResult,
        grader: Grader,
        grader_name: str,
    ) -> GradeResult:
        """Grade one execution, converting a raising ``grader.grade`` into an ERROR grade.

        Symmetric with ``_execute_isolated`` and ADR-0020: a grader that
        raises yields a single ``GradeStatus.ERROR`` result for this one
        sample rather than aborting the whole run. This isolates *arbitrary*
        graders -- ``JudgeGrader`` isolates its own transport failures
        internally, but a plain grader whose ``grade`` raises is only
        survivable because the runner wraps the call here. As in
        ``_execute_isolated``, ``asyncio.CancelledError`` is not caught.
        """
        try:
            return await grader.grade(sample, execution)
        except Exception as error:
            return self._grader_error_result(sample=sample, error=error, grader_name=grader_name)

    def _grader_error_result(
        self, *, sample: EvalSample, error: Exception, grader_name: str
    ) -> GradeResult:
        """Build the ``GradeStatus.ERROR`` grade for a grader that raised.

        Wraps the raise in a :class:`~agentic_evalkit.errors.GraderError` so
        the error taxonomy is actually used, and records -- mirroring the
        ADR-0020 ``judge_transport_error`` convention -- the original
        exception's type (``grader_error``), the wrapper's stable ``code``
        (``grader_error_code``), and a redacted, bounded message
        (``grader_error_message``). ``hard_gate`` is always ``False``: a
        grader breaking is never allowed to gate a release.
        """
        wrapped = GraderError(message=self._safe_error_message(error))
        return GradeResult(
            sample_id=sample.sample_id,
            grader=grader_name,
            status=GradeStatus.ERROR,
            hard_gate=False,
            evidence={
                "grader_error": type(error).__name__,
                "grader_error_code": wrapped.code,
                "grader_error_message": wrapped.message,
            },
            created_at=self._clock(),
        )

    def _safe_error_message(self, error: BaseException) -> str:
        """Redact secret-shaped substrings from ``str(error)`` and cap its length.

        Mirrors the redact-then-truncate order (and truncation marker)
        ADR-0018 applies to judge candidate output: the runner's own
        configured secret patterns are stripped first (reusing the same
        ``_compiled_secret_patterns``/``_redact`` this module already uses
        for spilling), then the message is bounded at
        ``_MAX_ERROR_MESSAGE_CHARS``. An exception message can echo target-
        or grader-controlled text, so it is never persisted raw. Takes
        ``BaseException`` rather than ``Exception`` because ``run()`` catches
        ``BaseException`` (a cancellation reaches this same path via
        ``_emit_run_failed``), not just ``Exception``.
        """
        message = str(error)
        patterns = self._compiled_secret_patterns()
        if patterns:
            message = _redact(message, patterns)
        if len(message) > _MAX_ERROR_MESSAGE_CHARS:
            omitted = len(message) - _MAX_ERROR_MESSAGE_CHARS
            kept = message[:_MAX_ERROR_MESSAGE_CHARS]
            message = f"{kept}...[truncated, {omitted} chars omitted]"
        return message

    def _spill_isolated(self, execution: NormalizedExecutionResult) -> NormalizedExecutionResult:
        """Spill an oversized output, converting a raising artifact store into a degraded result.

        The third per-sample fault-isolated boundary, added to the
        target/grader pair by the same ADR-0020 amendment reasoning: the
        artifact store is just as capable of raising as a target or a
        grader is (an oversized payload, a full disk, a directory that
        turned read-only), and ``_spill_large_output`` is called from inside
        the ``TaskGroup`` in ``_execute_all``. Left unguarded, one store
        failure cancelled every in-flight sibling and escaped
        ``EvalRunner.run`` as an ``ExceptionGroup`` -- which, not being an
        ``AgenticEvalkitError``, bypassed the CLI's documented exit-code
        mapping, so no report was ever written and every result already
        graded in that run was lost.

        As in ``_execute_isolated`` and ``_grade_isolated``, only
        ``Exception`` is caught: ``asyncio.CancelledError`` is a
        ``BaseException``, so cancelling a run still actually cancels it.
        """
        try:
            return self._spill_large_output(execution)
        except Exception as error:
            return self._spill_failure_result(execution=execution, error=error)

    def _spill_failure_message(self, error: Exception) -> str:
        """Describe ``error`` for the record, without ever raising while doing so.

        ``_safe_error_message`` is not itself guaranteed to succeed here.
        It calls ``str(error)`` on an exception this runner did not create,
        and it compiles ``self._redaction_policy``'s patterns -- the very
        call that can raise ``re.error`` inside ``_spill_large_output`` when
        a caller supplied a malformed pattern, which is one of the failures
        ``_spill_isolated`` is supposed to absorb. Letting it raise from the
        failure handler would put the original exception right back into the
        ``TaskGroup`` and cancel every sibling: the isolation would hold for
        every failure except the one it was handling.

        So the fallback is deliberately the least it can be: the exception's
        class name, which is an attribute of a class object rather than
        anything derived from target- or store-controlled text, and a note
        saying the message could not be rendered. Nothing is swallowed --
        the failure is still recorded, just with less detail than usual.
        """
        try:
            return self._safe_error_message(error)
        except Exception:
            return f"{type(error).__name__} (message unavailable: could not be rendered safely)"

    def _spill_failure_result(
        self, *, execution: NormalizedExecutionResult, error: Exception
    ) -> NormalizedExecutionResult:
        """Build the degraded execution result for an artifact store that raised.

        The output is dropped (``output=None``): the whole point of the
        spill is that these bytes are too big to keep inline, and quietly
        keeping them anyway would reintroduce the unbounded payload
        requirement 8 exists to prevent. What replaces them is a typed
        record -- the original exception's class name, the stable
        :class:`~agentic_evalkit.errors.OutputSpillFailed` ``code``, and a
        redacted, bounded message -- so a reader can tell "the answer
        existed but could not be persisted" apart from "there was never an
        answer."

        ``status`` is deliberately left alone. The store refusing the bytes
        is a storage failure, not a verdict on the attempt: the target
        completed, and by ADR-0017 the grader already saw the full inline
        output before this ran. Flipping the status to ``ERROR`` would
        re-bucket an already-earned grade as an operational error in
        ``_summarize`` and in ``stats.aggregate``, and would break the
        standing invariant that a non-``COMPLETED`` execution carries
        ``grade=None``.

        The record lands in two places for two different reasons.
        ``artifacts`` is this boundary's own namespace -- it records what
        the spill did, ``output_ref`` on success and ``output_spill_error``
        on failure, uniformly and always. ``error`` is the sample's
        *primary* diagnosis, so the spill claims it only when nothing else
        has: a target that already reported its own failure alongside a
        large output keeps that diagnosis intact, and the reader still
        finds the spill failure under ``artifacts``.

        Each field gets its **own copy** of the record rather than a shared
        reference to one dict. ``model_copy`` does not deep-copy what it is
        handed, so storing the same object twice would make ``error`` and
        ``artifacts["output_spill_error"]`` a single mutable value wearing
        two names -- and every later stage that rewrites one field
        independently of the other (report-boundary redaction is exactly
        that) would silently be rewriting both.

        Dropping the output is conditional, because not every raise from
        ``_spill_large_output`` comes from the store. Three steps run before
        the size check and before the store is touched at all: ``str()`` on
        the output, compiling ``secret_patterns`` (which raises ``re.error``
        on a malformed caller-supplied pattern -- accepted at construction,
        so a reachable state), and encoding the result. A raise from any of
        those reaches this handler for an output that was never a spill
        candidate, and nulling it would destroy a perfectly good inline
        answer -- for *every* sample in the run, since the policy is
        per-runner -- while recording that the artifact store refused bytes
        it was never offered. So the output is dropped only when it really
        was too big to keep inline, measured on the raw serialization
        (``str`` on JSON-shaped data cannot raise, so this check is always
        available even when the redaction that follows it is not). An output
        within the threshold is bounded by definition, which is all
        requirement 8 asks; it stays inline and gets redacted at the report
        boundary like any other small output.
        """
        wrapped = OutputSpillFailed(message=self._spill_failure_message(error))
        record: dict[str, JsonValue] = {
            "type": type(error).__name__,
            "code": wrapped.code,
            "message": wrapped.message,
        }
        updates: dict[str, object] = {
            "error": execution.error if execution.error is not None else dict(record),
            "artifacts": {**execution.artifacts, OUTPUT_SPILL_ERROR_KEY: dict(record)},
        }
        if self._output_exceeds_inline_threshold(execution):
            updates["output"] = None
        return execution.model_copy(update=updates)

    @staticmethod
    def _output_exceeds_inline_threshold(execution: NormalizedExecutionResult) -> bool:
        """Was this output actually too big to keep inline, ignoring redaction?

        Deliberately measured on the raw serialization rather than the
        redacted one ``_spill_large_output`` sizes up: this is called from a
        failure handler, where the redaction step is exactly what may have
        just failed. Redaction can only substitute a fixed ``[REDACTED]``
        marker for matched runs of text, so the two sizes differ by a bounded
        amount -- and in the corner where redaction would have pushed a
        just-under-threshold output over the line, keeping it inline is the
        safe answer anyway.

        Like ``_spill_failure_message``, this never raises. It runs inside
        the same ``except`` block, so a raise from here would re-enter the
        ``TaskGroup`` and cancel every sibling -- the precise failure the
        isolation exists to prevent, reintroduced by its own handler.
        ``str`` on JSON-shaped data does not raise today, but that rests on
        pydantic coercing the field's contents upstream rather than on
        anything this module enforces, and the cost of not relying on it is
        three lines. On failure the answer is "not oversized", which keeps
        the output inline: unmeasurable is not a reason to destroy data, and
        an output that really was oversized loses only its size bound for
        one sample.
        """
        if execution.output is None:
            return False
        try:
            return len(str(execution.output).encode("utf-8")) > _LARGE_OUTPUT_THRESHOLD_BYTES
        except Exception:
            return False

    def _spill_large_output(
        self, execution: NormalizedExecutionResult
    ) -> NormalizedExecutionResult:
        """Move an oversized output to the artifact store, replacing it with a reference.

        (Requirement 8.)

        This only runs after grading has already happened (see
        ``_execute_and_grade``, ADR-0017) -- so the fact that an output got
        moved out to storage for being too big can never accidentally
        affect what the grader saw. This method's only job is deciding
        what gets saved to disk, not what gets graded.

        Inside a real run this is always reached through
        ``_spill_isolated``, never called directly, so the artifact store is
        free to raise here: a store that refuses or fails to write the bytes
        degrades that one sample (``output=None`` plus an
        ``output_spill_error`` record) instead of aborting the run around
        it.

        Rather than modifying the ``execution`` object that was passed in,
        this builds a brand new ``NormalizedExecutionResult`` via
        ``model_copy`` (the class is immutable/frozen, so its fields can't
        be changed in place). If the output is small enough to stay
        inline, it's returned completely unchanged.

        This is the only place in the whole runner that applies
        ``self._redaction_policy``'s ``secret_patterns`` (the patterns used
        to detect and blank out things that look like secrets). It does
        that redaction on the serialized output text *before* checking its
        size -- so if something does get spilled to disk, it's guaranteed
        to never contain a raw, unredacted credential, honoring the
        promise (made in the events module's docstring) that nothing in
        this pipeline writes out an unredacted output anywhere.

        Outputs that are small enough to stay inline are deliberately left
        alone here, redaction and all -- they're still part of the
        in-memory ``EvalRunResult``, and get redacted exactly once, later,
        at the point a report is actually generated, by
        :func:`agentic_evalkit.reporters.base.apply_redaction` (design
        §12). Redacting them here too, in addition to that later step,
        would be pointless duplicate work. But skipping redaction here
        entirely (and only ever doing it at the report stage) would mean
        the in-memory result -- and anything else that reads it besides a
        rendered report -- would still be holding the raw, unredacted
        secret. So the rule this method follows is: redact only the
        specific bytes that are actually about to leave memory and be
        written to disk as a stored artifact.
        """
        if execution.output is None:
            return execution
        original = str(execution.output)
        patterns = self._compiled_secret_patterns()
        candidate = _redact(original, patterns) if patterns else original
        encoded = candidate.encode("utf-8")
        if len(encoded) <= _LARGE_OUTPUT_THRESHOLD_BYTES:
            return execution
        was_redacted: bool = candidate != original
        ref = self._artifact_store.put_bytes(
            encoded, media_type="application/json", redacted=was_redacted
        )
        # A successful spill clears any spill-failure record already sitting
        # in ``artifacts`` that carries the runner's taxonomy code. The two
        # keys describe the same boundary's outcome and cannot both be true:
        # the bytes are on disk under ``output_ref`` now. Leaving such a
        # record behind would be read *in preference* to the reference,
        # because consumers check the failure key first (see
        # ``HarnessGrader``) -- telling a reader the output "was never
        # persisted" while pointing away from bytes that are sitting right
        # there.
        #
        # In practice the only thing that can have put one there is the
        # target: every attempt builds a fresh ``NormalizedExecutionResult``,
        # so this never inherits a record an earlier attempt's spill wrote.
        #
        # Only a record carrying the taxonomy code is removed. ``artifacts``
        # is target-controlled and this key is not reserved -- that is the
        # whole premise of ``is_output_spill_error_record`` -- so a target
        # that keeps its own upload diagnostics under that name must not have
        # them silently deleted just because its output happened to spill.
        # Such a value cannot cause the misreading above either, since every
        # consumer validates the record's shape before acting on it.
        artifacts = {
            key: value
            for key, value in execution.artifacts.items()
            if not (key == OUTPUT_SPILL_ERROR_KEY and is_output_spill_error_record(value))
        }
        return execution.model_copy(
            update={
                "output": None,
                "artifacts": {**artifacts, OUTPUT_REF_KEY: ref.digest},
            }
        )

    def _compiled_secret_patterns(self) -> tuple[re.Pattern[str], ...]:
        """Compile ``self._redaction_policy.secret_patterns`` into regexes, or none at all.

        Returns an empty tuple when a policy was given but its own
        ``secret_patterns`` list happens to be empty (``RedactionPolicy()``
        -- the supported way to opt out of spill redaction entirely). In
        the normal case, though, the constructor's default value is
        :data:`DEFAULT_REDACTION_POLICY`, which does have patterns defined
        -- so ordinarily, this compiles and returns those.
        """
        return tuple(re.compile(pattern) for pattern in self._redaction_policy.secret_patterns)


def _redact(value: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    """Replace every part of ``value`` matching any of ``patterns`` with the text ``[REDACTED]``.

    This is a plain function that takes a string and returns a new string
    -- it doesn't touch anything else. It deliberately does the same thing
    as ``agentic_evalkit.reporters.base._redact_string``: since that's a
    private helper this module isn't allowed to import directly, the same
    substitution logic is duplicated here, built against the same
    :class:`RedactionPolicy` rules.
    """
    redacted = value
    for pattern in patterns:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _summarize(sample_results: list[SampleResult]) -> RunSummary:
    """Count up how each sample turned out, keeping every kind of "not passed" strictly separate.

    (Requirement 7.)

    ``RunSummary.failed`` specifically means "the AI system under test ran
    successfully, a grader looked at its answer, and the grader said it
    was wrong" (this comes from a ``GradeResult`` outcome, described
    below). It does *not* include cases where the system under test never
    even finished running. An ``ExecutionStatus.FAILED`` result never
    makes it to grading in the first place (requirement 6) -- it means our
    own infrastructure or the target broke, not that the AI gave a wrong
    answer. So it's counted in ``errors`` (alongside
    ``ExecutionStatus.ERROR``), never in ``failed``. Mixing the two
    together would wrongly make an infrastructure problem look like the AI
    simply answered incorrectly.
    """
    passed = failed = partial = errors = timeouts = cancelled = abstained = unavailable = 0
    for result in sample_results:
        match result.execution.status:
            case ExecutionStatus.ERROR | ExecutionStatus.FAILED:
                errors += 1
            case ExecutionStatus.TIMEOUT:
                timeouts += 1
            case ExecutionStatus.CANCELLED:
                cancelled += 1
            case ExecutionStatus.COMPLETED:
                pass
        if result.grade is not None:
            match result.grade.status:
                case "pass":
                    passed += 1
                case "fail":
                    failed += 1
                case "partial":
                    partial += 1
                case "abstain":
                    abstained += 1
                case "unavailable":
                    unavailable += 1
                case "error":
                    errors += 1
    return RunSummary(
        total=len(sample_results),
        passed=passed,
        failed=failed,
        partial=partial,
        errors=errors,
        timeouts=timeouts,
        cancelled=cancelled,
        abstained=abstained,
        unavailable=unavailable,
    )


def _noop_sink(_event: RunEvent) -> None:
    """Default event sink used when the caller does not pass one."""


__all__ = ["Adapter", "EvalRunner", "EventSink"]
