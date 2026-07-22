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
)
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY, RedactionPolicy

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
            at all to deliberately turn this protection off.
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
                    message=str(error),
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

        Both the execute step and the grade step are fault-isolated per
        sample (``_execute_isolated``/``_grade_isolated``): if the target
        or the grader raises while working on *this* sample, that raise is
        converted into this sample's own error result
        (``ExecutionStatus.ERROR``/``GradeStatus.ERROR``) rather than being
        allowed to escape. Because this coroutine therefore never raises for
        an ordinary target/grader failure, the surrounding ``TaskGroup``
        (see ``_execute_all``) never cancels the other in-flight samples, no
        already-completed result is discarded, and ``RunCompleted`` still
        fires. This makes the target and grader boundaries symmetric with
        the judge-transport isolation ADR-0020 already applied inside
        ``JudgeGrader``. ``asyncio.CancelledError`` is deliberately *not*
        isolated (see the helpers), so cancelling a run still cancels it.
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
        # store the final result.
        execution = self._spill_large_output(execution)
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

    def _safe_error_message(self, error: Exception) -> str:
        """Redact secret-shaped substrings from ``str(error)`` and cap its length.

        Mirrors the redact-then-truncate order (and truncation marker)
        ADR-0018 applies to judge candidate output: the runner's own
        configured secret patterns are stripped first (reusing the same
        ``_compiled_secret_patterns``/``_redact`` this module already uses
        for spilling), then the message is bounded at
        ``_MAX_ERROR_MESSAGE_CHARS``. An exception message can echo target-
        or grader-controlled text, so it is never persisted raw.
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
        return execution.model_copy(
            update={
                "output": None,
                "artifacts": {**execution.artifacts, "output_ref": ref.digest},
            }
        )

    def _compiled_secret_patterns(self) -> tuple[re.Pattern[str], ...]:
        """Compile ``self._redaction_policy.secret_patterns`` into regexes, or none at all.

        Returns an empty tuple in two different cases: when the policy was
        explicitly set to ``None`` (meaning the caller opted out of spill
        redaction entirely), and when a policy object was given but its
        own ``secret_patterns`` list happens to be empty. In the normal
        case, though, the constructor's default value is
        :data:`DEFAULT_REDACTION_POLICY`, which does have patterns defined
        -- so ordinarily, this compiles and returns those.
        """
        if self._redaction_policy is None:
            return ()
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
