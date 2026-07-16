"""Pipeline orchestration: dataset -> adapter -> target -> grader (plan Task 11).

``EvalRunner`` is the single place that wires together an already-resolved
set of named components (adapters, execution targets, graders) and drives
one :class:`~agentic_evalkit.models.EvalRunManifest` through to a complete
:class:`~agentic_evalkit.models.EvalRunResult`. It does not select, import,
or construct those components itself -- the caller (typically the CLI or a
higher-level catalog/registry, added in a later task) injects them by name.

The runner is deliberately decoupled from the concrete dataset catalog: it
depends only on the small, local ``_CatalogProtocol`` defined below (an
async ``resolve`` plus an async-iterator ``iter_records``), not on
``agentic_evalkit.datasets.catalog.DatasetCatalog``. This keeps the runner
importable and testable without pulling in provider/cache machinery, and
lets any object with the right shape (a real catalog, a fake, a filtered
view) stand in for it.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import uuid4

from agentic_evalkit.errors import JsonValue, ManifestValidationError
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

#: Serialized ``NormalizedExecutionResult.output`` larger than this many bytes
#: is spilled to the artifact store and replaced with a reference (plan
#: Task 11, Step 5, requirement 8) instead of being kept inline in the run
#: result. This keeps large tool outputs, logs, or generated files out of the
#: in-memory/JSON-serialized ``EvalRunResult`` while remaining retrievable.
_LARGE_OUTPUT_THRESHOLD_BYTES = 8192

EventSink = Callable[[RunEvent], None]

#: A structural adapter boundary matching ``BenchmarkAdapter.prepare`` (design
#: §7). The runner only ever calls ``prepare``; it never validates oracles or
#: aggregates benchmark metadata itself.
Adapter = Callable[[SourceRecord], EvalSample]


@runtime_checkable
class _CatalogProtocol(Protocol):
    """The runner's own, minimal view of a dataset catalog.

    Deliberately local to this module rather than imported from
    ``agentic_evalkit.datasets.catalog``: the runner depends on this shape
    (resolve a ``DatasetRef`` once, then iterate records from the resolved
    dataset), not on any particular catalog implementation. Any object -- a
    real ``DatasetCatalog``, a single provider, or a test fake -- that
    exposes these two methods satisfies this protocol.
    """

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset: ...

    def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]: ...


class _PrepareAdapter(Protocol):
    """Matches the subset of ``BenchmarkAdapter`` the runner calls."""

    def prepare(self, record: SourceRecord) -> EvalSample: ...


ClockFactory = Callable[[], datetime]
IdFactory = Callable[[], str]


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _default_id_factory() -> str:
    return uuid4().hex


class EvalRunner:
    """Drives one manifest through resolve -> prepare -> execute -> grade.

    Args:
        catalog: Anything satisfying :class:`_CatalogProtocol` -- resolves a
            ``DatasetRef`` once per run and iterates ``SourceRecord`` values
            from the resolved dataset.
        adapters: Named ``BenchmarkAdapter``-shaped objects (only ``prepare``
            is called), keyed by the name a manifest's ``adapter`` field
            references.
        targets: Named :class:`~agentic_evalkit.targets.base.ExecutionTarget`
            instances, keyed by the name a manifest's ``target_name`` field
            references.
        graders: Named :class:`~agentic_evalkit.graders.base.Grader`
            instances, keyed by the name a manifest's ``grader`` field
            references.
        artifact_store: Where large outputs are spilled (see
            ``_LARGE_OUTPUT_THRESHOLD_BYTES``).
        clock: Injectable timestamp source; defaults to ``datetime.now(UTC)``.
            Tests can inject a deterministic clock.
        id_factory: Injectable run-ID source; defaults to a random UUID hex
            string. Tests can inject a deterministic sequence.
        redaction_policy: Policy applied to spilled artifact bytes before
            they are written (see ``_spill_large_output``). Defaults to
            :data:`~agentic_evalkit.reporters.base.DEFAULT_REDACTION_POLICY`,
            so spilled artifacts are redacted by default and real runs never
            spill raw secrets to disk. A caller may pass a custom
            ``RedactionPolicy`` to tune the patterns, or ``RedactionPolicy()``
            (empty patterns) to explicitly opt out of spill redaction.
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
        """Execute ``manifest`` and return the complete, provenance-carrying result.

        ``manifest`` is never mutated: every value this method needs is read
        from it, and nothing is written back (requirement 12). Cancellation
        of the awaiting task (e.g. ``task.cancel()``) propagates
        ``asyncio.CancelledError`` after any already-scheduled attempts are
        allowed to finish or observe cancellation themselves (requirement
        11); no attempt is left as an orphan background task.

        Manifest validation happens before any ``run_id`` is minted, so a
        ``ManifestValidationError`` from a typo'd component name is a
        precondition failure, not a run that started and then aborted -- it
        raises directly and never reaches the failure handling below.

        Everything from the dataset resolving onward runs with one ``run_id``
        already established (and ``RunStarted`` already emitted). Any
        exception that escapes that region -- a dataset provider error, the
        awaiting task being cancelled (``asyncio.CancelledError``), or any
        other unexpected failure -- is an infrastructure-level abort: exactly
        one :class:`~agentic_evalkit.events.RunFailed` is emitted for the
        already-known ``run_id`` and the original exception is re-raised
        unchanged (never swallowed, never replaced), so callers -- the CLI
        maps exception types to exit codes -- see the same behavior as
        before, plus the added event. ``RunCompleted`` is therefore emitted
        only on the success path and never alongside ``RunFailed``.
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
        """Emit one ``RunFailed`` for an infrastructure-level abort of ``run_id``.

        The original ``error`` is always re-raised by the caller regardless
        of what happens here: if the sink itself raises while handling this
        notification, that secondary failure is discarded rather than
        allowed to replace or mask the run's real failure cause.
        """
        # A broken sink must never replace or mask the caller's real failure
        # (`error`, re-raised by the caller regardless).
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
        """Requirement 1: validate component names and capabilities up front.

        Failing fast here -- before any dataset resolution or execution --
        means a typo'd component name never produces a partially-run,
        confusing result.
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
        """Requirements 3-4: iterate the selection and prepare each record.

        Iteration order from ``catalog.iter_records`` is preserved, which is
        what makes requirement 10 (deterministic sample/attempt order in the
        final result) achievable: sample order is fixed here, before any
        concurrent execution begins.
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
        """Requirements 5, 9, 10, 11: bounded concurrency, ordered events/results.

        Every ``(sample, attempt)`` pair becomes one task in an
        ``asyncio.TaskGroup``, gated by a semaphore sized to
        ``manifest.concurrency``. Results are collected into a
        pre-sized list indexed by task order (not completion order), so the
        returned list is in deterministic sample/attempt order regardless of
        which task happens to finish first. If the caller cancels the
        awaiting task, ``TaskGroup`` cancels every still-running child task
        and re-raises ``CancelledError`` once they have all unwound -- no
        child task is left running unsupervised.
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
        timeout_seconds: float | None,
        sink: EventSink,
    ) -> SampleResult:
        """One sample/attempt's full pipeline: execute, grade, spill.

        Requirement 6: grading only ever runs for an execution whose status
        is ``COMPLETED``. Requirement 7: every other status (failed, timeout,
        cancelled, error) is preserved as-is on the returned
        ``NormalizedExecutionResult`` with ``grade=None`` -- the runner never
        collapses these into a grader-level abstain/fail, so a caller can
        always distinguish "the system under test broke" from "the system
        under test answered incorrectly".

        Grading, when it runs, always sees the execution exactly as the
        target returned it -- ``_spill_large_output`` does not run until
        after grading has already happened (ADR-0017), so a grader is never
        handed a spilled ``output=None`` placeholder just because the real
        output was large. Spilling is purely a persistence concern applied
        to the result on its way out.
        """
        sink(
            SampleStarted(
                run_id=run_id,
                sample_id=sample.sample_id,
                attempt=attempt,
                started_at=self._clock(),
            )
        )

        execution = await target.execute(sample, attempt=attempt, timeout_seconds=timeout_seconds)
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
            grade = await grader.grade(sample, execution)
            sink(
                GradeCompleted(
                    run_id=run_id,
                    sample_id=sample.sample_id,
                    attempt=attempt,
                    status=grade.status,
                    completed_at=self._clock(),
                )
            )

        # Unconditional, and deliberately after grading (ADR-0017): a
        # non-completed execution can still carry a huge output that needs
        # spilling for storage even though it was never gradable.
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

    def _spill_large_output(
        self, execution: NormalizedExecutionResult
    ) -> NormalizedExecutionResult:
        """Requirement 8: store large outputs as artifacts, keep a reference.

        Called only after grading has already happened (see
        ``_execute_and_grade``, ADR-0017), so a grader is never handed a
        spilled/``None`` output as a result of size alone -- this method's
        job is exclusively what gets persisted, not what gets graded.

        Builds a *new* ``NormalizedExecutionResult`` via ``model_copy`` (the
        contract is frozen) rather than mutating the target's returned
        instance. Small outputs are left inline unchanged.

        This is the only place in the runner that applies
        ``self._redaction_policy``'s ``secret_patterns``, and it applies them
        to the serialized string *before* the size check -- so a spilled
        artifact never carries a raw credential to disk even though the
        events module docstring promises no unredacted output payload
        anywhere in the pipeline. Inline outputs (below the spill threshold)
        are deliberately left as-is here: they remain part of the in-memory
        ``EvalRunResult`` and are redacted exactly once, at the report
        boundary, by :func:`agentic_evalkit.reporters.base.apply_redaction`
        (design §12). Redacting twice -- once here, once at the report
        boundary -- would be redundant; redacting only here and never at the
        report boundary would leave the in-memory result (and any other
        consumer of it besides a rendered report) holding raw secrets. This
        method redacts only the bytes that are about to leave the in-memory
        result and land on disk as an artifact.
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
        """Compile ``self._redaction_policy.secret_patterns``, or none at all.

        Returns an empty tuple both when the policy was explicitly set to
        ``None`` (opting out of spill redaction) and when a policy was
        supplied with no ``secret_patterns`` of its own. The constructor
        default is :data:`DEFAULT_REDACTION_POLICY`, which does carry
        patterns, so the ordinary path compiles those.
        """
        if self._redaction_policy is None:
            return ()
        return tuple(re.compile(pattern) for pattern in self._redaction_policy.secret_patterns)


def _redact(value: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    """Replace every match of any ``patterns`` entry in ``value`` with ``[REDACTED]``.

    A pure string -> string function, mirroring
    ``agentic_evalkit.reporters.base._redact_string``: this module cannot
    import that private helper, so the same substitution behavior is
    reimplemented locally against the same :class:`RedactionPolicy` contract.
    """
    redacted = value
    for pattern in patterns:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _summarize(sample_results: list[SampleResult]) -> RunSummary:
    """Requirement 7: separated outcome counts, never collapsed together.

    ``RunSummary.failed`` means "the system under test was graded and got
    the task wrong" (a ``GradeResult`` outcome, below). An
    ``ExecutionStatus.FAILED`` result never reaches grading (requirement 6),
    so it is an operational failure, not a task failure; it is counted in
    ``errors`` alongside ``ExecutionStatus.ERROR`` rather than in ``failed``,
    which would otherwise make an infrastructure problem look like an
    incorrect answer.
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
