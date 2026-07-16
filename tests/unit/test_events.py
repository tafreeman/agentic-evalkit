"""Story 5.3 (R-009): unit coverage for the frozen progress events + sink.

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 5, Story 5.3) and
``agentic_evalkit.events``.

``agentic_evalkit.events`` defines the frozen event types
:class:`~agentic_evalkit.runner.EvalRunner` may hand to a caller-supplied
``EventSink`` (a plain ``Callable[[RunEvent], None]``; see
``agentic_evalkit.runner.EventSink`` / ``_noop_sink``). This module pins three
things independent of the runner: every enumerated event type can be built and
delivered intact through the sink contract, every event is frozen (mutation
raises), and no event field can carry a raw secret or a large inlined payload
-- large/sensitive data is referenced by digest through the ``ArtifactStore``
instead (per the module docstring's promise). The event set is iterated from
:data:`~agentic_evalkit.events.ALL_EVENT_TYPES`, never hard-coded, so a newly
added event type that is not given a factory here fails the coverage guard
below rather than silently escaping it.
"""

from __future__ import annotations

import types
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Union, get_args, get_origin

import pytest

from agentic_evalkit.events import (
    ALL_EVENT_TYPES,
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
from agentic_evalkit.models import ExecutionStatus, GradeStatus
from agentic_evalkit.runner import _noop_sink

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentic_evalkit.models.base import FrozenModel

_NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)


def _run_started() -> RunStarted:
    return RunStarted(run_id="run-1", run_name="r", total_samples=2, started_at=_NOW)


def _dataset_resolved() -> DatasetResolved:
    return DatasetResolved(
        run_id="run-1", dataset_id="openai/gsm8k", dataset_revision="sha256:abc", resolved_at=_NOW
    )


def _sample_started() -> SampleStarted:
    return SampleStarted(run_id="run-1", sample_id="s0", attempt=1, started_at=_NOW)


def _execution_completed() -> ExecutionCompleted:
    return ExecutionCompleted(
        run_id="run-1",
        sample_id="s0",
        attempt=1,
        status=ExecutionStatus.COMPLETED,
        completed_at=_NOW,
    )


def _grade_completed() -> GradeCompleted:
    return GradeCompleted(
        run_id="run-1",
        sample_id="s0",
        attempt=1,
        status=GradeStatus.PASS,
        completed_at=_NOW,
    )


def _sample_completed() -> SampleCompleted:
    return SampleCompleted(run_id="run-1", sample_id="s0", attempt=1, completed_at=_NOW)


def _run_completed() -> RunCompleted:
    return RunCompleted(run_id="run-1", total_samples=2, finished_at=_NOW)


def _run_failed() -> RunFailed:
    return RunFailed(run_id="run-1", error_type="RuntimeError", message="boom", failed_at=_NOW)


#: One factory per concrete event type. Keyed by the type so the coverage guard
#: below can assert this registry matches ``ALL_EVENT_TYPES`` exactly -- a new
#: event type with no factory here fails that test.
_EVENT_FACTORIES: dict[type[FrozenModel], Callable[[], RunEvent]] = {
    RunStarted: _run_started,
    DatasetResolved: _dataset_resolved,
    SampleStarted: _sample_started,
    ExecutionCompleted: _execution_completed,
    GradeCompleted: _grade_completed,
    SampleCompleted: _sample_completed,
    RunCompleted: _run_completed,
    RunFailed: _run_failed,
}

#: The wire-safe primitive types an event field may hold. No dict/list/bytes:
#: an event must never inline a raw output payload, request headers, or a
#: secret -- large/sensitive data is referenced by digest elsewhere.
_WIRE_SAFE_FIELD_TYPES = (str, int, float, datetime, ExecutionStatus, GradeStatus)


def _annotation_member_types(annotation: object) -> tuple[object, ...]:
    """Decompose a field annotation into its member types.

    A union (``X | None`` or ``Optional[X]``) yields its args; a bare type
    yields just itself. ``None`` (``NoneType``) is stripped, so the caller
    checks only the value-bearing members of an optional field.
    """
    origin = get_origin(annotation)
    members = get_args(annotation) if origin in (Union, types.UnionType) else (annotation,)
    return tuple(member for member in members if member is not type(None))


def _assert_annotation_is_wire_safe(event_type: type[FrozenModel], field_name: str) -> None:
    """Assert every value-bearing member of a field's annotation is wire-safe.

    This is the annotation-level guarantee: an optional container-typed field
    (e.g. ``dict[str, str] | None``) would be caught here even when its runtime
    value is ``None`` -- exactly the case the old per-instance ``continue``
    skipped silently.
    """
    for member in _annotation_member_types(event_type.model_fields[field_name].annotation):
        if get_origin(member) is Literal:
            # The schema_version discriminator every FrozenModel inherits is
            # Literal["1"] -- a Literal of wire-safe scalars is wire-safe.
            assert all(isinstance(arg, (str, int, float, bool)) for arg in get_args(member)), (
                f"{event_type.__name__}.{field_name} is a Literal of non-scalar values"
            )
            continue
        assert member in _WIRE_SAFE_FIELD_TYPES, (
            f"{event_type.__name__}.{field_name} is annotated with "
            f"{getattr(member, '__name__', member)}, which is not wire-safe"
        )


def _all_events() -> list[RunEvent]:
    return [factory() for factory in _EVENT_FACTORIES.values()]


def test_event_factories_cover_every_enumerated_event_type() -> None:
    """The per-type factory registry matches ``ALL_EVENT_TYPES`` exactly, so a
    newly added event type must be given a factory here to pass -- the rest of
    this module then exercises it automatically.
    """
    assert set(_EVENT_FACTORIES) == set(ALL_EVENT_TYPES)
    # No duplicates and no drift in the source enumeration itself.
    assert len(ALL_EVENT_TYPES) == len(set(ALL_EVENT_TYPES))


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES, ids=lambda t: t.__name__)
def test_every_event_type_delivers_intact_through_a_collecting_sink(
    event_type: type[FrozenModel],
) -> None:
    """Each event type is delivered through a collecting ``EventSink`` (the
    canonical ``list.append`` sink) and arrives as the same, unmodified
    instance -- byte-for-byte equal on a model round-trip.
    """
    event = _EVENT_FACTORIES[event_type]()
    collected: list[RunEvent] = []

    def _sink(received: RunEvent) -> None:
        collected.append(received)

    _sink(event)

    assert len(collected) == 1
    delivered = collected[0]
    assert delivered is event
    assert type(delivered) is event_type
    assert delivered.model_dump(mode="json") == event.model_dump(mode="json")


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES, ids=lambda t: t.__name__)
def test_noop_sink_accepts_every_event_type(event_type: type[FrozenModel]) -> None:
    """The shipped default sink (``_noop_sink``) accepts every event type
    without raising -- a run with no caller sink must never fail on delivery.
    """
    event = _EVENT_FACTORIES[event_type]()
    assert _noop_sink(event) is None


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES, ids=lambda t: t.__name__)
def test_every_event_type_is_frozen(event_type: type[FrozenModel]) -> None:
    """Every event is immutable: reassigning any field raises, so an event
    handed to a sink can never be mutated in place after emission.
    """
    event = _EVENT_FACTORIES[event_type]()
    field_name = next(iter(event_type.model_fields))
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError (frozen)
        setattr(event, field_name, "mutated")


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES, ids=lambda t: t.__name__)
def test_every_event_field_is_a_wire_safe_primitive(event_type: type[FrozenModel]) -> None:
    """No event field may hold a container or raw bytes: every populated field
    is a plain identifier, enum, count, or timestamp. This is the structural
    guarantee that an event can never inline a raw output payload, request
    header, or secret (large/sensitive data is referenced by digest instead).
    """
    event = _EVENT_FACTORIES[event_type]()
    for field_name in event_type.model_fields:
        # Annotation-level guarantee, checked for every field (not just the
        # populated ones): the declared type admits only wire-safe members, so
        # an optional container-typed field is rejected even when its runtime
        # value is None. This is what the old `continue`-on-None loop skipped.
        _assert_annotation_is_wire_safe(event_type, field_name)

        value = getattr(event, field_name)
        if value is None:
            # An unset optional field carries no value to type-check at
            # runtime; the annotation assertion above already covered it.
            continue
        assert isinstance(value, _WIRE_SAFE_FIELD_TYPES), (
            f"{event_type.__name__}.{field_name} is {type(value).__name__}, "
            "which is not a wire-safe primitive"
        )
        # Defense in depth: no field is a large inlined blob even as a string.
        if isinstance(value, str):
            assert len(value) < 2048


def test_a_sink_can_receive_the_full_event_stream_in_order() -> None:
    """A single collecting sink receives a heterogeneous stream of every event
    type, in emission order, each instance intact -- the delivery contract a
    live progress view / audit trail depends on.
    """
    stream = _all_events()
    collected: list[RunEvent] = []

    def _sink(event: RunEvent) -> None:
        collected.append(event)

    for event in stream:
        _sink(event)

    assert collected == stream
    assert [type(item).__name__ for item in collected] == [type(item).__name__ for item in stream]


def test_events_carry_no_field_named_like_a_raw_payload_or_secret() -> None:
    """Guard the contract by field-name too: no event declares a field whose
    name suggests it inlines raw output, headers, a token, or a credential.
    Large/sensitive data must be referenced (by digest), never named-and-held.
    """
    forbidden_substrings = ("output", "payload", "token", "secret", "header", "credential", "body")
    for event_type in ALL_EVENT_TYPES:
        for field_name in event_type.model_fields:
            lowered = field_name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), (
                f"{event_type.__name__}.{field_name} looks like it could inline "
                "a raw payload or secret"
            )
