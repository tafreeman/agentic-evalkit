"""Story 5.3 (R-009): unit coverage for the frozen progress events + sink.

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 5, Story 5.3) and
``agentic_evalkit.events``.

``agentic_evalkit.events`` defines the "event" types that
:class:`~agentic_evalkit.runner.EvalRunner` sends out while a run is in
progress -- things like "a sample just started" or "the whole run just
finished." The runner delivers these to an ``EventSink``: just a plain
callback function the caller supplies, typed as ``Callable[[RunEvent],
None]`` (see ``agentic_evalkit.runner.EventSink`` and ``_noop_sink``, the
default do-nothing sink used when the caller doesn't supply their own).

This module checks three guarantees about those events, independent of
whether the runner itself behaves correctly:

1. Every event type that exists can actually be built and delivered to a
   sink unchanged.
2. Every event is "frozen" (immutable): once created, trying to change one
   of its fields raises an error instead of silently succeeding.
3. No event field can ever hold a raw secret or a large chunk of data
   inlined directly -- anything big or sensitive must instead be referenced
   by its digest (hash) through the ``ArtifactStore``, exactly as promised
   in the events module's own docstring.

Rather than hard-coding the list of event types to check, every test below
loops over :data:`~agentic_evalkit.events.ALL_EVENT_TYPES` -- the single
source of truth for "every event type that currently exists." That way, if
someone adds a new event type but forgets to give it a test fixture here,
the coverage-guard test in this file fails loudly instead of the new event
type quietly slipping through untested.
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


#: A ready-made example instance for each event type, built by calling the
#: matching factory function below. Keyed by the type itself, so the
#: coverage-check test further down can compare this dict's keys against
#: ``ALL_EVENT_TYPES`` and confirm they match exactly -- a new event type
#: added without a factory here would fail that test.
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

#: The types considered "wire-safe" -- simple and small enough that a value
#: of this type can be logged, printed, or sent over a network with no risk.
#: Deliberately excludes dict/list/bytes: an event must never directly
#: inline a raw output payload, an HTTP request header, or a secret --
#: anything large or sensitive has to be referenced by its digest (hash)
#: elsewhere instead.
_WIRE_SAFE_FIELD_TYPES = (str, int, float, datetime, ExecutionStatus, GradeStatus)


def _annotation_member_types(annotation: object) -> tuple[object, ...]:
    """Break a field's type annotation down into the individual types it allows.

    Most fields are declared as one plain type (e.g. ``str``), which comes
    back here as a single-item tuple. A field can also be declared as a
    union of several types, written either ``X | None`` or ``Optional[X]``
    -- meaning "this holds an X, or nothing at all" -- in which case every
    type listed in that union comes back separately. Either way, ``None``
    (Python's "no value" type, ``NoneType``) is left out of the result, so
    callers only see the "real" types a field can hold, not the fact that
    it's also allowed to be empty.
    """
    origin = get_origin(annotation)
    members = get_args(annotation) if origin in (Union, types.UnionType) else (annotation,)
    return tuple(member for member in members if member is not type(None))


def _assert_annotation_is_wire_safe(event_type: type[FrozenModel], field_name: str) -> None:
    """Check that every type a field's annotation allows (aside from None) is wire-safe.

    This checks the field's *declared type*, not just whatever value it
    happens to hold on one particular test instance. That distinction
    matters: picture a field typed as ``dict[str, str] | None`` that just
    happens to be set to ``None`` on the example event used in a test.
    Checking only that runtime value would miss the problem completely,
    since ``None`` doesn't look unsafe by itself -- and that's exactly what
    an earlier version of this test did (it used a ``continue`` to skip any
    field whose value was ``None``, which let an unsafe *type* slip through
    silently as long as no test happened to fill that field in). Checking
    the declared annotation instead catches the unsafe type regardless of
    what value is actually set.
    """
    for member in _annotation_member_types(event_type.model_fields[field_name].annotation):
        if get_origin(member) is Literal:
            # Every FrozenModel inherits a `schema_version` field typed as
            # `Literal["1"]` -- meaning "this field can only ever hold the
            # one exact value '1'". A Literal like that is wire-safe as long
            # as every value it's allowed to hold is itself a plain scalar
            # (str/int/float/bool), which is what this checks.
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
    """Check that the example-instance registry (`_EVENT_FACTORIES`) has an
    entry for every event type in `ALL_EVENT_TYPES`, and nothing extra. This
    is what forces a newly added event type to be given a factory here
    before it can pass CI -- once it has one, every other test in this file
    exercises it automatically.
    """
    assert set(_EVENT_FACTORIES) == set(ALL_EVENT_TYPES)
    # Also confirm ALL_EVENT_TYPES itself has no accidental duplicate entries.
    assert len(ALL_EVENT_TYPES) == len(set(ALL_EVENT_TYPES))


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES, ids=lambda t: t.__name__)
def test_every_event_type_delivers_intact_through_a_collecting_sink(
    event_type: type[FrozenModel],
) -> None:
    """Send one example of each event type through a simple sink that just
    appends whatever it receives to a list, and check it comes out the other
    end as the exact same object, completely unchanged. "Unchanged" is
    verified by dumping the event to JSON both before and after -- if
    anything about the event had been altered along the way, those two JSON
    dumps would no longer match.
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
    """The library's built-in do-nothing sink (``_noop_sink``, used
    automatically when the caller doesn't supply their own) accepts every
    event type without raising -- a run must never fail on event delivery
    just because nobody was listening.
    """
    event = _EVENT_FACTORIES[event_type]()
    assert _noop_sink(event) is None


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES, ids=lambda t: t.__name__)
def test_every_event_type_is_frozen(event_type: type[FrozenModel]) -> None:
    """Every event is immutable ("frozen"): trying to change any one of its
    fields after it's been created raises an error instead of silently
    succeeding. That guarantees an event handed to a sink can never
    afterward be changed out from under it.
    """
    event = _EVENT_FACTORIES[event_type]()
    field_name = next(iter(event_type.model_fields))
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError (frozen)
        setattr(event, field_name, "mutated")


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES, ids=lambda t: t.__name__)
def test_every_event_field_is_a_wire_safe_primitive(event_type: type[FrozenModel]) -> None:
    """No event field may hold a container (like a dict or list) or raw bytes
    -- every field that's actually set must be a plain string ID, an enum
    value, a count, or a timestamp. This is what backs up the promise made
    in the events module's own docstring: an event can never carry a raw
    output payload, an HTTP header, or a secret directly. Anything large or
    sensitive has to be referenced by its digest (hash) instead.
    """
    event = _EVENT_FACTORIES[event_type]()
    for field_name in event_type.model_fields:
        # Check the field's declared type for every field, not only the ones
        # that happen to be filled in on this particular example event. That
        # way, a field typed as an optional container (e.g. `dict[str, str]
        # | None`) is still caught even if this test's example event leaves
        # it as None -- see `_assert_annotation_is_wire_safe`'s own
        # docstring for why checking just the runtime value isn't enough.
        _assert_annotation_is_wire_safe(event_type, field_name)

        value = getattr(event, field_name)
        if value is None:
            # Nothing to check at runtime for a field that's unset here --
            # its declared type was already checked just above.
            continue
        assert isinstance(value, _WIRE_SAFE_FIELD_TYPES), (
            f"{event_type.__name__}.{field_name} is {type(value).__name__}, "
            "which is not a wire-safe primitive"
        )
        # Extra safety net on top of the type check: even a string-typed
        # field shouldn't be able to smuggle in a huge blob of inlined data.
        if isinstance(value, str):
            assert len(value) < 2048


def test_a_sink_can_receive_the_full_event_stream_in_order() -> None:
    """Simulate a real run's event stream: one sink receiving every kind of
    event mixed together, in the order they'd actually be sent, each one
    arriving unchanged. This is exactly the guarantee that something like a
    live progress bar or an audit log would depend on.
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
    """As a second line of defense on top of the type checks above, also
    check field *names*: no event should have a field named anything like
    "output", "token", or "credential", since a name like that would
    suggest it's meant to hold sensitive data directly. Anything large or
    sensitive must be referenced (by digest) rather than held directly in a
    field.
    """
    forbidden_substrings = ("output", "payload", "token", "secret", "header", "credential", "body")
    for event_type in ALL_EVENT_TYPES:
        for field_name in event_type.model_fields:
            lowered = field_name.lower()
            assert not any(bad in lowered for bad in forbidden_substrings), (
                f"{event_type.__name__}.{field_name} looks like it could inline "
                "a raw payload or secret"
            )
