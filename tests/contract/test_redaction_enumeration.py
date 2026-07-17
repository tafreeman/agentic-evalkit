"""Reporter/event redaction-enumeration contract (Story 2.2, R-002 P0).

"Redaction" means blanking out secrets (API keys, tokens, and so on)
before a result is ever written out in any format. "Enumeration" here
means this test keeps an explicit, hand-written list of every format that
redaction must cover, kept separate from the list of formats that simply
exist -- so that adding a new report format without also wiring up its
redaction is a mistake this test can actually catch, instead of something
that silently falls through the cracks. This module also enumerates every
event type the runner can emit, to make sure none of them can carry an
un-redactable raw payload.

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 2, Story 2.2) and
the TEA test design (R-002, an internal test-design/risk reference). This
test has passed ever since the P0 (priority-0, i.e. highest-priority)
branch landed the "seams" it depends on -- the integration points, such as
a shared registry both the production code and this test read from, that
make a check like this possible. A 2026-07-04 code review then changed the
test so a real gap would actually make it fail (in testing terms, made it
"falsifiable"), instead of it being structurally unable to ever catch
anything.

What this contract pins:
  * ``REPORTER_FORMATS`` is the single canonical registry that both CLI
    code paths which write a report to disk select reporters from, and it
    must list every built-in reporter.
  * ``REDACTION_ROUTED_FORMATS`` is maintained BY HAND next to the
    reporters package (deliberately NOT computed from the registry above),
    so the equality check below is a real tripwire: registering a new
    format without consciously also pairing it with redaction routing
    fails CI.
  * ``ALL_EVENT_TYPES`` must equal the ``RunEvent`` union exactly, so an
    event type added to one but not the other fails CI, and no event type
    can escape the wire-safety check below.
"""

from __future__ import annotations

import types
import typing
from datetime import datetime
from enum import Enum

_WIRE_SAFE_SCALARS = (str, int, float, bool, type(None))


def _is_wire_safe(annotation: object) -> bool:
    """True if a field's type is safe to put on an event: an identifier,
    count, enum, or timestamp; an Optional/Union of such types; or a
    ``Literal`` made up of wire-safe scalar values (for example, the
    ``schema_version: Literal["1"]`` version marker every ``FrozenModel``
    inherits). False for anything that is (or contains) a dict, list, or
    any other container -- those could hold arbitrary, possibly sensitive,
    payload data, which is exactly what this check exists to keep off
    events.
    """
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return all(isinstance(arg, _WIRE_SAFE_SCALARS) for arg in typing.get_args(annotation))
    if origin is typing.Union or origin is types.UnionType:
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return bool(args) and all(_is_wire_safe(arg) for arg in args)
    if origin is not None:
        # Anything else with a type parameter -- list[...], dict[...],
        # tuple[...], Sequence[...], and so on -- is a container type,
        # which is exactly what an event must never carry (a container
        # could hold arbitrary, possibly sensitive, data). This branch
        # exists so a container type can't sneak past this check just
        # because it superficially resembled a Union.
        return False
    return (
        annotation in _WIRE_SAFE_SCALARS
        or annotation is datetime
        or (isinstance(annotation, type) and issubclass(annotation, Enum))
    )


def test_wire_safety_rejects_containers() -> None:
    # This test checks the checker itself: a 2026-07-04 code review found
    # a bug where the Union-handling branch of `_is_wire_safe` above
    # wrongly accepted ANY parameterized generic type, which meant
    # `list[str]` and `dict[str, str]` both incorrectly passed the very
    # check that exists to reject them. This test pins down that the bug
    # stays fixed.
    assert not _is_wire_safe(list[str])
    assert not _is_wire_safe(dict[str, str])
    assert not _is_wire_safe(tuple[str, ...])
    assert not _is_wire_safe(dict[str, str] | None)
    assert _is_wire_safe(str | None)
    assert _is_wire_safe(typing.Literal["1"])
    assert _is_wire_safe(datetime)


def test_reporter_registry_lists_all_known_formats() -> None:
    # REPORTER_FORMATS is the one canonical registry of report formats. It
    # must list every built-in reporter class, because the CLI code that
    # writes reports to disk can only ever reach a format through this
    # table -- a reporter class that exists but is not registered here is
    # simply unreachable. If a new reporter class were added but never
    # registered, this test would fail.
    from agentic_evalkit import reporters

    registry = reporters.REPORTER_FORMATS
    assert set(registry.values()) == {
        reporters.JsonReporter,
        reporters.JsonlReporter,
        reporters.MarkdownReporter,
        reporters.HtmlReporter,
    }


def test_every_registered_reporter_is_redaction_routed() -> None:
    # REDACTION_ROUTED_FORMATS is a separate, hand-maintained list living
    # next to REPORTER_FORMATS (the selectable registry of formats).
    # Because it is hand-maintained rather than computed from
    # REPORTER_FORMATS, this equality check is a real tripwire: if someone
    # registers a fifth report format, this test forces them to also
    # consciously add it to REDACTION_ROUTED_FORMATS -- wiring it up to
    # apply_redaction, the secret-scrubbing pass -- rather than letting the
    # new format ship with no redaction at all.
    from agentic_evalkit import reporters

    routed = reporters.REDACTION_ROUTED_FORMATS
    assert set(reporters.REPORTER_FORMATS) == set(routed)


def test_report_command_reporters_derive_from_the_registry() -> None:
    # There are two places in the CLI that write a report to disk. The
    # first is the `run` command; this test covers the second, the
    # standalone `report` command, which must read its list of reporters
    # from the same canonical REPORTER_FORMATS registry -- a code review
    # found it used to keep its own private, separately-typed-out reporter
    # table instead, which could quietly drift out of sync with the real
    # registry. "json" is excluded here because the `report` command
    # doesn't need a JSON reporter: it already starts from canonical run
    # JSON and regenerates the other formats from that.
    from agentic_evalkit import reporters
    from agentic_evalkit.cli.reports import _REPORTERS

    assert set(_REPORTERS) == set(reporters.REPORTER_FORMATS) - {"json"}
    for name, reporter in _REPORTERS.items():
        assert type(reporter) is reporters.REPORTER_FORMATS[name]


def test_all_event_types_matches_the_run_event_union() -> None:
    # This is what makes the event-safety contract below actually
    # meaningful: ALL_EVENT_TYPES (a plain tuple you can loop over) and
    # RunEvent (a type-level Union listing every event class) must name
    # exactly the same set of classes. If a new event type were added to
    # one but not the other, it would silently escape both the
    # wire-safety check below and any other coverage that loops over
    # ALL_EVENT_TYPES.
    from agentic_evalkit import events

    assert set(events.ALL_EVENT_TYPES) == set(typing.get_args(events.RunEvent))


def test_every_event_type_is_enumerated_and_wire_safe() -> None:
    # Every event the runner can emit must (1) be listed in
    # ALL_EVENT_TYPES, and (2) have every one of its fields pass the
    # wire-safety check above: identifiers, enums, counts, and timestamps
    # only -- never a dict, list, or raw output payload that could carry
    # arbitrary, possibly sensitive, data straight onto an event.
    from agentic_evalkit import events

    all_event_types = events.ALL_EVENT_TYPES
    assert all_event_types, "expected a non-empty ALL_EVENT_TYPES registry"

    for event_type in all_event_types:
        for field_name, field in event_type.model_fields.items():
            assert _is_wire_safe(field.annotation), (
                f"{event_type.__name__}.{field_name} is not wire-safe: {field.annotation!r}"
            )
