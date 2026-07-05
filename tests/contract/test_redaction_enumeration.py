"""Reporter/event redaction-enumeration contract (Story 2.2, R-002 P0).

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 2, Story 2.2) and
the TEA test design (R-002). Green since the P0 branch landed the seams; the
2026-07-04 code review then made them falsifiable.

What this contract pins:
  * ``REPORTER_FORMATS`` is the single canonical registry both CLI write
    boundaries select reporters from, and it lists every built-in reporter.
  * ``REDACTION_ROUTED_FORMATS`` is maintained BY HAND at the reporters
    package (deliberately not derived from the registry), so the equality
    check below is a real tripwire: registering a new format without
    consciously pairing it with redaction routing fails CI.
  * ``ALL_EVENT_TYPES`` equals the ``RunEvent`` union exactly, so an event
    type added to one but not the other fails CI and no event can escape the
    wire-safety check.
"""

from __future__ import annotations

import types
import typing
from datetime import datetime
from enum import Enum

_WIRE_SAFE_SCALARS = (str, int, float, bool, type(None))


def _is_wire_safe(annotation: object) -> bool:
    """True if an event field is an identifier/count/enum/timestamp, an
    optional/union of such, or a ``Literal`` of wire-safe scalar values (e.g.
    the ``schema_version: Literal["1"]`` discriminator every ``FrozenModel``
    inherits) -- never a dict, list, or any other container payload.
    """
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return all(isinstance(arg, _WIRE_SAFE_SCALARS) for arg in typing.get_args(annotation))
    if origin is typing.Union or origin is types.UnionType:
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return bool(args) and all(_is_wire_safe(arg) for arg in args)
    if origin is not None:
        # Any other parameterized generic (list[...], dict[...], tuple[...],
        # Sequence[...], ...) is a container -- exactly what an event must
        # never carry. This branch is what keeps a raw-payload field from
        # slipping through as "some parameterized thing that looked unionish".
        return False
    return (
        annotation in _WIRE_SAFE_SCALARS
        or annotation is datetime
        or (isinstance(annotation, type) and issubclass(annotation, Enum))
    )


def test_wire_safety_rejects_containers() -> None:
    # Meta-guard for the guard: the 2026-07-04 review found the original
    # union branch approved ANY parameterized generic, so list[str] and
    # dict[str, str] passed the very check that exists to forbid them.
    assert not _is_wire_safe(list[str])
    assert not _is_wire_safe(dict[str, str])
    assert not _is_wire_safe(tuple[str, ...])
    assert not _is_wire_safe(dict[str, str] | None)
    assert _is_wire_safe(str | None)
    assert _is_wire_safe(typing.Literal["1"])
    assert _is_wire_safe(datetime)


def test_reporter_registry_lists_all_known_formats() -> None:
    # The canonical registry must enumerate every built-in reporter, so the
    # write boundaries can only reach registered (enumerable) formats and a
    # newly added reporter that is not registered fails this contract.
    from agentic_evalkit import reporters

    registry = reporters.REPORTER_FORMATS
    assert set(registry.values()) == {
        reporters.JsonReporter,
        reporters.JsonlReporter,
        reporters.MarkdownReporter,
        reporters.HtmlReporter,
    }


def test_every_registered_reporter_is_redaction_routed() -> None:
    # REDACTION_ROUTED_FORMATS is hand-maintained next to the reporters
    # package while REPORTER_FORMATS is the selectable registry; because the
    # routed set is NOT derived from the registry, this equality is a real
    # tripwire -- registering a fifth format forces a conscious, reviewed
    # update pairing it with apply_redaction at its write boundary.
    from agentic_evalkit import reporters

    routed = reporters.REDACTION_ROUTED_FORMATS
    assert set(reporters.REPORTER_FORMATS) == set(routed)


def test_report_command_reporters_derive_from_the_registry() -> None:
    # The second write boundary (the ``report`` CLI command) must be driven
    # by the same canonical registry -- the review found it kept a private,
    # drift-prone reporter table. "json" is exempt: the command regenerates
    # FROM canonical run JSON.
    from agentic_evalkit import reporters
    from agentic_evalkit.cli.reports import _REPORTERS

    assert set(_REPORTERS) == set(reporters.REPORTER_FORMATS) - {"json"}
    for name, reporter in _REPORTERS.items():
        assert type(reporter) is reporters.REPORTER_FORMATS[name]


def test_all_event_types_matches_the_run_event_union() -> None:
    # The binding that makes every event contract falsifiable: the enumerable
    # tuple and the RunEvent union must list exactly the same types, so a new
    # event added to one but not the other fails CI instead of silently
    # escaping the wire-safety and factory-coverage guards.
    from agentic_evalkit import events

    assert set(events.ALL_EVENT_TYPES) == set(typing.get_args(events.RunEvent))


def test_every_event_type_is_enumerated_and_wire_safe() -> None:
    # Every event the runner may emit must be enumerable and carry only
    # wire-safe fields (identifier/enum/count/timestamp) -- never a dict,
    # list, or raw output payload.
    from agentic_evalkit import events

    all_event_types = events.ALL_EVENT_TYPES
    assert all_event_types, "expected a non-empty ALL_EVENT_TYPES registry"

    for event_type in all_event_types:
        for field_name, field in event_type.model_fields.items():
            assert _is_wire_safe(field.annotation), (
                f"{event_type.__name__}.{field_name} is not wire-safe: {field.annotation!r}"
            )
