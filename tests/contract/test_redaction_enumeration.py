"""ATDD red-phase scaffolds for Story 2.2 -- reporter/event redaction
enumeration contract (R-002 P0).

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 2, Story 2.2) and
the TEA test design (R-002).

Redaction is not STRUCTURALLY enforced today: reporters are plain module
exports (no registry) and events are a ``RunEvent`` union -- nothing fails CI
when a NEW reporter format or event type is added that bypasses
``apply_redaction`` at the write boundary. This contract requires a
discoverable registry/reflection seam so "add a persisted format without
redaction routing" fails the build.

Skip-marked (TDD red phase). Implementation notes for the dev:
  * Expose a canonical ``REPORTER_FORMATS`` registry (name -> Reporter type)
    that the canonical write boundary iterates, so every reporter is
    enumerable and routed through ``apply_redaction``.
  * Expose an enumerable ``ALL_EVENT_TYPES`` (e.g. derived from the
    ``RunEvent`` union) and keep every event field wire-safe (no raw
    payloads) -- see ``events.py`` module docstring.
"""

from __future__ import annotations

import typing
from datetime import datetime
from enum import Enum

_WIRE_SAFE_SCALARS = (str, int, float, bool, type(None))


def _is_wire_safe(annotation: object) -> bool:
    """True if an event field is an identifier/count/enum/timestamp, an
    optional/union of such, or a ``Literal`` of wire-safe scalar values (e.g.
    the ``schema_version: Literal["1"]`` discriminator every ``FrozenModel``
    inherits) -- never a dict, list, or raw output payload.
    """
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return all(isinstance(arg, _WIRE_SAFE_SCALARS) for arg in typing.get_args(annotation))
    if origin is not None:  # Optional[...], X | Y, or any other parameterized generic
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return bool(args) and all(_is_wire_safe(arg) for arg in args)
    return (
        annotation in _WIRE_SAFE_SCALARS
        or annotation is datetime
        or (isinstance(annotation, type) and issubclass(annotation, Enum))
    )


def test_reporter_registry_lists_all_known_formats() -> None:
    # A canonical registry must enumerate every built-in reporter, so the
    # write boundary can route them all through redaction and a newly added
    # reporter that is not registered fails this contract.
    from agentic_evalkit import reporters

    registry = reporters.REPORTER_FORMATS  # seam to add: dict[str, type[Reporter]]
    assert set(registry.values()) == {
        reporters.JsonReporter,
        reporters.JsonlReporter,
        reporters.MarkdownReporter,
        reporters.HtmlReporter,
    }


def test_every_registered_reporter_is_redaction_routed() -> None:
    # Structural guard: every reporter reachable from the registry must be
    # produced only through the canonical, redaction-applying write boundary.
    # The dev binds this to the real boundary (cli/runs.py:write_canonical_report);
    # here we assert the seam that ties reporters to redaction exists.
    from agentic_evalkit import reporters

    routed = reporters.REDACTION_ROUTED_FORMATS  # seam: names the boundary redacts
    assert set(reporters.REPORTER_FORMATS) == set(routed)


def test_every_event_type_is_enumerated_and_wire_safe() -> None:
    # Every event the runner may emit must be enumerable and carry only
    # wire-safe fields (identifier/enum/count/timestamp) -- never a dict, list,
    # or raw output payload. A new event type cannot escape this contract.
    from agentic_evalkit import events

    all_event_types = events.ALL_EVENT_TYPES  # seam to add: tuple[type[FrozenModel], ...]
    assert all_event_types, "expected a non-empty ALL_EVENT_TYPES registry"

    for event_type in all_event_types:
        for field_name, field in event_type.model_fields.items():
            assert _is_wire_safe(field.annotation), (
                f"{event_type.__name__}.{field_name} is not wire-safe: {field.annotation!r}"
            )
