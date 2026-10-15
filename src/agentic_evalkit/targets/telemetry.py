"""Build a target ``telemetry`` object from portable per-call records.

A target that runs an instrumented system in-process (for example a service
whose LLM calls flow through ExecutionKit's ``TraceCollector``) ends each
sample with a handful of per-call records. This module flattens those records
into the small ``telemetry`` object the target protocols carry — the same
field set :class:`~agentic_evalkit.targets.subprocess.SubprocessTarget`
validates on the wire — so every instrumented target speaks one envelope.

The records are *plain JSON-safe mappings*, not any producer's classes: this
module deliberately never imports ExecutionKit (or any other instrumented
library), so the mapping stays a documentation-level contract that any target
can honour. A record built by ``TraceCollector.to_jsonable()`` satisfies it
verbatim.

Two honesty rules, mirroring the study contract:

* **Missing stays missing.** A token total is only emitted when *every* call
  reported that token kind; a partial total would silently undercount.
  Coverage counts always ride along under ``environment_metadata`` so a
  report can see how much of the work any total actually covers.
* **Malformed fails.** A record that violates the portable shape raises
  ``ValueError`` rather than being silently dropped — an uncounted call would
  make the envelope misleading in exactly the way the coverage counts exist
  to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["COVERAGE_KEY", "envelope_from_records"]

#: Key under which per-envelope coverage statistics are namespaced inside
#: ``environment_metadata`` (a caller-supplied key of the same name is
#: overwritten — coverage is authoritative here).
COVERAGE_KEY = "call_telemetry"


@dataclass(frozen=True, slots=True)
class _CallFacts:
    """Validated, fully-typed view of one portable per-call record."""

    call_id: str
    attempt_ids: tuple[str, ...]
    input_tokens: int | None
    output_tokens: int | None
    duration_ms: float | None
    model: str | None
    tool_call_names: tuple[str, ...]


def _optional_int(record: Mapping[str, Any], field: str) -> int | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"record {field} must be a nonnegative integer or None")
    # ``value`` stays ``Any`` after the disjunctive guard; ``int()`` is a
    # no-op at runtime and gives mypy the narrowed type.
    return int(value)


def _optional_duration(record: Mapping[str, Any]) -> float | None:
    value = record.get("duration_ms")
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("record duration_ms must be a finite nonnegative number or None")
    return float(value)


def _optional_model(record: Mapping[str, Any], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"record {field} must be a nonempty string or None")
    return value


def _call_id(record: Mapping[str, Any]) -> str:
    value = record.get("call_id")
    if not isinstance(value, str) or not value:
        raise ValueError("record call_id must be a nonempty string")
    return str(value)


def _str_tuple(record: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = record.get(field, ())
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"record {field} must be a list of nonempty strings")
    return tuple(value)


def _parse(record: Mapping[str, Any]) -> _CallFacts:
    # The served model is the fact; the requested one is the fallback
    # attribution when a provider does not report what it served.
    model = _optional_model(record, "served_model") or _optional_model(record, "requested_model")
    return _CallFacts(
        call_id=_call_id(record),
        attempt_ids=_str_tuple(record, "attempt_ids"),
        input_tokens=_optional_int(record, "input_tokens"),
        output_tokens=_optional_int(record, "output_tokens"),
        duration_ms=_optional_duration(record),
        model=model,
        tool_call_names=_str_tuple(record, "tool_call_names"),
    )


def envelope_from_records(
    records: list[Mapping[str, Any]],
    *,
    environment_metadata: Mapping[str, Any] | None = None,
    input_rate: float | None = None,
    output_rate: float | None = None,
) -> dict[str, Any]:
    """Flatten portable per-call records into one telemetry envelope.

    Args:
        records: Portable per-call records (JSON-safe mappings carrying
            ``call_id`` plus optional ``attempt_ids``, ``input_tokens`` /
            ``output_tokens``, ``duration_ms``, ``requested_model`` /
            ``served_model``, and ``tool_call_names``).
        environment_metadata: Extra JSON-safe metadata merged into the
            envelope (for example the target's build or deployment label).
        input_rate: Optional cost per input token.
        output_rate: Optional cost per output token.

    Returns:
        A dict using exactly the field set the target protocols carry
        (``input_tokens``, ``output_tokens``, ``cost_usd``, ``latency_ms``,
        ``model_name``, ``trace_refs``, ``tool_calls``,
        ``environment_metadata``). Fields with no honest value are omitted
        rather than zero-filled; ``environment_metadata`` is always present
        and carries per-envelope coverage counts under
        :data:`COVERAGE_KEY`.

    Raises:
        ValueError: If a record is malformed, or if exactly one token rate
            is supplied.
    """
    if (input_rate is None) != (output_rate is None):
        raise ValueError("input_rate and output_rate must be supplied together or not at all")

    calls = [_parse(record) for record in records]
    total = len(calls)
    with_input = [c.input_tokens for c in calls if c.input_tokens is not None]
    with_output = [c.output_tokens for c in calls if c.output_tokens is not None]
    durations = [c.duration_ms for c in calls if c.duration_ms is not None]

    envelope: dict[str, Any] = {}
    # Token totals only when EVERY call reported — a partial sum would read
    # as the whole sample's consumption while silently omitting unmeasured
    # calls.
    if total and len(with_input) == total:
        envelope["input_tokens"] = sum(with_input)
    if total and len(with_output) == total:
        envelope["output_tokens"] = sum(with_output)
    if durations:
        envelope["latency_ms"] = sum(durations)
    if (
        input_rate is not None
        and output_rate is not None
        and total
        and len(with_input) == total
        and len(with_output) == total
    ):
        envelope["cost_usd"] = sum(with_input) * input_rate + sum(with_output) * output_rate

    # Distinct models in first-seen order. A single string only when one
    # model served the whole sample; a fallback chain that crossed models is
    # reported under coverage instead of being flattened into a lie.
    models = list(dict.fromkeys(c.model for c in calls if c.model is not None))
    if len(models) == 1:
        envelope["model_name"] = models[0]

    trace_refs = [f"call:{c.call_id}" for c in calls]
    trace_refs.extend(f"attempt:{attempt_id}" for c in calls for attempt_id in c.attempt_ids)
    if trace_refs:
        envelope["trace_refs"] = trace_refs

    tool_calls = [{"name": name} for c in calls for name in c.tool_call_names]
    if tool_calls:
        envelope["tool_calls"] = tool_calls

    metadata = dict(environment_metadata or {})
    metadata[COVERAGE_KEY] = {
        "records": total,
        "with_input_tokens": len(with_input),
        "with_output_tokens": len(with_output),
        "with_duration_ms": len(durations),
        "models": models,
        "unattributed": sum(1 for c in calls if c.model is None),
    }
    envelope["environment_metadata"] = metadata
    return envelope
