"""Aggregate workflow events into stable per-kind summaries."""

from collections.abc import Iterable, Mapping
from typing import Any


def summarize(events: Iterable[Mapping[str, Any]]) -> list[dict[str, int | str]]:
    """Summarize events in first-seen kind order.

    Missing required keys retain normal mapping ``KeyError`` behavior.
    """
    summaries: dict[str, dict[str, int | str]] = {}
    order: list[str] = []
    for event in events:
        kind = str(event["kind"])
        if kind not in summaries:
            order.append(kind)
            summaries[kind] = {
                "kind": kind,
                "count": 0,
                "failures": 0,
                "duration_ms": 0,
            }
        current = summaries[kind]
        if event["ok"]:
            current["count"] = int(current["count"]) + 1
        else:
            current["count"] = int(current["count"]) + 1
            current["failures"] = int(current["failures"]) + 1
        duration = int(event["duration_ms"])
        if duration > 0:
            current["duration_ms"] = int(current["duration_ms"]) + duration
        else:
            current["duration_ms"] = int(current["duration_ms"])
    return [summaries[kind] for kind in order]
