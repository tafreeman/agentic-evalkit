from collections.abc import Iterable, Mapping
from typing import Any


def _new_summary(kind: str) -> dict[str, int | str]:
    return {"kind": kind, "count": 0, "failures": 0, "duration_ms": 0}


def summarize(events: Iterable[Mapping[str, Any]]) -> list[dict[str, int | str]]:
    summaries: dict[str, dict[str, int | str]] = {}
    for event in events:
        kind = str(event["kind"])
        current = summaries.setdefault(kind, _new_summary(kind))
        current["count"] = int(current["count"]) + 1
        current["failures"] = int(current["failures"]) + int(not event["ok"])
        current["duration_ms"] = int(current["duration_ms"]) + max(0, int(event["duration_ms"]))
    return list(summaries.values())
