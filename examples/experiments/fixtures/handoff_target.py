"""Deterministic protocol fixture that consumes an explicit worker hand-off."""

from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.loads(sys.stdin.readline())
    handoff = str(request["input"]["worker_request"])
    answers = {"3 + 4": "7", "5 + 6": "11"}
    response = {
        "schema_version": "1",
        "sample_id": request["sample_id"],
        "output": {"answer": answers.get(handoff, "")},
    }
    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
