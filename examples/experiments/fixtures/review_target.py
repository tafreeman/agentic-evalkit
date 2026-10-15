"""A separate deterministic subprocess representing a review-loop arm."""

from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.loads(sys.stdin.readline())
    question = str(request["input"]["question"])
    reviewed_answers = {"1 + 1": "2", "2 + 2": "4"}
    response = {
        "schema_version": "1",
        "sample_id": request["sample_id"],
        "output": {"answer": reviewed_answers.get(question, "")},
    }
    sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
