"""Fixture script: a well-behaved target that just echoes back whatever
input it was given, unchanged, as its output. Used as the simple "happy
path" baseline in tests -- for example, to check that a valid response
gets parsed correctly, or that the target's fingerprint (an ID for its
exact configuration) stays the same across calls.
"""

import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    response = {
        "schema_version": "1",
        "sample_id": request["sample_id"],
        "output": request["input"],
        "metadata": {"fixture": "echo"},
    }
    print(json.dumps(response, separators=(",", ":")), flush=True)
