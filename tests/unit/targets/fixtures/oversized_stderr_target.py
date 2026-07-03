"""Fixture: writes far more to standard error than any reasonable byte bound
while still returning a valid response on standard output, to exercise
SubprocessTarget's concurrent, bounded standard-error drain.
"""

import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    sys.stderr.write("e" * (1024 * 1024))
    sys.stderr.flush()
    response = {
        "schema_version": "1",
        "sample_id": request["sample_id"],
        "output": request["input"],
        "metadata": {"fixture": "oversized-stderr"},
    }
    print(json.dumps(response, separators=(",", ":")), flush=True)
