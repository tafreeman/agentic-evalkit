"""Fixture: responds with a sample_id that does not match the request, to
exercise SubprocessTarget's sample-ID-matching validation.
"""

import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    response = {
        "schema_version": "1",
        "sample_id": "wrong-sample-id",
        "output": request["input"],
        "metadata": {},
    }
    print(json.dumps(response, separators=(",", ":")), flush=True)
