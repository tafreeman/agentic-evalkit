"""Fixture script: always answers with a fixed, wrong sample_id
("wrong-sample-id") instead of echoing back the sample_id it was actually
asked about. Used to check that SubprocessTarget notices this mismatch and
reports an error, rather than silently accepting a response that might
belong to a different request than the one that was sent.
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
