"""Fixture script: writes 1 MiB to standard error (far more than any byte
limit these tests configure) but still sends back a normal, valid response
on standard output. Used to check that SubprocessTarget reads and caps an
oversized stderr stream in the background without it interfering with a
perfectly good response arriving on stdout at the same time.
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
