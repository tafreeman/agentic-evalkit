"""Fixture script: simulates a target that sends its response in several
separate chunks (not all at once), ending the line with Windows-style
"\\r\\n" (CRLF) instead of just "\\n" (LF).

This exists to prove that SubprocessTarget correctly reassembles a full
response line no matter how many separate writes it arrived in, using
`StreamReader.readline()` -- rather than wrongly assuming that "one write
from the subprocess" always means "one complete line." The test using this
fixture must get the same, correctly-parsed result on both Windows and
Linux.
"""

import json
import sys
import time

for line in sys.stdin:
    request = json.loads(line)
    response = {
        "schema_version": "1",
        "sample_id": request["sample_id"],
        "output": request["input"],
        "metadata": {"fixture": "crlf-split"},
    }
    payload = json.dumps(response, separators=(",", ":"))
    # Write the JSON text in two separate pieces, and the line-ending bytes
    # as a third, separate write -- so the data never arrives as one single
    # write that happens to line up neatly with a full line. This is what
    # actually forces SubprocessTarget to reassemble the pieces itself,
    # instead of getting lucky with one complete write.
    buffer = sys.stdout.buffer
    midpoint = len(payload) // 2
    buffer.write(payload[:midpoint].encode("utf-8"))
    buffer.flush()
    time.sleep(0.01)
    buffer.write(payload[midpoint:].encode("utf-8"))
    buffer.flush()
    time.sleep(0.01)
    buffer.write(b"\r\n")
    buffer.flush()
