"""Fixture: emits a CRLF-terminated JSON line whose bytes are split across
several separate writes/flushes, to prove SubprocessTarget reassembles
complete lines via StreamReader.readline() rather than assuming one write
equals one line. Must parse identically on Windows and Linux.
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
    # Write the body in fragments, then a separate CRLF terminator write, so
    # the transport-level bytes never coincide with the JSON line boundary.
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
