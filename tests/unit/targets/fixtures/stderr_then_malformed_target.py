"""Fixture script: writes a marker message to standard error, then sends
back a broken (not valid) JSON response on standard output. Used to check
that when SubprocessTarget reports an error, it includes the captured
stderr text as part of the error evidence -- so someone debugging a failed
run gets a real clue about what went wrong, not just a bare "invalid JSON"
message.
"""

import sys

for _line in sys.stdin:
    sys.stderr.write("diagnostic-marker-from-stderr\n")
    sys.stderr.flush()
    print("not valid json {", flush=True)
