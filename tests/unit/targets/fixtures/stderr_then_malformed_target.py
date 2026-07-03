"""Fixture: writes a diagnostic message to standard error, then emits
malformed JSON on standard output, to exercise SubprocessTarget surfacing
captured stderr content as part of the error evidence.
"""

import sys

for _line in sys.stdin:
    sys.stderr.write("diagnostic-marker-from-stderr\n")
    sys.stderr.flush()
    print("not valid json {", flush=True)
