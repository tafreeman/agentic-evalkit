"""Fixture: emits a line that is not valid JSON, to exercise the
ExecutionStatus.ERROR mapping for malformed subprocess responses.
"""

import sys

for _line in sys.stdin:
    print("this is not { valid json", flush=True)
