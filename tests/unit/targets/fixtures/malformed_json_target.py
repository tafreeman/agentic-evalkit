"""Fixture script: sends back a line of text that is not valid JSON at all.
Used to check that SubprocessTarget correctly reports this as an
ExecutionStatus.ERROR result, instead of crashing while trying to parse it.
"""

import sys

for _line in sys.stdin:
    print("this is not { valid json", flush=True)
