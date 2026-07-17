"""Fixture script: writes a single line of standard output that is 1 MiB
(much larger than any byte limit these tests configure). Used to check
that SubprocessTarget's output-size cap actually kicks in and reports an
error, instead of accepting an unbounded amount of data.
"""

import sys

for _line in sys.stdin:
    # 1 MiB (1024 * 1024 bytes) of padding, all on one line -- large enough
    # to trip any small byte limit these tests configure, while still being
    # cheap and fast to generate.
    print("x" * (1024 * 1024), flush=True)
