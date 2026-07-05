"""Fixture: writes far more than any reasonable byte bound to *both* standard
error and standard output, to exercise SubprocessTarget's oversized-output
teardown while its concurrent, bounded standard-error drain is also running.

Standard error is flushed first so the drain task has real work in flight when
the oversized standard-output line trips the byte bound and teardown begins.
"""

import sys

for _line in sys.stdin:
    sys.stderr.write("e" * (1024 * 1024))
    sys.stderr.flush()
    # 1 MiB of padding on a single standard-output line -- large enough to
    # exceed a small test-configured byte bound and force the oversized-output
    # teardown path while the stderr drain above is concurrently active.
    print("x" * (1024 * 1024), flush=True)
