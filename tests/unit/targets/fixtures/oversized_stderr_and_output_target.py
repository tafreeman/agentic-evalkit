"""Fixture: writes far more than any reasonable byte bound to *both* standard
error and standard output, interleaving the two so both oversized paths are
genuinely exercised regardless of pipe-buffer timing.

The two streams are written in alternating, individually-flushed chunks. If a
large stderr burst were written up front with a small ``max_stderr_bytes``, the
child could block on stderr backpressure before the standard-output line under
test was ever produced -- the oversized-output teardown path would then never
run. Interleaving keeps both streams making progress so the oversized-output
line is reached (and trips the byte bound) whatever the OS pipe buffer sizes.

The standard-output chunks carry no newline, so ``StreamReader.readline()``
accumulates them into one ever-growing line that trips the output byte bound;
the process is then killed by the target's teardown before any terminator is
written. Kept a plain stdlib script consistent with the sibling fixtures.
"""

import sys

_CHUNK = "x" * 4096
_STDERR_CHUNK = "e" * 4096
# 64 chunks * 4096 bytes = 256 KiB per stream -- far past any small
# test-configured byte bound, but fast to generate.
_ITERATIONS = 64

for _line in sys.stdin:
    for _ in range(_ITERATIONS):
        # A flushed stderr burst so the concurrent, bounded stderr drain has
        # real work in flight...
        sys.stderr.write(_STDERR_CHUNK)
        sys.stderr.flush()
        # ...interleaved with an unterminated standard-output chunk that
        # accumulates into one oversized line (no newline until never).
        sys.stdout.write(_CHUNK)
        sys.stdout.flush()
    break
