"""Fixture script: writes far more data than any test's configured byte
limit to *both* standard error and standard output -- and deliberately
interleaves the two (alternating small writes) rather than writing
everything to one stream and then the other.

Why interleave? If this script wrote its entire oversized stderr burst
first, a small configured ``max_stderr_bytes`` limit could make this
process stall trying to keep writing to stderr (a form of "backpressure":
the operating system pauses a process's write call once the receiving end
isn't reading fast enough) before it ever got around to producing the
oversized standard-output line the test is actually trying to exercise.
That would mean the behavior this fixture exists to test --
SubprocessTarget cleanly tearing down after an oversized-*output* line --
would never actually get exercised. Writing both streams in small,
alternating pieces keeps both sides making steady progress, so the
oversized standard-output line reliably gets produced (and trips
SubprocessTarget's output-size limit) no matter how the operating system
happens to size or time its internal pipe buffers.

The standard-output chunks are written with no line-ending character at
all, so from SubprocessTarget's point of view (using
``StreamReader.readline()``, which waits for a full line) they just keep
accumulating into one single, ever-growing "line" until it trips the
output byte limit. At that point SubprocessTarget kills this process as
part of its normal teardown, so no line-ending is ever actually sent.

Written as a plain script using only the Python standard library, matching
the style of the other fixture files in this directory.
"""

import sys

_CHUNK = "x" * 4096
_STDERR_CHUNK = "e" * 4096
# 64 chunks x 4096 bytes = 256 KiB written to each stream in total -- far
# more than any small byte limit these tests configure, but still quick to
# generate.
_ITERATIONS = 64

for _line in sys.stdin:
    for _ in range(_ITERATIONS):
        # Write (and immediately flush) one stderr chunk, giving
        # SubprocessTarget's concurrent stderr-reading logic real data to
        # work through...
        sys.stderr.write(_STDERR_CHUNK)
        sys.stderr.flush()
        # ...then write one standard-output chunk with no line ending, so
        # it keeps building toward one giant, never-terminated line.
        sys.stdout.write(_CHUNK)
        sys.stdout.flush()
    break
