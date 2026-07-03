"""Fixture: emits a standard-output line far larger than any reasonable byte
bound, to exercise SubprocessTarget's output-size cap.
"""

import sys

for _line in sys.stdin:
    # 1 MiB of padding on a single line -- large enough to exceed a small
    # test-configured byte bound while staying fast to generate.
    print("x" * (1024 * 1024), flush=True)
