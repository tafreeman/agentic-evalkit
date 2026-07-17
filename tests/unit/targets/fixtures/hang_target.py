"""Fixture script: simulates a target that hangs and never sends back a
response. Used to check that SubprocessTarget's timeout actually fires,
and that it then kills the stuck process and waits for it to fully exit
(see SubprocessTarget._terminate), rather than leaving it running in the
background forever.
"""

import sys
import time

for _line in sys.stdin:
    time.sleep(60)
    sys.exit(0)
