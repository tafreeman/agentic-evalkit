"""Fixture: never responds, to exercise SubprocessTarget's timeout ->
kill-and-await path.
"""

import sys
import time

for _line in sys.stdin:
    time.sleep(60)
    sys.exit(0)
