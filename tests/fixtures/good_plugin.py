"""A fixture plugin object that declares a compatible API version.

Used by tests/unit/test_plugins.py to prove load_plugins() accepts a plugin
whose declared api_version matches the expected group version, and to test
sorting, immutability, and duplicate-name rejection.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _GoodPlugin:
    api_version: str = "1"


plugin = _GoodPlugin()
