"""A fixture plugin object that declares an incompatible API version.

Used by tests/unit/test_plugins.py to prove load_plugins() rejects a plugin
whose declared api_version does not match the expected group version.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _BadPlugin:
    api_version: str = "2"


plugin = _BadPlugin()
