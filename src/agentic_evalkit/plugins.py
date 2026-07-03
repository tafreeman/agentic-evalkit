"""Deterministic Python entry-point plugin discovery.

Extension points (dataset providers, benchmark adapters, graders, reporters,
harness executors) register through versioned entry-point groups such as
``agentic_evalkit.providers.v1`` (see ADR-0003 and ADR-0009). This module
implements the single discovery routine used by every extension point:
:func:`load_plugins`.

Discovery is deterministic (entry points are sorted by name before loading)
and never silently swallows a failure: a plugin that fails to import, that
declares an incompatible ``api_version``, or that collides with an
already-loaded plugin name raises :class:`PluginCompatibilityError`.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint, entry_points
from types import MappingProxyType
from typing import Any

from agentic_evalkit.errors import PluginCompatibilityError

__all__ = ["load_plugins"]


def _entry_points(group: str) -> tuple[EntryPoint, ...]:
    """Return the entry points registered for ``group``.

    Isolated as a module-level indirection so tests can monkeypatch
    ``agentic_evalkit.plugins._entry_points`` instead of touching the real
    installed-package entry-point registry.
    """
    return tuple(entry_points(group=group))


def load_plugins(group: str, expected_api_version: str) -> MappingProxyType[str, Any]:
    """Load and validate every plugin registered under ``group``.

    Args:
        group: The entry-point group name, e.g. ``"agentic_evalkit.providers.v1"``.
        expected_api_version: The plugin API version every loaded object must
            declare via its own ``api_version`` attribute.

    Returns:
        An immutable mapping of entry-point name to loaded plugin object,
        sorted by name.

    Raises:
        PluginCompatibilityError: If an entry point fails to load, declares
            an ``api_version`` other than ``expected_api_version``, or if two
            entry points in the group share the same name.
    """
    loaded: dict[str, Any] = {}
    for entry in sorted(_entry_points(group), key=lambda item: item.name):
        if entry.name in loaded:
            raise PluginCompatibilityError(
                message=(
                    f"duplicate plugin name '{entry.name}' registered more than "
                    f"once in entry-point group '{group}'"
                ),
                context={"entry_point": entry.name, "group": group},
            )

        try:
            plugin = entry.load()
        except Exception as error:
            raise PluginCompatibilityError(
                message=(
                    f"plugin '{entry.name}' in group '{group}' failed to load: "
                    f"{type(error).__name__}: {error}"
                ),
                context={
                    "entry_point": entry.name,
                    "group": group,
                    "exception_type": type(error).__name__,
                },
            ) from error

        actual_api_version = getattr(plugin, "api_version", None)
        if actual_api_version != expected_api_version:
            raise PluginCompatibilityError(
                message=(
                    f"plugin '{entry.name}' in group '{group}' declares "
                    f"api_version={actual_api_version} but expected "
                    f"api_version={expected_api_version}"
                ),
                context={
                    "entry_point": entry.name,
                    "group": group,
                    "expected_api_version": expected_api_version,
                    "actual_api_version": actual_api_version,
                },
            )

        loaded[entry.name] = plugin

    return MappingProxyType(dict(sorted(loaded.items())))
