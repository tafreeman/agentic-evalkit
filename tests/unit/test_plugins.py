from importlib.metadata import EntryPoint
from types import MappingProxyType

import pytest

from agentic_evalkit.errors import PluginCompatibilityError
from agentic_evalkit.plugins import load_plugins


def test_rejects_plugin_with_wrong_api_version(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = EntryPoint(
        name="bad",
        value="tests.fixtures.bad_plugin:plugin",
        group="agentic_evalkit.providers.v1",
    )
    monkeypatch.setattr("agentic_evalkit.plugins._entry_points", lambda group: (entry,))
    with pytest.raises(PluginCompatibilityError, match="api_version=2"):
        load_plugins("agentic_evalkit.providers.v1", expected_api_version="1")


def test_successful_load_returns_sorted_immutable_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_b = EntryPoint(
        name="b_provider",
        value="tests.fixtures.good_plugin:plugin",
        group="agentic_evalkit.providers.v1",
    )
    entry_a = EntryPoint(
        name="a_provider",
        value="tests.fixtures.good_plugin:plugin",
        group="agentic_evalkit.providers.v1",
    )
    # Both entries declare a compatible api_version; confirm output ordering
    # is by entry-point name regardless of input order.
    monkeypatch.setattr(
        "agentic_evalkit.plugins._entry_points",
        lambda group: (entry_b, entry_a),
    )

    plugins = load_plugins("agentic_evalkit.providers.v1", expected_api_version="1")

    assert isinstance(plugins, MappingProxyType)
    assert list(plugins.keys()) == ["a_provider", "b_provider"]
    with pytest.raises(TypeError):
        plugins["c_provider"] = object()  # type: ignore[index]


def test_duplicate_plugin_name_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_one = EntryPoint(
        name="dup",
        value="tests.fixtures.good_plugin:plugin",
        group="agentic_evalkit.providers.v1",
    )
    entry_two = EntryPoint(
        name="dup",
        value="tests.fixtures.good_plugin:plugin",
        group="agentic_evalkit.providers.v1",
    )
    monkeypatch.setattr(
        "agentic_evalkit.plugins._entry_points",
        lambda group: (entry_one, entry_two),
    )

    with pytest.raises(PluginCompatibilityError, match="dup"):
        load_plugins("agentic_evalkit.providers.v1", expected_api_version="1")


def test_import_failure_is_wrapped_with_entry_point_name_and_exception_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = EntryPoint(
        name="broken",
        value="tests.fixtures.does_not_exist:plugin",
        group="agentic_evalkit.providers.v1",
    )
    monkeypatch.setattr("agentic_evalkit.plugins._entry_points", lambda group: (entry,))

    with pytest.raises(PluginCompatibilityError, match="broken") as excinfo:
        load_plugins("agentic_evalkit.providers.v1", expected_api_version="1")

    message = str(excinfo.value)
    assert "broken" in message
    assert "ModuleNotFoundError" in message
