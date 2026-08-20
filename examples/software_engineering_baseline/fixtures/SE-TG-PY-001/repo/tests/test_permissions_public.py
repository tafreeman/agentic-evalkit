from src.permissions import is_allowed


def test_exact_allow() -> None:
    assert is_allowed([("invoice:read", "allow")], "invoice:read") is True


def test_unrelated_action_is_denied() -> None:
    assert is_allowed([("invoice:read", "allow")], "invoice:write") is False
