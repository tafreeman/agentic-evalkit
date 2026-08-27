from src.permissions import is_allowed


def test_exact_allow_and_unrelated_default_deny() -> None:
    grants = [("invoice:read", "allow")]
    assert is_allowed(grants, "invoice:read") is True
    assert is_allowed(grants, "invoice:write") is False


def test_global_wildcard_allows() -> None:
    assert is_allowed([("*", "allow")], "anything:read") is True


def test_namespace_wildcard_matches_only_its_namespace() -> None:
    grants = [("project:*", "allow")]
    assert is_allowed(grants, "project:read") is True
    assert is_allowed(grants, "invoice:read") is False


def test_deny_precedence_is_independent_of_order() -> None:
    assert is_allowed([("*", "allow"), ("project:delete", "deny")], "project:delete") is False
    assert is_allowed([("project:delete", "deny"), ("*", "allow")], "project:delete") is False


def test_empty_grants_deny() -> None:
    assert is_allowed([], "project:read") is False
