from collections.abc import Iterable


Grant = tuple[str, str]


def _matches(pattern: str, action: str) -> bool:
    if pattern == "*" or pattern == action:
        return True
    return pattern.endswith(":*") and action.startswith(pattern[:-1])


def is_allowed(grants: Iterable[Grant], action: str) -> bool:
    effects = [effect for pattern, effect in grants if _matches(pattern, action)]
    if "deny" in effects:
        return False
    return "allow" in effects
