from collections.abc import Iterable


Grant = tuple[str, str]


def is_allowed(grants: Iterable[Grant], action: str) -> bool:
    effects = [effect for pattern, effect in grants if pattern == "*" or pattern == action]
    if "deny" in effects:
        return False
    return "allow" in effects
