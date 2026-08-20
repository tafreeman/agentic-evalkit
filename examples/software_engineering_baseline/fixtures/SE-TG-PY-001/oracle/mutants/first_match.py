from collections.abc import Iterable


Grant = tuple[str, str]


def is_allowed(grants: Iterable[Grant], action: str) -> bool:
    for pattern, effect in grants:
        if pattern == "*" or pattern == action or (
            pattern.endswith(":*") and action.startswith(pattern[:-1])
        ):
            return effect == "allow"
    return False
