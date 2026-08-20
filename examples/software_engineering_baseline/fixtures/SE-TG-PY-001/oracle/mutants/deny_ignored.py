from collections.abc import Iterable


Grant = tuple[str, str]


def is_allowed(grants: Iterable[Grant], action: str) -> bool:
    return any(
        effect == "allow"
        and (pattern == "*" or pattern == action or (pattern.endswith(":*") and action.startswith(pattern[:-1])))
        for pattern, effect in grants
    )
