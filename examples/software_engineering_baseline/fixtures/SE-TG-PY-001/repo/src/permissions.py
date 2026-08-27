"""Permission evaluator.

Contract:
- A grant is a ``(pattern, effect)`` pair where effect is ``allow`` or ``deny``.
- Patterns may be exact actions, ``*``, or namespace wildcards such as
  ``project:*``.
- Any matching deny overrides every matching allow.
- Access is denied when no pattern matches.
"""

from collections.abc import Iterable


Grant = tuple[str, str]


def is_allowed(grants: Iterable[Grant], action: str) -> bool:
    """Return whether *action* is allowed by *grants*."""
    for pattern, effect in grants:
        if pattern == "*" or pattern == action:
            return effect == "allow"
    return False
