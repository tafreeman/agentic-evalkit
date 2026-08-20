"""Small pagination helper with a planted boundary defect."""

from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def paginate(items: Sequence[T], page: int, page_size: int) -> list[T]:
    """Return one one-indexed page from *items*."""
    if page < 1:
        raise ValueError("page must be at least 1")
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    start = (page - 1) * page_size
    stop = start + page_size - 1
    return list(items[start:stop])
