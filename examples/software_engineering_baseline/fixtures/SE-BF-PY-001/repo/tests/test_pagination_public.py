import pytest

from src.pagination import paginate


@pytest.mark.parametrize(("page", "page_size"), [(0, 2), (1, 0), (-1, 3)])
def test_paginate_rejects_non_positive_arguments(page: int, page_size: int) -> None:
    with pytest.raises(ValueError):
        paginate([1, 2, 3], page, page_size)


def test_paginate_beyond_end_is_empty() -> None:
    assert paginate([1, 2], 10, 3) == []
