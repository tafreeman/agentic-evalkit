from src.pagination import paginate


def test_full_first_page_keeps_last_boundary_item() -> None:
    assert paginate([1, 2, 3, 4], 1, 2) == [1, 2]


def test_second_page_has_no_gap_or_duplicate() -> None:
    assert paginate([1, 2, 3, 4], 2, 2) == [3, 4]


def test_partial_final_page_is_returned() -> None:
    assert paginate([1, 2, 3, 4, 5], 3, 2) == [5]


def test_page_size_one_returns_one_item() -> None:
    assert paginate(["a", "b"], 2, 1) == ["b"]
