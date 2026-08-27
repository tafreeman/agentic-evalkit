from src.reporting import summarize


def test_summarize_empty_input() -> None:
    assert summarize([]) == []


def test_summarize_one_success() -> None:
    assert summarize([{"kind": "build", "ok": True, "duration_ms": 12}]) == [
        {"kind": "build", "count": 1, "failures": 0, "duration_ms": 12}
    ]
