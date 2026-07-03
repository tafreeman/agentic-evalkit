"""A packaged transport/evaluation smoke target that always returns ``"0"``.

``zero_target`` is deliberately *not* a benchmark baseline: it never reasons
about ``sample`` input, so its grade on any real benchmark (GSM8K included)
carries no information about system quality. Its only purpose is to prove
the CLI pipeline -- manifest loading, dataset resolution, target invocation,
grading, and canonical JSON reporting -- works end to end without requiring
a developer to have a real system under test wired up yet (plan Task 14,
Step 7). ``agentic-evalkit init --preset gsm8k`` wires this callable in only
when the developer supplies no callable/subprocess/HTTP target of their own.
"""

from pydantic import JsonValue

__all__ = ["zero_target"]


def zero_target(sample_input: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Ignore ``sample_input`` and always answer ``"0"``.

    Args:
        sample_input: The ``EvalSample.input`` mapping for one attempt.
            Unused -- this target is a fixed-output smoke test, not a
            reasoning system.

    Returns:
        A mapping with a single ``"answer"`` key whose value is always the
        string ``"0"``, matching the shape :class:`CallableTarget` expects
        (``NormalizedExecutionResult.output``).
    """
    del sample_input  # Unused: this target's output never depends on input.
    return {"answer": "0"}
