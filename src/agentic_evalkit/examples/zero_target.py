"""A built-in stand-in "system under test" that always answers ``"0"``, used
to smoke-test the evaluation pipeline end to end.

``zero_target`` is deliberately *not* something to compare real systems
against: it never looks at the ``sample`` input at all, so no matter which
benchmark you run it against (including GSM8K, a well-known grade-school
math benchmark), its score tells you nothing about whether a real system is
any good. Its only job is to prove that the rest of the pipeline actually
works -- reading the manifest (the config file that describes an eval run),
picking the dataset, invoking the target, grading the output, and writing
the standard JSON report -- without requiring a developer to have a real
system wired up yet (see the project plan, Task 14, Step 7).
``agentic-evalkit init --preset gsm8k`` plugs this function in
automatically, but only when the developer hasn't supplied their own
target (whether that's a plain Python callable, a subprocess, or an HTTP
endpoint).
"""

from pydantic import JsonValue

__all__ = ["zero_target"]


def zero_target(sample_input: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Ignore ``sample_input`` and always answer ``"0"``.

    Args:
        sample_input: The input data for one evaluation attempt (the
            ``input`` field of an ``EvalSample``). This function doesn't
            look at it at all -- it's a fixed-output smoke test standing in
            for a real system, not something that actually reasons about
            the question.

    Returns:
        A dictionary with a single ``"answer"`` key, always set to the
        string ``"0"``. This matches the output format that
        :class:`CallableTarget` expects from any target function (the
        library's standard ``NormalizedExecutionResult.output`` shape), so
        it flows through the rest of the pipeline exactly like a real
        target's output would.
    """
    del sample_input  # Unused: this target's output never depends on input.
    return {"answer": "0"}
