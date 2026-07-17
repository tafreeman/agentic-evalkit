"""Benchmark adapter protocol (design §7).

Having a raw dataset of questions and answers isn't the same as having a
runnable benchmark. A ``BenchmarkAdapter`` is the piece of code that bridges
the two: it turns a raw dataset row into this project's standard internal
sample format, decides how the task is phrased as a prompt, declares what
environment the task needs, defines the output format the system under test
must produce, checks that a prepared sample actually has a usable answer key
to grade against, and supplies benchmark-specific details to attach when a
run's results are summarized. Adapters only prepare and check data this way
-- they never run the system under test themselves, and they never perform
the real, authoritative check of whether an answer is correct. That job
belongs to the ``HarnessExecutor`` boundary (see ``benchmarks.harness`` and
ADR-0005, which explains why that separation matters).
"""

from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from agentic_evalkit.models import EvalSample, SourceRecord


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """The adapter boundary (design §7).

    ``Protocol`` here means this is a structural interface: any class that
    happens to have the right attributes and methods counts as a
    ``BenchmarkAdapter``, without needing to explicitly inherit from a
    shared base class (this is Python's "duck typing," made explicit and
    checkable by type checkers). ``api_version`` and ``name`` are plain
    attributes rather than methods, so callers can find out which adapter
    they have and which version it is just by reading an attribute -- no
    need to create an instance or call into the adapter first.
    """

    api_version: str
    name: str

    def prepare(self, record: SourceRecord) -> EvalSample:
        """Turn one raw dataset row (in the original source's own format)
        into this project's typed ``EvalSample`` format."""
        ...

    def validate_oracle(self, sample: EvalSample) -> bool:
        """Check that a prepared sample has everything needed to grade it.

        "Oracle" means the answer key / ground truth this sample's grading
        will be checked against -- for example, confirming that a required
        field is actually present and non-empty. This does not run or judge
        any actual attempt at the task; it only confirms the sample itself
        is complete enough to be gradable at all.
        """
        ...

    def aggregate_metadata(self) -> dict[str, JsonValue]:
        """Return benchmark-specific details to attach to a run's summary.

        Records which version of the upstream benchmark this adapter
        targets, and its compatibility policy (design §7), so that reports
        can tell which adapter version produced a given run's results.
        """
        ...
