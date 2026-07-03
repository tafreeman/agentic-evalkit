"""Benchmark adapter protocol (design §7).

A dataset is not automatically a benchmark. A ``BenchmarkAdapter`` binds
source-record projection, prompting/task policy, environment requirements,
execution artifact format, oracle validation, and benchmark-specific
aggregation metadata to a dataset. Adapters project records; they never
execute a target and never perform authoritative verification themselves —
that is the ``HarnessExecutor`` boundary (see ``benchmarks.harness`` and
ADR-0005).
"""

from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from agentic_evalkit.models import EvalSample, SourceRecord


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """The adapter boundary (design §7).

    Implementations are structural (``Protocol``), so adapters do not need
    to inherit any framework base class. ``api_version`` and ``name`` are
    plain attributes so callers can inspect adapter identity without
    instantiating or calling into the adapter.
    """

    api_version: str
    name: str

    def prepare(self, record: SourceRecord) -> EvalSample:
        """Project one provider-native source record into a typed sample."""
        ...

    def validate_oracle(self, sample: EvalSample) -> bool:
        """Validate that a prepared sample carries a usable grading oracle.

        This checks row/sample completeness (e.g. required fields present),
        not correctness of any particular execution attempt.
        """
        ...

    def aggregate_metadata(self) -> dict[str, JsonValue]:
        """Return benchmark-specific metadata to attach to run aggregation.

        Records the upstream benchmark version and compatibility policy
        (design §7) so reports can distinguish adapter versions.
        """
        ...
