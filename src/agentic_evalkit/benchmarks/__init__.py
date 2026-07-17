"""Benchmark adapters (turn raw dataset rows into gradable tasks) and the
harness boundary (the piece that hands results off to real, authoritative
verification tools)."""

from agentic_evalkit.benchmarks.base import BenchmarkAdapter
from agentic_evalkit.benchmarks.grounding import GroundedCitationAdapter
from agentic_evalkit.benchmarks.gsm8k import Gsm8kAdapter, extract_final_answer
from agentic_evalkit.benchmarks.harness import (
    FakeHarnessExecutor,
    HarnessExecutor,
    HarnessRequest,
    HarnessResult,
    HarnessStatus,
    UnavailableHarnessExecutor,
)
from agentic_evalkit.benchmarks.swebench import SweBenchVerifiedAdapter
from agentic_evalkit.benchmarks.swebench_docker import (
    SweBenchDockerHarnessExecutor,
    swebench_prediction,
)

__all__ = [
    "BenchmarkAdapter",
    "FakeHarnessExecutor",
    "GroundedCitationAdapter",
    "Gsm8kAdapter",
    "HarnessExecutor",
    "HarnessRequest",
    "HarnessResult",
    "HarnessStatus",
    "SweBenchDockerHarnessExecutor",
    "SweBenchVerifiedAdapter",
    "UnavailableHarnessExecutor",
    "extract_final_answer",
    "swebench_prediction",
]
