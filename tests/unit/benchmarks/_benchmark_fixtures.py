"""Shared fixtures for benchmark tests.

Both the harness contract tests and the SWE-bench adapter tests build the
same :class:`~agentic_evalkit.benchmarks.harness.HarnessRequest` shape, so it
lives here once rather than being duplicated per-file.
"""

from agentic_evalkit.benchmarks.harness import HarnessRequest


def _harness_request(sample_id: str = "org__repo-1") -> HarnessRequest:
    return HarnessRequest(
        benchmark="swebench-verified@1",
        sample_id=sample_id,
        prediction={
            "instance_id": sample_id,
            "model_name_or_path": "agentic-evalkit-target",
            "model_patch": "diff --git a/x b/x",
        },
        source={"dataset_revision": "abc"},
        environment={},
        timeout_seconds=60,
        resource_limits={"cpus": 1, "memory_mb": 1024},
    )
