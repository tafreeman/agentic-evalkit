"""Shared test data for benchmark tests.

A ``HarnessRequest`` is the message this project sends to an external
harness -- an official tool that actually checks whether a submitted fix
works, rather than just guessing (see
:mod:`agentic_evalkit.benchmarks.harness` for the full explanation). Both
the harness contract tests and the SWE-bench adapter tests need to build one
of these with the same shape, so the helper that builds it lives here once
instead of being copy-pasted into every test file.
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
