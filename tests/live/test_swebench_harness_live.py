"""Live, Docker-backed test that the real SWE-bench harness actually works (ADR-0014, design §7.1).

This only runs when explicitly requested (``@pytest.mark.live``) -- it's
excluded from the normal, hermetic test suite, and in practice only runs
inside ``.github/workflows/live-swebench.yml``, the one CI workflow that
installs the optional ``agentic-evalkit[swebench]`` extra and has a real
Docker daemon available. This test skips itself only when that capability
is genuinely missing (no Docker, no ``swebench`` package) -- it deliberately
does NOT skip just because some test fixture file happens to be missing, so
that a scheduled or manually-triggered CI run actually proves the harness
works, instead of silently passing without checking anything.

The key thing this file proves (the design §7.1 "fidelity gate"): feeding a
patch that's known to actually fix the bug (the "gold" patch) through the
real ``execute()`` code path yields ``resolved=True``; feeding a patch that
changes nothing relevant through that *same* code path yields
``resolved=False``. The gold patch used here isn't a separately-maintained
fixture file -- it's simply the dataset's own official reference solution
(the ``patch`` field already present on each SWE-bench Verified row), so
nothing extra needs to be committed to this repo just to run this check:

- ``AGENTIC_EVALKIT_SWEBENCH_INSTANCE`` (optional environment variable) --
  picks a specific SWE-bench instance id to test against. When it's not
  set, this just uses the first row of the dataset, so the check always has
  a real instance to run against under Docker.
"""

from __future__ import annotations

import os

import pytest

from agentic_evalkit.benchmarks.harness import HarnessRequest, HarnessStatus
from agentic_evalkit.benchmarks.swebench_docker import (
    SweBenchDockerHarnessExecutor,
    _default_preflight,
)

pytestmark = pytest.mark.live

_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"


def _require_capability() -> None:
    reason = _default_preflight()
    if reason is not None:
        pytest.skip(f"SWE-bench harness capability unavailable: {reason}")


def _gold_instance() -> tuple[str, str]:
    """Look up an ``(instance_id, gold_patch)`` pair from SWE-bench Verified.

    The dataset's own ``patch`` field already IS the correct reference fix
    for that instance, so this check doesn't need its own separately
    committed fixture file (which could be several megabytes). Uses the
    instance named by the ``AGENTIC_EVALKIT_SWEBENCH_INSTANCE`` environment
    variable if it's set, otherwise just uses the dataset's first row.
    """
    try:
        from datasets import load_dataset  # comes from installing the swebench extra
    except ImportError:  # pragma: no cover - live only
        pytest.skip("the 'datasets' package (swebench extra) is required")

    dataset = load_dataset(_DATASET_NAME, split="test")
    wanted = os.environ.get("AGENTIC_EVALKIT_SWEBENCH_INSTANCE")
    if wanted:
        matches = [row for row in dataset if row["instance_id"] == wanted]
        if not matches:
            pytest.skip(f"instance {wanted!r} not found in {_DATASET_NAME}")
        row = matches[0]
    else:
        row = dataset[0]
    return str(row["instance_id"]), str(row["patch"])


def _request(instance_id: str, patch: str) -> HarnessRequest:
    return HarnessRequest(
        benchmark="swebench-verified@1",
        sample_id=f"swebench-verified:{instance_id}",
        prediction={
            "instance_id": instance_id,
            "model_name_or_path": "agentic-evalkit-live-test",
            "model_patch": patch,
        },
        timeout_seconds=1800.0,
    )


@pytest.mark.asyncio
async def test_gold_patch_resolves_through_the_real_execute_path() -> None:
    _require_capability()
    instance_id, gold_patch = _gold_instance()
    result = await SweBenchDockerHarnessExecutor().execute(_request(instance_id, gold_patch))
    assert result.status is HarnessStatus.COMPLETED
    assert result.resolved is True


#: A patch that applies cleanly but can't possibly fix anything -- all it
#: does is create a new, unrelated file. This project needs a patch that is
#: guaranteed NOT to fix the bug, to prove that `resolved=False` genuinely
#: works. The obvious-seeming way to build one would be to take the real
#: gold patch and deliberately corrupt it -- but that does NOT reliably
#: produce a non-fix: the standard `patch` command-line tool skips trailing
#: garbage it can't parse and still applies whatever valid part comes
#: before it. In a real run of this workflow on 2026-07-11, a corrupted
#: gold patch built this way still applied its still-valid portion and DID
#: resolve the issue. Using a patch that applies cleanly but is simply
#: unrelated to the fix avoids that trap: it always lands in the "applied
#: fine, but the tests still fail" branch -- a genuine, deliberate non-fix,
#: not an accident of how `patch` happens to parse broken input.
_NON_FIXING_PATCH = """\
diff --git a/agentic_evalkit_fidelity_probe.txt b/agentic_evalkit_fidelity_probe.txt
new file mode 100644
--- /dev/null
+++ b/agentic_evalkit_fidelity_probe.txt
@@ -0,0 +1 @@
+non-fixing patch: applies cleanly, resolves nothing (design 7.1 negative control)
"""


@pytest.mark.asyncio
async def test_non_fixing_patch_does_not_resolve_through_the_same_path() -> None:
    _require_capability()
    instance_id, _ = _gold_instance()
    result = await SweBenchDockerHarnessExecutor().execute(_request(instance_id, _NON_FIXING_PATCH))
    # This patch applies cleanly but doesn't touch anything relevant, so the
    # tests that were supposed to start passing (FAIL_TO_PASS) still fail.
    # That is a real, deliberate "no" verdict (status COMPLETED, resolved
    # False) -- not an ERROR standing in for one.
    assert result.status is HarnessStatus.COMPLETED
    assert result.resolved is False
