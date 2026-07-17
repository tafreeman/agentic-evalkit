"""Regression guards for Story 2.1 -- making sure spilled artifacts get
redacted (R-002 P0, test-design gap #3).

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 2, Story 2.1) and
the TEA test design (R-002).

Recall from ``runner.py``: when a sample's output is too big to keep
inline, ``EvalRunner._spill_large_output`` writes it out to its own file on
disk (it "spills" the output) and leaves behind just a reference to it.
Before writing those bytes, it redacts them -- blanks out anything that
looks like a secret -- according to the runner's ``redaction_policy``
setting. As of Story 2.1, that setting defaults to
``DEFAULT_REDACTION_POLICY`` rather than "no redaction," so this protection
is on by default: a runner built with no explicit ``redaction_policy``
argument (exactly what ``_default_runner`` below sets up, mirroring how the
shipped CLI builds its own runner) will still catch and blank out
secret-shaped text before it ever reaches disk.

These tests exercise that default path directly (by calling
``_spill_large_output`` the same way a real run does internally), as
regression guards: they prove that a run whose target produces a large
output containing what looks like a credential no longer writes that raw
credential to disk.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.models import ExecutionStatus, NormalizedExecutionResult
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY
from agentic_evalkit.runner import _LARGE_OUTPUT_THRESHOLD_BYTES, EvalRunner

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

_SECRET = "sk-" + "A" * 40
# The three specific credential formats DEFAULT_REDACTION_POLICY is designed
# to catch (design §12): an OpenAI-style secret key (starts with "sk-"), a
# Hugging Face access token (starts with "hf_"), and an HTTP "Authorization:
# Bearer ..." header value. Each is planted here exactly as it would really
# look, so these tests prove the spill path redacts all three formats, not
# just the "sk-" case. Keep these in sync with the actual regex patterns in
# ``agentic_evalkit.reporters.base.DEFAULT_REDACTION_POLICY``.
_SK_SECRET = _SECRET
_HF_SECRET = "hf_" + "B" * 40
_BEARER_SECRET = "Bearer " + "c" * 40
_ALL_SECRETS = (_SK_SECRET, _HF_SECRET, _BEARER_SECRET)


def _default_runner(store: ArtifactStore) -> EvalRunner:
    """Build a runner configured the same way the shipped CLI builds one:
    with no explicit ``redaction_policy`` argument, so it falls back to the
    library's default. The behavior under test here
    (``_spill_large_output``) only ever touches the artifact store and the
    redaction policy, so every other collaborator this runner would
    normally need (catalog, adapters, targets, graders) is left as an empty
    placeholder -- they're irrelevant to what these tests check.
    """
    return EvalRunner(
        catalog=cast("Any", None),
        adapters={},
        targets={},
        graders={},
        artifact_store=store,
    )


def _execution_with_output(output: dict[str, JsonValue]) -> NormalizedExecutionResult:
    now = datetime.now(UTC)
    return NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output=output,
        status=ExecutionStatus.COMPLETED,
        started_at=now,
        finished_at=now,
    )


def _large_secret_execution() -> NormalizedExecutionResult:
    # Repeating the word "padding" 2000 times pushes this well past the
    # 8192-byte spill threshold, even after redaction runs -- the word
    # "padding" doesn't match any secret pattern, so it passes through
    # untouched and still counts fully toward the size. Exactly one secret
    # is planted in the text, so once redaction runs, the resulting bytes
    # should contain exactly one "[REDACTED]" marker.
    return _execution_with_output({"log": f"token={_SECRET} " + ("padding " * 2000)})


def _large_multi_secret_execution() -> NormalizedExecutionResult:
    # Also well over the 8192-byte spill threshold, but this time carrying
    # all three credential formats at once (sk-, hf_, and Bearer), so a test
    # using this fixture can check that the spill path redacts every format
    # in one pass, not just the "sk-" case.
    planted = f"sk={_SK_SECRET} hf={_HF_SECRET} auth={_BEARER_SECRET} "
    return _execution_with_output({"log": planted + ("padding " * 2000)})


def _output_serializing_to_exactly(n_bytes: int) -> dict[str, JsonValue]:
    """Build an output dict whose text form -- ``str(...)``, UTF-8-encoded,
    exactly how the spill logic measures size -- comes out to precisely
    ``n_bytes`` bytes, and contains nothing that looks like a secret.

    The spill logic decides whether to spill by checking
    ``len(str(output).encode("utf-8"))``, so to hit an exact byte count we
    first measure how many bytes the dict's own formatting takes up on its
    own (using an empty string value as a baseline, "overhead"), then pad
    the value with just enough filler characters to reach the target size.
    The filler character is ``"x"``, which is always exactly one byte in
    UTF-8 and doesn't appear in any of the default secret patterns -- so the
    final encoded length works out to exactly overhead + pad, with nothing
    else affecting the count.
    """
    overhead = len(str({"log": ""}).encode("utf-8"))
    pad = n_bytes - overhead
    assert pad >= 0, f"target {n_bytes} is below the {overhead}-byte serialization overhead"
    return {"log": "x" * pad}


def _only_payload_text(root: Path) -> str:
    payloads = list(root.glob("*.bin"))
    assert len(payloads) == 1, f"expected exactly one spilled payload, found {payloads}"
    return payloads[0].read_bytes().decode("utf-8")


def _only_sidecar(root: Path) -> dict[str, Any]:
    sidecars = list(root.glob("*.json"))
    assert len(sidecars) == 1, f"expected exactly one sidecar, found {sidecars}"
    return cast("dict[str, Any]", json.loads(sidecars[0].read_text(encoding="utf-8")))


def test_default_run_path_redacts_spilled_secret_bytes(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    spilled = runner._spill_large_output(_large_multi_secret_execution())
    # Seeing "output_ref" here means the output was moved out to the
    # artifact store and replaced with a reference, instead of staying
    # inline on the result.
    assert "output_ref" in spilled.artifacts

    stored = _only_payload_text(root)
    # None of the three credential formats (sk-, hf_, Bearer) show up in the
    # bytes actually written to disk, proving the spill path redacts all
    # three -- not just the "sk-" case.
    for secret in _ALL_SECRETS:
        assert secret not in stored
    assert "[REDACTED]" in stored


def test_spilled_artifact_sidecar_marks_redacted(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    runner._spill_large_output(_large_secret_execution())

    assert _only_sidecar(root)["redacted"] is True


def test_output_exactly_at_threshold_stays_inline(tmp_path: Path) -> None:
    # runner._spill_large_output only spills an output when its encoded size
    # is STRICTLY GREATER than the threshold (the check for staying inline
    # is ``len(encoded) <= _LARGE_OUTPUT_THRESHOLD_BYTES``), so an output
    # that comes out to exactly the threshold size should still be kept
    # inline. This builds an output of exactly that size and checks: nothing
    # gets written to disk, and the output is still sitting right there on
    # the returned result, unchanged.
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    output = _output_serializing_to_exactly(_LARGE_OUTPUT_THRESHOLD_BYTES)
    assert len(str(output).encode("utf-8")) == _LARGE_OUTPUT_THRESHOLD_BYTES

    result = runner._spill_large_output(_execution_with_output(output))

    # Exactly at the threshold: this must stay inline, not get spilled.
    assert "output_ref" not in result.artifacts
    assert result.output == output
    assert list(root.glob("*.bin")) == []


def test_one_byte_over_threshold_spills(tmp_path: Path) -> None:
    # One byte past the threshold crosses over into "strictly greater than
    # the limit," so it must get spilled. This pins down the other side of
    # the boundary from the exactly-at-threshold test above.
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    output = _output_serializing_to_exactly(_LARGE_OUTPUT_THRESHOLD_BYTES + 1)
    assert len(str(output).encode("utf-8")) == _LARGE_OUTPUT_THRESHOLD_BYTES + 1

    result = runner._spill_large_output(_execution_with_output(output))

    assert "output_ref" in result.artifacts
    assert result.output is None
    assert len(list(root.glob("*.bin"))) == 1


def test_spill_redaction_is_idempotent_at_the_boundary(tmp_path: Path) -> None:
    # This checks that redaction is "idempotent" -- running it a second time
    # over text that's already been redacted doesn't change anything
    # further. That's a genuinely useful thing to check, but not a trivial
    # one: a test that only counts how many "[REDACTED]" markers appear
    # wouldn't actually catch a broken, non-idempotent redaction, because
    # running redaction again over already-redacted text is a no-op no
    # matter what (there's no more raw secret left to find and replace, so
    # of course the marker count stays the same either way). So instead,
    # this test redacts the already-spilled bytes a second time, by hand,
    # and checks the result comes back byte-for-byte identical to what was
    # already on disk -- that's the part a marker count alone couldn't
    # prove. The single-marker assertion just below still holds too, for
    # the one secret planted in this test's own fixture.
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    runner = _default_runner(store)

    runner._spill_large_output(_large_secret_execution())

    persisted = _only_payload_text(root)
    # Exactly one secret was planted in this fixture, so exactly one
    # "[REDACTED]" marker should show up in what's on disk.
    assert persisted.count("[REDACTED]") == 1

    # Now manually redo the same redaction step, the same way the runner
    # does it internally (compile each pattern, substitute "[REDACTED]" for
    # any match) -- but this time running it over bytes that are ALREADY
    # redacted. If redaction is working correctly (idempotent: applying it
    # twice has the same effect as applying it once), the result should be
    # byte-for-byte identical to what was already there, since there's
    # nothing left to redact the second time around.
    reapplied = persisted
    for pattern in DEFAULT_REDACTION_POLICY.secret_patterns:
        reapplied = re.sub(pattern, "[REDACTED]", reapplied)
    assert reapplied == persisted
