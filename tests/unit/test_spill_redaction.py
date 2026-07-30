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

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from agentic_evalkit.artifacts import ArtifactRef, ArtifactStore
from agentic_evalkit.models import ExecutionStatus, NormalizedExecutionResult
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY, RedactionPolicy
from agentic_evalkit.runner import _LARGE_OUTPUT_THRESHOLD_BYTES, EvalRunner

if TYPE_CHECKING:
    from collections.abc import Callable
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


# --- a store that refuses the bytes degrades one sample, not the whole run ---
#
# ``_spill_large_output`` hands the encoded output to the artifact store, and
# a store is perfectly capable of refusing it: the payload can exceed the
# store's own ``max_bytes``, the disk can be full, the directory can have
# turned read-only. That call used to run unguarded inside the runner's
# ``TaskGroup``, so one such refusal cancelled every in-flight sibling and
# escaped ``EvalRunner.run`` as an ``ExceptionGroup`` -- no report written,
# every already-graded result lost. ``_spill_isolated`` now absorbs it into
# this one sample's result instead. The tests below drive that wrapper
# directly, the same way the tests above drive ``_spill_large_output``.

#: An output comfortably past the 8192-byte spill threshold, so the spill is
#: actually attempted rather than the output being left inline.
_OVERSIZED_OUTPUT_BYTES = _LARGE_OUTPUT_THRESHOLD_BYTES + 1024

#: A store limit sitting *between* the spill threshold and that payload. A
#: real ``ArtifactStore`` configured this way reproduces the failure exactly
#: as production hits it -- the runner decides to spill, and then the store
#: raises ``ArtifactStoreLimitExceeded`` -- with no test double involved.
_REJECTING_STORE_MAX_BYTES = _LARGE_OUTPUT_THRESHOLD_BYTES + 1


class _FailingStore(ArtifactStore):
    """An ``ArtifactStore`` whose ``put_bytes`` always raises what it is told to.

    ``ArtifactStoreLimitExceeded`` is the only failure a genuine store can be
    provoked into raising on demand (hand it a payload past ``max_bytes``),
    and the tests that need *that* use a real store to get it. Disk failures
    (``OSError``) and cancellation are just as real but can't be triggered by
    choosing a payload, so they are injected here instead. This subclasses
    rather than reimplements ``ArtifactStore`` because ``EvalRunner`` takes
    the concrete class, not a protocol -- so there is no structural interface
    to implement, and overriding the single method under test keeps the rest
    of the store's genuine behavior.

    ``error_factory`` builds a fresh exception per call rather than re-raising
    one stored instance, which would accumulate ``__traceback__`` and
    ``__context__`` across calls.
    """

    def __init__(self, root: Path, *, error_factory: Callable[[], BaseException]) -> None:
        super().__init__(root)
        self._error_factory = error_factory

    def put_bytes(self, data: bytes, *, media_type: str, redacted: bool = False) -> ArtifactRef:
        raise self._error_factory()


def _oversized_execution() -> NormalizedExecutionResult:
    return _execution_with_output(_output_serializing_to_exactly(_OVERSIZED_OUTPUT_BYTES))


def _oversized_execution_the_target_already_failed() -> NormalizedExecutionResult:
    """An execution carrying both a large output and the target's own error.

    A target can report a failure of its own and still hand back a big
    payload (a truncated transcript, a partial log). This is the case where
    the spill record must not overwrite a diagnosis that was already there.
    """
    now = datetime.now(UTC)
    return NormalizedExecutionResult(
        sample_id="s1",
        attempt=1,
        output=_output_serializing_to_exactly(_OVERSIZED_OUTPUT_BYTES),
        status=ExecutionStatus.ERROR,
        error={"type": "RuntimeError", "code": "target_failure", "message": "target reported this"},
        started_at=now,
        finished_at=now,
    )


def test_store_rejecting_the_payload_degrades_the_sample_instead_of_raising(
    tmp_path: Path,
) -> None:
    """The core guarantee: a store that refuses the bytes must not raise out
    of the spill. The sample keeps the status it earned -- the target really
    did complete, and the grader already saw this output inline (ADR-0017) --
    while the output itself is dropped and replaced by a typed
    ``output_spill_failed`` record, so a reader can tell "the answer existed
    but could not be stored" from "there was never an answer." Nothing at all
    is left on disk: the store rejected the payload before writing a byte.
    """
    root = tmp_path / "artifacts"
    store = ArtifactStore(root, max_bytes=_REJECTING_STORE_MAX_BYTES)
    runner = _default_runner(store)

    degraded = runner._spill_isolated(_oversized_execution())

    assert degraded.status is ExecutionStatus.COMPLETED  # a storage failure, not a task failure
    assert degraded.output is None
    assert degraded.error is not None
    assert degraded.error["code"] == "output_spill_failed"
    assert degraded.error["type"] == "ArtifactStoreLimitExceeded"
    spill_record = degraded.artifacts["output_spill_error"]
    assert isinstance(spill_record, dict)
    assert spill_record["code"] == "output_spill_failed"
    assert spill_record["type"] == "ArtifactStoreLimitExceeded"
    # Equal in content but two independent dicts, not one object stored
    # twice: a later stage that rewrites one field (report-boundary
    # redaction does exactly that) must not silently rewrite the other.
    assert spill_record is not degraded.error
    assert list(root.glob("*.bin")) == []
    assert list(root.glob("*.json")) == []


def test_an_error_the_target_already_reported_is_never_overwritten(tmp_path: Path) -> None:
    """``error`` is the sample's *primary* diagnosis, so the spill record only
    claims it when nothing else has. A target that already explained its own
    failure keeps that explanation verbatim; the spill failure is still
    recorded, in the ``artifacts`` namespace this boundary owns, so neither
    piece of evidence is lost.
    """
    store = ArtifactStore(tmp_path / "artifacts", max_bytes=_REJECTING_STORE_MAX_BYTES)
    runner = _default_runner(store)
    execution = _oversized_execution_the_target_already_failed()

    degraded = runner._spill_isolated(execution)

    assert degraded.error == execution.error  # the target's own diagnosis, untouched
    assert degraded.status is ExecutionStatus.ERROR
    assert degraded.output is None
    spill_record = degraded.artifacts["output_spill_error"]
    assert isinstance(spill_record, dict)
    assert spill_record["code"] == "output_spill_failed"
    assert spill_record["type"] == "ArtifactStoreLimitExceeded"


def test_a_store_failure_that_is_not_a_value_error_degrades_identically(tmp_path: Path) -> None:
    """Nothing about this depends on the store's exception being an
    ``ArtifactStoreLimitExceeded``. A full disk raises ``OSError``, which
    shares no base class with it below ``Exception`` -- and it has to be
    absorbed just the same, with ``error["type"]`` naming the class that
    actually failed so the real cause stays diagnosable.
    """
    store = _FailingStore(
        tmp_path / "artifacts", error_factory=lambda: OSError("no space left on device")
    )
    runner = _default_runner(store)

    degraded = runner._spill_isolated(_oversized_execution())

    assert degraded.output is None
    assert degraded.error is not None
    assert degraded.error["type"] == "OSError"
    assert degraded.error["code"] == "output_spill_failed"
    assert degraded.error["message"] == "no space left on device"


def test_a_secret_in_the_store_failure_message_is_redacted_before_it_is_recorded(
    tmp_path: Path,
) -> None:
    """The recorded message comes from ``str(error)``, and a store's exception
    text is not guaranteed to be framework-authored -- a remote or wrapping
    store can echo a URL, a header, or a credential into it. It therefore gets
    the same redact-then-bound treatment (ADR-0018 order) the runner already
    applies to a raising target's message, under the default policy, with no
    redaction configuration required of the caller.
    """
    store = _FailingStore(
        tmp_path / "artifacts",
        error_factory=lambda: OSError(f"upload rejected: token={_HF_SECRET}"),
    )
    runner = _default_runner(store)

    degraded = runner._spill_isolated(_oversized_execution())

    assert degraded.error is not None
    message = degraded.error["message"]
    assert isinstance(message, str)
    assert _HF_SECRET not in message
    assert "[REDACTED]" in message


def test_cancellation_raised_by_the_store_is_deliberately_not_absorbed(tmp_path: Path) -> None:
    """``_spill_isolated`` catches ``Exception``, never ``BaseException``, so
    ``asyncio.CancelledError`` still tears the run down exactly as it does at
    the target and grader boundaries. Absorbing it here would leave a
    cancelled run looking like it finished normally.
    """
    store = _FailingStore(tmp_path / "artifacts", error_factory=asyncio.CancelledError)
    runner = _default_runner(store)

    with pytest.raises(asyncio.CancelledError):
        runner._spill_isolated(_oversized_execution())


def test_the_isolation_holds_even_when_the_failure_handler_itself_would_raise(
    tmp_path: Path,
) -> None:
    """The subtle way this isolation could have failed: the handler that
    records the failure calls ``_safe_error_message``, which compiles the
    caller's ``secret_patterns`` -- the very call that raises ``re.error``
    inside ``_spill_large_output`` when a pattern is malformed. If the
    handler re-raised, the original exception would land straight back in
    the runner's ``TaskGroup`` and cancel every sibling: the isolation would
    hold for every failure except the one it was handling. It must degrade
    instead, recording at least the exception's class name.
    """
    store = ArtifactStore(tmp_path / "artifacts", max_bytes=_REJECTING_STORE_MAX_BYTES)
    runner = EvalRunner(
        catalog=cast("Any", None),
        adapters={},
        targets={},
        graders={},
        artifact_store=store,
        redaction_policy=RedactionPolicy(secret_patterns=("(",)),
    )

    degraded = runner._spill_isolated(_oversized_execution())

    assert degraded.output is None
    assert degraded.error is not None
    assert degraded.error["code"] == "output_spill_failed"
    # The bad pattern makes the store's own message unrenderable, so the
    # record falls back to naming the exception class -- less detail, but
    # the failure is still reported rather than swallowed or re-raised.
    # The class name itself is not pinned: `re.error` is spelled
    # `PatternError` from 3.13 on, and which alias `__name__` reports is not
    # what this test is about.
    message = degraded.error["message"]
    assert isinstance(message, str)
    assert "message unavailable" in message
    # The recorded ``type`` must name the class that actually raised -- here
    # the malformed pattern's ``re.error``, since compiling the patterns
    # happens before the store is ever reached. Comparing against
    # ``re.error.__name__`` rather than a literal keeps the assertion about
    # behaviour while staying agnostic about the spelling (``error`` before
    # 3.13, ``PatternError`` from 3.13 on). Re-deriving the expectation from
    # the message instead, as this once did, pinned the fallback f-string's
    # formatting and would have passed no matter which class was named.
    assert degraded.error["type"] == re.error.__name__


# --- the isolation must not destroy an output that was never a spill candidate ---
#
# ``_spill_isolated`` guards the whole of ``_spill_large_output``, but three of
# that method's steps run *before* the size check and before the artifact store
# is touched at all: ``str()`` on the output, compiling the caller's
# ``secret_patterns``, and encoding the result. A raise from any of those --
# ``re.error`` from a malformed pattern is the reachable one, since
# ``RedactionPolicy`` accepts an invalid regex at construction -- reaches the
# same failure handler. Nulling the output there would destroy a small inline
# answer the store was never even offered, and because the policy is a
# per-runner setting it would do so for *every* sample in the run, which would
# then exit 0 while reporting that the store had dropped the outputs.


def test_a_failure_before_the_size_check_leaves_a_small_output_inline(tmp_path: Path) -> None:
    """The output here is ~30 bytes -- three orders of magnitude below the
    spill threshold, so it was never going to reach the artifact store. A
    malformed secret pattern makes the spill boundary raise anyway, and the
    sample must keep the answer it earned: an output within the threshold is
    bounded by definition, which is all requirement 8 asks of it, and it is
    redacted at the report boundary like any other small output.

    The failure is still recorded -- this is a degradation, not a silent
    swallow -- so ``artifacts`` carries the record either way.
    """
    runner = EvalRunner(
        catalog=cast("Any", None),
        adapters={},
        targets={},
        graders={},
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        redaction_policy=RedactionPolicy(secret_patterns=("(",)),
    )
    small = _execution_with_output({"answer": "42"})

    degraded = runner._spill_isolated(small)

    assert degraded.output == {"answer": "42"}  # never oversized, never dropped
    # Not merely "the key is present": that is true of anything at all under
    # that name, including the target-planted junk a sibling test spends a
    # whole parametrize block rejecting. The record has to be one the runner
    # wrote, which is what every consumer downstream actually tests for.
    record = degraded.artifacts["output_spill_error"]
    assert isinstance(record, dict)
    assert record["code"] == "output_spill_failed"
    assert degraded.error is not None
    assert degraded.error["code"] == "output_spill_failed"


def test_an_oversized_output_is_still_dropped_when_the_failure_is_not_the_store(
    tmp_path: Path,
) -> None:
    """The other side of the rule above. A payload genuinely past the
    threshold cannot be kept inline no matter *which* step failed -- that is
    the unbounded payload requirement 8 exists to prevent -- so it is still
    dropped even though the store was never reached.
    """
    runner = EvalRunner(
        catalog=cast("Any", None),
        adapters={},
        targets={},
        graders={},
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        redaction_policy=RedactionPolicy(secret_patterns=("(",)),
    )

    degraded = runner._spill_isolated(_oversized_execution())

    assert degraded.output is None
    assert degraded.error is not None
    assert degraded.error["code"] == "output_spill_failed"


def test_a_successful_spill_clears_a_stale_spill_error_record(tmp_path: Path) -> None:
    """``output_ref`` and ``output_spill_error`` describe the same boundary's
    outcome and cannot both be true. A stale failure record must not survive a
    spill that then succeeded: consumers check the failure key *first*, so
    leaving it would make the grader announce that bytes sitting on disk "were
    never persisted", while steering the reader away from the reference that
    finds them.

    The reachable producer of such a record is the target: every attempt
    builds a fresh ``NormalizedExecutionResult``, so ``_spill_large_output``
    never inherits one the runner wrote on an earlier attempt. What that
    leaves is a target echoing the taxonomy code back -- planted directly
    here, since going through a target adds nothing to what is under test.
    ``test_a_successful_spill_keeps_a_targets_own_data_under_that_key`` is the
    complement: only a value carrying the code is removed, so the clearing
    cannot eat a target's unrelated notes.
    """
    runner = _default_runner(ArtifactStore(tmp_path / "artifacts"))
    stale = _oversized_execution().model_copy(
        update={
            "artifacts": {
                "output_spill_error": {
                    "type": "OSError",
                    "code": "output_spill_failed",
                    "message": "an earlier attempt",
                }
            }
        }
    )

    spilled = runner._spill_isolated(stale)

    assert "output_ref" in spilled.artifacts
    assert "output_spill_error" not in spilled.artifacts
    assert spilled.output is None


def test_a_successful_spill_keeps_a_targets_own_data_under_that_key(tmp_path: Path) -> None:
    """The clearing above is scoped to records the *runner* wrote.

    ``artifacts`` is target-controlled and this key is not reserved -- that
    is the whole premise of ``is_output_spill_error_record``. A target that
    keeps its own upload diagnostics under that name must not have them
    silently deleted just because its output happened to be large enough to
    spill. Nor can such a value cause the misreading the clearing exists to
    prevent: every consumer validates the record's shape before acting on
    it, so a value without the taxonomy code is already inert.
    """
    runner = _default_runner(ArtifactStore(tmp_path / "artifacts"))
    targets_own = _oversized_execution().model_copy(
        update={"artifacts": {"output_spill_error": {"note": "my own upload log"}}}
    )

    spilled = runner._spill_isolated(targets_own)

    assert "output_ref" in spilled.artifacts
    assert spilled.artifacts["output_spill_error"] == {"note": "my own upload log"}
