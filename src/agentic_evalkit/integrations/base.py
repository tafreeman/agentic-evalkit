"""Shared plumbing every host-platform export goes through (ADR-0022).

An "export" here means handing a finished run to somebody else's system --
an MLflow tracking server, a Langfuse project -- so the validity evidence
this package produces lands where a team already works, instead of only in
a local report file. That direction of travel is the whole point of this
subpackage, and it is also what makes it the riskiest surface in the
codebase: a report file stays on the machine that wrote it, while an export
leaves it. Everything below exists to make the leaving safe and boring.

Three rules are enforced here rather than restated in each exporter:

* **Redaction happens exactly once, and it happens here.** Every exporter
  calls :func:`redact_for_export` as its first act and works only with what
  comes back. Because that is a single named function rather than a habit,
  ``tests/contract/test_integration_redaction.py`` can count the calls and
  fail CI if a future exporter forgets -- the same tripwire design the
  reporters use with ``REDACTION_ROUTED_FORMATS``.
* **The host library is never imported at module scope.** ``import
  agentic_evalkit.integrations.mlflow`` must succeed on a machine with no
  MLflow installed, so that ``from agentic_evalkit import integrations``
  stays free and the optional extra stays genuinely optional (ADR-0009).
  :func:`require_dependency` is the one place that import happens, and the
  one place that turns a missing package into an explanation.
* **This package is a guest.** Nothing here is allowed to become a required
  dependency of the rest of the codebase, and nothing in the rest of the
  codebase imports from here. The dependency arrow points one way: outward.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar

from agentic_evalkit.errors import IntegrationUnavailable
from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.reporters import apply_redaction

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from types import ModuleType

    from agentic_evalkit.graders.calibration import CalibrationArtifact
    from agentic_evalkit.models import EvalRunResult
    from agentic_evalkit.reporters import RedactionPolicy

_T = TypeVar("_T")

__all__ = [
    "AuthorityLevel",
    "JudgeAuthority",
    "judge_authority",
    "redact_for_export",
    "require_dependency",
    "run_blocking",
]


class AuthorityLevel(StrEnum):
    """How much a judge's verdict is allowed to decide, given its calibration evidence.

    This is the three-way outcome of ADR-0007's decision D-1 (as amended
    2026-07-04), named so it can travel into a host platform's metadata as
    a label rather than as a bare boolean. The whole reason it has three
    members instead of two is that "we proved this judge is bad" and "we
    have no proof either way" are different facts about the world, and
    collapsing them loses the distinction that makes the control honest.

    - ``GATING``: full calibration evidence clears every floor, so this
      judge's verdict may block a release.
    - ``ADVISORY``: evidence is *absent* or too thin to prove reliability
      (no artifact, no ``calibrated_at``, too few held-out samples, a
      Wilson lower bound that does not clear the floor, a fingerprint that
      does not match the live judge). The verdict is still reported --
      it may well be right -- but it can never gate.
    - ``UNAVAILABLE``: evidence is *present and bad* (expired, or a
      measured TNR/TPR genuinely below the project floor on a sufficient
      sample). There is proof this judge should not be trusted here, so
      no verdict is reported at all.
    """

    GATING = "gating"
    ADVISORY = "advisory"
    UNAVAILABLE = "unavailable"


class JudgeAuthority(FrozenModel):
    """The verdict on a judge's own evidence, and why it came out that way.

    Attributes:
        level: How much this judge's verdict may decide -- see
            :class:`AuthorityLevel`.
        reason: Why, in words a reviewer can act on. ``None`` only when
            ``level`` is :attr:`AuthorityLevel.GATING`, where there is no
            failure to explain.
        calibration_id: Which calibration record produced this verdict.
            ``None`` when no artifact was supplied at all.
    """

    level: AuthorityLevel
    reason: str | None = None
    calibration_id: str | None = None


def judge_authority(
    calibration: CalibrationArtifact | None,
    *,
    judge_fingerprint: str | None = None,
    now: datetime | None = None,
) -> JudgeAuthority:
    """Decide what a judge's calibration evidence entitles its verdict to do.

    This is the control the whole bridge exists to carry into somebody
    else's platform, so it is worth being exact about what it is and is not.
    It does not score anything, does not call a model, and does not look at
    a verdict -- it looks only at the *evidence about the judge* and returns
    how far that evidence reaches. A platform that already has a judge it
    likes can wrap it with this and gain the one property the judge was
    missing: an inability to block a release it has not earned the right to
    block.

    The order of checks below is load-bearing and matches
    :class:`~agentic_evalkit.graders.judge.JudgeGrader` exactly. Proof of a
    *bad* judge is established first, before anything else, so a judge with
    expired or sub-floor calibration can never slip out as an advisory pass;
    only after that does the weaker "not enough proof" family get
    considered. Every threshold is read from
    :class:`~agentic_evalkit.graders.calibration.CalibrationArtifact` itself
    rather than recomputed here, so the project floors (TNR >= 0.95,
    TPR >= 0.85, age <= 90 days, the 95% Wilson lower bound) have exactly
    one definition in this codebase and this function cannot drift from it.

    Args:
        calibration: The judge's calibration record, or ``None`` if it has
            none. ``None`` is not an error -- it is the ordinary case for a
            judge someone just wrote, and it yields
            :attr:`AuthorityLevel.ADVISORY`.
        judge_fingerprint: The live judge's fingerprint (model + prompt). If
            given and it does not match what the calibration was measured
            against, the calibration describes a different judge and cannot
            gate this one. Left ``None`` to skip that check, which is right
            when the host platform does not expose a stable fingerprint.
        now: The moment to evaluate expiry and age against. Defaults to the
            current UTC time; tests pass a fixed value.

    Returns:
        A :class:`JudgeAuthority` carrying the level and the reason.
    """
    effective_now = now or datetime.now(UTC)

    if calibration is None:
        return JudgeAuthority(
            level=AuthorityLevel.ADVISORY,
            reason="no calibration artifact was supplied; judge is advisory-only",
        )

    # Tier one: solid proof the judge is unreliable. Expiry first, then a
    # measured rate genuinely below the project floor -- the two cases where
    # the evidence exists and is bad, rather than merely being thin.
    if calibration.is_expired(now=effective_now):
        return JudgeAuthority(
            level=AuthorityLevel.UNAVAILABLE,
            reason=(
                f"calibration {calibration.calibration_id!r} expired at "
                f"{calibration.expires_at.isoformat()}"
            ),
            calibration_id=calibration.calibration_id,
        )
    floor_reason = calibration.floor_failure_reason()
    if floor_reason is not None:
        return JudgeAuthority(
            level=AuthorityLevel.UNAVAILABLE,
            reason=floor_reason,
            calibration_id=calibration.calibration_id,
        )

    # Tier two: the evidence is not bad, it is just not enough. A judge here
    # still reports its verdict; it simply may not gate on it.
    if judge_fingerprint is not None and calibration.judge_fingerprint != judge_fingerprint:
        return JudgeAuthority(
            level=AuthorityLevel.ADVISORY,
            reason=(
                f"calibration fingerprint {calibration.judge_fingerprint!r} does not "
                f"match live judge fingerprint {judge_fingerprint!r}"
            ),
            calibration_id=calibration.calibration_id,
        )
    usability_reason = calibration.usability_failure_reason(now=effective_now)
    if usability_reason is not None:
        return JudgeAuthority(
            level=AuthorityLevel.ADVISORY,
            reason=usability_reason,
            calibration_id=calibration.calibration_id,
        )

    return JudgeAuthority(
        level=AuthorityLevel.GATING,
        calibration_id=calibration.calibration_id,
    )


def require_dependency(module_name: str, *, extra: str) -> ModuleType:
    """Import a host platform's client library, or explain how to install it.

    Called at the top of every exported function rather than at module
    scope, so importing this package never requires the host library to be
    present. The cost of a repeated import is a dictionary lookup in
    ``sys.modules``; the benefit is that the extra stays optional in the
    real sense -- a user who wants the Langfuse bridge does not have to
    install MLflow to get it.

    Args:
        module_name: The import root to load, e.g. ``"mlflow"``.
        extra: The name of this package's optional extra that provides it,
            used to build the install hint in the error message.

    Returns:
        The imported module.

    Raises:
        IntegrationUnavailable: If the module cannot be imported. The
            message names the exact ``pip install`` line that fixes it,
            because "No module named 'mlflow'" arriving from three frames
            inside an exporter tells a user what broke but not what to do.
    """
    # Imported here rather than at module scope purely for symmetry with the
    # rule this function exists to enforce; importlib itself is stdlib and
    # always present.
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise IntegrationUnavailable(
            message=(
                f"the {extra!r} integration needs the {module_name!r} package, which is "
                f"not installed; install it with: pip install 'agentic-evalkit[{extra}]'"
            ),
            context={"module": module_name, "extra": extra},
        ) from exc


def redact_for_export(run: EvalRunResult, policy: RedactionPolicy) -> EvalRunResult:
    """Scrub secrets out of ``run`` once, before any of it leaves this machine.

    This is a thin wrapper over :func:`~agentic_evalkit.reporters.apply_redaction`
    and deliberately adds no behaviour of its own. It exists to be a *name*:
    one function that every export path calls exactly once, which a contract
    test can patch and count. Wrapping also keeps the guarantee honest in the
    other direction -- because exporters never call ``apply_redaction``
    directly, "did this exporter redact?" is answerable by reading its first
    statement rather than by auditing the whole function.

    Note the difference from the reporters, and treat it as deliberate: a
    reporter writes to a path the caller chose on a machine the caller
    already controls, so the library leaves the policy entirely to them. An
    exporter transmits to a shared server that colleagues, and often other
    teams, can read. So the exporters here default to
    :data:`~agentic_evalkit.reporters.DEFAULT_REDACTION_POLICY` instead of to
    no redaction, and a caller who genuinely wants raw output on the wire
    has to say so by passing ``RedactionPolicy()`` explicitly.

    Args:
        run: The finished run about to be exported.
        policy: What to scrub. ``RedactionPolicy()`` (no patterns) is the
            supported way to opt out.

    Returns:
        A new, redacted run. ``run`` itself is never modified (ADR-0002).
    """
    return apply_redaction(run, policy)


def run_blocking(coroutine: Coroutine[object, object, _T]) -> _T:
    """Run an async grader to completion from synchronous host-platform code.

    Every grader in this package is ``async`` (``graders.base.Grader``),
    while the scorer interfaces both MLflow and Langfuse expose are ordinary
    synchronous callables. Something has to bridge that, and the naive
    bridge -- :func:`asyncio.run` -- raises ``RuntimeError`` the moment it is
    called from a thread that already has a running event loop. That is not
    an exotic case: it is what happens when a scorer runs inside a notebook,
    inside an async web handler, or inside an evaluation harness that is
    itself async.

    So this checks first. With no loop running, :func:`asyncio.run` is
    correct and cheap. With a loop already running on this thread, the
    coroutine is handed to a short-lived worker thread that owns its own
    fresh loop, and this thread blocks on the result. The worker cannot
    deadlock against the caller's loop because it shares nothing with it.

    Args:
        coroutine: The coroutine to drive to completion.

    Returns:
        Whatever the coroutine returns.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop on this thread: the ordinary case, and the cheap path.
        return asyncio.run(coroutine)

    # A loop is already running here, so asyncio.run would refuse. Give the
    # coroutine its own thread and its own loop, and wait for it. max_workers
    # is 1 because there is exactly one piece of work; the executor is
    # created and torn down per call rather than pooled, since a scorer may
    # be invoked from arbitrary host-platform threads and a module-level
    # pool would outlive the process's interest in it.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()
