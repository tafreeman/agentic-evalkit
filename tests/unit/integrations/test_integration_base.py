"""The shared plumbing under both host-platform bridges.

The centre of gravity here is :func:`judge_authority`, because it is the
control the whole bridge exists to carry into somebody else's platform. Its
three outcomes are not a preference -- they are ADR-0007's decision D-1 as
amended 2026-07-04 -- and the distinction that matters most is the one an
implementation is most likely to lose: *bad* evidence and *absent* evidence
must not produce the same answer. A judge proven unreliable has to be
silenced; a judge merely unproven has to be heard and disbelieved. Collapse
those two and the control still looks like it works, while quietly either
suppressing usable signal or letting a known-bad judge speak.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agentic_evalkit.errors import IntegrationUnavailable
from agentic_evalkit.graders.calibration import CalibrationArtifact
from agentic_evalkit.integrations.base import (
    AuthorityLevel,
    judge_authority,
    require_dependency,
    run_blocking,
)

_NOW = datetime(2026, 8, 4, 12, 0, 0, tzinfo=UTC)


def _calibration(
    *,
    true_positive: int = 200,
    true_negative: int = 400,
    false_positive: int = 2,
    false_negative: int = 5,
    calibrated_at: datetime | None = _NOW - timedelta(days=1),
    expires_at: datetime = _NOW + timedelta(days=30),
    threshold: float = 0.8,
    judge_fingerprint: str = "sha256:judge-a",
) -> CalibrationArtifact:
    """A calibration that clears every project floor, unless a test bends one.

    The counts are large on purpose, and the reason is the check most
    easily forgotten: the Wilson lower bound, not the raw rate, is what has
    to clear the floor. A tempting fixture of TP=40/FN=2, TN=60/FP=1 has a
    raw TNR of 0.984 and a raw TPR of 0.952 -- both comfortably above the
    0.95 and 0.85 floors -- and still fails to gate, because on those sample
    sizes the 95% Wilson lower bounds are only 0.913 and 0.842. These counts
    (TNR lower bound 0.982, TPR lower bound 0.944) clear the floors on the
    conservative measure that actually governs. Getting this wrong would
    make every ADVISORY assertion below ambiguous: it would be unclear
    whether a test found the condition it was aiming at or merely tripped
    the sample-size bar.
    """
    return CalibrationArtifact(
        calibration_id="cal-001",
        judge_fingerprint=judge_fingerprint,
        expires_at=expires_at,
        calibrated_at=calibrated_at,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        threshold=threshold,
    )


def test_full_calibration_earns_the_authority_to_gate() -> None:
    authority = judge_authority(_calibration(), now=_NOW)
    assert authority.level is AuthorityLevel.GATING
    assert authority.reason is None
    assert authority.calibration_id == "cal-001"


def test_no_calibration_at_all_is_advisory_not_unavailable() -> None:
    """A judge nobody has measured is unproven, not disproven.

    This is the single most consequential branch in the function. Returning
    UNAVAILABLE here would look more cautious and would be wrong: it would
    throw away the verdict of every judge that simply has not been
    calibrated yet, which is nearly all of them, and would make adopting the
    gate cost a team all their existing signal on day one.
    """
    authority = judge_authority(None, now=_NOW)
    assert authority.level is AuthorityLevel.ADVISORY
    assert authority.reason is not None
    assert "advisory-only" in authority.reason
    assert authority.calibration_id is None


def test_expired_calibration_is_unavailable() -> None:
    """Evidence that has aged out is present and bad, so the verdict is withheld."""
    authority = judge_authority(
        _calibration(
            calibrated_at=_NOW - timedelta(days=200),
            expires_at=_NOW - timedelta(days=1),
        ),
        now=_NOW,
    )
    assert authority.level is AuthorityLevel.UNAVAILABLE
    assert authority.reason is not None
    assert "expired" in authority.reason


def test_measured_tnr_below_the_project_floor_is_unavailable() -> None:
    """A judge that demonstrably waves bad answers through is silenced, not demoted.

    30 false positives against 30 true negatives is a 0.50 TNR on a
    sufficient sample -- not a thin measurement, a bad one. The project floor
    is 0.95, and no per-judge ``threshold`` can be configured loose enough to
    get around it.
    """
    authority = judge_authority(
        _calibration(true_negative=30, false_positive=30, threshold=0.1),
        now=_NOW,
    )
    assert authority.level is AuthorityLevel.UNAVAILABLE
    assert authority.reason is not None
    assert "TNR" in authority.reason


def test_measured_tpr_below_the_project_floor_is_unavailable() -> None:
    authority = judge_authority(
        _calibration(true_positive=30, false_negative=30, threshold=0.1),
        now=_NOW,
    )
    assert authority.level is AuthorityLevel.UNAVAILABLE
    assert authority.reason is not None
    assert "TPR" in authority.reason


def test_too_few_held_out_samples_is_advisory_not_unavailable() -> None:
    """A perfect score on five examples is thin evidence, not bad evidence.

    Nothing here says the judge is wrong -- there simply is not enough of it
    to say anything. That is the ADVISORY case, and marking it UNAVAILABLE
    would treat "we did not measure much" as "we measured badly".
    """
    authority = judge_authority(
        _calibration(true_positive=5, true_negative=5, false_positive=0, false_negative=0),
        now=_NOW,
    )
    assert authority.level is AuthorityLevel.ADVISORY
    assert authority.reason is not None
    assert "minimum" in authority.reason


def test_missing_calibrated_at_cannot_buy_its_way_past_the_age_check() -> None:
    """Leaving the measurement date out must not be a way to look fresh forever."""
    authority = judge_authority(_calibration(calibrated_at=None), now=_NOW)
    assert authority.level is AuthorityLevel.ADVISORY
    assert authority.reason is not None
    assert "calibrated_at" in authority.reason


def test_calibration_older_than_the_age_limit_is_advisory() -> None:
    authority = judge_authority(
        _calibration(calibrated_at=_NOW - timedelta(days=120)),
        now=_NOW,
    )
    assert authority.level is AuthorityLevel.ADVISORY
    assert authority.reason is not None
    assert "age exceeds" in authority.reason


def test_calibration_for_a_different_judge_cannot_gate_this_one() -> None:
    """Evidence is about a specific model+prompt, and does not transfer.

    Without this check the gate would be trivially defeatable: calibrate one
    cheap judge properly, then point the artifact at whatever judge you
    actually wanted to ship.
    """
    authority = judge_authority(
        _calibration(judge_fingerprint="sha256:judge-a"),
        judge_fingerprint="sha256:judge-b",
        now=_NOW,
    )
    assert authority.level is AuthorityLevel.ADVISORY
    assert authority.reason is not None
    assert "does not match" in authority.reason


def test_matching_fingerprint_still_gates() -> None:
    authority = judge_authority(
        _calibration(judge_fingerprint="sha256:judge-a"),
        judge_fingerprint="sha256:judge-a",
        now=_NOW,
    )
    assert authority.level is AuthorityLevel.GATING


def test_expiry_is_decided_before_thin_evidence() -> None:
    """Order matters: an expired calibration reports as expired, not as thin.

    Both problems are present in this fixture. If the thin-evidence family
    were checked first, an expired judge would come back ADVISORY -- still
    speaking, and reported under the wrong reason -- which is exactly the
    demotion-tier confusion the two-tier rule exists to prevent.
    """
    authority = judge_authority(
        _calibration(
            true_positive=2,
            true_negative=2,
            false_positive=0,
            false_negative=0,
            calibrated_at=_NOW - timedelta(days=200),
            expires_at=_NOW - timedelta(days=1),
        ),
        now=_NOW,
    )
    assert authority.level is AuthorityLevel.UNAVAILABLE
    assert authority.reason is not None
    assert "expired" in authority.reason


def test_authority_verdict_is_immutable() -> None:
    """ADR-0002: the verdict is a frozen wire model like everything else here."""
    authority = judge_authority(_calibration(), now=_NOW)
    with pytest.raises(ValueError, match="frozen"):
        authority.level = AuthorityLevel.ADVISORY  # type: ignore[misc]


# --- require_dependency ----------------------------------------------------


def test_missing_dependency_names_the_install_command() -> None:
    """The error a user actually hits has to say what to type next.

    A bare ImportError arriving from inside an exporter tells someone what
    broke but not what to do, and "which extra provides this?" is not
    guessable from the module name.
    """
    with pytest.raises(IntegrationUnavailable) as excinfo:
        require_dependency("definitely_not_a_real_package_xyz", extra="mlflow")
    message = str(excinfo.value)
    assert "pip install 'agentic-evalkit[mlflow]'" in message
    assert excinfo.value.context["module"] == "definitely_not_a_real_package_xyz"


def test_present_dependency_is_returned() -> None:
    module = require_dependency("json", extra="mlflow")
    assert module.dumps({"a": 1}) == '{"a": 1}'


# --- run_blocking ----------------------------------------------------------


async def _answer() -> int:
    await asyncio.sleep(0)
    return 42


def test_run_blocking_drives_a_coroutine_with_no_loop_running() -> None:
    assert run_blocking(_answer()) == 42


async def test_run_blocking_works_inside_an_already_running_loop() -> None:
    """The case a naive ``asyncio.run`` bridge gets wrong.

    Every grader here is async while both host platforms' scorer interfaces
    are synchronous, so something has to bridge them -- and the obvious
    bridge raises ``RuntimeError`` the moment a scorer is called from a
    notebook, an async web handler, or an async evaluation harness. This
    test runs inside a live event loop precisely because that is the
    environment where the bug appears.
    """
    assert run_blocking(_answer()) == 42


def test_run_blocking_propagates_the_coroutine_s_exception() -> None:
    """A grader that raises must not have its failure swallowed by the bridge."""

    async def boom() -> int:
        raise RuntimeError("grader exploded")

    with pytest.raises(RuntimeError, match="grader exploded"):
        run_blocking(boom())
