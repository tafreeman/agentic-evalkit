"""The one place an :class:`IntegrityFlag` is built, so evidence is always redacted.

A flag quotes the exact line that triggered it, and that line can hold a
token (a ``curl`` with an auth header, a ``.env`` read). Every detector builds
its flags through :func:`make_flag`; ``tests/contract/test_integrity_boundary.py``
fails if any other module constructs an ``IntegrityFlag`` directly.

The default redaction policy covers model and bearer tokens. A transcript is
likelier than most text to hold a GitHub token, an AWS key or credentials in a
URL, so this module adds patterns for those on top of it.
"""

from __future__ import annotations

from agentic_evalkit.integrity.models import IntegrityFlag, IntegrityStage, Severity
from agentic_evalkit.reporters.base import DEFAULT_REDACTION_POLICY, RedactionPolicy, redact_text

__all__ = ["EVIDENCE_LIMIT", "INTEGRITY_REDACTION_POLICY", "make_flag", "redact"]

#: Longest evidence line kept, after redaction. Long enough to show a full
#: command, short enough that a whole transcript can't ride in one flag.
EVIDENCE_LIMIT = 300

INTEGRITY_REDACTION_POLICY = RedactionPolicy(
    secret_patterns=(
        *DEFAULT_REDACTION_POLICY.secret_patterns,
        r"gh[pousr]_[A-Za-z0-9]{20,}",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"(?:AKIA|ASIA)[0-9A-Z]{16}",
        r"(?i:token)\s+[A-Za-z0-9._~+/=-]{16,}",
        r"(?<=://)[^\s/@:]+:[^\s/@]+(?=@)",
    ),
)


def redact(value: str) -> str:
    """Redact ``value`` with the integrity policy."""
    return redact_text(value, INTEGRITY_REDACTION_POLICY)


def make_flag(
    *,
    instance_id: str,
    stage: IntegrityStage,
    rule: str,
    severity: Severity,
    evidence: str,
    call_id: str | None = None,
) -> IntegrityFlag:
    """Build a flag whose evidence is redacted, then bounded to :data:`EVIDENCE_LIMIT`."""
    line = redact(evidence.strip())
    if len(line) > EVIDENCE_LIMIT:
        line = line[: EVIDENCE_LIMIT - 3] + "..."
    return IntegrityFlag(
        instance_id=instance_id,
        stage=stage,
        rule=rule,
        severity=severity,
        evidence=line,
        call_id=redact(call_id) if call_id is not None else None,
    )
