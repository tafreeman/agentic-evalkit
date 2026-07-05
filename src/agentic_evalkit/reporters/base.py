"""Reporter protocol and redaction policy shared by every reporter format.

Reporters consume a completed :class:`~agentic_evalkit.models.EvalRunResult`
only; they never execute targets, grade samples, or perform aggregation
(design §11.3). Redaction happens once, immutably, before any reporter sees
the model, so every output format observes the same redacted evidence
(design §12).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from agentic_evalkit.models import EvalRunResult, GradeResult, SampleResult
from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.execution import NormalizedExecutionResult

_REDACTED = "[REDACTED]"


class RedactionPolicy(FrozenModel):
    """Declares what must never appear in a rendered report.

    ``evidence_keys`` names grade-evidence dictionary keys to drop entirely.
    ``secret_patterns`` are regular expressions; any matching substring found
    in a string evidence value is replaced with ``"[REDACTED]"`` rather than
    dropping the whole value, so surrounding context survives.
    """

    evidence_keys: tuple[str, ...] = ()
    secret_patterns: tuple[str, ...] = ()


#: Conservative default applied at the CLI report boundaries (``run`` writes
#: the canonical JSON through it; ``report`` re-applies it before rendering).
#: Patterns target well-known credential shapes only -- Hugging Face user
#: tokens, OpenAI-style secret keys, and HTTP bearer/authorization values --
#: with length guards so ordinary evidence text (words that merely start with
#: "hf_" or "sk-") is never mangled. Library callers are unaffected: reporters
#: apply no policy themselves, and callers compose their own via
#: :func:`apply_redaction`.
DEFAULT_REDACTION_POLICY = RedactionPolicy(
    secret_patterns=(
        r"hf_[A-Za-z0-9]{16,}",
        r"sk-[A-Za-z0-9_-]{16,}",
        r"(?i:bearer)\s+[A-Za-z0-9._~+/=-]{8,}",
        r"(?i:authorization)\s*[:=]\s*\S{8,}",
    ),
)


@runtime_checkable
class Reporter(Protocol):
    """A pure function from a completed run to a written report file.

    Implementations must not mutate ``run``; every reporter renders from an
    immutable model, optionally decorated with ``aggregates`` supplied by a
    caller that already ran ``agentic_evalkit.stats`` (this package never
    imports that module itself).
    """

    def write(
        self,
        run: EvalRunResult,
        destination: Path,
        *,
        aggregates: dict[str, JsonValue] | None = None,
        generated_at: str | None = None,
    ) -> Path:
        """Render ``run`` to ``destination`` and return the written path."""
        ...


def _redact_string(value: str, patterns: tuple[re.Pattern[str], ...]) -> str:
    redacted = value
    for pattern in patterns:
        redacted = pattern.sub(_REDACTED, redacted)
    return redacted


def _redact_json_value(value: JsonValue, patterns: tuple[re.Pattern[str], ...]) -> JsonValue:
    if isinstance(value, str):
        return _redact_string(value, patterns)
    if isinstance(value, dict):
        return {key: _redact_json_value(item, patterns) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_json_value(item, patterns) for item in value]
    return value


def _redact_evidence(
    evidence: dict[str, JsonValue],
    *,
    evidence_keys: frozenset[str],
    patterns: tuple[re.Pattern[str], ...],
) -> dict[str, JsonValue]:
    return {
        key: _redact_json_value(value, patterns)
        for key, value in evidence.items()
        if key not in evidence_keys
    }


def _redact_grade(
    grade: GradeResult,
    *,
    evidence_keys: frozenset[str],
    patterns: tuple[re.Pattern[str], ...],
) -> GradeResult:
    if not grade.evidence:
        return grade
    redacted_evidence = _redact_evidence(
        grade.evidence, evidence_keys=evidence_keys, patterns=patterns
    )
    if redacted_evidence == grade.evidence:
        return grade
    return grade.model_copy(update={"evidence": redacted_evidence})


def _redact_execution(
    execution: NormalizedExecutionResult, *, patterns: tuple[re.Pattern[str], ...]
) -> NormalizedExecutionResult:
    """Redact secret-shaped substrings from a system-under-test's raw output.

    Unlike grade evidence, ``output``/``structured_output``/``error`` are the
    target's own words, not the harness's -- there is no meaningful
    ``evidence_keys``-style drop list for keys the harness doesn't define, so
    only pattern substitution applies here, never key removal. Without this,
    an output that happens to embed a credential-shaped value is written to
    the canonical report unredacted regardless of the spill threshold: the
    spill boundary (``EvalRunner._spill_large_output``) only redacts bytes
    that are about to leave the in-memory result as an oversized artifact,
    and small/never-spilled outputs reach the report exactly as the target
    returned them until this function also covers them.
    """
    if not patterns:
        return execution
    updates: dict[str, object] = {}
    for field_name in ("output", "structured_output", "error"):
        value = getattr(execution, field_name)
        if value is None:
            continue
        redacted_value = _redact_json_value(value, patterns)
        if redacted_value != value:
            updates[field_name] = redacted_value
    if not updates:
        return execution
    return execution.model_copy(update=updates)


def _redact_sample(
    sample: SampleResult,
    *,
    evidence_keys: frozenset[str],
    patterns: tuple[re.Pattern[str], ...],
) -> SampleResult:
    updates: dict[str, object] = {}
    redacted_execution = _redact_execution(sample.execution, patterns=patterns)
    if redacted_execution is not sample.execution:
        updates["execution"] = redacted_execution
    if sample.grade is not None:
        redacted_grade = _redact_grade(sample.grade, evidence_keys=evidence_keys, patterns=patterns)
        if redacted_grade is not sample.grade:
            updates["grade"] = redacted_grade
    if not updates:
        return sample
    return sample.model_copy(update=updates)


def apply_redaction(run: EvalRunResult, policy: RedactionPolicy) -> EvalRunResult:
    """Return a new :class:`EvalRunResult` with ``policy`` applied.

    Covers both the harness's own grade evidence (key removal via
    ``evidence_keys``, plus pattern substitution) and each execution's raw
    ``output``/``structured_output``/``error`` fields -- the system-under-
    test's own words, pattern-substituted only, since there is no harness-
    defined key list for content the harness didn't author.

    ``run`` is never mutated (design ADR-0002); this always returns a fresh
    model built with ``model_copy``, even when the policy removes nothing.
    """
    evidence_keys = frozenset(policy.evidence_keys)
    patterns = tuple(re.compile(pattern) for pattern in policy.secret_patterns)
    if not evidence_keys and not patterns:
        return run.model_copy(deep=True)
    redacted_samples = tuple(
        _redact_sample(sample, evidence_keys=evidence_keys, patterns=patterns)
        for sample in run.samples
    )
    return run.model_copy(update={"samples": redacted_samples})


__all__ = ["DEFAULT_REDACTION_POLICY", "RedactionPolicy", "Reporter", "apply_redaction"]
