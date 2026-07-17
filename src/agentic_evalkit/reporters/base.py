"""Shared contract for every report format, plus the rules for hiding secrets in a report.

A reporter's only job is to take a finished evaluation run (an
:class:`~agentic_evalkit.models.EvalRunResult` -- the full record of what
happened during one evaluation) and write it out as a file. A reporter never
runs the system under test, never decides whether a sample passed or failed,
and never computes summary statistics -- all of that already happened before
a reporter ever sees the data (see design doc section 11.3). Secret-scrubbing
("redaction") happens exactly once, producing a fresh copy of the data rather
than editing it in place, before any reporter runs -- so every output format
(JSON, HTML, etc.) starts from the same already-cleaned data and none of them
can accidentally leak something that should have been hidden (see design doc
section 12).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agentic_evalkit.models.base import FrozenModel

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from agentic_evalkit.models import EvalRunResult, GradeResult, SampleResult
    from agentic_evalkit.models.execution import NormalizedExecutionResult

_REDACTED = "[REDACTED]"


class RedactionPolicy(FrozenModel):
    """Declares what must never appear in a rendered report.

    ``evidence_keys`` lists specific keys inside a grade's "evidence" data
    (the extra details a grader records to justify its verdict -- things
    like reasoning text or retrieved snippets) that should be deleted
    outright. ``secret_patterns`` are regular expressions describing what a
    secret looks like; any matching substring found inside a piece of
    evidence text is replaced with the literal text ``"[REDACTED]"`` instead
    of deleting the whole value, so the rest of the surrounding text stays
    readable.
    """

    evidence_keys: tuple[str, ...] = ()
    secret_patterns: tuple[str, ...] = ()


#: The safe-by-default policy used at the two places the command-line tool
#: writes a report (the ``run`` command writes the main JSON report through
#: it, and the ``report`` command re-applies it before turning that JSON into
#: another format). Its patterns only catch well-known secret formats --
#: Hugging Face access tokens (which start with "hf_"), OpenAI-style secret
#: keys (which start with "sk-"), and HTTP "Authorization: Bearer ..."
#: header values -- and each pattern requires a minimum length, so an
#: ordinary word that merely happens to start with "hf_" or "sk-" is never
#: mistaken for a real secret and mangled. This default only applies to the
#: CLI: if you use this library directly in your own code, no redaction
#: happens automatically -- reporters never apply a policy on their own, so
#: you must call :func:`apply_redaction` yourself with whatever policy you
#: want (this default one, or your own).
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
    """The shared interface every report format implements: turn a finished run into a written file.

    A reporter must not modify the ``run`` object it's given -- it only
    reads from it. Callers may also hand in ``aggregates``: pre-computed
    summary statistics (for example, a pass rate together with a margin of
    error showing how much to trust it) that were calculated separately by
    the ``agentic_evalkit.stats`` module. This package deliberately never
    imports ``agentic_evalkit.stats`` itself -- that computation is kept out
    of the reporters entirely, so a caller who wants those numbers in the
    report has to compute them first and pass them in.
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
    """Scrub secret-looking text out of the raw output produced by the system being evaluated.

    Grade evidence (see ``_redact_grade`` above) is data our own harness
    generates, so we can define a fixed list of keys to strip out entirely
    (``evidence_keys``). The ``output``, ``structured_output``, and ``error``
    fields handled here are different: they're free-form text written by
    whatever system is under test, not by us, so there's no fixed set of
    keys to drop -- all we can do is scan the text for secret-shaped
    patterns and blank out any matches.

    This function matters even though very large outputs get special
    handling elsewhere: an output too big to keep inline gets written out to
    its own separate file instead (see ``EvalRunner._spill_large_output``),
    and that separate step does its own redaction of the bytes it moves out.
    But an ordinary, small output that's never moved out that way would
    otherwise reach the final report exactly as the tested system produced
    it, secrets and all -- this function is what redacts those
    normal-sized outputs before they're written.
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
    """Return a new, redacted copy of ``run`` with ``policy`` applied.

    Two kinds of data get cleaned, for the two reasons explained on
    ``_redact_grade`` and ``_redact_execution`` above: the harness's own
    grade evidence (where whole keys can be dropped via ``evidence_keys``,
    plus pattern-based scrubbing), and each sample's raw
    ``output``/``structured_output``/``error`` fields -- the tested
    system's own words, which only get pattern-based scrubbing since
    there's no fixed list of keys to drop from free-form text.

    ``run`` itself is left completely untouched (per ADR-0002, which
    requires every model in this codebase to be treated as read-only once
    created, never modified in place). This function always returns a
    brand-new object built with ``model_copy``, even in the edge case where
    the policy doesn't actually redact anything.
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
