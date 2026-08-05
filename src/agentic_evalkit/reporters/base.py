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
import warnings
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from agentic_evalkit.models.base import FrozenModel
from agentic_evalkit.models.execution import (
    OUTPUT_REF_KEY,
    OUTPUT_SPILL_ERROR_KEY,
    OUTPUT_SPILL_FAILED_CODE,
    is_output_spill_error_record,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import JsonValue

    from agentic_evalkit.models import EvalRunResult, EvalSample, GradeResult, SampleResult
    from agentic_evalkit.models.execution import NormalizedExecutionResult

_REDACTED = "[REDACTED]"

#: Exactly the shape ``ArtifactStore`` mints (``artifacts._digest_of``): the
#: literal ``sha256:`` followed by a 64-character lowercase hex digest. The
#: redaction sweep exempts ``artifacts["output_ref"]`` from rewriting, and
#: that exemption is only safe for a value this harness actually produced --
#: ``artifacts`` is target-controlled, so anything else found under that key
#: is free-form text that must be swept like the rest of the dict.
_OUTPUT_REF_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


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


def _resolve_redaction_policy(policy: RedactionPolicy | None, *, caller: str) -> RedactionPolicy:
    """Normalize the deprecated ``None`` opt-out spelling to ``RedactionPolicy()``.

    ``RedactionPolicy()`` (no patterns) is the one supported way to opt out
    of redaction. ``None`` is still accepted for backward compatibility and
    behaves identically, but warns: deprecated since 0.4.0, support for it
    will be removed in 0.5.0. Every constructor that accepts a policy calls
    this one function so the deprecation window and wording cannot drift
    between call sites. ``stacklevel=3`` points the warning at the code that
    called the constructor, not at the constructor or this helper.
    """
    if policy is not None:
        return policy
    warnings.warn(
        f"{caller}(redaction_policy=None) is deprecated; pass "
        "RedactionPolicy() instead, which is the supported way to opt "
        "out of redaction. None is still accepted for backward "
        "compatibility (deprecated since 0.4.0); support for it will "
        "be removed in 0.5.0.",
        DeprecationWarning,
        stacklevel=3,
    )
    return RedactionPolicy()


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
    """Scrub both free-form dicts a grade carries.

    ``oracle_provenance`` is swept alongside ``evidence`` because it records
    how an authoritative external checker was invoked -- which harness ran,
    and how. That routinely means a command line, an image reference, or a
    registry URL, and any of those can carry a credential. Only
    ``evidence_keys`` drops whole keys; ``oracle_provenance`` gets the
    pattern sweep, since its keys are a machine-readable evidence trail
    rather than free prose to be discarded.
    """
    updates: dict[str, object] = {}
    if grade.evidence:
        redacted_evidence = _redact_evidence(
            grade.evidence, evidence_keys=evidence_keys, patterns=patterns
        )
        if redacted_evidence != grade.evidence:
            updates["evidence"] = redacted_evidence
    if grade.oracle_provenance and patterns:
        redacted_oracle = _redact_json_value(grade.oracle_provenance, patterns)
        if redacted_oracle != grade.oracle_provenance:
            updates["oracle_provenance"] = redacted_oracle
    if not updates:
        return grade
    return grade.model_copy(update=updates)


def _redact_eval_sample(sample: EvalSample, *, patterns: tuple[re.Pattern[str], ...]) -> EvalSample:
    """Scrub the sample itself, not just what the system under test did with it.

    This was the gap that made the sweep incomplete for a long time. Every
    other redaction here works on the *output* side -- what the target
    produced, what the grader concluded -- on the assumption that the input
    side is benign dataset content. It is not always. ``input`` is the
    prompt, and a prompt routinely contains the credential the agent is
    supposed to use, an authorization header to replay, or customer data
    from a production trace turned into a regression case. ``metadata`` and
    ``expected_artifacts`` are adapter-authored and can carry the same.

    ``tags`` is deliberately left alone: it holds short structural labels an
    adapter assigns for filtering, never free-form payload, and rewriting a
    label would break selection without protecting anything.
    """
    if not patterns:
        return sample
    updates: dict[str, object] = {}
    for field_name in ("input", "metadata", "expected_artifacts"):
        value = getattr(sample, field_name)
        if not value:
            continue
        redacted_value = _redact_json_value(value, patterns)
        if redacted_value != value:
            updates[field_name] = redacted_value
    if sample.reference is not None:
        redacted_reference = _redact_string(sample.reference, patterns)
        if redacted_reference != sample.reference:
            updates["reference"] = redacted_reference
    if not updates:
        return sample
    return sample.model_copy(update=updates)


def _is_artifact_digest(value: JsonValue | None) -> bool:
    """Is ``value`` a reference ``ArtifactStore`` minted, rather than target text?

    Only a value of this exact shape earns the ``output_ref`` exemption from
    the redaction sweep (see ``_redact_execution``). ``artifacts`` is
    target-controlled, so the key name alone proves nothing -- exempting
    whatever happens to sit under it would let a target hand a live
    credential through the sweep untouched.
    """
    return isinstance(value, str) and _OUTPUT_REF_DIGEST_RE.match(value) is not None


def _restore_machine_readable_artifacts(
    execution: NormalizedExecutionResult, redacted: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    """Put back the two values in ``artifacts`` that are identifiers, not prose.

    Everything in ``artifacts`` is swept (see ``_redact_execution``), but two
    values there are machine-readable contracts rather than free-form text,
    and rewriting either breaks a consumer rather than protecting anyone:

    * ``output_ref`` -- the only pointer back to bytes already on disk.
      Rewriting one character orphans the artifact permanently.
    * the ``code`` inside an ``output_spill_error`` record -- what every
      consumer matches on via ``is_output_spill_error_record`` to tell a
      genuine spill failure from target-authored data. Redact it and
      ``HarnessGrader`` re-grading that report falls back to the
      "produced no output" ``UNAVAILABLE`` this whole change exists to
      eliminate, and the CLI's dropped-output warning stops firing.

    Neither restoration can smuggle anything, because neither is
    target-supplied text being handed back. The digest is exempt only when it
    matches what ``ArtifactStore`` mints, and the code is not restored from
    the record at all -- it is rewritten to the fixed ``OUTPUT_SPILL_FAILED_CODE``
    literal, and only for a record that already carried it before the sweep.
    Every other key and every other value in both, including the record's
    ``message`` and ``type``, stays redacted.

    The default policy matches neither value, so this changes nothing for the
    shipped CLI. It matters because ``apply_redaction`` is public and takes
    whatever policy a caller hands it: a "long hex string looks like a key"
    rule matches the digest, and a broad identifier rule matches
    ``output_spill_failed``.
    """
    restored = redacted
    if _is_artifact_digest(execution.artifacts.get(OUTPUT_REF_KEY)):
        restored = {**restored, OUTPUT_REF_KEY: execution.artifacts[OUTPUT_REF_KEY]}
    if is_output_spill_error_record(execution.artifacts.get(OUTPUT_SPILL_ERROR_KEY)):
        record = restored.get(OUTPUT_SPILL_ERROR_KEY)
        if isinstance(record, dict):
            restored = {
                **restored,
                OUTPUT_SPILL_ERROR_KEY: {**record, "code": OUTPUT_SPILL_FAILED_CODE},
            }
    return restored


def _redact_execution(
    execution: NormalizedExecutionResult, *, patterns: tuple[re.Pattern[str], ...]
) -> NormalizedExecutionResult:
    """Scrub secret-looking text out of the raw output produced by the system being evaluated.

    Grade evidence (see ``_redact_grade`` above) is data our own harness
    generates, so we can define a fixed list of keys to strip out entirely
    (``evidence_keys``). The ``output``, ``structured_output``, ``error``,
    and ``artifacts`` fields handled here are different: they're free-form
    text written by whatever system is under test, not by us, so there's no
    fixed set of keys to drop -- all we can do is scan the text for
    secret-shaped patterns and blank out any matches.

    ``artifacts`` is swept for the same reason as the rest, even though it
    mostly holds harness-authored values like the ``output_ref`` digest: a
    failed spill records the artifact store's own exception message there
    (``output_spill_error``), and a store that talks to something remote is
    free to echo a URL, a header, or a credential into that text. So the
    whole field goes through rather than a hand-picked list of keys that
    would need extending every time the dict grows -- except for the two
    values in it that are identifiers rather than prose, which
    ``_restore_machine_readable_artifacts`` puts back afterwards. The first is
    below; the second is the ``code`` inside a spill-failure record, which is
    what every consumer matches on to recognise that record at all.

    ``output_ref`` is restored verbatim afterwards -- but only when it holds
    a digest this harness actually minted. It is the *only* pointer back to a
    payload that has already been written to the artifact store, so a pattern
    that rewrites even one character of it orphans those bytes permanently:
    the report then references an artifact nobody can look up, which is a
    worse outcome than the one redaction is guarding against. The default
    policy's patterns cannot match a hex digest, but ``apply_redaction`` is
    public and takes any policy a caller supplies, and a generic "long hex
    string looks like a key" rule is an entirely reasonable thing for someone
    to add.

    The exemption is gated on the value's *shape* rather than on the key
    name, because the key name is not ours to trust. ``artifacts`` is
    target-controlled -- the same premise that makes
    ``is_output_spill_error_record`` necessary one field over -- so a target
    is perfectly free to return ``artifacts={"output_ref": "hf_…"}`` and, if
    the key alone bought the exemption, walk a live credential straight
    through this sweep and into the canonical report under the *default*
    policy the CLI writes every report with. Only a value matching what
    ``ArtifactStore`` mints (``_OUTPUT_REF_DIGEST_RE``) is a real reference;
    anything else is free-form target text and gets swept with everything
    else. A genuine digest is harness-authored and structurally incapable of
    carrying a secret, so exempting *that* gives up nothing.

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
    for field_name in ("output", "structured_output", "error", "artifacts", "environment_metadata"):
        value = getattr(execution, field_name)
        if value is None:
            continue
        redacted_value = _redact_json_value(value, patterns)
        if field_name == "artifacts":
            redacted_value = _restore_machine_readable_artifacts(
                execution, cast("dict[str, JsonValue]", redacted_value)
            )
        if redacted_value != value:
            updates[field_name] = redacted_value
    # tool_calls needs its own pass because it is a tuple of dicts, and
    # _redact_json_value turns every sequence into a list -- assigning that
    # back would change the field's type on a frozen model whose collections
    # are tuples by contract (ADR-0002). It is swept rather than trusted
    # because a tool call records the arguments the target sent to a tool,
    # which is exactly where an API key travels.
    if execution.tool_calls:
        redacted_calls = tuple(
            cast("dict[str, JsonValue]", _redact_json_value(call, patterns))
            for call in execution.tool_calls
        )
        if redacted_calls != execution.tool_calls:
            updates["tool_calls"] = redacted_calls
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
    redacted_sample = _redact_eval_sample(sample.sample, patterns=patterns)
    if redacted_sample is not sample.sample:
        updates["sample"] = redacted_sample
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


def redact_text(value: str, policy: RedactionPolicy) -> str:
    """Apply ``policy``'s secret patterns to one free-standing string.

    :func:`apply_redaction` is the right entry point whenever there is a
    whole :class:`~agentic_evalkit.models.EvalRunResult` to scrub, and it
    remains the only one the reporters use. This exists for the narrower
    case where a single caller-visible string is about to leave the process
    without any run around it -- the rationale
    ``agentic_evalkit.integrations.mlflow.as_mlflow_scorer`` attaches to a
    host-platform feedback object, for example, which is synthesized from a
    grade's evidence rather than copied out of a run.

    ``policy.evidence_keys`` is not consulted, because a bare string has no
    keys to drop; only ``secret_patterns`` applies. Passing a policy with no
    patterns returns ``value`` unchanged, which is the same opt-out
    :class:`RedactionPolicy` gives everywhere else.
    """
    if not policy.secret_patterns:
        return value
    return _redact_string(value, tuple(re.compile(p) for p in policy.secret_patterns))


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


__all__ = [
    "DEFAULT_REDACTION_POLICY",
    "RedactionPolicy",
    "Reporter",
    "apply_redaction",
    "redact_text",
]
