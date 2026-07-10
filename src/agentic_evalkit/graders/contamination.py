"""Contamination tripwire helpers (ADR-0013, C9).

Pure, deterministic, stdlib-only canary-leak detection any grader can call.
Policy-free by design: these functions report evidence, they never decide a
``GradeStatus`` or ``hard_gate`` -- matching ``CompositeGrader``'s separation
of "component reports evidence" from "composing grader applies policy".

Matching is **normalization-insensitive** (Unicode NFC, whitespace collapse,
case fold) -- the same containment semantics
:mod:`agentic_evalkit.graders.grounding` uses for quote faithfulness and its
canary check, so the ecosystem carries exactly one tripwire semantics and a
case-mangled canary echo can never evade detection (adversarial review
finding, 2026-07-09).
"""

import re
import unicodedata
from collections.abc import Sequence

from pydantic import JsonValue

__all__ = ["canary_leak_evidence", "find_canary_leaks", "normalize_for_containment"]

_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_for_containment(text: str) -> str:
    """NFC-normalize, collapse whitespace, and case-fold for substring checks.

    Mirrors ``exact._canonicalize``'s Unicode/whitespace/case steps but not
    its numeric-shape rewrite: that rewrite is an equality rule
    ("5.0" == "5"), and applying it inside substring containment would
    corrupt text containing numbers.
    """
    normalized = unicodedata.normalize("NFC", text)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip().casefold()


def find_canary_leaks(text: str, canary_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the subset of ``canary_ids`` that appear in ``text``.

    Deterministic, no LLM, no network; never raises. Empty ``text`` or empty
    ``canary_ids`` returns ``()``. Matching is normalization-insensitive
    containment (see :func:`normalize_for_containment`); returned tokens are
    the caller's original spellings, in ``canary_ids`` order, deduplicated.
    """
    if not text or not canary_ids:
        return ()
    haystack = normalize_for_containment(text)
    if not haystack:
        return ()
    seen: set[str] = set()
    leaked: list[str] = []
    for token in canary_ids:
        if token in seen:
            continue
        seen.add(token)
        normalized_token = normalize_for_containment(token)
        if normalized_token and normalized_token in haystack:
            leaked.append(token)
    return tuple(leaked)


def canary_leak_evidence(leaked: Sequence[str]) -> dict[str, JsonValue]:
    """Standard ``GradeResult.evidence``-shaped payload for a canary check.

    A fixed shape any grader can merge into its own evidence dict, so leak
    reports stay machine-readable and uniform across graders instead of each
    grader inventing its own:
    ``{"canary_check": "leaked" | "clean", "leaked_canary_ids": [...]}``.
    """
    return {
        "canary_check": "leaked" if leaked else "clean",
        "leaked_canary_ids": list(leaked),
    }
