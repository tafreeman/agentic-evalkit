"""Canary-leak detection: a "do-not-say" marker showing up in an AI's answer (ADR-0013, C9).

The idea: plant a unique, made-up string (a "canary token") somewhere the AI
has no legitimate reason to reproduce, and later check whether that exact
string shows up in what the AI actually said. If it does, that's a strong
signal something leaked -- for example, the AI may be repeating a marker it
saw during its own training on this exact question ("dataset contamination":
the test question may have already been in the AI's training data, which
would make the test unfairly easy), or it echoed a token that was only ever
supposed to appear in a part of the source material it wasn't meant to cite
from. This module is pure, deterministic, standard-library-only detection
logic that any grader can call.

These functions are policy-free by design: they only report *whether* a
canary leaked, and never decide what that means for the final
``GradeStatus``, or whether it should force an overall failure
(``hard_gate``). That decision is left to whichever grader calls these
functions -- the same split ``CompositeGrader`` uses between "a component
reports evidence" and "the thing composing the components decides policy".

Matching is **normalization-insensitive**: before comparing, text is run
through the same Unicode-normalize / whitespace-collapse / lowercase steps
that :mod:`agentic_evalkit.graders.grounding` uses when checking whether a
quote is genuine. Sharing this exact cleanup logic means the whole codebase
has one single definition of "this text contains that token" -- so nobody
can dodge detection just by changing the capitalization or spacing of a
leaked canary (a gap closed after a finding from an adversarial review on
2026-07-09).
"""

import re
import unicodedata
from collections.abc import Sequence

from pydantic import JsonValue

__all__ = ["canary_leak_evidence", "find_canary_leaks", "normalize_for_containment"]

_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_for_containment(text: str) -> str:
    """Clean up text so two strings can be compared regardless of formatting differences.

    Three steps: normalize the Unicode encoding to one canonical form (so
    visually identical characters that happen to be encoded differently
    still compare equal), collapse any run of whitespace down to a single
    space, and lowercase everything (using ``casefold``, a more thorough
    version of lowercasing meant for case-insensitive comparisons). This is
    almost the same cleanup ``exact._canonicalize`` does, but deliberately
    skips its number-reformatting step (the one that treats "5.0" and "5" as
    equal). That step is correct when checking whether two whole answers are
    equal, but would be wrong here: this function's result is used to check
    whether one string is a *substring* of another, and rewriting a number
    in the middle of a larger text could corrupt anything else nearby.
    """
    normalized = unicodedata.normalize("NFC", text)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip().casefold()


def find_canary_leaks(text: str, canary_ids: Sequence[str]) -> tuple[str, ...]:
    """Check ``text`` for any planted marker string in ``canary_ids``, and return the ones found.

    This is plain string matching -- no AI model call, no network access --
    and it never raises an exception. An empty ``text`` or an empty
    ``canary_ids`` simply returns ``()`` (nothing found). The comparison
    ignores formatting differences like casing and extra whitespace (see
    :func:`normalize_for_containment`), so a leaked token can't dodge
    detection just by appearing with different capitalization. The strings
    returned are the original spellings the caller passed in (not the
    cleaned-up internal form used for matching), listed in the same order as
    ``canary_ids``, with duplicates removed.
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
    """Package a canary check's result into the standard shape used in ``GradeResult.evidence``.

    Returns a small, fixed-shape dictionary that any grader can merge into
    its own evidence:
    ``{"canary_check": "leaked" | "clean", "leaked_canary_ids": [...]}``.
    Using one shared shape means every grader's leak report looks the same
    and can be read the same way, instead of each grader inventing its own
    format for saying "here's what leaked."
    """
    return {
        "canary_check": "leaked" if leaked else "clean",
        "leaked_canary_ids": list(leaked),
    }
