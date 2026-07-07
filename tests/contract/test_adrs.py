"""ADR shape and cross-ADR consistency (plan Task 15 Step 1, design §16).

This module *verifies* the nine ADRs already committed under
`docs/adr/0001-standalone-boundary.md` through
`docs/adr/0009-optional-dependencies-and-plugins.md`; it never edits their
content. Every ADR conforms to the required seven-heading template (Status,
Context, Decision, Alternatives, Consequences, Validation, Supersession) and
is `Accepted` -- Tasks 1-10 recorded each ADR before its governing production
code, per the plan's "Every ADR is committed before the production code it
governs" table. If any check here fails, that is a real documentation defect
to report, not something this test should be weakened to tolerate.
"""

import re
from pathlib import Path

import pytest

ADR_DIR = Path("docs/adr")

#: Every ratified ADR this suite enforces, in filename-prefix order: the
#: nine ADRs design §16 and the plan's ADR/task table require (Task 1-10
#: committed them with these exact numeric prefixes), plus 0010 (offline
#: dataset contract). Each prefix matches exactly one file under docs/adr/.
REQUIRED_ADR_PREFIXES = (
    "0001",
    "0002",
    "0003",
    "0004",
    "0005",
    "0006",
    "0007",
    "0008",
    "0009",
    "0010",
)

#: The six section headings every ADR must contain beyond its "## Status"
#: line, matching design §16: "Each ADR records context, decision,
#: alternatives, consequences, validation evidence, and supersession
#: policy."
REQUIRED_HEADINGS = (
    "## Context",
    "## Decision",
    "## Alternatives",
    "## Consequences",
    "## Validation",
    "## Supersession",
)

#: Decisions this package's ADRs must never contradict: the ADR-0001
#: dependency boundary (no ARP/agentic-tools/EK imports; evaluation only
#: through public targets) and the ADR-0003 Hugging Face baseline (shipped
#: in the base wheel, not an optional extra; remote code disabled).
_DEPENDENCY_BOUNDARY_MARKERS = (
    "imports no modules from ARP",
    "ExecutionTarget",
)
_HUGGING_FACE_BASELINE_MARKERS = (
    "base install ships two built-in dataset providers",
    "not behind an optional extra",
)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including line-wrap newlines) to one space.

    ADR prose wraps markers like "not behind an\\n  optional extra" across
    lines; comparing against the normalized text lets a marker match
    regardless of where Markdown line-wrapping happens to fall.
    """
    return re.sub(r"\s+", " ", text)


#: Phrases that would directly contradict the dependency-boundary or
#: Hugging-Face-baseline decisions if found in any ADR body. None of these
#: may appear anywhere in the nine ADRs.
_CONTRADICTING_PHRASES = (
    "may import agentic_v2",
    "may import executionkit",
    "may import tools.agents",
    "huggingface is an optional extra",
    "huggingface support requires an extra",
    "trust_remote_code=true",
)


def _find_adr_path(prefix: str) -> Path:
    matches = sorted(ADR_DIR.glob(f"{prefix}-*.md"))
    assert len(matches) == 1, (
        f"expected exactly one ADR file with prefix {prefix!r} under {ADR_DIR}, found {matches}"
    )
    return matches[0]


@pytest.fixture(scope="module")
def adr_paths() -> dict[str, Path]:
    """Map each required ADR prefix to its resolved file path."""
    return {prefix: _find_adr_path(prefix) for prefix in REQUIRED_ADR_PREFIXES}


@pytest.mark.parametrize("prefix", REQUIRED_ADR_PREFIXES)
def test_adr_file_exists(prefix: str, adr_paths: dict[str, Path]) -> None:
    assert adr_paths[prefix].is_file()


@pytest.mark.parametrize("prefix", REQUIRED_ADR_PREFIXES)
def test_adr_status_is_accepted(prefix: str, adr_paths: dict[str, Path]) -> None:
    text = adr_paths[prefix].read_text(encoding="utf-8")
    # "## Status" is followed (after a blank line) by exactly the word
    # "Accepted" on its own line, per every existing ADR's structure.
    status_match = re.search(r"^## Status\s*\n+\s*(\S+)", text, flags=re.MULTILINE)
    assert status_match is not None, f"{adr_paths[prefix]} has no '## Status' section"
    assert status_match.group(1) == "Accepted", (
        f"{adr_paths[prefix]} status is {status_match.group(1)!r}, expected 'Accepted'"
    )


@pytest.mark.parametrize("prefix", REQUIRED_ADR_PREFIXES)
def test_adr_has_all_required_headings(prefix: str, adr_paths: dict[str, Path]) -> None:
    text = adr_paths[prefix].read_text(encoding="utf-8")
    missing = [heading for heading in REQUIRED_HEADINGS if heading not in text]
    assert missing == [], f"{adr_paths[prefix]} is missing headings: {missing}"


@pytest.mark.parametrize("prefix", REQUIRED_ADR_PREFIXES)
def test_adr_headings_appear_in_canonical_order(prefix: str, adr_paths: dict[str, Path]) -> None:
    """Context/Decision/Alternatives/Consequences/Validation/Supersession, in that order."""
    text = adr_paths[prefix].read_text(encoding="utf-8")
    positions = [text.index(heading) for heading in REQUIRED_HEADINGS]
    assert positions == sorted(positions), (
        f"{adr_paths[prefix]} headings are not in the canonical order {REQUIRED_HEADINGS}"
    )


def test_no_adr_contradicts_dependency_or_baseline_decisions(
    adr_paths: dict[str, Path],
) -> None:
    """None of the nine ADRs may contain text contradicting ADR-0001/ADR-0003."""
    violations: list[str] = []
    for prefix, path in adr_paths.items():
        lowered = _normalize_whitespace(path.read_text(encoding="utf-8").lower())
        for phrase in _CONTRADICTING_PHRASES:
            if phrase in lowered:
                violations.append(f"{path} (ADR {prefix}) contains contradicting phrase {phrase!r}")
    assert violations == []


def test_adr_0001_states_the_dependency_boundary(adr_paths: dict[str, Path]) -> None:
    text = _normalize_whitespace(adr_paths["0001"].read_text(encoding="utf-8"))
    missing = [marker for marker in _DEPENDENCY_BOUNDARY_MARKERS if marker not in text]
    assert missing == [], f"docs/adr/0001-standalone-boundary.md is missing markers: {missing}"


def test_adr_0003_states_the_hugging_face_baseline(adr_paths: dict[str, Path]) -> None:
    text = _normalize_whitespace(adr_paths["0003"].read_text(encoding="utf-8"))
    missing = [marker for marker in _HUGGING_FACE_BASELINE_MARKERS if marker not in text]
    assert missing == [], (
        f"docs/adr/0003-provider-plugins-and-hugging-face-baseline.md is missing markers: {missing}"
    )
