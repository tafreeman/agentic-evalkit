"""ADR shape and cross-ADR consistency (plan Task 15 Step 1, design §16).

This module *verifies* the ADR log already committed under `docs/adr/`
(``REQUIRED_ADR_PREFIXES`` below names each ratified record, and a
completeness check below asserts that tuple matches the committed file
listing); it never edits ADR content. Every ADR conforms to the required
seven-heading template (Status, Context, Decision, Alternatives,
Consequences, Validation, Supersession) and
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
#: dataset contract), 0011 (offline resolution cache), 0012
#: (grounded-citation probe), 0013 (contamination metadata and canaries),
#: 0014 (SWE-bench Docker harness executor), and 0015 (environment/code
#: fingerprints gate comparability). Each prefix matches exactly one file
#: under docs/adr/.
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
    "0011",
    "0012",
    "0013",
    "0014",
    "0015",
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
#: may appear anywhere in any committed ADR.
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
    """No committed ADR may contain text contradicting ADR-0001/ADR-0003."""
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


#: Word-form numbers docs prose may use for the ADR count claim.
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
}


def _committed_adr_prefixes() -> tuple[str, ...]:
    """Numeric prefixes of every ADR file actually committed under docs/adr/."""
    return tuple(sorted(path.name[:4] for path in ADR_DIR.glob("[0-9][0-9][0-9][0-9]-*.md")))


def test_required_prefixes_cover_every_committed_adr_file() -> None:
    """``REQUIRED_ADR_PREFIXES`` cannot lag the ADR log.

    A new docs/adr/ file must be added to the tuple above in the same
    change, or every shape/status/heading check in this module silently
    skips it.
    """
    assert _committed_adr_prefixes() == REQUIRED_ADR_PREFIXES


def test_landing_page_adr_claims_match_committed_adr_count() -> None:
    """docs/index.md's ADR stat tile and prose must track docs/adr/.

    The 2026-07-08 audit found the landing page still claiming nine ADRs
    (stat tile "9", prose "0001 through 0009") after
    0010-offline-dataset-contract.md landed. Both claims are derived here
    from the committed file listing so the undercount cannot recur
    silently. CLAUDE.md repeats the range claim but is gitignored, so a CI
    checkout deliberately cannot assert on it.
    """
    prefixes = _committed_adr_prefixes()
    count = len(prefixes)
    index = Path("docs/index.md").read_text(encoding="utf-8")

    tile = re.search(
        r'<div class="stat-value">(\d+)</div>\s*<div class="stat-label">ADRs</div>',
        index,
    )
    assert tile is not None, "docs/index.md: ADR stat tile not found"
    assert int(tile.group(1)) == count, (
        f"docs/index.md ADR stat tile says {tile.group(1)}; docs/adr/ ships {count}"
    )

    prose = re.search(r"(\w+) architecture decision records, 0001 through (\d{4})", index)
    assert prose is not None, "docs/index.md: ADR-index prose claim not found"
    assert NUMBER_WORDS.get(prose.group(1).lower()) == count, (
        f"docs/index.md prose says {prose.group(1)!r} architecture decision records; "
        f"docs/adr/ ships {count}"
    )
    assert prose.group(2) == prefixes[-1], (
        f"docs/index.md prose says the ADR log ends at {prose.group(2)}; "
        f"the last committed ADR is {prefixes[-1]}"
    )
