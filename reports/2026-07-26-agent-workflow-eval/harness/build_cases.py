"""Extract real source files produced by ARP's own fullstack_generation runs into eval cases.

Every case is a genuine artifact from a past ARP workflow run -- nothing synthetic.

Corpus hygiene
--------------
ARP's run logs do not store bare source files. Generated code is wrapped in the
sentinel bundle format ``FILE: <path>\\n<content>\\nENDFILE`` (the format parsed
by ``agentic_v2/workflows/artifact_extractor.py::_FILE_BLOCK_RE``, which is the
authority on those delimiters), and the run logger truncates any string value
over 10,000 characters, appending the literal marker ``... (<n> chars)``
(``agentic_v2/workflows/run_logger.py::_truncate``).

Both are log artifacts, not source code. Handing them to a reviewer measures the
harness -- reviewers report the synthetic truncation as a genuine defect -- so
this builder strips the framing and **rejects** any payload the log has already
lost data from. Rejections are counted and printed; nothing is silently dropped,
and nothing truncated is repaired. A final validation pass re-checks every
selected case and fails the build if any delimiter or truncation marker survived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

#: Mirrors ``agentic_v2/workflows/artifact_extractor.py::_FILE_BLOCK_RE`` --
#: ARP's own parser for this bundle format, and the authority on the delimiters.
#: Kept as a local mirror rather than a private cross-repo import; the trailing
#: ``ENDFILE`` is required, so an unterminated block simply does not match and
#: is counted as a rejection below.
FILE_BLOCK = re.compile(
    r"^FILE:\s*(?P<path>[^\r\n]+)\r?\n(?P<content>.*?)^ENDFILE\s*$",
    re.MULTILINE | re.DOTALL,
)

#: Any surviving bundle delimiter in stored content means the framing was not
#: stripped cleanly (or the payload holds more than one file).
BUNDLE_MARKER = re.compile(r"^(?:FILE:|ENDFILE\s*$)", re.MULTILINE)

#: The run logger's display-truncation marker: ``value[:10000] + f"... ({n} chars)"``.
TRUNCATION_MARKER = re.compile(r"\.\.\.\s*\(\d+\s*chars\)")

MIN_CHARS = 400
MAX_CHARS = 5000
TARGET_CASES = 48

EXT_LANG = {
    ".cs": "csharp",
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript-react",
    ".js": "javascript",
    ".jsx": "javascript-react",
    ".csproj": "msbuild",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sql": "sql",
}

#: Languages kept in the suite -- real source code only.
SKIP_LANGS = frozenset({"json", "msbuild", "yaml"})


class ContaminatedCaseError(RuntimeError):
    """Raised when a case that reached the suite still carries log framing."""


def iter_blobs(run_dir: Path) -> list[str]:
    """Return every string field from run inputs/outputs that may hold generated code."""
    blobs: list[str] = []
    for path in sorted(run_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or data.get("workflow_name") != "fullstack_generation":
            continue
        containers = []
        for step in data.get("steps") or []:
            if isinstance(step, dict):
                for key in ("input", "output"):
                    if isinstance(step.get(key), dict):
                        containers.append(step[key])
        if isinstance(data.get("final_output"), dict):
            containers.append(data["final_output"])
        for container in containers:
            for value in container.values():
                if isinstance(value, str) and "FILE:" in value:
                    blobs.append(value)
    return blobs


def split_files(blob: str) -> tuple[list[tuple[str, str]], int]:
    """Split a bundle into ``(path, content)`` pairs with the framing removed.

    Returns the parsed blocks and the count of ``FILE:`` headers that had no
    matching ``ENDFILE`` -- those are payloads the log lost data from and are
    reported as rejections rather than repaired.
    """
    blocks = [
        (match.group("path").strip(), match.group("content").strip())
        for match in FILE_BLOCK.finditer(blob)
    ]
    headers = len(re.findall(r"^FILE:", blob, re.MULTILINE))
    return blocks, max(0, headers - len(blocks))


def classify(path: str, content: str) -> str | None:
    """Return a rejection reason for this candidate, or ``None`` when it is usable."""
    if TRUNCATION_MARKER.search(content):
        return "display-truncated by the run logger"
    if BUNDLE_MARKER.search(content):
        return "bundle framing survived (multi-file payload)"
    if not (MIN_CHARS <= len(content) <= MAX_CHARS):
        return f"outside the {MIN_CHARS}-{MAX_CHARS} char window"
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
    lang = EXT_LANG.get(ext.lower())
    if lang is None:
        return "unrecognized file extension"
    if lang in SKIP_LANGS:
        return f"not source code ({lang})"
    return None


def validate(cases: list[dict[str, object]]) -> None:
    """Fail the build if any selected case still carries log framing.

    Belt-and-braces against a future extraction change reintroducing the
    contamination: a rejected payload must never be able to reach the suite.
    """
    for case in cases:
        content = str(case["content"])
        case_id = case.get("case_id", case["file_path"])
        if TRUNCATION_MARKER.search(content):
            raise ContaminatedCaseError(f"{case_id}: truncation marker in stored content")
        if BUNDLE_MARKER.search(content):
            raise ContaminatedCaseError(f"{case_id}: bundle delimiter in stored content")


def collect(run_dir: Path) -> tuple[dict[str, list[dict[str, object]]], Counter[str], int]:
    """Extract, clean, and bucket candidate files by language."""
    seen: set[str] = set()
    by_lang: dict[str, list[dict[str, object]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    candidates = 0

    for blob in iter_blobs(run_dir):
        blocks, unterminated = split_files(blob)
        candidates += len(blocks) + unterminated
        if unterminated:
            rejected["unterminated block (no ENDFILE)"] += unterminated
        for path, content in blocks:
            reason = classify(path, content)
            if reason is not None:
                rejected[reason] += 1
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if digest in seen:
                rejected["duplicate content"] += 1
                continue
            seen.add(digest)
            ext = "." + path.rsplit(".", 1)[-1]
            by_lang[EXT_LANG[ext.lower()]].append(
                {
                    "file_path": path,
                    "language": EXT_LANG[ext.lower()],
                    "content": content,
                    "sha256": digest[:16],
                    "chars": len(content),
                }
            )
    return by_lang, rejected, candidates


def select(by_lang: dict[str, list[dict[str, object]]], target: int) -> list[dict[str, object]]:
    """Round-robin across languages for a diverse suite, largest-first within a language.

    The suite is whatever survives extraction: if fewer than *target* clean
    files exist, the smaller suite is returned rather than backfilled.
    """
    ordered = {k: sorted(v, key=lambda x: -int(x["chars"])) for k, v in by_lang.items()}
    picked: list[dict[str, object]] = []
    index = 0
    while len(picked) < target:
        added = False
        for lang in sorted(ordered):
            if index < len(ordered[lang]):
                picked.append(ordered[lang][index])
                added = True
                if len(picked) >= target:
                    break
        if not added:
            break
        index += 1
    return picked


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arp-root",
        type=Path,
        required=True,
        help="Path to the agentic-runtime-platform checkout.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Workflow run logs. Defaults to <arp-root>/runs/default.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--target-cases", type=int, default=TARGET_CASES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir or (args.arp_root / "runs" / "default")
    if not run_dir.is_dir():
        print(f"run dir not found: {run_dir}")
        return 2

    by_lang, rejected, candidates = collect(run_dir)
    kept = sum(len(v) for v in by_lang.values())

    print("=== extraction ===")
    print(f"candidate FILE blocks found : {candidates}")
    print(f"accepted (clean, distinct)  : {kept}")
    print(f"rejected                    : {sum(rejected.values())}")
    for reason, count in sorted(rejected.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:5}  {reason}")

    print("\n=== clean source files by language ===")
    for lang, items in sorted(by_lang.items(), key=lambda kv: -len(kv[1])):
        print(f"{len(items):5}  {lang}")

    picked = select(by_lang, args.target_cases)
    for number, case in enumerate(picked, start=1):
        case["case_id"] = f"arp-fsgen-{number:03d}"
    validate(picked)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for case in picked:
            handle.write(json.dumps(case) + "\n")

    suite_langs = Counter(str(case["language"]) for case in picked)
    print(f"\n=== suite ({len(picked)} cases; target was {args.target_cases}) ===")
    for lang, count in sorted(suite_langs.items()):
        print(f"{count:5}  {lang}")
    print(f"\nwrote {len(picked)} cases -> {args.out}")
    for case in picked[:10]:
        print(f"  {case['case_id']} {case['language']:18} {case['chars']:5}  {case['file_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
