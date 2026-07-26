"""Extract real source files produced by ARP's own fullstack_generation runs into eval cases.

Every case is a genuine artifact from a past ARP workflow run -- nothing synthetic.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
from collections import defaultdict

RUN_DIR = r"C:\Users\tandf\source\agentic-runtime-platform\runs\default"
OUT = (
    r"C:\Users\tandf\AppData\Local\Temp\claude"
    r"\C--Users-tandf-source-agentic-runtime-platform"
    r"\34357779-797c-455e-874a-f2791d3aff35\scratchpad\cases_raw.jsonl"
)

FILE_SPLIT = re.compile(r"^FILE:\s*(?P<path>\S+)\s*$", re.MULTILINE)

MIN_CHARS = 400
MAX_CHARS = 5000

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


def iter_blobs():
    """Yield every string field from run inputs/outputs that may hold generated code."""
    for path in sorted(glob.glob(RUN_DIR + r"\*.json")):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001,S112
            continue
        if data.get("workflow_name") != "fullstack_generation":
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
                    yield value


def split_files(blob: str) -> list[tuple[str, str]]:
    """Split a 'FILE: path\\n<content>' bundle into (path, content) pairs."""
    matches = list(FILE_SPLIT.finditer(blob))
    out = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(blob)
        out.append((match.group("path"), blob[start:end].strip()))
    return out


def main() -> None:
    seen: set[str] = set()
    by_lang: dict[str, list[dict[str, object]]] = defaultdict(list)

    for blob in iter_blobs():
        for path, content in split_files(blob):
            if not (MIN_CHARS <= len(content) <= MAX_CHARS):
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
            lang = EXT_LANG.get(ext.lower())
            if lang is None or lang in ("json", "msbuild", "yaml"):
                continue  # keep it to real source code
            by_lang[lang].append(
                {
                    "file_path": path,
                    "language": lang,
                    "content": content,
                    "sha256": digest[:16],
                    "chars": len(content),
                }
            )

    print("=== extracted distinct real source files by language ===")
    for lang, items in sorted(by_lang.items(), key=lambda kv: -len(kv[1])):
        print(f"{len(items):5}  {lang}")
    total = sum(len(v) for v in by_lang.values())
    print(f"total: {total}")

    # Round-robin across languages for a diverse suite, largest-first within a language.
    ordered = {k: sorted(v, key=lambda x: -int(x["chars"])) for k, v in by_lang.items()}
    picked: list[dict[str, object]] = []
    index = 0
    while len(picked) < 48:
        added = False
        for lang in sorted(ordered):
            if index < len(ordered[lang]):
                picked.append(ordered[lang][index])
                added = True
                if len(picked) >= 48:
                    break
        if not added:
            break
        index += 1

    with open(OUT, "w", encoding="utf-8") as handle:
        for number, case in enumerate(picked, start=1):
            case["case_id"] = f"arp-fsgen-{number:03d}"
            handle.write(json.dumps(case) + "\n")

    print(f"\nwrote {len(picked)} cases -> {OUT}")
    for case in picked[:10]:
        print(f"  {case['case_id']} {case['language']:18} {case['chars']:5}  {case['file_path']}")


if __name__ == "__main__":
    main()
