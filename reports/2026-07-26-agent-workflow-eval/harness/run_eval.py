"""Drive the ARP agent-workflow eval through evalkit's real EvalRunner."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.cli.runs import write_canonical_report
from agentic_evalkit.models import DatasetRef, EvalRunManifest
from agentic_evalkit.models.runs import DatasetSelection
from agentic_evalkit.runner import EvalRunner

from arp_eval_harness import (
    JUDGE_MODEL,
    RUBRIC_PATH,
    TARGET_MODEL,
    ArpReviewTarget,
    ArpRubricGrader,
    CodeReviewAdapter,
    JsonlCatalog,
    Rubric,
)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="arp-agent-workflow-eval")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    cases_path = Path(args.cases)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rubric = Rubric(RUBRIC_PATH)
    catalog = JsonlCatalog(cases_path)
    adapter = CodeReviewAdapter()
    target = ArpReviewTarget(TARGET_MODEL)
    grader = ArpRubricGrader(rubric, JUDGE_MODEL)

    total_rows = len(catalog._rows)  # noqa: SLF001 - local driver, not library code
    limit = args.limit if args.limit is not None else total_rows

    manifest = EvalRunManifest(
        run_name=args.run_name,
        dataset_ref=DatasetRef(provider="local", dataset_id=str(cases_path)),
        adapter=adapter.name,
        grader=grader.name,
        target_name="arp-reviewer-agent",
        selection=DatasetSelection(offset=0, limit=limit),
        attempts=1,
        timeout_seconds=args.timeout,
        concurrency=args.concurrency,
    )

    runner = EvalRunner(
        catalog=catalog,
        adapters={adapter.name: adapter},
        targets={"arp-reviewer-agent": target},
        graders={grader.name: grader},
        artifact_store=ArtifactStore(output_dir / "artifacts"),
    )

    print(f"cases={limit} target={TARGET_MODEL} judge={JUDGE_MODEL} concurrency={args.concurrency}")
    wall_start = time.perf_counter()
    result = await runner.run(manifest)
    wall_clock_s = time.perf_counter() - wall_start

    report_path = write_canonical_report(result, output_dir)

    summary = result.summary
    in_tok = sum(s.execution.input_tokens or 0 for s in result.samples)
    out_tok = sum(s.execution.output_tokens or 0 for s in result.samples)
    judge_in = sum(
        int(s.grade.evidence.get("judge_input_tokens") or 0)
        for s in result.samples
        if s.grade is not None
    )
    judge_out = sum(
        int(s.grade.evidence.get("judge_output_tokens") or 0)
        for s in result.samples
        if s.grade is not None
    )
    priced = [s.execution.cost_usd for s in result.samples if s.execution.cost_usd is not None]
    cost = sum(priced) if priced else None

    # Per-criterion tally, straight off the recorded grades.
    criterion_totals: dict[str, list[float]] = {name: [] for name in rubric.names}
    for sample in result.samples:
        if sample.grade is None:
            continue
        for name, value in (sample.grade.evidence.get("criterion_scores") or {}).items():
            criterion_totals.setdefault(str(name), []).append(float(value))

    stats = {
        "run_id": result.run_id,
        "report_path": str(report_path),
        "wall_clock_seconds": round(wall_clock_s, 2),
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "errors": summary.errors,
        "timeouts": summary.timeouts,
        "abstained": summary.abstained,
        "target_input_tokens": in_tok,
        "target_output_tokens": out_tok,
        "judge_input_tokens": judge_in,
        "judge_output_tokens": judge_out,
        "cost_usd": cost,
        "cost_note": (
            "unverified - NVIDIA endpoint exposes no per-token price"
            if cost is None
            else "measured"
        ),
        "target_model": TARGET_MODEL,
        "judge_model": JUDGE_MODEL,
        "criterion_means": {
            name: round(sum(v) / len(v), 3) for name, v in criterion_totals.items() if v
        },
        "criterion_counts": {name: len(v) for name, v in criterion_totals.items()},
        "criterion_pass_at_4plus": {
            name: sum(1 for x in v if x >= 4) for name, v in criterion_totals.items() if v
        },
    }
    (output_dir / "run-stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
