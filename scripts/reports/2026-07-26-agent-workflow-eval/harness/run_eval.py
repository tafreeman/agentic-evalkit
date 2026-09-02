"""Drive the ARP agent-workflow eval through evalkit's real EvalRunner."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from arp_eval_harness import (
    JUDGE_MODEL,
    RUBRIC_RELATIVE_PATH,
    TARGET_MODEL,
    ArpReviewTarget,
    ArpRubricGrader,
    CodeReviewAdapter,
    JsonlCatalog,
    Rubric,
    default_arp_root,
    default_rubric_path,
)

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.cli.runs import write_canonical_report
from agentic_evalkit.models import DatasetRef, EvalRunManifest, EvalRunResult
from agentic_evalkit.models.runs import DatasetSelection
from agentic_evalkit.provenance import (
    compute_code_fingerprint,
    compute_environment_fingerprint,
)
from agentic_evalkit.runner import EvalRunner

TARGET_NAME = "arp-reviewer-agent"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="arp-agent-workflow-eval")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--arp-root",
        type=Path,
        default=None,
        help=(
            "Path to the agentic-runtime-platform checkout. Defaults to the "
            "checkout that owns the importable agentic_v2 package."
        ),
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=None,
        help=f"Rubric YAML. Defaults to <arp-root>/{RUBRIC_RELATIVE_PATH.as_posix()}.",
    )
    return parser.parse_args(argv)


def build_stats(
    result: EvalRunResult,
    *,
    rubric: Rubric,
    target: ArpReviewTarget,
    report_path: Path,
    wall_clock_s: float,
) -> dict[str, object]:
    """Aggregate the recorded run into the summary block written to disk."""
    samples = result.samples
    summary = result.summary

    in_tok = sum(s.execution.input_tokens or 0 for s in samples)
    out_tok = sum(s.execution.output_tokens or 0 for s in samples)
    judge_in = sum(
        int(s.grade.evidence.get("judge_input_tokens") or 0) for s in samples if s.grade is not None
    )
    judge_out = sum(
        int(s.grade.evidence.get("judge_output_tokens") or 0)
        for s in samples
        if s.grade is not None
    )
    priced = [s.execution.cost_usd for s in samples if s.execution.cost_usd is not None]
    cost = sum(priced) if priced else None

    # Per-criterion tally, straight off the recorded grades.
    criterion_totals: dict[str, list[float]] = {name: [] for name in rubric.names}
    for sample in samples:
        if sample.grade is None:
            continue
        for name, value in (sample.grade.evidence.get("criterion_scores") or {}).items():
            criterion_totals.setdefault(str(name), []).append(float(value))

    return {
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
        # Proves which ARP agent and which prompt version produced these numbers.
        "target_fingerprint": target.fingerprint,
        "target_provenance": target.provenance,
        "criterion_means": {
            name: round(sum(v) / len(v), 3) for name, v in criterion_totals.items() if v
        },
        "criterion_counts": {name: len(v) for name, v in criterion_totals.items()},
        "criterion_pass_at_4plus": {
            name: sum(1 for x in v if x >= 4) for name, v in criterion_totals.items() if v
        },
    }


async def run(
    manifest: EvalRunManifest,
    *,
    catalog: JsonlCatalog,
    adapter: CodeReviewAdapter,
    target: ArpReviewTarget,
    grader: ArpRubricGrader,
    artifact_dir: Path,
) -> tuple[EvalRunResult, float]:
    """Execute the manifest through evalkit's runner, returning the result and wall clock."""
    runner = EvalRunner(
        catalog=catalog,
        adapters={adapter.name: adapter},
        targets={TARGET_NAME: target},
        graders={grader.name: grader},
        artifact_store=ArtifactStore(artifact_dir),
    )
    wall_start = time.perf_counter()
    result = await runner.run(manifest)
    return result, time.perf_counter() - wall_start


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cases_path = Path(args.cases)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --arp-root has to bind the checkout that is actually imported, not just
    # pick a rubric. Otherwise the run prints one root while ArpReviewTarget and
    # the revision fingerprint resolve whichever agentic_v2 was already on the
    # path -- evaluating checkout A's agent against checkout B's rubric and
    # labelling the result B.
    if args.arp_root is not None:
        requested = args.arp_root.resolve()
        package_parent = requested / "agentic-workflows-v2"
        if not (package_parent / "agentic_v2").is_dir():
            print(f"--arp-root does not contain agentic-workflows-v2/agentic_v2: {requested}")
            return 2
        sys.path.insert(0, str(package_parent))
        resolved = default_arp_root().resolve()
        if resolved != requested:
            print(
                "refusing to run: agentic_v2 resolves to a different checkout than "
                f"--arp-root.\n  --arp-root: {requested}\n  imported:   {resolved}\n"
                "Run from an environment where the requested checkout is importable."
            )
            return 2

    arp_root = args.arp_root or default_arp_root()
    rubric_path = args.rubric or default_rubric_path(arp_root)
    if not rubric_path.is_file():
        print(f"rubric not found: {rubric_path} (pass --arp-root or --rubric)")
        return 2

    rubric = Rubric(rubric_path)
    catalog = JsonlCatalog(cases_path)
    adapter = CodeReviewAdapter()
    target = ArpReviewTarget(TARGET_MODEL)
    grader = ArpRubricGrader(rubric, JUDGE_MODEL)

    limit = args.limit if args.limit is not None else len(catalog)
    manifest = EvalRunManifest(
        run_name=args.run_name,
        dataset_ref=DatasetRef(provider="local", dataset_id=str(cases_path)),
        adapter=adapter.name,
        grader=grader.name,
        target_name=TARGET_NAME,
        selection=DatasetSelection(offset=0, limit=limit),
        attempts=1,
        timeout_seconds=args.timeout,
        concurrency=args.concurrency,
        # The manifest is the field evalkit's own compare_runs reads to decide
        # whether two runs are comparable, and it treats two nulls as a match --
        # so a fingerprint recorded only in run-stats.json or per-execution
        # metadata is invisible to the gate that exists to use it. Pin it here.
        target_fingerprint_policy="required",
        target_fingerprint=target.fingerprint,
        # evalkit's own provenance helpers -- which interpreter/platform and
        # which evalkit build produced the run. Left null, the canonical report
        # silently under-describes what it can be compared against.
        environment_fingerprint=compute_environment_fingerprint(),
        code_fingerprint=compute_code_fingerprint(),
    )

    print(f"arp_root={arp_root}")
    print(f"rubric={rubric_path} ({rubric.rubric_id})")
    print(
        f"agent={target.provenance['arp_agent']} "
        f"prompt={target.provenance['arp_prompt_qualified_version']}"
    )
    print(f"cases={limit} target={TARGET_MODEL} judge={JUDGE_MODEL} concurrency={args.concurrency}")

    result, wall_clock_s = asyncio.run(
        run(
            manifest,
            catalog=catalog,
            adapter=adapter,
            target=target,
            grader=grader,
            artifact_dir=output_dir / "artifacts",
        )
    )

    report_path = write_canonical_report(result, output_dir)
    stats = build_stats(
        result,
        rubric=rubric,
        target=target,
        report_path=report_path,
        wall_clock_s=wall_clock_s,
    )
    (output_dir / "run-stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
