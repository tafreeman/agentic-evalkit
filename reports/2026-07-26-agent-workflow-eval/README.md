# ARP agent-workflow eval — 2026-07-26

A real evaluation run: 48 code-review tasks executed against a reviewer-prompted
model routed through `agentic-runtime-platform`'s (ARP) model layer and graded
with ARP's own YAML rubric, driven by `agentic-evalkit`'s `EvalRunner`. Every
number on this page came out of the run recorded in
`9895486e028d44ef9bbaa9df912b0e9c.json`.

> **Amended 2026-07-26 after post-publication review.** Caveats 6–8 below were
> added and the system-under-test description was tightened in response to
> external review of PR #29. The harness and run outputs are unchanged — they
> are the pinned evidence exactly as run.

## Headline numbers

| Metric | Value |
| --- | --- |
| Cases | 48 |
| Executions completed | 48 / 48 |
| Passed | 48 (100%) |
| Failed / errors / timeouts / abstains | 0 / 0 / 0 / 0 |
| Mean weighted score | 0.9648 (median 1.00, min 0.70, max 1.00) |
| Pass threshold | 0.70 (from `agent.yaml`) |
| Wall-clock | 1026.88 s (17 m 07 s) |
| Target tokens | 36,648 in / 212,174 out |
| Judge tokens | 98,182 in / 35,703 out |
| Total tokens | 382,707 |
| Cost | Not measured — see [Cost](#cost) |
| Per-case latency | mean 46.9 s, p50 37.5 s, p95 97.2 s (min 9.6 s, max 221.2 s) |

Concurrency 4, 1 attempt per sample, 300 s per-sample timeout.

## What was evaluated

**System under test:** ARP's target model under a reviewer system prompt,
resolved and invoked through ARP's own LangChain model layer
(`agentic_v2.langchain.models.get_chat_model`). The system prompt mirrors the
Reviewer Agent's role ("expert at code review and security analysis") but is
defined in the harness — the run does **not** construct ARP's Reviewer Agent,
workflow graph, or tool wiring (see caveat 6). What these numbers characterize
is the prompted base model on ARP's routing path, not the full agent stack.

**Task:** given one real source file, produce a code review identifying
correctness bugs, missing functionality, code-quality problems, and security
vulnerabilities.

**Cases:** 48 source files extracted from ARP's own `runs/default/` history —
specifically the outputs of past `fullstack_generation` workflow runs. 257
distinct files were recovered; 48 were selected round-robin across languages to
keep the suite diverse. No case was invented for this eval.

| Language | Cases |
| --- | ---: |
| C# | 12 |
| Python | 12 |
| TypeScript | 11 |
| TypeScript (React) | 11 |
| SQL | 2 |

48 distinct content hashes across 28 distinct file paths — repeated paths are
different generations of the same file from different runs. The exact suite is
`cases.jsonl`.

## How it was graded

ARP's **`agent.yaml`** rubric
(`agentic-v2-eval/src/agentic_v2_eval/rubrics/agent.yaml`, v1.0.0), used
unmodified. Six weighted criteria scored 0–5 by an LLM judge; the weighted score
is normalized to 0–1 and passes at ≥ 0.70.

| Criterion | Weight | Mean score | Scored ≥ 4/5 |
| --- | ---: | ---: | ---: |
| Correctness | 0.25 | 4.708 | 46 / 48 |
| Completeness | 0.20 | 4.562 | 45 / 48 |
| Clarity | 0.15 | 4.938 | 47 / 48 |
| Relevance | 0.15 | 5.000 | 48 / 48 |
| Efficiency | 0.10 | 4.938 | 47 / 48 |
| Safety | 0.15 | 5.000 | 48 / 48 |

Every one of the 48 samples received a score on all six criteria — no criterion
was skipped or defaulted.

## Models

| Role | Model |
| --- | --- |
| Agent under test | `nvidia:nvidia/nemotron-3-super-120b-a12b` |
| Judge | `nvidia:deepseek-ai/deepseek-v4-flash` |

The judge is deliberately a different model family from the target, so the agent
never grades its own output.

## Cost

**Not measured, and deliberately not reported as $0.** Both models run on
NVIDIA's hosted API, which exposes no per-token price in its responses, so
`cost_usd` is recorded as `null` throughout the canonical JSON rather than
carrying an invented figure. Token counts above are exact, taken from each
provider response's `usage_metadata`.

An earlier attempt at this suite used OpenRouter `:free` models, which are
genuinely $0 — but that tier is capped at 50 requests/day and was exhausted
mid-run (see below), so it could not produce a complete run.

## Honest caveats

These matter for reading the 100% pass rate correctly.

1. **The grader is a single uncalibrated LLM judge, and it is noisy.** In an
   earlier NVIDIA run of this same suite, case `arp-fsgen-015` scored **0.66
   (fail)**; in the recorded run the same case with the same target model scored
   **1.00**. One judge, one attempt, no calibration — treat individual case
   scores as indicative, not authoritative.
2. **Scores cluster at the ceiling.** Median 1.00 and only two cases below 0.80
   means this rubric/judge pairing is not very discriminative at the top end. It
   demonstrates the pipeline works end to end; it does not finely rank agent
   quality.
3. **The pass threshold is the rubric's own 0.70**, not a bar chosen to flatter
   the result.
4. **This grades review quality, not review correctness against ground truth.**
   No human verified that the issues the agent reported are real. The judge
   assesses the review against the rubric, and it sees the source file.
5. **Two earlier runs were discarded, not committed** — one 24/48 (OpenRouter
   daily quota exhausted), one 44/48 (transient NVIDIA 503s). Only the complete
   48/48 run is published here. The transient failures were fixed with bounded
   retry (4 attempts, linear backoff) on 429/503, which is in the harness.
6. **The system under test is a prompted model, not ARP's full Reviewer
   Agent.** The harness sends a locally-defined reviewer system prompt to the
   model resolved by ARP's `get_chat_model`; it never constructs ARP's agent
   configuration, workflow graph, or tools. The pass rate supports claims about
   the reviewer-prompted base model on ARP's routing path — not about the
   Reviewer Agent as deployed inside ARP workflows.
7. **The judge graded truncated inputs on a third of the suite.** The judge
   prompt slices sources to 4,000 chars and reviews to 6,000
   (`arp_eval_harness.py`, `_prompt`): 15 of 48 source files exceed the source
   slice, and one review (`arp-fsgen-041`) exceeds the review slice. Several
   recorded rationales penalize "truncated" inputs — those deductions measure
   harness truncation, not review quality. Scores on the 15 affected cases
   should be read with that bias in mind.
8. **The grader renormalizes over present criteria instead of abstaining.** If
   the judge returns valid JSON but omits a criterion, the weighted score is
   computed over the criteria present and a PASS/FAIL is still emitted
   (`missing_criteria` is recorded in evidence but does not gate). **This did
   not affect the published run** — all 48 samples have zero missing criteria,
   verified from the canonical JSON — but a re-run with a different judge could
   silently inflate scores through this path. A fixed harness should abstain
   when any criterion is missing.

## Reproducing

`agentic-evalkit`'s CLI has a closed component registry — `_KNOWN_ADAPTERS` and
`_KNOWN_GRADERS` in `cli/runs.py` resolve only the gsm8k / grounded-citation /
swebench components — so `agentic-evalkit run` cannot name a custom ARP adapter
or a rubric grader. This run therefore drives `EvalRunner` through its documented
Python API, which is exactly how it is designed to be extended: the runner
"never chooses, imports, or constructs any of those components — the caller is
responsible for building them and handing them to the runner."

The canonical JSON is written by evalkit's own `write_canonical_report`, so it is
identical in shape to CLI output, and the HTML/Markdown were generated by the
real CLI:

```bash
agentic-evalkit report 9895486e028d44ef9bbaa9df912b0e9c.json --format html
```

`harness/` holds everything needed to re-run:

| File | Purpose |
| --- | --- |
| `build_cases.py` | Extracts the suite from ARP's `runs/default/` logs |
| `arp_eval_harness.py` | Catalog, adapter, ARP target, rubric grader |
| `run_eval.py` | Builds the manifest and drives `EvalRunner` |

**Before running:** `RUBRIC_PATH` in `harness/arp_eval_harness.py` is an
absolute path pinned to the machine that produced this run. Edit it to point at
your ARP checkout's `agentic-v2-eval/src/agentic_v2_eval/rubrics/agent.yaml`
first — the harness is committed exactly as run, so it is not edited here even
to parameterize this.

Both packages must be importable by one interpreter. ARP's venv plus
`PYTHONPATH` pointing at evalkit's `src/` works without installing anything:

```bash
PYTHONPATH="/path/to/agentic-evalkit/src:harness" \
  /path/to/agentic-runtime-platform/.venv/bin/python harness/run_eval.py \
  --cases cases.jsonl --output-dir . --concurrency 4 --timeout 300
```

## Files

| File | Contents |
| --- | --- |
| `9895486e028d44ef9bbaa9df912b0e9c.json` | Canonical `EvalRunResult` — every sample, review, grade, and provenance record |
| `9895486e028d44ef9bbaa9df912b0e9c.html` | Self-contained HTML report |
| `9895486e028d44ef9bbaa9df912b0e9c.md` | Markdown summary |
| `run-stats.json` | Aggregates quoted above |
| `cases.jsonl` | The 48-case suite |
| `harness/` | The code that produced the run |
