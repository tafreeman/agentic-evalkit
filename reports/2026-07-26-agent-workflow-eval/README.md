# ARP agent-workflow eval — 2026-07-26

A real evaluation run: 48 code-review tasks executed against
`agentic-runtime-platform`'s (ARP) own reviewer agent and graded with ARP's own
YAML rubric, driven by `agentic-evalkit`'s `EvalRunner`. Every number on this
page came out of the run recorded in
`fb694424348643379922f0bd9c22f348.json`.

> **This run supersedes the invalidated run published at commit `c65f5b1`.**
> External review of that run found five validity defects — the wrong system
> under test, silently truncated judge inputs, a machine-pinned rubric path, a
> grader that renormalized over missing rubric criteria, and a case corpus
> contaminated with ARP's own log framing. All five are fixed in the harness
> under `harness/`, and every number below comes from a full re-run against the
> fixed harness and a rebuilt corpus. **The two runs are not comparable** and no
> figure here should be read as a delta against the old one: decontaminating the
> corpus changed the content of all 48 cases (zero shared content hashes), and
> under ARP's real persona the agent emits a structured artifact block instead of
> the free prose the old harness elicited. The superseded outputs remain in git
> history at `c65f5b1`.

## Headline numbers

| Metric | Value |
| --- | --- |
| Cases | 48 |
| Executions completed | 47 / 48 (1 operational timeout) |
| Graded | 47 |
| Passed / failed | 47 / 0 — of the 47 graded |
| Errors / timeouts / abstains | 0 / 1 / 0 |
| Mean weighted score | 0.9611 (median 1.00, min 0.72, max 1.00) |
| Pass threshold | 0.70 (from `agent.yaml`) |
| Wall-clock | 1574.62 s (26 m 15 s) |
| Target tokens | 131,203 in / 364,072 out |
| Judge tokens | 146,188 in / 43,632 out |
| Total tokens | 685,095 |
| Cost | Not measured — see [Cost](#cost) |
| Per-case latency | mean 76.9 s, p50 64.3 s, p95 173.3 s (min 13.2 s, max 197.0 s) |
| Target fingerprint | `sha256:d4e4a7e8592a8655…` (prompt `reviewer` `v1@f1555690`) |

Concurrency 4, 1 attempt per sample, 300 s per-sample timeout.

**The pass rate is 47/47 of the samples that produced a review, not 48/48.**
One case (`arp-fsgen-044`, `src/database.py`) hit the 300 s per-sample timeout
and was never graded. Per ADR-0008 an operational failure is never folded into
task results, so it is reported as a timeout rather than a failure — and it is
also not quietly dropped from the denominator: 48 cases were attempted.

## What was evaluated

**System under test:** ARP's `tier2_reviewer` agent, constructed by ARP's own
agent factory (`agentic_v2.langchain.agents.create_agent`) from the `review_code`
step of ARP's shipped `code_review` workflow, carrying ARP's canonical
`prompts/reviewer.md` persona. The harness defines no prompt of its own. The
workflow DAG is not executed and no tools are bound — see caveat 6 for exactly
what is and is not in scope.

**Task:** given one real source file, produce a code review identifying
correctness bugs, missing functionality, code-quality problems, and security
vulnerabilities.

**Cases:** 48 source files extracted from ARP's own `runs/default/` history —
specifically the outputs of past `fullstack_generation` workflow runs. No case
was invented for this eval. `build_cases.py` scanned 624 run logs and found
1,843 candidate file blocks, of which 195 survived decontamination and
de-duplication; 48 were selected round-robin across languages to keep the suite
diverse. Rejections are counted, not silent: 909 duplicate, 410 outside the size
window, 196 unterminated (no `ENDFILE`, i.e. the log itself was cut off), 89
unknown extension, 43 non-source, 1 multi-file payload.

| Language | Cases |
| --- | ---: |
| C# | 12 |
| Python | 12 |
| TypeScript | 11 |
| TypeScript (React) | 11 |
| SQL | 2 |

48 distinct content hashes across 31 distinct file paths — repeated paths are
different generations of the same file from different runs. The exact suite is
`cases.jsonl`.

## How it was graded

ARP's **`agent.yaml`** rubric
(`agentic-v2-eval/src/agentic_v2_eval/rubrics/agent.yaml`, v1.0.0), used
unmodified. Six weighted criteria scored 0–5 by an LLM judge; the weighted score
is normalized to 0–1 and passes at ≥ 0.70.

| Criterion | Weight | Mean score | Scored ≥ 4/5 |
| --- | ---: | ---: | ---: |
| Correctness | 0.25 | 4.638 | 42 / 47 |
| Completeness | 0.20 | 4.553 | 45 / 47 |
| Clarity | 0.15 | 4.957 | 47 / 47 |
| Relevance | 0.15 | 5.000 | 47 / 47 |
| Efficiency | 0.10 | 4.915 | 47 / 47 |
| Safety | 0.15 | 5.000 | 47 / 47 |

Every one of the 47 graded samples received a score on all six criteria — and
that is now enforced rather than observed: a missing or non-numeric criterion
makes the grader abstain instead of renormalizing (caveat 8). The 48th sample
timed out before producing a review and so was never graded.

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

These matter for reading the 47/47 pass rate correctly.

1. **The grader is a single uncalibrated LLM judge, and it is noisy.** One judge,
   one attempt per sample, no calibration against human labels — treat individual
   case scores as indicative, not authoritative. Across earlier runs of this suite
   the same case under the same target model has swung by more than 0.30 on the
   judge's score alone.
2. **Scores cluster at the ceiling.** Median 1.00, only two cases below 0.80, and
   nothing below the 0.70 threshold means this rubric/judge pairing is not very
   discriminative at the top end. It demonstrates the pipeline works end to end;
   it does not finely rank agent quality. A 47/47 pass rate is evidence the
   harness runs, not evidence the agent is flawless.
3. **The pass threshold is the rubric's own 0.70**, not a bar chosen to flatter
   the result.
4. **This grades review quality, not review correctness against ground truth.**
   No human verified that the issues the agent reported are real. The judge
   assesses the review against the rubric, and it sees the source file.
5. **Earlier runs were discarded, not committed** — one 24/48 (OpenRouter daily
   quota exhausted) and one 44/48 (transient NVIDIA 503s) before the harness was
   fixed. Bounded retry (4 attempts, linear backoff) on 429/503 is in the
   harness and handled the transient failures; the one timeout in this run
   exhausted the 300 s per-sample budget rather than erroring, so retry does not
   apply to it.
6. **The system under test is ARP's own reviewer agent — and the scope is
   exactly one step of it.** The harness builds the agent with
   `agentic_v2.langchain.agents.create_agent`, the same factory ARP's LangGraph
   engine calls for an LLM step, for the `tier2_reviewer` agent named by the
   `review_code` step of ARP's shipped `code_review` workflow (read from that
   YAML at runtime). The persona is ARP's canonical
   `agentic_v2/prompts/reviewer.md`, never a string copied into the harness; the
   task text is assembled by ARP's own `build_task_description` and the response
   read back with ARP's own `extract_agent_response_text` /
   `extract_agent_metadata`. **Which prompt version ran is provable, not
   asserted:** the ADR-056 prompt registry's content fingerprint for `reviewer`
   is verified at startup against both the file on disk and the role
   `tier2_reviewer` resolves to — any drift aborts the run — and is then recorded
   on every execution result (`environment_metadata.arp_prompt_qualified_version`
   and `arp_prompt_sha256`), folded into `target_fingerprint`, and repeated in
   `run-stats.json`. **Still out of scope:** the workflow DAG. The four sibling
   steps of `code_review` (`parse_code`, `style_check`, `complexity_analysis`,
   `generate_summary`) are not executed — they sit upstream and downstream of the
   review and have no bearing on reviewing one file — and tools are explicitly
   unbound, because the file is handed to the agent inline and binding ARP's
   tier-2 tool surface would exercise the fail-closed approval gate rather than
   review quality. These numbers therefore characterize ARP's reviewer agent on a
   single-file review, not a full multi-step workflow run.
7. **The judge grades complete inputs; nothing is truncated.** The source file
   and the agent's review are passed to the judge in full — the old 4,000/6,000
   character slices are gone. In their place is a hard guard: if the assembled
   judge prompt would exceed 200,000 characters, the sample is failed with
   `GradeStatus.ERROR` and evidence naming the prompt, source, and review
   lengths against the limit. Both models are large-context hosted models, so no
   case in this suite approaches the guard; it exists so that a pathological
   input fails loudly instead of being quietly shortened, which is what made the
   old behavior a bug.
8. **Missing rubric criteria abstain — they never pass or fail.** If the judge
   omits a criterion, or returns a non-numeric value for one (booleans and
   NaN/Infinity included, both of which JSON permits), the grader returns
   `GradeStatus.ABSTAIN` with the absent criteria named in
   `evidence.missing_criteria` and no score. Renormalizing over whatever the
   judge happened to return is structurally impossible now, not merely avoided:
   `Rubric.weighted_score` raises on an incomplete score set rather than
   reweighting the remainder, so a PASS/FAIL can only be reached from a complete
   set of six criterion scores.
9. **The case corpus carries no log framing.** ARP's run logs do not store bare
   source files: generated code is wrapped in the sentinel bundle format
   `FILE: <path>` … `ENDFILE`, and the run logger truncates any string over
   10,000 characters with a literal `... (<n> chars)` marker. `build_cases.py`
   parses the bundle with ARP's own delimiters (mirroring
   `agentic_v2/workflows/artifact_extractor.py`), stores the file content only,
   and **rejects** — never repairs — any payload that is display-truncated,
   unterminated, or holds more than one file, since the log has already lost data
   in those cases. Rejection counts by reason are printed at build time so
   nothing is silently dropped, and a post-build validation pass fails the build
   outright if any delimiter or truncation marker survives into a selected case.
   The suite is whatever survives this filter; it is not backfilled to hit a
   target count.
10. **One case never produced a review.** `arp-fsgen-044` (`src/database.py`)
    exhausted the 300 s per-sample timeout, so 47 of 48 attempted cases were
    graded. It is reported as a timeout, not a failure — ADR-0008 keeps
    operational outcomes out of task results — and it is not dropped from the
    denominator either: the suite is 48 cases, 47 of which yielded a gradeable
    review.

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
agentic-evalkit report fb694424348643379922f0bd9c22f348.json --format html
```

`harness/` holds everything needed to re-run:

| File | Purpose |
| --- | --- |
| `build_cases.py` | Extracts the suite from ARP's `runs/default/` logs, strips ARP's bundle framing, rejects truncated payloads |
| `arp_eval_harness.py` | Catalog, adapter, ARP reviewer-agent target, rubric grader |
| `run_eval.py` | Builds the manifest and drives `EvalRunner` |

**No path is baked in.** `build_cases.py` takes `--arp-root`, `--run-dir`
(default `<arp-root>/runs/default`) and `--out`. `run_eval.py` takes `--arp-root`
and `--rubric`; `--arp-root` defaults to the checkout that owns the importable
`agentic_v2` package, and `--rubric` defaults to
`<arp-root>/agentic-v2-eval/src/agentic_v2_eval/rubrics/agent.yaml`, so on a
normal checkout neither flag needs to be passed to `run_eval.py`.

Both packages must be importable by one interpreter. ARP's venv plus
`PYTHONPATH` pointing at evalkit's `src/` works without installing anything:

```bash
ARP=/path/to/agentic-runtime-platform
EVK=/path/to/agentic-evalkit
HARNESS="$EVK/reports/2026-07-26-agent-workflow-eval/harness"

# 1. Rebuild the case suite from ARP's own run logs
PYTHONPATH="$EVK/src:$HARNESS" "$ARP/.venv/bin/python" "$HARNESS/build_cases.py" \
  --arp-root "$ARP" --out cases.jsonl

# 2. Run the eval
PYTHONPATH="$EVK/src:$HARNESS" "$ARP/.venv/bin/python" "$HARNESS/run_eval.py" \
  --cases cases.jsonl --output-dir . --concurrency 4 --timeout 300
```

On Windows the only substitutions are `.venv/Scripts/python.exe` for the
interpreter and `;` for the `PYTHONPATH` separator.

## Files

| File | Contents |
| --- | --- |
| `fb694424348643379922f0bd9c22f348.json` | Canonical `EvalRunResult` — every sample, review, grade, and provenance record |
| `fb694424348643379922f0bd9c22f348.html` | Self-contained HTML report |
| `fb694424348643379922f0bd9c22f348.md` | Markdown summary |
| `run-stats.json` | Aggregates quoted above, plus `target_fingerprint` and `target_provenance` |
| `cases.jsonl` | The 48-case suite |
| `harness/` | The code that produced the run |
