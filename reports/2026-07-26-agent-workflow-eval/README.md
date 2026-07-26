# ARP agent-workflow eval — 2026-07-26

A real evaluation run: 48 code-review tasks executed against
`agentic-runtime-platform`'s (ARP) own reviewer agent and graded with ARP's own
YAML rubric, driven by `agentic-evalkit`'s `EvalRunner`. Every number on this
page came out of the run recorded in
`e2f77bdddafc4b74b324a57851587317.json`.

> **This run supersedes the invalidated run published at commit `c65f5b1`.**
> External review found sixteen validity defects across six rounds: the wrong
> system under test, silently truncated judge inputs, a machine-pinned rubric
> path, a grader that renormalized over missing rubric criteria, a case corpus
> contaminated with ARP's own log framing, spilled review artifacts that were
> referenced but never committed, a target fingerprint that ignored the ARP
> revision, a grader that clamped out-of-range judge scores into valid ones, a
> fingerprint absent from the one manifest field evalkit's own `compare_runs`
> reads, a dirty-checkout marker that gave every modified tree at a commit the
> same identity, a grader identity that ignored which rubric and judge scored
> the run, a `--arp-root` flag that selected a rubric without binding the
> checkout actually imported, a working-tree hash that never descended into new
> directories, a manifest identity that covered evalkit and ARP but never this
> harness's own source, and an `"unknown"` revision fallback that gave every
> unidentifiable ARP tree the same identity, and a dataset fingerprint over raw
> file bytes that made an identical corpus hash differently on Windows and
> POSIX.
> All sixteen are fixed in the harness under `harness/`, and every number below
> comes from a full re-run against the fixed harness and a rebuilt corpus.
> **The runs are not comparable** and no figure here should be read as a delta
> against an earlier one: decontaminating the corpus changed the content of all
> 48 cases (zero shared content hashes), and under ARP's real persona the agent
> emits a structured artifact block instead of the free prose the old harness
> elicited. The superseded outputs remain in git history at `c65f5b1`.

## Headline numbers

| Metric | Value |
| --- | --- |
| Cases | 48 |
| Executions completed | 48 / 48 |
| Graded | 48 |
| Passed / failed | 48 / 0 |
| Errors / timeouts / abstains | 0 / 0 / 0 |
| Mean weighted score | 0.9575 (median 0.96, min 0.75, max 1.00) |
| Pass threshold | 0.70 (from `agent.yaml`) |
| Wall-clock | 1268.61 s (21 m 09 s) |
| Target tokens | 133,415 in / 365,822 out |
| Judge tokens | 129,978 in / 44,450 out |
| Total tokens | 673,665 |
| Cost | Not measured — see [Cost](#cost) |
| Target fingerprint | `sha256:2a8fe8b0fdcbd41f…` (policy `required`, pinned in the manifest) |
| Grader identity | `arp-agent-rubric@1+960aa0c318ac` (rubric contents + judge model + harness source) |
| Harness source | `398e74c90933fb37…` (digest of the three files in `harness/`) |
| Dataset revision | `1f5a426c23609c37` (canonical record digest — line-ending invariant, see caveat 13) |
| ARP revision | `872d723a…-dirty+cec21bca21f8` (see caveat 11) |
| Prompt | `reviewer` `v1@f1555690` |

Concurrency 4, 1 attempt per sample, 300 s per-sample timeout.

**This run completed all 48 cases with no timeouts — and scored the lowest
mean of the six runs on record (0.9575).** That coincidence is the point of
caveat 10: the most complete execution and the weakest scores arrived together,
minutes apart from runs that timed out and scored higher, with nothing about
the agent changed. A 48/48 pass rate here is not a better result than run 5's
46/46; it is a different sample.

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
| Correctness | 0.25 | 4.583 | 45 / 48 |
| Completeness | 0.20 | 4.500 | 48 / 48 |
| Clarity | 0.15 | 4.958 | 47 / 48 |
| Relevance | 0.15 | 5.000 | 48 / 48 |
| Efficiency | 0.10 | 4.979 | 48 / 48 |
| Safety | 0.15 | 5.000 | 48 / 48 |

Every one of the 48 graded samples received a usable score on all six criteria —
and that is now enforced rather than observed: a criterion that is missing,
non-numeric, fractional, or outside 0–5 makes the grader abstain rather than be
repaired into a gradeable value (caveat 8). The judge's scores as parsed are
recorded verbatim in each grade's `evidence.judge_raw_scores`, so a reader can
confirm no value was reshaped on the way in. Every case produced a review in
this run; five of the six runs on record did not.

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

These matter for reading the 48/48 pass rate correctly.

1. **The grader is a single uncalibrated LLM judge, and it is noisy.** One judge,
   one attempt per sample, no calibration against human labels — treat individual
   case scores as indicative, not authoritative. Across earlier runs of this suite
   the same case under the same target model has swung by more than 0.30 on the
   judge's score alone.
2. **Scores cluster at the ceiling.** Median 0.96, no case below 0.75, and
   nothing near the 0.70 threshold means this rubric/judge pairing is not very
   discriminative at the top end. It demonstrates the pipeline works end to end;
   it does not finely rank agent quality. A 48/48 pass rate is evidence the
   harness runs, not evidence the agent is flawless — one of the six runs of
   this same suite, same models, produced a genuine rubric failure (0.69).
   Nothing about the agent changed between them.
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
8. **Unusable rubric criteria abstain — they are never repaired.** A criterion
   counts as unusable when the judge omits it, returns a non-numeric value
   (booleans and NaN/Infinity included, both of which JSON permits), or returns
   a number that is fractional or outside the rubric's 0–5 levels. Any of those
   yields `GradeStatus.ABSTAIN` with the criteria named in
   `evidence.missing_criteria` and no score. Nothing is coerced into range: an
   earlier version clamped, which silently turned a judge answering `50` for
   every criterion into a perfect PASS and a negative into a legitimate zero —
   manufacturing a gradeable result out of a malformed one. Renormalizing is
   structurally impossible rather than merely avoided: `Rubric.weighted_score`
   raises on an incomplete set rather than reweighting the remainder, so
   PASS/FAIL is reachable only from all six criterion scores. Each grade also
   records `evidence.judge_raw_scores` — the judge's values exactly as parsed —
   so that a recorded `5` can be distinguished from a rejected `50` by anyone
   reading the report rather than taken on trust.
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
10. **Operational variance is real, and this run happens to look best on
    completion while scoring worst.** No case timed out here, so all 48 were
    graded. **This is the single most important caveat on the page**, because
    six runs of this suite now exist and they disagree:

    | Run | Timeouts | Timed-out case(s) | Graded | Passed | Mean |
    | ---: | ---: | --- | ---: | ---: | ---: |
    | 1 | 1 | `-044` | 47 | 47 | 0.9611 |
    | 2 | 3 | `-005`, `-013`, `-021` | 45 | 44 (one 0.69 failure) | 0.9584 |
    | 3 | 1 | `-017` | 47 | 47 | 0.9636 |
    | 4 | 1 | `-033` | 47 | 47 | 0.9709 |
    | 5 | 2 | `-030`, `-031` | 46 | 46 | 0.9646 |
    | 6 (published) | 0 | — | 48 | 48 | 0.9575 |

    Same corpus, same models, same concurrency, and **no case has timed out in
    more than one run** — nine distinct cases have timed out across six runs,
    each exactly once. A genuine rubric failure appeared in run 2 and never
    recurred. Nothing about the agent changed across any of them. Which cases
    complete — and whether any case fails — is substantially a property of
    provider latency and judge noise on the day. Treat any single run's
    completion count and pass rate as one sample from a noisy distribution, not
    a measurement of the agent: across six runs the mean spans 0.9575–0.9709
    (range 0.0134) and the graded count spans 45–48, so no difference smaller
    than that is distinguishable from noise. **Note what run 6 shows:** it is
    the only run to complete every case, and it has the lowest mean of the six.
    Completion rate and score quality moved in opposite directions on identical
    inputs. The published run is simply the most recent — not the best.
11. **The ARP checkout was not clean, and the fingerprint says so.** The
    recorded revision is `872d723a…-dirty+cec21bca21f8`: the checkout carried
    two untracked files unrelated to the reviewer path. The digest covers the
    tracked diff and the contents of every untracked file (git is queried with
    `--untracked-files=all`, so files inside a new directory are hashed
    individually rather than the directory being noted and skipped). Where git
    cannot describe a tree at all — an unpacked or copied checkout with no
    `.git` — the revision is `nogit+<digest>` over the package source rather
    than a literal `"unknown"`, which would have given every unidentifiable
    tree one shared identity inside a `required`-policy fingerprint.
12. **The harness's own source is part of the run's identity.** The judge
    prompt, the score-parsing rules and the target integration all live in
    `harness/`, and nothing else in the manifest covers them: evalkit's
    `code_fingerprint` hashes only the installed package, and the target
    fingerprint and rubric digest cover ARP and the rubric. A digest of the
    three harness files is therefore folded into both the target fingerprint
    and the grader identity, so changing how scores are parsed or how the agent
    is invoked makes two runs non-comparable rather than silently equivalent.
13. **The dataset revision identifies the records, not the file's bytes.** It is
    a SHA-256 over a canonical serialization of the parsed rows, so it is
    invariant to line endings, key order and insignificant whitespace. Hashing
    raw bytes — as this harness originally did — meant the suite built on
    Windows recorded `f6cb6d74…` while the committed LF blob hashes to
    `94918c71…`: a POSIX reproduction of a byte-for-byte identical corpus would
    have been rejected by `compare_runs` as a different dataset. The digest
    still moves when any record's content changes. The suffix is reported rather than
    suppressed because a fingerprint claiming a clean tree it did not have would
    overstate how reproducible this run is. Two runs are safely comparable only
    when the whole of `target_fingerprint` matches, and a `-dirty` revision
    cannot guarantee that.

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
agentic-evalkit report e2f77bdddafc4b74b324a57851587317.json --format html
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
| `e2f77bdddafc4b74b324a57851587317.json` | Canonical `EvalRunResult` — every sample, review, grade, and provenance record |
| `e2f77bdddafc4b74b324a57851587317.html` | Self-contained HTML report |
| `e2f77bdddafc4b74b324a57851587317.md` | Markdown summary |
| `run-stats.json` | Aggregates quoted above, plus `target_fingerprint` and `target_provenance` |
| `cases.jsonl` | The 48-case suite |
| `artifacts/` | Spilled execution outputs, content-addressed — every `output_ref` in the canonical JSON resolves here |
| `harness/` | The code that produced the run |
