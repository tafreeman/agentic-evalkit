# Agentic EvalKit Operating Plan

**Status:** Active — replaces the master delivery plan as the execution authority

**Created:** 2026-08-31

**Supersedes:** `docs/plans/2026-08-16-agentic-evalkit-master-delivery-plan.md`,
`docs/plans/2026-08-16-arp-evalkit-evaluation-calibration-tracker.md`, and the
sequencing (not the design content) of
`docs/plans/2026-08-16-agentic-evalkit-low-code-integration-layer.md`. Those
documents remain on disk as design references and historical record.

> Naming note: this document names `agentic-v2-eval` and `agentic_v2`. That is
> safe only under `docs/plans/` — `tests/contract/test_public_docs.py` exempts
> `docs/plans`, `docs/specs`, and `docs/adr` from the codename scan. Paths are
> written in backticks, not links, so `mkdocs build --strict` stays clean.

## Goal

EvalKit is the harness the maintainer's own evaluations actually run on. It
earns each new capability by being used, not by being planned.

## Operating principle

A feature enters the roadmap when a real evaluation needed it and the need was
met by hand-written code outside this package. That hand-written code is the
requirements document. Nothing is built on the strength of an anticipated
consumer.

The corollary: this plan is short by construction. When it grows past one page,
something has been added speculatively and should be cut.

## Verified state — 2026-08-31

Re-verify before acting; every line here is a point-in-time snapshot.

- `main` at `459de8b`. 1,189 non-live tests pass, 94.24% branch coverage.
- Published: **v0.3.0** (2026-07-23). Unreleased on `main`: `ClaudeAgentTarget`
  (ADR-0025), host-platform bridges (ADR-0022), judge calibration measurement
  (ADR-0024), a report-boundary redaction fix, `comparability_snapshot`.
- Built in: 5 execution targets (callable, subprocess, HTTP, MCP, Claude
  subscription), 3 benchmark adapters, 5 graders. Manifests carry typed
  `callable` / `subprocess` / `http` target blocks; `adapter:` and `grader:`
  resolve through a hardcoded table in `src/agentic_evalkit/cli/runs.py`.
- `LocalDatasetProvider` already decodes JSON, JSONL, CSV, and YAML.
- **EvalKit has a live consumer.** `agentic-runtime-platform` runs
  `agentic-workflows-v2/evals/swe_ab/` — a workflow A/B study driving EvalKit
  through the subprocess JSONL target, with ~2,550 lines of consumer-side
  harness and 132 mutation cases mined from this portfolio's own repositories.
  Active as of 2026-08-30. It uses no low-code layer, no calibrated judge, and
  no unreleased EvalKit feature.
- That consumer pins `agentic-evalkit>=0.3.0,<0.4.0`. **Publishing 0.4.0 drops
  it out of range silently** — the pin bump is part of the release, not a
  follow-up.

## Standing decisions

1. **Model judges are advisory. Indefinitely.** Every packaged `JudgeGrader` is
   wired `calibration=None` and cannot hard-gate. Promoting one requires an
   independently labeled corpus this project has no labelers for. The
   measurement machinery (ADR-0024) ships and stays; the authority ambition is
   withdrawn until a concrete evaluation is blocked by its absence.
2. **No configuration DSL is built ahead of a caller.** The low-code design
   document stands as a reference. Individual pieces ship only when an actual
   evaluation cannot be expressed without them.
3. **Nothing is deleted from the consumer.** `agentic-v2-eval` retirement is
   removed from scope. It is the consumer's decision, made in the consumer's
   repository, when its replacement has been proven there.
4. **Evidence discipline is unchanged.** The `RUNTIME_VERIFIED` /
   `STRUCTURAL_VERIFIED` / `ADVISORY_ONLY` distinction from
   `docs/release/2026-08-16-m0-baseline-acceptance.md` continues to govern every
   claim this package makes about itself.
5. **Progress is not tracked in a hand-maintained table.** Status lives in git
   history, the changelog, and release acceptance records under `docs/release/`.

## The short list

Ordered. One at a time. Each ships or is abandoned before the next starts.

1. **Release 0.4.0.** Five weeks of merged, tested, documented work is invisible
   to consumers. Coordinated three-repo change: publish via a GitHub *release*
   (not a tag), raise the consumer's pin to `<0.5.0`, update the profile hub's
   version string in both `repo-data.jsx` and `social-cards.jsx`.
2. **Harvest what `swe_ab` had to hand-write.** Candidates observed in the
   consumer-side harness, each to be judged generic or consumer-specific on its
   own merits: a local pytest harness executor (only a Docker one ships today),
   a mutation grader, multi-report union and merge for backfilled waves, and
   McNemar alongside the existing paired bootstrap in `stats.compare`. Port only
   what is generic; leave worktree and workflow plumbing in the consumer.
3. **A generic field-mapping adapter.** `BenchmarkAdapter` is a three-method
   protocol and the local provider already reads four formats. One adapter plus
   a manifest block turns "my rows in a JSONL file" into a runnable evaluation
   with no user Python — the majority of the low-code promise, at a fraction of
   the design's scope.

Nothing is scheduled beyond item 3. The next item is chosen from what the next
evaluation actually needed.

## Not doing

- A 12-phase serial delivery sequence with per-phase gates. It assumed a team.
- Judge authority promotion, held-out labeling corpora, and adjudication
  protocols — see standing decision 1.
- Deleting or re-platforming the consumer's legacy evaluation package.
- Conformance pilots against sibling repositories as a release gate.
- Any capability whose only justification is a plan document.
