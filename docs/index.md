---
title: agentic-evalkit
description: Evaluate agentic systems with reproducible, evidence-first grading — dataset discovery, typed contracts, objective-first grading, and statistical comparison, framework-neutral.
hide:
  - toc
---

<div class="console-hero" markdown="1">
<div class="hero-inner" markdown="1">

<div class="hero-meta">
  <span class="hero-eyebrow">portfolio tooling · evaluation</span>
  <span class="status-tag">status: alpha</span>
</div>

# agentic-evalkit

<p class="hero-sub">A standalone evaluation toolkit for agentic systems — dynamic dataset discovery, typed evaluation contracts, benchmark-valid grading, calibrated judges, and statistical reporting behind a developer-friendly Python API and CLI. Datasets, grading, and reporting are separated from the system under test through a neutral <code>ExecutionTarget</code> protocol, so any callable, subprocess, or HTTP system is evaluable without framework lock-in.</p>

<div class="hero-actions" markdown>
[Get started](guides/quickstart.md){ .md-button .md-button--primary }
[View source](https://github.com/agentic-evalkit/agentic-evalkit){ .md-button }
</div>

<div class="term">
  <span class="term-prompt">$</span>
  <span class="term-cmd">pip install agentic-evalkit && agentic-evalkit doctor</span>
  <span class="term-comment"># no datasets/pyarrow/Docker required for the base install</span>
</div>

</div>
</div>

<div class="trusted-stack" markdown>
<span>Python 3.11+</span>
<span>Pydantic v2</span>
<span>Typer</span>
<span>Hugging Face Hub</span>
<span>httpx</span>
<span>mypy --strict</span>
</div>

[part of the Console portfolio](https://tafreeman.github.io/tafreeman/){ .link-forward }

<p class="section-kicker">get running</p>

## Quick start

```bash
pip install agentic-evalkit
agentic-evalkit doctor
agentic-evalkit init --preset gsm8k --output eval.yaml
agentic-evalkit run eval.yaml --limit 5 --yes
```

This resolves the curated GSM8K preset from Hugging Face, runs five samples through the packaged smoke target, grades them with a normalized exact-match grader, and writes a canonical JSON report — no importer code, manual dataset download, `datasets`, `pyarrow`, or Docker required.

[Full walkthrough](guides/quickstart.md){ .link-forward }

<div class="stat-strip" markdown>
<div class="stat-item">
  <div class="stat-value">9</div>
  <div class="stat-label">ADRs</div>
</div>
<div class="stat-item">
  <div class="stat-value">3</div>
  <div class="stat-label">Execution target adapters</div>
</div>
<div class="stat-item">
  <div class="stat-value">2</div>
  <div class="stat-label">Dataset providers</div>
</div>
<div class="stat-item">
  <div class="stat-value">80%</div>
  <div class="stat-label">Coverage gate, CI-enforced</div>
</div>
</div>

---

<p class="section-kicker">design</p>

## Objective-first, framework-neutral

Existing evaluation frameworks couple dataset access, grading, and reporting to specific agent platforms or model-provider SDKs. `agentic-evalkit` grades with the strongest valid evidence available, in order — authoritative benchmark verifier, executable tests, schema/type validation, exact or normalized comparison, documented domain metric, calibrated model judge, human review — and a model judge is never the first check for anything an objective grader can decide.

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<h3 class="fc-title">Dynamic dataset discovery</h3>
<p class="fc-body">Two providers — <code>local</code> (JSON/JSONL/CSV/YAML on disk) and <code>huggingface</code> (Hub search plus Dataset Viewer) — behind one async <code>DatasetProvider</code> protocol. A content-addressed cache backs offline runs.</p>
[Providers guide](guides/providers.md){ .fc-link }
</div>

<div class="feature-card" markdown>
<h3 class="fc-title">Typed, immutable contracts</h3>
<p class="fc-body">Every manifest, resolved dataset, and report is a versioned Pydantic model. Canonical JSON reports pin <code>dataset_revision</code>, <code>adapter</code>, <code>grader</code>, <code>target_name</code>, and an <code>environment_fingerprint</code> for reproducibility.</p>
[ADR-0002: Immutable versioned contracts](adr/0002-immutable-versioned-contracts.md){ .fc-link }
</div>

<div class="feature-card" markdown>
<h3 class="fc-title">Objective-first grading</h3>
<p class="fc-body">A strict evidence order — benchmark verifier, executable tests, schema validation, normalized comparison, domain metric, calibrated judge, human review. Hard objective requirements can't be averaged away by a judge score.</p>
[Graders guide](guides/graders.md){ .fc-link }
</div>

<div class="feature-card" markdown>
<h3 class="fc-title">Neutral execution targets</h3>
<p class="fc-body">Callable, subprocess-JSONL, and HTTP adapters normalize every outcome to an <code>ExecutionStatus</code> before grading — no target-specific response shape ever reaches a grader.</p>
[Targets guide](guides/targets.md){ .fc-link }
</div>

<div class="feature-card" markdown>
<h3 class="fc-title">Statistical comparison</h3>
<p class="fc-body"><code>compare</code> rejects incompatible runs — different dataset revision, adapter, grader, target, or sampling policy — and reports paired success-rate deltas with a seeded bootstrap interval rather than a misleading number.</p>
[ADR-0008: Statistical comparability](adr/0008-statistical-comparability.md){ .fc-link }
</div>

<div class="feature-card" markdown>
<h3 class="fc-title">Typed errors, stable exit codes</h3>
<p class="fc-body">Every CLI command that talks to a provider raises a typed <code>AgenticEvalkitError</code> with a stable <code>code</code>, mapped to one of a small set of exit codes — never a raw traceback unless <code>--debug</code> is passed.</p>
[Quickstart](guides/quickstart.md){ .fc-link }
</div>

</div>

---

<p class="section-kicker">boundary</p>

## Standalone by design

`agentic-evalkit` imports no modules from the Agentic Runtime Platform, ExecutionKit, or any other host repository, at build time, runtime, or in its test suite. Those systems — or any other agentic system — are evaluated only through the public `ExecutionTarget` protocol: callable, subprocess, or HTTP. Legacy evaluation code may remain in host repositories; this package neither imports nor migrates it.

[ADR-0001: Standalone boundary](adr/0001-standalone-boundary.md){ .link-forward } · [ADR-0006: Execution target boundary](adr/0006-execution-target-boundary.md){ .link-forward }

---

<p class="section-kicker">documentation</p>

## Where to go next

<div class="doc-grid" markdown>

<div class="doc-card" markdown>
<h3 class="dc-title">Guides</h3>

- [Quickstart](guides/quickstart.md) — install to first report, including the standalone `report` command for self-contained HTML
- [Providers](guides/providers.md) — local formats, Hugging Face auth, the content-addressed cache, and `--offline` mode
- [Graders](guides/graders.md) — the objective-first evidence order and calibrated-judge requirements
- [Targets](guides/targets.md) — callable, subprocess-JSONL, and HTTP execution targets
- [SWE-bench](guides/swebench.md) — the preview/prediction-export workflow and harness boundary
- [HTTP agent example](guides/http-agent-example.md) — evaluating a real tool-using agent over HTTP
</div>

<div class="doc-card" markdown>
<h3 class="dc-title">Reference</h3>

- [Architecture specification](specs/2026-07-02-agentic-evalkit-design.md) — the full design
- [Implementation plan](plans/2026-07-02-agentic-evalkit-initial-release.md) — initial release plan
- [ADR index](adr/0001-standalone-boundary.md) — nine architecture decision records, 0001 through 0009
</div>

</div>

<div class="cta-card" markdown>
### Evaluating a real system?

Start with the [quickstart](guides/quickstart.md) for the pipeline end to end, then the [targets guide](guides/targets.md) to wire `run` to your own callable, subprocess, or HTTP system under test.

<div class="hero-actions" markdown>
[Read the quickstart](guides/quickstart.md){ .md-button .md-button--primary }
[Browse the ADRs](adr/0001-standalone-boundary.md){ .md-button }
</div>
</div>
