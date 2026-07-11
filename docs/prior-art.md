# Prior art & build-vs-buy

This page records the build-vs-buy rationale for `agentic-evalkit` against the
existing evaluation-tooling landscape, with every framework claim verified
against its primary documentation on **2026-07-11**.

**Honest provenance:** the founding design's "Approaches Considered"
(design §3) weighed three *build* shapes — standalone library, host-repo
plugin, evaluation platform — and never listed *adopt an existing framework*
as an alternative. A later positioning statement differentiated against
Inspect, Harbor, LightEval, OpenAI Evals, and Langfuse, but that was
post-decision framing, not an evaluation — and it omitted promptfoo. This
page closes that gap retroactively: the conclusion below is the recorded
answer to "why not just use promptfoo (or Inspect, or DeepEval, …)?"

## The landscape, verified

| Framework | What it is | Where it excels |
|---|---|---|
| [promptfoo](https://www.promptfoo.dev/docs/intro/) | Open-source CLI/library for evaluating and red-teaming LLM apps | Config-driven assertions and metrics, prompt/model comparison, red-teaming, CI/CD integration. OpenAI's own cookbook [recommends it as the migration target](https://developers.openai.com/cookbook/examples/evaluation/moving-from-openai-evals-to-promptfoo) now that OpenAI is "winding down the Evals product" (2026-06-02) |
| [Inspect](https://inspect.aisi.org.uk/) | Frontier-AI evaluation framework from the UK AI Security Institute | Solvers/scorers architecture, agent evals (can drive external agents like Claude Code), untrusted-code sandboxing across Docker/K8s/Modal. The closest neighbor; its dataset contracts are cited as prior art in this package's design §18 |
| [DeepEval](https://github.com/confident-ai/deepeval) | Open-source (Apache-2.0) pytest-style LLM evaluation framework | Broad metric library: G-Eval, RAG metrics (faithfulness, contextual precision/recall), agentic metrics (task completion, tool correctness), multi-turn metrics; pytest + framework integrations |
| [Braintrust](https://www.braintrust.dev/docs/start) | Hosted AI observability platform with an eval framework | Playground + experiments UI, production logging/monitoring; account-based hosted service |
| [LangSmith](https://docs.langchain.com/langsmith/evaluation) | LangChain's dataset/experiment evaluation product | Datasets from production traces, human/code/LLM-judge/pairwise evaluators, experiment comparison; cloud, hybrid, or self-hosted platform |

Also in the original positioning set: Harbor, LightEval, Langfuse, and
[OpenAI Evals](https://github.com/openai/evals) — the last now sunsetting per
the cookbook note above.

## What none of them do

Every framework above solves the eval **workflow** problem well. This
package exists for the eval **validity** problem — being *structurally hard
to overclaim a result*. On 2026-07-11 we checked each framework's primary
documentation for the five validity controls this package treats as
load-bearing; none documents any of them as a first-class concept:

| Validity control | Here | promptfoo / Inspect / DeepEval / Braintrust / LangSmith |
|---|---|---|
| Model judges gated by calibration evidence (TNR/TPR floors, held-out human labels, position-bias probe, expiry; uncalibrated judges can never hard-gate) | [ADR-0007](adr/0007-objective-first-grading.md) | Model-graded assertions/judges are available everywhere, uncalibrated |
| Run comparison refused on provenance mismatch (dataset revision, adapter, grader, target, sampling, environment/code fingerprints) instead of producing a misleading delta | [ADR-0008](adr/0008-statistical-comparability.md), [ADR-0015](adr/0015-environment-and-code-fingerprints-gate-comparability.md) | Side-by-side comparison without comparability gating |
| Operational failure is never a task failure (error/timeout/unavailable are separate outcome categories, never folded into fail rates) | [ADR-0005](adr/0005-benchmark-adapters-and-harnesses.md), [ADR-0008](adr/0008-statistical-comparability.md) | Not documented |
| Authoritative-verifier boundary: a missing benchmark harness returns typed `unavailable`, never a substitute score; only a real harness verdict may claim "resolved" | [ADR-0005](adr/0005-benchmark-adapters-and-harnesses.md), [ADR-0014](adr/0014-swebench-docker-harness-executor.md) | Not documented |
| Dataset contamination metadata + canary tripwires (built-in public presets ship labeled `SUSPECT`) | [ADR-0013](adr/0013-contamination-metadata-and-canaries.md) | Not documented |

Statistical honesty is in the same family: cluster-robust intervals for
repeated attempts and visible uncertainty in reports
([ADR-0016](adr/0016-cluster-robust-intervals-for-repeated-attempts.md))
rather than pooled attempt counts presented as independent trials.

## When you should use one of them instead

This comparison only means something if it cuts both ways:

- **Prompt-level CI assertions, model shootouts, red-teaming** — use
  **promptfoo**. It is mature, community-backed, and that lane is not this
  package's lane.
- **Frontier-style agent evaluations with heavy sandboxing**, or an
  ecosystem standard backed by a safety institute — **Inspect** is the
  serious alternative and the closest overlap. If the calibration/
  provenance/contamination controls above ever land there as first-class
  features, the build-vs-buy math for this package should be revisited —
  that is this page's supersession trigger.
- **A large off-the-shelf RAG/agent metric library in pytest** — **DeepEval**.
- **Hosted observability with evals attached to production traces** —
  **Braintrust** or **LangSmith**, accepting the platform coupling.

## The recorded decision

Build. Three reasons, in order of weight:

1. **The validity controls do not exist elsewhere** (verified above), and
   they are the point: the 2026 eval-validity literature this package's
   recent work is grounded in (SWE-bench Verified regrade, the NeurIPS-2025
   Agentic Benchmark Checklist, UK AISI and NIST AREP guidance) shows
   grading defects distort measured agent performance more than capability
   differences do. A harness that makes those defects structurally hard to
   commit is a different product than a harness that runs assertions.
2. **The standalone boundary is a requirement, not a preference**
   ([ADR-0001](adr/0001-standalone-boundary.md)): host systems are evaluated
   only through the neutral `ExecutionTarget` protocol, with typed frozen
   contracts on every wire. None of the surveyed tools offered that
   contract-first shape without adopting their runtime or platform.
3. **Coexistence, not competition:** nothing prevents running promptfoo for
   prompt CI alongside this package for benchmark-grade claims. The lanes
   are complementary, and this package deliberately does not compete in the
   prompt-assertion/red-teaming lane.

## Supersession

Revisit this page if (a) Inspect or another maintained framework ships
calibration-gated judging, provenance-gated comparison, and typed
operational/task outcome separation as first-class features, or (b) this
package's maintenance cost starts crowding out the validity work that
justifies it.
