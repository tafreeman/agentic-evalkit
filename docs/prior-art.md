# Prior art & build-vs-buy

This page records the build-vs-buy rationale for `agentic-evalkit` against the
existing evaluation-tooling landscape, with every framework claim verified
against its primary documentation on **2026-07-11**.

**Honest provenance:** the founding design's "Approaches Considered"
([design §3](specs/2026-07-02-agentic-evalkit-design.md#3-approaches-considered)) weighed three *build* shapes — standalone library, host-repo
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
| [Inspect](https://inspect.aisi.org.uk/) | Frontier-AI evaluation framework from the UK AI Security Institute | Solvers/scorers architecture, agent evals (can drive external agents like Claude Code), untrusted-code sandboxing across Docker/K8s/Modal. The closest neighbor; its dataset contracts are cited as prior art in this package's [design §18](specs/2026-07-02-agentic-evalkit-design.md#18-source-derived-principles) |
| [DeepEval](https://github.com/confident-ai/deepeval) | Open-source (Apache-2.0) pytest-style LLM evaluation framework | Broad metric library: G-Eval, RAG metrics (faithfulness, contextual precision/recall), agentic metrics (task completion, tool correctness), multi-turn metrics; pytest + framework integrations |
| [Braintrust](https://www.braintrust.dev/docs/start) | Hosted AI observability platform with an eval framework | Playground + experiments UI, production logging/monitoring; account-based hosted service |
| [LangSmith](https://docs.langchain.com/langsmith/evaluation) | LangChain's dataset/experiment evaluation product | Datasets from production traces, human/code/LLM-judge/pairwise evaluators, experiment comparison; cloud, hybrid, or self-hosted platform |
| [MLflow](https://mlflow.org/docs/latest/genai/) | Apache-2.0 ML/GenAI lifecycle platform, self-hostable | Tracking server and UI, experiment/model/prompt registries, 60+ framework integrations, and **judge alignment** — a SIMBA optimizer over ≥10 human-labeled traces that raises judge–human agreement. Added to this table 2026-08-04 |
| [Langfuse](https://langfuse.com/docs) | Self-hostable LLM observability and evaluation platform | OpenTelemetry-based tracing, datasets and dataset runs, scores attached to traces. Added to this table 2026-08-04 |

Also in the original positioning set: Harbor, LightEval, and
[OpenAI Evals](https://github.com/openai/evals) — the last now sunsetting per
the cookbook note above.

**On MLflow specifically, since it is the closest thing to a direct rival and
also this package's largest distribution channel.** MLflow's judge alignment
and this package's judge gating are routinely conflated and are not the same
control. Alignment takes a judge that disagrees with humans and *makes it
agree better*. Gating takes a judge with no proof of agreement and *revokes
its authority* — advisory grade only, cannot hard-gate a release. Alignment
improves a signal; gating constrains a decision. Neither substitutes for the
other, and a team holding both is strictly better off than a team holding
either, which is why the response to MLflow was a bridge
([ADR-0022](adr/0022-host-platform-integration-boundary.md),
`agentic_evalkit.integrations.mlflow`) rather than a competing harness.

## What none of them do

Every framework above solves the eval **workflow** problem well. This
package exists for the eval **validity** problem — being *structurally hard
to overclaim a result*. On 2026-07-11 we checked each framework's primary
documentation for the five validity controls this package treats as
load-bearing; none documents any of them as a core concept:

The MLflow and Langfuse column was added on **2026-08-04** from the same kind
of primary-documentation review; the other five columns retain their
2026-07-11 verification. As always, "not documented" is a documentation-review
finding on the stated date, not a source audit of that project.

| Validity control | Here | promptfoo / Inspect / DeepEval / Braintrust / LangSmith | MLflow / Langfuse |
|---|---|---|---|
| Model judges gated by calibration evidence (TNR/TPR floors, held-out human labels, position-bias probe, expiry; uncalibrated judges can never hard-gate) | [ADR-0007](adr/0007-objective-first-grading.md) | Model-graded assertions/judges are available everywhere, uncalibrated | Judge **alignment** (MLflow SIMBA optimizer) raises judge–human agreement, but no control withholds an uncalibrated judge's authority |
| Run comparison refused on provenance mismatch (dataset revision, adapter, grader, target, sampling, environment/code fingerprints) instead of producing a misleading delta | [ADR-0008](adr/0008-statistical-comparability.md), [ADR-0015](adr/0015-environment-and-code-fingerprints-gate-comparability.md) | Side-by-side comparison without comparability gating | Runs and experiments are compared side by side; no documented refusal on provenance mismatch |
| Operational failure is never a task failure (error/timeout/unavailable are separate outcome categories, never folded into fail rates) | [ADR-0005](adr/0005-benchmark-adapters-and-harnesses.md), [ADR-0008](adr/0008-statistical-comparability.md) | Not documented | Not documented |
| Authoritative-verifier boundary: a missing benchmark harness returns typed `unavailable`, never a substitute score; only a real harness verdict may claim "resolved" | [ADR-0005](adr/0005-benchmark-adapters-and-harnesses.md), [ADR-0014](adr/0014-swebench-docker-harness-executor.md) | Not documented | Not documented |
| Dataset contamination metadata + canary tripwires (built-in public presets ship labeled `SUSPECT`) | [ADR-0013](adr/0013-contamination-metadata-and-canaries.md) | Not documented | Not documented |

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
  provenance/contamination controls above ever land there as built-in
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
operational/task outcome separation as built-in features, or (b) this
package's maintenance cost starts crowding out the validity work that
justifies it.

**(c) — added 2026-08-04, the MLflow trigger.** Revisit if **MLflow ships
calibration gating natively**: any control that withholds a judge's authority
when its calibration evidence is missing, expired, or below a floor, as
opposed to the alignment optimizer it ships today. This is the sharpest of
the three triggers, because it would undercut two things at once. It would
remove the differentiator in the table above, *and* it would remove the
argument for `agentic_evalkit.integrations.mlflow`
([ADR-0022](adr/0022-host-platform-integration-boundary.md)), whose entire
premise is that the combination of alignment and gating is unavailable in one
place. Concretely, watch for a calibration or evidence requirement attached
to `mlflow.genai.judges.make_judge`, or a scorer-level notion of a judge that
may not gate. The same trigger applies to Langfuse for the equivalent
control. If either lands it, the honest response is to say so here, retire
the corresponding bridge's headline claim, and re-run the build-vs-buy
question rather than defend the position.
