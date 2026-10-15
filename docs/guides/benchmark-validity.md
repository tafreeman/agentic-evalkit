# Benchmark validity

Verified as of: **2026-09-07**. These are dated suitability decisions,
not certifications of a benchmark or model. Use the
[experiment workflow](experiments.md) to save the decision with a run.

## What changed

| Benchmark | Evidence reviewed | Use in this workspace |
| --- | --- | --- |
| SWE-bench Verified | OpenAI stopped reporting frontier capability scores after finding contamination and defective tasks. Its 59.4% defect figure applies to an intentionally difficult 138-task subset, not all 500 tasks. [February 23, 2026 audit](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | Historical regression and harness checks. A score alone does not establish current frontier coding capability. |
| SWE-Bench Pro public split | OpenAI subsequently **retracted its recommendation to switch to Pro**. Its two review paths flagged 200 and 249 broken tasks out of 731, including underspecified prompts and tests that reject valid solutions or accept incomplete ones. [July 8, 2026 audit](https://openai.com/index/separating-signal-from-noise-coding-evaluations/) | Do not substitute it automatically for Verified. Audit the selected tasks and scoring before making a capability claim. |
| BrowseComp | Anthropic found leaked answers and benchmark identification followed by answer-key extraction in a multiagent evaluation. This is evidence about the investigated configuration, not a universal contamination rate. [March 6, 2026 report](https://www.anthropic.com/engineering/eval-awareness-browsecomp) | Open-web runs require exposure checks and transcript review. Keep unaudited scores as regression evidence. |
| Sierra tau benchmark | Version 1.0.1 changes `banking_knowledge` grading; its maintainers explicitly prohibit comparing older scores with the new scoring scheme. Other domains are unaffected by that specific warning. [July 15, 2026 changelog](https://github.com/sierra-research/tau2-bench/blob/main/CHANGELOG.md#101---2026-07-15) | Candidate for stateful agent tasks. Pin task data, simulator and grader revision. Regrade old trajectories with the same scheme before comparison. |
| SWE-rebench | The project continuously collects tasks and standardizes its evaluation setup. Public task freshness reduces some exposure risks; it does not establish that a particular model never saw a task. [Project methodology](https://swe-rebench.com/about) | Candidate for dated coding task cohorts, conditional on task review, model exposure evidence and a working execution harness. |
| HELM | HELM selects scenarios, models and run specifications and preserves native benchmark results. [Official quick start](https://crfm-helm.readthedocs.io/en/latest/quick_start/) | Supported harness integration. Judge each underlying scenario separately. Preparing instances is not running a model or obtaining a score. |

## Choose a claim before running

The following policy is this project's synthesis of the evidence above:

- **`regression_only`**: detect behavior changes under a fixed setup. This
  includes old public benchmarks and the small original demonstration fixtures.
- **`conditional`**: a candidate for a stated capability claim, with a dated
  review and concrete validation evidence. This is not a claim that the data
  is clean or that every task is sound.
- **`blocked`**: do not execute this saved experiment until its known blocking
  issue is resolved and the decision is updated.

An experiment with `purpose: capability` requires a conditional assessment,
sources and existing evidence files. Reviews older than 90 days must be
refreshed. Ninety days is an operational review interval chosen here, not a
scientific contamination threshold. File presence and checksums prove which
evidence was supplied; the CLI cannot certify its scientific quality.

## Evidence needed for a useful A/B result

Record the source revision, selected task IDs, task and grader fixes, exclusions,
reference-solution checks, model identity, target implementation, tool access,
budgets and environment. Freeze both arms before observing results. Use fresh
task state and repeated attempts where behavior is stochastic. Inspect failed
executions separately from completed but incorrect answers.

Evaluate the final state as well as the transcript when the task changes files,
records or remote systems. Anthropic's agent-evaluation guidance distinguishes
outcomes from trajectories and recommends multiple trials and task-specific
graders. [January 2026 guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

The experiment comparator pairs matching task instances and reports missing or
operationally invalid pairs separately. Repeated attempts are grouped by task;
they are not independent extra benchmark questions. The resulting comparison
describes the selected tasks under the saved setup. It does not turn a small
regression fixture into evidence about general model capability.

Keep evaluation references outside the target's input. A published case, local
copy, renamed identifier or synthetic mutation is not automatically a private
holdout. Reassess exposure and task validity when updating a dataset, grader,
model or tool environment.
