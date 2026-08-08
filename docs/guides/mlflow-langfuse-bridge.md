# Adding calibration gating to MLflow judge alignment

MLflow can make your LLM judge agree with humans more often. It cannot stop
that judge from blocking a release it has not earned the right to block.
Those are different problems, and this guide is about running both solutions
at once — without leaving MLflow.

The same bridge exists for Langfuse. The MLflow examples come first because
its judge features make the contrast sharpest.

```bash
pip install 'agentic-evalkit[mlflow]'      # or [langfuse], or both
```

---

## The distinction worth getting right

These two controls are routinely conflated, and everything below depends on
keeping them apart.

|  | Judge **alignment** (MLflow) | Judge **authority gating** (here) |
|---|---|---|
| Question it answers | "Does this judge agree with our humans?" | "Have we *proved* it does?" |
| Mechanism | SIMBA optimizer over ≥10 human-labeled traces | Calibration floors: TNR ≥ 0.95, TPR ≥ 0.85, 95% Wilson lower bound, ≤ 90 days old |
| What it changes | The judge's prompt | What the judge's verdict is permitted to decide |
| Failure it prevents | A judge that is wrong too often | A judge whose accuracy nobody measured deciding a release |

Alignment improves a signal. Gating constrains a decision. Neither is a
substitute for the other, and running both is strictly better than running
either: align the judge so it is right more often, gate it so it cannot act
beyond what it has been shown to be.

---

## Gate a judge you already have

Take any MLflow scorer — including one from `make_judge` — and wrap it:

```python
from mlflow.genai.judges import make_judge
from agentic_evalkit.integrations.mlflow import calibration_gate
from agentic_evalkit.graders.calibration import CalibrationArtifact

judge = make_judge(
    name="faithfulness",
    instructions="Is the answer supported by the retrieved context?",
    model="openai:/gpt-4o",
)

gated = calibration_gate(judge, calibration=my_calibration_artifact)

mlflow.genai.evaluate(data=eval_dataset, scorers=[gated])
```

`gated` behaves identically to `judge` in the case that matters least, and
differently in the two that matter most.

**With full calibration** — every floor cleared, evidence unexpired — the
verdict passes through untouched, tagged `evalkit_authority=gating`.

**With thin evidence** — no artifact yet, no `calibrated_at`, too few
held-out labels, or a Wilson lower bound short of the floor — the verdict
still comes through, tagged `evalkit_authority=advisory` with the reason
spelled out. Nothing is suppressed. The judge may well be right; you simply
have not proved it, so a release gate can be written to ignore advisory
results while a human still reads them.

**With evidence that is present and bad** — expired, or a measured TNR/TPR
genuinely below the floor on a sufficient sample — the verdict is *withheld*.
The feedback carries an error instead of a value, which keeps it out of every
aggregate MLflow computes. The wrapped judge is not even called, so a
known-untrustworthy LLM judge costs you nothing per row.

That three-way split is the whole control. The distinction that does the work
is between **absent** evidence and **bad** evidence: a judge nobody has
measured is unproven, not disproven, and silencing it would cost you all your
existing signal the day you adopt the gate.

Writing a release gate against it:

```python
for assessment in result.assessments:
    if assessment.metadata.get("evalkit_can_gate") == "true" and assessment.value is False:
        raise SystemExit("blocked by a judge entitled to block")
```

---

## Send a whole evaluation run to MLflow

If you run evaluations with this package's own `EvalRunner`, export the
result:

```python
from agentic_evalkit.integrations.mlflow import log_eval_run

mlflow_run_id = log_eval_run(
    result,
    tracking_uri="http://mlflow.internal:5000",
    experiment="agent-regression",
    calibration=my_calibration_artifact,
)
```

What lands in the tracking server:

- **Params** — the manifest: adapter, grader, target, dataset id/revision/
  config/split, sampling seed and temperature, selection window.
- **Metrics** — outcome counts recounted from the samples themselves, plus
  the pass rate *with its confidence interval*. Every operational outcome
  keeps its own counter, so a harness that crashed on ten samples never reads
  as a system that got ten answers wrong.
- **Tags** — every provenance field `compare_runs` checks, namespaced under
  `evalkit.provenance.*` so MLflow's run search can filter on them, plus the
  judge's authority level.
- **Artifacts** — the full run body at `evalkit/run.json` and the calibration
  record at `evalkit/calibration.json`.

Three properties are worth knowing about:

**Secrets are scrubbed before anything is transmitted.** Redaction runs once,
on the way out, defaulting to `DEFAULT_REDACTION_POLICY`. This differs from
the reporters, which apply only what you pass them — a report file stays on
your disk, an export does not. Pass `redaction_policy=RedactionPolicy()` to
opt out deliberately.

**Your MLflow configuration is left exactly as it was.** The export uses
`MlflowClient` with explicit run IDs, never `mlflow.set_experiment` or
`mlflow.start_run`, so it cannot redirect your own logging afterwards. You
can call it from inside your own `with mlflow.start_run()` block; it creates
a sibling run and leaves yours active.

**Absent measurements are omitted, not zeroed.** If no sample recorded
latency or cost, those metrics simply do not exist on the run. A `0.0` would
be a fabricated observation in somebody's chart.

---

## Compare two runs, or be told why you cannot

This is the part with no equivalent anywhere in the surveyed field:

```python
from agentic_evalkit.integrations.mlflow import compare_mlflow_runs

result = compare_mlflow_runs(baseline_run_id, candidate_run_id, seed=1234)
print(result.estimate, result.lower_percentile, result.upper_percentile)
```

Two outcomes are possible, and there is deliberately no third:

```text
IncompatibleRuns: MLflow runs 'a1b2' and 'c3d4' are not comparable:
  adapter differs: 'gsm8k@1' != 'gsm8k@2';
  sampling seed differs: 7 != 99
```

There is no "here's a delta, but note the caveat" mode, because a caveat
beside a number does not survive being pasted into a slide. `seed` is
required and keyword-only for the same reason: a comparison read off a shared
tracking server is the *most* likely one to be quoted later, so it is the last
place a silently irreproducible number should be possible.

A run this package did not export is refused too. There is no manifest and no
provenance on a foreign run, so no claim about comparability can be
supported.

---

## Use an evalkit grader as an MLflow scorer

The other direction — bring a grader from here into an evaluation you already
run there:

```python
from agentic_evalkit.graders.exact import ExactMatchGrader
from agentic_evalkit.integrations.mlflow import as_mlflow_scorer

scorer = as_mlflow_scorer(
    ExactMatchGrader(name="exact@1", extractor=lambda out: str(out["answer"])),
    name="evk_exact",
)

mlflow.genai.evaluate(data=eval_dataset, scorers=[scorer])
```

`PASS` becomes `True` and `FAIL`/`PARTIAL` become `False` — but `ABSTAIN`,
`ERROR` and `UNAVAILABLE` become a feedback *error*, never `False`. None of
those three says the system under test got the answer wrong; they say the
grader declined, broke, or could not be trusted. As errors, MLflow keeps them
out of the aggregate rather than averaging them in as failures.

---

## The same surface on Langfuse

```python
from langfuse import Langfuse
from agentic_evalkit.integrations.langfuse import log_eval_run, score_with_calibration_gate

client = Langfuse()

trace_id = log_eval_run(result, client=client, calibration=my_calibration_artifact)

score_with_calibration_gate(
    client,
    name="faithfulness",
    value=0.9,
    calibration=my_calibration_artifact,
    trace_id=trace_id,
)
```

A run becomes one root observation carrying the manifest, provenance,
recounted summary and full run body in its metadata, with one child
observation and score per sample.

Two Langfuse-specific behaviours follow from how Langfuse aggregates:

**Demotion renames the score.** A gating judge writes to `faithfulness`; an
advisory one writes to `faithfulness.advisory`; one proven unreliable writes
no numeric score at all, only a categorical `faithfulness.unavailable`.
Langfuse averages numeric scores by name, so writing an advisory value under
the gating name would move the aggregate before anyone read the metadata
explaining that it should not have.

**Non-verdicts never become `0.0`.** A sample that errored before grading, or
a grader that abstained, is recorded as a categorical `evalkit.grade_status`
rather than a numeric zero, where nothing can average it into your dashboard.

**There is no `compare_langfuse_runs`, and that is deliberate.** Langfuse has
no artifact store, so a full run body cannot be retrieved the way it can from
MLflow. Reconstructing one from trace metadata would be inference presented
as measurement. Export both runs, and compare the `EvalRunResult` objects
directly with `agentic_evalkit.stats.compare_runs`.

---

## What this does not do

Worth stating plainly, since the point of the package is being hard to
overclaim with:

- It does not make any judge more accurate. That is what alignment is for.
- It does not replace MLflow or Langfuse, and is not trying to. There is no
  UI, no trace store, no registry, and no gateway here.
- Calibration gating is only as good as the calibration artifact. This
  package enforces the floors; it does not collect your human labels.
- The Langfuse bridge is tested against the client protocol it declares, not
  against a live server. The MLflow bridge is tested against a real local
  MLflow store.

The design rules behind all of the above — including why redaction defaults
differently here than in the reporters — are recorded in
[ADR-0022](../adr/0022-host-platform-integration-boundary.md).
