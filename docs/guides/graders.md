# Graders

`agentic-evalkit` grades with the strongest valid evidence available,
strictly in this order:

1. authoritative benchmark verifier or state transition;
2. executable tests;
3. schema, type, or format validation;
4. exact or normalized deterministic comparison;
5. documented domain metric;
6. calibrated model judge;
7. human review.

A model judge is never the *first* check for anything an objective grader
can decide. Hard objective requirements cannot be averaged away by a
model-judge score.

## Objective graders

### `ExactMatchGrader`

Compares a normalized candidate value against a normalized reference:
Unicode normalization, optional case folding, whitespace normalization,
numeric canonicalization (so `"5"`, `"5.0"`, and `"5,000"`-style separators
can compare equal where appropriate), and an injected extractor function
that pulls the comparable value out of the target's raw output.

```python
from agentic_evalkit.graders.exact import ExactMatchGrader

def extract_answer(output: dict) -> str:
    return str(output.get("answer", ""))

grader = ExactMatchGrader(name="normalized-exact@1", extractor=extract_answer)
```

This is the grader behind the curated `gsm8k` preset (`normalized-exact@1`),
paired with `extract_final_answer` from the GSM8K adapter, which parses the
text after the dataset's `####` marker in the reference answer.

### `SchemaGrader`

Validates `NormalizedExecutionResult.output` against a supplied Pydantic
`TypeAdapter`. Useful for structured-output agents where "did the system
return a well-formed response" is itself an objective, deterministic
check — see [the HTTP agent example](http-agent-example.md), which uses a
`SchemaGrader` as its objective check.

## Composite graders and hard gates

`CompositeGrader` runs multiple component graders, preserves every child
result, and computes the weighted mean over the *available* numeric
sub-scores — a missing, abstained, or unavailable component score is
excluded from the mean, never treated as zero. Any component marked
`hard_gate=True` that fails forces the composite result to `FAIL`
regardless of how well the other components scored:

```python
from agentic_evalkit.graders.composite import CompositeGrader, WeightedGrader

grader = CompositeGrader(
    name="quality@1",
    graders=(
        WeightedGrader(schema_check, weight=0.2, hard_gate=True),
        WeightedGrader(style_judge, weight=0.8, hard_gate=False),
    ),
)
```

If `schema_check` fails, the composite result is `FAIL` with `hard_gate=True`
even if `style_judge` scored perfectly — a hard gate is noncompensable. If a
component grader itself raises, that component's result is recorded as
`ERROR`, not a silent zero, and the composite's evidence still shows every
child's status, score, weight, and gate flag.

## Rubrics

`Rubric` and `RubricCriterion` express atomic, holistic-scoring criteria:
every criterion has a stable ID, a binary or bounded scale, an evidence
requirement, a weight, and a hard-gate flag. Duplicate criterion IDs,
negative weights, broad criteria with no evidence requirement, and rubrics
whose weights sum to zero are all rejected at construction time. Broad
holistic scores remain advisory only — they cannot substitute for an
authoritative or executable check.

## Calibrated judges

A model judge can gate a release only after it passes a versioned
calibration check. `JudgeGrader` verifies, before it will ever set
`hard_gate=True` on a result:

- the live judge's `fingerprint` matches the calibration's
  `judge_fingerprint` exactly (a different model or prompt invalidates the
  calibration);
- the calibration has not expired (`expires_at` is in the future) and was
  calibrated within the last 90 days (`calibrated_at`);
- the held-out calibration has at least 30 positive and 30 negative labels;
- both TPR (true positive rate) and TNR (true negative rate) meet the
  calibration's threshold **and** the project floor of TNR ≥ 0.95 and
  TPR ≥ 0.85;
- a reversed-order ("position-bias") probe agrees with the primary verdict;
- the judge returns a parseable, non-abstained structured response (parse
  failures retry at most twice — three attempts total).

```python
from datetime import UTC, datetime, timedelta
from agentic_evalkit.graders.judge import CalibrationArtifact, JudgeGrader

calibration = CalibrationArtifact(
    calibration_id="cal-2026-07",
    judge_fingerprint="judge:my-model:v3-prompt",
    calibrated_at=datetime.now(UTC),
    expires_at=datetime.now(UTC) + timedelta(days=30),
    true_positive=95, true_negative=97, false_positive=3, false_negative=5,
    threshold=0.85,
)
grader = JudgeGrader(my_judge_client, calibration=calibration, gate=True)
```

Calibration failures demote in two tiers (D-1 as amended 2026-07-04), and the
specific reason is always recorded in `evidence["reason"]`:

- **Affirmatively bad evidence → `UNAVAILABLE` outright.** An expired
  calibration, or one whose held-out TNR/TNR falls below the project floor,
  is unusable: the result carries no score and can never be mistaken for a
  verdict.
- **Absent evidence → advisory only, never gates.** An uncalibrated judge, an
  undated or stale (older than 90 days) calibration, insufficient held-out
  samples, a fingerprint mismatch, or a position-bias disagreement still
  yields an advisory score, but `hard_gate` stays `False`.

Either way the grader never silently converts a calibration failure into a
task failure or a false pass.

## Abstention and error semantics

Every `GradeResult` carries one of six statuses: `pass`, `fail`, `partial`,
`error`, `abstain`, or `unavailable`. These are kept distinct on purpose:

- `error` means the grader itself could not produce a verdict (a bug, a
  malformed judge response after retries, an exception in a component
  grader) — it is an infrastructure problem, not evidence the system under
  test did something wrong.
- `abstain` means the grader explicitly declined to render a verdict (for
  example, a judge that determined it could not confidently judge this
  case).
- `unavailable` means the check that would produce an authoritative
  verdict is not installed or not configured (a missing harness capability,
  an uncalibrated judge, a composite with no definitive component).

None of these three statuses is ever silently collapsed into `fail` — a
report that conflates "the system failed" with "we could not check" would
misrepresent both the system under test and the evaluation's own
reliability. See [the graders' aggregate reporting](../guides/quickstart.md)
for how these statuses roll up into a run's separated outcome counts.

See [ADR-0007](../adr/0007-objective-first-grading.md) for the full
objective-first and calibration policy this guide summarizes.
