# ADR-0024: Judge calibration is measured by the package, not hand-authored

## Status

Accepted

## Context

ADR-0007 established that a model judge may gate a release only on
statistical evidence, and ADR-0020 tightened that evidence with a Wilson
lower bound. Both are fully implemented on the *enforcement* side:
`CalibrationArtifact` holds the confusion-matrix counts and every floor
check, `JudgeGrader` refuses `hard_gate=True` without them, and
`judge_authority` reduces an artifact to `GATING` / `ADVISORY` /
`UNAVAILABLE` with a reason.

Nothing implemented the *measurement* side. `CalibrationArtifact` was
constructed in zero production paths — the only occurrences in `src/` were
its own class definition — and there was no command to produce one. The
practical consequence was that this package's most differentiated control
could only ever say no. A user who genuinely had a reliable judge and
wanted it to gate had to hand-write four integer counts and invent a
`judge_fingerprint` that happened to match their live judge, with no tool
to check either. Evidence a human types is not evidence, and an artifact
whose fingerprint is a guess describes an unknown judge.

Three further facts shaped the design. `calibrated_at` is optional on the
artifact, and its absence blocks gating forever — correct for records
arriving from elsewhere, but a trap for a first measurement. The artifact
is a small file intended to be committed and attached to a release, so
what it may contain is a disclosure question, not a formatting one. And
`judge_authority` lived in `agentic_evalkit.integrations`, whose
dependency arrow ADR-0022 declares to point outward only.

## Decision

This package measures judge calibration itself, through two public
surfaces.

`agentic_evalkit.graders.measure.measure_calibration` runs a judge over a
sequence of `LabeledJudgeSample` and returns a `CalibrationArtifact`. It
takes the fingerprint from `judge.fingerprint` rather than from a
parameter, so a caller cannot type one; it always sets `calibrated_at` and
derives `expires_at` from `PROJECT_MAX_CALIBRATION_AGE_DAYS`; it calls the
judge once per sample and isolates a per-sample failure the way `EvalRunner`
does, so one bad sample cannot abort a measurement; and it counts a
non-verdict — an abstention, a refusal, a reported timeout or rate limit,
an unparseable response, a response bearing another judge's fingerprint, or
a parsed verdict carrying no score — into `abstained_count` or
`error_count` rather than into a class. A judge therefore cannot improve
its measured accuracy by declining the questions it would have got wrong.

The measurement applies the same decision function the grader applies: a
score at or above `pass_score_threshold` is a "good" verdict, defaulting to
the same value `JudgeGrader` defaults to. An artifact measured under any
other rule would faithfully describe a decision nobody makes in production.

`agentic-evalkit calibrate <labeled-set> --output <path>` is the command
form. It reads the labeled set, measures the judge, writes the artifact as
JSON, and prints the authority level and its reason. **A calibration too
thin to gate is still measured, still written, and still reported** — the
command exits `3` (`MISSING_CAPABILITY`, an existing member; no new exit
code was added) and says why, rather than refusing to produce a file. The
verdict it prints comes from calling `judge_authority` on the artifact it
has just produced; the command contains no second implementation of that
decision.

The input is a new wire model, `agentic_evalkit.models.calibration`, obeying
the ADR-0002 invariants: frozen, `extra="forbid"`, `schema_version="1"`,
and a `CalibrationLabel` StrEnum of `good`/`bad` rather than a boolean.
Reading the file goes through `LocalDatasetProvider`, which supplies the
JSON/JSONL/YAML/CSV decoders, `yaml.safe_load`, and confinement of reads to
the working directory, so no second file-reading path exists.

The artifact carries counts, identifiers and timestamps only. No prompt, no
candidate output, no judge rationale, and no exception message reaches it —
including the message of an exception raised by the judge, which is
discarded rather than recorded, the incremented `error_count` being the
whole record. This keeps the artifact structurally outside the
redaction-routing problem instead of depending on a redaction pass to catch
a leak later.

`AuthorityLevel`, `JudgeAuthority` and `judge_authority` move from
`agentic_evalkit.integrations.base` to `agentic_evalkit.graders.calibration`.
Nothing about the decision is integration-specific — it reads a
`CalibrationArtifact` and nothing else, and it encodes ADR-0007's decision
D-1 — while ADR-0022 states that nothing outside `integrations` imports
from it. `integrations.base` re-exports all three names, so every existing
import path resolves unchanged and ADR-0022's boundary is respected rather
than quietly breached by a CLI command.

## Alternatives

**Refuse to write an artifact below the 30-sample class minimum.** The
argument for it is real and worth recording: an artifact that cannot gate
is inert, a file on disk implies an accomplished measurement, and refusing
early gives the clearest possible signal that more labeling is needed.
Rejected because it introduces a third policy into a codebase that
deliberately keeps one. The ratified two-tier rule (D-1 as amended
2026-07-04) already distinguishes evidence that is present and bad from
evidence that is absent or thin, and thin evidence is the second kind — a
fact about the world that the artifact is the right place to record.
Refusing would also leave a user unable to write down the honest state of a
judge they have partially measured, which is precisely the state most
judges are in on the day someone first runs this. The exit code and the
printed level carry the signal instead, and `usability_failure_reason`
names the shortfall in the artifact itself.

**Reuse `EvalSample` as the input model.** Rejected because it carries no
candidate output — an output arrives later as a
`NormalizedExecutionResult` — and no label, since nothing in the ordinary
evaluation path has one. It also requires `source_digest` and `adapter`,
which record which adapter converted which dataset row; a hand-labeled row
came from neither, so reuse would mean asking a human to invent two
provenance values that would then be false.

**Carry the label in `EvalSample.metadata`.** Rejected because `metadata`
is explicitly the free-form field that is not used to grade, and a
ground-truth label is the most load-bearing value in the whole measurement.
It would also be untyped, so a typo would be counted rather than rejected.

**Let the caller pass `judge_fingerprint`.** Rejected: it is the one field
that must not be assertable. Reading it from the judge makes an artifact
that describes a different judge unconstructable rather than merely
discouraged.

**Compute the verdict in the CLI from the artifact's counts.** Rejected.
A second implementation that agreed on the day it was written would be free
to drift from the one the release gate consults, and the control's entire
value is that it cannot.

**Leave `judge_authority` in `integrations` and import it from the CLI.**
Rejected because ADR-0022 states the dependency arrow points outward only;
no test enforces it today, which makes an unnoticed breach more likely, not
less costly.

**Record the judge's error messages in the artifact for debuggability.**
Rejected: an exception message can quote the candidate output, and the
artifact is written to be committed and shared with people who were never
shown the labeled set. Anything worth persisting from a model or a sample
would have to route through `apply_redaction` exactly once; keeping text
out entirely is the stronger guarantee and costs only diagnostic detail
that a `--debug` traceback still provides at the moment of failure.

## Consequences

A judge can now earn gating authority through the same code path the gate
reads, which was previously unreachable. `agentic-evalkit calibrate` plus
`JudgeGrader(calibration=...)` is an end-to-end route from labeled answers
to a result that carries `hard_gate=True`.

Producing a gating calibration requires substantially more labeling than
the 30-sample class minimum suggests: at a perfect true-negative rate the
95% Wilson lower bound reaches the 0.95 floor only at 73 negative samples.
This was already true of the enforcement side; the command makes the cost
visible instead of leaving a user to discover it after labeling 30 rows.

The `pass_score_threshold` used at measurement time must match the one the
consuming `JudgeGrader` is built with. The defaults agree and a unit test
pins them together, but a caller who overrides one and not the other gets
an artifact describing a decision they do not make. This is a real sharp
edge, documented on the parameter.

`judge_authority` now has a `graders` home, so future callers inside the
package reach it without touching `integrations`. The re-export means the
move is invisible to existing code and to the published API.

A labeled set is an input file this package reads but does not produce.
Building one is a human activity, and nothing here validates that the
labels are correct — only that they are well-formed. A calibration is
evidence about a judge relative to a set of labels; it inherits whatever
care went into them.

## Validation

- `tests/unit/graders/test_measure.py` — the measurement: `calibrated_at`
  always set, `expires_at` derived from the project maximum age, the
  fingerprint taken from the judge, each verdict landing in the correct
  confusion-matrix cell, every non-verdict counted on its own and never as
  a class, per-sample isolation of a raised exception, `CancelledError`
  still propagating, the score threshold pinned to `JudgeGrader`'s default,
  a thin set yielding `ADVISORY` with the shortfall named, a measurably bad
  judge yielding `UNAVAILABLE` rather than `ADVISORY`, and a fully measured
  judge producing an artifact that lets `JudgeGrader` return
  `hard_gate=True`.
- `tests/integration/test_cli_calibrate.py` — the command: the printed
  verdict compared against `judge_authority` on the same artifact rather
  than a hardcoded expectation, a thin set still writing its artifact and
  exiting `3`, the artifact containing no prompt or candidate-output text,
  JSON/JSONL/YAML input, reads confined to the working directory, and each
  malformed-input path exiting with its classified code instead of a
  traceback.
- `tests/contract/test_models.py` — the ADR-0002 invariants that
  `LabeledJudgeSample` is checked against with every other wire model.
- `tests/contract/test_dependency_boundary.py` — the ADR-0001 boundary,
  unchanged by this work: this package still imports no modules from ARP,
  agentic-tools, or ExecutionKit.
- `uv run pytest -m "not live" --cov` — the 80% branch floor.

## Supersession

This ADR may be revisited by a later ADR that names it. Two changes would
warrant one. If a labeled set ever needs to carry per-sample text into the
artifact — a rationale, a disagreement note — the no-free-text rule stated
here is what is being reversed, and the replacement must route that text
through `apply_redaction` exactly once and say so. If the emit-below-the-
minimum decision is reversed in favour of refusing, the replacement must
also say what a user is expected to do with a partially labeled set
instead. Amending the floors themselves is ADR-0007's and ADR-0020's
business, not this one's; this ADR reads them and never restates their
values.
