# ADR-0022: Host-Platform Integration Boundary

## Status

Accepted

## Context

This package holds a set of validity controls — calibration-gated judging,
provenance-gated comparison, typed operational/task outcome separation,
contamination canaries — that a 2026-08-04 landscape review found no surveyed
comparator documents. The same review found the package has effectively no
distribution: two GitHub stars and 410 PyPI downloads a month, against
MLflow's 40.9M and Langfuse's 24.6M. The controls are real and nobody can
reach them.

The review's conclusion was that there is no credible path from
"differentiated and undistributed" to "differentiated and adopted" that runs
*around* the established platforms, only one that runs *through* them. A team
already on MLflow will not migrate to a two-star harness to obtain one
statistical property; the same team will happily add a decorator to a judge
they already have.

This is sharpened by what MLflow actually ships. MLflow has judge
*alignment*: an optimizer that takes a judge disagreeing with human labels
and improves the agreement. It does not have judge *authority gating*: a rule
that a judge lacking proof of agreement may not block a release. Those are
different mechanisms against different failure modes, and they compose —
alignment raises the quality of the signal, gating constrains what the signal
is permitted to decide. Positioning this package as a rival to MLflow's
evaluation stack misreads that relationship. Positioning it as the validity
layer a platform user adds without leaving their platform matches it.

Building that bridge, however, points the data flow outward for the first
time. Every existing output path writes a file to a location the caller
already controls. An export transmits to a server that colleagues, other
teams, and sometimes other companies can read. The redaction guarantee, the
optional-dependency posture (ADR-0009), and the standalone boundary
(ADR-0001) all need restating for a surface that leaves the machine, because
the consequences of getting them wrong are not the same as for a local file.

## Decision

- A new `agentic_evalkit.integrations` subpackage holds one module per host
  platform. It ships `mlflow` and `langfuse` today. Each is reached through
  its own `[project.optional-dependencies]` extra of the same name, and the
  MLflow extra depends on `mlflow-skinny` rather than `mlflow`, since only
  the tracking client and the GenAI scorer surface are used.
- **No host library is imported at module scope.** `import
  agentic_evalkit.integrations.mlflow` must succeed on a machine with no
  MLflow installed. The single import site is
  `integrations.base.require_dependency`, which converts a missing package
  into a typed `IntegrationUnavailable` naming the `pip install` line that
  fixes it.
- **Every export applies redaction exactly once, through one named
  function.** `integrations.base.redact_for_export` is that function, and it
  is called as the first act of every sink. Two independently maintained
  registries — `EXTERNAL_SINKS` (every export that exists) and
  `REDACTION_ROUTED_SINKS` (every export that scrubs first) — are compared by
  `tests/contract/test_integration_redaction.py`, which additionally patches
  and counts the call so a sink cannot claim routing it does not perform.
  This mirrors the `REPORTER_FORMATS` / `REDACTION_ROUTED_FORMATS` pair and
  is deliberately not derived from it.
- **Exports default to `DEFAULT_REDACTION_POLICY`, unlike reporters, which
  default to whatever the caller passes.** The destination is a shared
  server. Opting out remains possible and explicit, by passing
  `RedactionPolicy()`.
- **The one transmit path that cannot use `redact_for_export` scrubs its own
  text instead.** A scorer (`as_mlflow_scorer`) is handed one row at a time
  and never sees an `EvalRunResult`, so there is nothing for the run-level
  pass to operate on; but the rationale it attaches to a feedback object is
  synthesized from a grade's evidence, which is not always a fixed string —
  `HarnessGrader` interpolates an exception message into it, and a caller's
  own grader may put anything there. `reporters.redact_text` applies the same
  patterns to that single string. Sinks that transmit a whole run continue to
  go through `redact_for_export`, and only through it.
- **Exports never mutate the host library's process-global state.** The
  MLflow bridge goes through `MlflowClient` with explicit run IDs rather
  than the fluent `mlflow.start_run` / `set_experiment` API, so exporting a
  result cannot redirect the caller's own subsequent logging, and works
  inside a caller's already-active run.
- **The provenance surface exported is derived, never re-listed.**
  `stats.compare.comparability_snapshot` reads the same two tables
  `compare_runs` loops over, so an exporter can never advertise a provenance
  surface narrower than the one actually enforced.
- **A demoted judge publishes under a different name, on both platforms.**
  Advisory feedback is written as `<name>.advisory` and a withheld verdict as
  a categorical marker. Both hosts aggregate by name, so renaming *is* the
  demotion: an advisory value written under the gating name has already moved
  the aggregate, and any gate reading it, before anyone reads the metadata
  explaining that it should not have counted.
- **Judge authority is re-evaluated per row, never captured once.** A scorer
  object is typically built at import and reused for the length of an
  evaluation or the lifetime of a service, so an authority resolved at
  construction would let a calibration that expires mid-run keep gating
  indefinitely. Both time-dependent halves of ADR-0007 D-1 — expiry and the
  90-day age limit — are only honest if they are asked again each time.
- **A grade outcome that is not a verdict is never rendered as a failing
  score.** `ABSTAIN`, `ERROR` and `UNAVAILABLE` become a host-platform error
  or a categorical marker, never `False` or `0.0`, so no aggregate a host
  platform computes can absorb them as task failures (ADR-0008).
- **The dependency arrow points outward only.** Nothing outside
  `agentic_evalkit.integrations` imports from it, no host platform may become
  a required dependency, and the ADR-0001 boundary is untouched: this package
  still imports no modules from ARP, agentic-tools, or ExecutionKit, and
  still reaches a system under test only through the `ExecutionTarget`
  protocol.
- **Comparison is offered only where it can be honest.** MLflow gets
  `compare_mlflow_runs`, because the full run body is stored as an artifact
  and the real `compare_runs` can be run over it. Langfuse gets no
  equivalent: it has no artifact store, and reconstructing a run body from
  trace metadata would be inference presented as measurement.

## Alternatives

1. **Build a self-hosted evaluation dashboard instead.** Rejected: MLflow
   (27.4K stars, Apache-2.0, self-hostable, 60+ framework integrations) and
   Langfuse (32.5K stars, self-hostable) already occupy that position.
   Shipping a rival would mean building a UI, a trace store, and an
   integration surface — years of work already done elsewhere — in order to
   deliver one statistical feature. The rigor is the product; the dashboard
   is not.
2. **Vendor the integrations as a separate distribution
   (`agentic-evalkit-mlflow`).** Rejected: it doubles the release surface and
   splits the contract tests away from the code they constrain, so the
   redaction tripwire above would no longer be enforced in the same CI run as
   the exporters. The extras mechanism already provides the isolation the
   separate package would buy.
3. **Reimplement `compare_runs`' provenance rules inside the MLflow bridge,
   reading tags only.** Rejected: it would produce a second, drifting
   definition of comparability whose verdicts could disagree with the
   library's own. Tags are used as a cheap pre-check that can only fail a
   pair earlier, never permit one, and the real comparison always runs.
4. **Depend on full `mlflow` rather than `mlflow-skinny`.** Rejected: the
   full distribution additionally installs scikit-learn, scipy, pyarrow and a
   web server, none of which any export path touches. `mlflow` itself depends
   on `mlflow-skinny`, so the narrower pin is already satisfied for anyone
   holding the full package.
5. **Let exports default to no redaction, matching the reporters.** Rejected:
   the reporters' default is safe because their output stays on the caller's
   own disk. Applying the same default to a shared tracking server would make
   the most dangerous destination the least protected one.

## Consequences

- A team already on MLflow or Langfuse can adopt calibration gating,
  provenance-gated comparison, and honest outcome separation without leaving
  their platform or changing their harness.
- The base install is unchanged. A user who wants neither bridge pays
  nothing, and `agentic_evalkit.integrations` remains free to import.
- Adding a third host platform requires three deliberate edits — the module,
  an `EXTERNAL_SINKS` entry, and a `REDACTION_ROUTED_SINKS` entry — and CI
  fails until the third is made and is truthful.
- Both host libraries are now development dependencies, so the bridges are
  tested and type-checked against the real client APIs rather than against a
  guess. The MLflow bridge is tested hermetically against a local tracking
  store; Langfuse has no offline mode, so its bridge is tested against the
  `LangfuseClient` protocol this package declares.
- This package now tracks two external APIs it does not control. A breaking
  change in either surfaces as a test or type failure here rather than at a
  user's first export.

## Validation

- `tests/contract/test_integration_redaction.py` asserts `EXTERNAL_SINKS`
  equals `REDACTION_ROUTED_SINKS`, that every registered sink resolves to a
  real callable, that importing the registry imports neither host library,
  and — by patching and counting — that each sink calls `redact_for_export`
  exactly once.
- `tests/unit/integrations/test_mlflow_bridge.py` runs against a real MLflow
  store on local disk with no server: it pins that a planted secret never
  reaches the exported artifact, that operational failures stay separate from
  task failures, that absent measurements are omitted rather than logged as
  zero, that every field `compare_runs` checks is exported as a tag, that a
  provably incomparable pair is refused, that a foreign MLflow run is
  refused, and that exporting leaves the caller's MLflow configuration and
  active run untouched.
- `tests/unit/integrations/test_integration_base.py` pins the three authority
  levels against the ADR-0007 D-1 two-tier rule, including that absent
  evidence yields advisory and present-but-bad evidence yields unavailable,
  and that a judge proven unreliable is never called at all.
- `tests/unit/integrations/test_langfuse_bridge.py` pins that a non-verdict
  outcome never becomes a `0.0` numeric score, that demotion changes the
  score name rather than only its metadata, and that supplying a client makes
  the Langfuse package unnecessary.
- `tests/unit/integrations/test_mlflow_bridge.py` additionally pins that a
  secret planted in a grader's rationale is scrubbed before it reaches a
  feedback object, and that `_truncate` never returns more than the limit it
  was given for any limit.
- `tests/unit/reporters/test_redaction_policy.py` pins that the sweep
  reaches every free-form field a run carries — `sample.input`,
  `sample.reference`, `sample.metadata`, `sample.expected_artifacts`,
  `execution.tool_calls`, `execution.environment_metadata`, and
  `grade.oracle_provenance` — not only the four output-side fields it
  originally covered.
- `tests/unit/integrations/test_mlflow_bridge.py` additionally pins that the
  gated scorer exposes the parameter names MLflow dispatches on (a wrapper
  declaring only `**kwargs` is handed no row data at all), that authority is
  re-evaluated per row, that an advisory verdict is renamed, and that
  `allow_cross_environment` actually reaches `compare_runs` rather than being
  refused by the tag pre-check in front of it.
- `uv run mypy` type-checks both bridges against the installed client
  libraries, which ship `py.typed`.

## Supersession

Revisit this decision if either host platform ships calibration-gated judging
natively — MLflow doing so is the specific, monitored trigger recorded
alongside the existing Inspect trigger in `docs/prior-art.md` — since the
bridge's argument rests on the combination being unavailable in one place. It
should also be revisited if maintaining the two client APIs begins to cost
more than the distribution it buys, measured against whether anyone is
actually using the extras. Any future change to the redaction-routing rule
for external sinks must supersede this ADR explicitly; loosening it inside an
exporter is not permitted.
