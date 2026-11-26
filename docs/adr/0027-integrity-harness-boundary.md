# ADR-0027: Integrity harness boundary

## Status

Accepted

## Context

Coding-agent scores are inflated by agents that look up the known fix, edit the
tests, or exploit the grader. A pass produced that way is not evidence the task
was solved, yet the oracle (the official test harness) reports it as a pass.
Existing tools each cover one stage and none audits a real run end to end.

The package must audit a finished run without becoming an agent framework
(ADR-0006) and without importing the system that ran the agent (ADR-0001). It
also must not turn a suspicious run into a task failure: an operational or
trust problem is never folded into a failed task (ADR-0008). Finally, a
composite grader (ADR-0007) can let a flagged run pass if an integrity grader
is wired in by hand: a non-hard-gate child that returns `UNAVAILABLE` drops out
of the weighted mean, and a hard-gate child vetoes only on `FAIL` or a
non-verdict, so a hard-gate `PARTIAL` beside an oracle `PASS` still passes.

## Decision

This package adds `agentic_evalkit.integrity`, three stages that each read
plain data and return flags:

1. `preflight`: the repository has no commit, ref or unreachable commit after
   `base_commit`; test and verifier paths are outside every writable mount; the
   network egress policy is recorded; and what an instance's `eval.sh` resets
   from the base commit is parsed and recorded. Each check reports `passed`,
   `failed` or `error` (it could not run, which is never a pass), and a check
   that does not pass flags every instance it covers.
2. `diff_guard`: a patch that edits a file the grader reads, or `conftest.py` or
   `pytest.ini`, is `exploit` unless grading restores that file first. Modifying,
   renaming or deleting another existing test file, or changing pytest settings
   in shared config, is `suspect`. A brand-new test file is not flagged. Patches
   are parsed hunk by hunk, so renames, deletions and quoted paths are seen and
   a diff that yields no readable path is itself flagged.
3. `trajectory`: tool calls are scanned for the solution SHA, fetches of the
   upstream pull request, issue or commit, reads or writes of verifier paths,
   history probes past `HEAD`, and expected outputs hard-coded into non-test
   files.

The detectors are pure and policy-free like `graders/contamination.py`
(ADR-0013): they report evidence and never choose a `GradeStatus`. Every flag is
built through one function, `integrity.flags.make_flag`, which redacts the
evidence line with the default redaction policy and bounds its length.

Two versioned files cross the boundary with whatever runs the agent, both at
`schema_version "1"` (ADR-0002): `tool_calls.jsonl`, one `ToolCallRecord` per
line, published as `schemas/tool-call-record.v1.schema.json`, and
`integrity_report.json`, an `IntegrityReport` with `raw_pass_rate`,
`clean_pass_rate` and per-instance `IntegrityFlag` entries of severity `clean`,
`suspect` or `exploit`. No `GradeStatus` value is added.

The audit fails closed. A patch the runner spilled to the artifact store is read
back and checked against its digest, and is flagged `patch_unavailable` if it
cannot be. Tool calls that cannot be parsed are flagged `tool_calls_unparseable`.
A named context file that does not exist, a context value of the wrong type, a
context key or tool-call record that names no sample, and a tool-call file that
mixes runs are errors. The report records the harness version, the rules it knows
and the digests of the context and tool-call files it read.

A flagged run grades `UNAVAILABLE`, never `FAIL`. The integrity grader is wired
only through `integrity_gated()`, which adds it as `hard_gate=True`,
`weight=0` and narrows its result to `PASS`, `ERROR` or `UNAVAILABLE`, so the
composite can never grade `PASS` when integrity is not `PASS`. Both rates share
one denominator, so flagging lowers `clean_pass_rate` without removing samples.

Both `suspect` and `exploit` keep an instance out of `clean_pass_rate`; the
report keeps the severity so the cost of `suspect` alone can be measured before
any live blocking is added. v1 records and flags; it does not block.

`agentic-evalkit integrity <run_dir>` writes the report. An experiment may
declare an `integrity:` block, which runs the audit on each completed native arm
after the arms finish; the context file it names is part of the run's input
identity. `ClaudeAgentTarget` records its tool calls on the normalized result.

## Alternatives

- **Detectors inside the grader or the runner.** Rejected: the same logic is
  needed by the CLI, experiments and external hosts, and a grader that also owns
  policy repeats the composite-wiring trap this ADR exists to close.
- **A new `GradeStatus` such as `FLAGGED`.** Rejected: it breaks every consumer
  that switches over the enum and adds nothing `UNAVAILABLE` does not already
  say.
- **Flag as `FAIL`.** Rejected: the agent may have solved the task; the score is
  untrustworthy, not wrong (ADR-0008).
- **Add the integrity child to a composite by hand.** Rejected: both traps above
  were reproduced; `integrity_gated` is the only supported wiring.
- **An LLM judge for the trajectory.** Deferred: rule-based detection first, so
  false positives are measurable.
- **Calling the host system's tracing as the record.** Rejected: the host
  records step-level data with redacted arguments; the detectors need the full
  call, so the host writes `tool_calls.jsonl` against a published schema.

## Consequences

- A host that runs agents can adopt the audit by writing one JSONL file and
  reading one JSON report, with no shared imports.
- `clean_pass_rate` is always at most `raw_pass_rate` over the same
  denominator, so reports cannot show the audit improving a score.
- Detectors depend on what the host records. A run with no tool calls can only
  be audited by its patch, and the report says nothing about calls it never saw.
  Tool names are matched case-insensitively so the names Claude Code records and
  snake_case names are both read. A command built from shell variables, an encoded
  URL or a fetch inside a script is not seen; the egress policy is the control for
  those.
- `hardcoded_expected` is a heuristic: a genuine fix can add the same message the
  tests assert. It is `suspect` and runs only when expected outputs are supplied.
- Pre-flight reads the audited repository, which is untrusted input. Git runs with
  the repository's own config and hooks off, a timeout, and only a hex revision.
- Rule-based matching will miss a paraphrased lookup and will raise some false
  positives. Restoration is checked before a test edit is flagged, and
  precision is reported against a control set rather than assumed.
- Training-data contamination scoring, grader fuzzing and live blocking of test
  writes are not part of this decision.

## Validation

- `tests/unit/integrity/test_gate.py` pins the gate over all 36 status pairs and
  reproduces both hand-wiring traps.
- `tests/unit/integrity/test_integrity_acceptance.py` runs 10 cheating and 10
  clean trajectories: all 10 cheating runs flagged with their evidence line, no
  clean run flagged, including plain `git log`.
- `tests/unit/integrity/test_integrity_diff_and_preflight.py` covers restoration,
  `eval.sh` parsing and the git-history check against a real repository.
- `tests/unit/integrity/test_integrity_report_and_grader.py` checks the report
  arithmetic and that a malformed tool call is an error, not a silent pass.
- `tests/unit/integrity/test_integrity_regressions.py` pins the findings of the
  pre-merge review: tool names as the targets record them, fail-closed input
  handling, severity tiers, redaction of tokens in evidence, and linear time on
  hostile input. `detection_metrics` computes precision and recall against a
  labelled control set; the acceptance test asserts both are 1.0 on its batch.
- `tests/contract/test_integrity_boundary.py` requires that flags are only built
  through `make_flag`, that the detectors import only the standard library,
  pydantic and package contracts, and that the committed schema matches `ToolCallRecord`.

## Supersession

A later ADR may change the severity policy, add blocking, or add a judge-based
stage, naming this ADR. The two file contracts keep `schema_version "1"` until a
field is removed, renamed or changes meaning; additions stay optional.
