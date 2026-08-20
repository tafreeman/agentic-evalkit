# ADR-0025: Claude Subscription Execution Target

## Status

Accepted

## Context

Every built-in target reaches the system under test through something the
operator already owns: a Python callable, a subprocess, an HTTP endpoint, or an
MCP stdio server. Each of those assumes the operator has already stood the
system up and, where a hosted model is involved, that they hold an API key for
it.

That excludes a whole class of operator. Someone who pays for a Claude
subscription rather than API credits cannot express "grade Claude on this
dataset" at all. Their options today are to buy API access they do not otherwise
need, or to hand-write a `CallableTarget` shim around a client they must
themselves authenticate — which puts credential handling, usage accounting, and
failure classification in operator code, outside every guarantee this package
makes about evidence.

Subscription credentials are not an environment variable. They live in the
Claude Code CLI's own credential store, and `claude-agent-sdk` — which drives
that CLI — is the only supported way to spend them programmatically. There is no
documented wire contract that the `httpx` client this package already carries
could speak to reach the same place, so unlike ADR-0021's hand-rolled MCP client
there is no "write it ourselves and add no dependency" option here.

ADR-0006 fixed `ExecutionTarget` as the only system-under-test boundary and
requires that adding a new built-in target kind supersede it; ADR-0021 already
effected that once in adding a fourth.

## Decision

- Ship a fifth built-in adapter, `ClaudeAgentTarget`
  (`src/agentic_evalkit/targets/claude_agent.py`), which runs a prompt through
  the Claude Agent SDK and normalizes the outcome like every other target.
- **Optional extra, not a dependency.** `claude-agent-sdk` sits behind the
  `claude` extra. The base install is unchanged. Grading Claude is one target
  among five, and the SDK additionally requires the Claude Code CLI on `PATH` —
  a machine-level prerequisite pip cannot express — so forcing it on every
  install would trade a working default for a broken one.
- **Fail at construction, not mid-run.** A missing extra raises `TargetFailure`
  with the install instruction when the target is built. A missing CLI is
  recognized during execution and reported with the sign-in instruction rather
  than a bare import or transport error.
- **Injected callable, mirroring the HTTP target.** The SDK entry point is a
  constructor argument defaulting to `claude_agent_sdk.query`, the same way
  `HttpTarget` is handed an already-configured client. The whole test suite
  drives the target through that seam, so no test needs a CLI, a sign-in, or a
  network.
- **Tools off by default.** The harness is invoked with an empty tool set and an
  empty allow-list, so grading an answer cannot touch the filesystem, a shell,
  or the network. An operator who is evaluating agentic behaviour opts in
  explicitly via `allowed_tools`.
- **Operational failures stay operational (ADR-0008).** An exhausted
  subscription rate-limit window, an assistant-level SDK error, and a failed
  harness run each produce an `ERROR` result. None may be graded as an empty
  answer, because a spent usage window is not a wrong answer.
- **The fingerprint covers the whole ask.** Model id, system prompt, effort, tool
  allow-list, turn ceiling, and target name are folded into
  `target_fingerprint`, so `compare_runs` refuses to compare across any change
  that alters what the model was asked to do.
- **Credentials never enter this package.** Resolution belongs entirely to the
  CLI. Nothing here reads, stores, or forwards a credential, so none can reach a
  report.
- **Wire models reused unchanged.** `EvalSample` in;
  `NormalizedExecutionResult` with `ExecutionStatus` out. No new wire model, and
  no change to any existing one.

## Alternatives

**Require an API key from subscription holders.** The status quo. It keeps the
dependency story clean at the cost of leaving that class of operator unable to
use the package at all for its most obvious subject.

**Add the Anthropic API SDK instead.** It reaches the Messages API with full
sampling control, which would give better run-to-run reproducibility than the
harness offers. But it authenticates with an API key or a CLI-managed API
profile, neither of which is the subscription sign-in — so it solves a different
problem. It remains the right choice if sampling control ever matters more than
subscription reach.

**Hand-roll a client, as ADR-0021 did for MCP.** Rejected because it is not
possible rather than merely costly: the credential exchange is internal to the
CLI and has no documented wire contract. Reimplementing it would mean depending
on a private surface that can change without notice.

**Make it a `CallableTarget` recipe in the docs.** Rejected because it puts
credential handling, usage accounting, and the operational-versus-task failure
distinction into operator code, where none of this package's guarantees reach.

**Ship it in the base install.** Rejected because the CLI prerequisite cannot be
expressed as a package dependency, so a base install would advertise a target
that fails on most machines.

## Consequences

- An operator with only a Claude subscription can grade Claude directly, with no
  API key and no shim.
- Results carry token counts, cost, latency, model name, and the harness session
  id as a trace reference, so a subscription run is as auditable as an HTTP one.
- `environment_metadata` records `auth: claude-subscription`, so a reader of the
  evidence can tell which credential class produced a number.
- **Runs are less reproducible than an API-key target.** The harness exposes no
  temperature and no seed, so a configuration cannot be pinned the way an API
  client can and repeat runs vary by the model's own nondeterminism. Operators
  should use multiple attempts and report the spread rather than treat a single
  run as definitive. This is a real reduction in evidence strength and is
  documented at the target, in the guide, and here.
- The fingerprint cannot detect a silent server-side model revision under a
  stable model id. That is a limit of evaluating a hosted model, not of this
  adapter, and it applies equally to any hosted target.
- A new failure mode — subscription rate-limit exhaustion — appears in results as
  `ERROR`, and a long run can hit it partway through.
- The `claude` extra must track the SDK's major version; a breaking SDK release
  is a maintenance obligation this package did not previously carry.

## Validation

- `tests/unit/targets/test_claude_agent.py` covers protocol conformance, text
  assembly across multiple assistant messages, telemetry mapping, structured
  output, prompt-field extraction and its two failure shapes, option
  forwarding, tools-off-by-default, fingerprint stability and sensitivity to
  every setting that changes the ask, timeout mapping, rate-limit rejection
  versus warning, assistant and harness error mapping, traceback exclusion from
  recorded errors, and the missing-extra construction failure.
- The replayed messages are the SDK's own dataclasses, so an upstream field
  rename fails the suite rather than passing it and failing in production.
- `tests/contract/test_dependency_boundary.py` passes over the new module: the
  Agent SDK is a third-party package, not one of the sibling systems that
  contract forbids.
- `tests/contract/test_adrs.py` registers prefix 0025;
  `tests/contract/test_public_docs.py` stays green.
- `uv run mypy` (strict) covers the module; the 80% branch-coverage floor
  applies.

## Supersession

Adding a non-subscription authentication path to this target, persisting a
harness session across samples, enabling tools by default, reading credentials
inside this package, or grading a rate-limit rejection as a task failure rather
than an operational one is a material change and must supersede this ADR. This
ADR itself effects the supersession that ADR-0006 requires for adding a new
built-in target kind; ADR-0006 otherwise remains in force, as does ADR-0021.
