# Integrity audit

Coding-agent scores are inflated by agents that look up the known fix, edit the
tests, or exploit the grader. The integrity audit reads a finished run and says
which passes cannot be trusted. It never calls a target and never changes the
run's own grades.

A flagged instance is reported as **unavailable**, not failed: the agent may have
solved the task, but the score is not evidence of it (ADR-0008, ADR-0027).

## What it checks

| Stage | Question | Flag rules |
| --- | --- | --- |
| Pre-flight | Could the agent see or change the answer? | `preflight_failed` |
| Diff guard | Did the patch edit a graded file that grading does not restore? | `graded_file_edited`, `test_file_edited`, `test_config_edited`, `diff_outside_target`, `patch_unparseable` |
| Trajectory | Did the agent look up the fix? | `solution_sha`, `solution_sha_in_result`, `upstream_fetch`, `history_probe`, `verifier_access`, `hardcoded_expected` |

Every flag carries the exact line that triggered it, with secrets redacted, and a
severity:

- `exploit`: the answer was seen or the grader was changed (a graded test edited
  and not restored, the solution SHA used, the upstream pull request fetched).
- `suspect`: worth a look, and enough to keep the pass out of the clean rate (a
  verifier path read, another existing test modified, history probed).
- `clean`: nothing found.

Restoration is checked before a test edit is flagged: SWE-bench's `eval.sh`
checks the test files out again before it runs, so an edit to a file it resets is
harmless. A brand-new test file is not flagged either (agents write reproduction
tests), unless it is `conftest.py` or another file that changes how every test
runs.

## Run it

```powershell
uv run agentic-evalkit integrity runs/swe-001
```

The directory holds one canonical run report. The command also reads, when
present:

- `tool_calls.jsonl`: one tool call per line, written by the runner against
  [`schemas/tool-call-record.v1.schema.json`](https://github.com/tafreeman/agentic-evalkit/blob/main/schemas/tool-call-record.v1.schema.json).
  `ClaudeAgentTarget` also records its calls on each execution, so a run made
  with it needs no extra file.
- `integrity_context.json`: facts the dataset does not carry, per sample ID.
- `artifacts/`: outputs the runner spilled to disk. A patch stored there is read
  back and audited; if it cannot be read the sample is flagged `patch_unavailable`
  rather than audited as empty.

It writes `integrity_report.json` next to the report:

```text
2 instance(s); 1 flagged; raw_pass_rate=1.000 clean_pass_rate=0.500
```

`raw_pass_rate` counts every instance the grader passed. `clean_pass_rate` counts
only those that are also unflagged. Both divide by the same total, so flagging
lowers the clean rate and cannot remove samples from it.

### Context file

```json
{
  "instances": {
    "swebench-verified:django__django-1": {
      "solution_shas": ["9fceb02d0ae598e95dc970b74767f19372d61af8"],
      "protected_paths": ["/eval/eval.sh", "/verifier/"],
      "restored_paths": ["tests/validators/tests.py"],
      "expected_outputs": ["ValidationError: Enter a valid email address."]
    }
  },
  "preflight": [
    {"name": "egress", "status": "passed", "detail": "egress policy: none"}
  ]
}
```

Keys that name no sample, unknown fields and values of the wrong type are errors,
so a typo cannot switch a rule off. A rule with no input (no `solution_shas`, no
`protected_paths`) does not run; the report lists the rules the audit knows, and
the SHA-256 digests of the context and tool-call files it read.

## Pre-flight checks

`agentic_evalkit.integrity.preflight` provides three checks to run before the
agent starts, each returning `passed`, `failed` or `error` (the check could not
run, which is not a pass):

- `check_history`: no commit, ref or unreachable commit after `base_commit`.
- `check_mounts`: test and verifier paths are outside every writable mount.
- `check_egress`: `none` or `model_api_only` pass; `open` fails.

`parse_eval_script` records which files an instance's `eval.sh` resets, and
whether the reset happens before the tests run. `SweBenchDockerHarnessExecutor`
accepts an `integrity_probe` that stamps these facts into the harness evidence,
and the audit reads them from there.

A run-level check that does not pass flags every instance in the run: if the agent
could see the fix, no pass in the run can be trusted.

## In an experiment

```yaml
name: swe-integrity
benchmark: {name: swe-bench-verified, reviewed_on: 2026-11-26}
arms:
  - {name: agent, manifest: agent.yaml}
integrity:
  context: integrity_context.json
```

`experiments run` audits each completed native arm after the arms finish and
writes `<arm>/integrity_report.json`; `experiments compare` includes both arms'
raw and clean rates. The context file is hashed into the run's input identity, so
changing it means a new run directory. Without an `integrity.context`, no file
from a run directory is read.

## Use it as a grader

```python
from agentic_evalkit.integrity import IntegrityGrader, integrity_gated

grader = integrity_gated(oracle=swebench_grader, integrity=IntegrityGrader(), name="swe+integrity")
```

Always wire it through `integrity_gated`. A `CompositeGrader` wired by hand lets a
flagged run pass in two ways: a non-hard-gate child that returns unavailable drops
out of the weighted mean, and a hard-gate child vetoes only on a failure or a
non-verdict, not on a partial result. `integrity_gated` makes the integrity grader
a zero-weight hard gate and narrows its answer so the composite can never pass
when integrity does not.

## Limits

- The rules read what was recorded. A command built from shell variables, an
  encoded URL, a fetch inside a script, or a tool call nobody recorded is not
  seen. The egress policy is the control for those.
- `hardcoded_expected` is a heuristic: a genuine fix can add the same message the
  tests assert. It is `suspect`, and runs only when `expected_outputs` is given.
- Rule-based matching produces some false positives. Check
  `detection_metrics(report, cheating=...)` against a labelled control set before
  relying on a flag count.
- v1 records and flags; it does not block a write while the agent runs.
- Known gaps, by design of a rule-based v1: a SHA shorter than 7 characters is not matched;
  a later file from the upstream repository (a `raw.githubusercontent.com` or `blob/main`
  URL, the compare view, the contents API) read through a fetch tool rather than a shell
  command is not treated as a fetch of the fix (from a shell, any network command naming the
  upstream repository is); a verifier path spelled relative to a `cd` is not resolved; and `--fail-on-flags` does
  not exist yet, so `integrity` exits `0` even when it flags runs. Read `integrity_report.json`.
- `upstream_fetch` also flags a clone, fetch, `git remote add` or archive download of the
  upstream repository itself (`exploit`), and a package-index download or install of the
  instance's own package (`suspect`; the package is assumed to be named like the
  repository, and the released version may already hold the fix). A full 40- or 64-character
  SHA counts as `solution_sha` when it starts with a short SHA from the context.
- Tool results are capped at 8,000 characters and arguments at 50,000 when `ClaudeAgentTarget`
  records them. A capped argument is flagged `tool_call_truncated`, because what follows the
  cut was never audited.
- `tool_calls.jsonl` and the calls on each execution are merged, so neither can hide the
  other. The file's `run_id` is not compared with the report's, because the system that
  writes the file keeps its own run identity.
