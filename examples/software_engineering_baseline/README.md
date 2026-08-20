# Software-engineering workflow baseline seeds

> **Status: PROPOSAL — NOT TESTED AS AGENT WORKFLOWS.** This directory is
> separate from the established Agentic EvalKit evaluation evidence. The fixture
> structure and gold/no-op controls have been exercised, but no agent has run the
> tasks and the suite is not integrated with `EvalRunner`. See `suite.json` for
> the machine-readable boundary.

This directory contains five synthetic repository tasks, one for each initial
software-engineering workflow track:

| Case | Track | Primary grade |
| --- | --- | --- |
| `SE-BF-PY-001` | Bug fix | Hidden regression tests |
| `SE-FE-PY-001` | Feature implementation | Hidden acceptance and state-transition tests |
| `SE-TG-PY-001` | Test generation | Reference pass plus seeded-mutant kill rate |
| `SE-RV-PY-001` | Code review | Structured finding precision and recall against a defect ledger |
| `SE-RF-PY-001` | Refactor | Behavioral equivalence plus an advisory structure metric |

Each fixture separates `repo/`, which is visible to the agent, from `oracle/`,
which is mounted only for grading. A future repository-task adapter should
materialize only `repo/` for the execution target. Giving the target access to
`oracle/` invalidates the sample.

`cases.jsonl` is the target-facing catalog. It deliberately contains no hidden
test source, gold patch, defect ledger, or reference implementation. Each
`oracle/oracle.json` describes the verifier-only contract.

These are seed fixtures, not a statistically sufficient benchmark. Before
publishing comparative results, add multiple repositories and difficulty bands,
run known-weak and known-capable sentinel targets, and demonstrate meaningful
score spread within every track.

Validate the catalog and fixture separation with:

```powershell
python examples/software_engineering_baseline/validate_cases.py
python examples/software_engineering_baseline/validate_controls.py
```

The validators check schema shape, unique IDs, matching directories, required
oracle controls, relative paths, and that no verifier-only canary appears in the
target-facing catalog or repository files. They also execute gold/no-op controls,
including all seeded permission mutants. They do not execute an agent or claim
the oracle implementations are already wired into `EvalRunner`.

## Target output contracts

- Bug fix, feature, and refactor tasks return a unified diff plus a short final
  summary.
- Test-generation tasks may modify only `tests/` and return a unified diff.
- Code-review tasks return JSON matching `repo/review-output.schema.json`; they
  do not modify the repository.

## Grading policy

Executable or ledger-based checks are primary. Operational failures remain
separate from task failures. Explanation quality and refactor elegance may be
reported as advisory signals, but they cannot compensate for a failed objective
gate.
