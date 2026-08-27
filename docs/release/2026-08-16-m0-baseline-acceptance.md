# M0 Baseline Acceptance Record

**Date:** 2026-08-16

**Master plan:** `docs/plans/2026-08-16-agentic-evalkit-master-delivery-plan.md`

**Gate:** G0 `CLOSED`

## Scope and interpretation

This record closes M0 only. It freezes repository identities, reconciles the
EvalKit candidate, refreshes ARP's evaluation usage map, captures the current
EvalKit and ARP baselines, and classifies existing evidence. It is not a
release, calibration-authority, consumer-cutover, or legacy-removal decision.

Commands ran on Windows with PowerShell, Python 3.12.11 for EvalKit, Python
3.11.9 for ARP, uv 0.11.23, Node 25.9.0, and npm 11.12.1. Remote references
were refreshed with `git fetch --prune --tags origin` before identities were
recorded.

## MDP-001 — Repository ownership and starting state

| Repository | Shared checkout state | Pinned committed state | Remote and tag | Package/dependency baseline |
|---|---|---|---|---|
| Agentic EvalKit | `C:\Users\tandf\source\agentic-evalkit`; `main`; modified `docs/plans/README.md`; three untracked plan/tracker files; untracked `examples/software_engineering_baseline/` | `fcbcd365e1f58eb4ee7f6392fd89d4d208a0b28d`, equal to `origin/main` after a safe fast-forward | `https://github.com/tafreeman/agentic-evalkit.git`; latest version-sorted tag `v0.3.0` at `cb706a0a32e7c5a3f06eb7a21ddeddb5f4e9db64` | project and lock version `0.3.0` |
| Agentic Runtime Platform | `C:\Users\tandf\source\agentic-runtime-platform`; `rag_removal`; 3 commits ahead of `origin/main`; 21 modified documentation/style files preserved | baseline worktree pinned to `27bcdbb63d558a6b32fe90b7459f1eaef6db5ac1`; its merge base is current `origin/main` `b41567ca2cf047b050ca034ce6c2966d2552de69` | `https://github.com/tafreeman/agentic-runtime-platform.git`; latest version-sorted tag `v0.3.0` at `7fb2b59fe620c183992aa84d00110c5af8ab43a8` | root `agentic-tools` `0.1.0`; runtime `0.1.0`; legacy eval `0.3.0`; lock and CI constraint pin EvalKit `0.3.0`; runtime extra declares `>=0.3.0,<0.4.0`; lock resolves ExecutionKit `0.3.0` |
| ExecutionKit | `C:\Users\tandf\source\executionkit`; clean `main` after fast-forward; the prior squash-equivalent `landing_css` commit remains recoverable on its local branch | `b6ba189dbbe9a40515698795f30af7161f374f36`, equal to `origin/main` | `https://github.com/tafreeman/executionkit.git`; `v0.3.0` at `80749945a0e36e65a1dc4a798a3423f774ab3d2e` | `executionkit.__version__ == "0.3.0"` |
| Financial Scenario Engine | `C:\Users\tandf\source\financial-scenario-engine`; `main`; modified `docs/index.md` preserved | `c100d350d2c84648d48650ed38a2c112200e86fd`, equal to `origin/main` | `https://github.com/tafreeman/financial-scenario-engine.git`; `v0.1.0` at `94067b0ba75ac81a6adb500af71120a60a18f6f6` | root and client package version `0.1.0` |

The pre-existing modified and untracked files above were neither staged nor
overwritten. A stale, zero-byte ExecutionKit `.git/index.lock` dated 2026-08-10
was removed only after confirming that no Git process was running.

## MDP-002 — Reconciled EvalKit candidate

The shared EvalKit `main` was safely fast-forwarded from `5f6b524` to fetched
`origin/main` `fcbcd365`. The incoming paths did not overlap the user's dirty
planning or example paths.

A separate clean detached worktree was created at
`C:\Users\tandf\source\agentic-evalkit-m0-candidate` with exact SHA
`fcbcd365e1f58eb4ee7f6392fd89d4d208a0b28d`.

The reconciled calibration history is:

- `0bb0141bf04f76486868b719840bf69b06824590` — measurement feature;
- `c424c15eddbc3645c03457f53a1efee472b34ac8` — reviewed coverage/preparation fixes;
- `fcbcd365e1f58eb4ee7f6392fd89d4d208a0b28d` — merge commit on `origin/main`.

The clean candidate has no tracked diff and does not contain the untracked
`examples/software_engineering_baseline/` fixtures.

## MDP-003 — Current ARP evaluation usage map

The inventory was refreshed against committed ARP SHA `27bcdbb6`, not copied
from the July analysis.

### Production and runtime consumers

- `agentic-workflows-v2/agentic_v2/models/llm.py` imports the legacy
  `LLMClientProtocol` unconditionally.
- `agentic-workflows-v2/agentic_v2/scoring/step_scoring.py` imports the legacy
  rubric loader and scorer behind an optional-import guard and owns the active
  step-scoring listener path.
- `scripts/eval_gate.py` imports the legacy scorer/rubrics and runs the four
  deterministic and optional live golden cases.
- `scripts/score-trace.py` imports legacy scoring and report writers.
- `agentic-workflows-v2/agentic_v2/scoring/evalkit_bridge.py` is the current
  partial EvalKit bridge. It still reads legacy rubric shapes; it is not the
  permanent configuration-first golden gate.

### Packages, data, tests, and workspace wiring

- `agentic-v2-eval/` remains the complete legacy distribution, source, rubric,
  benchmark, runner, reporter, sandbox, and test tree.
- Root `pyproject.toml`, `uv.lock`, and `ci-constraints.txt` retain the legacy
  workspace package and released EvalKit `0.3.0` dependency.
- `datasets/default/golden_cases.json` and `datasets/default/README.md` define
  the four legacy workflow cases and their saved-output semantics.
- Runtime and cross-package coverage remains in
  `agentic-workflows-v2/tests/test_evalkit_bridge.py`,
  `agentic-workflows-v2/tests/contract/test_evalkit_boundary.py`, legacy-import
  scoring tests, and `tests/e2e/test_cross_package.py`.

### CI, build, release, and development wiring

The current deletion/cutover inventory includes:

- `.github/workflows/eval-package-ci.yml`, plus legacy installs or paths in
  `ci.yml`, `dependency-audit.yml`, `deploy.yml`, and `sbom.yml`;
- `.github/dependabot.yml`, `.github/CODEOWNERS`,
  `.pre-commit-config.yaml`, `.devcontainer/devcontainer.json`, `Dockerfile`,
  `justfile`, and `release-manifest.toml`;
- workspace test-runner package enumeration and documentation-stat generation;
- root and package documentation, including `README.md`, `CONTRIBUTING.md`,
  `CLAUDE.md`, `docs/architecture-eval.md`,
  `docs/deep-dive-agentic-v2-eval.md`, `docs/evaluation/`, architecture,
  onboarding, deployment, development, roadmap, and ADR material.

No item in this inventory was deleted or rewired in M0. ARP's independent
server evaluation pipeline and benchmark data remain separately owned and are
not automatic deletion targets.

## MDP-004 — EvalKit baseline

All commands ran from the clean candidate SHA `fcbcd365` after
`uv sync --all-groups --frozen`.

| Command | Result |
|---|---|
| `uv run --no-sync ruff check .` | PASS; cache-write warning only |
| `uv run --no-sync ruff format --check .` | PASS; 162 files formatted; cache-write warnings only |
| `uv run --no-sync mypy` | PASS; 75 source files |
| `uv run --no-sync pytest -m "not live" --cov --cov-branch --cov-report=term-missing` | PASS; 1,147 passed, 6 live tests deselected, 1 warning; 94.16% coverage |
| `uv run --no-sync mkdocs build --strict` | PASS |
| `uv build` | PASS; `agentic_evalkit-0.3.0.tar.gz` and wheel built |
| `uv run --no-sync pytest tests/integration/test_clean_wheel.py -v` | PASS; isolated wheel install and CLI smoke |

The six deselected cases are explicitly marked live tests; they are not counted
as runtime evidence for this gate.

## MDP-005 — ARP baseline

Commands ran from the clean detached worktree
`C:\Users\tandf\source\agentic-runtime-platform-m0-baseline` at committed SHA
`27bcdbb63d558a6b32fe90b7459f1eaef6db5ac1` after a frozen all-package,
all-extra, all-group uv sync.

| Surface | Command/result |
|---|---|
| Four deterministic golden workflows | `uv run --no-sync python scripts/eval_gate.py --cases datasets/default/golden_cases.json --threshold 0.80` — PASS; scores `1.0`, `0.8`, `0.8`, `0.8`; aggregate `0.85` |
| Legacy package | 273 tests PASS; 86.27% coverage; Ruff, format, and strict mypy PASS; sdist and wheel build PASS |
| EvalKit bridge/boundary | 28 tests PASS against installed EvalKit `0.3.0` |
| Runtime CI subset and coverage | 3,924 passed, 5 skipped, 53 deselected, 2 xfailed; 84.63% coverage; explicit 80% gate PASS |
| Runtime full suite | 3,964 passed, 18 skipped, 2 xfailed; 10 warnings |
| No-LLM path | deterministic validate/run PASS (`processed_text: Hello from CI`, `step_count: 13`); 6 LangChain tests PASS; 3,924-test unit subset PASS |
| Root cross-package E2E | 18 tests PASS |
| Tools suite | 418 passed, 2 skipped, 11 explicitly deselected stale tests |
| Runtime quality | Ruff PASS; strict engine/contracts mypy PASS; suppression ratchet PASS |
| UI | npm clean install reports 0 vulnerabilities; production build PASS; 446 tests PASS; configured coverage gate PASS |
| Documentation references/stats | local reference check PASS; generated doc stats current |
| Strict docs | FAIL on the pinned committed branch: removed `docs/rag/index.md` is still linked from seven documents; Windows also lacks the native Cairo library required by the imaging plugin |
| Pre-commit | FAIL: Black/Ruff/docformatter would modify committed files, and detect-secrets reports committed fingerprint/test/config candidates. Mypy hook passes. All hook-generated tracked changes were restored; the baseline worktree is clean at the pinned SHA. |

These failures are captured baseline findings. They do not become release
exceptions and must be resolved before a later ARP release/cutover gate can
claim a green required matrix.

## MDP-006 — Evidence classification

| Evidence | Classification | Reason |
|---|---|---|
| `reports/2026-07-26-agent-workflow-eval/` | `RUNTIME_VERIFIED`, `ADVISORY_ONLY` | The report records 48/48 completed executions and 48 passing grades for one `review_code` step, but every grade has `hard_gate=false` and `judge_calibration_ref=null`. It proves plumbing and an advisory measurement, not judge authority or four-workflow coverage. |
| `examples/software_engineering_baseline/` | `STRUCTURAL_VERIFIED`, `UNVALIDATED` | The untracked suite declares `proposal` and `NOT TESTED AS AGENT WORKFLOWS`. Its structural/gold/no-op validators do not establish an executed Agentic EvalKit or ARP result. It is excluded from release and integration claims. |
| M0 EvalKit and ARP command results above | `RUNTIME_VERIFIED` baseline | They establish the current reproducible starting behavior at pinned SHAs. They do not authorize calibration, release, cutover, or deletion. |

## G0 decision

G0 is `CLOSED` because all four repositories have pinned identities and
preserved inventories, EvalKit and ARP commands/results are archived, current
ARP consumers and deletion targets are enumerated, and advisory/unvalidated
evidence is labeled accurately. M1 remains `NOT_STARTED`.
