# REVIEW.md
<!-- Read by the AI reviewer from the BASE branch only. Byte-stable. Owner: @tafreeman. Changes route to high_risk. -->

## What this repo is
`agentic-evalkit` is a Python 3.11+ library and Typer CLI that grades the output of agentic systems and records the evidence behind each verdict. It is an evaluation harness, not an agent framework: it never runs an agent's reasoning loop and reaches a system under test only through `ExecutionTarget`. Its value is being hard to overclaim a result.

## Invariants a change must not break
- No import of sibling systems (ARP, agentic-tools, ExecutionKit) — lives in `src/agentic_evalkit/**`. Checked: `tests/contract/test_dependency_boundary.py`.
- Every wire model is `frozen=True`, `extra="forbid"`, carries `schema_version`, uses tuples and `StrEnum` status, never a boolean status — lives in `src/agentic_evalkit/models/base.py` and `models/**`. Checked: `tests/contract/test_models.py`.
- A judge may set `hard_gate=True` only with full calibration and the ratified floor (TNR >= 0.95, TPR >= 0.85, age <= 90 days). Bad evidence demotes to `UNAVAILABLE`; absent evidence leaves the judge advisory — lives in `src/agentic_evalkit/graders/calibration.py`, `graders/judge.py` (sets `hard_gate`), `graders/composite.py`. Checked: `tests/unit/graders/test_judge_calibration_floor.py`, `tests/unit/graders/test_composite.py`.
- ABSTAIN, ERROR and UNAVAILABLE grades are excluded from the weighted mean, never scored as zero; any failing `hard_gate` component fails the composite — lives in `graders/composite.py`. Checked: `tests/unit/graders/test_composite.py`.
- Operational failures (error, timeout, cancel) are never folded into task failures, and `compare_runs` refuses runs whose provenance differs, except the sanctioned `allow_cross_environment=True` opt-in, which waives only the environment and code fingerprints and records them in `waived_provenance_fields` (ADR-0015) — lives in `stats/compare.py`, `provenance.py`, `manifest.py`. Checked: `tests/unit/stats/test_compare.py`, `tests/contract/test_provenance_drift.py`.
- Persisted reports are redacted through `apply_redaction` before any reporter writes, so no format leaks on its own; large-output spill has a separate `_redact`, so keep the two in step — lives in `src/agentic_evalkit/reporters/base.py`, `reporters/__init__.py`, `runner.py`. Checked: `tests/contract/test_redaction_enumeration.py`, `tests/contract/test_integration_redaction.py`.
- YAML is parsed only with `yaml.safe_load`; `trust_remote_code` is never enabled; secrets arrive only through runtime credential hooks — lives in `manifest.py`, `datasets/local.py`, `datasets/huggingface.py`. Checked: ruff `S506` flags `yaml.load` in the CI lint job; nothing checks `trust_remote_code` or credential handling.
- Dataset cache correctness is checksum-on-read, not `Path.replace()` atomicity; `_cache_io.py` retries exactly `PermissionError` and `FileNotFoundError` — lives in `src/agentic_evalkit/datasets/_cache_io.py`, `datasets/cache.py`. Checked: `tests/integration/test_cache_read_race.py`, `tests/unit/datasets/test_cache.py`.
- Default tests are hermetic (`-m "not live"`; live tests only under `tests/live/`) — lives in `pyproject.toml`, `.github/workflows/ci.yml`. Checked: `tests/contract/test_live_test_boundary.py`.

## High-risk areas and what to look for there
| Path | Look for |
|---|---|
| `src/agentic_evalkit/graders/**` | Any path where a sample can pass that should fail: a default that grades PASS, an error or abstain counted as pass, a loosened match, a gate that no longer fails the composite, a judge gating without calibration. |
| `src/agentic_evalkit/stats/**` | Interval, power, reliability or comparison changes that make a difference look significant or a run comparable when it is not. |
| `src/agentic_evalkit/provenance.py` | Fingerprint inputs dropped or made optional. |
| `src/agentic_evalkit/models/**` | Loosened `frozen` or `extra="forbid"`, a changed `schema_version`, a new `EvalRunManifest` or `SamplingPolicy` field without its drift-test category, a status turned into a boolean. |
| `examples/**/oracle/**` | Gold patches or hidden tests changed or exposed to the system under test; controls altered so a baseline passes or fails wrongly. |
| `src/agentic_evalkit/datasets/**`, `targets/**`, `integrations/**`, `manifest.py` | `trust_remote_code`, `yaml.load`, credentials outside runtime hooks, anything logged or exported unredacted. |
| `.github/**` | Unpinned actions, widened `permissions`, secrets in new places, a changed `pytest -m "not live"` invocation, release or publish steps made automatic. |

## Conventions the reviewer should assume
- Run everything through `uv run` (`pytest`, `ruff check`, `ruff format --check`, `mypy`); `mypy` is strict and scoped to the package by `pyproject.toml`.
- Ruff line length is 100; the 80% coverage floor must not drop; keep `--cov-config=pyproject.toml` in `addopts`.
- Core seams (`Grader`, `ExecutionTarget`, `DatasetProvider`, `BenchmarkAdapter`, `Reporter`) are `@runtime_checkable` Protocols in their subpackage's `base.py`; `EvalRunner` constructs nothing, the CLI builds and injects.
- Adding a report format means editing both `REPORTER_FORMATS` and `REDACTION_ROUTED_FORMATS` in `reporters/__init__.py`, by design.
- A new ADR uses the seven-heading template and `Accepted` status (`tests/contract/test_adrs.py`); its mkdocs nav entry is not checked by CI.

## Do not flag (repo-specific)
- Pydantic field types imported at runtime instead of under `if TYPE_CHECKING:`; `runtime-evaluated-base-classes` is pinned in `pyproject.toml` on purpose.
- Per-file ruff ignores (`S101`/`S105` in tests, `N818` in `errors.py`/`artifacts.py`); they are deliberate.
- Fake tokens in redaction tests (for example `hf_AbCdEfGh0123456789`).

## Severity calibration for this repo
- Any change that can make a grader pass a sample it should fail is critical.
- Critical: a persisted output format that bypasses `apply_redaction`, a judge gating below the calibration floor, remote code enabled or a credential leaked, an operational failure scored as a task failure, a forbidden sibling import.
- High: a weakened provenance or comparability check, a loosened wire model, a new unpinned or over-permissioned workflow step, an oracle exposed to the target.
- Medium: missing tests for new grader or stats branches, a coverage drop.
- Low: style and wording that ruff and doc tests cover.

## Out of scope for AI review
- Gitignored local output (`_bmad*`, ad-hoc reports and run outputs). Tracked evidence such as `scripts/reports/**` stays in scope: flag corrupted pinned artifacts or newly added generated runs.
- `uv.lock` contents; dependency bumps are Dependabot's.
- Prose quality of `docs/` and ADR wording.
