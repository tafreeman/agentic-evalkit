# CLAUDE.md

`agentic-evalkit` is a standalone Python 3.11+ library + Typer CLI that evaluates agentic
systems with reproducible, evidence-first grading. It is an evaluation harness, **not** an
agent framework: it never runs an agent's reasoning loop and reaches a system-under-test
only through the `ExecutionTarget` protocol (callable / subprocess / HTTP). Its core value
is being structurally hard to overclaim a result — treat every rule below as load-bearing.

## Commands (always via `uv run`; PATH tools are unreliable here)

- `uv run pytest -m "not live" --cov --cov-report=term-missing` — default suite is hermetic; 80% branch floor
- `uv run ruff check .` and `uv run ruff format --check .`
- `uv run mypy` — strict
- Full pre-PR matrix and release gates: see [CONTRIBUTING.md](CONTRIBUTING.md); one-shot release checks: `.claude/skills/release-gate/`

## Invariants (enforced by `tests/contract/` — executable architecture)

- Never import sibling-system packages under `src/agentic_evalkit`; the forbidden import
  roots are pinned in `tests/contract/test_dependency_boundary.py` (ADR-0001).
- Every wire model: `frozen=True`, `extra="forbid"`, `schema_version="1"`, `tuple` collections,
  `StrEnum` status — never a boolean status, never in-place mutation (ADR-0002).
- A judge may `hard_gate=True` only under full calibration **and** the ratified floor
  (TNR ≥ 0.95, TPR ≥ 0.85, age ≤ 90 days); anything less demotes to `UNAVAILABLE` (ADR-0007).
- Operational failures (error/timeout/cancel) are never folded into task failures;
  `compare_runs` is provenance-gated (ADR-0008).
- Redaction routes every persisted format through `apply_redaction` exactly once;
  `yaml.safe_load` only; secrets only via runtime credential hooks; never `trust_remote_code`.
- Cache correctness = checksum-on-read, never `Path.replace()` atomicity; parallel runs use a
  per-worker `AGENTIC_EVALKIT_CACHE_DIR` (D-2, 2026-07-04).
- Tests stay hermetic (`-m "not live"` default); never commit generated reports or `_bmad*` output.

## Deeper context

- **Tracked source of truth:** `docs/adr/0001`–`0009` + `docs/specs/` + [CONTRIBUTING.md](CONTRIBUTING.md).
- **Generated agent context (gitignored, BMAD-regenerated; present on maintainer machines):**
  `_bmad-output/project-context.md` — the full 50-rule agent rulebook — and
  `docs/codebase/architecture.md`. Prefer them when present; when absent, the ADRs govern.
