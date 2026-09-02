# AGENTS.md

`agentic-evalkit` is a standalone Python 3.11+ library + Typer CLI that evaluates agentic
systems with reproducible, evidence-first grading. It is an evaluation harness, **not** an
agent framework: it never runs an agent's reasoning loop and reaches a system-under-test
only through the `ExecutionTarget` protocol (callable / subprocess / HTTP). Its core value
is being structurally hard to overclaim a result — treat every rule below as load-bearing.

## Commands (always via `uv run`; PATH tools are unreliable here)

- `uv run pytest -m "not live" --cov --cov-report=term-missing` — default suite is hermetic; 80% branch floor
- `uv run ruff check .` and `uv run ruff format --check .`
- `uv run mypy` — strict
- Full pre-PR matrix and release gates: see [CONTRIBUTING.md](CONTRIBUTING.md); one-shot release checks: `.Codex/skills/release-gate/`

## Invariants (enforced by `tests/contract/` — executable architecture)

- Never import sibling-system packages under `src/agentic_evalkit`; the forbidden import
  roots are pinned in `tests/contract/test_dependency_boundary.py` (ADR-0001).
- Every wire model: `frozen=True`, `extra="forbid"`, `schema_version="1"`, `tuple` collections,
  `StrEnum` status — never a boolean status, never in-place mutation (ADR-0002).
- A judge may `hard_gate=True` only under full calibration **and** the ratified floor
  (TNR ≥ 0.95, TPR ≥ 0.85, age ≤ 90 days). Two-tier demotion (D-1 as amended 2026-07-04):
  bad evidence (expired, sub-floor TNR/TPR) ⇒ `UNAVAILABLE`; absent evidence (undated/stale)
  ⇒ advisory only, never gates (ADR-0007).
- Operational failures (error/timeout/cancel) are never folded into task failures;
  `compare_runs` is provenance-gated (ADR-0008).
- Redaction routes every persisted format through `apply_redaction` exactly once;
  `yaml.safe_load` only; secrets only via runtime credential hooks; never `trust_remote_code`.
- Cache correctness = checksum-on-read, never `Path.replace()` atomicity; parallel runs use a
  per-worker `AGENTIC_EVALKIT_CACHE_DIR` (D-2, 2026-07-04).
- Tests stay hermetic (`-m "not live"` default); never commit generated reports or `_bmad*` output.

## Deeper context

- **Tracked source of truth:** `docs/adr/` + `docs/specs/` + [CONTRIBUTING.md](CONTRIBUTING.md).
- **Generated agent context (gitignored, BMAD-regenerated; present on maintainer machines):**
  `_bmad-output/project-context.md` — the full 50-rule agent rulebook — and
  `docs/codebase/architecture.md`. Prefer them when present; when absent, the ADRs govern.

---

> Everything below moved here from `C:/Users/tandf/source/AGENTS.md` §4 on 2026-07-28 so it
> loads only when working in this repo. **Note: this file is gitignored (`.gitignore:28`), so a
> fresh clone has none of it** — on a fresh clone `CONTRIBUTING.md` says the ADRs under
> `docs/adr/` govern. Cross-repo coupling stays in the workspace-root file.

## Portfolio context

Standalone publishable library + Typer CLI (`agentic-evalkit`, v0.3.0) that grades AI/agent outputs and produces the evidence for the verdict. Tracked skills: `.Codex/skills/release-gate/` and `.Codex/skills/write-adr/`.

### Command surface deltas vs CONTRIBUTING.md

CONTRIBUTING.md has the full offline verification matrix, documentation build, and release-gate command sequences — read it for the actual invocations. Not obvious from there:

- `uv run mypy` takes no path argument; `pyproject.toml` already scopes it to the package.
- `uv run pytest tests/live -m live -v` needs the explicit `-m live` because addopts defaults to `-m 'not live'`.
- `.Codex/skills/release-gate/scripts/run-gates.sh` takes `[--release]` and `[--live]` flags.
- `uv sync --all-groups --locked` is what the release-gates/live-provider/live-swebench workflows install with; `ci.yml`'s own `test` job uses plain `uv sync --all-groups`.

CLI surface: `doctor`, `init`, `validate`, `run`, `compare`, `report`, plus the `datasets` group (curated/search/inspect/preview/pull). `pytest-repeat` is available for flake hunting (`--count=N`).

**Architecture:** every seam is a `@runtime_checkable` `typing.Protocol` defined once in its subpackage's `base.py` — `DatasetProvider`, `BenchmarkAdapter`, `ExecutionTarget`, `Grader`. `EvalRunner` drives dataset → adapter → target → grader → reporter but **constructs nothing**; the CLI builds everything and hands it in, and the runner declares its own local `_CatalogProtocol` rather than importing the catalog. Everything crossing a boundary subclasses `FrozenModel` (`frozen=True, extra="forbid", schema_version`). `CompositeGrader` excludes ABSTAIN/ERROR/UNAVAILABLE from the weighted mean rather than scoring them zero, and any failing `hard_gate=True` component fails the composite; a model judge may hold that gate only with calibration clearing TNR ≥ 0.95 / TPR ≥ 0.85 / age ≤ 90d plus a Wilson lower bound. Redaction is applied **once**, before any reporter runs, so no output format can leak independently.

**Gotchas:**

- `tests/contract/` is executable architecture, and several gates fail on **documentation and workflow text**. `test_live_test_boundary.py` asserts the literal substrings `-m 'not live'` (pyproject), `pytest -m "not live"` (ci.yml) and `pytest tests/live -m live` (live-provider.yml) — rewording any of those three invocations breaks the suite.
- Adding a report format needs two edits by design: register it in `REPORTER_FORMATS` **and** hand-add it to `REDACTION_ROUTED_FORMATS` (both in `src/agentic_evalkit/reporters/__init__.py`, deliberately not derived from each other). Similarly `ALL_EVENT_TYPES` (`events.py`) must equal the `RunEvent` union exactly and every event field must be a wire-safe scalar/enum/timestamp.
- Adding a field to `EvalRunManifest` or `SamplingPolicy` fails CI until a human categorizes it — `test_provenance_drift.py` reflects over the live model fields and `compare._PROVENANCE_CHECKS`.
- A new ADR must be added to `REQUIRED_ADR_PREFIXES`, use the seven-heading template in canonical order, be `Accepted`, and get an mkdocs nav entry. Use the `write-adr` skill; the test docstring says explicitly: do not weaken the test.
- Public docs, examples and live CLI `--help` are scanned case-sensitively for `agentic_v2`, `agentic-v2-eval`, `tools.agents`, `executionkit` — lowercase fails, the prose name "ExecutionKit" is fine; `docs/plans|specs|adr` are exempt.
- Do **not** move a Pydantic field's type import under `if TYPE_CHECKING:` to satisfy ruff TC00x — `runtime-evaluated-base-classes` is pinned in pyproject for exactly this.
- `uv run mypy` covers only the package, so type errors under `tests/` are invisible to CI. Existing per-file-ignores (`S101`/`S105` for tests, `N818` for `errors.py`/`artifacts.py`) are API-stability decisions, not debt.
- `datasets/_cache_io.py` retries **exactly** `PermissionError` and `FileNotFoundError` (Windows `Path.replace()` sharing violations) and deliberately lets every other `OSError` propagate — broadening to `except OSError` would be a bug.
- Do not remove `--cov-config=pyproject.toml` from addopts (pytest-cov #479 / coveragepy #512 combine crash).
- Live jobs that skip everything are treated as failures: `live-swebench.yml` parses the JUnit XML and fails unless both fidelity tests actually ran.
- Only `ci.yml` runs on `pull_request` — it is the sole merge-blocking workflow. Release gates are workflow_dispatch-only; live lanes are cron. Pre-commit is **gitleaks only**, so a locally clean commit can still be red.
- `docs/codebase/*.md` exists locally and reads like tracked guidance, but is gitignored and mkdocs-excluded — do not cite it as policy.
