# Contributing to agentic-evalkit

Thank you for your interest in contributing. This document describes how to
set up a local development environment and the verification matrix every
change must pass before it can be merged.

## Local development setup

`agentic-evalkit` uses [`uv`](https://docs.astral.sh/uv/) for dependency
management and Python 3.11+ as the minimum supported version.

```bash
# Install all dependency groups (runtime, dev, and capability extras)
uv sync --all-groups

# Run the CLI from the local checkout
uv run agentic-evalkit --help
```

## Offline verification matrix

Run the full matrix locally before opening a pull request. All commands are
run from the repository root:

```bash
uv sync --all-groups
uv run pytest -m "not live" --cov --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Tests marked `live` require network access to Hugging Face and are excluded
from the default local and CI runs; they are run separately (see
`tests/live/`). Tests marked `integration` exercise component boundaries
(providers, targets, the pipeline) without external network calls unless the
test lives under `tests/live/`. The contract test
`tests/contract/test_live_test_boundary.py` enforces that directory and
marker boundary. Default CLI integration tests inject canned providers and
fail immediately if they accidentally construct the real provider catalog.

## Coverage

The project enforces an 80% branch-aware coverage floor
(`tool.coverage.report.fail_under = 80`). New code should include unit,
integration, or contract tests as appropriate — see
`docs/plans/2026-07-02-agentic-evalkit-initial-release.md` for the
task-by-task test-first workflow this project follows.

## Code style

- Formatting and linting are enforced by Ruff (`uv run ruff check .` and
  `uv run ruff format --check .`).
- Type checking is enforced by strict mypy (`uv run mypy`).
- Public models are immutable Pydantic v2 models (`frozen=True`,
  `extra="forbid"`) — see ADR-0002.

## Dependency boundary

This package must never import ARP, `agentic-tools`, or ExecutionKit modules
— see ADR-0001. Do not add such imports in a pull request; integrations with
those systems go through the public `ExecutionTarget` protocol instead.
`tests/contract/test_dependency_boundary.py` enforces this with a static AST
scan of `src/agentic_evalkit`, and `tests/contract/test_public_docs.py`
separately scans every user-facing document, example, and CLI `--help`
snapshot for the same internal codenames (`agentic_v2`, `agentic-v2-eval`,
`tools.agents`, `executionkit`) so they cannot leak into README, guides, or
examples even though this package never imports the modules they name.

### Coexistence with legacy evaluation code

`agentic-evalkit` separates datasets, grading, and reporting from the system
under test through callable/subprocess/HTTP targets, and objective checks
gate before judges. Legacy evaluation code may remain in host repositories
that adopt this package — this package neither imports nor migrates it;
integration happens only through the public `ExecutionTarget` protocol.

## Documentation

Documentation lives under `docs/` and builds with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) in strict
mode:

```bash
uv run mkdocs build --strict
uv run mkdocs serve   # live-reloading local preview
```

`mkdocs.yml` excludes internal process records (plan review/modification
records, `docs/plans/README.md`, and `docs/release/`) from the published
site via `exclude_docs`; everything else under `docs/` — the design, the
implementation plan, every ADR (`docs/adr/`), and the guides — is part of
the published site and must build without strict-mode warnings.

## Release gates

Beyond the offline verification matrix above, a release additionally
requires (see `docs/plans/2026-07-02-agentic-evalkit-initial-release.md`,
Task 15):

```bash
uv run pytest tests/contract/test_dependency_boundary.py tests/contract/test_adrs.py tests/contract/test_public_docs.py -v
uv run mkdocs build --strict
uv run pytest tests/integration/test_clean_wheel.py -v   # slow: builds a real wheel + venv
uv run pytest tests/live -m live -v   # requires network access
```

The clean-wheel test builds the wheel, installs *only* the wheel into a
temporary virtual environment outside the repository, and confirms the CLI
and Python import work with no host repository on `sys.path`. The live
Hugging Face suite is also run on a weekly schedule and on demand via
`.github/workflows/live-provider.yml`; a classified transient outage there
is a known issue to document, not something to silently retry into a false
pass.

The offline release gates can also be run on demand in CI via the manually
triggered `.github/workflows/release-gates.yml` workflow, or locally in one
pass with `.claude/skills/release-gate/scripts/run-gates.sh` (the live
Hugging Face gate stays in `live-provider.yml`, the only workflow that
exercises the real network path).

## Commit and pull request workflow

1. Create a branch from `main` (`feature/`, `fix/`, `chore/`, or `docs/`
   prefix).
2. Make focused, test-first commits.
3. Run the full offline verification matrix above.
4. Open a pull request describing the change and the tests that cover it.

## AI agent context

The tracked source of truth for AI coding agents (and humans who want the condensed rulebook)
is the ADRs under `docs/adr/` plus the design under `docs/specs/`. Maintainer machines may also
carry local, gitignored quick-reference files that are not part of the shipped repo: a root
`CLAUDE.md` and/or a richer generated `_bmad-output/project-context.md` (BMAD-regenerated).
When neither is present — e.g. a fresh clone — the ADRs govern.

## Reporting security issues

Do not open a public issue for security vulnerabilities — see
[SECURITY.md](SECURITY.md).
