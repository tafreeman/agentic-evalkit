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
test itself is also marked `live`.

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

## Commit and pull request workflow

1. Create a branch from `main` (`feature/`, `fix/`, `chore/`, or `docs/`
   prefix).
2. Make focused, test-first commits.
3. Run the full offline verification matrix above.
4. Open a pull request describing the change and the tests that cover it.

## Reporting security issues

Do not open a public issue for security vulnerabilities — see
[SECURITY.md](SECURITY.md).
