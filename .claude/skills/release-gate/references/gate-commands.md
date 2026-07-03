# Gate Commands Reference

Every verification gate, what it proves, and how it typically fails. Commands
run from the repository root. `pytest` deselects `live`-marked tests by
default (`addopts` in `pyproject.toml`), so only the explicit `-m live`
invocation touches the network.

## Offline verification matrix (every change)

| Gate | Command | Proves | Typical failure |
|------|---------|--------|-----------------|
| sync | `uv sync --all-groups` | Lockfile and environment agree | Drifted `uv.lock`, missing interpreter |
| tests + coverage | `uv run pytest --cov --cov-report=term-missing` | Behavior + 80% branch coverage floor | Failing test, or coverage below `fail_under` |
| lint | `uv run ruff check .` | Lint rules (E, F, I, B, UP, ASYNC, RUF) | Unused import, un-sorted imports |
| format | `uv run ruff format --check .` | Canonical formatting | Any unformatted file |
| types | `uv run mypy` | Strict typing over `agentic_evalkit` | Missing annotation, bad Optional handling |

## Release gates (before tagging)

| Gate | Command | Proves | Typical failure |
|------|---------|--------|-----------------|
| dependency boundary | `uv run pytest tests/contract/test_dependency_boundary.py -v` | No ARP / agentic-tools / ExecutionKit import anywhere in `src/agentic_evalkit` (ADR-0001, static AST scan) | A new module imports a host-runtime symbol |
| ADR shape | `uv run pytest tests/contract/test_adrs.py -v` | All ADRs exist, are `Accepted`, carry the seven headings in canonical order, and contradict no standing decision | New ADR missing a heading or not registered in `REQUIRED_ADR_PREFIXES` |
| public docs | `uv run pytest tests/contract/test_public_docs.py -v` | README, guides, examples, and every CLI `--help` are free of internal codenames | A doc edit leaks an internal system name |
| docs build | `uv run mkdocs build --strict` | Published site builds with zero warnings | Broken nav entry or dead link |
| clean wheel | `uv run pytest tests/integration/test_clean_wheel.py -v` | The wheel installs and runs in a venv outside the repo with no forbidden modules | Packaging config drift; slow (~minutes) is normal |

## Live gate (network; release evidence)

| Gate | Command | Proves | Typical failure |
|------|---------|--------|-----------------|
| live Hugging Face | `uv run pytest tests/live/test_huggingface_live.py -m live -v` | Real Dataset Viewer path works for both curated presets | Provider outage — document as a known issue backed by the latest green `live-provider.yml` run; never retry into a false pass |

CI mapping: the offline matrix runs on every push/PR (`ci.yml`), the release
gates on demand (`release-gates.yml`), and the live gate weekly + on demand
(`live-provider.yml` — the only workflow that exercises the real network).
