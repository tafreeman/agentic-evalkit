---
name: release-gate
description: Run agentic-evalkit's offline verification matrix and release gates in order, reporting pass/fail per gate. Use before opening a PR, before tagging a release, or when asked to "run the gates", "verify the release", or "check release readiness".
---

# Release Gate Runner

Single source of truth for the verification every change and release must
pass. CONTRIBUTING.md describes these gates in prose and
`.github/workflows/release-gates.yml` runs the release subset in CI; this
skill executes them locally, in order, with one command per gate.

## Steps

1. Sync the environment once — and never run two gate sequences
   concurrently; parallel `uv` invocations race on `.venv`:

   ```bash
   uv sync --all-groups
   ```

2. Run the offline verification matrix, in order. Stop and report the first
   failure with its output; do not continue past a red gate:

   ```bash
   uv run pytest --cov --cov-report=term-missing
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy
   ```

   (`pytest` deselects `live`-marked tests by default via `addopts`; the
   coverage floor is 80% branch-aware and fails the run by itself.)

3. For a release, additionally run the offline release gates:

   ```bash
   uv run pytest tests/contract/test_dependency_boundary.py tests/contract/test_adrs.py tests/contract/test_public_docs.py -v
   uv run mkdocs build --strict
   uv run pytest tests/integration/test_clean_wheel.py -v   # slow: real wheel + venv
   ```

4. The live Hugging Face gate requires network access and is never part of
   the default matrix. Run it explicitly, or dispatch
   `.github/workflows/live-provider.yml` (the only CI workflow allowed to
   touch the real network path):

   ```bash
   uv run pytest tests/live/test_huggingface_live.py -m live -v
   ```

5. Report results as a table — gate, command, PASS/FAIL — followed by the
   failing output for any FAIL. A classified transient outage on the live
   gate is a known issue to document, never something to retry into a false
   pass.

To run everything unattended in one pass instead of step by step:

```bash
bash .claude/skills/release-gate/scripts/run-gates.sh            # offline matrix
bash .claude/skills/release-gate/scripts/run-gates.sh --release  # + release gates
bash .claude/skills/release-gate/scripts/run-gates.sh --release --live  # + network gate
```

## References

- `references/gate-commands.md` — every gate with what it proves and its
  typical failure modes.
