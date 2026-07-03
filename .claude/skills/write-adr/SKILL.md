---
name: write-adr
description: Draft a new agentic-evalkit ADR that passes tests/contract/test_adrs.py on the first run — correct filename, seven-heading template in canonical order, Accepted status, and no phrases contradicting standing decisions. Use when asked to write, add, or record an ADR or architecture decision.
---

# Write an ADR

`tests/contract/test_adrs.py` structurally enforces every ADR in this repo.
Follow these steps and the new ADR passes the contract test on the first run.

## Steps

1. **Number and name.** Take the next free four-digit prefix under
   `docs/adr/` and name the file `NNNN-kebab-case-title.md`. Exactly one
   file may match `NNNN-*.md` — never reuse or pad an existing prefix.

2. **Copy the template** from `references/adr-template.md` and fill every
   section. The seven headings, in this exact order, are mandatory:
   `## Status`, `## Context`, `## Decision`, `## Alternatives`,
   `## Consequences`, `## Validation`, `## Supersession`.

3. **Status wording matters.** `## Status` must be followed by the single
   word `Accepted` on its own line — the contract test extracts the first
   non-whitespace token after the heading and compares it exactly. This
   project records ADRs at acceptance time (no Draft/Proposed states).

4. **Do not contradict standing decisions.** The test scans all ADRs
   (normalized, lowercased) and fails on any of these phrases: "may import
   agentic_v2", "may import executionkit", "may import tools.agents",
   "huggingface is an optional extra", "huggingface support requires an
   extra", "trust_remote_code=true". Discuss alternatives without asserting
   them in those words.

5. **Cite validation evidence.** The `## Validation` section names the
   tests/commands that prove the decision holds (see any existing ADR for
   the style — each cites its governing contract or unit tests).

6. **Register the ADR.** Add the new prefix to `REQUIRED_ADR_PREFIXES` in
   `tests/contract/test_adrs.py` (the tuple is an explicit allowlist) and
   add a nav entry under the ADRs section of `mkdocs.yml` so the strict
   docs build publishes it.

7. **Verify:**

   ```bash
   uv run pytest tests/contract/test_adrs.py -v
   uv run mkdocs build --strict
   ```

## References

- `references/adr-template.md` — the seven-section skeleton to copy.
