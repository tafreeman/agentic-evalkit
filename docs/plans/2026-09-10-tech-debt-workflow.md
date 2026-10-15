# EvalKit technical-debt delivery workflow

Status: implementation plan, 2026-09-10; no implementation completion is implied.
This is the EvalKit half of the coordinated ARP/EvalKit audit follow-up.

## Baseline and scope

Authored against clean local source `e6cf57c2`. The last verified remote main
was `98f74b17`, including PR #55's strict documentation gate and the 8192
constant cross-check. Revalidate main before implementation. The numeric
cross-check is completed work; it does not make byte limits and character
limits the same setting.

The standalone dependency boundary, frozen wire models, calibrated judge
authority, failure taxonomy and exactly-once redaction remain unchanged.
ARP will own its deterministic required gate; this project retains its
abstention-aware composite grading. No sibling imports or new integration
package are needed to settle the arithmetic difference.

## Goals and model assignments

H means Opus/Sol class, L means Haiku/Luna class. H owns each goal's contracts,
high-risk implementation and integration review. L executes bounded patches
after the contract is fixed.

| Goal | Tasks | Prerequisite | Done |
| --- | --- | --- | --- |
| G0 | L verifies current main and audit ledger; captures baseline commands/results. H confirms the remaining scope. | None | Exact base SHA and passing baseline, or explicit blockers |
| G5A | H classifies hash/serialization contracts; L extracts byte hashing and identical row serialization; H reviews identity fixtures. | G0 | Identical bytes and fingerprints for all migrated callers |
| G5B | H designs storage primitives and fault tests; L migrates one cache per patch; H validates concurrency and Windows failures. | G5A | Old cache compatibility and complete fault matrix |
| G7 | L consolidates identical result construction, target taxonomy and reporter presentation; H reviews policy-sensitive differences. | G0; disjoint files | Existing grading, failure, redaction and output behavior preserved |

Implementation contracts:

* [Hashing and cache storage](tech-debt/cache-and-hashing-addendum.md)
* [Bounded grader, target and reporter cleanup](tech-debt/helper-cleanup-addendum.md)

ARP's companion workflow is `docs/plans/2026-09-10-tech-debt-workflow.md` in its
repository. Its scoring and execution tracks can run independently of G5.
Do not invent a shared cross-repo utility package for these extractions.

## Task execution contract

Each task receives goal ID, base SHA, prerequisite evidence, allowed files,
approved API, fixtures, acceptance commands and escalation rules. Preserve
unrelated work in isolated branches/worktrees. Return changed files, validation
counts, skips/failures, compatibility evidence and remaining risks.

Reusable goal instruction:

> Complete Gx using its addendum. Reconcile current main and record baseline
> gates. H fixes the contract; delegate only bounded L tasks. Integrate and
> validate the complete goal. Preserve public APIs, on-disk bytes, provenance,
> redaction and grading authority. Return a reviewable patch and evidence;
> do not merge, release or launch paid/provider runs without authorization.

L escalates any change to semantics, public fields, serialization, disk format,
dependency boundaries or assigned files. After two unsuccessful fixes, return
a reproduction and diff to H. Small helper extractions are direct work items;
do not spend a separate design phase documenting trivial edits.

## Validation and decision records

Before and after implementation run the current CONTRIBUTING matrix:
`uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and
`uv run pytest -m "not live" --cov --cov-report=term-missing`.
Use `uv run mkdocs build --strict` for docs and the relevant contract tests
for ADR, dependency, redaction and provenance changes. Preserve the configured
coverage floor. Never count all-skipped integration coverage as success.

The cache addendum refines ADR-0004 without changing its disk format or
checksum-on-read correctness model. If recording the accepted new ownership
decision as a separate ADR, allocate the next available number during G5B
(0026 was next at inspection), use the seven required headings, register its
prefix in the ADR contract test and add MkDocs navigation. Cite actual final
test evidence. These plans are not prematurely marked Accepted ADRs.

Keep shipping validation, CI success, push and merge distinct. No audit finding
is closed merely because its implementation has been documented.
