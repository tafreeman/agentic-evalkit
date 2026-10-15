# Architecture: shared result and report helpers

Status: proposed architecture. The changes below are not yet implemented.
Delivery steps: [G7](../2026-09-10-tech-debt-workflow.md).

## Problem

Graders repeat result construction, execution targets repeat failure labels,
and reporters repeat display and timestamp logic. These copies can drift when
one implementation changes.

## Decision

Share identical construction and display behavior through internal functions.
Keep grading decisions in each grader. Do not introduce new base classes or
change public result fields.

## Grader results

Add `graders/_results.py` with `make_grade_result(...) -> GradeResult`.
Use keyword-only arguments matching the current model:

- Required: `sample_id`, `grader`, `status`, `hard_gate`, `created_at`.
- Existing optional fields: `grader_type`, `score`, `evidence`, `artifact_refs`,
  `rubric_id`, `oracle_provenance`, `judge_calibration_ref`.

Keep current field types and defaults. Copy supplied mappings. This helper
must not decide whether evidence is trustworthy, grant permission to block a
run, turn an execution error into a failed grade, or replace a missing score
with zero.

Add `incomplete_execution_result(sample, execution, *, grader, grader_type,
created_at) -> GradeResult | None` for graders that already use this exact rule:
incomplete execution or missing output returns UNAVAILABLE, a null score,
`hard_gate=False`, and the existing reason.

Move only matching guards to this function. Keep other rules local, including
ABSTAIN when a reference answer is absent.

## Target failures

Add internal `targets/_results.py`:

```python
def failure_code(
    status: ExecutionStatus,
) -> Literal["target_timeout", "target_failure"]: ...
```

TIMEOUT returns `target_timeout`; other failure states return `target_failure`.
Call it only while constructing failure results. Keep status classification,
retry behavior and error details unchanged. Hashing changes belong to the
separate cache and hashing work.

## Report presentation

Add internal `reporters/_presentation.py`:

```python
def outcome_label(sample: SampleResult) -> str: ...
def resolve_generated_at(value: str | None) -> str: ...
```

Use the current outcome labels and timestamp format. Preserve an explicitly
supplied timestamp; generate a default only for None. Do not read the clock at
module import.

These functions do not remove secrets. Keep the existing redaction step before
reporting, applied exactly once for every format.

## Migration and verification

Migrate one helper family per change. Compare complete grader results,
including evidence and calibration fields, for successful execution, incomplete
execution, missing output and absent reference answers.

Test timeout and other target failures with existing fakes. Compare JSON,
JSONL, Markdown and HTML outputs using fixed timestamps and redacted secrets.
Test observable output, not merely whether a helper was called.

Do not combine the judge's character limit with the runner's byte threshold.
Their numeric equality check already exists, but their units and purposes
remain different.
