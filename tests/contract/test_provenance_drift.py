"""ATDD red-phase scaffolds for Story 3.1 -- provenance-drift reflection
contract (R-004 P0).

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 3, Story 3.1) and
the TEA test design (R-004).

``agentic_evalkit.stats.compare._describe_mismatches`` HAND-enumerates the
manifest provenance fields it checks (adapter, grader, target policy/
fingerprint, sampling temperature/seed, attempts, + resolved-dataset
identity). Nothing binds that enumeration to the manifest's actual fields, so
a NEW ``EvalRunManifest`` provenance/sampling field can be added and silently
escape ``compare_runs`` -- two runs that differ only in that field would
compare as "compatible" and produce a statistically invalid delta.

This contract requires two reflection seams so drift fails the build. Both
are referenced INSIDE the skipped bodies so collection never breaks before
they exist.

Skip-marked (TDD red phase). Implementation notes for the dev:
  * Declare the comparability-relevant provenance fields on the manifest,
    e.g. ``EvalRunManifest.provenance_field_names() -> frozenset[str]``.
  * Expose the set ``compare_runs`` actually checks, e.g.
    ``compare.PROVENANCE_FIELDS_CHECKED``, and drive ``_describe_mismatches``
    from it so the two can never diverge.
  * A behavioural follow-up (differ two runs by one field -> ``IncompatibleRuns``
    names it) belongs in ``tests/unit/stats``; reuse ``_stats_fixtures.py``.
"""

from __future__ import annotations

# The provenance fields compare_runs already enumerates today (from
# compare._describe_mismatches). The declaration seam must cover at least these.
_EXISTING_PROVENANCE_FIELDS = frozenset(
    {
        "adapter",
        "grader",
        "target_name",
        "target_fingerprint_policy",
        "target_fingerprint",
        "sampling.temperature",
        "sampling.seed",
        "attempts",
    }
)


def test_manifest_declares_its_provenance_fields() -> None:
    from agentic_evalkit.models.runs import EvalRunManifest

    declared = frozenset(EvalRunManifest.provenance_field_names())  # seam to add
    missing = _EXISTING_PROVENANCE_FIELDS - declared
    assert not missing, f"manifest provenance declaration omits: {missing}"


def test_compare_runs_checks_every_declared_provenance_field() -> None:
    # The drift guard: every field the manifest declares as provenance must be
    # covered by compare_runs, so adding a provenance field without extending
    # the mismatch enumeration fails here (R-004).
    from agentic_evalkit.models.runs import EvalRunManifest
    from agentic_evalkit.stats import compare

    declared = frozenset(EvalRunManifest.provenance_field_names())  # seam to add
    checked = frozenset(compare.PROVENANCE_FIELDS_CHECKED)  # seam to add
    missing = declared - checked
    assert not missing, f"compare_runs does not check declared provenance fields: {missing}"


def test_every_manifest_field_is_categorized_for_comparability() -> None:
    # R-004 (real drift guard): every EvalRunManifest field must be explicitly
    # categorized as either comparability-relevant provenance (checked by
    # compare_runs) or deliberately excluded. A NEW manifest field then fails
    # this test until the author consciously categorizes it, so it can never
    # silently escape comparability. Behavioural per-field coverage (differing
    # one field -> IncompatibleRuns) lives in tests/unit/stats/test_compare.py.
    from agentic_evalkit.models.runs import EvalRunManifest

    # Nested sampling leaves are declared with dotted paths; collapse to the
    # top-level container name to compare against manifest field names.
    provenance_top_level = {
        name.split(".")[0] for name in EvalRunManifest.provenance_field_names()
    }
    # Top-level manifest fields that intentionally do NOT affect whether a
    # run-to-run delta is meaningful. Comparability uses the *resolved* dataset
    # identity (on EvalRunResult.resolved_dataset), not the requested
    # dataset_ref, so dataset_ref is excluded here.
    comparability_excluded = {
        "run_name",
        "dataset_ref",
        "revision_policy",
        "selection",
        "timeout_seconds",
        "concurrency",
        "artifact_policy",
        "redaction_policy",
        "environment_fingerprint",
        "code_fingerprint",
        "baseline_compatibility_rules",
        "schema_version",
    }
    categorized = provenance_top_level | comparability_excluded
    uncategorized = set(EvalRunManifest.model_fields) - categorized
    assert not uncategorized, (
        f"new EvalRunManifest field(s) {sorted(uncategorized)} must be added to "
        "provenance_field_names() (comparability-relevant) or the "
        "comparability_excluded set above (deliberately not compared)"
    )
