"""Provenance-drift reflection contract (Story 3.1, R-004 P0).

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 3, Story 3.1) and
the TEA test design (R-004). Green since the P0 branch landed the seams; the
2026-07-04 code review then made the declared<->checked binding falsifiable.

How the guard works, end to end:
  * ``EvalRunManifest.provenance_field_names()`` DECLARES the
    comparability-relevant fields (dotted paths for nested sampling leaves).
  * ``compare._PROVENANCE_CHECKS`` is the table ``_describe_mismatches``
    actually iterates; ``compare.PROVENANCE_FIELDS_CHECKED`` is derived from
    that table's rows -- from the checks themselves, not re-declared -- so
    the equality test below fails when a declared field has no live check
    (or a check compares an undeclared field).
  * The categorization test reflects over the real model fields (manifest
    top-level AND SamplingPolicy leaves), so a NEW field anywhere in the
    comparability surface fails CI until consciously categorized.
  * Behavioural per-field coverage (differ two runs by one field ->
    ``IncompatibleRuns`` names it) lives in ``tests/unit/stats/test_compare.py``.
"""

from __future__ import annotations

# The provenance fields compare_runs enumerates (from compare._PROVENANCE_CHECKS).
# The declaration seam must cover at least these. environment_fingerprint and
# code_fingerprint joined this set under ADR-0015; both are still
# comparability-relevant declarations even though compare_runs's
# allow_cross_environment can waive a *mismatch* on them per-comparison.
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
        "environment_fingerprint",
        "code_fingerprint",
    }
)


def test_manifest_declares_its_provenance_fields() -> None:
    from agentic_evalkit.models.runs import EvalRunManifest

    declared = frozenset(EvalRunManifest.provenance_field_names())
    missing = _EXISTING_PROVENANCE_FIELDS - declared
    assert not missing, f"manifest provenance declaration omits: {missing}"


def test_compare_runs_checks_every_declared_provenance_field() -> None:
    # The drift guard, both directions: every declared provenance field must
    # have a live row in compare's checks table (PROVENANCE_FIELDS_CHECKED is
    # derived from the table _describe_mismatches iterates, so this cannot
    # pass by construction), and every checked field must be declared.
    from agentic_evalkit.models.runs import EvalRunManifest
    from agentic_evalkit.stats import compare

    declared = frozenset(EvalRunManifest.provenance_field_names())
    checked = frozenset(compare.PROVENANCE_FIELDS_CHECKED)
    unchecked = declared - checked
    undeclared = checked - declared
    assert not unchecked, f"compare_runs does not check declared provenance fields: {unchecked}"
    assert not undeclared, f"compare_runs checks fields the manifest never declared: {undeclared}"


def test_cross_environment_waiver_set_is_exactly_the_adr_0015_fields() -> None:
    # ADR-0015 authorizes waiving exactly environment_fingerprint and
    # code_fingerprint -- no more, no fewer. The waivable set is derived from
    # _PROVENANCE_CHECKS' waivable column, so this pins the table itself:
    # marking any other row waivable (or unmarking one of these two) is a
    # contract change that must supersede ADR-0015 before it can pass CI.
    from agentic_evalkit.stats import compare

    assert compare._WAIVABLE_UNDER_CROSS_ENVIRONMENT == frozenset(
        {"environment_fingerprint", "code_fingerprint"}
    )


def test_declared_provenance_names_resolve_to_real_fields() -> None:
    # A typo in a declared dotted path ("sampling.temprature") would otherwise
    # satisfy every set comparison while checking nothing that exists.
    from agentic_evalkit.models.runs import EvalRunManifest, SamplingPolicy

    for name in EvalRunManifest.provenance_field_names():
        head, _, leaf = name.partition(".")
        assert head in EvalRunManifest.model_fields, (
            f"declared provenance field {name!r} does not start with a real manifest field"
        )
        if leaf:
            assert head == "sampling", f"unexpected nested provenance container in {name!r}"
            assert leaf in SamplingPolicy.model_fields, (
                f"declared provenance leaf {name!r} does not exist on SamplingPolicy"
            )


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
    provenance_top_level = {name.split(".")[0] for name in EvalRunManifest.provenance_field_names()}
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
        "baseline_compatibility_rules",
        # ADR-0013: an informative dataset-provenance label, deliberately not a
        # comparability key -- two runs of the same dataset never differ in
        # meaning because one carried a SUSPECT label and the other did not.
        "contamination",
        "schema_version",
    }
    categorized = provenance_top_level | comparability_excluded
    uncategorized = set(EvalRunManifest.model_fields) - categorized
    assert not uncategorized, (
        f"new EvalRunManifest field(s) {sorted(uncategorized)} must be added to "
        "provenance_field_names() (comparability-relevant) or the "
        "comparability_excluded set above (deliberately not compared)"
    )


def test_every_sampling_policy_field_is_categorized_for_comparability() -> None:
    # The manifest-level reflection above collapses "sampling.*" to its
    # container, so a NEW SamplingPolicy leaf would otherwise be invisible to
    # the drift guard. Reflect the nested model directly: every leaf is either
    # declared as a dotted provenance path or deliberately excluded.
    from agentic_evalkit.models.runs import EvalRunManifest, SamplingPolicy

    declared_leaves = {
        name.partition(".")[2]
        for name in EvalRunManifest.provenance_field_names()
        if name.startswith("sampling.")
    }
    # SamplingPolicy.attempts is validator-mirrored to the manifest's own
    # top-level ``attempts`` field, which IS declared and checked -- comparing
    # it twice would only duplicate the mismatch line.
    sampling_excluded = {"attempts", "schema_version"}
    categorized = declared_leaves | sampling_excluded
    uncategorized = set(SamplingPolicy.model_fields) - categorized
    assert not uncategorized, (
        f"new SamplingPolicy field(s) {sorted(uncategorized)} must be declared as "
        "'sampling.<leaf>' in provenance_field_names() (and checked by compare_runs) "
        "or added to the sampling_excluded set above with a rationale"
    )
