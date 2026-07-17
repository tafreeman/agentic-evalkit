"""Provenance-drift reflection contract (Story 3.1, R-004 P0).

"Provenance" means the recorded conditions a run happened under -- which
adapter, which grader, which sampling settings, and so on -- the facts you'd
need before trusting that two runs are actually comparable. "Drift" means a
policy claim about what gets checked quietly stops matching what the code
actually checks -- for example, a field everyone assumes is compared, but
which a refactor silently stopped comparing. This module guards against
that by using Python reflection (inspecting the real model fields and the
real checks table at runtime, instead of comparing two hand-copied lists
that could each be edited independently) so a mismatch between "what's
declared as provenance" and "what's actually checked" cannot pass by
accident.

Source: ``_bmad-output/planning-artifacts/epics.md`` (Epic 3, Story 3.1) and
the TEA test design (R-004, an internal test-design/risk reference). This
test has passed ever since the P0 (priority-0, i.e. highest-priority)
branch landed the "seams" it depends on -- the integration points, such as
a shared table both the production code and this test read from, that make
a check like this possible. A 2026-07-04 code review then changed the test
so that a real gap between "declared" and "checked" fields would actually
make it fail (in testing terms, made it "falsifiable"), instead of it being
structurally unable to ever catch anything.

How the guard works, end to end:
  * ``EvalRunManifest.provenance_field_names()`` DECLARES which fields are
    comparability-relevant -- i.e., which settings must match between two
    runs before it is fair to compare their scores (dotted paths are used
    for nested fields inside ``sampling``, e.g. ``"sampling.seed"``).
  * ``compare._PROVENANCE_CHECKS`` is the actual table the comparison code
    (``_describe_mismatches``) loops over at runtime to decide what to
    compare. ``compare.PROVENANCE_FIELDS_CHECKED`` is computed FROM that
    same table -- never re-typed by hand -- so the equality test below
    really can fail: it does, whenever a declared field has no matching
    live check, or a check compares a field nobody declared.
  * The categorization test inspects the real model fields at runtime
    (both the manifest's own top-level fields and the nested
    SamplingPolicy fields), so a brand new field added anywhere in this
    surface fails CI until a person consciously categorizes it.
  * Tests that check the actual run-time behavior (e.g., that two runs
    differing in one field cause ``compare_runs`` to raise
    ``IncompatibleRuns`` naming that field) live separately, in
    ``tests/unit/stats/test_compare.py``.
"""

from __future__ import annotations

# The provenance fields compare_runs actually compares, read off
# compare._PROVENANCE_CHECKS (the live table -- see the module docstring
# above). EvalRunManifest.provenance_field_names() -- the "declaration"
# side of the contract -- must cover at least this set.
# environment_fingerprint and code_fingerprint joined this set under
# ADR-0015; they still count as comparability-relevant declarations even
# though compare_runs's allow_cross_environment flag can choose to waive
# (ignore) a *mismatch* on just these two fields, on a per-comparison
# basis.
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
    # This is the drift guard, checked in both directions at once: every
    # field declared as provenance must have a matching live row in
    # compare's checks table -- and because PROVENANCE_FIELDS_CHECKED is
    # computed from that same table rather than typed out separately by
    # hand, this side of the check reflects the real, running code rather
    # than two lists that merely happen to agree. And every field that IS
    # actually checked must also have been declared, so nothing gets
    # compared "in secret" without being documented as provenance.
    from agentic_evalkit.models.runs import EvalRunManifest
    from agentic_evalkit.stats import compare

    declared = frozenset(EvalRunManifest.provenance_field_names())
    checked = frozenset(compare.PROVENANCE_FIELDS_CHECKED)
    unchecked = declared - checked
    undeclared = checked - declared
    assert not unchecked, f"compare_runs does not check declared provenance fields: {unchecked}"
    assert not undeclared, f"compare_runs checks fields the manifest never declared: {undeclared}"


def test_cross_environment_waiver_set_is_exactly_the_adr_0015_fields() -> None:
    # ADR-0015 permits waiving (ignoring) a mismatch on exactly two
    # fields -- environment_fingerprint and code_fingerprint -- no more, no
    # fewer. The set of waivable fields is read directly off
    # _PROVENANCE_CHECKS' own "waivable" column, so this test pins that
    # table down: marking any other field waivable (or un-marking one of
    # these two) is a real contract change, and by ADR-0015's own rule, a
    # change like that must be recorded in a new ADR that formally
    # supersedes (replaces) ADR-0015 before it is allowed to pass CI.
    from agentic_evalkit.stats import compare

    assert (
        frozenset({"environment_fingerprint", "code_fingerprint"})
        == compare._WAIVABLE_UNDER_CROSS_ENVIRONMENT
    )


def test_declared_provenance_names_resolve_to_real_fields() -> None:
    # Guards against a silent typo: a misspelled dotted path like
    # "sampling.temprature" would still pass every set-equality check
    # above (it is just a string, and both the "declared" and "checked"
    # sides would agree on the same typo) while actually checking a field
    # that does not exist on the model at all. This test catches that by
    # confirming each declared name resolves to a real field.
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
    # This is the real drift guard behind R-004 (an internal risk-tracking
    # ID): every field on EvalRunManifest must be explicitly categorized as
    # either comparability-relevant provenance (something compare_runs
    # checks) or deliberately excluded from comparison. A brand new
    # manifest field therefore fails this test the moment it is added,
    # until whoever added it consciously decides which category it
    # belongs in -- a field can never silently slip through uncompared.
    # Tests that check the actual behavior (e.g., that differing in one
    # field causes compare_runs to raise IncompatibleRuns) live
    # separately, in tests/unit/stats/test_compare.py.
    from agentic_evalkit.models.runs import EvalRunManifest

    # Nested fields under `sampling` are declared as dotted paths (like
    # "sampling.seed"); this collapses each one down to just "sampling" so
    # it can be compared against the manifest's own top-level field names.
    provenance_top_level = {name.split(".")[0] for name in EvalRunManifest.provenance_field_names()}
    # Top-level manifest fields that deliberately do NOT affect whether
    # comparing two runs is meaningful -- explicitly excluded, not just
    # forgotten. For example, comparability is based on the *resolved*
    # dataset identity (EvalRunResult.resolved_dataset), not on the
    # dataset_ref the caller originally requested, so dataset_ref itself is
    # excluded here.
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
        # ADR-0013: records what's known about whether this dataset's rows
        # may have leaked into a model's training data -- an informative
        # label, not a comparability key. Two runs of the exact same
        # dataset never differ in meaning just because one happened to
        # carry a SUSPECT contamination label and the other did not.
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
    # The manifest-level check above collapses every "sampling.*" dotted
    # path down to just its container name ("sampling"), which means a
    # brand new field nested inside SamplingPolicy would be invisible to
    # that check -- "sampling" itself is already categorized, so the check
    # would never flag it. This test closes that gap by inspecting
    # SamplingPolicy's own fields directly: each one must either be
    # declared as a dotted provenance path (like "sampling.seed") or be
    # deliberately excluded below.
    from agentic_evalkit.models.runs import EvalRunManifest, SamplingPolicy

    declared_leaves = {
        name.partition(".")[2]
        for name in EvalRunManifest.provenance_field_names()
        if name.startswith("sampling.")
    }
    # SamplingPolicy.attempts is kept in sync with the manifest's own
    # top-level ``attempts`` field by a validator (a Pydantic check that
    # runs at construction time) -- and that top-level field IS already
    # declared and checked. Comparing SamplingPolicy.attempts too would
    # just report the identical mismatch a second time, so it is excluded
    # here instead of being declared as a redundant second provenance
    # field.
    sampling_excluded = {"attempts", "schema_version"}
    categorized = declared_leaves | sampling_excluded
    uncategorized = set(SamplingPolicy.model_fields) - categorized
    assert not uncategorized, (
        f"new SamplingPolicy field(s) {sorted(uncategorized)} must be declared as "
        "'sampling.<leaf>' in provenance_field_names() (and checked by compare_runs) "
        "or added to the sampling_excluded set above with a rationale"
    )
