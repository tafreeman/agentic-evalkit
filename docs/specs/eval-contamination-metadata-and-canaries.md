# Spec: Dataset contamination metadata + a reusable canary-leak helper

**Slug:** `contamination-metadata-and-canaries`
**Status:** Implemented (ADR-0013; this spec's PR)
**Repo:** `agentic-evalkit`
**Criteria:** C9 (contamination/memorization resistance); touches C8 (statistical rigor — a
contaminated score is not a valid sample) and C1 (spec validity — an unlabeled public preset
silently invites a capability claim two experts would not both sign off on)
**Grounded against:** `main` @ `0c967a1` (2026-07-09)

> **Refresh, 2026-07-10 (implemented against post-`8fcb404` main).** Three facts changed after
> this spec was grounded, and the implementation deliberately deviates from the original text in
> three places:
>
> 1. **`GroundedCitationGrader` now exists** (ADR-0012, merged 2026-07-09) — the Problem section's
>    "no such class exists" correction below is superseded. The helper is therefore built as the
>    *shared* implementation: `graders/grounding.py` now delegates its canary check to
>    `find_canary_leaks` and imports the shared `normalize_for_containment`, so the package
>    carries exactly one tripwire semantics.
> 2. **Matching is normalization-insensitive, not case-sensitive** — superseding acceptance
>    criterion 6's case-sensitivity assertion. The adversarial review (2026-07-09) found that a
>    case-sensitive helper alongside the grader's normalized check would put two divergent
>    tripwire semantics in one package, silently weakening leak detection if the grader ever
>    adopted the helper. Tests now assert case- and whitespace-mangled echoes ARE detected, plus
>    a cross-consistency test proving the helper and the grader agree on a case-mangled leak.
> 3. **ADR numbering resolved:** 0012 = grounded-citation probe (merged), so this spec's ADR is
>    **0013** as provisionally claimed; the sibling swebench spec renumbers to 0014. The docs
>    baseline in criterion 11 (eleven ADRs / ends at 0011) was updated in flight to
>    twelve → thirteen.
> 4. **Report propagation added (supersedes the "No CLI changes" non-goal).** Codex review
>    (P2) flagged that `_manifest_document_for_preset` and `catalog.resolve` dropped the label,
>    so preset-run reports carried `resolved_dataset.contamination == null` — the score never
>    saw the SUSPECT prompt. Fixed minimally: an additive `contamination` field on
>    `EvalRunManifest`, populated from the preset at init and stamped onto the report's
>    `resolved_dataset` by the `run` CLI (a provider-resolved value wins; the manifest only
>    fills a gap), mirroring the existing provenance-fingerprint flow. Threading into
>    `EvalSample`/grading via `BenchmarkAdapter.prepare` remains the deferred, ADR-superseding
>    change.

## Problem

`agentic-evalkit` ships exactly two built-in presets today (`src/agentic_evalkit/datasets/presets.py`,
`BUILTIN_PRESETS`): `gsm8k` (`openai/gsm8k` on the Hugging Face Hub) and `swe-bench-verified`
(`princeton-nlp/SWE-bench_Verified`). Both are widely mirrored, long-public datasets — exactly the
shape of data C9 warns is at highest risk of appearing in a model's pretraining corpus. A repo-wide
search confirms the framework currently carries **zero** machine-readable signal about this: no
field on `ResolvedDataset` (`src/agentic_evalkit/models/datasets.py`) or `DatasetPreset`
(`datasets/presets.py`) records freshness, public-release timing, or held-out status, and a
case-insensitive grep for `contaminat|memoriz|canary` across `src/` and `docs/` returns nothing
outside this spec's own sibling (`docs/specs/eval-swebench-harness-executor.md`, drafted
concurrently in this batch, which does not touch this area). A caller who runs `gsm8k` or
`swe-bench-verified` today and reports the resulting score has no structural prompt — no field to
check, no warning in the preset's own metadata — telling them that score cannot back a capability
claim without first checking for train/test overlap. That is silent, not absent: `ResolvedDataset`
already distinguishes "unavailable" from "empty" for other optional metadata (its own docstring:
"a missing value means 'unavailable', not 'empty'"); contamination risk deserves the same honest,
typed treatment instead of living only in a maintainer's head.

Separately, the framework has a real, already-shipped mechanism for the *other* half of C9 —
avoiding contamination entirely via a private, never-published set — but it is not documented as
such: `LocalDatasetProvider` (`src/agentic_evalkit/datasets/local.py`) reads JSON/JSONL/CSV/YAML
from allow-listed local roots, `requires_network = False` (ADR-0010), and never touches a network
socket on any method. A caller who authors their own eval rows and keeps them local already has a
held-out set in the C9 sense; the framework just never says so anywhere a caller would find it.

**Correction to this improvement's own premise, verified before scoping the design below:** the
sketch that generated this spec asks for a canary-leak helper "extracted from the prototype's
`GroundedCitationGrader`." No such class exists anywhere in this repository — a full-text search of
`src/`, `tests/`, both open branches (`feat/tier2-usability`, `fix/canned-hub-preview-async`), and
every doc under `docs/plans/`, `docs/specs/`, `docs/codebase/` for `GroundedCitation`, `AREP`, or
`faithfulness.*completeness.*sufficiency` returns zero matches. ARP's own `scoring_criteria.py`
(`agentic-runtime-platform/agentic-workflows-v2/agentic_v2/scoring/scoring_criteria.py`) has
generic RAG-style `faithfulness`/`relevance` criterion names but nothing resembling the NIST AREP
three-axis probe. This spec therefore builds the canary-leak helper as new, standalone code rather
than an extraction, and designs its interface so that a future grounded-citation grader (should
pattern P1 be adopted later) can call it unmodified — but it does not depend on that grader existing
first, and this document does not claim it does.

**Second correction, load-bearing for scope:** "held-out" is already a heavily used term in this
codebase, but exclusively in ADR-0007's sense — the ≥30/≥30 human-labeled positive/negative corpus
`CalibrationArtifact` (`graders/judge.py`) needs before a judge may `hard_gate`. That is a
judge-calibration-evidence axis, completely orthogonal to "was this dataset ever plausibly in a
model's pretraining data." Reusing the bare word `held_out` for the new field below risks exactly
the kind of silent conflation ADR-0002 exists to prevent. The proposed change names the field
`held_out` (matching the sketch) but requires its docstring to explicitly disambiguate it from
`CalibrationArtifact`'s held-out corpus, and the acceptance criteria below test that the
disambiguating sentence is actually present, not just requested.

## Research basis

- **C9 itself (contamination/memorization resistance), rated thin-in-the-literature in the source
  audit** (`_audit/agentic-eval-best-practices-2026-07-09.md`) — this spec is a direct instance of
  the pattern the audit flags as high-value-and-underserved: dynamic/held-out/freshly-authored
  tasks as the structural countermeasure, made machine-readable rather than left as an unstated
  convention.
- **NIST AREP grounded-citation probe (adoptable pattern P1)** — the audit's own framing: "LM-judges
  as adversarial verifiers that score claims against a trusted source corpus, returning a structured
  verdict + rationale + machine-readable audit trail." The `canary_leak_evidence()` helper below is
  this repo's minimal instance of "machine-readable audit trail": a fixed-shape evidence dict any
  grader can merge into `GradeResult.evidence` rather than each grader inventing its own leak-report
  shape.
- **C8 interaction, using this repo's own existing discipline as precedent.**
  `stats/aggregate.py`'s `pass_at_k_by_sample` (cited by the sibling `swebench-harness-executor`
  spec) already refuses to fabricate a defined estimate from zero real attempts. A `SUSPECT`-labeled
  public-preset score is the same category of problem one layer up the pipeline: the number is
  computable but not a valid instance of the capability claim it would be used to support without an
  overlap check first. This spec gives that risk a typed field instead of leaving it entirely to
  prose.
- **Anti-pattern this spec avoids:** the audit's headline finding is that task-setup defects swing
  measured performance by up to 100% relative, usually *understating* true capability (Opus 4.5
  42→95% on CORE-Bench after grading fixes). An unflagged contaminated public preset is the mirror
  failure — it can *overstate* capability by rewarding memorization instead of the tested skill —
  and the audit explicitly does not want either direction trusted without a validity check.
- **Repo-native precedent for the design shape, not an external pattern:** `ResolvedDataset.gated`,
  `SearchHit.private`, and `SearchHit.downloads` already establish that a dataset-level contract
  field can carry a trust/provenance signal without becoming an authoritative verdict on its own —
  `gated=True` does not itself refuse access, it informs a caller. `ContaminationMetadata.status`
  below follows the identical shape: informative, not enforcing, so this spec adds zero new runtime
  refusals.

## Proposed change

Reuse before new, per this repo's own boundary: this is two additive optional fields on existing
frozen contracts, two small pure functions in a new grader-adjacent module, two preset annotations,
and documentation — no new dependency, no protocol signature change, no new `Grader` implementation.

1. **`src/agentic_evalkit/models/datasets.py` (extend)** — add two new frozen contracts and two new
   optional fields, following the file's own `FrozenModel` convention:

   ```python
   class ContaminationStatus(StrEnum):
       UNKNOWN = "unknown"
       SUSPECT = "suspect"
       VERIFIED_CLEAN = "verified_clean"
       CONFIRMED_CONTAMINATED = "confirmed_contaminated"


   class ContaminationMetadata(FrozenModel):
       """Best-effort provenance signal for dataset contamination/memorization risk (C9).

       Not the ADR-0007 judge-calibration held-out corpus (`CalibrationArtifact`,
       `graders/judge.py`) — that is human-labeled evidence a judge is trustworthy;
       `held_out` here means this *eval* dataset itself was never published, so it
       cannot appear in any model's pretraining corpus by construction.
       """

       status: ContaminationStatus = ContaminationStatus.UNKNOWN
       authored_after: datetime | None = None
       public_since: datetime | None = None
       canary_ids: tuple[str, ...] = ()
       held_out: bool = False

       @model_validator(mode="after")
       def _validate_held_out_consistency(self) -> "ContaminationMetadata":
           if self.held_out and self.public_since is not None:
               raise ValueError(
                   "held_out=True is inconsistent with a non-null public_since "
                   "(a dataset cannot be both withheld from publication and have "
                   "a known public-release date)"
               )
           return self
   ```

   `status` is a `StrEnum`, not a `contamination_suspect: bool`, for the exact reason ADR-0002's own
   Alternative 3 already rejects boolean status fields: a boolean cannot distinguish "never checked"
   (`UNKNOWN`, the honest default) from "checked and clean" (`VERIFIED_CLEAN`) without the same lossy
   conflation ADR-0002 forbids for `GradeStatus`. `authored_after`/`public_since` are left `None`
   rather than populated with an asserted date on any built-in preset (below) — this repo's own
   numeric/factual discipline (never fabricate a value with no verified source) applies equally to
   dates. Add `contamination: ContaminationMetadata | None = None` to `ResolvedDataset`. Per
   ADR-0002's additive-evolution clause ("new optional fields with safe defaults may be added
   without a version bump"), this needs **no** `schema_version` bump on `ResolvedDataset` — the new
   nested `ContaminationMetadata` model carries its own `schema_version` by inheriting `FrozenModel`.

2. **`src/agentic_evalkit/models/__init__.py` (extend)** — export `ContaminationMetadata` and
   `ContaminationStatus` alongside the existing `ResolvedDataset` export, in the same curated
   `__all__` list. The top-level `agentic_evalkit/__init__.py` re-export surface is deliberately
   minimal ("the smallest set... everything else stays one import away") and does not currently
   export `ResolvedDataset` itself, so the new contracts are **not** added there either — consistent
   with, not an exception to, the existing curation policy.

3. **`src/agentic_evalkit/datasets/presets.py` (extend)** — add the identical
   `contamination: ContaminationMetadata | None = None` field to `DatasetPreset` (same additive
   reasoning; `DatasetPreset` already inherits `FrozenModel`). Annotate both built-in presets:

   ```python
   contamination=ContaminationMetadata(status=ContaminationStatus.SUSPECT)
   ```

   on `_GSM8K_PRESET` and `_SWE_BENCH_VERIFIED_PRESET`. No `canary_ids` on either — canaries only
   make sense for a set this project or its caller authored; neither built-in preset was. No
   `authored_after`/`public_since` — this spec verified neither date live and will not assert one
   (see Problem). `held_out` stays `False` (the class default) for both — they are Hub-hosted,
   public by construction.

4. **`src/agentic_evalkit/graders/contamination.py` (new)** — two pure, stdlib-only functions,
   sibling to `exact.py`/`rubric.py`, following `exact.py`'s stated principle ("this package owns
   grading policy only, not benchmark projection") by taking `canary_ids` as an explicit parameter
   rather than reaching into `EvalSample`/`ResolvedDataset` itself:

   ```python
   def find_canary_leaks(text: str, canary_ids: Sequence[str]) -> tuple[str, ...]:
       """Return the subset of canary_ids that appear verbatim in text.

       Pure, deterministic, case-sensitive substring containment — no LLM, no
       network, hermetic by construction. Never raises; empty/falsy input
       returns an empty tuple.
       """

   def canary_leak_evidence(leaked: tuple[str, ...]) -> dict[str, JsonValue]:
       """Standard GradeResult.evidence-shaped dict any grader can merge in.

       {"canary_check": "leaked" | "clean", "leaked_canary_ids": [...]}
       """
   ```

   Deliberately does **not** decide `GradeStatus` or `hard_gate` — matching `CompositeGrader`'s own
   separation of "component reports evidence" from "composing grader applies policy" — so this spec
   ships the detection primitive without dictating how any specific benchmark should react to a
   leak. Export both from `graders/__init__.py`.

5. **`docs/guides/providers.md` (extend)** — new section (after "Local files", before "Hugging
   Face", so the held-out recommendation lands before the public-provider section it is contrasted
   against) documenting: (a) both built-in presets are `ContaminationStatus.SUSPECT`; (b) a score on
   either must not back a capability claim without a train/test-overlap or decontamination check
   first; (c) `LocalDatasetProvider` plus `ContaminationMetadata(held_out=True, canary_ids=(...))` is
   the supported pattern for a defensible held-out set, cross-referencing `find_canary_leaks`.

**Explicit non-goals (to prevent scope drift):**

- **No automatic threading of `canary_ids` into `EvalSample.metadata` at projection time.**
  Verified structurally: `BenchmarkAdapter.prepare(self, record: SourceRecord) -> EvalSample`
  (`benchmarks/base.py`) never receives a `ResolvedDataset`, and `runner.py`'s only call site
  (`adapter.prepare(record) for record in records`, line 308) confirms this — the adapter has no
  access to dataset-level contamination metadata today. Wiring it through would require a breaking
  signature change to `BenchmarkAdapter.prepare` touching both concrete adapters
  (`benchmarks/gsm8k.py`, `benchmarks/swebench.py`) and the runner call site. That is a separate,
  larger architectural decision and deserves its own ADR, not a drive-by inside a metadata-only
  spec. Until it lands, a grader that wants `find_canary_leaks` must receive `canary_ids` through its
  own constructor injection (matching `ExactMatchGrader`'s injected-extractor pattern) or a
  `GraderSpec.parameters` entry — both already-supported extension points, requiring zero framework
  changes.
- **No `GroundedCitationGrader` and no new `Grader` implementation.** Per the Problem section's
  correction, none exists to extract from; inventing one is out of scope for a C9 metadata spec.
- **No automatic hard-gate policy from a canary leak.** `find_canary_leaks`/`canary_leak_evidence`
  report; whether a leak fails a sample is a grading-policy decision left to the grader that calls
  them, exactly as `CompositeGrader` already separates component evidence from composite policy.
- **No change to `ResolvedDataset.card_metadata`.** Verified in `datasets/huggingface.py` (lines
  349–363): `card_metadata=info.card_metadata` is populated verbatim from the Hub's own API
  response — it is provider-attested data, not framework-asserted data. Writing a
  framework-asserted "contamination-suspect" label into that field would blur exactly the kind of
  provenance distinction ADR-0002 exists to keep clean ("a shared instance could be silently
  modified... corrupting provenance without a stack trace" — the risk here is conflation, not
  mutation, but the same discipline applies). The dedicated `contamination` field keeps the two
  provenance sources structurally separate.
- **No CLI changes.** Surfacing `contamination.status` in `datasets inspect`/`datasets preview`
  output is a reasonable follow-up but is not required for the metadata and helper to be adoptable,
  and keeps this spec's diff scoped to models, presets, one new grader module, and docs.

## Acceptance criteria

1. `ContaminationStatus` and `ContaminationMetadata` are added to `models/datasets.py` and exported
   from `models/__init__.py`; `uv run mypy` (strict) passes with no `Any` leakage on the new code.
2. `ContaminationMetadata(held_out=True, public_since=datetime.now(UTC))` raises `ValidationError`
   at construction (new unit test); `ContaminationMetadata()` with no arguments constructs
   successfully with `status is ContaminationStatus.UNKNOWN` (the honest default).
3. `ResolvedDataset.contamination` and `DatasetPreset.contamination` are both optional
   (`| None = None`); every existing call site that constructs either model without the new field
   (`tests/contract/test_models.py::test_resolved_dataset_round_trips`,
   `datasets/huggingface.py`, `datasets/local.py`, both `_*_PRESET` constructions before this
   change) continues to pass unmodified.
4. `tests/contract/test_models.py` gains a round-trip test constructing a `ResolvedDataset` with a
   fully-populated `contamination=ContaminationMetadata(...)` and asserting
   `ResolvedDataset.model_validate_json(resolved.model_dump_json()) == resolved`, matching the
   file's own documented pattern ("construct -> model_dump_json -> model_validate_json ->
   equality"). A sibling test asserts `ContaminationStatus.SUSPECT` is preserved as a distinct enum
   member through the round-trip (mirrors `test_grade_status_is_not_collapsed_to_boolean`).
5. `tests/unit/datasets/test_catalog.py::test_builtin_presets_full_field_set` is extended to assert
   `BUILTIN_PRESETS["gsm8k"].contamination.status is ContaminationStatus.SUSPECT` and the same for
   `"swe-bench-verified"`. A new, loop-based test —
   `test_all_builtin_presets_declare_a_contamination_status` — iterates
   `BUILTIN_PRESETS.values()` and asserts every preset's `contamination is not None`, so a future
   third preset added without contamination annotation fails this test immediately rather than
   silently shipping unlabeled (mirrors `_build_builtin_presets`'s own eager-fail-at-import-time
   discipline for duplicate names).
6. New `tests/unit/graders/test_contamination.py` (hermetic, no network): `find_canary_leaks`
   returns `()` for empty `canary_ids` and for empty `text`; returns exactly the leaked subset for a
   text containing some-but-not-all configured canaries; confirms matching is case-sensitive (a
   canary differing only in case is not reported as leaked, and this behavior is asserted, not just
   assumed); `canary_leak_evidence(())` returns `{"canary_check": "clean", "leaked_canary_ids": []}`
   and `canary_leak_evidence(("id-1",))` returns `{"canary_check": "leaked", "leaked_canary_ids":
   ["id-1"]}`; every return value round-trips through `json.dumps` without error (proves the
   `JsonValue` contract holds).
7. At least one test demonstrates the stated reuse goal end-to-end: a fake/minimal `Grader.grade()`
   implementation calls `find_canary_leaks` and merges `canary_leak_evidence(...)` into its returned
   `GradeResult.evidence`, and the resulting `GradeResult` still round-trips through
   `model_dump_json()`/`model_validate_json()` (proves the evidence shape is a valid `GradeResult`
   payload, not just a standalone dict).
8. `docs/guides/providers.md`'s new section states, verbatim-checkable in a doc test or by manual
   review citing this criterion: both built-in presets are `SUSPECT`, and their scores "must not
   back a capability claim without an overlap or decontamination check." `uv run mkdocs build
   --strict` still passes.
9. `uv run ruff check .`, `uv run ruff format --check .`, and `uv run mypy` (strict) pass on all
   new/changed files; the repository-wide 80% branch-coverage floor
   (`tool.coverage.report.fail_under = 80`) is met without weakening it.
10. `tests/contract/test_dependency_boundary.py` and `tests/contract/test_public_docs.py` pass
    unmodified — no new third-party dependency is introduced (both new functions are stdlib-only)
    and no internal codename is added to `docs/guides/providers.md`.
11. **ADR numbering, explicitly re-verified against the live contract test, not assumed:**
    `docs/adr/` currently ends at `0011` (`0011-offline-resolution-cache.md`, confirmed via
    `git log`/`ls` at grounding time) and `tests/contract/test_adrs.py::REQUIRED_ADR_PREFIXES` /
    `docs/index.md` both assert exactly eleven committed ADRs today
    (`test_required_prefixes_cover_every_committed_adr_file`,
    `test_landing_page_adr_claims_match_committed_adr_count`). The sibling
    `docs/specs/eval-swebench-harness-executor.md`, drafted concurrently in this same batch, has
    already provisionally claimed **ADR-0012** for its own topic. This spec's ADR stub therefore
    claims **ADR-0013** rather than colliding on 0012 — but because both are proposals, not merged
    ADRs, whichever of the two is actually implemented and merged first keeps its number, and the
    other must renumber before its own merge. Landing either ADR requires, in the same commit:
    appending its prefix to `REQUIRED_ADR_PREFIXES`, and updating both `docs/index.md`'s ADR stat
    tile and its "N architecture decision records, 0001 through NNNN" prose line — or
    `test_required_prefixes_cover_every_committed_adr_file` and
    `test_landing_page_adr_claims_match_committed_adr_count` fail immediately. This criterion is
    satisfied only when both tests pass with the actually-assigned number, whatever it turns out to
    be at merge time.

## ADR stub — ADR-0013: Dataset Contamination Metadata and Canary-Leak Detection

**Status:** Proposed (number provisional — see acceptance criterion 11; renumber at merge if
ADR-0012 is not yet claimed by the sibling `swebench-harness-executor` spec at that time)

- **Context:** `ResolvedDataset` and `DatasetPreset` carry no signal distinguishing a freshly
  authored, held-out evaluation set from a widely mirrored public benchmark, even though C9
  (contamination/memorization resistance) names exactly that distinction as validity-critical, and
  `LocalDatasetProvider` (ADR-0010) already gives callers a mechanism to build the former without
  documenting it as such.
- **Decision:** Add `ContaminationStatus` (`StrEnum`: `unknown`/`suspect`/`verified_clean`/
  `confirmed_contaminated`) and `ContaminationMetadata` (`FrozenModel`: `status`, `authored_after`,
  `public_since`, `canary_ids`, `held_out`, with a construction-time invariant rejecting
  `held_out=True` alongside a non-null `public_since`) to `models/datasets.py`. Attach
  `contamination: ContaminationMetadata | None = None` to both `ResolvedDataset` and
  `DatasetPreset` — additive, no `schema_version` bump. Both built-in presets (`gsm8k`,
  `swe-bench-verified`) are annotated `status=SUSPECT`. Add
  `graders/contamination.py::find_canary_leaks`/`canary_leak_evidence` as pure, stdlib-only,
  policy-free detection primitives any `Grader` implementation may call.
- **Alternatives:**
  1. *Reuse `ResolvedDataset.card_metadata` instead of a new field.* Rejected: verified in
     `datasets/huggingface.py` that `card_metadata` is populated verbatim from the provider's own
     API response (provider-attested); writing a framework-asserted label into it would conflate two
     distinct provenance sources ADR-0002's discipline keeps separate elsewhere.
  2. *A plain `contamination_suspect: bool` instead of `ContaminationStatus`.* Rejected on the same
     grounds ADR-0002 itself already rejects boolean status fields (its own Alternative 3): a
     boolean cannot distinguish "never checked" from "checked and clean" without the exact lossy
     conflation that ADR-0002 exists to prevent.
  3. *Auto-thread `canary_ids` into `EvalSample.metadata` at adapter-projection time now.* Rejected
     for this ADR: `BenchmarkAdapter.prepare` structurally receives only a `SourceRecord`, not a
     `ResolvedDataset` (verified against `runner.py`'s call site); wiring this requires a breaking
     protocol change to two concrete adapters and deserves its own superseding ADR.
- **Consequences:** Both built-in presets are honestly labeled `SUSPECT`; a caller building a
  private eval set gains a documented, typed way to declare `held_out=True` and register canaries;
  no existing model, adapter, or grader call site changes behavior (purely additive); no new
  dependency.
- **Validation evidence:** [STUB — fill in at implementation time with real test paths/line numbers]
  `tests/contract/test_models.py` (round-trip + enum-preservation), `tests/unit/datasets/
  test_catalog.py` (built-in preset annotation + the all-presets loop test), `tests/unit/graders/
  test_contamination.py` (helper hermetic unit tests + one end-to-end `GradeResult.evidence`
  integration test).
- **Supersession:** A future change threading contamination metadata automatically into
  `EvalSample`/grading (the deferred `BenchmarkAdapter.prepare` signature change, or any change to
  how `ContaminationStatus` values are derived rather than caller-asserted) must supersede this ADR.

## Effort estimate: M

No new dependency, no protocol/signature change, no live/network test, no CLI surface — the reason
this is not L. But the honest touch count is real and cuts across every layer the repo's own
contract tests actively enforce, which is why this is not S: two new frozen contracts plus two
field additions (`models/datasets.py`, `models/__init__.py`), one preset field plus two annotations
(`datasets/presets.py`), a new module with two pure functions plus its `graders/__init__.py`
export, five-plus new/extended hermetic test files across three test directories
(`tests/contract/`, `tests/unit/datasets/`, `tests/unit/graders/`), a new documentation section
with an `mkdocs --strict` obligation, and a new ADR whose number is contested in-flight with a
sibling spec and whose landing is gated by two separate contract tests
(`test_required_prefixes_cover_every_committed_adr_file`,
`test_landing_page_adr_claims_match_committed_adr_count`) that fail loudly, by design, if the
prefix table and the landing page are not updated in the same commit.
