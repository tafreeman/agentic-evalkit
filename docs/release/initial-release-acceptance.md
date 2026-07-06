# Initial Release Acceptance Audit

**Date:** 2026-07-03
**Checkpoint decision:** `CONTINUE_FULL_V1` (see `v0.1-checkpoint.md`) — the full
v1 surface (calibrated judges, advanced statistics, rich reporters, and the
`compare`/`report` CLI commands) is implemented, so no Slice-4b criterion is
deferred.

This audit maps every acceptance criterion in design §17 to the concrete
test or command that proves it and the resulting artifact. Criteria requiring
live-provider or clean-wheel evidence are backed by an actual run, not code
inspection.

## Acceptance criteria (design §17)

| # | Criterion | Proof (test / command) | Result | Status |
|---|---|---|---|---|
| 1 | Clean env installs the wheel and runs the CLI outside ARP/EK checkouts | `tests/integration/test_clean_wheel.py`; orchestrator clean-room (`v0.1-checkpoint.md`) | Wheel installs in a bare venv outside the repo; `doctor`/`init`/`run` exit 0 | PASS |
| 2 | Static checks prove no ARP/`agentic-tools`/EK imports | `tests/contract/test_dependency_boundary.py` | AST scan of `src/`; zero imports rooted in `agentic_v2`/`tools`/`executionkit` | PASS |
| 3 | List curated datasets and search Hugging Face immediately after install | `tests/integration/test_cli.py::test_curated_and_init...`; clean-wheel `datasets curated --format json` | Both presets listed offline; HF search wired to the same catalog | PASS |
| 4 | GSM8K & SWE-bench Verified configs/splits have live Dataset Viewer evidence | `tests/live/test_huggingface_live.py -m live` | 2/2 passed; both presets resolve + preview 2 rows against the real Hub | PASS |
| 5 | Search/inspect/preview/retrieval without `datasets`/`pyarrow`/Docker/manual imports | `tests/unit/datasets/test_huggingface_provider.py`, `test_catalog.py`; clean-wheel `find_spec` check | Provider uses only `httpx`+`huggingface_hub`; forbidden modules absent in wheel env | PASS |
| 6 | Every resolved run pins immutable dataset+code provenance; outages fail before execution, no partial "latest" | `tests/integration/test_runner.py`; canonical JSON `provenance` | `dataset_revision` = commit SHA `740312a…`; runner resolves once and pins provenance | PASS |
| 7 | Provider failures stay typed and cannot appear as empty datasets | `test_huggingface_provider.py` (error mapping), `test_local_provider.py` (`DatasetSchemaMismatch`, never empty) | Typed errors on 401/403/404/422/429/5xx; malformed input raises, never yields `[]` | PASS |
| 8 | GSM8K completes through a target + objective grader via one manifest, Python and CLI | `test_runner.py`; `test_cli.py` run tests; clean-room `run eval.yaml --limit 5` | 5/5 executed and graded from one manifest; canonical JSON written | PASS |
| 9 | SWE-bench Verified previews, projects, and exports official predictions without Docker | `tests/unit/benchmarks/test_swebench_adapter.py`; live gate preview | `export_prediction` returns exactly the 3 official keys; no filesystem/Docker | PASS |
| 10 | Missing authoritative SWE-bench capability returns `unavailable`, never a substitute score | `tests/unit/benchmarks/test_harness.py`; ADR-0005 | `UnavailableHarnessExecutor` → status `unavailable`, `resolved=None` | PASS |
| 11 | Objective hard-gate failures cannot be offset by model-judge scores | `tests/unit/graders/test_composite.py::test_failed_hard_gate_cannot_be_averaged_away`; ADR-0007 | Hard-gate failure forces non-gating FAIL regardless of weighted mean | PASS |
| 12 | Uncalibrated judges cannot gate releases | `tests/unit/graders/test_judge.py::test_expired_calibration_cannot_gate` (+ calibration-minimum, fingerprint, position-bias tests); ADR-0007 | Expired/insufficient/mismatched calibration → `GradeStatus.UNAVAILABLE`, never a gating pass | PASS |
| 13 | Reports separate task failure from infra/timeout/abstention/cancellation/unavailable | `tests/unit/stats/test_aggregate.py`; `RunSummary`; reporters | Eight distinct outcome buckets; operational outcomes never folded into `failed` | PASS |
| 14 | Compatible runs → paired comparison with uncertainty; incompatible → rejected with reasons | `tests/unit/stats/test_compare.py`; CLI `compare` tests | Seeded paired bootstrap (2.5/97.5 percentiles); `IncompatibleRuns` lists all mismatches | PASS |
| 15 | All required ADRs accepted and consistent with code and docs | `tests/contract/test_adrs.py`; reconciliation below | ADR-0001..0009 all `Accepted`, 6-section template, non-contradictory | PASS |
| 16 | Clean-wheel/CLI/typing/test/coverage/docs gates pass; live-provider evidence passes or is a classified outage | Offline matrix + live gate (below) | ruff/mypy clean; 416 passed; 88.91% branch cov; `mkdocs build --strict` 0; live 2/2 | PASS |
| 17 | From a clean install, `init --preset gsm8k` + `run` against the demo target succeeds with no importer/manual/`datasets`/`pyarrow`/Docker; provider failures → stable code + remediation, not a traceback | `v0.1-checkpoint.md` clean-room; documented quickstart (Task 15 Step 7) | End-to-end run + canonical JSON + self-contained HTML; `datasets inspect hf:missing/not-found` → exit 4 `[dataset_access_denied]`, no traceback | PASS |

**Result: 17 / 17 criteria pass.** No criterion is deferred (CONTINUE_FULL_V1).

## ADR ↔ code ↔ metadata reconciliation (Step 2)

| Check | Evidence | Status |
|---|---|---|
| Package deps match ADR-0003/0009 | `pyproject.toml` base deps = pydantic/typer/rich/pyyaml/huggingface-hub/httpx/jinja2; extras `parquet`/`judges`/`swebench`; `test_dependency_boundary.py` | PASS |
| Cache identity matches ADR-0004 | `test_cache.py` (canonical-JSON digest, checksum manifest, corruption ≠ offline miss) | PASS |
| Harness status semantics match ADR-0005 | `test_harness.py` (`completed`/`unavailable`/`error`, tri-state `resolved`) | PASS |
| Forbidden imports match ADR-0001/0006 | `test_dependency_boundary.py`; clean-wheel `find_spec` None for `agentic_v2`/`tools.agents`/`executionkit` | PASS |
| Objective hard gates match ADR-0007 | `test_composite.py` (noncompensable hard gate) | PASS |
| Judge gates match ADR-0007 (CONTINUE_FULL_V1) | `test_judge.py` (calibration minimums, expiry, position-bias) | PASS |
| Report compatibility matches ADR-0008 (CONTINUE_FULL_V1) | `test_compare.py` (provenance gating, seeded bootstrap) | PASS |
| All required ADRs accepted and consistent (§17.15) | ADR-0001..0009 all `Accepted` and reconciled above. ADR-0001-0004/0009 were committed within their own tasks; ADR-0005-0008 were recorded in one batch immediately after their parallel implementation wave and before this audit (a consequence of parallelized execution, noted for transparency). | PASS |

## Release gate summary

- `ruff check .` / `ruff format --check .` — clean
- `mypy` — no issues in 56 source files
- `pytest -m "not live" --cov=agentic_evalkit --cov-branch` — 416 passed, 2 deselected, **88.91%** branch coverage (≥ 80% gate)
- `mkdocs build --strict` — exit 0, no warnings
- `uv build` — sdist + wheel produced
- Live gate `pytest tests/live -m live` — 2 passed (Hugging Face reachable; no outage to classify)
- Clean-wheel gate `tests/integration/test_clean_wheel.py` — passed (isolated wheel, forbidden modules absent)

## Known issues

- `doctor` prints Hugging Face latency unrounded (cosmetic); tracked in `v0.1-checkpoint.md`.
- **Resolved 2026-07-05:** all default CLI integration coverage is hermetic, provider failures and exit code 4 are exercised with canned providers, and real provider/CLI checks are confined to the opt-in `tests/live/` workflow.
- The self-contained HTML report embeds a provenance parquet URL as *data* (not a loaded resource); the report loads nothing remote and renders fully offline.

## Follow-on boundary

The official SWE-bench Docker executor is out of scope for this release and
requires its own plan (see `docs/plans/README.md`), which may begin only now
that this acceptance audit passes.
