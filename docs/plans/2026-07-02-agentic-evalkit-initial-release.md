# agentic-evalkit Initial Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an independently installable Python library and CLI that discovers Hugging Face and local datasets, evaluates callable/subprocess/HTTP targets with objective-first graders, and can ship an objective-only v0.1 before adding the deferred full-analytics surface.

**Architecture:** `agentic-evalkit` is a host-neutral pipeline with explicit provider, adapter, target, grader, aggregation, and reporter boundaries. Immutable Pydantic contracts connect those boundaries; Python entry points provide extensions; ARP and ExecutionKit remain outside the dependency graph. The initial release includes SWE-bench dataset projection and harness contracts, while the official Docker executor receives its own follow-on implementation plan.

**Tech Stack:** Python 3.11+, Hatchling, uv, Pydantic v2, Typer, Rich, PyYAML, huggingface-hub, HTTPX, Jinja2, pytest, pytest-asyncio, pytest-cov, Ruff, mypy, MkDocs Material, GitHub Actions.

---

## Scope and sequencing

This plan implements design Slices 1-4a, pauses at a formal working-product checkpoint, and then implements Slice 4b only if the checkpoint decision selects a full v1 rather than an objective-only v0.1. Slice 5, the official containerized SWE-bench executor, is deliberately excluded. Task 8 establishes `HarnessRequest`, `HarnessResult`, and `HarnessExecutor`, prediction export, capability reporting, and deterministic contract tests so the later executor does not change public schemas.

The canonical identity is `agentic-evalkit` / `agentic_evalkit` / `agentic-evalkit`. A legacy ARP-local draft uses `agentic-v2-eval`, but this standalone repository supersedes that name and does not require an ARP change or import integration. ARP and EK are systems under test, reachable only through public target boundaries.

Dependency decisions verified before execution:

- both companion repositories use MIT, so this public package uses MIT;
- PyPI lists HTTPX 0.28.1 as stable and 1.0 only as a development prerelease, so pin `httpx>=0.28.1,<1`;
- pin `typer>=0.12,<1` rather than an unbounded pre-1 range;
- begin with an 80% branch-aware coverage floor and report the achieved percentage at release rather than forcing 90% before the CLI/report surface exists.

### Milestone delivery

| Milestone | Scope | Merge/release boundary |
|---|---|---|
| A | Tasks 1-7: repository, contracts, plugins, cache, providers, catalog | Merge as the dataset-foundation PR |
| B | Tasks 8-11, Task 13 Steps 1-4, and Task 14 Steps 1-9: benchmark contracts, targets, objective graders, runner, JSON report, runnable CLI | Merge as the objective-evaluation PR and execute the v0.1 checkpoint |
| C | Task 10 Steps 6-9, Task 12, Task 13 Steps 5-7, Task 14 Steps 10-11, and Tasks 15-16 | Execute deferred analytics only after `CONTINUE_FULL_V1`; for `SHIP_V0_1`, move them to the v0.2 plan and run Tasks 15-16 in objective-only mode |

Each milestone starts from updated `main`, uses its own branch/worktree, runs its complete verification matrix, and lands independently. Do not carry one 16-task branch through all milestones.

Every ADR is committed before the production code it governs:

| ADR | Governing task |
|---|---|
| ADR-0001 standalone boundary | Task 1 |
| ADR-0002 immutable contracts | Task 2 |
| ADR-0003 providers and Hugging Face baseline | Task 3 |
| ADR-0004 immutable cache | Task 4 |
| ADR-0005 adapter/harness separation | Task 8 |
| ADR-0006 execution targets and ARP/EK boundary | Task 9 |
| ADR-0007 grading and calibrated judges | Task 10 |
| ADR-0008 statistics and comparability | Task 12 |
| ADR-0009 optional dependencies/plugins | Task 1 |

## Planned file map

```text
agentic-evalkit/
  .github/workflows/ci.yml               # supported Python matrix and quality gates
  docs/
    adr/0001-0009-*.md                    # accepted architecture decisions
    guides/quickstart.md                  # install-to-first-report workflow
    guides/providers.md                   # provider and dataset authoring
    guides/graders.md                     # objective and judge policy
    specs/...                             # approved design
    plans/...                             # this implementation plan and review record
  src/agentic_evalkit/
    __init__.py                            # intentionally small public surface
    errors.py                              # typed framework failures
    models/                                # immutable wire contracts only
    plugins.py                             # entry-point discovery
    datasets/                              # providers, cache, catalog, presets
    benchmarks/                            # adapter and harness boundaries
    targets/                               # callable, subprocess, HTTP targets
    graders/                               # objective, composite, judge graders
    stats/                                 # aggregation and compatibility
    reporters/                             # JSON, JSONL, Markdown, HTML
    artifacts.py                           # content-addressed run artifacts
    runner.py                              # pipeline orchestration
    cli/                                   # Typer commands and presentation
    py.typed
  tests/
    contract/                              # public schema/plugin compatibility
    unit/                                  # deterministic component tests
    integration/                           # provider/target/pipeline tests
    live/                                  # opt-in Hugging Face tests
    fixtures/                              # captured source responses and manifests
  pyproject.toml
  mkdocs.yml
  LICENSE                                 # MIT terms
  CHANGELOG.md                            # release history
  CONTRIBUTING.md                         # local development and review workflow
  SECURITY.md                             # private vulnerability reporting policy
```

Files stay focused: models do not perform I/O, providers do not grade, targets do not know benchmark types, graders do not execute targets, and reporters consume completed run models only.

### Task 1: Repository foundation and dependency decisions

**Files:**
- Create: `.gitignore`
- Create: `.github/workflows/ci.yml`
- Create: `LICENSE`
- Create: `CHANGELOG.md`
- Create: `CONTRIBUTING.md`
- Create: `SECURITY.md`
- Create: `pyproject.toml`
- Create: `src/agentic_evalkit/__init__.py`
- Create: `src/agentic_evalkit/cli/__init__.py`
- Create: `src/agentic_evalkit/cli/app.py`
- Create: `src/agentic_evalkit/py.typed`
- Create: `tests/unit/test_package.py`
- Create: `docs/adr/0001-standalone-boundary.md`
- Create: `docs/adr/0009-optional-dependencies-and-plugins.md`

- [ ] **Step 1: Record ADR-0001 before adding package dependencies**

Create `docs/adr/0001-standalone-boundary.md` with status `Accepted`, the decision that the package imports no ARP, `agentic-tools`, or EK modules, and these consequences: integrations use public target protocols; clean-wheel tests run outside all source trees; ARP/EK changes are out of scope.

- [ ] **Step 2: Record ADR-0009 before defining extras**

Create `docs/adr/0009-optional-dependencies-and-plugins.md` with status `Accepted`. Record that the base install contains Hugging Face discovery, while `parquet`, `judges`, and `swebench` are capability extras; extension points use versioned `agentic_evalkit.*` entry-point groups; plugin load failures are reported rather than ignored.

- [ ] **Step 3: Write the failing package metadata test**

Before the test, add the MIT license using the standard MIT text and copyright `2026 agentic-evalkit contributors`. Add `CHANGELOG.md` with an `Unreleased` section, `CONTRIBUTING.md` with uv setup and the offline verification matrix, and `SECURITY.md` directing vulnerability reports to private GitHub security advisories rather than public issues. Reference all four files from `pyproject.toml`/README metadata.

```python
# tests/unit/test_package.py
from importlib.metadata import version

import agentic_evalkit


def test_package_version_matches_distribution() -> None:
    assert agentic_evalkit.__version__ == version("agentic-evalkit")
```

- [ ] **Step 4: Run the test and verify the package does not exist yet**

Run: `uv run pytest tests/unit/test_package.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'agentic_evalkit'`.

- [ ] **Step 5: Add build, runtime, development, and CLI metadata**

Create `pyproject.toml` with Hatchling build metadata, Python `>=3.11`, MIT license metadata, package discovery under `src`, and script entry point `agentic-evalkit = "agentic_evalkit.cli:app"`. Use compatible base ranges: Pydantic 2, `typer>=0.12,<1`, Rich below 15, PyYAML below 7, huggingface-hub below 2, `httpx>=0.28.1,<1`, and Jinja2 below 4. Add `parquet`, `judges`, and `swebench` as empty capability groups until their implementation plans select concrete packages. Add a `dev` dependency group containing pytest, pytest-asyncio, pytest-cov, Ruff, mypy, build, MkDocs, and MkDocs Material.

Configure:

```toml
[tool.pytest.ini_options]
addopts = "--strict-config --strict-markers"
asyncio_mode = "auto"
markers = [
  "integration: component integration tests",
  "live: requires network access",
]

[tool.coverage.run]
branch = true
source = ["agentic_evalkit"]

[tool.coverage.report]
fail_under = 80
show_missing = true

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "ASYNC", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["agentic_evalkit"]
```

Create the initial package:

```python
# src/agentic_evalkit/__init__.py
from importlib.metadata import version

__version__ = version("agentic-evalkit")

__all__ = ["__version__"]
```

Create the initial CLI so the packaging gate is valid from the first commit:

```python
# src/agentic_evalkit/cli/app.py
import typer

from agentic_evalkit import __version__

app = typer.Typer(no_args_is_help=True, help="Evaluate agentic systems with reproducible evidence.")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show the installed version."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
```

```python
# src/agentic_evalkit/cli/__init__.py
from agentic_evalkit.cli.app import app

__all__ = ["app"]
```

Create an empty `src/agentic_evalkit/py.typed` marker and a `.gitignore` covering `.venv/`, `dist/`, `build/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/`, and `__pycache__/`.

- [ ] **Step 6: Add the CI matrix**

Create `.github/workflows/ci.yml` with Python 3.11, 3.12, and 3.13 jobs on Ubuntu and Windows. Each job installs uv, runs `uv sync --all-groups`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`, and `uv run pytest -m "not live" --cov --cov-report=term-missing`. Add one Ubuntu packaging job that runs `uv build` and installs the wheel into a temporary virtual environment before executing `agentic-evalkit --help`.

- [ ] **Step 7: Run foundation checks**

Run:

```powershell
uv sync --all-groups
uv run pytest tests/unit/test_package.py -v
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Expected: package test PASS and all quality commands exit 0.

- [ ] **Step 8: Commit the foundation**

```powershell
git add .gitignore .github LICENSE CHANGELOG.md CONTRIBUTING.md SECURITY.md pyproject.toml src tests/unit/test_package.py docs/adr/0001-standalone-boundary.md docs/adr/0009-optional-dependencies-and-plugins.md
git commit -m "build: establish standalone package foundation"
```

### Task 2: Immutable public contracts

**Files:**
- Create: `docs/adr/0002-immutable-versioned-contracts.md`
- Create: `src/agentic_evalkit/models/base.py`
- Create: `src/agentic_evalkit/models/datasets.py`
- Create: `src/agentic_evalkit/models/samples.py`
- Create: `src/agentic_evalkit/models/execution.py`
- Create: `src/agentic_evalkit/models/grades.py`
- Create: `src/agentic_evalkit/models/runs.py`
- Create: `src/agentic_evalkit/models/__init__.py`
- Create: `tests/contract/test_models.py`

- [ ] **Step 1: Accept ADR-0002**

Record Pydantic v2 frozen models, `extra="forbid"`, explicit `schema_version`, JSON-compatible values, string enums, backward-compatible additive evolution within schema version 1, and a new schema version for breaking wire changes.

- [ ] **Step 2: Write failing immutability and round-trip tests**

```python
# tests/contract/test_models.py
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentic_evalkit.models import DatasetRef, EvalSample, GradeResult, GradeStatus


def test_models_are_frozen_and_forbid_unknown_fields() -> None:
    ref = DatasetRef(provider="huggingface", dataset_id="openai/gsm8k")
    with pytest.raises(ValidationError):
        DatasetRef(provider="huggingface", dataset_id="openai/gsm8k", unknown=True)
    with pytest.raises(ValidationError):
        ref.dataset_id = "other/dataset"  # type: ignore[misc]


def test_sample_round_trips_through_versioned_json() -> None:
    sample = EvalSample(
        sample_id="gsm8k:main:test:0",
        input={"question": "1+1?"},
        reference="2",
        source_digest="sha256:abc",
        adapter="gsm8k@1",
    )
    assert EvalSample.model_validate_json(sample.model_dump_json()) == sample


def test_grade_status_is_not_collapsed_to_boolean() -> None:
    grade = GradeResult(
        sample_id="s1",
        grader="exact@1",
        status=GradeStatus.ABSTAIN,
        score=None,
        hard_gate=False,
        created_at=datetime.now(UTC),
    )
    assert grade.status is GradeStatus.ABSTAIN
```

- [ ] **Step 3: Run contract tests and confirm missing models**

Run: `uv run pytest tests/contract/test_models.py -v`

Expected: FAIL with import errors for `agentic_evalkit.models`.

- [ ] **Step 4: Implement the shared frozen base and dataset/sample contracts**

```python
# src/agentic_evalkit/models/base.py
from typing import Literal

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["1"] = "1"
```

Implement `DatasetRef`, `ResolvedDataset`, `SourceRecord`, `SearchHit`, `SearchPage`, and `SamplePage` in `models/datasets.py`. Implement `GraderSpec` and `EvalSample` in `models/samples.py`. Use `pydantic.JsonValue` for provider data and sample inputs, `datetime` for timestamps, and tuples instead of mutable lists in public models.

- [ ] **Step 5: Implement execution, grade, and run contracts**

Define string enums and models matching the approved spec:

```python
class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ERROR = "error"


class GradeStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"
    ABSTAIN = "abstain"
    UNAVAILABLE = "unavailable"
```

Add `ExecutionRequest`, `NormalizedExecutionResult`, `GradeResult`, `DatasetSelection`, `SamplingPolicy`, `EvalRunManifest`, `SampleResult`, and `EvalRunResult`. Require IDs, timestamps, component versions, attempt numbers, target fingerprints, evidence mappings, and provenance mappings where the design calls for them.

- [ ] **Step 6: Export the public models and run contract tests**

Expose the models from `models/__init__.py`, then run:

```powershell
uv run pytest tests/contract/test_models.py -v
uv run mypy
```

Expected: all model contract tests PASS and mypy exits 0.

- [ ] **Step 7: Commit the contracts**

```powershell
git add docs/adr/0002-immutable-versioned-contracts.md src/agentic_evalkit/models tests/contract/test_models.py
git commit -m "feat: add immutable evaluation contracts"
```

### Task 3: Extension discovery and typed errors

**Files:**
- Create: `docs/adr/0003-provider-plugins-and-hugging-face-baseline.md`
- Create: `src/agentic_evalkit/errors.py`
- Create: `src/agentic_evalkit/plugins.py`
- Create: `tests/unit/test_errors.py`
- Create: `tests/unit/test_plugins.py`

- [ ] **Step 1: Accept ADR-0003**

Record built-in `local` and `huggingface` providers, Python entry-point group `agentic_evalkit.providers.v1`, explicit plugin API version 1, Hugging Face support in the base wheel, remote code disabled, and classified provider errors.

- [ ] **Step 2: Write failing typed-error and plugin tests**

```python
# tests/unit/test_plugins.py
from importlib.metadata import EntryPoint

import pytest

from agentic_evalkit.errors import PluginCompatibilityError
from agentic_evalkit.plugins import load_plugins


def test_rejects_plugin_with_wrong_api_version(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = EntryPoint(
        name="bad",
        value="tests.fixtures.bad_plugin:plugin",
        group="agentic_evalkit.providers.v1",
    )
    monkeypatch.setattr("agentic_evalkit.plugins._entry_points", lambda group: (entry,))
    with pytest.raises(PluginCompatibilityError, match="api_version=2"):
        load_plugins("agentic_evalkit.providers.v1", expected_api_version="1")
```

Add `tests/fixtures/bad_plugin.py` with a frozen object exposing `api_version = "2"`.

- [ ] **Step 3: Run tests and confirm missing modules**

Run: `uv run pytest tests/unit/test_errors.py tests/unit/test_plugins.py -v`

Expected: FAIL because `errors` and `plugins` are not implemented.

- [ ] **Step 4: Implement the error hierarchy**

Create `AgenticEvalkitError` with stable `code`, human message, and context mapping. Add subclasses for dataset not found, config required, split not found, access denied, license rejected, integrity failure, schema mismatch, provider unavailable, unsafe code required, rate limited, offline cache miss, plugin compatibility, target failure, target timeout, grader error, and incompatible runs. Ensure `str(error)` contains the code and message but never serializes context values marked secret.

- [ ] **Step 5: Implement deterministic entry-point loading**

`load_plugins(group, expected_api_version)` must sort entry points by name, load each object, verify its `api_version`, reject duplicate plugin names, and return an immutable mapping. It must wrap import failures in `PluginCompatibilityError` with the entry-point name and original exception class.

- [ ] **Step 6: Run and commit**

```powershell
uv run pytest tests/unit/test_errors.py tests/unit/test_plugins.py -v
uv run ruff check src/agentic_evalkit/errors.py src/agentic_evalkit/plugins.py tests
uv run mypy
git add docs/adr/0003-provider-plugins-and-hugging-face-baseline.md src/agentic_evalkit/errors.py src/agentic_evalkit/plugins.py tests
git commit -m "feat: add extension discovery and typed errors"
```

Expected: tests and quality checks PASS; commit succeeds.

### Task 4: Immutable content-addressed cache

**Files:**
- Create: `docs/adr/0004-content-addressed-dataset-cache.md`
- Create: `src/agentic_evalkit/datasets/cache.py`
- Create: `tests/unit/datasets/test_cache.py`

- [ ] **Step 1: Accept ADR-0004**

Record canonical JSON hashing, SHA-256 object IDs, separate page/full-dataset record types, replace-based publication, checksums, platform user-cache default, exact offline lookup, and lock-scoped concurrent writes. Explicitly document that `Path.replace()` atomicity depends on the local Windows filesystem; correctness therefore relies on checksum validation and mandatory Windows CI cache tests rather than an unconditional cross-filesystem atomicity claim.

- [ ] **Step 2: Write failing cache identity and corruption tests**

```python
# tests/unit/datasets/test_cache.py
from pathlib import Path

import pytest

from agentic_evalkit.datasets.cache import CacheKey, DatasetCache
from agentic_evalkit.errors import DatasetIntegrityError, OfflineCacheMiss


def test_cache_key_changes_for_revision_config_split_and_page() -> None:
    base = CacheKey(
        provider="huggingface",
        dataset_id="openai/gsm8k",
        revision="abc",
        config="main",
        split="test",
        offset=0,
        limit=10,
    )
    variants = (
        base.model_copy(update={"revision": "def"}),
        base.model_copy(update={"config": "socratic"}),
        base.model_copy(update={"split": "train"}),
        base.model_copy(update={"offset": 10}),
    )
    assert all(item.digest() != base.digest() for item in variants)


def test_corruption_and_offline_miss_are_distinct(tmp_path: Path) -> None:
    cache = DatasetCache(tmp_path)
    key = CacheKey(
        provider="local",
        dataset_id="items.jsonl",
        revision="sha256:a",
        config=None,
        split=None,
        offset=0,
        limit=10,
    )
    with pytest.raises(OfflineCacheMiss):
        cache.read(key)
    cache.write(key, b"valid")
    cache.payload_path(key).write_bytes(b"changed")
    with pytest.raises(DatasetIntegrityError):
        cache.read(key)
```

- [ ] **Step 3: Run the tests and confirm the cache is absent**

Run: `uv run pytest tests/unit/datasets/test_cache.py -v`

Expected: FAIL with import error for `agentic_evalkit.datasets.cache`.

- [ ] **Step 4: Implement canonical keys and atomic entries**

Implement frozen `CacheKey` with all fields in the test plus projection/filter/data-file digests and `record_type`. `digest()` must SHA-256 hash UTF-8 canonical JSON using sorted keys and compact separators. `DatasetCache.write()` writes payload and manifest to temporary files, flushes them, then uses `Path.replace()` under a per-key lock. The manifest records payload checksum, byte count, creation time, and key JSON. `read()` verifies key identity, byte count, and checksum before returning bytes.

- [ ] **Step 5: Add concurrency and exact-page tests**

Use a `ThreadPoolExecutor` to write the same key concurrently and assert one valid final entry. Write two page keys with different offsets and assert both payloads remain addressable. Run the test module repeatedly with `pytest -x --count=5` after adding `pytest-repeat` to the dev group.

The Windows CI matrix from Task 1 must run these cache tests; do not mark Task 4 complete from a Linux-only result.

- [ ] **Step 6: Run and commit**

```powershell
uv run pytest tests/unit/datasets/test_cache.py -v
uv run mypy
git add pyproject.toml docs/adr/0004-content-addressed-dataset-cache.md src/agentic_evalkit/datasets/cache.py tests/unit/datasets/test_cache.py
git commit -m "feat: add content-addressed dataset cache"
```

### Task 5: Dataset provider protocol and local files

**Files:**
- Create: `src/agentic_evalkit/datasets/base.py`
- Create: `src/agentic_evalkit/datasets/local.py`
- Create: `src/agentic_evalkit/datasets/__init__.py`
- Create: `tests/unit/datasets/test_local_provider.py`
- Create: `tests/fixtures/datasets/items.jsonl`
- Create: `tests/fixtures/datasets/items.csv`
- Create: `tests/fixtures/datasets/items.yaml`

- [ ] **Step 1: Write failing provider contract tests**

```python
# tests/unit/datasets/test_local_provider.py
from pathlib import Path

import pytest

from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.models import DatasetRef


@pytest.mark.asyncio
async def test_resolve_preview_and_iterate_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "items.jsonl"
    source.write_text('{"id":"a","prompt":"alpha"}\n{"id":"b","prompt":"beta"}\n')
    provider = LocalDatasetProvider(allowed_roots=(tmp_path,))
    resolved = await provider.resolve(DatasetRef(provider="local", dataset_id=str(source)))
    page = await provider.preview(resolved, offset=1, limit=1)
    records = [record async for record in provider.iter_records(resolved, offset=0, limit=None)]
    assert resolved.revision.startswith("sha256:")
    assert page.total_rows == 2
    assert page.records[0].data["id"] == "b"
    assert [record.row_id for record in records] == ["0", "1"]


@pytest.mark.asyncio
async def test_rejects_path_outside_allowed_roots(tmp_path: Path) -> None:
    provider = LocalDatasetProvider(allowed_roots=(tmp_path / "allowed",))
    with pytest.raises(ValueError, match="outside allowed roots"):
        await provider.resolve(DatasetRef(provider="local", dataset_id=str(tmp_path / "x.json")))
```

- [ ] **Step 2: Run tests and verify the provider is missing**

Run: `uv run pytest tests/unit/datasets/test_local_provider.py -v`

Expected: FAIL with import error for `LocalDatasetProvider`.

- [ ] **Step 3: Define the async provider protocol**

In `datasets/base.py`, define runtime-checkable `DatasetProvider` with `api_version = "1"` and async `search`, `resolve`, `preview`, `iter_records`, and `healthcheck`. Add `ProviderHealth` as a frozen model with `status`, `latency_ms`, `capabilities`, and optional error code. Use keyword-only pagination parameters and `AsyncIterator[SourceRecord]`.

- [ ] **Step 4: Implement local decoding and immutable resolution**

`LocalDatasetProvider` must resolve the path, enforce allowed roots, reject directories and unsupported suffixes, calculate SHA-256 over raw bytes, and decode:

- JSON as a list of objects or an object containing a `records` list;
- JSONL as one object per nonblank line;
- CSV with `csv.DictReader`;
- YAML as a list of objects using `yaml.safe_load`.

Validate every row as `dict[str, JsonValue]`, use zero-based string row IDs, and calculate each `SourceRecord.digest` from canonical JSON. `search()` returns an empty successful page because local roots are not recursively indexed. `healthcheck()` verifies all configured roots are readable.

- [ ] **Step 5: Add fixture parity tests**

Populate the three fixture formats with the same two logical rows and assert all produce identical canonical `SourceRecord.data` and different source revisions. Add malformed JSONL and scalar-YAML cases and assert `DatasetSchemaMismatch`, not empty results.

- [ ] **Step 6: Run and commit**

```powershell
uv run pytest tests/unit/datasets/test_local_provider.py -v
uv run ruff check src/agentic_evalkit/datasets tests/unit/datasets
uv run mypy
git add src/agentic_evalkit/datasets tests/unit/datasets tests/fixtures/datasets
git commit -m "feat: add local dataset provider"
```

### Task 6: Hugging Face discovery and Dataset Viewer provider

**Files:**
- Create: `src/agentic_evalkit/datasets/huggingface.py`
- Create: `tests/fixtures/huggingface/dataset_info.json`
- Create: `tests/fixtures/huggingface/is_valid.json`
- Create: `tests/fixtures/huggingface/splits.json`
- Create: `tests/fixtures/huggingface/rows.json`
- Create: `tests/fixtures/huggingface/size.json`
- Create: `tests/fixtures/huggingface/statistics.json`
- Create: `tests/fixtures/huggingface/parquet.json`
- Create: `tests/unit/datasets/test_huggingface_provider.py`
- Create: `tests/live/test_huggingface_live.py`

- [ ] **Step 1: Write failing request-shape and pagination tests**

```python
# tests/unit/datasets/test_huggingface_provider.py
import httpx
import pytest

from agentic_evalkit.datasets.huggingface import HuggingFaceDatasetProvider
from agentic_evalkit.models import DatasetRef


@pytest.mark.asyncio
async def test_preview_always_sends_resolved_config_and_split() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/splits"):
            return httpx.Response(200, json={"splits": [{"config": "default", "split": "test"}]})
        return httpx.Response(
            200,
            json={
                "rows": [{"row_idx": 0, "row": {"instance_id": "x"}}],
                "num_rows_total": 1,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HuggingFaceDatasetProvider(client=client, hub=_FakeHub(sha="abc123"))
    resolved = await provider.resolve(
        DatasetRef(
            provider="huggingface",
            dataset_id="princeton-nlp/SWE-bench_Verified",
            config="default",
            split="test",
        )
    )
    await provider.preview(resolved, offset=0, limit=1)
    row_request = seen[-1]
    assert row_request.url.params["config"] == "default"
    assert row_request.url.params["split"] == "test"
    await client.aclose()
```

Define `_FakeHub` in the test with `dataset_info()` and `list_datasets()` methods returning frozen fixture objects.

- [ ] **Step 2: Run the test and confirm the provider is absent**

Run: `uv run pytest tests/unit/datasets/test_huggingface_provider.py -v`

Expected: FAIL with import error for `HuggingFaceDatasetProvider`.

- [ ] **Step 3: Implement search and immutable resolution**

Use injected `huggingface_hub.HfApi` and `httpx.AsyncClient`. Run synchronous Hub calls through `asyncio.to_thread`. `search()` maps `list_datasets(search=query, limit=...)` results to `SearchHit` with dataset ID, SHA, tags, gated/private flags, downloads, and card metadata. `resolve()`:

1. calls Dataset Viewer `/is-valid` and rejects datasets the viewer cannot safely serve;
2. calls `dataset_info(repo_id, revision)` and requires a nonempty commit SHA;
3. calls Dataset Viewer `/splits`;
4. validates or uniquely infers config and split;
5. raises `DatasetConfigRequired` for ambiguous configs;
6. calls `/size`, `/statistics`, and `/parquet` for the selected config/split;
7. stores license, citation, card, gated access, schema/statistics, size, Parquet-file metadata, and selected files in `ResolvedDataset`.

Never import `datasets` or `pyarrow`, and never set `trust_remote_code=True`.

- [ ] **Step 4: Implement preview and bounded iteration**

Use `/rows` with URL-encoded `dataset`, `config`, `split`, `offset`, and `length`; cap each page at 100. Convert each row to `SourceRecord` using `row_idx` and a canonical digest. `preview()` returns one page and the upstream `num_rows_total`. `iter_records()` requests successive pages until it reaches the caller limit, upstream total, or an empty page. A partial page is not cached as a full dataset.

Map HTTP 401/403 to `DatasetAccessDenied`, 404 to `DatasetNotFound`, 422 config errors to `DatasetConfigRequired`, 429 to `DatasetRateLimited` with retry metadata, and transport/5xx failures to `DatasetProviderUnavailable`.

`healthcheck()` calls `/is-valid` for `openai/gsm8k` with a short timeout and reports latency and rate-limit metadata. Use default `HfApi` authentication so public access needs no token and private/gated access honors `HF_TOKEN` or the standard Hugging Face credential store.

Add a bounded retry policy for connection failures, 429, and 502/503/504. Honor `Retry-After`, otherwise use jittered exponential backoff with at most three retries and a caller-injected sleep function for deterministic tests. Add tests proving 429-then-200 succeeds, repeated 429 raises `DatasetRateLimited`, and nonretryable 4xx responses are attempted once.

- [ ] **Step 5: Add opt-in live tests**

```python
# tests/live/test_huggingface_live.py
import pytest

from agentic_evalkit.datasets.huggingface import HuggingFaceDatasetProvider
from agentic_evalkit.models import DatasetRef


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dataset_id", "config", "split"),
    (
        ("openai/gsm8k", "main", "test"),
        ("princeton-nlp/SWE-bench_Verified", "default", "test"),
    ),
)
async def test_verified_presets_resolve_and_preview(
    dataset_id: str, config: str, split: str
) -> None:
    async with HuggingFaceDatasetProvider.create() as provider:
        resolved = await provider.resolve(
            DatasetRef(
                provider="huggingface",
                dataset_id=dataset_id,
                config=config,
                split=split,
            )
        )
        page = await provider.preview(resolved, offset=0, limit=2)
    assert len(resolved.revision) >= 7
    assert len(page.records) == 2
```

- [ ] **Step 6: Run unit and live verification**

```powershell
uv run pytest tests/unit/datasets/test_huggingface_provider.py -v
uv run pytest tests/live/test_huggingface_live.py -m live -v
```

Expected: unit fixtures PASS; both live presets resolve and return two rows. If Hugging Face is unavailable, record the provider error and rerun before claiming the task complete; do not replace the live gate with mocks.

- [ ] **Step 7: Commit the provider**

```powershell
git add src/agentic_evalkit/datasets/huggingface.py tests/fixtures/huggingface tests/unit/datasets/test_huggingface_provider.py tests/live/test_huggingface_live.py
git commit -m "feat: add Hugging Face dataset discovery"
```

### Task 7: Dataset catalog, presets, and cache integration

**Files:**
- Create: `src/agentic_evalkit/datasets/catalog.py`
- Create: `src/agentic_evalkit/datasets/presets.py`
- Create: `tests/unit/datasets/test_catalog.py`
- Modify: `src/agentic_evalkit/datasets/__init__.py`

- [ ] **Step 1: Write failing preset and provider-routing tests**

```python
# tests/unit/datasets/test_catalog.py
import pytest

from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.presets import BUILTIN_PRESETS


def test_builtin_presets_pin_configs_splits_and_adapters() -> None:
    gsm = BUILTIN_PRESETS["gsm8k"]
    swe = BUILTIN_PRESETS["swe-bench-verified"]
    assert (gsm.ref.dataset_id, gsm.ref.config, gsm.ref.split) == (
        "openai/gsm8k",
        "main",
        "test",
    )
    assert gsm.adapter == "gsm8k@1"
    assert swe.ref.config == "default"
    assert swe.readiness == "prediction_export"


@pytest.mark.asyncio
async def test_unknown_provider_is_explicit() -> None:
    catalog = DatasetCatalog(providers={})
    with pytest.raises(KeyError, match="provider 'missing'"):
        await catalog.search("x", provider="missing", limit=10)
```

- [ ] **Step 2: Run tests and confirm missing catalog**

Run: `uv run pytest tests/unit/datasets/test_catalog.py -v`

Expected: FAIL with import errors for catalog and presets.

- [ ] **Step 3: Implement immutable built-in presets**

Add frozen `DatasetPreset` with name, description, `DatasetRef`, adapter, grader, readiness, and required capabilities. Define exactly:

- `gsm8k`: `openai/gsm8k`, `main/test`, adapter `gsm8k@1`, grader `normalized-exact@1`, readiness `runnable`;
- `swe-bench-verified`: `princeton-nlp/SWE-bench_Verified`, `default/test`, adapter `swebench-verified@1`, grader `swebench-harness@1`, readiness `prediction_export`, required capability `swebench`.

Store the catalog in an immutable mapping and reject duplicate preset names at import time.

- [ ] **Step 4: Implement provider routing and cache decoration**

`DatasetCatalog` accepts built-ins plus entry-point providers. It exposes `list_presets`, `search`, `resolve`, `preview`, and `iter_records`. Route only by `DatasetRef.provider`. Before network preview, construct an exact `CacheKey`; on a verified hit return decoded cached records, and on a miss call the provider then write the exact page. `offline=True` must never call a provider.

- [ ] **Step 5: Test cache hit, offline miss, and plugin collision**

Use a counting fake provider to prove a second identical preview uses the cache, a different offset invokes the provider, and offline mode returns only exact cached pages. Register a plugin with the built-in name `huggingface` and assert a compatibility error rather than silent replacement.

- [ ] **Step 6: Run and commit**

```powershell
uv run pytest tests/unit/datasets/test_catalog.py tests/unit/datasets/test_cache.py -v
uv run mypy
git add src/agentic_evalkit/datasets tests/unit/datasets
git commit -m "feat: add dataset catalog and verified presets"
```

### Task 8: Benchmark adapters and harness contract

**Files:**
- Create: `docs/adr/0005-benchmark-adapters-and-harnesses.md`
- Create: `src/agentic_evalkit/benchmarks/base.py`
- Create: `src/agentic_evalkit/benchmarks/gsm8k.py`
- Create: `src/agentic_evalkit/benchmarks/harness.py`
- Create: `src/agentic_evalkit/benchmarks/swebench.py`
- Create: `src/agentic_evalkit/benchmarks/__init__.py`
- Create: `tests/unit/benchmarks/test_gsm8k.py`
- Create: `tests/unit/benchmarks/test_swebench.py`
- Create: `tests/contract/test_harness.py`

- [ ] **Step 1: Accept ADR-0005**

Record that adapters project records and define artifact/oracle policy, while harness executors perform authoritative isolated verification. A missing harness returns typed `unavailable`; an advisory grader cannot impersonate an authoritative benchmark result.

- [ ] **Step 2: Write failing GSM8K projection tests**

```python
# tests/unit/benchmarks/test_gsm8k.py
from agentic_evalkit.benchmarks.gsm8k import Gsm8kAdapter, extract_final_answer
from agentic_evalkit.models import SourceRecord


def test_projects_question_and_normalized_reference() -> None:
    record = SourceRecord(
        row_id="0",
        data={"question": "What is 20 / 4?", "answer": "Reasoning #### 5"},
        digest="sha256:row",
    )
    sample = Gsm8kAdapter().prepare(record)
    assert sample.input == {"question": "What is 20 / 4?"}
    assert sample.reference == "5"
    assert extract_final_answer("work\n#### 5.0") == "5"
```

- [ ] **Step 3: Write failing SWE-bench and unavailable-harness tests**

```python
# tests/unit/benchmarks/test_swebench.py
import pytest

from agentic_evalkit.benchmarks.harness import HarnessRequest, UnavailableHarnessExecutor
from agentic_evalkit.benchmarks.swebench import SweBenchVerifiedAdapter
from agentic_evalkit.models import SourceRecord


def _harness_request() -> HarnessRequest:
    return HarnessRequest(
        benchmark="swebench-verified@1",
        sample_id="org__repo-1",
        prediction={
            "instance_id": "org__repo-1",
            "model_name_or_path": "agentic-evalkit-target",
            "model_patch": "diff --git a/x b/x",
        },
        source={"dataset_revision": "abc"},
        environment={},
        timeout_seconds=60,
        resource_limits={"cpus": 1, "memory_mb": 1024},
    )


def test_exports_official_prediction_shape() -> None:
    row = SourceRecord(
        row_id="0",
        digest="sha256:row",
        data={
            "instance_id": "org__repo-1",
            "repo": "org/repo",
            "base_commit": "abc",
            "problem_statement": "Fix it",
            "test_patch": "diff --git a/test.py b/test.py",
            "FAIL_TO_PASS": '["test_x"]',
            "PASS_TO_PASS": '["test_y"]',
        },
    )
    sample = SweBenchVerifiedAdapter().prepare(row)
    prediction = SweBenchVerifiedAdapter().export_prediction(sample, "diff --git a/x b/x")
    assert prediction == {
        "instance_id": "org__repo-1",
        "model_name_or_path": "agentic-evalkit-target",
        "model_patch": "diff --git a/x b/x",
    }


@pytest.mark.asyncio
async def test_missing_harness_is_unavailable_not_failed() -> None:
    result = await UnavailableHarnessExecutor("install agentic-evalkit[swebench]").execute(
        _harness_request()
    )
    assert result.status == "unavailable"
    assert "agentic-evalkit[swebench]" in result.message
```

- [ ] **Step 4: Run tests and confirm adapters are absent**

Run: `uv run pytest tests/unit/benchmarks tests/contract/test_harness.py -v`

Expected: FAIL with import errors for benchmark modules.

- [ ] **Step 5: Implement adapter and harness protocols**

Define `BenchmarkAdapter` with `api_version = "1"`, `name`, `prepare`, `validate_oracle`, and `aggregate_metadata`. Define frozen `HarnessRequest` with benchmark, sample ID, prediction, source/environment metadata, timeout, and resource limits. Define `HarnessResult` with status, resolved flag, message, evidence, logs, image digests, and typed error. Define async `HarnessExecutor.execute()`, deterministic `UnavailableHarnessExecutor`, and a `FakeHarnessExecutor` that returns configured resolved/unresolved/error results only from tests.

- [ ] **Step 6: Implement verified adapters**

`Gsm8kAdapter` validates `question` and `answer`, extracts the text after the final `####`, normalizes integer-equivalent decimals and comma separators, and emits `EvalSample` with `normalized-exact@1`.

`SweBenchVerifiedAdapter` validates all required fields, parses fail/pass lists from JSON strings or arrays, preserves issue/repository/base/test metadata, never checks out code, and exports only the official prediction keys. Its oracle validation checks row completeness and prediction identity, not patch correctness.

- [ ] **Step 7: Add contract serialization and discrimination tests**

Round-trip `HarnessRequest` and `HarnessResult` JSON. Use `FakeHarnessExecutor` to assert unavailable, infrastructure error, resolved false, and resolved true remain distinct. Assert a generic grade cannot be converted to `resolved=True` without a harness result.

- [ ] **Step 8: Run and commit**

```powershell
uv run pytest tests/unit/benchmarks tests/contract/test_harness.py -v
uv run mypy
git add docs/adr/0005-benchmark-adapters-and-harnesses.md src/agentic_evalkit/benchmarks tests/unit/benchmarks tests/contract/test_harness.py
git commit -m "feat: add benchmark and harness contracts"
```

### Task 9: Host-neutral execution targets

**Files:**
- Create: `docs/adr/0006-execution-target-boundary.md`
- Create: `src/agentic_evalkit/targets/base.py`
- Create: `src/agentic_evalkit/targets/callable.py`
- Create: `src/agentic_evalkit/targets/subprocess.py`
- Create: `src/agentic_evalkit/targets/http.py`
- Create: `src/agentic_evalkit/targets/__init__.py`
- Create: `tests/fixtures/targets/echo_target.py`
- Create: `tests/contract/test_targets.py`
- Create: `tests/integration/test_subprocess_target.py`
- Create: `tests/integration/test_http_target.py`

- [ ] **Step 1: Accept ADR-0006**

Record `ExecutionTarget` as the only system-under-test boundary, with callable, JSONL subprocess, and HTTP adapters. Record that target results normalize before grading; ARP and EK types cannot appear in public models; ARP can be evaluated through existing public surfaces without repository changes.

- [ ] **Step 2: Write failing callable target contract tests**

```python
# tests/contract/test_targets.py
import pytest

from agentic_evalkit.models import EvalSample, ExecutionStatus
from agentic_evalkit.targets import CallableTarget


@pytest.mark.asyncio
async def test_callable_target_normalizes_output_and_timeout() -> None:
    sample = EvalSample(
        sample_id="s1",
        input={"question": "ping"},
        source_digest="sha256:s1",
        adapter="identity@1",
    )
    target = CallableTarget(lambda value: {"answer": value["question"]}, name="echo")
    result = await target.execute(sample, attempt=1, timeout_seconds=1.0)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.output == {"answer": "ping"}
    assert result.target_fingerprint.startswith("callable:echo:")
```

- [ ] **Step 3: Write subprocess and HTTP integration tests**

Create `echo_target.py` to read one JSON object per line from standard input and emit one JSON object containing `sample_id`, `output`, and `metadata`. Test that `SubprocessTarget` sends one request, enforces a one-second timeout, caps standard error and output bytes, and reports malformed JSON as `ExecutionStatus.ERROR`.

```python
# tests/fixtures/targets/echo_target.py
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    response = {
        "schema_version": "1",
        "sample_id": request["sample_id"],
        "output": request["input"],
        "metadata": {"fixture": "echo"},
    }
    print(json.dumps(response, separators=(",", ":")), flush=True)
```

Use `httpx.MockTransport` to test that `HttpTarget` POSTs schema version, sample ID, input, attempt, and trace ID; maps a valid response; redacts authorization headers from evidence; maps 429 to a retryable target error; and maps deadline expiry to `TIMEOUT`.

- [ ] **Step 4: Run tests and confirm targets are absent**

Run: `uv run pytest tests/contract/test_targets.py tests/integration/test_subprocess_target.py tests/integration/test_http_target.py -v`

Expected: FAIL with import errors for `agentic_evalkit.targets`.

- [ ] **Step 5: Implement the target protocol and callable adapter**

Define async `ExecutionTarget.execute(sample, *, attempt, timeout_seconds)`. `CallableTarget` accepts sync or async functions, invokes sync functions with `asyncio.to_thread`, wraps both with `asyncio.timeout`, hashes module/qualified-name/name into a stable fingerprint, and converts exceptions into typed `NormalizedExecutionResult` errors without leaking local variables.

- [ ] **Step 6: Implement bounded subprocess JSONL**

`SubprocessTarget` uses `asyncio.create_subprocess_exec` with an argument tuple and no shell. Send one compact UTF-8 JSON line and close standard input. Read standard output with `StreamReader.readline()` so chunks are reassembled into complete lines on Windows, strip both `\r` and `\n`, enforce the configured byte bound, and parse exactly one JSON object with matching sample ID. Drain bounded standard error concurrently. Kill and await the process on timeout. Record command executable name and configured protocol version, but not environment secret values. Add a fixture that emits CRLF and a response split across writes; it must parse identically on Windows and Linux.

- [ ] **Step 7: Implement HTTP execution**

`HttpTarget` receives an injected `httpx.AsyncClient`, URL, nonsecret target name, header provider callback, and retry policy. Retry only connection failures, 429, and 502/503/504 with bounded exponential backoff and server `Retry-After`; never retry validation errors or other 4xx responses. Require matching sample ID and normalized output. Store only redacted request/response metadata.

- [ ] **Step 8: Run and commit**

```powershell
uv run pytest tests/contract/test_targets.py tests/integration/test_subprocess_target.py tests/integration/test_http_target.py -v
uv run mypy
git add docs/adr/0006-execution-target-boundary.md src/agentic_evalkit/targets tests/contract/test_targets.py tests/integration tests/fixtures/targets
git commit -m "feat: add host-neutral execution targets"
```

### Task 10: Objective graders first; calibrated judges after the checkpoint

**Files:**
- Create: `docs/adr/0007-objective-first-grading.md`
- Create: `src/agentic_evalkit/graders/base.py`
- Create: `src/agentic_evalkit/graders/exact.py`
- Create: `src/agentic_evalkit/graders/composite.py`
- Create: `src/agentic_evalkit/graders/rubric.py`
- Create: `src/agentic_evalkit/graders/judge.py`
- Create: `src/agentic_evalkit/graders/__init__.py`
- Create: `tests/unit/graders/test_exact.py`
- Create: `tests/unit/graders/test_composite.py`
- Create: `tests/unit/graders/test_judge.py`

- [ ] **Step 1: Accept ADR-0007**

Record the objective-first evidence order, atomic rubric criteria, noncompensable hard gates, provider-neutral judge protocol, held-out human calibration requirements, expiry, abstention, and the rule that uncalibrated judges cannot gate releases.

- [ ] **Step 2: Write failing exact and composite tests**

```python
# tests/unit/graders/test_composite.py
import pytest

from agentic_evalkit.graders.composite import CompositeGrader, WeightedGrader
from agentic_evalkit.models import GradeStatus


@pytest.mark.asyncio
async def test_failed_hard_gate_cannot_be_averaged_away() -> None:
    grader = CompositeGrader(
        name="quality@1",
        graders=(
            WeightedGrader(_StaticGrader(GradeStatus.FAIL, 0.0), weight=0.2, hard_gate=True),
            WeightedGrader(_StaticGrader(GradeStatus.PASS, 1.0), weight=0.8, hard_gate=False),
        ),
    )
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.FAIL
    assert result.hard_gate is True
    assert result.score == pytest.approx(0.8)
```

Define `_StaticGrader`, `_sample`, and `_execution` in the test module using the Task 2 public models.

- [ ] **Step 3: Run objective tests and confirm grader modules are absent**

Run: `uv run pytest tests/unit/graders/test_exact.py tests/unit/graders/test_composite.py -v`

Expected: FAIL with import errors for objective grader modules.

- [ ] **Step 4: Implement objective and composite graders**

Define async `Grader.grade(sample, execution)`. Implement:

- `ExactMatchGrader` with Unicode normalization, optional case folding, whitespace normalization, numeric canonicalization, and an injected extractor;
- `SchemaGrader` that validates structured output against a supplied Pydantic `TypeAdapter`;
- `CompositeGrader` that runs components, preserves every child result, calculates the weighted score over available numeric results, fails when any hard gate fails, and returns error/unavailable rather than treating missing graders as zero.

Add immutable `RubricCriterion` and `Rubric` models. Every criterion requires a stable ID, binary or bounded scale, evidence requirement, weight, and hard-gate flag. Reject duplicate IDs, negative weights, broad criteria without evidence requirements, and rubrics whose numeric weights sum to zero. Use `extract_final_answer` from the GSM8K adapter to configure its normalized exact grader.

- [ ] **Step 5: Verify and commit the Slice 4a objective graders**

```powershell
uv run pytest tests/unit/graders/test_exact.py tests/unit/graders/test_composite.py -v
uv run mypy
git add docs/adr/0007-objective-first-grading.md src/agentic_evalkit/graders tests/unit/graders/test_exact.py tests/unit/graders/test_composite.py
git commit -m "feat: add objective-first grading"
```

Expected: objective and hard-gate tests PASS. Do not implement model judges before the runnable-v0.1 checkpoint.

- [ ] **Step 6 (Slice 4b): Write failing calibration enforcement tests**

```python
# tests/unit/graders/test_judge.py
from datetime import UTC, datetime, timedelta

import pytest

from agentic_evalkit.graders.judge import CalibrationArtifact, JudgeGrader
from agentic_evalkit.models import GradeStatus


@pytest.mark.asyncio
async def test_expired_calibration_cannot_gate() -> None:
    calibration = CalibrationArtifact(
        calibration_id="cal-1",
        judge_fingerprint="judge:model:prompt",
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
        true_positive=40,
        true_negative=40,
        false_positive=5,
        false_negative=5,
        threshold=0.7,
    )
    grader = JudgeGrader(_FakeJudge(0.9), calibration=calibration, gate=True)
    result = await grader.grade(_sample(), _execution())
    assert result.status is GradeStatus.UNAVAILABLE
    assert "expired" in result.evidence["reason"]
```

- [ ] **Step 7 (Slice 4b): Implement calibrated judge contracts**

Define `JudgeClient` with a provider-neutral async `judge(request) -> JudgeResponse`. Add immutable `JudgeRequest`, `JudgeResponse`, and `CalibrationArtifact`. Calculate TPR/TNR from confusion counts. `JudgeGrader` must verify fingerprint equality, nonexpired calibration, minimum 30 positive and 30 negative held-out labels, TPR and TNR each at least the manifest threshold, and structured response validity. It reports parse errors and abstentions explicitly and retries malformed responses at most twice.

- [ ] **Step 8 (Slice 4b): Add position-bias and fingerprint tests**

Test mismatched model/prompt fingerprints, insufficient calibration sample counts, reversed answer ordering, malformed structured output, and an explicit judge abstention. Assert none can produce a release-gating pass.

- [ ] **Step 9 (Slice 4b): Run and commit calibrated judges**

```powershell
uv run pytest tests/unit/graders/test_judge.py -v
uv run mypy
git add src/agentic_evalkit/graders/judge.py src/agentic_evalkit/graders/__init__.py tests/unit/graders/test_judge.py
git commit -m "feat: add calibrated model judges"
```

### Task 11: Artifact store and evaluation runner

**Files:**
- Create: `src/agentic_evalkit/artifacts.py`
- Create: `src/agentic_evalkit/events.py`
- Create: `src/agentic_evalkit/runner.py`
- Create: `tests/unit/test_artifacts.py`
- Create: `tests/integration/test_runner.py`

- [ ] **Step 1: Write failing artifact integrity tests**

```python
# tests/unit/test_artifacts.py
from pathlib import Path

from agentic_evalkit.artifacts import ArtifactStore


def test_artifacts_are_content_addressed_and_deduplicated(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    first = store.put_bytes(b"same", media_type="text/plain")
    second = store.put_bytes(b"same", media_type="text/plain")
    assert first == second
    assert store.read(first) == b"same"
    assert first.digest == "sha256:0967115f2813a3541eaef77de9d9d5773f1c0c04314b0bbfe4ff3b3b1c55b5d5"
```

- [ ] **Step 2: Write a failing end-to-end runner test**

```python
# tests/integration/test_runner.py
import pytest

from agentic_evalkit.runner import EvalRunner


@pytest.mark.asyncio
async def test_runner_preserves_sample_failure_and_infrastructure_error() -> None:
    runner = EvalRunner(
        catalog=_catalog_with_two_records(),
        adapters={"identity@1": _IdentityAdapter()},
        targets={"fake": _SequencedTarget.success_then_error()},
        graders={"exact@1": _ExactFixtureGrader()},
        artifact_store=_artifact_store(),
    )
    result = await runner.run(_manifest())
    assert result.summary.total == 2
    assert result.summary.failed == 0
    assert result.summary.errors == 1
    assert result.samples[0].grade.status == "pass"
    assert result.samples[1].execution.status == "error"
    assert result.samples[1].grade is None
```

Define deterministic fakes in the test; do not use network or model calls.

- [ ] **Step 3: Run tests and confirm runner modules are absent**

Run: `uv run pytest tests/unit/test_artifacts.py tests/integration/test_runner.py -v`

Expected: FAIL with import errors for artifacts and runner.

- [ ] **Step 4: Implement content-addressed artifacts and progress events**

`ArtifactStore` writes immutable blobs by SHA-256 with a sidecar containing media type, byte count, creation time, and redaction status. Writes are atomic and bounded by configured maximum bytes. Define frozen events for run started, dataset resolved, sample started, execution completed, grade completed, sample completed, run completed, and run failed. Events carry run/sample IDs and timestamps but not raw secrets.

- [ ] **Step 5: Implement the runner pipeline**

`EvalRunner.run(manifest, event_sink)` must:

1. validate component names and capabilities;
2. resolve the dataset once and pin the result in provenance;
3. iterate the requested selection;
4. prepare each record with the named adapter;
5. execute attempts under a concurrency semaphore;
6. grade only completed executions;
7. preserve errors, timeouts, cancellation, abstention, and unavailable separately;
8. store large outputs/logs as artifacts and retain references;
9. emit ordered progress events;
10. return samples in deterministic sample/attempt order;
11. cancel outstanding tasks when the caller cancels the run;
12. never mutate the supplied manifest.

Use `asyncio.TaskGroup` and inject clock/ID factories in tests.

- [ ] **Step 6: Add determinism, concurrency, and cancellation tests**

Run the same fixed manifest twice and assert equivalent result JSON after excluding run IDs/timestamps. Use a counting target to prove concurrency never exceeds the manifest limit. Cancel a run with pending targets and assert pending results become `cancelled` with no orphan subprocesses.

- [ ] **Step 7: Run and commit**

```powershell
uv run pytest tests/unit/test_artifacts.py tests/integration/test_runner.py -v
uv run mypy
git add src/agentic_evalkit/artifacts.py src/agentic_evalkit/events.py src/agentic_evalkit/runner.py tests/unit/test_artifacts.py tests/integration/test_runner.py
git commit -m "feat: add reproducible evaluation runner"
```

### Task 12 (Slice 4b): Advanced statistics, reliability, and run compatibility

Do not execute this task until the runnable-v0.1 checkpoint after Task 14 Part A selects the full-v1 path.

**Files:**
- Create: `docs/adr/0008-statistical-comparability.md`
- Create: `src/agentic_evalkit/stats/aggregate.py`
- Create: `src/agentic_evalkit/stats/reliability.py`
- Create: `src/agentic_evalkit/stats/compare.py`
- Create: `src/agentic_evalkit/stats/__init__.py`
- Create: `tests/unit/stats/test_aggregate.py`
- Create: `tests/unit/stats/test_reliability.py`
- Create: `tests/unit/stats/test_compare.py`

- [ ] **Step 1: Accept ADR-0008**

Record sample-level retention, Wilson 95% intervals for binary rates, deterministic seeded bootstrap for paired deltas, exact attempt accounting, `pass@k`, all-attempt consistency at `k`, separated operational outcomes, and rejection of comparisons with incompatible provenance.

- [ ] **Step 2: Write failing known-value tests**

```python
# tests/unit/stats/test_reliability.py
import pytest

from agentic_evalkit.stats.reliability import consistency_at_k, pass_at_k


def test_pass_at_k_known_values() -> None:
    assert pass_at_k(total_attempts=4, successful_attempts=1, k=1) == pytest.approx(0.25)
    assert pass_at_k(total_attempts=4, successful_attempts=1, k=4) == pytest.approx(1.0)


def test_consistency_at_k_requires_every_attempt_to_pass() -> None:
    assert consistency_at_k(success_probability=0.8, k=3) == pytest.approx(0.512)
```

```python
# tests/unit/stats/test_compare.py
import pytest

from agentic_evalkit.errors import IncompatibleRuns
from agentic_evalkit.stats.compare import compare_runs


def test_rejects_different_dataset_revisions() -> None:
    left = _run(dataset_revision="abc")
    right = _run(dataset_revision="def")
    with pytest.raises(IncompatibleRuns, match="dataset revision"):
        compare_runs(left, right, bootstrap_samples=1000, seed=7)
```

Define `_run()` in the module as a complete two-sample `EvalRunResult` fixture whose only parameterized provenance field is `dataset_revision`; keep adapter, grader, target, sampling, and attempt policy identical.

- [ ] **Step 3: Run tests and confirm stats modules are absent**

Run: `uv run pytest tests/unit/stats -v`

Expected: FAIL with import errors for `agentic_evalkit.stats`.

- [ ] **Step 4: Implement aggregation and intervals**

Implement Wilson bounds using `statistics.NormalDist().inv_cdf(0.975)`. `aggregate_run()` counts pass/fail/partial/error/timeout/cancelled/abstain/unavailable separately, reports exact numerator/denominator, score mean only over defined numeric scores, and latency/token/cost count/mean/p50/p95 where data exists. Empty denominators return `None` bounds, never zero confidence. Detailed subgroup manifest syntax and minimum-sample warnings are deferred past v1.

- [ ] **Step 5: Implement repeated-trial metrics**

`pass_at_k(total_attempts, successful_attempts, k)` computes `1 - C(n-c,k)/C(n,k)` with validation `0 <= c <= n` and `1 <= k <= n`; use `math.lgamma` in log space for large `n` rather than constructing huge integers. `consistency_at_k(success_probability, k)` uses `p**k` with `0 <= p <= 1` and is described as “all k attempts pass,” not as a second `pass@k` metric. Group attempts by sample ID and report attempt coverage so missing attempts cannot inflate results.

- [ ] **Step 6: Implement compatibility and paired bootstrap**

Compare dataset ID/revision/config/split, adapter, harness, grader, target policy, sampling temperature/seed policy, and attempt count. Return all mismatches in `IncompatibleRuns`. For compatible runs, pair by sample and attempt ID, calculate the success-rate delta, and bootstrap paired differences with a local `random.Random(seed)` instance. Default to 1,000 bootstrap samples; accept an API/CLI override from 100 through 10,000. Return estimate, 2.5/97.5 percentiles, paired count, sample count, and seed.

- [ ] **Step 7: Run and commit**

```powershell
uv run pytest tests/unit/stats -v
uv run mypy
git add docs/adr/0008-statistical-comparability.md src/agentic_evalkit/stats tests/unit/stats
git commit -m "feat: add statistically valid aggregation"
```

### Task 13: Canonical JSON first; rich reports after the checkpoint

**Files:**
- Create: `src/agentic_evalkit/reporters/base.py`
- Create: `src/agentic_evalkit/reporters/json.py`
- Create: `src/agentic_evalkit/reporters/jsonl.py`
- Create: `src/agentic_evalkit/reporters/markdown.py`
- Create: `src/agentic_evalkit/reporters/html.py`
- Create: `src/agentic_evalkit/reporters/__init__.py`
- Create: `src/agentic_evalkit/reporters/templates/report.html.j2`
- Create: `tests/unit/reporters/test_json.py`
- Create: `tests/unit/reporters/test_markdown.py`
- Create: `tests/unit/reporters/test_html.py`

- [ ] **Step 1: Write failing report evidence tests**

```python
# tests/unit/reporters/test_json.py
import json

from agentic_evalkit.reporters import JsonReporter


def test_json_and_jsonl_retain_sample_evidence(tmp_path) -> None:
    run = _run_with_pass_error_timeout_and_provenance()
    json_path = JsonReporter().write(run, tmp_path / "run.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["dataset_revision"] == "abc"
    assert {item["execution"]["status"] for item in payload["samples"]} == {
        "completed",
        "error",
        "timeout",
    }
```

Define `_run_with_pass_error_timeout_and_provenance()` in the test package as a frozen three-sample `EvalRunResult` fixture shared by all reporter tests.

- [ ] **Step 2: Run tests and confirm reporters are absent**

Run: `uv run pytest tests/unit/reporters/test_json.py -v`

Expected: FAIL with import errors for `agentic_evalkit.reporters`.

- [ ] **Step 3: Implement the reporter protocol and canonical JSON**

Define `Reporter.write(run, destination) -> Path`. `JsonReporter` writes the complete versioned `EvalRunResult` with deterministic indentation and sorted keys. Use a temporary file plus atomic replacement and UTF-8 newlines.

- [ ] **Step 4: Verify and commit the Slice 4a JSON reporter**

```powershell
uv run pytest tests/unit/reporters/test_json.py -v
uv run mypy
git add src/agentic_evalkit/reporters/base.py src/agentic_evalkit/reporters/json.py src/agentic_evalkit/reporters/__init__.py tests/unit/reporters/test_json.py
git commit -m "feat: add canonical JSON reports"
```

- [ ] **Step 5 (Slice 4b): Implement JSONL and human-readable formats**

`JsonlReporter` writes one header record with manifest/provenance/summary, one sample record per result, and one trailer record with aggregate statistics.

`MarkdownReporter` renders identity, provenance, compatibility, outcome counts, confidence intervals, reliability, resource distributions, and a sample table with evidence/artifact references.

`HtmlReporter` uses Jinja2 autoescape, embeds CSS and escaped JSON in one self-contained file, provides filter buttons for outcome categories, and includes a no-JavaScript summary. It must not load remote scripts, fonts, analytics, or styles.

- [ ] **Step 6 (Slice 4b): Add redaction and deterministic-output tests**

Provide a redaction policy that removes configured evidence keys and replaces matching secret strings with `[REDACTED]` before any reporter sees the model. Render the same frozen run twice and assert byte-identical output after injecting a fixed generated-at timestamp.

Add tests that Markdown contains exact numerator/denominator and compatibility details, and HTML escapes `<script>` in model output while containing embedded JSON data and no external asset URLs.

- [ ] **Step 7 (Slice 4b): Run and commit rich reports**

```powershell
uv run pytest tests/unit/reporters/test_markdown.py tests/unit/reporters/test_html.py -v
uv run mypy
git add src/agentic_evalkit/reporters/jsonl.py src/agentic_evalkit/reporters/markdown.py src/agentic_evalkit/reporters/html.py src/agentic_evalkit/reporters/templates tests/unit/reporters/test_markdown.py tests/unit/reporters/test_html.py
git commit -m "feat: add rich portable reports"
```

### Task 14: Runnable CLI first; comparison commands after the checkpoint

**Files:**
- Create: `src/agentic_evalkit/manifest.py`
- Modify: `src/agentic_evalkit/cli/__init__.py`
- Modify: `src/agentic_evalkit/cli/app.py`
- Create: `src/agentic_evalkit/cli/datasets.py`
- Create: `src/agentic_evalkit/cli/runs.py`
- Create: `src/agentic_evalkit/cli/doctor.py`
- Create: `src/agentic_evalkit/examples/__init__.py`
- Create: `src/agentic_evalkit/examples/zero_target.py`
- Create: `tests/fixtures/manifests/gsm8k.yaml`
- Create: `tests/unit/test_manifest.py`
- Create: `tests/integration/test_cli.py`
- Create: `docs/release/v0.1-checkpoint.md`
- Modify: `src/agentic_evalkit/__init__.py`

- [ ] **Step 1: Write failing manifest and CLI tests**

```python
# tests/integration/test_cli.py
from typer.testing import CliRunner

from agentic_evalkit.cli import app

runner = CliRunner()


def test_curated_and_init_work_without_manual_import(tmp_path) -> None:
    listed = runner.invoke(app, ["datasets", "curated", "--format", "json"])
    assert listed.exit_code == 0
    assert "swe-bench-verified" in listed.stdout
    destination = tmp_path / "eval.yaml"
    created = runner.invoke(app, ["init", "--preset", "gsm8k", "--output", str(destination)])
    assert created.exit_code == 0
    assert destination.exists()
    validated = runner.invoke(app, ["validate", str(destination)])
    assert validated.exit_code == 0
    assert "valid" in validated.stdout.lower()


def test_provider_failure_has_nonzero_exit_and_error_code() -> None:
    result = runner.invoke(app, ["datasets", "inspect", "hf:missing/not-found"])
    assert result.exit_code == 4
    assert "dataset_not_found" in result.stdout
```

- [ ] **Step 2: Run tests and confirm CLI modules are absent**

Run: `uv run pytest tests/unit/test_manifest.py tests/integration/test_cli.py -v`

Expected: FAIL with import errors for manifest and CLI modules.

- [ ] **Step 3: Implement safe manifest loading**

`load_manifest(path)` uses `yaml.safe_load`, requires one mapping, resolves no Python tags, validates with `EvalRunManifest`, and reports field paths in `ManifestValidationError`. `dump_manifest()` emits stable YAML with schema version and explicit dataset config/split, adapter, target, grader, attempts, timeout, concurrency, and artifact policy. Environment interpolation is forbidden in manifests; secret values enter only through target/provider hooks.

- [ ] **Step 4: Build the Typer application and exit-code policy**

Create one root `Typer(no_args_is_help=True)` application with commands and subcommands:

- `doctor`;
- `datasets curated/search/inspect/preview/pull`;
- `init` and `validate`;
- `run`.

Use exit codes: 0 success, 2 invalid input/manifest, 3 missing capability, 4 provider/target unavailable, 5 evaluation completed with infrastructure errors, and 130 cancelled. Catch only `AgenticEvalkitError` at the command boundary, print its stable code and actionable message, and allow unexpected exceptions to produce a traceback under `--debug`.

- [ ] **Step 5: Implement dataset commands**

`curated` works offline. `search`, `inspect`, and `preview` use the same catalog services as Python callers and display resolved revisions/configs/splits. `pull` records an immutable cache entry and manifest; it never means “latest” after resolution. Support `--format table|json` and `--offline` consistently.

- [ ] **Step 6: Implement the objective-only run command**

`run` loads a manifest, performs a preflight summary, requires `--yes` only when the command is noninteractive, streams Rich progress from runner events, stores canonical run JSON under the selected output directory, and prints separated outcome counts plus the JSON report path.

The initial CLI target configuration supports:

- a Python import string for `CallableTarget`;
- an argv list for `SubprocessTarget`;
- a URL plus named credential hook for `HttpTarget`.

- [ ] **Step 7: Implement doctor and end-to-end CLI fixtures**

`doctor` checks Python version, cache read/write, Hugging Face health, configured target health, optional capability availability, and judge calibration. Each check returns `ok`, `warning`, or `error` with remediation.

Create a packaged `agentic_evalkit.examples.zero_target` callable that always returns `"0"`; it is a transport/evaluation smoke target, not a benchmark baseline. `agentic-evalkit init --preset gsm8k` uses this demo target only when the developer does not provide a callable/subprocess/HTTP target. Run `agentic-evalkit run` through `CliRunner` and assert canonical JSON is created with completed objective grades, regardless of whether individual GSM8K samples pass.

- [ ] **Step 8: Run and commit**

```powershell
uv run pytest tests/unit/test_manifest.py tests/integration/test_cli.py -v
uv run agentic-evalkit --help
uv run agentic-evalkit datasets curated
uv run mypy
git add src/agentic_evalkit/manifest.py src/agentic_evalkit/cli src/agentic_evalkit/examples src/agentic_evalkit/__init__.py tests/fixtures/manifests tests/unit/test_manifest.py tests/integration/test_cli.py
git commit -m "feat: add runnable objective evaluation CLI"
```

- [ ] **Step 9: Execute and record the formal v0.1 checkpoint**

Build and install the wheel in a clean temporary environment, then run:

```powershell
agentic-evalkit doctor
agentic-evalkit init --preset gsm8k --output eval.yaml
agentic-evalkit run eval.yaml --limit 5 --yes
```

Verify that the commands require no importer code, manual dataset download, `datasets`, `pyarrow`, or Docker; that a canonical JSON report is produced; and that a simulated provider outage prints a stable code/remediation without a traceback. Record commands, artifacts, test results, known issues, and one explicit decision in `docs/release/v0.1-checkpoint.md`:

- `SHIP_V0_1`: complete Tasks 15-16 in objective-only mode, move Task 10 Steps 6-9, Task 12, Task 13 Steps 5-7, and Task 14 Steps 10-11 to the v0.2 plan; or
- `CONTINUE_FULL_V1`: execute those deferred parts, then complete Tasks 15-16.

Commit the checkpoint record before continuing.

```powershell
git add docs/release/v0.1-checkpoint.md
git commit -m "docs: record runnable v0.1 checkpoint"
```

- [ ] **Step 10 (Slice 4b): Implement compare and rich report commands**

Only for `CONTINUE_FULL_V1`, add `compare` and `report`. `compare` loads two canonical run files, uses Task 12 compatibility checks, and accepts `--bootstrap-samples` from 100 through 10,000 with a default of 1,000. `report` regenerates JSONL, Markdown, or self-contained HTML from canonical JSON.

- [ ] **Step 11 (Slice 4b): Verify and commit advanced CLI commands**

```powershell
uv run pytest tests/unit/stats tests/unit/reporters tests/integration/test_cli.py -v
uv run agentic-evalkit compare --help
uv run agentic-evalkit report --help
git add src/agentic_evalkit/cli tests/integration/test_cli.py
git commit -m "feat: add comparison and rich report commands"
```

### Task 15: Documentation, ADR consistency, clean-wheel, and release gates

**Files:**
- Create: `.github/workflows/live-provider.yml`
- Create: `.github/workflows/publish.yml`
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Create: `docs/guides/quickstart.md`
- Create: `docs/guides/providers.md`
- Create: `docs/guides/graders.md`
- Create: `docs/guides/targets.md`
- Create: `docs/guides/swebench.md`
- Create: `docs/guides/http-agent-example.md`
- Create: `examples/http_agent/README.md`
- Create: `tests/contract/test_dependency_boundary.py`
- Create: `tests/contract/test_adrs.py`
- Create: `tests/contract/test_public_docs.py`
- Create: `tests/integration/test_clean_wheel.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `CONTRIBUTING.md`
- Modify: `SECURITY.md`

- [ ] **Step 1: Write failing dependency-boundary and ADR tests**

```python
# tests/contract/test_dependency_boundary.py
import ast
from pathlib import Path

FORBIDDEN_ROOTS = {"agentic_v2", "tools", "executionkit"}


def test_package_does_not_import_arp_tools_or_executionkit() -> None:
    violations: list[str] = []
    for path in Path("src/agentic_evalkit").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in FORBIDDEN_ROOTS:
                    violations.append(f"{path}:{node.lineno}:{name}")
    assert violations == []
```

`test_adrs.py` must assert ADR files 0001 through 0009 exist, contain `Status: Accepted`, include Context/Decision/Alternatives/Consequences/Validation/Supersession headings, and do not contradict the dependency and baseline Hugging Face decisions.

Add a public-document hygiene test that scans README, guides, examples, CLI help snapshots, and error-message fixtures for internal codenames `agentic_v2`, `agentic-v2-eval`, `tools.agents`, and `executionkit`. The dependency-boundary test may contain forbidden names as test data; public user-facing artifacts may not.

- [ ] **Step 2: Run tests and verify documentation gaps**

Run: `uv run pytest tests/contract/test_dependency_boundary.py tests/contract/test_adrs.py -v`

Expected: dependency test PASS; ADR shape test FAIL until every ADR receives the required sections.

- [ ] **Step 3: Normalize all ADRs and build strict documentation**

Update each ADR to the required six-section template and add links from `docs/index.md`. Configure `mkdocs.yml` with strict mode, Material theme, repository metadata, and navigation for design, plan, ADRs, and guides. Do not expose internal machine paths in published pages.

Write guides with executable commands:

- quickstart: install, `doctor`, curated list, GSM8K init/run, and the canonical JSON produced by `run`; add the standalone rich `report` command only for `CONTINUE_FULL_V1`;
- providers: local formats, Hugging Face auth, cache/offline, plugin entry points;
- graders: objective order, hard gates, calibration, abstention/error semantics;
- targets: callable, subprocess JSONL, HTTP mappings, credential hooks;
- SWE-bench: initial preview/prediction workflow, typed unavailable result, and later Docker capability boundary;
- providers: document the `parquet` extra as the explicit fallback when Dataset Viewer cannot serve a dataset;
- HTTP agent example: evaluate a real tool-using agent endpoint with request/response mapping, authentication hook, timeout, objective schema grader, and canonical report. The example must not import ARP or EK and need not be an automated release test.

Update README and `docs/index.md` with the approved positioning statement: `agentic-evalkit` separates datasets, grading, and reporting from the system under test through callable/subprocess/HTTP targets, and objective checks gate before judges. Add a coexistence note: legacy evaluation code may remain in host repositories, but this package neither imports nor migrates it.

Create `.github/workflows/live-provider.yml` with `workflow_dispatch` and a weekly schedule. It installs the locked environment and runs only `uv run pytest tests/live/test_huggingface_live.py -m live -v`. The provider client applies Task 6's bounded backoff. Provider outages fail visibly with classified diagnostics; the workflow does not retry indefinitely or convert failures to success.

Create `.github/workflows/publish.yml` using PyPI trusted publishing with GitHub OIDC (`id-token: write`), an environment named `pypi`, artifact build/verification before upload, and a release-only trigger. Store no PyPI API token. Publishing remains inert until the repository/environment is configured and a GitHub release is intentionally created.

- [ ] **Step 4: Implement the clean-wheel integration test**

`test_clean_wheel.py` must build the wheel with `python -m build --wheel`, create a temporary virtual environment outside the repository, install only the wheel, set the working directory to that temporary directory, and run:

```text
python -c "import agentic_evalkit; print(agentic_evalkit.__version__)"
agentic-evalkit --help
agentic-evalkit datasets curated --format json
```

Assert all commands exit 0 and output contains `gsm8k` and `swe-bench-verified`. In the isolated verification script, call `importlib.util.find_spec()` through a helper that catches `ModuleNotFoundError` for missing parent packages; assert the helper returns `None` for `agentic_v2`, `tools.agents`, and `executionkit`.

- [ ] **Step 5: Run the complete offline verification matrix**

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -m "not live" --cov=agentic_evalkit --cov-branch --cov-report=term-missing
uv run mkdocs build --strict
uv build
```

Expected: every command exits 0, pytest reports at least 80% branch-aware coverage, strict docs contain no warnings, and both sdist and wheel are created. Record the achieved coverage; raising the gate to 90% is a later evidence-based decision, not an initial-release requirement.

- [ ] **Step 6: Run live Hugging Face and clean-wheel gates**

```powershell
uv run pytest tests/live/test_huggingface_live.py -m live -v
uv run pytest tests/integration/test_clean_wheel.py -v
```

Expected when Hugging Face is available: both verified presets resolve and preview; the isolated wheel imports and CLI commands pass with no host repositories on `sys.path`. If bounded retries still end in a classified transient provider outage, preserve the failed live evidence, confirm the latest scheduled/on-demand successful evidence, record a release known issue, and continue only if every offline provider contract and clean-wheel test passes.

- [ ] **Step 7: Execute the documented quickstart exactly**

Create a temporary directory, follow `docs/guides/quickstart.md` without using repository-relative paths, run a five-sample GSM8K evaluation against the packaged demo target, and verify canonical JSON output. For `CONTINUE_FULL_V1`, additionally verify self-contained HTML. Record the command transcript in the release verification section of the quickstart.

- [ ] **Step 8: Commit release documentation and gates**

```powershell
git add README.md CHANGELOG.md CONTRIBUTING.md SECURITY.md .github/workflows/live-provider.yml .github/workflows/publish.yml examples mkdocs.yml docs tests/contract/test_dependency_boundary.py tests/contract/test_adrs.py tests/integration/test_clean_wheel.py
git commit -m "docs: complete initial release verification"
```

### Task 16: Initial release acceptance audit and follow-on boundary

**Files:**
- Create: `docs/release/initial-release-acceptance.md`
- Create: `docs/plans/README.md`

- [ ] **Step 1: Audit every design acceptance criterion**

Create `initial-release-acceptance.md` with a table containing all 17 criteria from the approved design, the exact test/command that proves each criterion, the resulting artifact or test name, and pass/fail status. For `SHIP_V0_1`, identify the approved Slice 4b criteria as deferred to v0.2 rather than falsely passing them. Do not mark a criterion passed from code inspection alone when the design requires live or clean-wheel evidence.

- [ ] **Step 2: Reconcile ADRs, code, and package metadata**

Verify package dependencies match ADR-0003/0009, cache identity matches ADR-0004, harness status semantics match ADR-0005, forbidden imports match ADR-0001/0006, and objective hard gates match ADR-0007. For `CONTINUE_FULL_V1`, also verify judge gates match ADR-0007 and report compatibility matches ADR-0008. For `SHIP_V0_1`, record those checks as deferred rather than passed. Record any implemented-scope mismatch as failed acceptance and fix it in the owning task before continuing.

- [ ] **Step 3: Document the separate SWE-bench implementation plan gate**

In `docs/plans/README.md`, state that the Docker executor requires a new plan based on the accepted harness contracts. That plan may begin only after the initial acceptance audit passes and must include pinned upstream `swebench` compatibility, Docker/image resource preflight, gold/invalid patch tests, cancellation, log capture, and no public contract changes. Also list deferred v0.2 work: run resumption, async-first ADR-0010, performance/eviction targets, subgroup syntax, framework observability, and any Slice 4b work deferred at the checkpoint.

- [ ] **Step 4: Re-run release evidence from a clean checkout state**

Run:

```powershell
git status --short
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -m "not live" --cov=agentic_evalkit --cov-branch --cov-report=term-missing
uv run pytest tests/live/test_huggingface_live.py -m live -v
uv run mkdocs build --strict
uv build
```

Expected before the final commit: `git status --short` lists only the acceptance and plan-index files; every offline quality, test, docs, and build command exits 0. The live-provider command either passes or ends in a classified outage that is documented with the latest prior successful workflow evidence and a release known issue.

- [ ] **Step 5: Commit the accepted initial-release evidence**

```powershell
git add docs/release/initial-release-acceptance.md docs/plans/README.md
git commit -m "docs: record initial release acceptance"
```

- [ ] **Step 6: Verify final repository state**

Run:

```powershell
git status --short --branch
git log --oneline --decorate -25
```

Expected: clean `main` working tree, focused milestone commits, and every governing ADR committed before its production implementation.

## Design coverage matrix

| Design requirement | Implemented and verified by |
|---|---|
| Standalone identity and forbidden dependencies | Tasks 1, 9, 15, 16 |
| Immutable typed contracts and provenance | Tasks 2, 11, 15 |
| Plugin model and typed errors | Task 3 |
| Content-addressed cache and offline behavior | Tasks 4, 7 |
| Local dataset formats | Task 5 |
| Baseline Hugging Face search/resolve/preview/metadata | Tasks 6, 7, 14, 15 |
| Curated GSM8K and SWE-bench presets | Tasks 7, 8, 14 |
| Benchmark adapter and optional harness separation | Task 8 |
| Callable, subprocess, and HTTP targets | Task 9 |
| Objective grading, atomic rubrics, composite gates | Task 10 Steps 1-5 |
| Calibrated judges | Task 10 Steps 6-9 after the checkpoint |
| Reproducible orchestration and artifacts | Task 11 |
| Confidence intervals, repeated trials, paired comparison | Task 12 after the checkpoint |
| Canonical JSON and deferred rich reports | Task 13 Parts 4a/4b |
| Runnable CLI and deferred compare/report commands | Task 14 Parts 4a/4b |
| Security, clean-wheel, live-provider, docs, and ADR gates | Tasks 15, 16 |
| Follow-on SWE-bench Docker boundary | Tasks 8, 15, 16 |

## Execution notes

- Do not modify Agentic Runtime Platform or ExecutionKit while executing this plan.
- Use a dedicated worktree or isolated clone before implementation begins.
- Apply test-driven development to each production change: failing test, observed failure, minimal implementation, passing test, refactor, focused commit.
- Never hide live-provider failures or weaken clean-wheel, dependency-boundary, calibration, or authoritative-harness gates to make CI green; use the classified transient-outage release policy exactly as documented.
- Preserve sample-level evidence; no aggregate metric may discard the underlying statuses or provenance.
- Generate the official SWE-bench Docker executor plan only after Task 16 passes.
