# Task 5: Dataset provider protocol and local files

First read COMMON.md in this directory and follow every rule there. Your plan section: "Task 5: Dataset provider protocol and local files".

IMPORTANT DEVIATION: the plan says "Create src/agentic_evalkit/datasets/base.py" — it ALREADY EXISTS (orchestrator-created) with the DatasetProvider protocol and ProviderHealth exactly per plan Step 3. Import from it; extend it only if strictly necessary and report any change.

YOUR FILES:
- src/agentic_evalkit/datasets/local.py
- src/agentic_evalkit/datasets/__init__.py — you are the OWNER of this file this wave: append exports for DatasetProvider, ProviderHealth, LocalDatasetProvider (keep the docstring). Do not export cache/huggingface/catalog names (other agents own those modules).
- tests/unit/datasets/test_local_provider.py
- tests/fixtures/datasets/items.jsonl, items.csv, items.yaml
Do NOT touch cache.py or huggingface.py.

KEY DECISIONS:
- Plan test snippets verbatim: resolve() returns ResolvedDataset with revision "sha256:<hex>" over raw file bytes; preview(resolved, offset=1, limit=1) -> SamplePage with total_rows=2 and records[0].data["id"]=="b"; iter_records yields row_id "0","1"; path outside allowed_roots raises ValueError matching "outside allowed roots" (plain ValueError, verbatim per test).
- LocalDatasetProvider(allowed_roots: tuple[Path, ...]); api_version = "1". Resolve: Path.resolve(), enforce roots via is_relative_to, reject directories and unsupported suffixes.
- Decoding per plan Step 4: JSON (list of objects, or object with "records" list), JSONL (one object per nonblank line), CSV via csv.DictReader, YAML via yaml.safe_load (list of objects). Every row must be dict[str, JsonValue]; zero-based string row IDs; SourceRecord.digest = "sha256:" + hash of canonical JSON of the row.
- Malformed JSONL or scalar YAML -> raise DatasetSchemaMismatch (from agentic_evalkit.errors), never empty results.
- search() returns an empty successful SearchPage; healthcheck() verifies all roots readable -> ProviderHealth.
- Step 5 parity: all three fixture formats hold the same two logical rows; assert identical canonical SourceRecord.data across formats and different source revisions.
