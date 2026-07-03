# Task 4: Immutable content-addressed cache

First read COMMON.md in this directory and follow every rule there. Your plan section: "Task 4: Immutable content-addressed cache".

YOUR FILES:
- docs/adr/0004-content-addressed-dataset-cache.md (Accepted; six sections; MUST include the Windows note: Path.replace() atomicity depends on the local filesystem — correctness relies on checksum validation and mandatory Windows CI cache tests, not an unconditional atomicity claim)
- src/agentic_evalkit/datasets/cache.py
- tests/unit/datasets/test_cache.py
- pyproject.toml — ALLOWED ONLY to add pytest-repeat to the dev dependency group (run `uv sync --all-groups` afterward); change nothing else in it.
Do NOT touch datasets/__init__.py, base.py, local.py, or huggingface.py (other agents own them).

KEY DECISIONS:
- CacheKey subclasses agentic_evalkit.models.FrozenModel. Fields: provider, dataset_id, revision, config (str|None), split (str|None), offset, limit, plus projection_digest/filter_digest/data_files_digest (str|None defaults) and record_type ("page" | "full", default "page"). digest() = "sha256:" + SHA-256 of UTF-8 canonical JSON (sorted keys, compact separators) of all fields.
- DatasetCache(root: Path). write(key, payload: bytes): write payload+manifest to temp files in the same directory, flush+fsync, then Path.replace() under a per-key threading.Lock (module-level lock registry). Manifest JSON records payload sha256 checksum, byte count, created_at, and the full key JSON.
- read(key): missing entry -> raise OfflineCacheMiss; manifest/key mismatch, byte-count mismatch, or checksum mismatch -> raise DatasetIntegrityError. payload_path(key) exposes the payload file path (the plan's corruption test overwrites it directly).
- Layout: root / key.digest()[:2] / key.digest() / {payload.bin, manifest.json} (or similar stable scheme — document in ADR).
- Step 5: ThreadPoolExecutor concurrent same-key writes -> exactly one valid final entry; two page keys with different offsets both readable; run `uv run pytest tests/unit/datasets/test_cache.py -x --count=5` after adding pytest-repeat.

The plan's test snippet must pass verbatim (CacheKey(provider=..., dataset_id=..., revision=..., config=..., split=..., offset=..., limit=...), model_copy variants change digest; corruption vs offline miss distinct).
