# Task 6: Hugging Face discovery and Dataset Viewer provider

First read COMMON.md in this directory and follow every rule there. Your plan section: "Task 6: Hugging Face discovery and Dataset Viewer provider" (read the hardened notes about best-effort endpoints and captured fixtures — they are binding).

YOUR FILES:
- src/agentic_evalkit/datasets/huggingface.py
- tests/fixtures/huggingface/*.json (dataset_info, is_valid, splits, rows, size, statistics, parquet)
- tests/unit/datasets/test_huggingface_provider.py
- tests/live/test_huggingface_live.py
Do NOT touch datasets/__init__.py (another agent owns it), base.py, cache.py, or local.py. Import DatasetProvider/ProviderHealth from agentic_evalkit.datasets.base and errors from agentic_evalkit.errors.

KEY DECISIONS:
- CAPTURE REAL FIXTURES FIRST: you have network access. Fetch and save real responses for openai/gsm8k (config main, split test) and princeton-nlp/SWE-bench_Verified (default/test) from https://datasets-server.huggingface.co endpoints /is-valid, /splits, /rows (offset=0&length=2), /size, /statistics, /parquet, plus Hub dataset_info JSON from https://huggingface.co/api/datasets/<id>. Save verbatim under tests/fixtures/huggingface/ (you may nest per-dataset subdirs; keep the seven plan-listed names as the gsm8k set or document your layout in the test module).
- resolve() order per plan Step 3. LOAD-BEARING: /is-valid, dataset_info (nonempty commit SHA), /splits. BEST-EFFORT: /size, /statistics, /parquet — on their failure record absence in ResolvedDataset metadata and continue. Ambiguous configs -> DatasetConfigRequired.
- Injected huggingface_hub.HfApi + httpx.AsyncClient; sync Hub calls via asyncio.to_thread. HuggingFaceDatasetProvider.create() returns an async context manager that owns its AsyncClient (the live test uses `async with HuggingFaceDatasetProvider.create() as provider:`).
- preview/iter_records per plan Step 4: /rows with URL-encoded params, page cap 100, SourceRecord from row_idx + canonical digest, num_rows_total honored; a partial page is never presented as a full dataset. The provider itself does NOT cache (catalog does).
- Error mapping: 401/403->DatasetAccessDenied, 404->DatasetNotFound (load-bearing endpoints only), 422 config->DatasetConfigRequired, 429->DatasetRateLimited (with retry metadata), transport/5xx->DatasetProviderUnavailable.
- Retries: bounded (max 3), only connection errors/429/502/503/504, honor Retry-After else jittered exponential backoff, caller-injected sleep function for deterministic tests. Tests must prove: 429-then-200 succeeds; repeated 429 raises DatasetRateLimited; nonretryable 4xx attempted once.
- Unit tests use httpx.MockTransport serving the CAPTURED fixture payloads for EVERY endpoint resolve() calls, plus a _FakeHub. The plan's request-shape test (config/split params always sent on /rows) must pass verbatim.
- healthcheck(): /is-valid for openai/gsm8k, short timeout, latency + rate-limit metadata. Default HfApi auth (public needs no token; HF_TOKEN honored).
- LIVE GATE: `uv run pytest tests/live/test_huggingface_live.py -m live -v` must pass (both presets resolve + preview 2 rows). Do not mark complete without it; if Hugging Face is down, report the classified error to the orchestrator instead.
