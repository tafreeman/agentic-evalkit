"""Opt-in live Hugging Face verification (design §14, plan Task 6 Step 5).

Behind the ``live`` marker (``uv run pytest tests/live/test_huggingface_live.py
-m live -v``): both verified presets must resolve against the real Hub and
Dataset Viewer and preview two real rows. Mocks prove local behavior but
never live source integration, so this file is the only place that exercises
the real network path end to end.
"""

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
