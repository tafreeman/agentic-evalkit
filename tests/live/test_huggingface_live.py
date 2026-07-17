"""Live checks against the real Hugging Face services (design §14, plan Task 6 Step 5).

These tests only run when explicitly requested, via pytest's ``live``
marker (``uv run pytest tests/live -m live -v``) -- they need a real
network connection, so they're excluded from the normal, hermetic test
suite. They confirm that both of this project's "verified" dataset presets
(GSM8K and SWE-bench Verified) can actually be looked up on the real
Hugging Face Hub and Dataset Viewer, and that previewing a couple of real
rows from each works end to end. Tests that use fake/mocked data (as most
of this project's tests do) prove the code's own logic is correct, but they
can never prove the real Hugging Face integration actually works -- that's
what the tests in this directory are for.
"""

import pytest

from agentic_evalkit.datasets.huggingface import HuggingFaceDatasetProvider
from agentic_evalkit.models import DatasetRef

pytestmark = pytest.mark.live


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
