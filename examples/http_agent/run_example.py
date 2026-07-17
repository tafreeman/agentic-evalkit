"""Python-API driver for the http_agent example.

Demonstrates evaluating a real HTTP agent endpoint through the public
``EvalRunner`` API with a custom adapter and an objective ``SchemaGrader``
(design §9's schema/type/format validation step, ahead of any model
judge). The CLI's ``run`` command only recognizes the two curated presets'
adapter/grader names out of the box (``gsm8k@1`` / ``normalized-exact@1``);
a custom adapter and grader such as this example's are wired up in Python,
which is the fully general integration surface -- see
docs/guides/http-agent-example.md.

This module imports and depends only on ``agentic-evalkit`` public
contracts and the local ``httpx``/``pydantic`` runtime dependencies it
already ships with; it never imports ARP, ExecutionKit, or agentic-tools.

Usage (from this directory, in a separate terminal from the stub server):
    python stub_agent_server.py &
    python run_example.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, TypeAdapter

from agentic_evalkit.artifacts import ArtifactStore
from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.graders.composite import SchemaGrader
from agentic_evalkit.models import (
    DatasetRef,
    DatasetSelection,
    EvalRunManifest,
    EvalSample,
    GraderSpec,
    ResolvedDataset,
    SamplingPolicy,
    SourceRecord,
)
from agentic_evalkit.reporters.json import JsonReporter
from agentic_evalkit.runner import EvalRunner
from agentic_evalkit.targets.http import HttpTarget

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
_AGENT_URL = "http://127.0.0.1:8765"
_ADAPTER_NAME = "http-agent-questions@1"
_GRADER_NAME = "agent-response-schema@1"


class AgentResponse(BaseModel):
    """The schema this example's stub agent is expected to return."""

    answer: str
    tool_calls: list[str] = []


class QuestionsAdapter:
    """Projects one row of questions.jsonl into an EvalSample.

    A minimal, example-local adapter -- not part of the public package --
    showing the shape ``EvalRunner`` expects: a ``prepare(record)`` method
    (matching ``BenchmarkAdapter``, design §7) that performs pure
    projection from a ``SourceRecord`` to a typed ``EvalSample``, no I/O.
    """

    api_version = "1"
    name = _ADAPTER_NAME

    def prepare(self, record: SourceRecord) -> EvalSample:
        question = record.data["question"]
        expected = record.data.get("expected_answer")
        if not isinstance(question, str):
            raise TypeError(
                f"questions.jsonl row {record.row_id!r} has a non-string 'question' field"
            )
        return EvalSample(
            sample_id=f"http-agent:{record.row_id}",
            input={"question": question},
            reference=expected if isinstance(expected, str) else None,
            source_row_id=record.row_id,
            source_digest=record.digest,
            adapter=_ADAPTER_NAME,
            grader=GraderSpec(name=_GRADER_NAME, grader_type="schema", hard_gate=True),
        )


class _LocalCatalogAdapter:
    """Adapts LocalDatasetProvider to EvalRunner's minimal catalog protocol."""

    def __init__(self, provider: LocalDatasetProvider) -> None:
        self._provider = provider

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        return await self._provider.resolve(ref)

    def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        return self._provider.iter_records(dataset, offset=offset, limit=limit)


async def main() -> None:
    provider = LocalDatasetProvider(allowed_roots=(_QUESTIONS_PATH.parent,))
    catalog = _LocalCatalogAdapter(provider)

    client = httpx.AsyncClient(timeout=30.0)
    target = HttpTarget(client=client, url=_AGENT_URL, name="stub-agent")

    grader = SchemaGrader(name=_GRADER_NAME, adapter=TypeAdapter(AgentResponse))

    artifact_store = ArtifactStore(Path(__file__).parent / "artifacts")
    runner = EvalRunner(
        catalog=catalog,
        adapters={_ADAPTER_NAME: QuestionsAdapter()},
        targets={"stub-agent": target},
        graders={_GRADER_NAME: grader},
        artifact_store=artifact_store,
    )

    manifest = EvalRunManifest(
        run_name="http-agent-example",
        dataset_ref=DatasetRef(provider="local", dataset_id=str(_QUESTIONS_PATH)),
        adapter=_ADAPTER_NAME,
        grader=_GRADER_NAME,
        target_name="stub-agent",
        selection=DatasetSelection(limit=3),
        sampling=SamplingPolicy(attempts=1),
        attempts=1,
        timeout_seconds=30.0,
        concurrency=1,
    )

    result = await runner.run(manifest)
    await client.aclose()

    report_path = Path(__file__).parent / f"{result.run_id}.json"
    JsonReporter().write(result, report_path)

    summary = result.summary
    print(
        f"outcomes: total={summary.total} passed={summary.passed} "
        f"failed={summary.failed} errors={summary.errors}"
    )
    print(f"report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
