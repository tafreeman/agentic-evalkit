"""Real evalkit harness driving ARP's Reviewer Agent, graded by ARP's agent.yaml rubric.

Every component here is handed to the genuine ``agentic_evalkit.EvalRunner``;
nothing about the pipeline is simulated. The canonical JSON is written by
evalkit's own ``write_canonical_report``, byte-identical in shape to what the
CLI produces.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agentic_evalkit.models import (
    DatasetRef,
    EvalSample,
    ExecutionStatus,
    GradeResult,
    GradeStatus,
    NormalizedExecutionResult,
    ResolvedDataset,
    SourceRecord,
)

# --- configuration ------------------------------------------------------

#: The ARP agent under test, on NVIDIA's own API. An earlier run of this suite
#: used the same model via OpenRouter's free tier, but that tier caps at 50
#: requests/day and was exhausted mid-run; NVIDIA is a separate quota.
TARGET_MODEL = "nvidia:nvidia/nemotron-3-super-120b-a12b"

#: The judge. Deliberately a different model family from the target so the
#: agent is never grading its own output.
JUDGE_MODEL = "nvidia:deepseek-ai/deepseek-v4-flash"

RUBRIC_PATH = Path(
    r"C:\Users\tandf\source\agentic-runtime-platform"
    r"\agentic-v2-eval\src\agentic_v2_eval\rubrics\agent.yaml"
)

REVIEW_SYSTEM_PROMPT = (
    "You are the Reviewer Agent: an expert at code review and security analysis. "
    "Review the source file you are given. Report correctness bugs, missing "
    "functionality, code-quality problems, and security vulnerabilities. Be "
    "specific and cite the code you are referring to. Be concise."
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

#: Transient provider conditions worth retrying: NVIDIA returns 503
#: "ResourceExhausted: Worker local total request limit reached" under
#: concurrency pressure, and 429 for short-window rate limits. Both clear on
#: their own; a permanent 4xx does not and is never retried here.
_RETRYABLE = ("503", "429", "ResourceExhausted", "Too Many Requests", "timeout")


def _is_retryable(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}"
    return any(marker in text for marker in _RETRYABLE)


async def _with_retry(call, *, attempts: int = 4, base_delay: float = 3.0):
    """Await ``call()``, retrying transient provider failures with linear backoff."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as error:  # noqa: BLE001 - re-raised below when not retryable
            last = error
            if attempt == attempts - 1 or not _is_retryable(error):
                raise
            await asyncio.sleep(base_delay * (attempt + 1))
    raise last  # pragma: no cover - loop always returns or raises above


# --- rubric -------------------------------------------------------------


class Rubric:
    """ARP's YAML rubric: named criteria with weights, levels, and thresholds."""

    def __init__(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.criteria = data["criteria"]
        self.thresholds = data.get("thresholds", {})
        self.metadata = data.get("metadata", {})
        self.pass_threshold = float(self.thresholds.get("pass", 0.7))
        self.rubric_id = f"{path.stem}@{self.metadata.get('version', '?')}"
        self.names = [str(c["name"]) for c in self.criteria]

    def prompt_block(self) -> str:
        lines = []
        for criterion in self.criteria:
            levels = " | ".join(
                f"{score}={text}" for score, text in sorted(criterion["levels"].items(), reverse=True)
            )
            lines.append(
                f"- {criterion['name']} (weight {criterion['weight']}): "
                f"{criterion['description']}\n    Levels: {levels}"
            )
        return "\n".join(lines)

    def weighted_score(self, scores: dict[str, float]) -> float:
        """Weight-normalized score in [0,1]; each criterion is scored 0-5."""
        total = 0.0
        weight_sum = 0.0
        for criterion in self.criteria:
            name = str(criterion["name"])
            if name not in scores:
                continue
            weight = float(criterion["weight"])
            total += weight * (float(scores[name]) / 5.0)
            weight_sum += weight
        return total / weight_sum if weight_sum else 0.0


# --- dataset ------------------------------------------------------------


class JsonlCatalog:
    """Minimal catalog over a local JSONL file, matching evalkit's catalog protocol."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self._digest = hashlib.sha256(path.read_bytes()).hexdigest()

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        return ResolvedDataset(
            dataset_id=ref.dataset_id,
            revision=self._digest[:16],
            row_count=len(self._rows),
            retrieved_at=datetime.now(UTC),
            checksums={"sha256": self._digest},
        )

    async def iter_records(
        self, dataset: ResolvedDataset, *, offset: int = 0, limit: int | None = None
    ) -> AsyncIterator[SourceRecord]:
        rows = self._rows[offset : (offset + limit) if limit is not None else None]
        for row in rows:
            yield SourceRecord(
                row_id=str(row["case_id"]),
                data=row,
                digest=hashlib.sha256(
                    json.dumps(row, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            )


class CodeReviewAdapter:
    """Turns one extracted source file into an EvalSample for the Reviewer Agent."""

    name = "arp-code-review@1"

    def prepare(self, record: SourceRecord) -> EvalSample:
        row = record.data
        return EvalSample(
            sample_id=str(row["case_id"]),
            input={
                "file_path": row["file_path"],
                "language": row["language"],
                "content": row["content"],
            },
            metadata={
                "language": row["language"],
                "chars": row["chars"],
                "source_sha256": row["sha256"],
                "origin": "arp fullstack_generation run logs",
            },
            tags=(str(row["language"]),),
            source_row_id=record.row_id,
            source_digest=record.digest,
            adapter=self.name,
        )


# --- target -------------------------------------------------------------


class ArpReviewTarget:
    """Runs ARP's Reviewer Agent through the platform's own LangChain model layer."""

    def __init__(self, model_id: str) -> None:
        from agentic_v2.langchain.models import get_chat_model

        self._model_id = model_id
        self._model = get_chat_model(model_id, temperature=0.0)

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        started = datetime.now(UTC)
        clock = time.perf_counter()
        prompt = (
            f"File: {sample.input['file_path']}\n"
            f"Language: {sample.input['language']}\n\n"
            f"```\n{sample.input['content']}\n```"
        )
        messages = [
            SystemMessage(content=REVIEW_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = await _with_retry(
            lambda: asyncio.wait_for(
                self._model.ainvoke(messages), timeout=timeout_seconds
            )
        )
        latency_ms = (time.perf_counter() - clock) * 1000.0
        usage = getattr(response, "usage_metadata", None) or {}
        text = response.content if isinstance(response.content, str) else str(response.content)

        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output={"review": text},
            status=ExecutionStatus.COMPLETED if text.strip() else ExecutionStatus.FAILED,
            latency_ms=round(latency_ms, 2),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            # Left unset deliberately: the NVIDIA endpoint exposes no per-token
            # price, so any dollar figure here would be invented. Token counts
            # above are measured; cost is reported as unverified, not as $0.
            cost_usd=None,
            model_name=self._model_id,
            started_at=started,
            finished_at=datetime.now(UTC),
        )


# --- grader -------------------------------------------------------------


class ArpRubricGrader:
    """Scores the Reviewer Agent's output against ARP's agent.yaml rubric via an LLM judge."""

    name = "arp-agent-rubric@1"

    def __init__(self, rubric: Rubric, judge_model_id: str) -> None:
        from agentic_v2.langchain.models import get_chat_model

        self._rubric = rubric
        self._judge_model_id = judge_model_id
        self._judge = get_chat_model(judge_model_id, temperature=0.0)

    def _prompt(self, sample: EvalSample, review: str) -> str:
        return (
            "You are grading a code-review produced by an AI agent.\n\n"
            f"=== FILE UNDER REVIEW ({sample.input['file_path']}) ===\n"
            f"{sample.input['content'][:4000]}\n\n"
            f"=== AGENT'S REVIEW ===\n{review[:6000]}\n\n"
            "=== RUBRIC ===\n"
            f"{self._rubric.prompt_block()}\n\n"
            "Score the REVIEW on each criterion using the integer levels 0-5.\n"
            "Respond with ONLY a JSON object, no prose, no markdown fences, of the form:\n"
            '{"scores": {'
            + ", ".join(f'"{n}": <0-5>' for n in self._rubric.names)
            + '}, "rationale": "<one sentence>"}'
        )

    @staticmethod
    def _parse(text: str) -> dict[str, Any] | None:
        match = _JSON_BLOCK.search(text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    async def grade(
        self, sample: EvalSample, execution: NormalizedExecutionResult
    ) -> GradeResult:
        from langchain_core.messages import HumanMessage

        review = str((execution.output or {}).get("review", ""))
        now = datetime.now(UTC)

        judge_messages = [HumanMessage(content=self._prompt(sample, review))]
        response = await _with_retry(lambda: self._judge.ainvoke(judge_messages))
        raw = response.content if isinstance(response.content, str) else str(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        parsed = self._parse(raw)

        if parsed is None or not isinstance(parsed.get("scores"), dict):
            # Never invent a score: an unparseable judge is recorded as ABSTAIN.
            return GradeResult(
                sample_id=sample.sample_id,
                grader=self.name,
                status=GradeStatus.ABSTAIN,
                hard_gate=False,
                rubric_id=self._rubric.rubric_id,
                evidence={
                    "reason": "judge returned unparseable output",
                    "judge_model": self._judge_model_id,
                    "judge_raw": raw[:1000],
                    "judge_input_tokens": usage.get("input_tokens"),
                    "judge_output_tokens": usage.get("output_tokens"),
                },
                created_at=now,
            )

        scores: dict[str, float] = {}
        for name in self._rubric.names:
            value = parsed["scores"].get(name)
            if isinstance(value, (int, float)):
                scores[name] = max(0.0, min(5.0, float(value)))

        missing = [n for n in self._rubric.names if n not in scores]
        weighted = self._rubric.weighted_score(scores)
        passed = weighted >= self._rubric.pass_threshold

        return GradeResult(
            sample_id=sample.sample_id,
            grader=self.name,
            status=GradeStatus.PASS if passed else GradeStatus.FAIL,
            score=round(weighted, 4),
            hard_gate=False,
            rubric_id=self._rubric.rubric_id,
            evidence={
                "criterion_scores": scores,
                "missing_criteria": missing,
                "pass_threshold": self._rubric.pass_threshold,
                "rationale": str(parsed.get("rationale", ""))[:500],
                "judge_model": self._judge_model_id,
                "judge_input_tokens": usage.get("input_tokens"),
                "judge_output_tokens": usage.get("output_tokens"),
            },
            created_at=now,
        )
