"""Real evalkit harness driving ARP's own reviewer agent, graded by ARP's agent.yaml rubric.

Every component here is handed to the genuine ``agentic_evalkit.EvalRunner``;
nothing about the pipeline is simulated. The canonical JSON is written by
evalkit's own ``write_canonical_report``, byte-identical in shape to what the
CLI produces.

System under test
-----------------
:class:`ArpReviewTarget` drives **ARP's reviewer agent**, not a locally prompted
model. It builds the agent with ``agentic_v2.langchain.agents.create_agent`` --
the same factory the LangGraph engine's LLM node calls -- for the
``tier2_reviewer`` agent named by the ``review_code`` step of ARP's shipped
``code_review`` workflow. That factory loads ARP's canonical
``agentic_v2/prompts/reviewer.md`` persona; the prompt is never copied into this
file. Its content fingerprint (the ADR-056 prompt registry) is verified at
construction and recorded on every execution result, so a run can prove which
prompt version produced its numbers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

# --- configuration ------------------------------------------------------

#: The ARP agent under test, on NVIDIA's own API. An earlier run of this suite
#: used the same model via OpenRouter's free tier, but that tier caps at 50
#: requests/day and was exhausted mid-run; NVIDIA is a separate quota.
TARGET_MODEL = "nvidia:nvidia/nemotron-3-super-120b-a12b"

#: The judge. Deliberately a different model family from the target so the
#: agent is never grading its own output.
JUDGE_MODEL = "nvidia:deepseek-ai/deepseek-v4-flash"

#: ARP's shipped workflow and the step inside it that performs the review. The
#: step definition (agent name, description, declared outputs) is read from
#: ARP's YAML at runtime rather than restated here.
REVIEW_WORKFLOW = "code_review"
REVIEW_STEP = "review_code"

#: Registry key of ARP's canonical reviewer persona (``prompts/reviewer.md``).
REVIEWER_PROMPT_NAME = "reviewer"

#: Sampling for the reviewer agent. Deterministic, matching the LangGraph
#: engine's own default for a step that declares no ``model_params``.
REVIEW_TEMPERATURE = 0.0

#: Rubric location inside an ARP checkout. Resolved against ``--arp-root`` so
#: no machine-specific absolute path is baked into the harness.
RUBRIC_RELATIVE_PATH = Path("agentic-v2-eval/src/agentic_v2_eval/rubrics/agent.yaml")

#: Hard ceiling on the judge prompt. Both models are large-context hosted
#: models, so complete inputs fit comfortably; this exists only so a
#: pathological case fails loudly instead of being silently truncated.
MAX_JUDGE_PROMPT_CHARS = 200_000

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

#: Transient provider conditions worth retrying: NVIDIA returns 503
#: "ResourceExhausted: Worker local total request limit reached" under
#: concurrency pressure, and 429 for short-window rate limits. Both clear on
#: their own; a permanent 4xx does not and is never retried here.
_RETRYABLE = ("503", "429", "ResourceExhausted", "Too Many Requests", "timeout")

_T = TypeVar("_T")


def default_arp_root() -> Path:
    """Return the ARP checkout root that owns the importable ``agentic_v2``.

    ``agentic_v2`` lives at ``<arp-root>/agentic-workflows-v2/agentic_v2``, so
    the checkout root is two levels above the package directory. Deriving it
    keeps the harness runnable from any clone without an edit.
    """
    import agentic_v2

    package_dir = Path(str(agentic_v2.__file__)).resolve().parent
    return package_dir.parents[1]


def default_rubric_path(arp_root: Path | None = None) -> Path:
    """Return ``<arp-root>/agentic-v2-eval/.../rubrics/agent.yaml``."""
    return (arp_root or default_arp_root()) / RUBRIC_RELATIVE_PATH


def _arp_revision() -> str:
    """Identify the ARP checkout under test: ``<sha>`` or ``<sha>-dirty``.

    Returns ``"unknown"`` when the checkout is not a git repository or git is
    unavailable -- reported honestly rather than silently omitted, because a
    fingerprint that quietly drops this field would claim more comparability
    than it can back.
    """
    root = default_arp_root()
    try:
        head = subprocess.run(  # noqa: S603 - fixed argv, no shell, local driver
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if head.returncode != 0:
            return "unknown"
        revision = head.stdout.strip()
        status = subprocess.run(  # noqa: S603 - fixed argv, no shell, local driver
            ["git", "-C", str(root), "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{revision}-dirty" if status.stdout.strip() else revision


def _is_retryable(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}"
    return any(marker in text for marker in _RETRYABLE)


async def _with_retry(
    call: Callable[[], Awaitable[_T]], *, attempts: int = 4, base_delay: float = 3.0
) -> _T:
    """Await ``call()``, retrying transient provider failures with linear backoff."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await call()
        # Broad by design: provider SDKs raise unrelated exception types for the
        # same transient conditions, so the retry decision is made on the message.
        except Exception as error:
            last = error
            if attempt == attempts - 1 or not _is_retryable(error):
                raise
            await asyncio.sleep(base_delay * (attempt + 1))
    raise last  # pragma: no cover - loop always returns or raises above


def _json_safe(value: Any) -> Any:
    """Round-trip ``value`` through JSON so it satisfies evalkit's wire models."""
    return json.loads(json.dumps(value, default=str))


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
        #: Highest level the rubric itself defines, so the 0-1 normalization
        #: below is read off the YAML rather than hardcoded.
        self.max_level = float(max(int(level) for c in self.criteria for level in c["levels"]))

    def prompt_block(self) -> str:
        lines = []
        for criterion in self.criteria:
            levels = " | ".join(
                f"{score}={text}"
                for score, text in sorted(criterion["levels"].items(), reverse=True)
            )
            lines.append(
                f"- {criterion['name']} (weight {criterion['weight']}): "
                f"{criterion['description']}\n    Levels: {levels}"
            )
        return "\n".join(lines)

    def weighted_score(self, scores: Mapping[str, float]) -> float:
        """Weight-normalized score in [0,1] over **every** criterion.

        Raises:
            KeyError: If any rubric criterion is absent from ``scores``.
                Renormalizing over whatever the judge happened to return
                silently inflates the result, so a partial set is refused
                here and handled as an ABSTAIN by the grader.
        """
        total = 0.0
        weight_sum = 0.0
        for criterion in self.criteria:
            weight = float(criterion["weight"])
            total += weight * (float(scores[str(criterion["name"])]) / self.max_level)
            weight_sum += weight
        if weight_sum <= 0.0:
            raise ValueError(f"Rubric {self.rubric_id} has no positive criterion weight")
        return total / weight_sum


# --- dataset ------------------------------------------------------------


class JsonlCatalog:
    """Minimal catalog over a local JSONL file, matching evalkit's catalog protocol."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def __len__(self) -> int:
        return len(self._rows)

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
                digest=hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest(),
            )


class CodeReviewAdapter:
    """Turns one extracted source file into an EvalSample for ARP's reviewer agent."""

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
    """Drives ARP's own ``tier2_reviewer`` agent over one source file.

    Scope, stated precisely. This constructs the *agent object* ARP's LangGraph
    engine constructs for the ``review_code`` step of the shipped
    ``code_review`` workflow -- same factory (``create_agent``), same step
    definition read from ARP's YAML, same canonical ``prompts/reviewer.md``
    persona, same task-description assembly (``build_task_description``), same
    response and usage extraction (``extract_agent_response_text`` /
    ``extract_agent_metadata``) -- and invokes it once per sample. It does
    **not** run the workflow DAG: the four sibling steps (``parse_code``,
    ``style_check``, ``complexity_analysis``, ``generate_summary``) sit
    upstream/downstream of the review itself and have no bearing on reviewing
    one file, so they are deliberately not executed.

    Tools are explicitly unbound (``tool_names=[]``). The step declares no tools
    of its own, which in a live workflow resolves to "every tier-2 tool"; here
    the file content is handed to the agent inline, so binding that surface
    would only invite filesystem/network calls the fail-closed approval gate
    (ADR-047) denies -- that would measure governance, not review quality.

    The persona is resolved through ARP at runtime and its ADR-056 content
    fingerprint is checked against the registry at construction, then recorded
    in ``environment_metadata`` and folded into ``target_fingerprint`` on every
    result, so the run proves which prompt version it measured.
    """

    def __init__(self, model_id: str) -> None:
        from agentic_v2.engine.prompt_assembly import load_agent_system_prompt
        from agentic_v2.langchain.agents import create_agent, parse_agent_tier
        from agentic_v2.langchain.config import ModelParamsConfig, load_workflow_config
        from agentic_v2.langchain.models import get_model_candidates_for_tier
        from agentic_v2.prompts import get_prompt_path
        from agentic_v2.prompts.registry import (
            compute_content_hash,
            default_registry,
            normalize_prompt_text,
        )

        workflow = load_workflow_config(REVIEW_WORKFLOW)
        matching = [step for step in workflow.steps if step.name == REVIEW_STEP]
        if not matching:
            raise RuntimeError(f"ARP workflow {REVIEW_WORKFLOW!r} defines no {REVIEW_STEP!r} step")
        self._step = matching[0]
        self._model_id = model_id

        record = default_registry().get(REVIEWER_PROMPT_NAME)
        prompt_path = get_prompt_path(REVIEWER_PROMPT_NAME)
        if prompt_path is None:
            raise RuntimeError("ARP ships no reviewer persona file to fingerprint")
        # Two independent proofs that the persona measured here is ARP's
        # canonical one: the file ``create_agent`` reads off disk, and the role
        # this step's agent name resolves to, must both hash to the registry
        # record. A mismatch means prompt drift and fails the run at startup.
        checks = {
            f"file {prompt_path.name}": normalize_prompt_text(
                prompt_path.read_text(encoding="utf-8")
            ),
            f"role of {self._step.agent}": load_agent_system_prompt(self._step.agent) or "",
        }
        for label, text in checks.items():
            if compute_content_hash(text) != record.content_sha256:
                raise RuntimeError(
                    f"ARP reviewer prompt drift: {label} does not match registry "
                    f"record {record.qualified_version}"
                )

        # ``create_agent`` treats ``model_override`` as the *first* candidate in
        # the tier chain and quietly falls through to the next available model
        # if it cannot be built -- on a checkout with no NVIDIA credentials the
        # run would silently measure some other provider's model. ARP filters
        # candidates by *provider* availability, so this catches the missing-key
        # case at startup; a bad model id under a configured provider still
        # fails at invoke time (never substituted), and the model the provider
        # reports back is recorded per result.
        candidates = get_model_candidates_for_tier(
            parse_agent_tier(self._step.agent),
            model_id,
            include_unavailable=False,
            include_gh_backup=True,
        )
        if not candidates or candidates[0] != model_id:
            raise RuntimeError(
                f"ARP will not route to {model_id!r} (resolved chain: {candidates}). "
                "Check the provider credentials before running."
            )

        self._agent = create_agent(
            self._step.agent,
            tool_names=[],
            prompt_file=self._step.prompt_file,
            model_override=model_id,
            model_params=ModelParamsConfig(temperature=REVIEW_TEMPERATURE),
        )
        self._declared_outputs = [key for key in self._step.outputs if key != "raw_response"]
        self.provenance: dict[str, Any] = {
            # The ARP checkout itself is part of the system under test: the same
            # prompt fingerprint can sit on top of a different create_agent,
            # workflow YAML, or task-assembly implementation. Without this, two
            # runs of genuinely different systems produce the same target
            # fingerprint and would be wrongly treated as comparable.
            "arp_revision": _arp_revision(),
            "arp_workflow": REVIEW_WORKFLOW,
            "arp_step": self._step.name,
            "arp_agent": self._step.agent,
            "arp_bound_tools": [],
            "arp_declared_outputs": list(self._declared_outputs),
            "arp_temperature": REVIEW_TEMPERATURE,
            "arp_prompt_name": record.name,
            "arp_prompt_source": record.source,
            "arp_prompt_declared_version": record.declared_version,
            "arp_prompt_qualified_version": record.qualified_version,
            "arp_prompt_sha256": record.content_sha256,
            "model_id": model_id,
            "integration": (
                "agentic_v2.langchain.agents.create_agent -> LangGraph react agent; "
                "task text from agentic_v2.langchain.graph_wiring.build_task_description; "
                "single step, no workflow DAG, no bound tools"
            ),
        }
        self.fingerprint = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(self.provenance, sort_keys=True).encode("utf-8")
            ).hexdigest()
        )

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        from agentic_v2.langchain.graph_wiring import (
            build_task_description,
            extract_agent_metadata,
            extract_agent_response_text,
            parse_step_outputs,
        )
        from langchain_core.messages import HumanMessage

        started = datetime.now(UTC)
        clock = time.perf_counter()
        resolved_inputs = {
            "file_path": sample.input["file_path"],
            "language": sample.input["language"],
            "content": sample.input["content"],
        }
        task_description = build_task_description(self._step, resolved_inputs)
        agent_result = await _with_retry(
            lambda: asyncio.wait_for(
                self._agent.ainvoke({"messages": [HumanMessage(content=task_description)]}),
                timeout=timeout_seconds,
            )
        )
        latency_ms = (time.perf_counter() - clock) * 1000.0

        text = extract_agent_response_text(agent_result)
        usage = extract_agent_metadata(agent_result)
        parsed = parse_step_outputs(
            text, expected_output_keys=self._declared_outputs, warn_on_missing=False
        )
        # ``raw_response`` duplicates ``output["review"]`` verbatim; drop it so
        # the canonical report carries each review exactly once.
        parsed.pop("raw_response", None)

        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output={"review": text},
            structured_output=_json_safe(parsed) or None,
            status=ExecutionStatus.COMPLETED if text.strip() else ExecutionStatus.FAILED,
            latency_ms=round(latency_ms, 2),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            # Left unset deliberately: the NVIDIA endpoint exposes no per-token
            # price, so any dollar figure here would be invented. Token counts
            # above are measured; cost is reported as unverified, not as $0.
            cost_usd=None,
            model_name=str(usage.get("model") or self._model_id),
            environment_metadata=_json_safe(self.provenance),
            target_fingerprint=self.fingerprint,
            started_at=started,
            finished_at=datetime.now(UTC),
        )


# --- grader -------------------------------------------------------------


class ArpRubricGrader:
    """Scores the reviewer agent's output against ARP's agent.yaml rubric via an LLM judge."""

    name = "arp-agent-rubric@1"

    def __init__(self, rubric: Rubric, judge_model_id: str) -> None:
        from agentic_v2.langchain.models import get_chat_model

        self._rubric = rubric
        self._judge_model_id = judge_model_id
        self._judge = get_chat_model(judge_model_id, temperature=0.0)

    def _prompt(self, sample: EvalSample, review: str) -> str:
        """Build the judge prompt from **complete** inputs.

        Nothing is sliced: truncating the source or the review here measures the
        harness, not the review. Oversized prompts are refused by
        :meth:`grade` instead, which is loud where truncation was silent.
        """
        return (
            "You are grading a code-review produced by an AI agent.\n\n"
            f"=== FILE UNDER REVIEW ({sample.input['file_path']}) ===\n"
            f"{sample.input['content']}\n\n"
            f"=== AGENT'S REVIEW ===\n{review}\n\n"
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

    def _scores(self, raw_scores: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
        """Return the usable criterion scores and the names that are unusable.

        A criterion counts as missing when the judge omitted it or returned a
        non-numeric value. ``bool`` is excluded explicitly (it is a subclass of
        ``int``, so a JSON ``true`` would otherwise score as 1.0), and so are
        NaN/Infinity, which ``json.loads`` accepts.
        """
        scores: dict[str, float] = {}
        for name in self._rubric.names:
            value = raw_scores.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if not math.isfinite(number):
                continue
            # Out-of-range and fractional values are *rejected*, not repaired.
            # Clamping turned a judge that answered "50" for every criterion
            # into a perfect score, and a negative into a legitimate zero --
            # manufacturing a gradeable result out of a malformed one. The
            # prompt asks for integer levels 0-5; anything else means the
            # judge did not answer the question, which is what ABSTAIN is for.
            if number != int(number) or not 0 <= number <= self._rubric.max_level:
                continue
            scores[name] = number
        missing = [name for name in self._rubric.names if name not in scores]
        return scores, missing

    def _result(
        self,
        sample: EvalSample,
        status: GradeStatus,
        evidence: dict[str, Any],
        *,
        score: float | None = None,
    ) -> GradeResult:
        return GradeResult(
            sample_id=sample.sample_id,
            grader=self.name,
            status=status,
            score=score,
            hard_gate=False,
            rubric_id=self._rubric.rubric_id,
            evidence=_json_safe(evidence),
            created_at=datetime.now(UTC),
        )

    async def grade(self, sample: EvalSample, execution: NormalizedExecutionResult) -> GradeResult:
        from langchain_core.messages import HumanMessage

        review = str((execution.output or {}).get("review", ""))
        source = str(sample.input["content"])
        prompt = self._prompt(sample, review)

        if len(prompt) > MAX_JUDGE_PROMPT_CHARS:
            # Returned rather than raised so the evidence names the exact sizes;
            # the runner would otherwise reduce a raise to a redacted message.
            return self._result(
                sample,
                GradeStatus.ERROR,
                {
                    "reason": "judge prompt exceeds the hard size guard",
                    "detail": "inputs are never truncated; this sample is failed instead",
                    "prompt_chars": len(prompt),
                    "limit_chars": MAX_JUDGE_PROMPT_CHARS,
                    "source_chars": len(source),
                    "review_chars": len(review),
                    "judge_model": self._judge_model_id,
                },
            )

        response = await _with_retry(lambda: self._judge.ainvoke([HumanMessage(content=prompt)]))
        raw = response.content if isinstance(response.content, str) else str(response.content)
        usage = getattr(response, "usage_metadata", None) or {}
        judge_tokens = {
            "judge_model": self._judge_model_id,
            "judge_input_tokens": usage.get("input_tokens"),
            "judge_output_tokens": usage.get("output_tokens"),
        }
        parsed = self._parse(raw)

        if parsed is None or not isinstance(parsed.get("scores"), dict):
            # Never invent a score: an unparseable judge is recorded as ABSTAIN.
            return self._result(
                sample,
                GradeStatus.ABSTAIN,
                {
                    "reason": "judge returned unparseable output",
                    "judge_raw": raw[:1000],
                    **judge_tokens,
                },
            )

        scores, missing = self._scores(parsed["scores"])
        rationale = str(parsed.get("rationale", ""))[:500]

        if missing:
            # A partial score set can only be turned into PASS/FAIL by
            # renormalizing over the criteria that happen to be present, which
            # inflates the result. Abstain and name what was missing instead.
            return self._result(
                sample,
                GradeStatus.ABSTAIN,
                {
                    "reason": "judge omitted or unusably scored rubric criteria",
                    "missing_criteria": missing,
                    "criterion_scores": scores,
                    "judge_raw_scores": parsed["scores"],
                    "rationale": rationale,
                    **judge_tokens,
                },
            )

        weighted = self._rubric.weighted_score(scores)
        return self._result(
            sample,
            GradeStatus.PASS if weighted >= self._rubric.pass_threshold else GradeStatus.FAIL,
            {
                "criterion_scores": scores,
                "missing_criteria": [],
                # The judge's scores exactly as parsed, so a reader can verify
                # no value was reshaped on the way in. Without this, a score of
                # 5.0 in the record is indistinguishable from a rejected-or-
                # repaired 50, and the grader's own honesty is unauditable.
                "judge_raw_scores": parsed["scores"],
                "pass_threshold": self._rubric.pass_threshold,
                "rationale": rationale,
                **judge_tokens,
            },
            score=round(weighted, 4),
        )
