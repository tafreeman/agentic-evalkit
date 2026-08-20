"""Tests for ClaudeAgentTarget.

Every test drives the target through the injected ``query_fn`` seam, the same
way the HTTP target tests hand in a ``MockTransport`` client -- so nothing here
needs the Claude Code CLI, a sign-in, or the network. The replayed messages are
the SDK's own dataclasses rather than stand-ins, so an upstream field rename
fails these tests instead of passing them and failing in production.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    ResultMessage,
    TextBlock,
)
from claude_agent_sdk.types import RateLimitInfo

from agentic_evalkit.errors import TargetFailure
from agentic_evalkit.models import EvalSample, ExecutionStatus
from agentic_evalkit.targets import ClaudeAgentTarget
from agentic_evalkit.targets.base import ExecutionTarget
from agentic_evalkit.targets.claude_agent import subscription_env

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def sample(prompt: str = "What is 2+2?", *, key: str = "prompt") -> EvalSample:
    return EvalSample(
        sample_id="s-1",
        input={key: prompt},
        source_digest="sha256:s1",
        adapter="identity@1",
    )


def assistant(text: str, *, error: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)],
        model="claude-opus-5",
        error=error,  # type: ignore[arg-type]
    )


def result(
    *,
    is_error: bool = False,
    usage: dict[str, Any] | None = None,
    cost: float | None = None,
    session_id: str = "sess-1",
    duration_ms: int = 1234,
    num_turns: int = 1,
    stop_reason: str | None = "end_turn",
    structured_output: Any = None,
    api_error_status: int | None = None,
    errors: list[str] | None = None,
) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=duration_ms,
        duration_api_ms=duration_ms,
        is_error=is_error,
        num_turns=num_turns,
        session_id=session_id,
        stop_reason=stop_reason,
        total_cost_usd=cost,
        usage=usage,
        structured_output=structured_output,
        api_error_status=api_error_status,
        errors=errors,
    )


def rate_limit(status: str) -> RateLimitEvent:
    return RateLimitEvent(
        rate_limit_info=RateLimitInfo(
            status=status,  # type: ignore[arg-type]
            resets_at=1_800_000_000,
            rate_limit_type="five_hour",
        ),
        uuid="u-1",
        session_id="sess-1",
    )


def replay(*messages: Any) -> Any:
    """Build a ``query`` stand-in that replays *messages* and records its calls."""
    calls: list[dict[str, Any]] = []

    async def _query(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        calls.append({"prompt": prompt, "options": options})
        for message in messages:
            yield message

    _query.calls = calls  # type: ignore[attr-defined]
    return _query


def target(query_fn: Any, **kwargs: Any) -> ClaudeAgentTarget:
    kwargs.setdefault("name", "claude")
    return ClaudeAgentTarget(query_fn=query_fn, **kwargs)


async def run(tgt: ClaudeAgentTarget, smpl: EvalSample | None = None, *, attempt: int = 1) -> Any:
    return await tgt.execute(smpl or sample(), attempt=attempt, timeout_seconds=30.0)


# ---------------------------------------------------------------------------
# Credential scrub
# ---------------------------------------------------------------------------


def test_subscription_env_blanks_both_credential_vars() -> None:
    """Blank, not absent: ClaudeAgentOptions.env merges over os.environ.

    The SDK spawns the CLI with ``{**os.environ, **options.env}``, so an entry
    can override a value but never remove the key. Empty is what the CLI
    treats as absent.
    """
    assert subscription_env() == {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""}


def test_subscription_env_lets_an_explicit_override_win() -> None:
    env = subscription_env({"ANTHROPIC_API_KEY": "explicit"})
    assert env["ANTHROPIC_API_KEY"] == "explicit"
    assert env["ANTHROPIC_AUTH_TOKEN"] == ""


async def test_cli_subprocess_cannot_inherit_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inherited key would make the recorded evidence a lie.

    The run would complete, be graded, and be reported with
    ``auth: claude-subscription`` in environment_metadata while actually having
    authenticated against -- and billed -- an API account.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-never-reach-the-cli")
    fake = replay(assistant("4"), result())
    await run(target(fake))
    assert fake.calls[0]["options"].env["ANTHROPIC_API_KEY"] == ""


async def test_explicit_env_override_reaches_the_harness() -> None:
    fake = replay(assistant("4"), result())
    await run(target(fake, env={"ANTHROPIC_API_KEY": "deliberate"}))
    assert fake.calls[0]["options"].env["ANTHROPIC_API_KEY"] == "deliberate"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_the_execution_target_protocol() -> None:
    assert isinstance(target(replay()), ExecutionTarget)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_completed_result_carries_the_assistant_text() -> None:
    tgt = target(replay(assistant("4"), result()))
    outcome = await run(tgt)
    assert outcome.status is ExecutionStatus.COMPLETED
    assert outcome.output == {"output": "4"}
    assert outcome.error is None


async def test_text_blocks_are_concatenated() -> None:
    tgt = target(replay(assistant("The answer "), assistant("is 4."), result()))
    outcome = await run(tgt)
    assert outcome.output == {"output": "The answer is 4."}


async def test_harness_telemetry_lands_on_the_normalized_result() -> None:
    """Token counts, cost, latency, model and session are all reportable here."""
    tgt = target(
        replay(
            assistant("4"),
            result(usage={"input_tokens": 17, "output_tokens": 2}, cost=0.42, duration_ms=987),
        )
    )
    outcome = await run(tgt)
    assert outcome.input_tokens == 17
    assert outcome.output_tokens == 2
    assert outcome.cost_usd == 0.42
    assert outcome.latency_ms == 987
    assert outcome.model_name == "claude-opus-5"
    assert outcome.trace_refs == ("sess-1",)


async def test_structured_output_is_kept_separate_from_text() -> None:
    tgt = target(replay(assistant("4"), result(structured_output={"answer": 4})))
    outcome = await run(tgt)
    assert outcome.output == {"output": "4"}
    assert outcome.structured_output == {"answer": 4}


async def test_non_dict_structured_output_is_ignored() -> None:
    tgt = target(replay(assistant("4"), result(structured_output="not-a-mapping")))
    outcome = await run(tgt)
    assert outcome.structured_output is None


async def test_environment_metadata_records_the_auth_mode() -> None:
    """A reader of the evidence can tell this ran on a subscription."""
    tgt = target(replay(assistant("4"), result(num_turns=3)))
    outcome = await run(tgt)
    assert outcome.environment_metadata["transport"] == "claude-agent-sdk"
    assert outcome.environment_metadata["auth"] == "claude-subscription"
    assert outcome.environment_metadata["num_turns"] == 3
    assert outcome.environment_metadata["stop_reason"] == "end_turn"


async def test_missing_usage_leaves_token_counts_unset_not_zero() -> None:
    """Absent means 'not reported', never 'measured as zero'."""
    tgt = target(replay(assistant("4"), result(usage=None)))
    outcome = await run(tgt)
    assert outcome.input_tokens is None
    assert outcome.output_tokens is None
    assert outcome.cost_usd is None


async def test_non_integer_usage_values_are_discarded() -> None:
    tgt = target(replay(assistant("4"), result(usage={"input_tokens": "many"})))
    outcome = await run(tgt)
    assert outcome.input_tokens is None


async def test_result_message_absent_still_completes() -> None:
    tgt = target(replay(assistant("4")))
    outcome = await run(tgt)
    assert outcome.status is ExecutionStatus.COMPLETED
    assert outcome.output == {"output": "4"}
    assert outcome.latency_ms is None


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------


async def test_prompt_is_read_from_the_configured_field() -> None:
    fake = replay(assistant("ok"), result())
    tgt = target(fake, prompt_field="question")
    await run(tgt, sample("Why?", key="question"))
    assert fake.calls[0]["prompt"] == "Why?"


async def test_missing_prompt_field_is_an_error_result_not_a_crash() -> None:
    tgt = target(replay(assistant("ok"), result()), prompt_field="question")
    outcome = await run(tgt, sample("Why?", key="prompt"))
    assert outcome.status is ExecutionStatus.ERROR
    assert outcome.error is not None
    assert outcome.error["code"] == "target_failure"
    assert "found nothing" in outcome.error["message"]


async def test_non_string_prompt_is_an_error_result() -> None:
    tgt = target(replay(assistant("ok"), result()))
    outcome = await run(
        tgt,
        EvalSample(
            sample_id="s-1",
            input={"prompt": 42},
            source_digest="sha256:s1",
            adapter="identity@1",
        ),
    )
    assert outcome.status is ExecutionStatus.ERROR
    assert outcome.error is not None
    assert "found int" in outcome.error["message"]


# ---------------------------------------------------------------------------
# Harness options
# ---------------------------------------------------------------------------


async def test_tools_are_disabled_by_default() -> None:
    """Grading answers must not give the model filesystem or shell access."""
    fake = replay(assistant("ok"), result())
    await run(target(fake))
    options = fake.calls[0]["options"]
    assert options.tools == []
    assert options.allowed_tools == []
    assert options.max_turns == 1


async def test_configured_options_are_forwarded() -> None:
    fake = replay(assistant("ok"), result())
    tgt = target(
        fake,
        model="claude-sonnet-5",
        system_prompt="Answer with a single number.",
        effort="low",
        allowed_tools=["Read", "Glob"],
        max_turns=4,
        max_budget_usd=0.25,
    )
    await run(tgt)
    options = fake.calls[0]["options"]
    assert options.model == "claude-sonnet-5"
    assert options.system_prompt == "Answer with a single number."
    assert options.effort == "low"
    assert options.allowed_tools == ["Read", "Glob"]
    assert options.max_turns == 4
    assert options.max_budget_usd == 0.25


# ---------------------------------------------------------------------------
# Fingerprint / provenance
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_for_one_configuration() -> None:
    first = target(replay(), model="claude-opus-5")
    second = target(replay(), model="claude-opus-5")
    assert first._fingerprint == second._fingerprint
    assert first._fingerprint.startswith("claude-agent:claude:")


@pytest.mark.parametrize(
    "override",
    [
        {"model": "claude-sonnet-5"},
        {"system_prompt": "Be terse."},
        {"effort": "max"},
        {"allowed_tools": ["Read"]},
        {"max_turns": 9},
        {"name": "other"},
    ],
)
def test_every_setting_that_changes_the_ask_changes_the_fingerprint(
    override: dict[str, Any],
) -> None:
    """compare_runs must refuse to compare across any of these (ADR-0008)."""
    baseline = target(replay())._fingerprint
    assert target(replay(), **override)._fingerprint != baseline


def test_tool_order_does_not_change_the_fingerprint() -> None:
    """The allow-list is a set in effect; ordering it differently is the same config."""
    a = target(replay(), allowed_tools=["Read", "Glob"])._fingerprint
    b = target(replay(), allowed_tools=["Glob", "Read"])._fingerprint
    assert a == b


# ---------------------------------------------------------------------------
# Operational failures (ADR-0008: never folded into task failures)
# ---------------------------------------------------------------------------


async def test_timeout_becomes_a_timeout_result() -> None:
    async def _slow(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        raise TimeoutError
        yield  # pragma: no cover - unreachable, makes this an async generator

    outcome = await run(target(_slow))
    assert outcome.status is ExecutionStatus.TIMEOUT
    assert outcome.error is not None
    assert outcome.error["code"] == "target_timeout"


async def test_rejected_rate_limit_is_an_error_not_an_empty_answer() -> None:
    """A spent subscription window must never be graded as a blank response."""
    tgt = target(replay(rate_limit("rejected")))
    outcome = await run(tgt)
    assert outcome.status is ExecutionStatus.ERROR
    assert outcome.error is not None
    assert outcome.error["type"] == "TargetFailure"
    assert "rate limit" in outcome.error["message"]


async def test_warning_level_rate_limit_does_not_interrupt() -> None:
    tgt = target(replay(rate_limit("allowed_warning"), assistant("4"), result()))
    outcome = await run(tgt)
    assert outcome.status is ExecutionStatus.COMPLETED
    assert outcome.output == {"output": "4"}


async def test_assistant_error_is_an_error_result() -> None:
    tgt = target(replay(assistant("", error="authentication_failed")))
    outcome = await run(tgt)
    assert outcome.status is ExecutionStatus.ERROR
    assert outcome.error is not None
    assert "authentication_failed" in outcome.error["message"]


async def test_failed_run_result_is_an_error_result() -> None:
    tgt = target(replay(result(is_error=True, errors=["boom"], api_error_status=500)))
    outcome = await run(tgt)
    assert outcome.status is ExecutionStatus.ERROR
    assert outcome.error is not None
    assert "boom" in outcome.error["message"]


async def test_failed_run_without_errors_falls_back_to_subtype() -> None:
    tgt = target(replay(result(is_error=True)))
    outcome = await run(tgt)
    assert outcome.status is ExecutionStatus.ERROR
    assert outcome.error is not None
    assert "success" in outcome.error["message"]


async def test_error_results_never_carry_a_traceback() -> None:
    """Frames here hold the prompt and options; only type and message are kept."""

    async def _boom(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        raise RuntimeError("upstream exploded")
        yield  # pragma: no cover - unreachable, makes this an async generator

    outcome = await run(target(_boom))
    assert outcome.error is not None
    assert set(outcome.error) == {"type", "message", "code"}
    assert outcome.error["type"] == "RuntimeError"
    assert outcome.error["message"] == "upstream exploded"


async def test_error_results_still_carry_the_fingerprint() -> None:
    """A failed attempt is still evidence and must be traceable to its config."""
    tgt = target(replay(result(is_error=True)))
    outcome = await run(tgt)
    assert outcome.target_fingerprint == tgt._fingerprint
    assert outcome.environment_metadata["auth"] == "claude-subscription"


async def test_attempt_number_is_recorded() -> None:
    outcome = await run(target(replay(assistant("4"), result())), attempt=3)
    assert outcome.attempt == 3


# ---------------------------------------------------------------------------
# Missing extra
# ---------------------------------------------------------------------------


def test_missing_extra_fails_at_construction_with_an_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agentic_evalkit.targets.claude_agent._SDK_AVAILABLE", False)
    with pytest.raises(TargetFailure, match=r"agentic-evalkit\[claude\]"):
        ClaudeAgentTarget(name="claude")


def test_injected_query_fn_works_without_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller supplying their own query_fn does not need the SDK installed."""
    monkeypatch.setattr("agentic_evalkit.targets.claude_agent._SDK_AVAILABLE", False)
    assert ClaudeAgentTarget(name="claude", query_fn=replay()) is not None
