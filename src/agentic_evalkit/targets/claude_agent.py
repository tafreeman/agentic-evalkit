"""ClaudeAgentTarget: evaluates Claude itself, via a Claude subscription (design §8).

The other targets reach the system under test through something the caller
already owns -- a Python function, a subprocess, an HTTP endpoint, an MCP
server. This one reaches Claude directly through ``claude-agent-sdk``, which
drives a locally installed Claude Code CLI and therefore authenticates with the
operator's **Claude subscription sign-in** rather than an API key. That makes
"grade Claude on this dataset" possible for someone who pays for a subscription
instead of API credits, which was previously not expressible at all: they would
have had to stand up an HTTP service or wrap an API-key client in a callable
first.

Nothing about the evaluation contract changes. Like every other target this one
returns a :class:`~agentic_evalkit.models.NormalizedExecutionResult`, so graders
and reporters cannot tell which kind of target produced a result.

Two properties of this target are worth knowing before trusting a number from it:

* **The harness has no sampling controls.** ``ClaudeAgentOptions`` exposes no
  temperature and no seed, so a run cannot be pinned to a fixed sampling
  configuration the way an API-key client can. Repeat runs vary by the model's
  own nondeterminism. Use ``attempts`` and report the spread rather than
  treating one run as definitive.
* **The fingerprint covers the configuration, not the model weights.** Every
  setting that changes what the model is asked to do -- model id, system prompt,
  effort, tool allow-list, turn ceiling -- is folded into
  ``target_fingerprint``, so ``compare_runs`` refuses to compare across a config
  change. It cannot detect a silent server-side model revision under a stable
  model id; that is a limit of evaluating a hosted model, not of this class.

Credentials are resolved entirely by the CLI. This module never reads, stores,
or forwards them, and no credential can reach a report through it.

Requires the ``claude`` extra::

    pip install 'agentic-evalkit[claude]'
    claude          # interactive sign-in, once per machine
"""

import hashlib
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from agentic_evalkit.errors import TargetFailure
from agentic_evalkit.models import EvalSample, ExecutionStatus, NormalizedExecutionResult

if TYPE_CHECKING:
    from pydantic import JsonValue

# ---------------------------------------------------------------------------
# Optional dependency probe (done once at import time)
# ---------------------------------------------------------------------------
#
# Probed at module scope rather than imported lazily inside ``execute`` because
# the message handling is a sequence of ``isinstance`` checks against the SDK's
# own dataclasses. The names are only ever touched after ``__init__`` has
# verified availability, so an absent extra fails at construction with an
# instruction, never mid-run with a NameError.

try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        CLINotFoundError,
        RateLimitEvent,
        ResultMessage,
        TextBlock,
        query,
    )

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - import-time state, not a branch
    _SDK_AVAILABLE = False

DEFAULT_MODEL: Final[str] = "claude-opus-5"
"""Model used when the caller does not name one."""

_DEFAULT_PROMPT_FIELD: Final[str] = "prompt"
_OUTPUT_KEY: Final[str] = "output"

_INSTALL_HINT: Final[str] = (
    "ClaudeAgentTarget needs the 'claude-agent-sdk' package, which is not "
    "installed; install it with: pip install 'agentic-evalkit[claude]'"
)

_SIGN_IN_HINT: Final[str] = (
    "the Claude Code CLI was not found; install it and sign in once. This "
    "target authenticates with a Claude subscription, not an API key."
)

QueryFn = Callable[..., AsyncIterator[Any]]
"""Shape of ``claude_agent_sdk.query``; injectable so tests never touch a CLI."""


def _fingerprint(
    name: str,
    model: str,
    system_prompt: str | None,
    effort: str | None,
    allowed_tools: Sequence[str],
    max_turns: int,
) -> str:
    """Build a stable ID covering every setting that changes what is asked.

    A fingerprint is an ID for one exact target configuration, used to detect
    whether the target's identity changed between two runs. Everything folded
    in here changes the model's instructions or its latitude, so changing any
    of them makes two runs incomparable and ``compare_runs`` should say so.
    """
    material = "\x1f".join(
        (
            name,
            model,
            system_prompt or "",
            effort or "",
            ",".join(sorted(allowed_tools)),
            str(max_turns),
        )
    )
    digest = hashlib.sha256(material.encode()).hexdigest()[:16]
    return f"claude-agent:{name}:{digest}"


def _usage_tokens(result: "ResultMessage | None") -> tuple[int | None, int | None]:
    """Read Anthropic-style ``input_tokens`` / ``output_tokens`` off a result."""
    if result is None or not result.usage:
        return None, None
    usage = result.usage
    raw_in = usage.get("input_tokens")
    raw_out = usage.get("output_tokens")
    tokens_in = raw_in if isinstance(raw_in, int) else None
    tokens_out = raw_out if isinstance(raw_out, int) else None
    return tokens_in, tokens_out


class ClaudeAgentTarget:
    """Invokes Claude through the Agent SDK as the system under test."""

    def __init__(
        self,
        *,
        name: str,
        model: str = DEFAULT_MODEL,
        prompt_field: str = _DEFAULT_PROMPT_FIELD,
        system_prompt: str | None = None,
        effort: str | None = None,
        allowed_tools: Sequence[str] = (),
        max_turns: int = 1,
        max_budget_usd: float | None = None,
        cwd: str | None = None,
        query_fn: QueryFn | None = None,
    ) -> None:
        """
        Args:
            name: Short label for this target, recorded in the fingerprint.
            model: Claude model id.
            prompt_field: Key in ``sample.input`` holding the prompt text.
            system_prompt: Instructions prepended to every sample.
            effort: Harness reasoning-depth knob (``low`` .. ``max``).
            allowed_tools: Built-in SDK tools the model may use. Empty (the
                default) means a pure completion with no filesystem, shell, or
                network side effects -- the right setting for grading answers.
            max_turns: Turn ceiling; ``1`` keeps the harness to one completion.
            max_budget_usd: Hard per-sample spend ceiling.
            cwd: Working directory handed to the CLI. Only meaningful when
                ``allowed_tools`` grants filesystem access.
            query_fn: Override for ``claude_agent_sdk.query``. Injected the same
                way :class:`~agentic_evalkit.targets.http.HttpTarget` is handed a
                client, so tests drive this target without a CLI or a sign-in.

        Raises:
            TargetFailure: If the ``claude`` extra is not installed.
        """
        if query_fn is None and not _SDK_AVAILABLE:
            raise TargetFailure(message=_INSTALL_HINT, context={"extra": "claude"})

        self._name = name
        self._model = model
        self._prompt_field = prompt_field
        self._system_prompt = system_prompt
        self._effort = effort
        self._allowed_tools = tuple(allowed_tools)
        self._max_turns = max_turns
        self._max_budget_usd = max_budget_usd
        self._cwd = cwd
        self._query: QueryFn = query_fn if query_fn is not None else query
        self._fingerprint = _fingerprint(
            name, model, system_prompt, effort, self._allowed_tools, max_turns
        )

    async def execute(
        self, sample: EvalSample, *, attempt: int, timeout_seconds: float | None
    ) -> NormalizedExecutionResult:
        started_at = datetime.now(UTC)

        prompt = sample.input.get(self._prompt_field)
        if not isinstance(prompt, str):
            found = type(prompt).__name__ if self._prompt_field in sample.input else "nothing"
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error_type="TypeError",
                message=(
                    f"sample input field {self._prompt_field!r} must hold a string "
                    f"prompt, found {found}"
                ),
            )

        try:
            text, result = await self._run(prompt, timeout_seconds=timeout_seconds)
        except TimeoutError:
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.TIMEOUT,
                started_at=started_at,
                error_type="TimeoutError",
                message=(f"claude agent target {self._name!r} exceeded {timeout_seconds}s timeout"),
            )
        except Exception as exc:  # deliberately broad -- turned into an ERROR result below
            # Only the type and message, never a traceback: the frames here hold
            # the prompt and the harness options, and a traceback would carry
            # both into recorded evidence.
            message = _SIGN_IN_HINT if _is_missing_cli(exc) else str(exc)
            return self._error_result(
                sample,
                attempt=attempt,
                status=ExecutionStatus.ERROR,
                started_at=started_at,
                error_type=type(exc).__name__,
                message=message,
            )

        return self._completed_result(
            sample, attempt=attempt, started_at=started_at, text=text, result=result
        )

    # -- harness driving ---------------------------------------------------

    def _build_options(self) -> "ClaudeAgentOptions":
        options: dict[str, Any] = {
            "model": self._model,
            "tools": list(self._allowed_tools),
            "allowed_tools": list(self._allowed_tools),
            "max_turns": self._max_turns,
            "permission_mode": "default",
        }
        if self._system_prompt is not None:
            options["system_prompt"] = self._system_prompt
        if self._effort is not None:
            options["effort"] = self._effort
        if self._max_budget_usd is not None:
            options["max_budget_usd"] = self._max_budget_usd
        if self._cwd is not None:
            options["cwd"] = self._cwd
        return ClaudeAgentOptions(**options)

    async def _run(
        self, prompt: str, *, timeout_seconds: float | None
    ) -> tuple[str, "ResultMessage | None"]:
        """Drive one harness run to completion, returning its text and result.

        No ``asyncio.timeout`` wrapper here: the runner already races every
        attempt against ``timeout_seconds`` and cancels it, and a second,
        inner deadline would only race the outer one to report the same
        expiry under a different code.
        """
        chunks: list[str] = []
        result: ResultMessage | None = None

        async for message in self._query(prompt=prompt, options=self._build_options()):
            if isinstance(message, RateLimitEvent):
                _raise_for_rate_limit(message)
            elif isinstance(message, AssistantMessage):
                _raise_for_assistant_error(message)
                chunks.extend(
                    block.text for block in message.content if isinstance(block, TextBlock)
                )
            elif isinstance(message, ResultMessage):
                _raise_for_result(message)
                result = message

        return "".join(chunks), result

    # -- result assembly ---------------------------------------------------

    def _completed_result(
        self,
        sample: EvalSample,
        *,
        attempt: int,
        started_at: datetime,
        text: str,
        result: "ResultMessage | None",
    ) -> NormalizedExecutionResult:
        finished_at = datetime.now(UTC)
        tokens_in, tokens_out = _usage_tokens(result)

        environment_metadata: dict[str, JsonValue] = {
            "transport": "claude-agent-sdk",
            "auth": "claude-subscription",
            "model": self._model,
            "allowed_tools": list(self._allowed_tools),
        }
        structured: dict[str, JsonValue] | None = None
        trace_refs: tuple[str, ...] = ()
        if result is not None:
            environment_metadata["num_turns"] = result.num_turns
            if result.stop_reason is not None:
                environment_metadata["stop_reason"] = result.stop_reason
            if isinstance(result.structured_output, dict):
                structured = dict(result.structured_output)
            if result.session_id:
                trace_refs = (result.session_id,)

        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            output={_OUTPUT_KEY: text},
            structured_output=structured,
            trace_refs=trace_refs,
            latency_ms=(result.duration_ms if result is not None else None),
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            cost_usd=(result.total_cost_usd if result is not None else None),
            model_name=self._model,
            status=ExecutionStatus.COMPLETED,
            environment_metadata=environment_metadata,
            target_fingerprint=self._fingerprint,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _error_result(
        self,
        sample: EvalSample,
        *,
        attempt: int,
        status: ExecutionStatus,
        started_at: datetime,
        error_type: str,
        message: str,
    ) -> NormalizedExecutionResult:
        # Same stable taxonomy codes the runner's isolation path records, so
        # ``error["code"]`` has one schema regardless of which layer produced it.
        code = "target_timeout" if status is ExecutionStatus.TIMEOUT else "target_failure"
        return NormalizedExecutionResult(
            sample_id=sample.sample_id,
            attempt=attempt,
            status=status,
            error={"type": error_type, "message": message, "code": code},
            model_name=self._model,
            environment_metadata={
                "transport": "claude-agent-sdk",
                "auth": "claude-subscription",
                "model": self._model,
            },
            target_fingerprint=self._fingerprint,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )


# ---------------------------------------------------------------------------
# Failure translation
# ---------------------------------------------------------------------------


def _is_missing_cli(exc: BaseException) -> bool:
    """True when *exc* means the Claude Code CLI is not installed.

    Checked by name rather than ``isinstance`` so the guard still works when
    the extra is absent and a caller supplied their own ``query_fn``.
    """
    return type(exc).__name__ == CLINotFoundError.__name__ if _SDK_AVAILABLE else False


def _raise_for_rate_limit(event: "RateLimitEvent") -> None:
    """Fail the attempt when the subscription's rate-limit window is exhausted.

    A subscription has usage windows an API key does not. Exhausting one is an
    operational failure, not a wrong answer, so it must reach the runner as an
    error rather than be graded as an empty response (ADR-0008).
    """
    info = event.rate_limit_info
    if info.status != "rejected":
        return
    window = info.rate_limit_type or "unknown window"
    raise TargetFailure(
        message=f"Claude subscription rate limit reached ({window})",
        context={"rate_limit_type": window, "resets_at": info.resets_at},
    )


def _raise_for_assistant_error(message: "AssistantMessage") -> None:
    """Fail the attempt on an assistant-level error rather than grading a blank."""
    if message.error is None:
        return
    raise TargetFailure(
        message=f"Claude Agent SDK reported {message.error!r}",
        context={"sdk_error": message.error},
    )


def _raise_for_result(result: "ResultMessage") -> None:
    """Fail the attempt when the harness itself reports the run failed."""
    if not result.is_error:
        return
    detail = "; ".join(result.errors or []) or result.subtype
    raise TargetFailure(
        message=f"Claude Agent SDK run failed: {detail}",
        context={"subtype": result.subtype, "api_error_status": result.api_error_status},
    )
