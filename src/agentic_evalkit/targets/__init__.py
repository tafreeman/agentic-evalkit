"""Callable, subprocess, HTTP, MCP, and Claude Agent SDK execution targets."""

from agentic_evalkit.targets.base import ExecutionTarget
from agentic_evalkit.targets.callable import CallableTarget
from agentic_evalkit.targets.claude_agent import ClaudeAgentTarget
from agentic_evalkit.targets.http import HttpTarget
from agentic_evalkit.targets.mcp import McpTarget
from agentic_evalkit.targets.subprocess import SubprocessTarget

__all__ = [
    "CallableTarget",
    "ClaudeAgentTarget",
    "ExecutionTarget",
    "HttpTarget",
    "McpTarget",
    "SubprocessTarget",
]
