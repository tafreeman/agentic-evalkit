"""Callable, subprocess, HTTP, and MCP execution targets."""

from agentic_evalkit.targets.base import ExecutionTarget
from agentic_evalkit.targets.callable import CallableTarget
from agentic_evalkit.targets.http import HttpTarget
from agentic_evalkit.targets.mcp import McpTarget
from agentic_evalkit.targets.subprocess import SubprocessTarget

__all__ = [
    "CallableTarget",
    "ExecutionTarget",
    "HttpTarget",
    "McpTarget",
    "SubprocessTarget",
]
