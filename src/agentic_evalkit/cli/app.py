"""The root Typer application and shared CLI error/output policy (design §11.1).

This module owns three things every subcommand needs:

1. The single root ``Typer(no_args_is_help=True)`` app that
   ``src/agentic_evalkit/cli/__init__.py`` exports as ``app``, with the
   ``datasets`` command group mounted and ``doctor``/``init``/``validate``/
   ``run`` registered directly on it (plan Task 14 Step 4's command list).
2. The exit-code policy: 0 success, 2 invalid input/manifest, 3 missing
   capability, 4 provider/dataset resolution or availability errors
   (including dataset not found), 5 evaluation completed with
   infrastructure errors, 130 cancelled.
3. :func:`run_cli_command`, the single error boundary every command function
   goes through: it catches only :class:`AgenticEvalkitError`, prints its
   stable ``.code`` and message, and exits with the mapped code. Any other
   exception is left to propagate into a normal Python traceback -- visible
   only when the caller passed ``--debug`` (checked by the command itself
   before calling into framework code that might raise unexpected errors);
   otherwise Typer's default handling still prints it, but commands are
   expected to have already converted foreseeable failures into typed
   errors before this boundary is reached.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import IntEnum
from typing import Annotated, TypeVar

import typer
from rich.console import Console
from rich.text import Text

from agentic_evalkit import __version__
from agentic_evalkit.errors import (
    AgenticEvalkitError,
    DatasetAccessDenied,
    DatasetConfigRequired,
    DatasetIntegrityError,
    DatasetLicenseRejected,
    DatasetNotFound,
    DatasetProviderUnavailable,
    DatasetRateLimited,
    DatasetSchemaMismatch,
    DatasetSplitNotFound,
    ManifestValidationError,
    OfflineCacheMiss,
    PluginCompatibilityError,
    UnsafeCodeRequired,
)

__all__ = [
    "ExitCode",
    "OutputFormat",
    "app",
    "console",
    "err_console",
    "run_cli_command",
    "safe_text",
]

T = TypeVar("T")

#: Minimum render width for table output. Rich falls back to 80 columns
#: when stdout is not a real terminal (piped output, CI logs, CliRunner
#: tests) -- too narrow to show a full dataset ID next to its adapter and
#: grader names without truncating identifiers a user needs to copy
#: verbatim. A genuine narrow terminal is still honored: this only raises
#: the floor for the non-terminal fallback case.
_MIN_CONSOLE_WIDTH = 120

#: Stdout for normal command output (tables, JSON payloads, success messages).
console = Console(width=_MIN_CONSOLE_WIDTH if not Console().is_terminal else None)
#: Stderr for error/diagnostic output, kept separate so ``--format json``
#: stdout always parses cleanly even when a warning was also printed.
err_console = Console(stderr=True)


class ExitCode(IntEnum):
    """The CLI's stable exit-code policy (plan Task 14 Step 4)."""

    SUCCESS = 0
    INVALID_INPUT = 2
    MISSING_CAPABILITY = 3
    PROVIDER_ERROR = 4
    INFRASTRUCTURE_ERROR = 5
    CANCELLED = 130


class OutputFormat(str):
    """String-valued ``--format`` choices, kept as plain strings for Typer."""

    TABLE = "table"
    JSON = "json"


#: Maps each typed error subclass to its exit code. Order matters only in
#: that ``AgenticEvalkitError`` (the catch-all base) is checked last, via
#: ``isinstance`` fallthrough in :func:`_exit_code_for_error` -- every other
#: entry here is a genuine subclass with a more specific code.
_ERROR_EXIT_CODES: dict[type[AgenticEvalkitError], ExitCode] = {
    ManifestValidationError: ExitCode.INVALID_INPUT,
    DatasetSchemaMismatch: ExitCode.INVALID_INPUT,
    UnsafeCodeRequired: ExitCode.INVALID_INPUT,
    PluginCompatibilityError: ExitCode.MISSING_CAPABILITY,
    DatasetNotFound: ExitCode.PROVIDER_ERROR,
    DatasetConfigRequired: ExitCode.PROVIDER_ERROR,
    DatasetSplitNotFound: ExitCode.PROVIDER_ERROR,
    DatasetAccessDenied: ExitCode.PROVIDER_ERROR,
    DatasetLicenseRejected: ExitCode.PROVIDER_ERROR,
    DatasetIntegrityError: ExitCode.PROVIDER_ERROR,
    DatasetProviderUnavailable: ExitCode.PROVIDER_ERROR,
    DatasetRateLimited: ExitCode.PROVIDER_ERROR,
    OfflineCacheMiss: ExitCode.PROVIDER_ERROR,
}


def _exit_code_for_error(error: AgenticEvalkitError) -> ExitCode:
    """Look up the most specific mapped exit code for ``error``.

    Falls back to :attr:`ExitCode.INFRASTRUCTURE_ERROR` (5) for any
    ``AgenticEvalkitError`` subclass not explicitly listed above (for
    example, ``TargetFailure``/``TargetTimeout``/``GraderError`` raised
    while a run was otherwise in progress) -- those represent an evaluation
    that reached execution but hit an infrastructure-level problem, matching
    the plan's "evaluation completed with infrastructure errors" code.
    """
    for error_type, code in _ERROR_EXIT_CODES.items():
        if isinstance(error, error_type):
            return code
    return ExitCode.INFRASTRUCTURE_ERROR


def safe_text(value: object) -> Text:
    """Wrap dynamic content so Rich renders it literally, never as markup.

    Every string this CLI prints or puts in a table cell that did not come
    from a hardcoded template literal here (dataset IDs, error messages,
    error codes, remediation hints, JSON-derived field values, row data) can
    legitimately contain a ``[...]``-shaped substring -- an error code like
    ``dataset_not_found`` wrapped in brackets, a pip extra like
    ``agentic-evalkit[parquet]``, or arbitrary upstream text. Passed as a
    raw ``str`` to ``console.print``/``Table.add_row``, Rich parses that
    ``[...]`` as an (unrecognized) style tag and silently drops it from the
    rendered output rather than raising -- corrupting exactly the
    identifiers a user needs to read or copy. A :class:`rich.text.Text`
    instance bypasses markup parsing entirely and is always rendered
    character-for-character, so every call site that interpolates dynamic
    content wraps it with this function instead of building an f-string
    Rich will re-parse.
    """
    return Text(str(value))


def run_cli_command(action: Callable[[], T], *, debug: bool) -> T:
    """The single error boundary every CLI command routes through.

    Catches only :class:`AgenticEvalkitError`, prints its stable
    ``[code] message`` to stdout, and exits with the exit code
    :func:`_exit_code_for_error` maps it to. Printed to ``console`` (stdout)
    rather than ``err_console`` deliberately: an error always terminates the
    command before any success payload would print, so there is never a
    case where both share one stream, and a caller scripting against this
    CLI's output (including ``--format json`` success payloads) can always
    find the one message that explains a nonzero exit in the same stream.
    Everything else propagates unchanged -- with ``debug=True`` that means a
    normal Python traceback; with ``debug=False`` a caller-facing command
    should have already turned every foreseeable failure into a typed error
    before reaching this boundary, so an unexpected exception here is
    treated as a genuine bug rather than something this function should
    mask.
    """
    try:
        return action()
    except AgenticEvalkitError as error:
        # error.code/message are dynamic content, not markup (see
        # safe_text): a code like "dataset_not_found" wrapped in literal
        # brackets, or a message containing "[...]", must print literally
        # rather than being parsed -- and silently dropped -- as an
        # unrecognized Rich style tag.
        prefix = Text("error ", style="bold red")
        body = safe_text(f"[{error.code}] {error.message}")
        console.print(Text.assemble(prefix, body))
        if debug:
            raise
        raise typer.Exit(code=int(_exit_code_for_error(error))) from None


def print_output(payload: object, *, format_: str) -> None:
    """Print ``payload`` as JSON (``--format json``) or let the caller
    render a Rich table for the default ``table`` format.

    Only handles the JSON branch: callers needing a table pass
    ``format_="table"`` and render their own ``rich.table.Table`` before or
    instead of calling this, since table shape is command-specific.
    """
    if format_ == "json":
        console.print_json(json.dumps(payload, sort_keys=True, default=str))


app = typer.Typer(no_args_is_help=True, help="Evaluate agentic systems with reproducible evidence.")


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[bool, typer.Option("--version", help="Show the installed version.")] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
