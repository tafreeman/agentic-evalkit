"""The root Typer application, plus the CLI-wide rules for exit codes and error handling.

Design doc, section 11.1.

Every subcommand in this CLI depends on this module for three things:

1. The single root Typer app itself (``Typer(no_args_is_help=True)``), which
   ``src/agentic_evalkit/cli/__init__.py`` re-exports as ``app``. The
   ``datasets`` subcommands are attached to it as a named group, and
   ``doctor``/``init``/``validate``/``run`` are registered directly on it
   (this is the exact command list from the implementation plan, Task 14
   Step 4).
2. The exit-code policy -- the number this process hands back to the shell
   in each situation: 0 success, 2 invalid input/manifest, 3 a required
   capability is missing, 4 a provider or dataset could not be resolved or
   reached (including "dataset not found"), 5 the evaluation ran but hit an
   infrastructure problem along the way, 130 the user cancelled it.
3. :func:`run_cli_command`, the single choke point that every command's
   logic is run through, so errors get handled the same way everywhere
   instead of separately in each command. It only catches
   :class:`AgenticEvalkitError` (this project's own base error class),
   prints that error's stable ``.code`` and message, and exits with the
   matching code from the policy above. Any other, unanticipated exception
   is deliberately left to propagate as a normal Python traceback -- shown
   to the user only when they passed ``--debug`` (each command checks for
   that flag itself before calling into framework code that might raise
   something unexpected); without ``--debug``, Typer's own default error
   handling still prints the traceback, but by that point a command is
   expected to have already converted every failure it could anticipate
   into one of this project's typed errors, so hitting this fallback at all
   signals a genuine bug rather than a case this function is meant to hide.
"""

from __future__ import annotations

import json
from enum import IntEnum
from typing import TYPE_CHECKING, Annotated, TypeVar

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
    IncompatibleRuns,
    ManifestValidationError,
    OfflineCacheMiss,
    PluginCompatibilityError,
    UnsafeCodeRequired,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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

#: Minimum width to render tables at. Rich (the library that draws our
#: tables) falls back to a default width of 80 columns whenever stdout
#: isn't a real terminal -- for example, when output is piped somewhere,
#: captured in CI logs, or captured by Typer's CliRunner test harness. 80
#: columns is too narrow to show a full dataset ID next to its adapter and
#: grader names without cutting off text a user needs to copy exactly.
#: A genuinely narrow terminal window is still respected as-is: this
#: constant only raises the floor used for that non-terminal fallback case.
_MIN_CONSOLE_WIDTH = 120

#: Stdout for normal command output (tables, JSON payloads, success messages).
console = Console(width=_MIN_CONSOLE_WIDTH if not Console().is_terminal else None)
#: Stderr for error/diagnostic output, kept separate so ``--format json``
#: stdout always parses cleanly even when a warning was also printed.
err_console = Console(stderr=True)


class ExitCode(IntEnum):
    """The CLI's exit codes (plan Task 14 Step 4).

    These are a stable contract: any script or CI job calling this CLI can
    rely on these numbers staying the same across releases.
    """

    SUCCESS = 0
    INVALID_INPUT = 2
    MISSING_CAPABILITY = 3
    PROVIDER_ERROR = 4
    INFRASTRUCTURE_ERROR = 5
    CANCELLED = 130


class OutputFormat(str):
    """The allowed values for a ``--format`` option, defined as plain strings
    (rather than, say, a Python ``Enum``) because that's the type Typer
    expects when it validates a command-line option's choices.
    """

    TABLE = "table"
    JSON = "json"


#: Maps each of this project's specific error types to the exit code that
#: type should produce. Every key here is a distinct subclass of
#: ``AgenticEvalkitError`` (this project's shared base error class) -- the
#: base class itself is deliberately left out of this dict.
#: :func:`_exit_code_for_error` checks a raised error against every entry
#: here looking for the most specific match; only if none of these specific
#: types match does it fall back to one default exit code for "any other
#: AgenticEvalkitError." That fallback (defined right below) is what stands
#: in for the base class here, so there is no ordering to worry about among
#: the specific entries above.
_ERROR_EXIT_CODES: dict[type[AgenticEvalkitError], ExitCode] = {
    ManifestValidationError: ExitCode.INVALID_INPUT,
    DatasetSchemaMismatch: ExitCode.INVALID_INPUT,
    UnsafeCodeRequired: ExitCode.INVALID_INPUT,
    # A comparison of two incompatible runs is an invalid *choice of inputs*
    # by the user (they picked two runs that cannot be meaningfully
    # compared), not a provider or infrastructure failure -- so it exits 2
    # (invalid input) with every mismatch listed, per Task 14 Step 10.
    IncompatibleRuns: ExitCode.INVALID_INPUT,
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
    """Look up the exit code for ``error``, using the most specific match available.

    Falls back to :attr:`ExitCode.INFRASTRUCTURE_ERROR` (5) for any
    ``AgenticEvalkitError`` subclass that isn't explicitly listed in
    ``_ERROR_EXIT_CODES`` above -- for example, ``TargetFailure``,
    ``TargetTimeout``, or ``GraderError`` raised while a run was otherwise
    in progress. Those all represent a run that successfully started and
    reached execution, but then hit an infrastructure-level problem partway
    through, which is exactly what exit code 5 ("evaluation completed with
    infrastructure errors") means.
    """
    for error_type, code in _ERROR_EXIT_CODES.items():
        if isinstance(error, error_type):
            return code
    return ExitCode.INFRASTRUCTURE_ERROR


def safe_text(value: object) -> Text:
    """Wrap dynamic text so Rich prints it as-is, instead of trying to interpret it as formatting.

    Rich is the library this CLI uses to print colored/styled output, and it
    has its own mini-syntax for that: text like ``[red]`` inside a string
    tells Rich "start red text here." That's convenient for the fixed
    messages we write ourselves, but risky for any text that came from
    somewhere else -- a dataset ID, an error message, an error code, a
    remediation hint, a value pulled out of JSON, or table-cell data. Any of
    those can innocently contain a real ``[...]``-shaped chunk that was
    never meant as a style tag: an error code like ``dataset_not_found``
    that this project happens to print inside brackets, a pip "extra" name
    like ``agentic-evalkit[swebench]``, or just arbitrary text from
    somewhere upstream. If you hand text like that to ``console.print`` or
    ``Table.add_row`` as a plain ``str``, Rich reads the ``[...]`` part as a
    style tag it doesn't recognize and quietly deletes it from the output --
    it doesn't raise an error, it just silently mangles exactly the
    identifier a user was trying to read or copy. Wrapping the same text in
    a :class:`rich.text.Text` object, as this function does, tells Rich to
    skip that interpretation step entirely and print the text
    character-for-character. So every place in this CLI that prints dynamic
    content calls this function first, rather than building a plain string
    or f-string that Rich would then try to re-parse.
    """
    return Text(str(value))


def run_cli_command(action: Callable[[], T], *, debug: bool) -> T:
    """The single place where every CLI command's error handling happens.

    Each command function hands its real logic to this function as the
    ``action`` argument, instead of writing its own try/except -- so error
    handling is written once, here, instead of being repeated (and
    potentially done a little differently) in every command. This function
    catches only :class:`AgenticEvalkitError` (this project's own base
    error class), prints its stable ``[code] message`` text to stdout, and
    exits the process with whatever exit code :func:`_exit_code_for_error`
    picks for that error. It prints to ``console`` (stdout) rather than
    ``err_console`` (stderr) on purpose: an error always ends the command
    before any success output would print, so the two never need to share
    one stream, and it means a caller scripting against this CLI's output
    (including a ``--format json`` success payload) can always find the one
    message explaining a nonzero exit in that same stream, without checking
    both. Any exception that is not an :class:`AgenticEvalkitError` is left
    completely alone and allowed to propagate: with ``debug=True`` that
    shows up as a normal Python traceback; with ``debug=False``, a
    well-behaved command should already have converted every failure it
    could reasonably anticipate into one of this project's typed errors
    before its logic ever reaches this function -- so an unexpected
    exception getting here at all is treated as a genuine bug in that
    command, not a case this function is meant to quietly paper over.
    """
    try:
        return action()
    except AgenticEvalkitError as error:
        # error.code and error.message are dynamic text, not something we
        # want Rich to interpret as style markup (see the safe_text
        # docstring above for why). A code like "dataset_not_found" ends up
        # wrapped in literal brackets right here, and either it or the
        # message could contain other "[...]"-shaped text -- that needs to
        # print exactly as written, not get parsed (and silently deleted)
        # as an unrecognized Rich style tag.
        prefix = Text("error ", style="bold red")
        body = safe_text(f"[{error.code}] {error.message}")
        console.print(Text.assemble(prefix, body))
        if debug:
            raise
        raise typer.Exit(code=int(_exit_code_for_error(error))) from None


def print_output(payload: object, *, format_: str) -> None:
    """Print ``payload`` as JSON when the user asked for ``--format json``; for
    the default ``table`` format, do nothing here and let the calling
    command render its own Rich table instead.

    This function only implements the JSON branch. A command that needs a
    table still calls this function, but when ``format_="table"`` it is a
    no-op, so that command is responsible for building and printing its own
    ``rich.table.Table``. There's no single generic table shape this
    function could render on every command's behalf, since the columns a
    table needs are different for each command.
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
