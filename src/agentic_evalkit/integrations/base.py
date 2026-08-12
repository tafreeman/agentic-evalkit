"""Shared plumbing every host-platform export goes through (ADR-0022).

An "export" here means handing a finished run to somebody else's system --
an MLflow tracking server, a Langfuse project -- so the validity evidence
this package produces lands where a team already works, instead of only in
a local report file. That direction of travel is the whole point of this
subpackage, and it is also what makes it the riskiest surface in the
codebase: a report file stays on the machine that wrote it, while an export
leaves it. Everything below exists to make the leaving safe and boring.

Three rules are enforced here rather than restated in each exporter:

* **Redaction happens exactly once, and it happens here.** Every exporter
  calls :func:`redact_for_export` as its first act and works only with what
  comes back. Because that is a single named function rather than a habit,
  ``tests/contract/test_integration_redaction.py`` can count the calls and
  fail CI if a future exporter forgets -- the same tripwire design the
  reporters use with ``REDACTION_ROUTED_FORMATS``.
* **The host library is never imported at module scope.** ``import
  agentic_evalkit.integrations.mlflow`` must succeed on a machine with no
  MLflow installed, so that ``from agentic_evalkit import integrations``
  stays free and the optional extra stays genuinely optional (ADR-0009).
  :func:`require_dependency` is the one place that import happens, and the
  one place that turns a missing package into an explanation.
* **This package is a guest.** Nothing here is allowed to become a required
  dependency of the rest of the codebase, and nothing in the rest of the
  codebase imports from here. The dependency arrow points one way: outward.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, TypeVar

# Re-exported, not defined here. All three moved down to graders.calibration
# in ADR-0024: the ``calibrate`` command needs the same authority decision
# this subpackage carries into a host platform, and this subpackage's own
# dependency arrow points outward only (ADR-0022), so a CLI command may not
# import from it. Keeping the names importable from here means every
# existing caller -- and this module's own exporters -- are unaffected by
# where the definition physically lives.
from agentic_evalkit.errors import IntegrationUnavailable
from agentic_evalkit.graders.calibration import (
    AuthorityLevel as AuthorityLevel,
)
from agentic_evalkit.graders.calibration import (
    JudgeAuthority as JudgeAuthority,
)
from agentic_evalkit.graders.calibration import (
    judge_authority as judge_authority,
)
from agentic_evalkit.reporters import apply_redaction

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from types import ModuleType

    from agentic_evalkit.models import EvalRunResult
    from agentic_evalkit.reporters import RedactionPolicy

_T = TypeVar("_T")

__all__ = [
    "AuthorityLevel",
    "JudgeAuthority",
    "judge_authority",
    "redact_for_export",
    "require_dependency",
    "run_blocking",
]


def require_dependency(module_name: str, *, extra: str) -> ModuleType:
    """Import a host platform's client library, or explain how to install it.

    Called at the top of every exported function rather than at module
    scope, so importing this package never requires the host library to be
    present. The cost of a repeated import is a dictionary lookup in
    ``sys.modules``; the benefit is that the extra stays optional in the
    real sense -- a user who wants the Langfuse bridge does not have to
    install MLflow to get it.

    Args:
        module_name: The import root to load, e.g. ``"mlflow"``.
        extra: The name of this package's optional extra that provides it,
            used to build the install hint in the error message.

    Returns:
        The imported module.

    Raises:
        IntegrationUnavailable: If the module cannot be imported. The
            message names the exact ``pip install`` line that fixes it,
            because "No module named 'mlflow'" arriving from three frames
            inside an exporter tells a user what broke but not what to do.
    """
    # Imported here rather than at module scope purely for symmetry with the
    # rule this function exists to enforce; importlib itself is stdlib and
    # always present.
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise IntegrationUnavailable(
            message=(
                f"the {extra!r} integration needs the {module_name!r} package, which is "
                f"not installed; install it with: pip install 'agentic-evalkit[{extra}]'"
            ),
            context={"module": module_name, "extra": extra},
        ) from exc


def redact_for_export(run: EvalRunResult, policy: RedactionPolicy) -> EvalRunResult:
    """Scrub secrets out of ``run`` once, before any of it leaves this machine.

    This is a thin wrapper over :func:`~agentic_evalkit.reporters.apply_redaction`
    and deliberately adds no behaviour of its own. It exists to be a *name*:
    one function that every export path calls exactly once, which a contract
    test can patch and count. Wrapping also keeps the guarantee honest in the
    other direction -- because exporters never call ``apply_redaction``
    directly, "did this exporter redact?" is answerable by reading its first
    statement rather than by auditing the whole function.

    Note the difference from the reporters, and treat it as deliberate: a
    reporter writes to a path the caller chose on a machine the caller
    already controls, so the library leaves the policy entirely to them. An
    exporter transmits to a shared server that colleagues, and often other
    teams, can read. So the exporters here default to
    :data:`~agentic_evalkit.reporters.DEFAULT_REDACTION_POLICY` instead of to
    no redaction, and a caller who genuinely wants raw output on the wire
    has to say so by passing ``RedactionPolicy()`` explicitly.

    Args:
        run: The finished run about to be exported.
        policy: What to scrub. ``RedactionPolicy()`` (no patterns) is the
            supported way to opt out.

    Returns:
        A new, redacted run. ``run`` itself is never modified (ADR-0002).
    """
    return apply_redaction(run, policy)


def run_blocking(coroutine: Coroutine[object, object, _T]) -> _T:
    """Run an async grader to completion from synchronous host-platform code.

    Every grader in this package is ``async`` (``graders.base.Grader``),
    while the scorer interfaces both MLflow and Langfuse expose are ordinary
    synchronous callables. Something has to bridge that, and the naive
    bridge -- :func:`asyncio.run` -- raises ``RuntimeError`` the moment it is
    called from a thread that already has a running event loop. That is not
    an exotic case: it is what happens when a scorer runs inside a notebook,
    inside an async web handler, or inside an evaluation harness that is
    itself async.

    So this checks first. With no loop running, :func:`asyncio.run` is
    correct and cheap. With a loop already running on this thread, the
    coroutine is handed to a short-lived worker thread that owns its own
    fresh loop, and this thread blocks on the result. The worker cannot
    deadlock against the caller's loop because it shares nothing with it.

    Args:
        coroutine: The coroutine to drive to completion.

    Returns:
        Whatever the coroutine returns.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop on this thread: the ordinary case, and the cheap path.
        return asyncio.run(coroutine)

    # A loop is already running here, so asyncio.run would refuse. Give the
    # coroutine its own thread and its own loop, and wait for it. max_workers
    # is 1 because there is exactly one piece of work; the executor is
    # created and torn down per call rather than pooled, since a scorer may
    # be invoked from arbitrary host-platform threads and a module-level
    # pool would outlive the process's interest in it.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()
