"""``agentic-evalkit datasets ...``: curated/search/inspect/preview/pull.

Every command in this module goes through the same
:class:`~agentic_evalkit.datasets.catalog.DatasetCatalog` class that a
Python caller using this as a library (rather than the CLI) would use
(design §11.2) -- there's no separate, CLI-only shortcut that reaches a
dataset provider directly. ``curated`` works fully offline because it only
reads a built-in, hardcoded table of presets -- no network call involved.
``search``, ``inspect``, and ``preview`` build a real catalog object, backed
by a cache that's keyed by a hash of its contents (which is what makes
``--offline`` mode possible), and then display exactly which dataset
revision/config/split got resolved -- the same exact, fixed identifiers a
manifest-driven eval run would lock in and use. ``pull`` writes one exact
page of data into the cache as a permanent, unchangeable entry -- it's a
one-time snapshot, never a "keep this updated" operation, matching design
§6.3's model where a cache entry's identity is fixed forever once written.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypeVar, cast

import httpx
import typer
from huggingface_hub import HfApi
from rich.table import Table

from agentic_evalkit.cli.app import app, console, print_output, run_cli_command, safe_text
from agentic_evalkit.datasets.cache import DatasetCache
from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.huggingface import HuggingFaceDatasetProvider
from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.datasets.presets import BUILTIN_PRESETS
from agentic_evalkit.datasets.resolution_cache import ResolutionCache
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.models import DatasetRef, ResolvedDataset, SamplePage, SearchPage

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from agentic_evalkit.datasets.base import DatasetProvider

datasets_app = typer.Typer(help="Discover, inspect, and preview datasets.")
app.add_typer(datasets_app, name="datasets")

T = TypeVar("T")

_CACHE_DIR_NAME = "agentic-evalkit"
#: Subdirectory of the cache root set aside for the "resolution" cache
#: (ADR-0011) -- this caches the answer to "which exact revision/config/
#: split does this dataset reference currently point to?", kept separate
#: from the page cache below, which stores the actual downloaded data (split
#: across many hash-prefixed subdirectories so no single directory ends up
#: with too many files). Keeping the two caches in physically separate
#: directories means their entries can never accidentally collide.
_RESOLUTION_CACHE_SUBDIR = "resolutions"


def default_cache_dir() -> Path:
    """Return the platform's standard user-cache directory for agentic-evalkit (design §6.3).

    Resolved using only the Python standard library, with no third-party
    "find the cache dir" package. Honors the ``AGENTIC_EVALKIT_CACHE_DIR``
    environment variable first (mainly so tests and CI can point this
    somewhere isolated), then falls back to each platform's own convention:
    ``%LOCALAPPDATA%`` on Windows, or ``$XDG_CACHE_HOME``/``~/.cache`` on
    Linux/macOS.
    """
    override = os.environ.get("AGENTIC_EVALKIT_CACHE_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / _CACHE_DIR_NAME
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / _CACHE_DIR_NAME


def parse_dataset_locator(locator: str) -> DatasetRef:
    """Parse a ``<provider>:<dataset_id>[#config/split]`` CLI locator.

    Accepts ``hf:`` and ``huggingface:`` as aliases for the ``huggingface``
    provider, and ``local:`` for the ``local`` provider (design §11.1's
    ``datasets inspect hf:princeton-nlp/SWE-bench_Verified`` example).
    ``config``/``split`` are optional and, when given, are separated from
    the dataset ID with ``#config/split``.

    Raises:
        ManifestValidationError: ``locator`` has no ``provider:`` prefix.
    """
    if ":" not in locator:
        raise ManifestValidationError(
            message=(
                f"dataset locator {locator!r} must be of the form "
                "'<provider>:<dataset_id>', e.g. 'hf:openai/gsm8k'"
            ),
            context={"locator": locator},
        )
    prefix, _, remainder = locator.partition(":")
    provider = {"hf": "huggingface", "huggingface": "huggingface", "local": "local"}.get(
        prefix, prefix
    )
    dataset_id, _, config_split = remainder.partition("#")
    config: str | None = None
    split: str | None = None
    if config_split:
        config, _, split = config_split.partition("/")
        config = config or None
        split = split or None
    return DatasetRef(provider=provider, dataset_id=dataset_id, config=config, split=split)


def build_catalog(*, offline: bool) -> DatasetCatalog:
    """Build a real ``DatasetCatalog``, connected to the local and Hugging Face dataset providers.

    Uses :func:`default_cache_dir` (the same default the library uses on its
    own, per design §6.3) so that ``--offline`` runs can still serve pages
    that were already downloaded and cached from an earlier, online run. It
    also builds a
    :class:`~agentic_evalkit.datasets.resolution_cache.ResolutionCache`,
    rooted in that same cache directory's ``resolutions/`` subdirectory
    (ADR-0011). That second cache remembers *which exact version* a dataset
    reference resolved to, so an ``--offline`` run can reuse a resolution
    that succeeded online at least once before (e.g. via ``datasets pull``)
    without needing to contact the provider again just to re-confirm it.

    This function accepts an ``offline`` parameter (rather than dropping it)
    only to keep every caller's function signature uniform.
    ``DatasetCatalog`` itself takes ``offline`` separately on each
    individual method call, never once at construction time (ADR-0010; see
    the module docstring of ``agentic_evalkit.datasets.catalog`` for why) --
    so this function has nothing to actually *do* with the flag beyond
    accepting and holding it. A past bug here was never that this parameter
    existed pointlessly: it was that every caller below was silently
    forgetting to pass its own ``offline`` value through to the actual
    ``DatasetCatalog`` method calls it made, so ``--offline`` silently did
    nothing in places. Each command function below now passes that value
    through explicitly.
    """
    cache = DatasetCache(default_cache_dir())
    resolution_cache = ResolutionCache(default_cache_dir() / _RESOLUTION_CACHE_SUBDIR)
    client = httpx.AsyncClient(timeout=30.0)
    # HfApi has all the right methods/behavior to satisfy
    # datasets.huggingface's private _HubClient Protocol ("structural"
    # typing: Python doesn't require HfApi to explicitly declare that it
    # implements _HubClient, just to have matching methods) -- and that
    # module's own tests confirm this at runtime with an isinstance() check
    # against the @runtime_checkable protocol. But mypy can't verify it
    # statically: HfApi's actual methods declare their keyword arguments by
    # name, while _HubClient's protocol methods are declared with a generic
    # **kwargs. We also can't just annotate against _HubClient directly,
    # because it's private to that other module (not exported), so it can't
    # be imported here. So we cast through Any instead -- mirroring the
    # identical workaround datasets/huggingface.py's own
    # HuggingFaceDatasetProvider.create() uses internally for this exact
    # same mismatch.
    hf_provider = HuggingFaceDatasetProvider(client=client, hub=cast("Any", HfApi()))
    local_provider = LocalDatasetProvider(allowed_roots=(Path.cwd(),))
    # Both providers satisfy the DatasetProvider Protocol structurally at
    # runtime (confirmed via isinstance() against the @runtime_checkable
    # protocol, in Task 5/6's own test suites). But even with the type
    # annotation on this dict, mypy doesn't always widen concrete-class
    # values up to the Protocol type on its own -- left alone, it would
    # infer this dict's value type as the narrower "either
    # LocalDatasetProvider or HuggingFaceDatasetProvider" union instead of
    # DatasetProvider. So each value below is cast explicitly to avoid that
    # wrong, narrower inference.
    providers: dict[str, DatasetProvider] = {
        "local": cast("DatasetProvider", local_provider),
        "huggingface": cast("DatasetProvider", hf_provider),
    }
    # This function is registering the CLI's own genuine "local" and
    # "huggingface" providers -- it is not loading any third-party plugins
    # here. DatasetCatalog has a safety check elsewhere that normally stops a
    # third-party plugin from registering a provider under a name that
    # collides with ("shadows") a built-in one. That check doesn't apply to
    # this case, since these ARE the built-ins, not a plugin trying to
    # impersonate one -- so we pass an empty tuple of "reserved" names here,
    # rather than accidentally tripping that same safety check
    # (PluginCompatibilityError) against our own code.
    return DatasetCatalog(
        providers=providers,
        cache=cache,
        resolution_cache=resolution_cache,
        builtin_provider_names=(),
    )


def _run_async(coroutine_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run one coroutine to completion; the CLI is not itself async."""
    return asyncio.run(coroutine_factory())


@datasets_app.command("curated")
def curated(
    format_: Annotated[
        str, typer.Option("--format", help="Output format: table or json.")
    ] = "table",
) -> None:
    """List the built-in, verified dataset presets. Works fully offline."""
    presets = list(BUILTIN_PRESETS.values())
    if format_ == "json":
        print_output(
            [
                {
                    "name": preset.name,
                    "description": preset.description,
                    "provider": preset.ref.provider,
                    "dataset_id": preset.ref.dataset_id,
                    "config": preset.ref.config,
                    "split": preset.ref.split,
                    "adapter": preset.adapter,
                    "grader": preset.grader,
                    "readiness": preset.readiness,
                    "required_capabilities": list(preset.required_capabilities),
                }
                for preset in presets
            ],
            format_=format_,
        )
        return
    table = Table(title="Curated dataset presets")
    # Name/Dataset/Adapter/Grader are values a user is meant to copy exactly
    # into other commands (e.g. "agentic-evalkit init --preset <name>"), so
    # none of them may ever get cut off, even in a narrow terminal window --
    # only the free-form, human-readable Readiness column is allowed to wrap
    # onto multiple lines. Every cell's text ultimately comes from a
    # provider or preset (it's not a literal string we wrote ourselves), so
    # every cell is wrapped in safe_text() -- this displays the text exactly
    # as-is, instead of Rich trying to interpret something like "[bold]"
    # inside it as a markup tag.
    table.add_column("Name", no_wrap=True)
    table.add_column("Dataset", no_wrap=True)
    table.add_column("Readiness")
    table.add_column("Adapter", no_wrap=True)
    table.add_column("Grader", no_wrap=True)
    for preset in presets:
        table.add_row(
            safe_text(preset.name),
            safe_text(preset.ref.dataset_id),
            safe_text(preset.readiness),
            safe_text(preset.adapter),
            safe_text(preset.grader),
        )
    console.print(table)


@datasets_app.command("search")
def search(
    query: str,
    provider: Annotated[
        str, typer.Option("--provider", help="Provider to search.")
    ] = "huggingface",
    limit: Annotated[int, typer.Option("--limit", help="Maximum number of results.")] = 20,
    format_: Annotated[
        str, typer.Option("--format", help="Output format: table or json.")
    ] = "table",
    offline: Annotated[bool, typer.Option("--offline", help="Never contact a provider.")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Search a provider's dataset catalog."""

    def _action() -> SearchPage:
        catalog = build_catalog(offline=offline)
        return _run_async(
            lambda: catalog.search(query, provider=provider, limit=limit, offline=offline)
        )

    page = run_cli_command(_action, debug=debug)
    if format_ == "json":
        print_output(page.model_dump(mode="json"), format_=format_)
        return
    table = Table(title=safe_text(f"Search results for {query!r}"))
    table.add_column("Dataset")
    table.add_column("Revision")
    table.add_column("Gated")
    table.add_column("Downloads")
    for hit in page.hits:
        table.add_row(
            safe_text(hit.dataset_id),
            safe_text(hit.revision or ""),
            safe_text(hit.gated),
            safe_text(hit.downloads or ""),
        )
    console.print(table)


@datasets_app.command("inspect")
def inspect(
    locator: str,
    format_: Annotated[
        str, typer.Option("--format", help="Output format: table or json.")
    ] = "table",
    offline: Annotated[bool, typer.Option("--offline", help="Only use exact cached data.")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Resolve a dataset locator (e.g. 'hf:openai/gsm8k') and show its metadata."""
    ref = run_cli_command(lambda: parse_dataset_locator(locator), debug=debug)

    def _action() -> ResolvedDataset:
        catalog = build_catalog(offline=offline)
        return _run_async(lambda: catalog.resolve(ref, offline=offline))

    resolved = run_cli_command(_action, debug=debug)
    if format_ == "json":
        print_output(resolved.model_dump(mode="json"), format_=format_)
        return
    table = Table(title=safe_text(resolved.dataset_id))
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("revision", safe_text(resolved.revision))
    table.add_row("config", safe_text(resolved.config or ""))
    table.add_row("split", safe_text(resolved.split or ""))
    table.add_row(
        "row_count", safe_text(resolved.row_count if resolved.row_count is not None else "")
    )
    table.add_row("license", safe_text(resolved.license or ""))
    table.add_row("gated", safe_text(resolved.gated))
    console.print(table)


@datasets_app.command("preview")
def preview(
    locator: str,
    config: Annotated[str | None, typer.Option("--config", help="Dataset config.")] = None,
    split: Annotated[str | None, typer.Option("--split", help="Dataset split.")] = None,
    offset: Annotated[int, typer.Option("--offset", help="Row offset.")] = 0,
    limit: Annotated[int, typer.Option("--limit", help="Number of rows to preview.")] = 3,
    format_: Annotated[
        str, typer.Option("--format", help="Output format: table or json.")
    ] = "table",
    offline: Annotated[bool, typer.Option("--offline", help="Only use exact cached data.")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Resolve a dataset and preview a page of raw source records."""
    base_ref = run_cli_command(lambda: parse_dataset_locator(locator), debug=debug)
    ref = base_ref.model_copy(
        update={"config": config or base_ref.config, "split": split or base_ref.split}
    )

    def _action() -> tuple[ResolvedDataset, SamplePage]:
        catalog = build_catalog(offline=offline)

        async def _run() -> tuple[ResolvedDataset, SamplePage]:
            # Both the resolve() call and the page-preview call below need
            # to respect --offline. Without this, resolve() -- which may
            # need to make a live network call to a provider like Hugging
            # Face to figure out things like the latest revision -- would
            # still go over the network even when --offline was passed,
            # even though the actual page-fetch right after it already
            # correctly refused to make a live call (serving from cache
            # instead). That made `preview --offline` only halfway offline:
            # one of its two steps could still silently reach the network
            # (ADR-0010).
            resolved = await catalog.resolve(ref, offline=offline)
            page = await catalog.preview(ref, resolved, offset=offset, limit=limit, offline=offline)
            return resolved, page

        return _run_async(_run)

    resolved, page = run_cli_command(_action, debug=debug)
    if format_ == "json":
        print_output(
            {
                "resolved_dataset": resolved.model_dump(mode="json"),
                "page": page.model_dump(mode="json"),
            },
            format_=format_,
        )
        return
    table = Table(title=safe_text(f"{resolved.dataset_id} @ {resolved.revision[:12]}"))
    table.add_column("Row ID")
    table.add_column("Data")
    for record in page.records:
        table.add_row(safe_text(record.row_id), safe_text(record.data))
    console.print(table)


@datasets_app.command("pull")
def pull(
    locator: str,
    config: Annotated[str | None, typer.Option("--config", help="Dataset config.")] = None,
    split: Annotated[str | None, typer.Option("--split", help="Dataset split.")] = None,
    offset: Annotated[int, typer.Option("--offset", help="Row offset.")] = 0,
    limit: Annotated[int, typer.Option("--limit", help="Number of rows to cache.")] = 100,
    format_: Annotated[
        str, typer.Option("--format", help="Output format: table or json.")
    ] = "table",
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on error.")] = False,
) -> None:
    """Resolve a dataset locator and cache one exact page of it, permanently.

    ``pull`` records a frozen snapshot at whatever revision it resolves to
    right now. It does not mean "keep this dataset up to date" after the
    call returns -- the cached copy will not automatically refresh later
    just because the dataset changes upstream (design §6.3).
    """
    base_ref = run_cli_command(lambda: parse_dataset_locator(locator), debug=debug)
    ref = base_ref.model_copy(
        update={"config": config or base_ref.config, "split": split or base_ref.split}
    )

    def _action() -> tuple[ResolvedDataset, SamplePage]:
        catalog = build_catalog(offline=False)

        async def _run() -> tuple[ResolvedDataset, SamplePage]:
            resolved = await catalog.resolve(ref)
            page = await catalog.preview(ref, resolved, offset=offset, limit=limit)
            return resolved, page

        return _run_async(_run)

    resolved, page = run_cli_command(_action, debug=debug)
    payload = {
        "dataset_id": resolved.dataset_id,
        "revision": resolved.revision,
        "config": resolved.config,
        "split": resolved.split,
        "cached_rows": len(page.records),
    }
    if format_ == "json":
        print_output(payload, format_=format_)
        return
    console.print(
        safe_text(
            f"Cached {len(page.records)} rows of "
            f"{resolved.dataset_id}@{resolved.revision[:12]} "
            f"(config={resolved.config}, split={resolved.split})"
        ),
        soft_wrap=True,
    )
