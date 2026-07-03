"""``agentic-evalkit datasets ...``: curated/search/inspect/preview/pull.

Every command in this module goes through the same
:class:`~agentic_evalkit.datasets.catalog.DatasetCatalog` a Python caller
would use (design §11.2): there is no CLI-only shortcut path to a provider.
``curated`` works fully offline (it only reads the built-in preset table);
``search``, ``inspect``, and ``preview`` build a real catalog (with the
content-addressed cache backing ``--offline``) and display the resolved
revision/config/split the same way a manifest run would pin them. ``pull``
writes an immutable cache entry for one exact page -- it is a snapshot, not
a "sync to latest" operation, matching design §6.3's cache-identity model.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any, TypeVar, cast

import httpx
import typer
from huggingface_hub import HfApi
from rich.table import Table

from agentic_evalkit.cli.app import app, console, print_output, run_cli_command, safe_text
from agentic_evalkit.datasets.base import DatasetProvider
from agentic_evalkit.datasets.cache import DatasetCache
from agentic_evalkit.datasets.catalog import DatasetCatalog
from agentic_evalkit.datasets.huggingface import HuggingFaceDatasetProvider
from agentic_evalkit.datasets.local import LocalDatasetProvider
from agentic_evalkit.datasets.presets import BUILTIN_PRESETS
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.models import DatasetRef, ResolvedDataset, SamplePage, SearchPage

datasets_app = typer.Typer(help="Discover, inspect, and preview datasets.")
app.add_typer(datasets_app, name="datasets")

T = TypeVar("T")

_CACHE_DIR_NAME = "agentic-evalkit"


def default_cache_dir() -> Path:
    """Return the platform user-cache directory for agentic-evalkit (design §6.3).

    Stdlib-only, cross-platform resolution (no third-party cache-dir
    dependency): honors ``AGENTIC_EVALKIT_CACHE_DIR`` first (mainly for
    tests and CI isolation), then the platform convention -- ``%LOCALAPPDATA%``
    on Windows, ``$XDG_CACHE_HOME`` or ``~/.cache`` elsewhere.
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
    """Build a real ``DatasetCatalog`` wired to the local + Hugging Face providers.

    Uses :func:`default_cache_dir` (design §6.3's standalone default) so
    ``--offline`` runs can serve exact previously-cached pages.
    """
    del offline  # DatasetCatalog itself takes offline per-call, not at construction.
    cache = DatasetCache(default_cache_dir())
    client = httpx.AsyncClient(timeout=30.0)
    # HfApi structurally satisfies datasets.huggingface's private _HubClient
    # protocol (verified at runtime by that module's own tests via
    # @runtime_checkable isinstance checks) but mypy cannot prove it: HfApi's
    # real methods use enumerated keyword-only parameters where _HubClient's
    # protocol methods use **kwargs. _HubClient itself is private
    # (module-internal, not exported) so it cannot be imported here to
    # annotate against directly -- cast through Any, mirroring the identical
    # cast datasets/huggingface.py's own HuggingFaceDatasetProvider.create()
    # uses internally for this exact same mismatch.
    hf_provider = HuggingFaceDatasetProvider(client=client, hub=cast("Any", HfApi()))
    local_provider = LocalDatasetProvider(allowed_roots=(Path.cwd(),))
    # Both providers satisfy DatasetProvider structurally at runtime
    # (confirmed via isinstance against the @runtime_checkable protocol in
    # Task 5/6's own test suites); mypy's dict-literal check does not always
    # narrow concrete-class values to a Protocol type even with this
    # annotated dict, so each value is cast explicitly rather than left to
    # infer incorrectly as the narrower concrete-class union.
    providers: dict[str, DatasetProvider] = {
        "local": cast(DatasetProvider, local_provider),
        "huggingface": cast(DatasetProvider, hf_provider),
    }
    # This CLI supplies the genuine "local"/"huggingface" built-in providers
    # itself (it is not loading third-party plugins here), so the
    # collision guard that stops a *plugin* from shadowing a built-in name
    # does not apply -- pass an empty reserved-name tuple rather than
    # tripping PluginCompatibilityError on our own built-ins.
    return DatasetCatalog(providers=providers, cache=cache, builtin_provider_names=())


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
    # Name/Dataset/Adapter/Grader are lookup keys a user copies verbatim
    # into other commands (e.g. "agentic-evalkit init --preset <name>"), so
    # they must never truncate even on a narrow terminal; only the
    # free-text Readiness column is allowed to wrap. Every cell is dynamic,
    # provider/preset-derived text, so every cell is wrapped with
    # safe_text to render literally rather than be re-parsed as markup.
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
        return _run_async(lambda: catalog.search(query, provider=provider, limit=limit))

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
        return _run_async(lambda: catalog.resolve(ref))

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
            resolved = await catalog.resolve(ref)
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
    """Resolve and cache one exact page as an immutable cache entry.

    ``pull`` records a snapshot at the resolved revision; it never means
    "keep this dataset up to date" after the call returns (design §6.3).
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
