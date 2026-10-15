"""Explicit extension bundle for domain adapters and graders."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agentic_evalkit.benchmarks.base import BenchmarkAdapter
from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.graders.base import Grader

if TYPE_CHECKING:
    from agentic_evalkit.manifest import ComponentFactoryConfig


@dataclass(frozen=True)
class ComponentBundle:
    adapter: BenchmarkAdapter
    grader: Grader


@dataclass(frozen=True)
class ComponentIdentity:
    """Immutable proof of the exact local factory selected for a run."""

    source_fingerprint: str
    function_fingerprint: str
    config_fingerprint: str
    fingerprint: str

    def as_manifest_value(self) -> dict[str, str]:
        return {
            "source_fingerprint": self.source_fingerprint,
            "function_fingerprint": self.function_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class LoadedComponentBundle:
    bundle: ComponentBundle
    identity: ComponentIdentity


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def component_identity(config: ComponentFactoryConfig) -> ComponentIdentity:
    """Fingerprint the selected source file, symbol, and literal configuration."""
    return _identity_from_source(config, Path(config.path).read_bytes())


def _identity_from_source(config: ComponentFactoryConfig, source: bytes) -> ComponentIdentity:
    source_fingerprint = _sha256(source)
    function_fingerprint = _sha256(config.function.encode("utf-8"))
    config_json = json.dumps(
        config.config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    config_fingerprint = _sha256(config_json.encode("utf-8"))
    fingerprint = _sha256(
        json.dumps(
            {
                "source": source_fingerprint,
                "function": function_fingerprint,
                "config": config_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return ComponentIdentity(
        source_fingerprint=source_fingerprint,
        function_fingerprint=function_fingerprint,
        config_fingerprint=config_fingerprint,
        fingerprint=fingerprint,
    )


def load_component_bundle(config: ComponentFactoryConfig) -> LoadedComponentBundle:
    """Load only the file and function explicitly selected by the manifest.

    This executes trusted local Python, just like the callable target. It is
    never invoked by manifest validation or automatic plugin discovery.
    """
    path = Path(config.path)
    if not path.is_file() or path.suffix != ".py":
        raise ManifestValidationError(message="component factory must name an existing .py file")
    source = path.read_bytes()
    identity = _identity_from_source(config, source)
    module_name = "_evalkit_component_" + identity.source_fingerprint.removeprefix("sha256:")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ManifestValidationError(message="could not load component factory")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        # Execute exactly the bytes fingerprinted above, never an mtime-based
        # bytecode cache from a different version of this explicitly trusted file.
        exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
        factory = getattr(module, config.function, None)
        if not callable(factory):
            raise ManifestValidationError(message="component factory function is not callable")
        bundle = factory(config.config)
        if not isinstance(bundle, ComponentBundle):
            raise ManifestValidationError(message="component factory must return ComponentBundle")
        if not isinstance(bundle.adapter, BenchmarkAdapter) or not isinstance(
            bundle.grader, Grader
        ):
            raise ManifestValidationError(
                message="component bundle does not implement the public protocols"
            )
        return LoadedComponentBundle(bundle=bundle, identity=identity)
    finally:
        sys.modules.pop(module_name, None)
