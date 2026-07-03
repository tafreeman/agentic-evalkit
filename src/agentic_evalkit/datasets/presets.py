"""Verified built-in dataset presets (design §6.2, plan Task 7 Step 3).

A ``DatasetPreset`` pins the exact provider, dataset ID, config, split,
adapter, and grader a caller gets from a short preset name, plus a
``readiness`` label distinguishing presets that are fully runnable today from
presets that are discoverable/previewable/projectable but whose authoritative
grading requires an optional capability (e.g. the ``swebench`` harness
extra).

``BUILTIN_PRESETS`` is built by :func:`_build_builtin_presets`, which raises
``ValueError`` at *import time* if two presets ever share a name, so a
duplicate preset name can never silently shadow another and reach a caller.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from agentic_evalkit.models import DatasetRef
from agentic_evalkit.models.base import FrozenModel

__all__ = ["BUILTIN_PRESETS", "DatasetPreset"]


class DatasetPreset(FrozenModel):
    """A named, pinned dataset/adapter/grader combination (design §6.2).

    ``required_capabilities`` names optional-extra capabilities (e.g.
    ``"swebench"``) that authoritative grading for this preset needs but
    that discovery, preview, and projection do not; an empty tuple means no
    optional capability is required for any readiness level this preset
    reaches.
    """

    name: str
    description: str
    ref: DatasetRef
    adapter: str
    grader: str
    readiness: str
    required_capabilities: tuple[str, ...] = ()


def _build_builtin_presets(*presets: DatasetPreset) -> MappingProxyType[str, DatasetPreset]:
    """Freeze ``presets`` into an immutable, name-keyed mapping.

    Raises:
        ValueError: Two presets share the same ``name``. Raised eagerly at
            import time (this function is called once, at module scope)
            rather than deferred to lookup time, so a duplicate preset name
            fails the very first import of this module instead of silently
            shadowing an earlier preset.
    """
    by_name: dict[str, DatasetPreset] = {}
    for preset in presets:
        if preset.name in by_name:
            raise ValueError(f"duplicate dataset preset name {preset.name!r}")
        by_name[preset.name] = preset
    return MappingProxyType(by_name)


_GSM8K_PRESET: Final[DatasetPreset] = DatasetPreset(
    name="gsm8k",
    description="Grade-school math word problems with normalized exact-answer grading.",
    ref=DatasetRef(
        provider="huggingface",
        dataset_id="openai/gsm8k",
        config="main",
        split="test",
    ),
    adapter="gsm8k@1",
    grader="normalized-exact@1",
    readiness="runnable",
)

_SWE_BENCH_VERIFIED_PRESET: Final[DatasetPreset] = DatasetPreset(
    name="swe-bench-verified",
    description=(
        "Real-world GitHub issue resolution tasks. Discoverable, previewable, and "
        "projectable out of the box; authoritative resolved/unresolved grading "
        "requires the optional swebench harness extra."
    ),
    ref=DatasetRef(
        provider="huggingface",
        dataset_id="princeton-nlp/SWE-bench_Verified",
        config="default",
        split="test",
    ),
    adapter="swebench-verified@1",
    grader="swebench-harness@1",
    readiness="prediction_export",
    required_capabilities=("swebench",),
)

BUILTIN_PRESETS: Final[MappingProxyType[str, DatasetPreset]] = _build_builtin_presets(
    _GSM8K_PRESET,
    _SWE_BENCH_VERIFIED_PRESET,
)
