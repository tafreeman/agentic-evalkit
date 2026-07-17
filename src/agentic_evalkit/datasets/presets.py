"""Built-in dataset presets, verified and ready to use (design §6.2, plan
Task 7 Step 3).

A ``DatasetPreset`` bundles together everything needed to run a known
benchmark under one short name: the exact provider, dataset ID, config,
split, adapter, and grader a caller gets just by naming the preset. It also
carries a ``readiness`` label that distinguishes presets that are fully
usable today from presets where you can browse, preview, and reformat
("project") the data out of the box, but where *authoritative* grading (the
official, trustworthy scoring for that benchmark) needs an extra, optional
piece of software installed (for example, the ``swebench`` preset needs the
optional ``swebench`` harness package to grade results).

``BUILTIN_PRESETS`` is built by calling :func:`_build_builtin_presets`,
which raises ``ValueError`` immediately, at the moment this module is first
imported, if it's ever given two presets with the same name. That way, a
duplicate preset name breaks loudly the first time the module loads,
instead of quietly letting one preset hide ("shadow") another and reach a
caller unnoticed.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from agentic_evalkit.models import ContaminationMetadata, ContaminationStatus, DatasetRef
from agentic_evalkit.models.base import FrozenModel

__all__ = ["BUILTIN_PRESETS", "DatasetPreset"]


class DatasetPreset(FrozenModel):
    """One named preset that pins together an exact dataset, adapter, and
    grader (design §6.2).

    ``required_capabilities`` lists the names of optional, extra software
    packages (for example, ``"swebench"``) that this preset's
    *authoritative* grading needs -- but that browsing, previewing, or
    reformatting ("projecting") the data do not need. An empty tuple means
    no optional package is required for anything this preset can currently
    do.

    ``contamination`` is this preset's best-effort label describing its
    "data contamination" risk -- this is the C9 provenance label from
    ADR-0013. Data contamination is the concern that a public benchmark's
    test questions (and often their answers) may have leaked into a model's
    training data, simply because the benchmark has been public and copied
    ("mirrored") in many places online for a long time; if that has
    happened, a model could score well by having memorized the answers
    rather than by actually being capable of solving the task. This label
    is informative only -- it does not block or gate anything by itself.
    Both built-in presets here are long-public, widely mirrored datasets, so
    they are honestly labeled ``SUSPECT``: before treating a score on
    either of them as evidence of real capability, someone must first check
    for train/test overlap (whether test examples show up in the model's
    training data) or run a decontamination pass (a process that filters
    out compromised examples) -- this label alone is not enough
    justification on its own.
    """

    name: str
    description: str
    ref: DatasetRef
    adapter: str
    grader: str
    readiness: str
    required_capabilities: tuple[str, ...] = ()
    contamination: ContaminationMetadata | None = None


def _build_builtin_presets(*presets: DatasetPreset) -> MappingProxyType[str, DatasetPreset]:
    """Turn ``presets`` into a locked-down (immutable) mapping, keyed by
    name.

    Raises:
        ValueError: Two of the given presets share the same ``name``. This
            check happens right away -- this function only ever runs once,
            at the moment this module is first imported -- rather than
            being delayed until someone actually looks a preset up by name.
            That way, a duplicate preset name causes the very first import
            of this module to fail loudly, instead of quietly letting one
            preset hide an earlier one with the same name.
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
    contamination=ContaminationMetadata(status=ContaminationStatus.SUSPECT),
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
    contamination=ContaminationMetadata(status=ContaminationStatus.SUSPECT),
)

BUILTIN_PRESETS: Final[MappingProxyType[str, DatasetPreset]] = _build_builtin_presets(
    _GSM8K_PRESET,
    _SWE_BENCH_VERIFIED_PRESET,
)
