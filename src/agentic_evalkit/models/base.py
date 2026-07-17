"""The shared base class every wire model in this codebase builds on."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Base class for every data model passed around or saved to disk in this codebase (ADR-0002).

    Inheriting from this gives a model two guarantees, set up by
    ``model_config`` below:

    - It's immutable ("frozen"): once you build an instance, its fields
      can't be changed. Code that wants a modified version has to build a
      new instance instead of editing the old one in place. This matters
      because these objects get handed between different parts of the
      pipeline, and sometimes saved as evidence -- if one piece of code
      could quietly edit a shared instance, another piece of code holding
      the same object would see that change too, which is exactly the kind
      of bug that's painful to track down later.
    - It rejects unknown fields (``extra="forbid"``): trying to construct
      one with a field name that doesn't exist raises an error immediately,
      instead of silently ignoring what might be a typo.

    ``schema_version`` records which version of this model's shape a saved
    copy was written with. It's pinned at ``"1"`` for now: as long as
    changes only ever *add* new optional fields, old saved data stays
    readable without needing a version bump -- this number only has to
    change if a field is ever removed, renamed, or has its meaning changed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["1"] = "1"
