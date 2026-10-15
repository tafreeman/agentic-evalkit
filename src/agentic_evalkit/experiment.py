"""Saved experiment definitions and durable, versioned execution state."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.models.base import FrozenModel


class BenchmarkPolicy(FrozenModel):
    name: str
    assessment: Literal["regression_only", "conditional", "blocked"] = "regression_only"
    reviewed_on: date
    sources: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    validation_evidence: tuple[str, ...] = ()


class HelmConfig(FrozenModel):
    python: str
    version: str
    run_entries: tuple[str, ...] = Field(min_length=1)
    max_eval_instances: int = Field(default=5, ge=1)
    num_threads: int = Field(default=1, ge=1)
    local_path: str
    phase: Literal["prepare", "evaluate"] = "prepare"
    timeout_seconds: float = Field(default=1800, gt=0, allow_inf_nan=False)


class ExperimentArm(FrozenModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
    manifest: str | None = None
    helm: HelmConfig | None = None

    @model_validator(mode="after")
    def _one_backend(self) -> ExperimentArm:
        if (self.manifest is None) == (self.helm is None):
            raise ValueError("each arm requires exactly one manifest or helm block")
        return self


class ExperimentConfig(FrozenModel):
    name: str = Field(min_length=1)
    purpose: Literal["regression", "capability"] = "regression"
    benchmark: BenchmarkPolicy
    arms: tuple[ExperimentArm, ...] = Field(min_length=1)
    provenance_files: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_experiment(self) -> ExperimentConfig:
        if len({arm.name for arm in self.arms}) != len(self.arms):
            raise ValueError("arm names must be unique")
        if self.benchmark.assessment == "blocked":
            raise ValueError("this benchmark is blocked by its recorded validity assessment")
        if self.purpose == "capability":
            if self.benchmark.assessment != "conditional":
                raise ValueError("regression-only datasets cannot support a capability experiment")
            if not self.benchmark.sources or not self.benchmark.validation_evidence:
                raise ValueError(
                    "capability experiments require sources and task validation evidence"
                )
        return self


class ArmStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PREPARED = "prepared"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ArmState(FrozenModel):
    name: str
    status: ArmStatus = ArmStatus.PENDING
    report: str | None = None
    report_digest: str | None = None
    error: str | None = None


class ExperimentState(FrozenModel):
    name: str
    config_path: str
    input_digest: str
    started_at: datetime
    arms: tuple[ArmState, ...]


def load_experiment(path: Path) -> ExperimentConfig:
    try:
        config = ExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ManifestValidationError(message=f"invalid experiment: {error}") from error
    age = (date.today() - config.benchmark.reviewed_on).days
    if age < 0:
        raise ManifestValidationError(message="benchmark review date cannot be in the future")
    if config.purpose == "capability" and age > 90:
        raise ManifestValidationError(
            message="benchmark validity review is older than 90 days; review before running"
        )
    for item in (*config.provenance_files, *config.benchmark.validation_evidence):
        if not (path.resolve().parent / item).is_file():
            raise ManifestValidationError(message=f"missing provenance or validation file: {item}")
    return config


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Replace one state snapshot only after its complete bytes are flushed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".eval-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
