"""Contracts for explicitly selected local component factories."""

import os
from pathlib import Path

from agentic_evalkit.components import component_identity, load_component_bundle
from agentic_evalkit.manifest import ComponentFactoryConfig


def _factory(path: Path, *, suffix: str = "") -> ComponentFactoryConfig:
    path.write_text(
        f"""from agentic_evalkit.components import ComponentBundle
from agentic_evalkit.models import EvalSample, GradeResult

class Adapter:
    api_version = "1"
    name = "adapter{suffix}"
    def prepare(self, record):
        return EvalSample(
            sample_id=record.row_id,
            input={{}},
            reference="ok",
            source_digest=record.digest,
            adapter=self.name,
        )
    def validate_oracle(self, sample):
        return True
    def aggregate_metadata(self):
        return {{}}

class Grader:
    async def grade(self, sample, execution):
        return GradeResult(sample_id=sample.sample_id, grader="local", status="PASS", score=1.0)

def make(config):
    assert config == {{"threshold": 2}}
    return ComponentBundle(adapter=Adapter(), grader=Grader())
""",
        encoding="utf-8",
    )
    return ComponentFactoryConfig(path=str(path), function="make", config={"threshold": 2})


def test_factory_identity_records_source_function_and_config(tmp_path: Path) -> None:
    config = _factory(tmp_path / "components.py")

    loaded = load_component_bundle(config)

    assert loaded.bundle.adapter.name == "adapter"
    assert loaded.identity.source_fingerprint.startswith("sha256:")
    assert loaded.identity.function_fingerprint.startswith("sha256:")
    assert loaded.identity.config_fingerprint.startswith("sha256:")
    assert loaded.identity.as_manifest_value()["fingerprint"] == loaded.identity.fingerprint


def test_factory_identity_changes_for_each_semantic_input(tmp_path: Path) -> None:
    path = tmp_path / "components.py"
    config = _factory(path)
    original = component_identity(config)

    changed_function = component_identity(config.model_copy(update={"function": "other"}))
    changed_config = component_identity(config.model_copy(update={"config": {"threshold": 3}}))
    changed_source = component_identity(_factory(path, suffix="2"))

    assert original.fingerprint != changed_function.fingerprint
    assert original.fingerprint != changed_config.fingerprint
    assert original.fingerprint != changed_source.fingerprint


def test_factory_never_executes_stale_bytecode(tmp_path: Path) -> None:
    path = tmp_path / "components.py"
    config = _factory(path, suffix="1")
    timestamp = path.stat()
    assert load_component_bundle(config).bundle.adapter.name == "adapter1"
    _factory(path, suffix="2")
    os.utime(path, ns=(timestamp.st_atime_ns, timestamp.st_mtime_ns))
    assert load_component_bundle(config).bundle.adapter.name == "adapter2"
