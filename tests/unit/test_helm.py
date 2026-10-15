"""Native HELM runs require retained, nonempty evidence after process success."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from agentic_evalkit.errors import ManifestValidationError
from agentic_evalkit.experiment import HelmConfig
from agentic_evalkit.helm import run_helm

if TYPE_CHECKING:
    from pathlib import Path


def _config(*, phase: str = "prepare") -> HelmConfig:
    return HelmConfig(
        python=sys.executable,
        version="test",
        run_entries=("fixture",),
        local_path=".",
        phase=phase,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_successful_helm_process_requires_and_records_native_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentic_evalkit.helm as helm

    monkeypatch.setattr(helm, "helm_version", lambda config: config.version)
    evidence = tmp_path / "arm" / "helm" / "input_instances.json"
    script = (
        "from pathlib import Path; import json; "
        f"path=Path({str(evidence)!r}); path.parent.mkdir(parents=True, exist_ok=True); "
        "path.write_text(json.dumps({'instances': [1]}), encoding='utf-8')"
    )
    monkeypatch.setattr(
        helm, "helm_command", lambda config, directory, suite: [sys.executable, "-c", script]
    )

    report = await run_helm(_config(), tmp_path / "arm", "fixture")

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["backend"] == "helm"
    assert payload["evidence"] == [
        {"path": "helm/input_instances.json", "sha256": payload["evidence"][0]["sha256"]}
    ]


@pytest.mark.asyncio
async def test_zero_exit_without_native_evidence_is_not_a_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentic_evalkit.helm as helm

    monkeypatch.setattr(helm, "helm_version", lambda config: config.version)
    monkeypatch.setattr(
        helm, "helm_command", lambda config, directory, suite: [sys.executable, "-c", "pass"]
    )

    with pytest.raises(
        ManifestValidationError, match=r"produced no input_instances\.json evidence"
    ):
        await run_helm(_config(), tmp_path / "arm", "fixture")
