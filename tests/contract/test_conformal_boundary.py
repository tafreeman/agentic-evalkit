"""Static proof that a conformal threshold certificate cannot reach a hard gate (ADR-0028).

ADR-0007 lets a judge gate a release only under a calibration that clears the
ratified floors. A ``ThresholdCertificate`` is evidence for *choosing* a score
threshold; it is a different statement (expected or high-probability *risk*,
not TNR/TPR) and carries no ratified floor. If a grader imported it, a
certificate could quietly become a second, unratified way to open the gate.

So nothing under ``src/`` may import the module. A caller that wants to use a
certificate does so from outside the package, in its own code, where the choice
is visible. If a later ADR deliberately wires it in, that ADR updates this test.
"""

import ast
from pathlib import Path

CONFORMAL_MODULE = "agentic_evalkit.graders.conformal"
SOURCE_ROOT = Path("src/agentic_evalkit")


def _imports_conformal(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == CONFORMAL_MODULE for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == CONFORMAL_MODULE:
                return True
            if node.module == "agentic_evalkit.graders" and any(
                alias.name == "conformal" for alias in node.names
            ):
                return True
            if node.level > 0 and (node.module or "").split(".")[-1] == "conformal":
                return True
    return False


def test_no_package_module_imports_the_conformal_calibrator() -> None:
    offenders = [
        str(path)
        for path in SOURCE_ROOT.rglob("*.py")
        if path.name != "conformal.py"
        and _imports_conformal(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    ]
    assert offenders == [], (
        "a conformal ThresholdCertificate must not feed a grader's hard gate "
        f"(ADR-0028, ADR-0007); imported by: {offenders}"
    )


def test_the_scan_detects_an_import_when_there_is_one() -> None:
    # Guard against the scan going blind: it must flag each way of importing.
    for source in (
        f"import {CONFORMAL_MODULE}",
        f"from {CONFORMAL_MODULE} import calibrate_score_threshold",
        "from agentic_evalkit.graders import conformal",
        "from .conformal import ThresholdCertificate",
    ):
        assert _imports_conformal(ast.parse(source)), source
    assert not _imports_conformal(ast.parse("from agentic_evalkit.graders import judge"))


def test_every_certificate_model_is_a_frozen_wire_model() -> None:
    from agentic_evalkit.graders.conformal import (
        ExactRational,
        ScoreObservation,
        ThresholdCertificate,
    )
    from agentic_evalkit.models.base import FrozenModel

    for model in (ExactRational, ScoreObservation, ThresholdCertificate):
        assert issubclass(model, FrozenModel)
        assert model.model_config["frozen"] is True
        assert model.model_config["extra"] == "forbid"
