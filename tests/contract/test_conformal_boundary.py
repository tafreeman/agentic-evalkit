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
SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "agentic_evalkit"


def _names_conformal(module: str) -> bool:
    return module == "conformal" or module.endswith(".conformal")


def _imports_conformal(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == CONFORMAL_MODULE for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == CONFORMAL_MODULE or (node.level > 0 and _names_conformal(module)):
                return True
            # ``from agentic_evalkit.graders import conformal``, ``from . import
            # conformal`` and ``from ..graders import judge, conformal``.
            parent = module == "agentic_evalkit.graders" or (
                node.level > 0 and (module == "" or module.split(".")[-1] == "graders")
            )
            if parent and any(alias.name == "conformal" for alias in node.names):
                return True
        # importlib.import_module / __import__ take the module as a string.
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and (node.value == CONFORMAL_MODULE or node.value.endswith(".graders.conformal"))
        ):
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
        "from . import conformal",
        "from . import judge, conformal",
        "from ..graders import conformal",
        "from ..graders.conformal import ThresholdCertificate",
        f"importlib.import_module({CONFORMAL_MODULE!r})",
        f"__import__({CONFORMAL_MODULE!r})",
    ):
        assert _imports_conformal(ast.parse(source)), source
    for clean in (
        "from agentic_evalkit.graders import judge",
        "from . import judge",
        "from ..targets import conformal_helpers",
    ):
        assert not _imports_conformal(ast.parse(clean)), clean


def test_the_scan_reads_the_package_wherever_pytest_starts() -> None:
    assert (SOURCE_ROOT / "graders" / "conformal.py").is_file()


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
