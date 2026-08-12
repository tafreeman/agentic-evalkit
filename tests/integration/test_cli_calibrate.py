"""Hermetic integration tests for ``agentic-evalkit calibrate`` (ADR-0024).

These exercise the command end to end -- reading a labeled set off disk,
measuring a judge, writing the artifact, printing the verdict, choosing the
exit code -- with no network and no provider. The judge under measurement is
the packaged ``ReferenceJudgeClient``, which is a substring matcher rather
than a model, so the measurement is real but deterministic.

Every test runs with the working directory moved to ``tmp_path``, because
the command confines its reads to the working directory. That is the
behaviour under test in ``test_a_labeled_set_outside_the_working_directory_is_refused``,
not an artifact of the harness.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from typer.testing import CliRunner

from agentic_evalkit.cli import app
from agentic_evalkit.cli.app import ExitCode
from agentic_evalkit.examples.reference_judge import ReferenceJudgeClient
from agentic_evalkit.graders.calibration import (
    AuthorityLevel,
    CalibrationArtifact,
    judge_authority,
)

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

#: See ``tests/unit/graders/test_measure.py`` for why a perfect true-negative
#: rate needs far more than the 30-sample class minimum: its 95% Wilson lower
#: bound only reaches the 0.95 floor at 73 samples.
_GATING_NEGATIVES = 80
_GATING_POSITIVES = 40


def _rows(*, positives: int, negatives: int) -> list[dict[str, str]]:
    """Rows the reference judge grades correctly: it looks for the reference
    text inside the candidate output, so a positive embeds it and a negative
    does not."""
    good = [
        {
            "sample_id": f"pos-{i}",
            "prompt": "what is the answer?",
            "candidate_output": f"after working it through, the answer is {i}",
            "reference": str(i),
            "label": "good",
        }
        for i in range(positives)
    ]
    bad = [
        {
            "sample_id": f"neg-{i}",
            "prompt": "what is the answer?",
            "candidate_output": "the answer is somewhere else entirely",
            "reference": f"unrepeated-token-{i}",
            "label": "bad",
        }
        for i in range(negatives)
    ]
    return good + bad


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def _invoke(*args: str) -> Any:
    return runner.invoke(app, ["calibrate", *args])


# --- the earned-authority path ---------------------------------------------


def test_a_fully_measured_judge_exits_zero_and_reports_gating(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "labeled.jsonl", _rows(positives=_GATING_POSITIVES, negatives=_GATING_NEGATIVES)
    )

    result = _invoke("labeled.jsonl", "--output", "cal.json")

    assert result.exit_code == ExitCode.SUCCESS
    assert "GATING" in result.stdout
    assert (tmp_path / "cal.json").is_file()


def test_the_written_artifact_is_dated_and_names_the_measured_judge(tmp_path: Path) -> None:
    """The two properties that made a hand-written artifact unusable in practice."""
    _write_jsonl(
        tmp_path / "labeled.jsonl", _rows(positives=_GATING_POSITIVES, negatives=_GATING_NEGATIVES)
    )

    _invoke("labeled.jsonl", "--output", "cal.json")
    artifact = CalibrationArtifact.model_validate_json(
        (tmp_path / "cal.json").read_text(encoding="utf-8")
    )

    assert artifact.calibrated_at is not None
    assert artifact.judge_fingerprint == ReferenceJudgeClient().fingerprint
    assert artifact.total_labeled == _GATING_POSITIVES + _GATING_NEGATIVES
    assert artifact.usability_failure_reason() is None


@pytest.mark.parametrize(
    ("positives", "negatives"),
    [
        pytest.param(_GATING_POSITIVES, _GATING_NEGATIVES, id="gating"),
        pytest.param(12, 12, id="thin-advisory"),
        pytest.param(1, 1, id="very-thin-advisory"),
    ],
)
def test_the_printed_verdict_equals_judge_authority_on_the_same_artifact(
    tmp_path: Path, positives: int, negatives: int
) -> None:
    """The command must never form its own opinion of the artifact it wrote.

    It prints whatever ``judge_authority`` says. Recomputing the decision at
    the CLI -- even correctly, today -- would create a second definition free
    to drift from the one the release gate actually consults, so this
    compares the printed level against the function's own answer rather than
    against a hardcoded expectation.
    """
    _write_jsonl(tmp_path / "labeled.jsonl", _rows(positives=positives, negatives=negatives))

    result = _invoke("labeled.jsonl", "--output", "cal.json", "--format", "json")
    payload = json.loads(result.stdout)
    artifact = CalibrationArtifact.model_validate(payload["artifact"])
    expected = judge_authority(artifact, judge_fingerprint=ReferenceJudgeClient().fingerprint)

    assert payload["authority"]["level"] == expected.level.value
    if expected.reason is not None:
        assert payload["authority"]["reason"] == expected.reason


# --- the emit-do-not-refuse decision ----------------------------------------


def test_a_thin_labeled_set_still_writes_an_artifact_and_explains_itself(
    tmp_path: Path,
) -> None:
    """ADR-0024: thin evidence is a fact worth recording, not an error."""
    _write_jsonl(tmp_path / "thin.jsonl", _rows(positives=6, negatives=6))

    result = _invoke("thin.jsonl", "--output", "thin-cal.json")

    assert result.exit_code == ExitCode.MISSING_CAPABILITY
    assert "ADVISORY" in result.stdout
    assert "below the required minimum of 30" in result.stdout
    assert (tmp_path / "thin-cal.json").is_file()


def test_a_thin_artifact_records_its_real_counts(tmp_path: Path) -> None:
    """The artifact is honest, not blanked out because it cannot gate."""
    _write_jsonl(tmp_path / "thin.jsonl", _rows(positives=6, negatives=6))

    _invoke("thin.jsonl", "--output", "thin-cal.json")
    artifact = CalibrationArtifact.model_validate_json(
        (tmp_path / "thin-cal.json").read_text(encoding="utf-8")
    )

    assert (artifact.true_positive, artifact.true_negative) == (6, 6)
    assert judge_authority(artifact).level is AuthorityLevel.ADVISORY


def test_the_artifact_carries_no_prompt_or_candidate_output(tmp_path: Path) -> None:
    """Counts, IDs and timestamps only -- so the file can be committed safely."""
    rows = _rows(positives=2, negatives=2)
    rows[0]["candidate_output"] = "SENTINEL-CANDIDATE-TEXT"
    rows[0]["prompt"] = "SENTINEL-PROMPT-TEXT"
    _write_jsonl(tmp_path / "labeled.jsonl", rows)

    _invoke("labeled.jsonl", "--output", "cal.json")
    written = (tmp_path / "cal.json").read_text(encoding="utf-8")

    assert "SENTINEL-CANDIDATE-TEXT" not in written
    assert "SENTINEL-PROMPT-TEXT" not in written


# --- input handling ---------------------------------------------------------


def test_a_yaml_labeled_set_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "labeled.yaml").write_text(
        yaml.safe_dump(_rows(positives=2, negatives=2)), encoding="utf-8"
    )

    result = _invoke("labeled.yaml", "--output", "cal.json")

    assert result.exit_code == ExitCode.MISSING_CAPABILITY  # thin, but read fine
    assert (tmp_path / "cal.json").is_file()


def test_the_calibration_id_defaults_to_something_derived_from_the_file(
    tmp_path: Path,
) -> None:
    _write_jsonl(tmp_path / "labeled.jsonl", _rows(positives=2, negatives=2))

    _invoke("labeled.jsonl", "--output", "cal.json")
    artifact = CalibrationArtifact.model_validate_json(
        (tmp_path / "cal.json").read_text(encoding="utf-8")
    )

    assert artifact.calibration_id.startswith("labeled-")


def test_an_explicit_calibration_id_is_used_verbatim(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "labeled.jsonl", _rows(positives=2, negatives=2))

    _invoke("labeled.jsonl", "--output", "cal.json", "--calibration-id", "cal-2026-08")
    artifact = CalibrationArtifact.model_validate_json(
        (tmp_path / "cal.json").read_text(encoding="utf-8")
    )

    assert artifact.calibration_id == "cal-2026-08"


def test_a_judge_named_by_import_string_is_measured(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "labeled.jsonl", _rows(positives=2, negatives=2))

    result = _invoke(
        "labeled.jsonl",
        "--output",
        "cal.json",
        "--judge",
        "agentic_evalkit.examples.reference_judge:ReferenceJudgeClient",
    )
    artifact = CalibrationArtifact.model_validate_json(
        (tmp_path / "cal.json").read_text(encoding="utf-8")
    )

    assert result.exit_code == ExitCode.MISSING_CAPABILITY
    assert artifact.judge_fingerprint == ReferenceJudgeClient().fingerprint


# --- refusals ---------------------------------------------------------------


def test_a_row_that_is_not_a_labeled_sample_is_rejected(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "bad.jsonl", [{"sample_id": "s0", "prompt": "q"}])

    result = _invoke("bad.jsonl", "--output", "cal.json")

    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "is not a labeled judge sample" in result.stdout
    assert not (tmp_path / "cal.json").exists()


def test_a_label_outside_the_named_pair_is_rejected(tmp_path: Path) -> None:
    """A bare ``true``/``yes``/``1`` must not be silently read as one of the classes."""
    _write_jsonl(
        tmp_path / "bad.jsonl",
        [
            {
                "sample_id": "s0",
                "prompt": "q",
                "candidate_output": "a",
                "reference": "a",
                "label": "yes",
            }
        ],
    )

    result = _invoke("bad.jsonl", "--output", "cal.json")

    assert result.exit_code == ExitCode.INVALID_INPUT


def test_a_missing_labeled_set_is_reported_not_traced(tmp_path: Path) -> None:
    result = _invoke("nope.jsonl", "--output", "cal.json")

    assert result.exit_code == ExitCode.PROVIDER_ERROR
    assert "could not be read" in result.stdout


def test_a_labeled_set_outside_the_working_directory_is_refused(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.jsonl"
    _write_jsonl(outside, _rows(positives=1, negatives=1))

    result = _invoke(str(outside), "--output", "cal.json")

    assert result.exit_code == ExitCode.PROVIDER_ERROR


def test_an_unusable_judge_specification_is_reported(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "labeled.jsonl", _rows(positives=1, negatives=1))

    result = _invoke("labeled.jsonl", "--output", "cal.json", "--judge", "not-a-known-judge")

    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "import string" in result.stdout


def test_a_judge_import_string_naming_a_missing_module_is_reported(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "labeled.jsonl", _rows(positives=1, negatives=1))

    result = _invoke("labeled.jsonl", "--output", "cal.json", "--judge", "no_such_module_xyz:Judge")

    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "could not import module" in result.stdout


def test_a_judge_import_string_naming_a_missing_attribute_is_reported(
    tmp_path: Path,
) -> None:
    _write_jsonl(tmp_path / "labeled.jsonl", _rows(positives=1, negatives=1))

    result = _invoke(
        "labeled.jsonl",
        "--output",
        "cal.json",
        "--judge",
        "agentic_evalkit.examples.reference_judge:NoSuchJudge",
    )

    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "has no attribute" in result.stdout


# --- output placement -------------------------------------------------------


def test_the_output_directory_is_created_if_it_does_not_exist(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "labeled.jsonl", _rows(positives=2, negatives=2))

    _invoke("labeled.jsonl", "--output", "evidence/2026-08/cal.json")

    assert (tmp_path / "evidence" / "2026-08" / "cal.json").is_file()


def test_the_command_appears_in_the_root_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == ExitCode.SUCCESS
    assert "calibrate" in result.stdout
