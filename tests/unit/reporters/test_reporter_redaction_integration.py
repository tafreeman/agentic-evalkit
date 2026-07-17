"""Integration tests proving redaction happens before any reporter runs.

Per the design doc (§12), a run is redacted (secrets scrubbed out) exactly
once, producing a new copy rather than editing the run in place, before it
reaches any reporter. Every output format (JSON, JSONL, Markdown, HTML) must
see and write out that same already-redacted copy -- never the original,
unredacted evidence.
"""

import json
from pathlib import Path

from agentic_evalkit.models import EvalRunResult
from agentic_evalkit.reporters import (
    HtmlReporter,
    JsonlReporter,
    JsonReporter,
    MarkdownReporter,
    RedactionPolicy,
    apply_redaction,
)


def _leaking_run(run: EvalRunResult) -> EvalRunResult:
    assert run.samples[0].grade is not None
    leaking_grade = run.samples[0].grade.model_copy(
        update={"evidence": {"expected": "42", "actual": "42", "api_key": "sk-super-secret"}}
    )
    leaking_sample = run.samples[0].model_copy(update={"grade": leaking_grade})
    return run.model_copy(update={"samples": (leaking_sample, *run.samples[1:])})


def test_json_reporter_never_writes_redacted_evidence_key(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = _leaking_run(pass_error_timeout_and_provenance_run)
    policy = RedactionPolicy(evidence_keys=("api_key",))
    redacted_run = apply_redaction(run, policy)

    json_path = JsonReporter().write(redacted_run, tmp_path / "run.json", generated_at="fixed")
    content = json_path.read_text(encoding="utf-8")
    assert "sk-super-secret" not in content
    payload = json.loads(content)
    assert "api_key" not in payload["samples"][0]["grade"]["evidence"]


def test_jsonl_reporter_never_writes_redacted_evidence_key(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = _leaking_run(pass_error_timeout_and_provenance_run)
    policy = RedactionPolicy(evidence_keys=("api_key",))
    redacted_run = apply_redaction(run, policy)

    jsonl_path = JsonlReporter().write(redacted_run, tmp_path / "run.jsonl", generated_at="fixed")
    content = jsonl_path.read_text(encoding="utf-8")
    assert "sk-super-secret" not in content


def test_markdown_reporter_never_writes_secret_pattern_match(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = _leaking_run(pass_error_timeout_and_provenance_run)
    policy = RedactionPolicy(secret_patterns=(r"sk-[a-zA-Z0-9-]+",))
    redacted_run = apply_redaction(run, policy)

    md_path = MarkdownReporter().write(redacted_run, tmp_path / "run.md", generated_at="fixed")
    content = md_path.read_text(encoding="utf-8")
    assert "sk-super-secret" not in content


def test_html_reporter_never_writes_redacted_evidence_key(
    tmp_path: Path, pass_error_timeout_and_provenance_run: EvalRunResult
) -> None:
    run = _leaking_run(pass_error_timeout_and_provenance_run)
    policy = RedactionPolicy(evidence_keys=("api_key",))
    redacted_run = apply_redaction(run, policy)

    html_path = HtmlReporter().write(redacted_run, tmp_path / "run.html", generated_at="fixed")
    content = html_path.read_text(encoding="utf-8")
    assert "sk-super-secret" not in content
