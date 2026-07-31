# Deferred Work

- source_spec: `docs/specs/spec-runner-redaction-single-path.md`
  summary: The CLI's console error path prints an `AgenticEvalkitError`'s raw, unredacted `.message`
    (and lets any other exception's raw traceback propagate), so a secret-shaped substring in an
    underlying failure still reaches stdout/stderr even after `RunFailed.message` was fixed to redact
    on the persisted-event path.
  evidence: Traced `src/agentic_evalkit/cli/runs.py` into
    `src/agentic_evalkit/cli/app.py::run_cli_command`, which prints
    `f"[{error.code}] {error.message}"` for any `AgenticEvalkitError` (e.g. `DatasetProviderUnavailable`)
    with no redaction applied, and re-raises/propagates any non-framework exception as an ordinary,
    unredacted Python traceback. This is pre-existing behavior, not introduced or regressed by the
    `runner-redaction-single-path` bundle, which only routes `RunFailed.message` (the persisted event)
    through the redact-then-bound helper -- the console boundary is a distinct, untouched code path.
