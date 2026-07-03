#!/usr/bin/env bash
# Runs agentic-evalkit's verification gates in order and prints PASS/FAIL per
# gate, continuing past failures so one run reports the full picture.
#
#   run-gates.sh             offline verification matrix (CONTRIBUTING.md)
#   run-gates.sh --release   + offline release gates (contract tests, strict
#                              docs build, clean-wheel install)
#   run-gates.sh --live      + live Hugging Face gate (requires network)
#
# Exit status: 0 when every executed gate passed, 1 otherwise.
set -u

RELEASE=0
LIVE=0
for arg in "$@"; do
  case "$arg" in
    --release) RELEASE=1 ;;
    --live) LIVE=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

NAMES=()
RESULTS=()
FAILED=0

run_gate() {
  name="$1"
  shift
  echo ""
  echo "=== ${name}: $* ==="
  if "$@"; then
    NAMES+=("$name"); RESULTS+=("PASS")
  else
    NAMES+=("$name"); RESULTS+=("FAIL"); FAILED=1
  fi
}

run_gate "sync"        uv sync --all-groups
run_gate "tests"       uv run pytest --cov --cov-report=term-missing
run_gate "ruff-check"  uv run ruff check .
run_gate "ruff-format" uv run ruff format --check .
run_gate "mypy"        uv run mypy

if [ "$RELEASE" -eq 1 ]; then
  run_gate "contract-gates" uv run pytest tests/contract/test_dependency_boundary.py tests/contract/test_adrs.py tests/contract/test_public_docs.py -v
  run_gate "docs-strict"    uv run mkdocs build --strict
  run_gate "clean-wheel"    uv run pytest tests/integration/test_clean_wheel.py -v
fi

if [ "$LIVE" -eq 1 ]; then
  run_gate "live-hf" uv run pytest tests/live/test_huggingface_live.py -m live -v
fi

echo ""
echo "=== gate summary ==="
i=0
for name in "${NAMES[@]}"; do
  printf '%-16s %s\n' "$name" "${RESULTS[$i]}"
  i=$((i + 1))
done
exit "$FAILED"
