#!/usr/bin/env bash
# Reproduces the CI "test (ubuntu-latest, pyX.Y)" job locally in a disposable
# Linux container, from a pristine `git archive` checkout of the current
# HEAD -- not a bind-mounted working tree -- so the result reflects what CI
# actually sees (no local .venv, caches, or uncommitted artifacts leaking in).
#
# Use this when a check fails only on ubuntu-latest and you need to reproduce
# it on a non-Linux dev machine (e.g. a coverage-combine or platform-specific
# subprocess bug) without waiting on a CI round-trip.
#
#   scripts/repro-linux-ci.sh                    # python 3.11, ephemeral container
#   scripts/repro-linux-ci.sh --python 3.13      # pin a different matrix version
#   scripts/repro-linux-ci.sh --keep             # keep the container for `docker exec` post-mortem
#   scripts/repro-linux-ci.sh --debug-coverage   # additionally dump any leftover
#                                                 # `.coverage.*` parallel data files'
#                                                 # sqlite meta/file tables
#
# Requires: git, a running Docker daemon. Exit status mirrors the containerized
# pytest run (0 = passed).
set -euo pipefail

PYTHON_VERSION="3.11"
KEEP=0
DEBUG_COVERAGE=0

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --python) PYTHON_VERSION="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --debug-coverage) DEBUG_COVERAGE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not on PATH" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "docker daemon is not reachable -- is Docker Desktop running?" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
COMMIT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
ARCHIVE="$(mktemp -t agentic-evalkit-ci-repro-XXXXXX.tar)"
trap 'rm -f "$ARCHIVE"' EXIT

echo "=== archiving HEAD ($COMMIT) -- committed tree only, no local artifacts ==="
git -C "$REPO_ROOT" archive HEAD -o "$ARCHIVE"

# Git Bash/MSYS mangles container-absolute paths like `/work` into a host
# path (e.g. `C:\Program Files\Git\work`) unless path conversion is disabled.
# A no-op on native Linux/macOS shells.
export MSYS_NO_PATHCONV=1

DOCKER_ARGS=(--rm)
CONTAINER_NAME="agentic-evalkit-ci-repro-py${PYTHON_VERSION}-${COMMIT}"
if [ "$KEEP" -eq 1 ]; then
  DOCKER_ARGS=(--name "$CONTAINER_NAME")
  echo "=== container will be kept as '$CONTAINER_NAME' for post-mortem (docker exec -it $CONTAINER_NAME bash) ==="
fi

echo "=== running the CI test job on python:${PYTHON_VERSION}-slim ==="
docker run "${DOCKER_ARGS[@]}" -i \
  -e DEBUG_COVERAGE="$DEBUG_COVERAGE" \
  python:"${PYTHON_VERSION}"-slim bash -s <<'CONTAINER_SCRIPT' < "$ARCHIVE"
set -euo pipefail
mkdir -p /work && cd /work
tar -x
pip install -q uv
uv sync --all-groups
set +e
uv run pytest -m "not live" --cov --cov-report=term-missing
status=$?
set -e

if [ "$DEBUG_COVERAGE" -eq 1 ]; then
  echo "=== leftover coverage data files ==="
  shopt -s nullglob
  files=(.coverage.*)
  if [ ${#files[@]} -eq 0 ]; then
    echo "(none -- combine() consumed everything cleanly)"
  fi
  for f in "${files[@]}"; do
    echo "--- $f ---"
    uv run python -c "
import sqlite3
conn = sqlite3.connect('$f')
print('meta:', conn.execute('select key, value from meta').fetchall())
print('file count:', conn.execute('select count(*) from file').fetchall())
print('sample files:', conn.execute('select path from file limit 20').fetchall())
"
  done
fi

exit "$status"
CONTAINER_SCRIPT
