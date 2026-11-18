#!/usr/bin/env bash
# Execute inside iobench-gdcp-v1, visibly through H20's existing tmux session.
set -euo pipefail
export TZ=America/Los_Angeles
source_root=${GDCP_SOURCE_ROOT:-/data00/iobench/gdcp}
export PYTHONPATH="$source_root"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export TMPDIR=/data00/iobench/gdcp-test-checkpoints/tmp
export GDTENSOR_TORCH_TESTS=1
mkdir -p "$TMPDIR"
cd "$source_root"
trap 'status=$?; date "+%Y-%m-%d %H:%M:%S PST"; echo "GDCP_PYTEST_RC=$status"' EXIT
date '+%Y-%m-%d %H:%M:%S PST'
python -m pytest -v --capture=tee-sys --log-cli-level=INFO \
  -p no:cacheprovider "$@"
