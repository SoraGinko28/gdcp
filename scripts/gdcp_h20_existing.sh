#!/usr/bin/env bash
set -euo pipefail
export TZ=America/Los_Angeles
export PYTHONPATH=/data00/iobench/gdcp
export OMP_NUM_THREADS=1
export TMPDIR=/data00/iobench/gdcp-test-checkpoints/tmp
mkdir -p "$TMPDIR"
cd /data00/iobench/gdcp
date '+%Y-%m-%d %H:%M:%S PST'
GDTENSOR_TORCH_TESTS=1 python -m pytest -v --capture=tee-sys \
  --log-cli-level=INFO -p no:cacheprovider \
  tests/core tests/io tests/checkpoint tests/torch tests/sim tests/demo
date '+%Y-%m-%d %H:%M:%S PST'
