#!/usr/bin/env bash
set -euo pipefail
export TZ=America/Los_Angeles
export PYTHONPATH=/data00/iobench/gdcp
export GDCP_SPIKE_ROOT=/data00/iobench/gdcp-test-checkpoints/spikes
export OMP_NUM_THREADS=1
cd /data00/iobench/gdcp
date '+%Y-%m-%d %H:%M:%S PST'
python -m tests.gdcp.spikes.b_scalars_empty
torchrun --standalone --nproc-per-node=4 --module tests.gdcp.spikes.a_boundaries
torchrun --standalone --nproc-per-node=2 --module tests.gdcp.spikes.c_restore
torchrun --standalone --nproc-per-node=2 --module tests.gdcp.spikes.d_gloo
date '+%Y-%m-%d %H:%M:%S PST'
