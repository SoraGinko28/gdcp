#!/usr/bin/env bash
# Run on H20 through the existing h20 tmux session. No GPUs are exposed.
set -euo pipefail
mkdir -p /data00/iobench/logs/gdcp-v1
export TZ=America/Los_Angeles
date '+%Y-%m-%d %H:%M:%S PST'
docker run -d --name iobench-gdcp-v1 --network host \
  -v /data00/iobench:/data00/iobench \
  -w /data00/iobench/gdcp iobench:h20-cu126 sleep infinity
if compgen -G '/data00/iobench/gdcp-env/wheels/*.whl' > /dev/null; then
  docker exec iobench-gdcp-v1 python -m pip install --no-index --no-deps \
    /data00/iobench/gdcp-env/wheels/*.whl
else
  docker exec iobench-gdcp-v1 python -m pip install \
    'numpy==2.5.3' 'safetensors==0.8.0' 'pytest==9.1.1' \
    'ruff==0.17.1' 'ty==0.0.14' 'pyrefly==1.4.0'
fi
docker exec iobench-gdcp-v1 python -m pip freeze
docker exec iobench-gdcp-v1 python -c \
  'import sys, torch; print(sys.version); print("torch", torch.__version__); print("cuda available", torch.cuda.is_available())'
date '+%Y-%m-%d %H:%M:%S PST'
