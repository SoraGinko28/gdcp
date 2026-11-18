#!/usr/bin/env bash
# Host entry point; run visibly in the existing h20 tmux session.
set -euo pipefail
export TZ=America/Los_Angeles
mkdir -p /data00/iobench/logs/gdcp-v2
name=${1:-iobench-gdcp-v2-cpu}
mode=${2:-cpu}
owner=${3:-standalone}
evidence=${4:-${GDCP_GPU_EVIDENCE:-/data00/iobench/logs/gdcp-v2/gpu-guard.jsonl}}
case "$name" in
  iobench-gdcp-*) ;;
  *) echo "refusing non-task container name" >&2; exit 2 ;;
esac
case "$mode" in
  cpu|gpu) ;;
  *) echo "expected cpu or gpu mode" >&2; exit 2 ;;
esac
args=()
if [[ $mode == gpu ]]; then
  GDCP_GPU_EVIDENCE="$evidence" python3 /data00/iobench/gdcp/scripts/gdcp_v2_safety.py gpu
  args+=(--gpus '"device=2,3,4,5,6,7"')
fi
date '+%Y-%m-%d %H:%M:%S PST'
docker run -d --name "$name" --label "gdcp.v2.owner=$owner" --network host --ipc host "${args[@]}" \
  -v /data00/iobench:/data00/iobench \
  -w /data00/iobench/gdcp iobench:h20-cu126 sleep infinity
docker exec "$name" python -m pip install --no-index --no-deps /data00/iobench/gdcp-env/wheels/*.whl
docker exec "$name" python -c 'import sys,torch,transformers,safetensors; print(sys.version); print("torch",torch.__version__,"transformers",transformers.__version__,"safetensors",safetensors.__version__); print("CUDA devices",torch.cuda.device_count())'
docker inspect "$name" --format 'Mounts={{json .Mounts}} GPURequests={{json .HostConfig.DeviceRequests}}'
date '+%Y-%m-%d %H:%M:%S PST'
