#!/usr/bin/env bash
# Equivalent static scope to full_check.sh, using the H20 image interpreter.
set -euo pipefail
export TZ=America/Los_Angeles
source_root=${GDCP_SOURCE_ROOT:-/data00/iobench/gdcp}
export PYTHONPATH="$source_root"
export PYTHONDONTWRITEBYTECODE=1
cd "$source_root"
trap 'status=$?; date "+%Y-%m-%d %H:%M:%S PST"; echo "GDCP_STATIC_RC=$status"' EXIT
date '+%Y-%m-%d %H:%M:%S PST'
ruff --version
ty --version
pyrefly --version
ruff format --check gdtensor gdtensor_conn gdcp tests
ruff check --no-cache gdtensor gdtensor_conn gdcp tests
ruff format --check scripts/gdcp_v2_safety.py scripts/gdcp_v2_pipeline.py scripts/gdcp_v2_hf_interop.py scripts/gdcp_v2_worker.py
ruff check --no-cache scripts/gdcp_v2_safety.py scripts/gdcp_v2_pipeline.py scripts/gdcp_v2_hf_interop.py scripts/gdcp_v2_worker.py
ty check --python /opt/iobench/bin/python --error-on-warning
pyrefly check --python-interpreter-path /opt/iobench/bin/python --min-severity=warn
PYTHONPYCACHEPREFIX=/data00/iobench/gdcp-env/final-pycache python -m compileall -q gdtensor gdtensor_conn gdcp tests scripts/gdcp_v2_safety.py scripts/gdcp_v2_pipeline.py scripts/gdcp_v2_hf_interop.py scripts/gdcp_v2_worker.py
python -c 'import sys; import gdcp, gdcp.backend, gdcp.core, gdcp.desc, gdcp.policy, gdcp.state; assert "torch" not in sys.modules; print("neutral imports: torch-free")'
