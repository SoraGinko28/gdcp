"""Shared launcher for isolated GDCP FSDP2 acceptance workers."""

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_fsdp2(world, action, path, *arguments):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        "--module",
        "tests.gdcp.fsdp2_worker",
        action,
        str(path),
        *map(str, arguments),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env.pop("PYTHONOPTIMIZE", None)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _stderr = process.communicate(timeout=240)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, _stderr = process.communicate()
        pytest.fail(f"torchrun timed out: {command}\n{stdout}")
    print(stdout, flush=True)
    assert process.returncode == 0, f"{command}\n{stdout}"
    return stdout
