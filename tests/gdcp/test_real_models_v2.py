"""Per-layer FSDP2 of the three real model types on CPU worlds one to four."""

import json
import os
import shutil
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)


def _run(world, action, root):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        "--module",
        "tests.gdcp.real_models_worker",
        action,
        str(root),
    ]
    env = dict(
        os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT), HF_HUB_OFFLINE="1"
    )
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
        stdout, _stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, _stderr = process.communicate()
        pytest.fail(f"torchrun timed out: {command}\n{stdout}")
    print(stdout, flush=True)
    assert process.returncode == 0, f"{command}\n{stdout}"
    result = json.loads((root / f"results-{action}-{world}.json").read_text())
    assert result["status"] == "pass"
    return result


def test_tiny_real_architectures_all_formats_values_identity_and_optimizer_step(
    tmp_path,
):
    pytest.importorskip("transformers")
    root = tmp_path / "real-architecture-checkpoints"
    try:
        assert len(_run(2, "source", root)["checks"]) == 9
        for world in (1, 3, 4):
            result = _run(world, "target", root)
            assert all(
                row.get("V") == "pass" and row.get("R") == "pass"
                for row in result["checks"]
            )
        assert len(_run(2, "final", root)["checks"]) == 27
    finally:
        shutil.rmtree(root, ignore_errors=True)
