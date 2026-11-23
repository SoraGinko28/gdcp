"""Real CPU FSDP2 acceptance tests, launched as Gloo torchrun processes."""

import builtins
import io
import json
import os
import shutil
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest
import safetensors

from gdtensor.builder import Ok
from gdtensor_conn.iobackend import FileIO
from tests.support.gdcp_reader import independent_reader

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)


def _run(world, action, path, *arguments):
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


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("gdcp-fsdp2-acceptance")
    cache = {}

    def source(world):
        if world not in cache:
            path, oracle = root / f"source-{world}", root / f"oracle-{world}.json"
            _run(world, "save", path, "--oracle", oracle)
            cache[world] = path, oracle
        return cache[world]

    yield source
    shutil.rmtree(root)


def test_01_same_world_roundtrip(checkpoints):
    source, _oracle = checkpoints(2)
    _run(2, "load", source, "--source-world", 2)


@pytest.mark.parametrize("source_world,target_world", [(2, 3), (3, 2), (1, 4), (4, 1)])
def test_02_reshard_matrix(checkpoints, source_world, target_world):
    source, _oracle = checkpoints(source_world)
    _run(target_world, "load", source, "--source-world", source_world)


def test_03_training_continues_after_reshard(checkpoints):
    source, _oracle = checkpoints(2)
    _run(2, "load", source, "--source-world", 2, "--continue-step")
    _run(3, "load", source, "--source-world", 2, "--continue-step")


@pytest.mark.parametrize("backend", ["nccl", "cuda:nccl,cpu:gloo"])
def test_gpu_save_failure_and_reshard(tmp_path, backend):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 3 or not torch.distributed.is_nccl_available():
        pytest.skip("requires three CUDA devices and NCCL")
    _run(2, "gpu_save", tmp_path, "--backend", backend)
    _run(3, "gpu_load", tmp_path, "--backend", backend)


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
def test_independent_process_subgroups(tmp_path, backend):
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < 4 or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("requires four CUDA devices and NCCL")
    output = _run(4, "subgroups", tmp_path, "--backend", backend)
    assert output.count("subgroup PASS") == 4


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
def test_hsdp_checkpoint_layouts(tmp_path, backend):
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < 4 or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("requires four CUDA devices and NCCL")
    output = _run(4, "hsdp", tmp_path, "--backend", backend)
    assert output.count("HSDP PASS") == 4
    # The reader checks that each logical byte appears on disk exactly once.
    records, _, _ = independent_reader(tmp_path / "hsdp" / "safetensors")
    flat, _, _ = independent_reader(tmp_path / "fsdp" / "safetensors")
    assert records == flat


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
def test_parameter_layouts(tmp_path, backend):
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < 4 or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("requires four CUDA devices and NCCL")
    output = _run(4, "param_layouts", tmp_path, "--backend", backend)
    assert output.count("parameter layouts PASS") == 4
    records, _, _ = independent_reader(tmp_path / "mixed" / "safetensors")
    flat, _, _ = independent_reader(tmp_path / "flat" / "safetensors")
    assert records == flat


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
def test_dp_tp_same_and_different_dimensions(tmp_path, backend):
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < 4 or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("requires four CUDA devices and NCCL")
    output = _run(4, "dp_tp", tmp_path, "--backend", backend)
    assert output.count("DP+TP PASS") == 4


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
@pytest.mark.parametrize("world", [4, 8])
def test_hsdp_tp_checkpoint_layouts(tmp_path, backend, world):
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < world or not torch.distributed.is_nccl_available()
    ):
        pytest.skip(f"requires {world} CUDA devices and NCCL")
    output = _run(world, "hsdp_tp", tmp_path, "--backend", backend)
    assert output.count("HSDP+TP PASS") == world
    for root in tmp_path.iterdir():
        independent_reader(root / "hsdptp" / "safetensors")


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
@pytest.mark.parametrize("flat_world", [2, 3])
def test_cross_world_native_interchange(tmp_path, backend, flat_world):
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < 4 or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("requires four CUDA devices and NCCL")
    for world, action in (
        (flat_world, "cross_save"),
        (4, "cross_exchange"),
        (flat_world, "cross_exchange"),
    ):
        output = _run(world, action, tmp_path, "--backend", backend)
        assert output.count("CROSS PASS") == world


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
def test_lazy_adamw_resume(tmp_path, backend):
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("requires two CUDA devices and NCCL")
    output = _run(2, "lazy_adamw", tmp_path, "--backend", backend)
    assert output.count("lazy AdamW PASS") == 2


def test_hsdp_tp_cross_world_dcp_interchange(tmp_path):
    for world, action in (
        (4, "hsdp_tp_source"),
        (8, "hsdp_tp_expand"),
        (4, "hsdp_tp_contract"),
    ):
        output = _run(world, action, tmp_path)
        assert output.count("HSDP+TP interchange PASS") == world


def _identity_files(path):
    files = {"gdcp_manifest.json": (path / "gdcp_manifest.json").read_bytes()}
    files.update(
        {
            str(file.relative_to(path)): file.read_bytes()
            for file in path.rglob("*.safetensors")
        }
    )
    return files


def test_04_roundtrip_reshard_identity(checkpoints, tmp_path):
    source, _oracle = checkpoints(2)
    middle, final = tmp_path / "middle-3", tmp_path / "final-2"
    _run(3, "load", source, "--source-world", 2, "--output", middle)
    _run(2, "load", middle, "--source-world", 3, "--output", final)
    assert _identity_files(source) == _identity_files(final)
    print(
        "ACCEPTANCE 4 PASS 2->3->2 tensor files and manifest are byte-identical",
        flush=True,
    )


def test_05_independent_reader(checkpoints):
    source, oracle = checkpoints(2)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.gdcp.independent_reader",
            str(source),
            str(oracle),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    print(result.stdout, flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    reconstructed, _coverage, _manifest = independent_reader(source)
    expected = json.loads(oracle.read_text())
    assert reconstructed == expected


def test_06_replicated_tensor_dedup(checkpoints):
    source, _oracle = checkpoints(2)
    manifest = json.loads((source / "gdcp_manifest.json").read_text())
    checked = []
    for component, description in manifest["components"].items():
        counts = _tensor_entry_counts(source / component, description)
        for tensor in description["tensors"]:
            _assert_dedup(component, tensor, counts, checked)
    assert "model.running_mean" in checked and "model.count" in checked
    assert sum(key.endswith(".step") for key in checked) == 3
    print(f"ACCEPTANCE 6 PASS replicated tensors stored once: {checked}", flush=True)


def _tensor_entry_counts(root, description):
    counts = {}
    for path in description["files"]:
        entries = dict(safetensors.deserialize((root / path).read_bytes()))
        for name in entries:
            key = name.partition("/slot")[0]
            counts[key] = counts.get(key, 0) + 1
    return counts


def _assert_dedup(component, tensor, counts, checked):
    replicated = tensor["placements"] == [{"replicate": None}]
    nonempty = all(size > 0 for size in tensor["shape"])
    if replicated and nonempty:
        assert counts[tensor["id"]] == 1, (component, tensor)
        checked.append(f"{component}.{tensor['id']}")


def test_07_atomicity_rank_fault_and_retry(checkpoints, tmp_path):
    source, _oracle = checkpoints(2)
    _run(2, "atomic", source, "--output", tmp_path / "atomic")


def test_08_strict_metadata_before_mutation(checkpoints, tmp_path):
    source, _oracle = checkpoints(2)
    _run(2, "strict", source, "--output", tmp_path / "strict")


def test_09_extra_state_per_rank_and_fallback(checkpoints, tmp_path):
    source, _oracle = checkpoints(2)
    _run(2, "load", source, "--source-world", 2)
    output = _run(3, "load", source, "--source-world", 2)
    assert "ACCEPTANCE 9 fallback warning=" in output
    _run(2, "extra_missing", source, "--output", tmp_path / "missing-own-extra")


def test_10_partial_state(checkpoints, tmp_path):
    source, _oracle = checkpoints(2)
    _run(2, "partial", source, "--output", tmp_path / "partial")


def test_11_global_steps(checkpoints, tmp_path):
    source, _oracle = checkpoints(2)
    _run(2, "global_steps", source, "--output", tmp_path / "global")


def test_13_inspect_without_tensor_open(checkpoints, monkeypatch):
    from gdcp.core import inspect

    source, _oracle = checkpoints(2)
    original_open, original_io_open = builtins.open, io.open
    opened = []

    def checked_open(file, *args, **kwargs):
        opened.append(str(file))
        assert not str(file).endswith(".safetensors")
        return original_open(file, *args, **kwargs)

    def checked_io_open(file, *args, **kwargs):
        opened.append(str(file))
        assert not str(file).endswith(".safetensors")
        return original_io_open(file, *args, **kwargs)

    def blocked_pread(*args, **kwargs):
        raise AssertionError("inspect opened a tensor file through FileIO")

    monkeypatch.setattr(builtins, "open", checked_open)
    monkeypatch.setattr(io, "open", checked_io_open)
    monkeypatch.setattr(FileIO, "pread", blocked_pread)
    result = inspect(source)
    assert type(result) is Ok, result
    description = result.value
    assert description is not None
    assert description.world_size == 2
    tensors = {
        tensor["id"]: tensor for tensor in description.components["model"]["tensors"]
    }
    assert tensors["weight"]["shape"] == [10, 4]
    assert tensors["tiny"]["dtype"] == "BF16"
    assert tensors["count"]["logical_shape"] == []
    assert set(tensors) == {"weight", "tiny", "bias", "running_mean", "count", "empty"}
    assert any(path.endswith("gdcp_manifest.json") for path in opened)
    print(
        "ACCEPTANCE 13 PASS inspect returned saved world, keys, shapes and dtypes with no tensor opens",
        flush=True,
    )


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
def test_general_optimizer_state_layouts(tmp_path, backend):
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < 4 or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("requires four CUDA devices and NCCL")
    output = _run(4, "optimizer_matrix", tmp_path, "--backend", backend)
    assert output.count("optimizer matrix PASS") == 4


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
@pytest.mark.parametrize("world", [4, 8])
def test_restored_optimizer_state_reuses_parameter_meshes(tmp_path, backend, world):
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < world or not torch.distributed.is_nccl_available()
    ):
        pytest.skip(f"requires {world} CUDA devices and NCCL")
    output = _run(world, "optimizer_meshes", tmp_path, "--backend", backend)
    assert output.count("optimizer meshes PASS") == world
