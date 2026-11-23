"""Focused adapter invariants independent of model size and rank launcher."""

import ctypes
import json
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from shutil import copytree
from threading import local
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from torch.distributed import HashStore, destroy_process_group, init_process_group
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint import save as torch_dcp_save
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard

from gdcp import FSDP2Checkpointer, core, desc, hf, load_hf, save_hf
from gdcp.adapters import dcp, fsdp2, torch_access
from gdcp.adapters import hf as hf_adapter
from gdcp.adapters.torch_access import TorchAccessor
from gdcp.hf import read_headers
from gdcp.state import CheckpointState, RankContext
from gdtensor.builder import Err, Ok
from gdtensor_conn.coord import GlooCoordinator, NcclCoordinator, ThreadCoordinator
from gdtensor_conn.iobackend import FileIO

_FORMATS = ("safetensors", "dcp", "hf")


def test_hf_dtype_inference_failure_votes_on_every_rank(tmp_path, monkeypatch):
    coordinator = ThreadCoordinator(2, timeout=2)
    context = local()
    monkeypatch.setattr(hf_adapter, "_context", lambda: context.value)
    monkeypatch.setattr(
        hf_adapter,
        "read_headers",
        lambda path: {"weight": {"dtype": "F32"}} if context.value.rank else {},
    )

    def load(rank):
        context.value = RankContext(rank, 2, coordinator)
        expected = "another rank failed" if rank else "cannot infer HF dtype"
        with pytest.raises((ValueError, RuntimeError), match=expected):
            hf_adapter.load_hf(tmp_path, {"weight": torch.zeros(1)})

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))


def _batch_norm(dtype, *, initialized):
    model = torch.nn.BatchNorm1d(3, dtype=dtype)
    model.register_buffer("mask", torch.tensor([initialized, False, initialized]))
    for bits in (16, 32, 64):
        values = [0, 2**bits - 1, 2 ** (bits - 1) + 1] if initialized else [0, 0, 0]
        name = f"uint{bits}"
        model.register_buffer(name, torch.tensor(values, dtype=getattr(torch, name)))
    if initialized:
        model.num_batches_tracked.fill_(16777217)
        model.running_mean.copy_(torch.tensor([1.125, -2.25, 3.5]))
    return model


def _assert_tensor_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
    )


def _assert_hf_weights(checkpoint, headers, source, target, dtype):
    from safetensors import safe_open

    for name, value in source.items():
        expected = value.to(dtype) if value.is_floating_point() else value
        with safe_open(checkpoint / headers[name]["file"], framework="pt") as file:
            saved = file.get_tensor(name)
        _assert_tensor_bits(saved, expected)
        _assert_tensor_bits(target[name], expected)


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_hf_weights_round_trip_integer_and_bool_buffers(
    tmp_path, monkeypatch, world, dtype
):
    checkpoint = tmp_path / "weights"
    torch_dtype, code = {
        "fp32": (torch.float32, "F32"),
        "bf16": (torch.bfloat16, "BF16"),
    }[dtype]
    source_dtype = torch.bfloat16 if dtype == "fp32" else torch.float32
    context = local()
    coordinator = ThreadCoordinator(world, timeout=5) if world > 1 else None
    monkeypatch.setattr(hf_adapter, "_context", lambda: context.value)

    def round_trip(rank):
        context.value = RankContext(rank, world, coordinator)
        source = _batch_norm(source_dtype, initialized=True)
        target = _batch_norm(torch_dtype, initialized=False)
        save_hf(
            checkpoint, source, dtype=dtype, max_shard_size=16 if world > 1 else None
        )
        headers = read_headers(checkpoint)
        assert headers["num_batches_tracked"]["dtype"] == "I64"
        assert headers["mask"]["dtype"] == "BOOL"
        for bits in (16, 32, 64):
            assert headers[f"uint{bits}"]["dtype"] == f"U{bits}"
        assert headers["running_mean"]["dtype"] == code
        load_hf(checkpoint, target)
        _assert_hf_weights(
            checkpoint, headers, source.state_dict(), target.state_dict(), torch_dtype
        )

    with ThreadPoolExecutor(max_workers=world) as pool:
        list(pool.map(round_trip, range(world)))


def test_hf_weights_load_integer_and_bool_without_float_dtype(tmp_path):
    from safetensors.torch import save_file

    tensors = {"counter": torch.tensor(16777217), "mask": torch.tensor([True, False])}
    save_file(tensors, tmp_path / "model.safetensors")
    for dtype in (None, "bf16"):
        target = {name: torch.zeros_like(value) for name, value in tensors.items()}
        load_hf(tmp_path, target, dtype=dtype)
        for name, value in tensors.items():
            _assert_tensor_bits(target[name], value)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_full_state_round_trip_integer_and_bool_buffers(tmp_path, monkeypatch, fmt):
    monkeypatch.setattr(fsdp2, "FSDPModule", torch.nn.Module)
    source = _batch_norm(torch.float32, initialized=True)
    target = _batch_norm(torch.float32, initialized=False)
    checkpoint = tmp_path / "checkpoint"
    FSDP2Checkpointer.save(checkpoint, {"model": source}, checkpoint_format=fmt)
    FSDP2Checkpointer.load(checkpoint, {"model": target})
    restored = target.state_dict()
    for name, value in source.state_dict().items():
        _assert_tensor_bits(restored[name], value)


@pytest.mark.parametrize("bits", [16, 32, 64])
def test_hf_weights_validate_optimizer_prefixed_model_keys(tmp_path, bits):
    import numpy as np
    from safetensors.numpy import save_file

    key = "optimizer.ids"
    values = [0, 2**bits - 1]
    save_file(
        {"weight": np.ones(2, dtype=np.float32), key: np.array(values, f"uint{bits}")},
        tmp_path / "model.safetensors",
    )
    weight = torch.zeros(2)
    with pytest.raises(RuntimeError, match=r"HF unexpected key optimizer\.ids"):
        load_hf(tmp_path, {"weight": weight})
    _assert_tensor_bits(weight, torch.zeros(2))
    with pytest.raises(
        RuntimeError, match=r"HF shape/dtype mismatch for optimizer\.ids"
    ):
        load_hf(tmp_path, {"weight": weight, key: torch.zeros(2, dtype=torch.int64)})
    _assert_tensor_bits(weight, torch.zeros(2))
    target_dtype = getattr(torch, f"uint{bits}")
    target = torch.zeros(2, dtype=target_dtype)
    load_hf(tmp_path, {"weight": weight, key: target})
    _assert_tensor_bits(target, torch.tensor(values, dtype=target_dtype))
    _assert_tensor_bits(weight, torch.ones(2))


def test_hf_weights_can_overwrite_native_unsigned_checkpoint(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            f"uint{bits}": torch.zeros(2, dtype=getattr(torch, f"uint{bits}"))
            for bits in (16, 32, 64)
        },
        tmp_path / "model.safetensors",
    )
    save_hf(tmp_path, {"weight": torch.ones(2)})
    assert set(read_headers(tmp_path)) == {"weight"}


@pytest.fixture(params=[1, 2])
def sharded_world(request, monkeypatch):
    world = request.param
    init_process_group("fake", store=HashStore(), rank=0, world_size=world)
    context = local()
    coordinator = ThreadCoordinator(world, timeout=5) if world > 1 else None
    monkeypatch.setattr(fsdp2, "FSDPModule", torch.nn.Module)
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context.value)
    monkeypatch.setattr(hf_adapter, "_context", lambda: context.value)
    try:
        yield DeviceMesh("cpu", list(range(world))), world, context, coordinator
    finally:
        destroy_process_group()


def _sharded_buffer(mesh, values, rank, world):
    buffer = DTensor.from_local(
        values.chunk(world)[rank].clone(),
        mesh,
        [Shard(0)],
        run_check=False,
        shape=values.shape,
        stride=values.stride(),
    )
    model = torch.nn.Module()
    model.register_buffer("ids", buffer)
    return model


@pytest.mark.parametrize("mode", ["fp32", "bf16", "native_hf", *_FORMATS])
def test_sharded_int16_buffer_round_trip(tmp_path, sharded_world, mode):
    from safetensors.torch import save_file

    mesh, world, context, coordinator = sharded_world
    values = torch.tensor([-32768, -1234, 0, 17, 32767], dtype=torch.int16)
    if mode == "native_hf":
        save_file({"ids": values}, tmp_path / "model.safetensors")

    def round_trip(rank):
        context.value = RankContext(rank, world, coordinator)
        source = _sharded_buffer(mesh, values, rank, world)
        target = _sharded_buffer(mesh, torch.zeros_like(values), rank, world)
        if mode in ("fp32", "bf16"):
            save_hf(tmp_path, source, dtype=mode)
        if mode in _FORMATS:
            FSDP2Checkpointer.save(tmp_path, {"model": source}, checkpoint_format=mode)
            FSDP2Checkpointer.load(tmp_path, {"model": target})
        else:
            assert read_headers(tmp_path)["ids"]["dtype"] == "I16"
            load_hf(tmp_path, target)
        _assert_tensor_bits(target.ids.to_local(), values.chunk(world)[rank])

    with ThreadPoolExecutor(max_workers=world) as pool:
        list(pool.map(round_trip, range(world)))


@pytest.mark.parametrize("fmt", _FORMATS)
def test_load_latest_numeric_committed_step(tmp_path, mesh, fmt):
    root = tmp_path / "run"
    for step in (10, 2):
        source = fully_shard(_model(float(step), "a"), mesh=mesh)
        FSDP2Checkpointer.save(
            root,
            {"model": source, "extra_state": step},
            global_steps=step,
            checkpoint_format=fmt,
        )
    (root / "global_step_20").mkdir()
    (root / "global_step_20" / "model.safetensors").touch()
    pending = root / "global_step_30"
    pending.mkdir()
    (pending / desc.PENDING_NAME).touch()
    (pending / desc.MANIFEST_NAME).touch()
    (root / "global_step_40").symlink_to(
        root / "global_step_2", target_is_directory=True
    )
    (root / ".global_step_100.gdcp-stage").mkdir()
    (root / "global_step_last").mkdir()
    for path, expected in ((root, 10), (root / "global_step_2", 2)):
        target = fully_shard(_model(0.0, "a"), mesh=mesh)
        state = {"model": target, "extra_state": None}
        FSDP2Checkpointer.load(path, state)
        assert state["extra_state"] == expected
        assert torch.all(target.a.to_local() == expected)


def test_latest_failed_publish_preserves_previous_step(tmp_path, monkeypatch):
    root = tmp_path / "run"
    FSDP2Checkpointer.save(root, {"extra_state": 2}, global_steps=2)

    def fail(transaction):
        raise OSError("injected publish failure")

    monkeypatch.setattr(core.DirectoryTransaction, "publish", fail)
    with pytest.raises(RuntimeError, match="injected publish failure"):
        FSDP2Checkpointer.save(root, {"extra_state": 3}, global_steps=3)
    result = {"extra_state": None}
    FSDP2Checkpointer.load(root, result)
    assert result["extra_state"] == 2


def test_latest_reports_missing_or_damaged_generation(tmp_path):
    state = {"extra_state": "untouched"}
    (tmp_path / "global_step_3").mkdir()
    with pytest.raises(RuntimeError, match="no committed global_step_N"):
        FSDP2Checkpointer.load(tmp_path, state)
    FSDP2Checkpointer.save(tmp_path, {"extra_state": 1}, global_steps=1)
    (tmp_path / "global_step_3" / desc.MANIFEST_NAME).write_text("{}")
    with pytest.raises(RuntimeError):
        FSDP2Checkpointer.load(tmp_path, state)
    assert state["extra_state"] == "untouched"


def test_latest_prefers_explicit_checkpoint_root(tmp_path):
    FSDP2Checkpointer.save(tmp_path, {"extra_state": 1})
    FSDP2Checkpointer.save(tmp_path, {"extra_state": 2}, global_steps=2)
    state = {"extra_state": None}
    FSDP2Checkpointer.load(tmp_path, state)
    assert state["extra_state"] == 1


def test_latest_agrees_on_step_across_rank_local_paths(tmp_path, monkeypatch):
    roots = [tmp_path / str(rank) for rank in range(2)]
    for rank, root in enumerate(roots):
        FSDP2Checkpointer.save(root, {"extra_state": rank}, global_steps=rank + 10)
    coordinator = ThreadCoordinator(2, timeout=5)
    context = local()
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context.value)

    def load(rank):
        context.value = RankContext(rank, 2, coordinator)
        state = {"extra_state": "untouched"}
        with pytest.raises(RuntimeError, match="ranks disagree"):
            FSDP2Checkpointer.load(roots[rank], state)
        assert state["extra_state"] == "untouched"

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))


def test_fsdp2_format_lookup_failure_votes_on_every_rank(tmp_path, monkeypatch):
    FSDP2Checkpointer.save(tmp_path, {})
    coordinator = ThreadCoordinator(2, timeout=2)
    context = local()
    is_file = Path.is_file
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context.value)

    def rank_local_is_file(path):
        if path == tmp_path / desc.MANIFEST_NAME and context.value.rank == 0:
            raise OSError("injected checkpoint lookup failure")
        return is_file(path)

    monkeypatch.setattr(Path, "is_file", rank_local_is_file)

    def load(rank):
        context.value = RankContext(rank, 2, coordinator)
        expected = "another rank failed" if rank else "checkpoint lookup failure"
        with pytest.raises(RuntimeError, match=expected):
            fsdp2.load_checkpoint(tmp_path, {}, common_only=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize(
    "parts",
    [
        (),
        ("extra_state",),
        ("model",),
        ("model", "extra_state"),
        ("model", "optimizer"),
        ("model", "optimizer", "extra_state"),
    ],
    ids=["empty", "extra", "model", "model-extra", "model-optimizer", "all"],
)
def test_fsdp2_checkpoint_subsets_round_trip(mesh, tmp_path, monkeypatch, fmt, parts):
    source = _checkpoint_subset(mesh, parts, 1.0, 0.01, {"step": 7})
    target = _checkpoint_subset(mesh, parts, 9.0, 0.25, None)
    options = {"max_shard_size": 1} if fmt == "hf" else {}
    FSDP2Checkpointer.save(tmp_path, source, checkpoint_format=fmt, **options)
    FSDP2Checkpointer.load(tmp_path, target)
    if "model" in parts:
        assert _values(target["model"], target.get("optimizer")) == _values(
            source["model"], source.get("optimizer")
        )
    expected = {key: source[key] for key in ("extra_state",) if key in source}
    assert {key: target[key] for key in expected} == expected

    def reject_tensor_open(path, *args, **kwargs):
        if path.is_relative_to(tmp_path):
            allowed = {desc.MANIFEST_NAME, ".gdtensor_committed", "extra_state"}
            assert path.relative_to(tmp_path).parts[0] in allowed, path
        return original_open(path, *args, **kwargs)

    original_open = Path.open
    monkeypatch.setattr(Path, "open", reject_tensor_open)
    common = {}
    FSDP2Checkpointer.load(tmp_path, common, load_common_state_only=True)
    assert common == expected
    if "extra_state" in parts:
        (tmp_path / "extra_state" / "extra_state_rank_0.pt").unlink()
        with pytest.raises(RuntimeError, match="missing declared rank state file"):
            FSDP2Checkpointer.load(tmp_path, common, load_common_state_only=True)


def _checkpoint_subset(mesh, parts, value, lr, extra):
    state = {}
    if "model" in parts:
        state["model"] = fully_shard(_model(value, "a"), mesh=mesh)
    if "optimizer" in parts:
        state["optimizer"] = _optimizer(state["model"], lr)
    if "extra_state" in parts:
        state["extra_state"] = extra
    return state


@pytest.mark.parametrize("fmt", ["safetensors", "dcp"])
@pytest.mark.parametrize("directory", [True, False])
def test_fsdp2_manifest_wins_over_uncommitted_hf_names(tmp_path, fmt, directory):
    FSDP2Checkpointer.save(tmp_path, {"extra_state": 7}, checkpoint_format=fmt)
    for name in ("model.safetensors", "model.safetensors.index.json"):
        path = tmp_path / name
        if directory:
            path.mkdir()
        else:
            path.write_bytes(b"foreign file")
    common = {}
    FSDP2Checkpointer.load(tmp_path, common, load_common_state_only=True)
    assert common == {"extra_state": 7}


@pytest.mark.parametrize("fmt", _FORMATS)
def test_fsdp2_save_rejects_foreign_completion_marker(tmp_path, fmt):
    if fmt == "hf":
        marker = tmp_path / desc.MANIFEST_NAME
        document = {
            "gdcp_version": 1,
            "world_size": 1,
            "components": {"foreign": {"format": "unsupported"}},
            "rank_state": {},
        }
    else:
        marker = tmp_path / ".gdtensor_committed"
        document = {"format": "safetensors", "files": []}
    contents = desc.json_bytes(document)
    marker.write_bytes(contents)
    inode = marker.stat().st_ino

    with pytest.raises(RuntimeError, match=f"foreign entry: {marker.name}"):
        FSDP2Checkpointer.save(tmp_path, {"extra_state": 42}, checkpoint_format=fmt)

    assert set(tmp_path.iterdir()) == {marker}
    assert marker.read_bytes() == contents
    assert marker.stat().st_ino == inode


@pytest.mark.parametrize("fmt", _FORMATS)
def test_fsdp2_common_load_rejects_conflicting_completion_markers(tmp_path, fmt):
    source, other = tmp_path / "source", tmp_path / "other"
    other_fmt = "safetensors" if fmt == "hf" else "hf"
    marker = desc.MANIFEST_NAME if fmt == "hf" else ".gdtensor_committed"
    FSDP2Checkpointer.save(source, {"extra_state": 42}, checkpoint_format=fmt)
    FSDP2Checkpointer.save(other, {"extra_state": 13}, checkpoint_format=other_fmt)
    (source / marker).write_bytes((other / marker).read_bytes())

    target = {"extra_state": 0}
    with pytest.raises(RuntimeError, match="ambiguous checkpoint format"):
        FSDP2Checkpointer.load(source, target, load_common_state_only=True)
    assert target == {"extra_state": 0}


@pytest.mark.parametrize("fmt", ["safetensors", "dcp"])
def test_fsdp2_overwrite_replaces_owned_completion_marker(tmp_path, fmt):
    cases = [
        (fmt, desc.HF_COMMITTED_NAME),
        ("hf", desc.MANIFEST_NAME),
        (fmt, desc.HF_COMMITTED_NAME),
    ]
    for step, (checkpoint_format, previous_marker) in enumerate(cases):
        FSDP2Checkpointer.save(
            tmp_path, {"extra_state": step}, checkpoint_format=checkpoint_format
        )
        assert not (tmp_path / previous_marker).exists()
        target = {}
        FSDP2Checkpointer.load(tmp_path, target, load_common_state_only=True)
        assert target == {"extra_state": step}


def test_safetensors_manifest_wins_over_foreign_dcp_metadata(mesh, tmp_path):
    source = _checkpoint_subset(mesh, ("model",), 1.0, 0.01, None)
    target = _checkpoint_subset(mesh, ("model",), 9.0, 0.25, None)
    FSDP2Checkpointer.save(tmp_path, source)
    (tmp_path / "model" / ".metadata").write_bytes(b"foreign file")
    FSDP2Checkpointer.load(tmp_path, target)
    assert _values(target["model"]) == _values(source["model"])


def test_fsdp2_loads_legacy_hf_with_rank_state_only(tmp_path):
    FSDP2Checkpointer.save(tmp_path, {"extra_state": 7}, checkpoint_format="hf")
    (tmp_path / "model.safetensors").unlink(missing_ok=True)
    marker = tmp_path / ".gdtensor_committed"
    saved = json.loads(marker.read_bytes())
    saved["files"] = []
    marker.write_bytes(desc.json_bytes(saved))
    common = {}
    FSDP2Checkpointer.load(tmp_path, common, load_common_state_only=True)
    assert common == {"extra_state": 7}


def test_fsdp2_common_only_detects_hf_without_tensor_files(mesh, tmp_path):
    source = _checkpoint_subset(mesh, ("model", "extra_state"), 1.0, 0.01, 7)
    FSDP2Checkpointer.save(tmp_path, source, checkpoint_format="hf")
    (tmp_path / "model.safetensors").unlink()
    (tmp_path / "optimizer.json").unlink()
    (tmp_path / desc.MANIFEST_NAME).mkdir()
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / ".metadata").write_bytes(b"foreign file")
    common = {}
    FSDP2Checkpointer.load(tmp_path, common, load_common_state_only=True)
    assert common == {"extra_state": 7}


def test_empty_hf_weights_round_trip(tmp_path):
    save_hf(tmp_path, {})
    assert hf.read_headers(tmp_path) == {}
    load_hf(tmp_path, {}, dtype="fp32")


@pytest.mark.parametrize("native", [False, True])
def test_hf_weights_load_rejects_unexpected_optimizer_prefixed_model_key(
    tmp_path, native
):
    from safetensors.torch import save_file

    saved = {"optimizer.weight": torch.ones(2), "optimizer.bias": torch.ones(2)}
    if native:
        save_file(saved, tmp_path / "model.safetensors")
    else:
        save_hf(tmp_path, saved)

    target = {"optimizer.weight": torch.zeros(2)}
    with pytest.raises(RuntimeError, match=r"HF unexpected key optimizer\.bias"):
        load_hf(tmp_path, target)
    assert torch.equal(target["optimizer.weight"], torch.zeros(2))


def test_hf_model_only_load_distinguishes_weights_from_optimizer_slots(mesh, tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    model.optimizer = torch.nn.Linear(2, 2)
    source = fully_shard(model, mesh=mesh)
    optimizer = _optimizer(source, 0.01)
    FSDP2Checkpointer.save(
        tmp_path, {"model": source, "optimizer": optimizer}, checkpoint_format="hf"
    )

    target = torch.nn.Linear(2, 2, bias=False)
    target.optimizer = torch.nn.Linear(2, 2, bias=False)
    target = fully_shard(target, mesh=mesh)
    before = _values(target)
    with pytest.raises(RuntimeError, match=r"HF unexpected key optimizer\.bias"):
        FSDP2Checkpointer.load(tmp_path, {"model": target})
    assert _values(target) == before

    match = torch.nn.Linear(2, 2, bias=False)
    match.optimizer = torch.nn.Linear(2, 2)
    match = fully_shard(match, mesh=mesh)
    FSDP2Checkpointer.load(tmp_path, {"model": match})
    assert _values(match) == _values(source)


def _use_legacy_hf_metadata(path):
    # e471fa1 used the same tensor layout without either of these inventories.
    sidecar = path / "optimizer.json"
    objects = desc.read_objects(sidecar)
    del objects["/optimizer_tensors"]
    desc.write_objects(sidecar, objects)
    marker = path / desc.HF_COMMITTED_NAME
    completion = json.loads(marker.read_bytes())
    del completion["rank_state"]
    marker.write_bytes(desc.json_bytes(completion))


@pytest.mark.parametrize("fmt", (*_FORMATS, "legacy_hf"))
@pytest.mark.parametrize("prefixed", [False, True], ids=["plain", "optimizer-prefix"])
@pytest.mark.parametrize(
    "parts",
    [(), ("model",), ("optimizer",), ("model", "optimizer")],
    ids=["empty", "model", "optimizer", "both"],
)
def test_full_checkpoint_restores_requested_components(tmp_path, fmt, prefixed, parts):
    ctx = RankContext(0, 1)
    source, target = _checkpoint_linear(prefixed), _checkpoint_linear(prefixed)
    saved_optimizer, optimizer = _optimizer(source, 0.01), _optimizer(target, 0.25)
    for fields in saved_optimizer.state.values():
        fields["step"].fill_(7)
    save, load = {
        "safetensors": (core.save, core.load),
        "dcp": (dcp.save, dcp.load),
        "hf": (hf.save_state, hf.load_state),
        "legacy_hf": (hf.save_state, hf.load_state),
    }[fmt]
    saved = CheckpointState(
        [
            fsdp2._component("model", source.state_dict(), {}, ctx),
            _adamw_checkpoint_component(source, saved_optimizer, ctx),
        ],
        {"extra_state": b"saved"},
    )
    assert type(save(tmp_path, saved, ctx)) is Ok
    if fmt == "legacy_hf":
        _use_legacy_hf_metadata(tmp_path)

    targets = {
        "model": fsdp2._component("model", target.state_dict(), {}, ctx),
        "optimizer": _adamw_checkpoint_component(target, optimizer, ctx),
    }
    expected = _values(
        source if "model" in parts else target,
        saved_optimizer if "optimizer" in parts else optimizer,
    )
    requested = CheckpointState([targets[name] for name in parts], {"extra_state": b""})
    result = load(tmp_path, requested, ctx)
    assert type(result) is Ok, result
    assert requested.rank_state == saved.rank_state
    assert _values(target, optimizer) == expected
    if "optimizer" in parts:
        assert targets["optimizer"].objects == saved.components[1].objects


def _adamw_checkpoint_component(model, optimizer, ctx):
    tensors, objects, _ = fsdp2._optimizer_values(
        get_optimizer_state_dict(model, optimizer)
    )
    return fsdp2._component("optimizer", tensors, objects, ctx)


def _checkpoint_linear(prefixed, *, bias=True):
    linear = torch.nn.Linear(2, 2, bias=bias)
    if not prefixed:
        return linear
    model = torch.nn.Module()
    model.add_module("optimizer", linear)
    return model


@pytest.mark.parametrize("prefixed", [False, True], ids=["plain", "optimizer-prefix"])
def test_legacy_hf_model_only_still_validates_model_keys(
    tmp_path, monkeypatch, prefixed
):
    monkeypatch.setattr(fsdp2, "FSDPModule", torch.nn.Module)
    source = _checkpoint_linear(prefixed)
    FSDP2Checkpointer.save(
        tmp_path,
        {"model": source, "optimizer": _optimizer(source, 0.01)},
        checkpoint_format="hf",
    )
    _use_legacy_hf_metadata(tmp_path)
    target = _checkpoint_linear(prefixed, bias=False)
    before = _values(target)
    key = r"optimizer\.bias" if prefixed else "bias"
    with pytest.raises(RuntimeError, match=f"HF unexpected key {key}"):
        FSDP2Checkpointer.load(tmp_path, {"model": target})
    assert _values(target) == before


@pytest.fixture
def hf_component_checkpoint(tmp_path):
    ctx = RankContext(0, 1)
    model = {"weight": torch.ones(2), "optimizer.bias": torch.full((2,), 2.0)}
    optimizer = {"step": torch.full((2,), 3.0)}
    state = CheckpointState(
        [
            fsdp2._component("model", model, {}, ctx),
            fsdp2._component("optimizer", optimizer, {}, ctx),
        ]
    )
    assert type(hf.save_state(tmp_path, state, ctx)) is Ok
    return tmp_path, ctx


def test_hf_optimizer_only_ignores_optimizer_prefixed_model_key(
    hf_component_checkpoint,
):
    path, ctx = hf_component_checkpoint
    step = torch.zeros(2)
    target = CheckpointState([fsdp2._component("optimizer", {"step": step}, {}, ctx)])
    result = hf.load_state(path, target, ctx)
    assert type(result) is Ok, result
    assert torch.equal(step, torch.full((2,), 3.0))


@pytest.mark.parametrize("both", [False, True])
@pytest.mark.parametrize("owner", ["model", "optimizer"])
def test_hf_rejects_tensor_from_wrong_component(hf_component_checkpoint, both, owner):
    path, ctx = hf_component_checkpoint
    model = {"weight": torch.zeros(2)}
    optimizer = {}
    if owner == "optimizer":
        optimizer = {"step": torch.zeros(2), "bias": torch.zeros(2)}
    else:
        model["optimizer.bias"] = torch.zeros(2)
        model["optimizer.step"] = torch.zeros(2)
    tensors = {"model": model, "optimizer": optimizer}
    parts = tuple(tensors) if both else (owner,)
    target = CheckpointState(
        [fsdp2._component(name, tensors[name], {}, ctx) for name in parts]
    )
    result = hf.load_state(path, target, ctx)
    assert type(result) is Err, result
    key = "optimizer.bias" if owner == "optimizer" else "optimizer.step"
    assert f"HF missing key {key}" in result.value.msg
    assert all(torch.equal(tensor, torch.zeros(2)) for tensor in model.values())
    assert all(torch.equal(tensor, torch.zeros(2)) for tensor in optimizer.values())


def test_hf_model_alias_resolves_within_recorded_ownership(hf_component_checkpoint):
    path, ctx = hf_component_checkpoint
    bias, step = torch.zeros(2), torch.zeros(2)
    model = {"weight": torch.zeros(2), "optimizer.step": bias}
    aliases = {"aliases": {"optimizer.bias": "optimizer.step"}}
    target = CheckpointState(
        [
            fsdp2._component("model", model, aliases, ctx),
            fsdp2._component("optimizer", {"step": step}, {}, ctx),
        ]
    )
    result = hf.load_state(path, target, ctx)
    assert type(result) is Ok, result
    assert torch.equal(bias, torch.full((2,), 2.0))
    assert torch.equal(step, torch.full((2,), 3.0))


@pytest.mark.parametrize("dtype", ["bf16", None], ids=["explicit", "inferred"])
def test_hf_load_uses_model_alias_for_dtype(tmp_path, dtype):
    ctx = RankContext(0, 1)
    bias = torch.full((2,), 2.0, dtype=torch.bfloat16)
    saved = CheckpointState(
        [
            fsdp2._component("model", {"optimizer.bias": bias}, {}, ctx),
            fsdp2._component("optimizer", {"step": torch.full((2,), 3.0)}, {}, ctx),
        ]
    )
    assert type(hf.save_state(tmp_path, saved, ctx, max_shard_size=8)) is Ok

    target = torch.nn.Module()
    layer = torch.nn.Module()
    target.add_module("optimizer", layer)
    tied = torch.nn.Parameter(torch.zeros_like(bias))
    layer.register_parameter("step", tied)
    layer.register_parameter("bias", tied)

    load_hf(tmp_path, target, dtype=dtype)
    assert layer.step is layer.bias is tied
    assert torch.equal(tied, bias)


@pytest.mark.parametrize(
    "options, expected",
    [
        ({"checkpoint_format": "bad"}, "unsupported checkpoint_format"),
        ({"checkpoint_format": "dcp", "hf_dtype": "fp32"}, "require checkpoint_format"),
        ({"async_fast_checkpoint": True}, "async_fast_checkpoint"),
    ],
)
def test_invalid_fsdp2_save_arguments_vote_on_every_rank(
    tmp_path, monkeypatch, options, expected
):
    coordinator = ThreadCoordinator(2, timeout=2)
    context = local()
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context.value)

    def save(rank):
        context.value = RankContext(rank, 2, coordinator)
        match = "another rank failed" if rank else expected
        with pytest.raises(
            (ValueError, RuntimeError, NotImplementedError), match=match
        ):
            FSDP2Checkpointer.save(
                tmp_path / "checkpoint", {}, **(options if not rank else {})
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(save, range(2)))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "options",
    [
        ({"checkpoint_format": "safetensors"}, {"checkpoint_format": "dcp"}),
        ({"checkpoint_format": "hf"}, {"checkpoint_format": "hf", "hf_dtype": "bf16"}),
    ],
)
def test_fsdp2_save_routes_agree_before_dispatch(tmp_path, monkeypatch, options):
    coordinator = ThreadCoordinator(2, timeout=2)
    context = local()
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context.value)
    dispatched = []

    def track_dispatch(*args, **kwargs):
        dispatched.append(True)

    monkeypatch.setattr(core, "_save_agreed", track_dispatch)
    monkeypatch.setattr(dcp, "_save_agreed", track_dispatch)
    monkeypatch.setattr(hf, "_save_agreed", track_dispatch)

    def save(rank):
        context.value = RankContext(rank, 2, coordinator)
        with pytest.raises(RuntimeError, match="ranks disagree on .*format options"):
            FSDP2Checkpointer.save(tmp_path / "checkpoint", {}, **options[rank])

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(save, range(2)))
    assert not dispatched
    assert not list(tmp_path.iterdir())


def test_invalid_fsdp2_load_arguments_vote_on_every_rank(tmp_path, monkeypatch):
    coordinator = ThreadCoordinator(2, timeout=2)
    context = local()
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context.value)

    def load(rank):
        context.value = RankContext(rank, 2, coordinator)
        match = "another rank failed" if rank else "ignore_model_keys"
        with pytest.raises((RuntimeError, NotImplementedError), match=match):
            FSDP2Checkpointer.load(
                tmp_path,
                {},
                ignore_model_keys=None if rank else set(),
                load_common_state_only=True,
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))


def test_rejected_save_preserves_first_adamw_step(tmp_path, monkeypatch):
    current = local()
    coordinator = ThreadCoordinator(2, timeout=5)
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: current.ctx)
    monkeypatch.setattr(fsdp2, "FSDPModule", torch.nn.Module)

    def save(rank):
        current.ctx = RankContext(rank, 2, coordinator)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        with pytest.raises(RuntimeError, match="ranks disagree on .*format options"):
            FSDP2Checkpointer.save(
                tmp_path / "checkpoint",
                {"model": model, "optimizer": optimizer},
                checkpoint_format=("safetensors", "dcp")[rank],
            )
        assert not optimizer.state
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        assert all(fields["step"].item() == 1 for fields in optimizer.state.values())

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(save, range(2)))
    assert not list(tmp_path.iterdir())


def _range(key, offset, size):
    return SimpleNamespace(global_tensor_id=key, global_tensor_offset=offset, size=size)


def _read(accessor, key, offset, size):
    result = []

    def take(pointer, length):
        result.append(ctypes.string_at(pointer, length))
        return Ok(None)

    assert type(accessor.read_to_buffer(_range(key, offset, size), take)) is Ok
    return b"".join(result)


def _write(accessor, key, offset, data):
    def give(pointer, length):
        assert len(data) == length
        ctypes.memmove(pointer, data, length)
        return Ok(None)

    result = accessor.write_from_buffer(_range(key, offset, len(data)), give)
    assert type(result) is Ok, result


def test_staging_reuses_one_host_tensor_for_all_save_ranges(monkeypatch):
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    accessor = TorchAccessor({"first": tensor, "second": tensor[:2]}, {})
    calls, original = [], torch.Tensor.cpu

    def cpu(value, *args, **kwargs):
        calls.append(value.numel())
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    first = _read(accessor, "first", 0, 8)
    assert accessor._host is not None
    pointer = accessor._host.data_ptr()
    rest = _read(accessor, "first", 8, 40)
    assert accessor._host is not None
    assert accessor._host.data_ptr() == pointer
    assert first + rest == tensor.contiguous().view(torch.uint8).numpy().tobytes()
    assert calls == [12]
    _read(accessor, "second", 0, 24)
    assert calls == [12, 6]
    assert accessor._host is not None
    assert accessor._host.numel() == 6
    accessor.finish()
    assert accessor._host is None


def test_load_staging_writes_back_once_after_all_ranges(monkeypatch):
    tensor = torch.full((3, 4), -1.0).t()
    expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    raw = expected.view(torch.uint8).numpy().tobytes()
    accessor = TorchAccessor({"weight": tensor}, {})
    calls, original = [], torch_access._finish_write

    def finish(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(torch_access, "_finish_write", finish)
    _write(accessor, "weight", 0, raw[:8])
    _write(accessor, "weight", 8, raw[8:])
    assert torch.equal(tensor, torch.full_like(tensor, -1))
    assert calls == []
    accessor.finish()
    assert calls == ["weight"]
    assert torch.equal(tensor, expected)
    assert accessor._host is None


def test_incomplete_load_rejected_before_staging_writeback():
    tensor = torch.full((2, 3), -1.0).t()
    accessor = TorchAccessor({"weight": tensor}, {})
    _write(accessor, "weight", 0, bytes(4))
    with pytest.raises(ValueError, match="incomplete local tensor load"):
        accessor.finish()
    assert torch.equal(tensor, torch.full_like(tensor, -1))


@pytest.mark.parametrize(
    "backend", ["gloo", "cuda:nccl,cpu:gloo", "cpu:gloo,cuda:nccl"]
)
def test_context_accepts_cpu_gloo_in_composite_group(monkeypatch, backend):
    monkeypatch.setattr(fsdp2, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp2, "get_backend", lambda group=None: backend)
    monkeypatch.setattr(fsdp2, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(fsdp2, "get_world_size", lambda group=None: 4)
    ctx = fsdp2._context()
    assert (ctx.rank, ctx.world_size) == (1, 4)
    assert isinstance(ctx.coordinator, GlooCoordinator)


@pytest.mark.parametrize("backend", ["nccl", "cuda:nccl"])
@pytest.mark.parametrize("bound_device", [None, torch.device("cuda:2")])
def test_context_accepts_cuda_nccl(monkeypatch, backend, bound_device):
    monkeypatch.setattr(fsdp2, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp2, "get_backend", lambda group=None: backend)
    monkeypatch.setattr(fsdp2, "get_rank", lambda group=None: 5)
    monkeypatch.setattr(fsdp2, "get_world_size", lambda group=None: 8)
    monkeypatch.setattr(
        fsdp2,
        "group",
        SimpleNamespace(WORLD=SimpleNamespace(bound_device_id=bound_device)),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    ctx = fsdp2._context()
    assert isinstance(ctx.coordinator, NcclCoordinator)
    assert ctx.coordinator.device_index == (2 if bound_device is not None else 1)


def test_context_rejects_unsupported_backend(monkeypatch):
    monkeypatch.setattr(fsdp2, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp2, "get_backend", lambda group=None: "mpi")
    with pytest.raises(NotImplementedError, match="CPU Gloo or CUDA NCCL"):
        fsdp2._context()


def test_strided_hsdp_keeps_replica_coordinates_on_rectangular_mesh():
    from itertools import product
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.placement_types import _StridedShard

    ranks = torch.arange(12).reshape(2, 3, 2)
    source = SimpleNamespace(
        device_mesh=SimpleNamespace(mesh=ranks),
        placements=(Replicate(), _StridedShard(1, split_factor=2), Shard(1)),
    )
    ordered, placements = fsdp2._ordered_shards("weight", source)
    assert placements == [Replicate(), Shard(1), Shard(1)]
    assert ordered.shape == (2, 2, 3)
    for replica, dp, tp in product(range(2), range(3), range(2)):
        assert ordered[replica, tp, dp] == ranks[replica, dp, tp]
    source.placements = (Replicate(), _StridedShard(1, split_factor=3), Shard(1))
    with pytest.raises(NotImplementedError, match="strided DP/TP"):
        fsdp2._ordered_shards("weight", source)


def test_tied_parameter_has_one_tensor_and_alias_metadata():
    model = torch.nn.Module()
    model.embed_tokens = torch.nn.Embedding(7, 4)
    model.lm_head = torch.nn.Linear(4, 7, bias=False)
    model.lm_head.weight = model.embed_tokens.weight
    aliases = fsdp2._model_aliases(model)
    assert aliases == {"lm_head.weight": "embed_tokens.weight"}
    values = fsdp2._model_values(model)
    assert list(values) == ["embed_tokens.weight"]
    component = fsdp2._component(
        "model", values, {"aliases": aliases}, RankContext(0, 1)
    )
    assert component.table.named_d_tensors[0].alias == ["lm_head.weight"]


def test_optimizer_tensor_fields_validate_geometry_and_preserve_dtypes():
    model = torch.nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)
    fields: dict[str, Any] = {
        "step": torch.tensor(3.0),
        "master_param": torch.ones((7, 4), dtype=torch.float32),
        "exp_avg": torch.zeros((7, 4), dtype=torch.float32),
        "exp_avg_sq": torch.ones((7, 4), dtype=torch.float32),
        "another_moment": torch.ones((7, 4), dtype=torch.float64),
        "scalar_flag": 3,
    }
    fsdp2._validate_optimizer_tensors(model, {"state": {"weight": fields}})
    assert fields["another_moment"].dtype == torch.float64
    fields["another_moment"] = torch.zeros(2)
    fsdp2._validate_optimizer_tensors(model, {"state": {"weight": fields}})
    assert fields["another_moment"].shape == (2,)


def test_optimizer_restore_preserves_fp32_master_and_extra_fields():
    model = torch.nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    values = cast(dict[str, Any], fsdp2.get_optimizer_state_dict(model, optimizer))
    fields = values["state"]["weight"]
    fields["master_param"] = torch.arange(28, dtype=torch.float32).reshape(7, 4) / 17
    fields["exp_avg"] = torch.ones((7, 4), dtype=torch.float32) / 19
    fields["extra_moment"] = torch.ones((7, 4), dtype=torch.float64) / 23
    adapter = fsdp2.FSDP2Adapter(loading=True)
    adapter.optimizer_state = values
    _tensors, objects, adapter.optimizer_paths = fsdp2._optimizer_values(values)
    target = {"model": model, "optimizer": optimizer}
    adapter.prepare_restore(target, {"optimizer": objects})
    adapter.restore(target, CheckpointState())
    restored = optimizer.state[model.weight]
    assert restored["master_param"] is fields["master_param"]
    assert restored["exp_avg"].dtype == torch.float32
    assert restored["extra_moment"].dtype == torch.float64
    assert torch.equal(restored["master_param"], fields["master_param"])


def test_optimizer_adamw_subclass_is_supported(monkeypatch):
    class MixedAdamW(torch.optim.AdamW):
        pass

    monkeypatch.setattr(fsdp2, "FSDPModule", torch.nn.Module)
    model = torch.nn.Linear(2, 3)
    optimizer = MixedAdamW(model.parameters())
    fsdp2._validate_state({"model": model, "optimizer": optimizer}, common_only=False)
    fsdp2._validate_state(
        {"model": model, "optimizer": torch.optim.SGD(model.parameters())},
        common_only=False,
    )


def test_invalid_accessor_range_returns_result_error():
    accessor = TorchAccessor({"weight": torch.zeros(2)}, {})
    result = accessor.read_to_buffer(_range("weight", 0, 12), lambda *args: Ok(None))
    assert type(result) is Err


@pytest.fixture
def mesh(monkeypatch):
    pytest.importorskip("torch.testing._internal.distributed.fake_pg")
    init_process_group("fake", store=HashStore(), rank=0, world_size=1)
    context = RankContext(0, 1)
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context)
    monkeypatch.setattr(hf_adapter, "_context", lambda: context)
    try:
        yield DeviceMesh("cpu", [0])
    finally:
        destroy_process_group()


def _model(value, tied_to):
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.full((4,), value))
    model.b = torch.nn.Parameter(torch.full((4,), value + 1))
    model.alias = getattr(model, tied_to)
    return model


def _optimizer(model, lr, *, split=False, fresh=False):
    params = (
        [{"params": [model.a]}, {"params": [model.b]}] if split else model.parameters()
    )
    optimizer = torch.optim.AdamW(params, lr=lr, foreach=False)
    if not fresh:
        for param in model.parameters():
            param.grad = torch.full_like(param, lr)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return optimizer


@pytest.mark.parametrize("fmt", _FORMATS)
def test_save_rejects_colliding_adamw_fields(mesh, tmp_path, fmt):
    model = fully_shard(_model(1.0, "a"), mesh=mesh)
    optimizer = _optimizer(model, 0.01)
    fields = optimizer.state[model.a]
    fields[1], fields["1"] = 10, 20
    with pytest.raises(ValueError, match="optimizer state key collision"):
        FSDP2Checkpointer.save(
            tmp_path / "checkpoint",
            {"model": model, "optimizer": optimizer},
            checkpoint_format=fmt,
        )
    assert fields[1] == 10 and fields["1"] == 20
    assert not list(tmp_path.iterdir())


def _values(model, optimizer=None):
    tensors = list(model.state_dict().values())
    if optimizer is not None:
        tensors += [
            value for fields in optimizer.state.values() for value in fields.values()
        ]
    return [
        (tensor.to_local() if isinstance(tensor, DTensor) else tensor).detach().tolist()
        for tensor in tensors
    ]


def _load_rejected(path, state, match):
    with pytest.raises((ValueError, RuntimeError), match=match):
        FSDP2Checkpointer.load(path, state)


@pytest.mark.parametrize("fmt", (*_FORMATS, "native_dcp"))
def test_alias_validation_preserves_model(mesh, tmp_path, fmt):
    source = fully_shard(_model(1.0, "a"), mesh=mesh)
    target = fully_shard(_model(8.0, "b"), mesh=mesh)
    checkpoint_format = "dcp" if fmt == "native_dcp" else fmt
    FSDP2Checkpointer.save(
        tmp_path, {"model": source}, checkpoint_format=checkpoint_format
    )
    if fmt == "native_dcp":
        (tmp_path / desc.MANIFEST_NAME).unlink()
        (tmp_path / "model" / "objects.json").unlink()
    before = _values(target)
    error = RuntimeError if fmt == "native_dcp" else ValueError
    with pytest.raises(error, match="tied.*alias"):
        FSDP2Checkpointer.load(tmp_path, {"model": target})
    assert _values(target) == before
    assert target.alias is target.b


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("fresh", (False, True))
def test_optimizer_group_validation_preserves_live_state(mesh, tmp_path, fmt, fresh):
    source = fully_shard(_model(1.0, "a"), mesh=mesh)
    target = fully_shard(_model(8.0, "a"), mesh=mesh)
    saved = _optimizer(source, 0.01, split=True)
    optimizer = _optimizer(target, 0.25, fresh=fresh)
    extra_state = {"step": -1}
    state = {"model": target, "optimizer": optimizer, "extra_state": extra_state}
    FSDP2Checkpointer.save(
        tmp_path,
        {"model": source, "optimizer": saved, "extra_state": {"step": 1}},
        checkpoint_format=fmt,
    )
    before = _values(target, optimizer)
    _load_rejected(tmp_path, state, "optimizer parameter group count")
    assert _values(target, optimizer) == before
    assert optimizer.param_groups[0]["lr"] == 0.25
    assert state["extra_state"] is extra_state


def _grouped_model(mesh, value):
    return fully_shard(
        torch.nn.ParameterDict(
            {name: torch.nn.Parameter(torch.full((4,), value)) for name in "abcd"}
        ),
        mesh=mesh,
    )


def _grouped_optimizer(model, orders, *, named=False, initialized=True):
    parameters = dict(model.named_parameters())
    groups = [
        {
            "params": [
                (name, parameters[name]) if named else parameters[name]
                for name in order
            ],
            "lr": (index + 1) / 100,
        }
        for index, order in enumerate(orders)
    ]
    optimizer = torch.optim.AdamW(groups, foreach=False)
    if not initialized:
        return optimizer
    optimized = {
        parameter for group in optimizer.param_groups for parameter in group["params"]
    }
    for index, parameter in enumerate(parameters.values(), start=1):
        if parameter not in optimized:
            continue
        optimizer.state[parameter] = {
            "step": torch.tensor(float(index)),
            "exp_avg": torch.full_like(parameter, 2 * index),
            "exp_avg_sq": torch.full_like(parameter, 3 * index),
        }
    return optimizer


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize(
    "saved_named,target_named",
    [(False, True), (True, False), (True, True), (False, False)],
)
@pytest.mark.parametrize("reorder_groups", [False, True])
def test_optimizer_group_names_and_slots_follow_saved_parameters(
    mesh, tmp_path, fmt, saved_named, target_named, reorder_groups
):
    source, target = _grouped_model(mesh, 1.0), _grouped_model(mesh, 9.0)
    saved = _grouped_optimizer(source, ("ab", "cd"), named=saved_named)
    target_orders = ("dc", "ba") if reorder_groups else ("ba", "dc")
    optimizer = _grouped_optimizer(
        target, target_orders, named=target_named, initialized=False
    )
    parameters = dict(target.named_parameters())
    FSDP2Checkpointer.save(
        tmp_path, {"model": source, "optimizer": saved}, checkpoint_format=fmt
    )
    FSDP2Checkpointer.load(tmp_path, {"model": target, "optimizer": optimizer})
    for group, order, saved_group in zip(
        optimizer.param_groups, ("ab", "cd"), saved.param_groups, strict=True
    ):
        assert group["param_names"] == list(order)
        assert all(
            param is parameters[name]
            for name, param in zip(order, group["params"], strict=True)
        )
        assert group["lr"] == saved_group["lr"]
    assert _values(target) == _values(source)
    _assert_optimizer_slots(source, saved, parameters, optimizer)


def _assert_optimizer_slots(source, saved, parameters, optimizer):
    assert optimizer.state.keys() == set(parameters.values())
    for name, param in source.named_parameters():
        for field, value in saved.state[param].items():
            actual = optimizer.state[parameters[name]][field]
            torch.testing.assert_close(
                actual.to_local() if isinstance(actual, DTensor) else actual,
                value.to_local() if isinstance(value, DTensor) else value,
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize(
    "saved_orders,target_orders,error",
    [
        (("ab", "cd"), ("ac", "bd"), "members do not match"),
        pytest.param(("ab",), ("ac",), "members do not match", id="different-subsets"),
        (("abcd", "", ""), ("", "abcd", ""), "ambiguous"),
    ],
)
def test_unreconcilable_optimizer_groups_fail_before_payload_reads(
    mesh, tmp_path, monkeypatch, fmt, saved_orders, target_orders, error
):
    source, target = _grouped_model(mesh, 1.0), _grouped_model(mesh, 9.0)
    saved = _grouped_optimizer(source, saved_orders)
    optimizer = _grouped_optimizer(target, target_orders)
    extra_state = {"step": -1}
    state = {"model": target, "optimizer": optimizer, "extra_state": extra_state}
    FSDP2Checkpointer.save(
        tmp_path,
        {"model": source, "optimizer": saved, "extra_state": {"step": 1}},
        checkpoint_format=fmt,
    )
    before = _values(target, optimizer)
    groups_before = optimizer.param_groups
    read_bytes = Path.read_bytes

    def read_metadata_only(path):
        assert path.suffix != ".pt", f"read rank-state payload {path.name}"
        return read_bytes(path)

    def unexpected_read(*args):
        pytest.fail("read tensor payload before validating optimizer groups")

    monkeypatch.setattr(Path, "read_bytes", read_metadata_only)
    monkeypatch.setattr(FileIO, "pread", unexpected_read)
    _load_rejected(tmp_path, state, f"optimizer parameter group.*{error}")
    assert _values(target, optimizer) == before
    assert optimizer.param_groups is groups_before
    assert state["extra_state"] is extra_state


def _native_grouped_optimizer(value, orders, *, named=False):
    model = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(torch.full((4,), value)) for name in "abcd"}
    )
    optimizer = _grouped_optimizer(model, orders, named=named)
    for index, parameter in enumerate(model.parameters(), start=1):
        optimizer.state[parameter]["scalar_flag"] = value * index
    return model, optimizer


@pytest.fixture
def native_optimizer_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(fsdp2, "FSDPModule", torch.nn.Module)
    source, optimizer = _native_grouped_optimizer(1.0, ("ab", "cd"))
    torch_dcp_save(source.state_dict(), checkpoint_id=tmp_path / "model", no_dist=True)
    torch_dcp_save(
        get_optimizer_state_dict(source, optimizer),
        checkpoint_id=tmp_path / "optimizer",
        no_dist=True,
    )
    return source, optimizer


@pytest.mark.parametrize("corrupt", [None, "state.a.scalar_flag", "param_groups.0.lr"])
def test_native_dcp_group_validation_precedes_payload_reads_on_every_rank(
    native_optimizer_checkpoint, tmp_path, monkeypatch, corrupt
):
    root = tmp_path / "optimizer"
    metadata = dcp.read_metadata(root)
    keys_by_record = {
        (record.relative_path, record.offset): index.fqn
        for index, record in metadata.storage_data.items()
    }
    if corrupt is not None:
        record = next(
            record
            for index, record in metadata.storage_data.items()
            if index.fqn == corrupt
        )
        with (root / record.relative_path).open("r+b") as handle:
            handle.seek(record.offset)
            handle.write(bytes(record.length))

    coordinator = ThreadCoordinator(2, timeout=10)
    context = local()
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context.value)
    read_object = dcp._read_object
    reads = []

    def record_read(path, record, *, weights_only=False):
        reads.append(
            (
                context.value.rank,
                keys_by_record[record.relative_path, record.offset],
                weights_only,
            )
        )
        return read_object(path, record, weights_only=weights_only)

    def unexpected_tensor_read(*args):
        pytest.fail("read native tensor record before validating optimizer groups")

    monkeypatch.setattr(dcp, "_read_object", record_read)
    monkeypatch.setattr(dcp, "_native_payload", unexpected_tensor_read)

    def load(rank):
        context.value = RankContext(rank, 2, coordinator)
        orders = ("ac", "bd") if rank else ("dc", "ba")
        target, optimizer = _native_grouped_optimizer(9.0, orders, named=True)
        expected = (
            "optimizer parameter group members do not match"
            if rank
            else "another rank failed"
        )
        groups_before = optimizer.param_groups
        _load_rejected(tmp_path, {"model": target, "optimizer": optimizer}, expected)
        assert optimizer.param_groups is groups_before
        assert all(torch.all(parameter == 9) for parameter in target.parameters())
        assert optimizer.state[target["a"]]["scalar_flag"] == 9

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))
    assert sorted(reads) == [
        (rank, f"param_groups.{index}.params", True)
        for rank in range(2)
        for index in range(2)
    ]


def test_native_dcp_restores_reordered_optimizer_groups_and_scalar_state(
    native_optimizer_checkpoint, tmp_path
):
    source, saved = native_optimizer_checkpoint
    target, optimizer = _native_grouped_optimizer(9.0, ("dc", "ba"), named=True)
    FSDP2Checkpointer.load(tmp_path, {"model": target, "optimizer": optimizer})
    for group, saved_group, order in zip(
        optimizer.param_groups, saved.param_groups, ("ab", "cd"), strict=True
    ):
        assert group["param_names"] == list(order)
        assert group["lr"] == saved_group["lr"]
        assert all(
            param is target[name]
            for name, param in zip(order, group["params"], strict=True)
        )
    assert _values(target) == _values(source)
    for name, param in source.items():
        torch.testing.assert_close(
            optimizer.state[target[name]], saved.state[param], rtol=0, atol=0
        )


def test_native_dcp_validates_inventory_on_all_ranks_before_unrestricted_load(
    native_optimizer_checkpoint, tmp_path, monkeypatch
):
    coordinator = ThreadCoordinator(2, timeout=10)
    context = local()
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context.value)
    unrestricted = []
    torch_load, pickle_load = torch.load, dcp.pickle.load

    def record_torch_load(file, *args, **kwargs):
        if kwargs.get("weights_only") is not True:
            unrestricted.append((context.value.rank, "torch.load"))
        return torch_load(file, *args, **kwargs)

    def record_pickle_load(file, *args, **kwargs):
        if str(getattr(file, "name", "")).endswith(".metadata"):
            unrestricted.append((context.value.rank, "pickle.load"))
        return pickle_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", record_torch_load)
    monkeypatch.setattr(dcp.pickle, "load", record_pickle_load)

    def load(rank):
        context.value = RankContext(rank, 2, coordinator)
        target, optimizer = _native_grouped_optimizer(9.0, ("ab", "cd"))
        expected = "another rank failed"
        if rank:
            optimizer.state[target["a"]]["extra_moment"] = torch.zeros_like(target["a"])
            expected = r"missing keys.*state\.a\.extra_moment"
        _load_rejected(tmp_path, {"model": target, "optimizer": optimizer}, expected)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))
    assert not unrestricted


@pytest.mark.parametrize("fault", ["model-file", "optimizer-file", "optimizer-record"])
def test_native_dcp_validates_records_on_all_ranks_before_unrestricted_load(
    tmp_path, monkeypatch, fault
):
    monkeypatch.setattr(fsdp2, "FSDPModule", torch.nn.Module)
    intact, damaged = tmp_path / "intact", tmp_path / "damaged"
    source, saved = _native_grouped_optimizer(1.0, ("ab", "cd"))
    torch_dcp_save(source.state_dict(), checkpoint_id=intact / "model", no_dist=True)
    torch_dcp_save(
        get_optimizer_state_dict(source, saved),
        storage_writer=FileSystemWriter(
            intact / "optimizer", single_file_per_rank=False
        ),
        no_dist=True,
    )
    copytree(intact, damaged)
    component, key = (
        ("model", "a") if fault == "model-file" else ("optimizer", "state.a.exp_avg")
    )
    root = damaged / component
    metadata = dcp.read_metadata(root)
    record = next(
        record for index, record in metadata.storage_data.items() if index.fqn == key
    )
    payload = root / record.relative_path
    if fault == "optimizer-record":
        payload.write_bytes(bytes(payload.stat().st_size))
        error = "zip file"
    else:
        payload.unlink()
        error = re.escape(str(payload))

    coordinator = ThreadCoordinator(2, timeout=10)
    context = local()
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context.value)
    unrestricted = []
    torch_load = torch.load

    def record_load(file, *args, **kwargs):
        if kwargs.get("weights_only") is not True:
            unrestricted.append(context.value.rank)
        return torch_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)

    def load(rank):
        context.value = RankContext(rank, 2, coordinator)
        target, optimizer = _native_grouped_optimizer(9.0, ("ab", "cd"))
        before = deepcopy(optimizer.state_dict())
        _load_rejected(
            damaged if rank else intact,
            {"model": target, "optimizer": optimizer},
            error if rank else "another rank failed",
        )
        assert all(torch.all(parameter == 9) for parameter in target.parameters())
        torch.testing.assert_close(optimizer.state_dict(), before, rtol=0, atol=0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))
    assert not unrestricted


@pytest.mark.parametrize("fmt", ("safetensors", "dcp"))
def test_later_component_header_failure_preserves_optimizer(mesh, tmp_path, fmt):
    source = fully_shard(_model(1.0, "a"), mesh=mesh)
    target = fully_shard(_model(8.0, "a"), mesh=mesh)
    saved, optimizer = _optimizer(source, 0.01), _optimizer(target, 0.25)
    FSDP2Checkpointer.save(
        tmp_path, {"model": source, "optimizer": saved}, checkpoint_format=fmt
    )
    suffix = "distcp" if fmt == "dcp" else "safetensors"
    payload = tmp_path / "model" / f"__0_0.{suffix}"
    payload.write_bytes(bytes(payload.stat().st_size))
    before = _values(target, optimizer)
    _load_rejected(
        tmp_path, {"model": target, "optimizer": optimizer}, "Expecting value|zip file"
    )
    assert _values(target, optimizer) == before


@pytest.mark.parametrize("fmt", _FORMATS)
def test_rank_specific_alias_failure_stops_writes_on_every_rank(
    tmp_path, monkeypatch, fmt
):
    monkeypatch.setattr(fsdp2, "FSDPModule", torch.nn.Module)
    source = _model(1.0, "a")
    FSDP2Checkpointer.save(tmp_path, {"model": source}, checkpoint_format=fmt)
    coordinator = ThreadCoordinator(2, timeout=10)
    context = local()
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: context.value)
    writes = []
    write = fsdp2.TorchAccessor.write_from_buffer

    def record_write(accessor, tensor_range, callback):
        writes.append(context.value.rank)
        return write(accessor, tensor_range, callback)

    monkeypatch.setattr(fsdp2.TorchAccessor, "write_from_buffer", record_write)

    def load(rank):
        context.value = RankContext(rank, 2, coordinator)
        target = _model(8.0, "b" if rank else "a")
        before = _values(target)
        _load_rejected(tmp_path, {"model": target}, "tied.*alias|another rank")
        assert _values(target) == before

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))
    assert not writes


@pytest.mark.parametrize("fmt", _FORMATS)
def test_missing_rank_state_preserves_fresh_optimizer(mesh, tmp_path, fmt, monkeypatch):
    source = fully_shard(_model(1.0, "a"), mesh=mesh)
    target = fully_shard(_model(8.0, "a"), mesh=mesh)
    saved, optimizer = _optimizer(source, 0.01), _optimizer(target, 0.25, fresh=True)
    FSDP2Checkpointer.save(
        tmp_path, {"model": source, "optimizer": saved}, checkpoint_format=fmt
    )
    before = _values(target, optimizer)
    extra_state = object()
    state = {"model": target, "optimizer": optimizer, "extra_state": extra_state}

    def unexpected_read(*args):
        pytest.fail("tensor file opened before checking rank state presence")

    if fmt != "hf":
        monkeypatch.setattr(FileIO, "pread", unexpected_read)
    _load_rejected(tmp_path, state, "missing rank state|extra_state_rank_0")
    assert _values(target, optimizer) == before
    assert not optimizer.state
    assert state["extra_state"] is extra_state


@pytest.mark.parametrize("fmt", _FORMATS)
def test_corrupt_rank_state_preserves_live_state(mesh, tmp_path, fmt):
    source = fully_shard(_model(1.0, "a"), mesh=mesh)
    target = fully_shard(_model(8.0, "a"), mesh=mesh)
    saved, optimizer = _optimizer(source, 0.01), _optimizer(target, 0.25)
    extra_state = object()
    state = {"model": target, "optimizer": optimizer, "extra_state": extra_state}
    FSDP2Checkpointer.save(
        tmp_path,
        {"model": source, "optimizer": saved, "extra_state": {"step": 1}},
        checkpoint_format=fmt,
    )
    (tmp_path / "extra_state" / "extra_state_rank_0.pt").write_bytes(b"")
    before = _values(target, optimizer)
    with pytest.raises((EOFError, RuntimeError), match="^$|extra_state|input"):
        FSDP2Checkpointer.load(tmp_path, state)
    assert _values(target, optimizer) == before
    assert optimizer.param_groups[0]["lr"] == 0.25
    assert state["extra_state"] is extra_state


@pytest.mark.parametrize("fmt", _FORMATS)
def test_valid_load_installs_fresh_optimizer_state(mesh, tmp_path, fmt):
    source = fully_shard(_model(1.0, "a"), mesh=mesh)
    target = fully_shard(_model(8.0, "a"), mesh=mesh)
    saved, optimizer = _optimizer(source, 0.01), _optimizer(target, 0.25, fresh=True)
    FSDP2Checkpointer.save(
        tmp_path, {"model": source, "optimizer": saved}, checkpoint_format=fmt
    )
    FSDP2Checkpointer.load(tmp_path, {"model": target, "optimizer": optimizer})
    assert _values(target, optimizer) == _values(source, saved)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert target.alias is target.a


def test_rejected_fp32_export_preserves_empty_optimizer(mesh, tmp_path):
    model = _grouped_model(mesh, 1.0)
    opt = _grouped_optimizer(model, ("ab", "cd"), initialized=False)
    with pytest.raises((ValueError, RuntimeError), match="missing master_param"):
        FSDP2Checkpointer.save(
            tmp_path,
            {"model": model, "optimizer": opt},
            checkpoint_format="hf",
            hf_dtype="fp32",
        )
    assert not opt.state


def test_pristine_adamw_dcp_component_readable_by_pytorch(mesh, tmp_path):
    model = _grouped_model(mesh, 1.0)
    opt = _grouped_optimizer(model, ("ab", "cd"), initialized=False)
    for _ in range(2):
        FSDP2Checkpointer.save(
            tmp_path, {"model": model, "optimizer": opt}, checkpoint_format="dcp"
        )
        restored = {
            "state": {},
            "param_groups": fsdp2._named_optimizer_groups(
                opt, {p: name for name, p in model.named_parameters()}
            ),
        }
        torch.distributed.checkpoint.load(
            restored, checkpoint_id=tmp_path / "optimizer"
        )
        assert not restored["state"]
        assert restored["param_groups"][1]["params"] == ["c", "d"]
        assert not opt.state


def test_native_dcp_bundle_model_only_and_missing_optimizer_moment(mesh, tmp_path):
    source = _grouped_model(mesh, 1.0)
    opt = _grouped_optimizer(source, ("ab", "cd"))
    saved = cast(dict[str, Any], get_optimizer_state_dict(source, opt))
    del saved["state"]["a"]["exp_avg"]
    torch_dcp_save(
        {"model": source.state_dict(), "optimizer": saved}, checkpoint_id=tmp_path
    )
    target = _grouped_model(mesh, 9.0)
    fresh = _grouped_optimizer(target, ("ab", "cd"), initialized=False)
    _load_rejected(
        tmp_path,
        {"model": target, "optimizer": fresh},
        r"missing keys.*state\.a\.exp_avg",
    )
    assert not fresh.state
    assert _values(target) != _values(source)
    FSDP2Checkpointer.load(tmp_path, {"model": target})
    assert _values(target) == _values(source)


def test_hf_weights_only_export_loads_tied_model_without_alias_sidecar(mesh, tmp_path):
    source = fully_shard(_model(1.0, "a"), mesh=mesh)
    target = fully_shard(_model(8.0, "a"), mesh=mesh)
    FSDP2Checkpointer.save(
        tmp_path, {"model": source}, checkpoint_format="hf", hf_dtype="fp32"
    )
    FSDP2Checkpointer.load(tmp_path, {"model": target})
    assert _values(target) == _values(source)
    assert target.alias is target.a
