"""Focused adapter invariants independent of model size and rank launcher."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from torch.distributed import HashStore, destroy_process_group, init_process_group
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from gdcp import FSDP2Checkpointer, desc
from gdcp.adapters import fsdp2, torch_access
from gdcp.adapters import hf as hf_adapter
from gdcp.adapters.torch_access import TorchAccessor
from gdcp.state import CheckpointState, RankContext
from gdtensor.builder import Err, Ok
from gdtensor_conn.coord import ThreadCoordinator
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
            hf_adapter.load_hf(tmp_path, {})

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))


def test_fsdp2_format_lookup_failure_votes_on_every_rank(tmp_path, monkeypatch):
    coordinator = ThreadCoordinator(2, timeout=2)
    context = local()
    exists = Path.exists
    monkeypatch.setattr(fsdp2, "_context", lambda: context.value)

    def rank_local_exists(path):
        if path == tmp_path / "model.safetensors" and context.value.rank == 0:
            raise OSError("injected checkpoint lookup failure")
        return exists(path)

    monkeypatch.setattr(Path, "exists", rank_local_exists)

    def load(rank):
        context.value = RankContext(rank, 2, coordinator)
        expected = "another rank failed" if rank else "checkpoint lookup failure"
        with pytest.raises(RuntimeError, match=expected):
            fsdp2.load_checkpoint(tmp_path, {}, common_only=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(load, range(2)))


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
    monkeypatch.setattr(fsdp2, "_context", lambda: context.value)

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
    monkeypatch.setattr(fsdp2, "_context", lambda: context.value)
    described = []
    describe = fsdp2.FSDP2Adapter.describe

    def track_describe(adapter, state, ctx):
        described.append(ctx.rank)
        return describe(adapter, state, ctx)

    monkeypatch.setattr(fsdp2.FSDP2Adapter, "describe", track_describe)

    def save(rank):
        context.value = RankContext(rank, 2, coordinator)
        with pytest.raises(RuntimeError, match="ranks disagree on checkpoint save"):
            FSDP2Checkpointer.save(tmp_path / "checkpoint", {}, **options[rank])

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(save, range(2)))
    assert not described
    assert not list(tmp_path.iterdir())


def test_invalid_fsdp2_load_arguments_vote_on_every_rank(tmp_path, monkeypatch):
    coordinator = ThreadCoordinator(2, timeout=2)
    context = local()
    monkeypatch.setattr(fsdp2, "_context", lambda: context.value)

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
    accessor = TorchAccessor({"first": tensor, "second": tensor[:2]}, 0, 1)
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
    accessor = TorchAccessor({"weight": tensor}, 0, 1)
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
    accessor = TorchAccessor({"weight": tensor}, 0, 1)
    _write(accessor, "weight", 0, bytes(4))
    with pytest.raises(ValueError, match="incomplete local tensor load"):
        accessor.finish()
    assert torch.equal(tensor, torch.full_like(tensor, -1))


@pytest.mark.parametrize(
    "backend", ["gloo", "cuda:nccl,cpu:gloo", "cpu:gloo,cuda:nccl"]
)
def test_context_accepts_cpu_gloo_in_composite_group(monkeypatch, backend):
    monkeypatch.setattr(fsdp2, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp2, "get_backend", lambda: backend)
    monkeypatch.setattr(fsdp2, "get_rank", lambda: 1)
    monkeypatch.setattr(fsdp2, "get_world_size", lambda: 4)
    assert fsdp2._context().rank == 1
    assert fsdp2._context().world_size == 4


def test_context_rejects_group_without_cpu_gloo(monkeypatch):
    monkeypatch.setattr(fsdp2, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp2, "get_backend", lambda: "nccl")
    with pytest.raises(NotImplementedError, match="Gloo for CPU"):
        fsdp2._context()


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
    with pytest.raises(ValueError, match="another_moment.*shape"):
        fsdp2._validate_optimizer_tensors(model, {"state": {"weight": fields}})


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
    adapter.prepare_restore(target, {"optimizer": objects}, {})
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
    fsdp2._validate_state(
        {"model": model, "optimizer": optimizer}, loading=False, common_only=False
    )
    with pytest.raises(NotImplementedError, match="AdamW or a subclass"):
        fsdp2._validate_state(
            {"model": model, "optimizer": torch.optim.SGD(model.parameters())},
            loading=False,
            common_only=False,
        )


def test_invalid_accessor_range_returns_result_error():
    accessor = TorchAccessor({"weight": torch.zeros(2)}, 0, 1)
    result = accessor.read_to_buffer(_range("weight", 0, 12), lambda *args: Ok(None))
    assert type(result) is Err


@pytest.fixture
def mesh(monkeypatch):
    pytest.importorskip("torch.testing._internal.distributed.fake_pg")
    init_process_group("fake", store=HashStore(), rank=0, world_size=1)
    context = RankContext(0, 1)
    monkeypatch.setattr(fsdp2, "_context", lambda: context)
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
    monkeypatch.setattr(fsdp2, "_context", lambda: context.value)
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


def test_hf_weights_only_export_loads_tied_model_without_alias_sidecar(mesh, tmp_path):
    source = fully_shard(_model(1.0, "a"), mesh=mesh)
    target = fully_shard(_model(8.0, "a"), mesh=mesh)
    FSDP2Checkpointer.save(
        tmp_path, {"model": source}, checkpoint_format="hf", hf_dtype="fp32"
    )
    FSDP2Checkpointer.load(tmp_path, {"model": target})
    assert _values(target) == _values(source)
    assert target.alias is target.a
