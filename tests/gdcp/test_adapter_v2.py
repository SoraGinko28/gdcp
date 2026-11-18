"""Focused adapter invariants independent of model size and rank launcher."""

import ctypes
from types import SimpleNamespace
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from gdcp.adapters import fsdp2, torch_access
from gdcp.adapters.torch_access import TorchAccessor
from gdcp.state import RankContext
from gdtensor.builder import Err, Ok


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
    adapter._restore_optimizer({"model": model, "optimizer": optimizer}, objects)
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
