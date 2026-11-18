"""Independent dense FSDP2 oracle and manifest/safetensors-only reader."""

from typing import cast

import torch
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


class AcceptanceModel(torch.nn.Module):
    """Uneven and empty local shards, mixed precision, replicated buffers."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(40).reshape(10, 4) / 16 + 1)
        self.tiny = torch.nn.Parameter(
            (torch.arange(6).reshape(2, 3) / 8 + 1).bfloat16()
        )
        self.bias = torch.nn.Parameter(torch.arange(10) / 16 - 0.25)
        self.register_buffer("running_mean", torch.arange(4).float() / 4)
        self.register_buffer("count", torch.tensor(7, dtype=torch.int64))
        self.register_buffer("empty", torch.empty((0, 3), dtype=torch.bfloat16))


def optimizer_for(model, *, target=False):
    return torch.optim.AdamW(
        model.parameters(),
        lr=0.25 if target else 0.03125,
        betas=(0.5, 0.75),
        eps=0.125,
        weight_decay=0.0625,
        foreach=False,
    )


def train_step(model, optimizer, step):
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.ones_like(parameter) * ((index + step + 1) / 8)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def make_model(world, *, target=False):
    model = AcceptanceModel()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
    fully_shard(model, mesh=mesh)
    optimizer = optimizer_for(model, target=target)
    return model, optimizer


def dense_oracle(steps=2):
    model = AcceptanceModel()
    optimizer = optimizer_for(model)
    for step in range(steps):
        train_step(model, optimizer, step)
    return model, optimizer


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def raw(tensor):
    return tensor.detach().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()


def expected_local(expected, actual, rank, world):
    if not isinstance(actual, DTensor):
        return expected
    chunk = (expected.shape[0] + world - 1) // world
    return expected[rank * chunk : (rank + 1) * chunk]


def assert_model(model, oracle, rank, world):
    actual, expected = model.state_dict(), oracle.state_dict()
    assert actual.keys() == expected.keys()
    for key, tensor in actual.items():
        want = expected_local(expected[key], tensor, rank, world)
        assert local(tensor).shape == want.shape, (key, local(tensor).shape, want.shape)
        assert raw(local(tensor)) == raw(want), (key, rank)
        if isinstance(tensor, DTensor):
            assert raw(tensor.full_tensor()) == raw(expected[key]), key


def assert_optimizer(model, optimizer, oracle, oracle_optimizer, rank, world):
    actual = get_optimizer_state_dict(model, optimizer)
    expected = get_optimizer_state_dict(oracle, oracle_optimizer)
    assert actual["param_groups"] == expected["param_groups"]
    actual_state = cast(dict[str, dict[str, torch.Tensor]], actual["state"])
    expected_state = cast(dict[str, dict[str, torch.Tensor]], expected["state"])
    assert actual_state.keys() == expected_state.keys()
    for key, fields in actual_state.items():
        assert fields.keys() == expected_state[key].keys()
        _assert_fields(fields, expected_state[key], rank, world, key)


def _assert_fields(actual, expected, rank, world, key):
    for name, tensor in actual.items():
        want = expected_local(expected[name], tensor, rank, world)
        assert local(tensor).shape == want.shape, (key, name)
        assert raw(local(tensor)) == raw(want), (key, name, rank)
        if isinstance(tensor, DTensor):
            assert raw(tensor.full_tensor()) == raw(expected[name]), (key, name)


def extra_for(rank):
    return {"step": 2, "rank": rank, "nested": {"tuple": (3, "state")}, "blob": b"gdcp"}


def _oracle_tensor(tensor):
    return {
        "data": raw(tensor).hex(),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def oracle_records(steps=2):
    model, optimizer = dense_oracle(steps)
    state = get_optimizer_state_dict(model, optimizer)
    tensor_state = cast(dict[str, dict[str, torch.Tensor]], state["state"])
    opt = {
        f"state.{key}.{name}": _oracle_tensor(value)
        for key, fields in tensor_state.items()
        for name, value in fields.items()
    }
    return {
        "model": {
            key: _oracle_tensor(value) for key, value in model.state_dict().items()
        },
        "optimizer": opt,
    }
