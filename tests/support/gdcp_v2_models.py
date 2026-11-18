"""Architecture-only v2 fixtures and a real fp32-master AdamW optimizer."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
import numpy as np
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "fixtures/gdcp-v2/configs"


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def global_start(tensor, rank, world):
    if not isinstance(tensor, DTensor) or tensor.ndim == 0:
        return 0
    rows = math.ceil(tensor.shape[0] / world)
    return rank * rows * math.prod(tensor.shape[1:])


def pattern(name, size, start, device, *, step=0):
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    index = torch.arange(start, start + size, device=device, dtype=torch.int64)
    return ((index + seed + step * 17) % 257 - 128).float() / 1024


def fill(tensor, name, rank, world, *, positive=False):
    target = local(tensor).reshape(-1)
    start = global_start(tensor, rank, world)
    for offset in range(0, target.numel(), 1_048_576):
        count = min(1_048_576, target.numel() - offset)
        value = pattern(name, count, start + offset, target.device)
        if positive:
            value = value.abs() + 0.25
        target[offset : offset + count].copy_(value)


def _poison_groups(optimizer):
    for group in optimizer.param_groups:
        group.update(lr=0.25, betas=(0.25, 0.5), eps=0.5, weight_decay=0.25)


def _poison_parameter(parameter, state):
    local(parameter).fill_(77)
    for value in state.values():
        local(value).fill_(77)


def _master_offset(tensor, field):
    if field == "master_param":
        local(tensor).add_(2**-19)


def _sqrt_fp32(tensor):
    # Torch's CPU vector sqrt can differ by one ULP from CUDA IEEE sqrt.
    # NumPy keeps float32 and uses the CPU IEEE sqrt instruction.
    if tensor.device.type == "cpu":
        return torch.from_numpy(np.sqrt(tensor.numpy()))
    return tensor.sqrt()


class MasterAdamW(torch.optim.AdamW):
    """AdamW over fp32 master weights, with per-parameter DTensor state."""

    def __init__(self, model, rank, world, *, poison=False):
        super().__init__(
            model.named_parameters(),
            lr=0.03125,
            betas=(0.5, 0.75),
            eps=0.125,
            weight_decay=0.0625,
            foreach=False,
        )
        self.fixture_names = {id(p): n for n, p in model.named_parameters()}
        self.fixture_rank, self.fixture_world = rank, world
        for name, param in model.named_parameters():
            self._initialize(name, param, poison)
        if poison:
            _poison_groups(self)

    @torch.no_grad()
    def _initialize(self, name, param, poison):
        state = self.state[param]
        state["step"] = torch.tensor(3.0, device="cpu")
        for field in ("master_param", "exp_avg", "exp_avg_sq"):
            value = torch.empty_like(param, dtype=torch.float32)
            fill(
                value,
                name + "." + field,
                self.fixture_rank,
                self.fixture_world,
                positive=field == "exp_avg_sq",
            )
            _master_offset(value, field)
            state[field] = value
        local(param).copy_(local(state["master_param"]).to(torch.bfloat16))
        if poison:
            _poison_parameter(param, state)

    @torch.no_grad()
    def step(self, closure=None, *, synthetic_step=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            self._step_group(group, synthetic_step)
        return loss

    def _step_group(self, group, synthetic_step):
        for param in group["params"]:
            self._step_parameter(param, group, synthetic_step)

    def _step_parameter(self, param, group, synthetic_step):
        if synthetic_step is None and param.grad is None:
            return
        state = self.state[param]
        master = local(state["master_param"])
        if synthetic_step is None:
            gradient = local(param.grad).float()
        else:
            name = self.fixture_names[id(param)]
            start = global_start(param, self.fixture_rank, self.fixture_world)
            gradient = pattern(
                name + ".gradient",
                master.numel(),
                start,
                master.device,
                step=synthetic_step,
            ).reshape(master.shape)
        beta1, beta2 = group["betas"]
        state["step"].add_(1)
        step = state["step"].item()
        avg, square = local(state["exp_avg"]), local(state["exp_avg_sq"])
        master.mul_(1 - group["lr"] * group["weight_decay"])
        avg.mul_(beta1).add_(gradient * (1 - beta1))
        square.mul_(beta2).add_(gradient.square() * (1 - beta2))
        denom = (
            _sqrt_fp32(square).mul_(1 / math.sqrt(1 - beta2**step)).add_(group["eps"])
        )
        master.add_((avg / denom) * (-group["lr"] / (1 - beta1**step)))
        local(param).copy_(master.to(torch.bfloat16))


def _tiny_layer_types(config):
    for attribute in ("layer_types", "mlp_layer_types"):
        layers = getattr(config, attribute, None)
        if isinstance(layers, list):
            setattr(config, attribute, layers[:2])


def build(key, world, rank, device, *, tiny=False, poison=False):
    config = AutoConfig.from_pretrained(CONFIG_ROOT / key, local_files_only=True)
    if tiny:
        config.hidden_size, config.intermediate_size = 32, 64
        config.num_hidden_layers, config.num_attention_heads = 2, 4
        config.num_key_value_heads, config.head_dim, config.vocab_size = 2, 8, 67
        config.max_position_embeddings = 128
        _tiny_layer_types(config)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.tie_weights()
    parameters = sum(p.numel() for p in model.parameters())
    mesh = init_device_mesh(device, (world,), mesh_dim_names=("dp",))
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    model.to_empty(device=torch.device(device, rank) if device == "cuda" else "cpu")
    optimizer = MasterAdamW(model, rank, world, poison=poison)
    return (
        model,
        optimizer,
        {
            "parameters": parameters,
            "nominal_state_bytes": 14 * parameters,
            "model_class": type(model).__name__,
            "config": config.to_dict(),
        },
    )


def state_tensors(model, optimizer):
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    tensors = {"model." + k: v for k, v in model.state_dict().items()}
    state = cast(dict[str, Any], get_optimizer_state_dict(model, optimizer))
    for name, fields in state["state"].items():
        tensors.update(
            {
                f"optimizer.state.{name}.{field}": value
                for field, value in fields.items()
                if isinstance(value, torch.Tensor)
            }
        )
    return tensors, state["param_groups"]


def jsonable(value):
    return json.loads(json.dumps(value, sort_keys=True))
