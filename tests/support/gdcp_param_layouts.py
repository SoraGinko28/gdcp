"""Mixed FSDP2 parameter meshes, checked against an unsharded AdamW model."""

from pathlib import Path

import pytest
import torch
from torch.distributed import get_rank
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    HSDPMeshInfo,
    ShardPlacementResult,
)
from torch.distributed.tensor import DTensor, Shard

from gdcp import FSDP2Checkpointer, load_hf, save_hf
from tests.support.gdcp_fsdp2 import local, optimizer_for


class MixedModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        for name, shape in (
            ("rows", (7, 8)),
            ("cols", (5, 8)),
            ("last", (3, 4, 8)),
            ("middle", (2, 8, 3)),
        ):
            values = (
                torch.arange(torch.Size(shape).numel(), device=device)
                .float()
                .reshape(shape)
            )
            self.register_parameter(name, torch.nn.Parameter(values / 128))

    def forward(self, inputs):
        return sum(
            (p * x).sum() for p, x in zip(self.parameters(), inputs, strict=True)
        )


def _make(config, mixed):
    device, flat, hybrid, subset = config
    model = MixedModel(device)
    layouts = {
        model.cols: Shard(1),
        model.last: ShardPlacementResult(
            Shard(-1), FSDPMeshInfo(subset, shard_mesh_dim=0)
        ),
        model.middle: ShardPlacementResult(
            Shard(1), HSDPMeshInfo(hybrid, shard_mesh_dim=1, replicate_mesh_dim=0)
        ),
    }
    fully_shard(model, mesh=flat, shard_placement_fn=layouts.get if mixed else None)
    if mixed:
        for name, mesh, dim in (
            ("rows", flat, 0),
            ("cols", flat, 1),
            ("last", subset, 2),
            ("middle", hybrid, 1),
        ):
            param = getattr(model, name)
            assert param.device_mesh == mesh and param.placements[-1] == Shard(dim)
    return model, optimizer_for(model)


def _expected(tensor, param):
    if not isinstance(param, DTensor):
        return tensor
    mesh = param.device_mesh
    coordinate = mesh.get_coordinate()
    assert coordinate is not None
    for axis, placement in enumerate(param.placements):
        if isinstance(placement, Shard):
            length = tensor.size(placement.dim)
            chunk = -(-length // mesh.size(axis))
            start = min(length, coordinate[axis] * chunk)
            return tensor.narrow(placement.dim, start, min(chunk, length - start))
    return tensor


def _update(model, optimizer, step, device):
    inputs = [
        (torch.arange(p.numel(), device=device).reshape(p.shape) + step + 1) / 128
        for p in model.parameters()
    ]
    if device == "cuda":
        model(inputs).backward()
    else:
        for p, value in zip(model.parameters(), inputs, strict=True):
            p.grad = torch.empty_like(p)
            local(p.grad).copy_(_expected(value, p))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _compare(model, optimizer, ref, oracle):
    for (name, p), ref_p in zip(
        model.named_parameters(), ref.parameters(), strict=True
    ):
        torch.testing.assert_close(
            local(p), _expected(ref_p, p), rtol=0, atol=0, msg=name
        )
        assert optimizer.state[p].keys() == oracle.state[ref_p].keys()
        for key, value in optimizer.state[p].items():
            expected = oracle.state[ref_p][key]
            expected = _expected(expected, p) if expected.ndim else expected
            torch.testing.assert_close(
                local(value), expected, rtol=0, atol=0, msg=f"{name}.{key}"
            )


def _restore(path, config, mixed):
    device = config[0]
    model, opt = _make(config, mixed)
    with torch.no_grad():
        for p in model.parameters():
            local(p).fill_(-8)
    state = {"model": model, "optimizer": opt, "extra_state": None}
    FSDP2Checkpointer.load(path, state)
    assert state["extra_state"] == get_rank()
    ref = MixedModel(device)
    oracle = optimizer_for(ref)
    _update(ref, oracle, 0, device)
    _compare(model, opt, ref, oracle)
    _update(model, opt, 1, device)
    _update(ref, oracle, 1, device)
    _compare(model, opt, ref, oracle)


def _hf_masters(path, model, opt):
    expected = {}
    for p in model.parameters():
        master = local(p).detach().clone() + 2
        opt.state[p]["master_param"] = master
        expected[p] = master.clone()
    state = {"model": model, "optimizer": opt}
    save_hf(path, state, dtype="fp32")
    for p in model.parameters():
        opt.state[p]["master_param"].zero_()
    load_hf(path, state, dtype="fp32")
    for p in model.parameters():
        torch.testing.assert_close(
            opt.state[p].pop("master_param"), expected[p], rtol=0, atol=0
        )


def _roundtrip(root, config, mixed):
    model, opt = _make(config, mixed)
    _update(model, opt, 0, config[0])
    state = {"model": model, "optimizer": opt, "extra_state": get_rank()}
    for fmt in ("safetensors", "hf", "dcp"):
        path = root / fmt
        FSDP2Checkpointer.save(path, state, checkpoint_format=fmt)
        _restore(path, config, True)
        if mixed:
            _restore(path, config, False)
    if mixed:
        _hf_masters(root / "masters", model, opt)
        # Independent training groups can diverge under one FQN. Publishing a
        # single copy would discard a group's weights or optimizer progress.
        if get_rank() == 1:
            opt.state[model.get_parameter("last")]["step"].add_(1)
        with pytest.raises(ValueError, match="different values"):
            FSDP2Checkpointer.save(root / "conflict", state)
        assert not (root / "conflict").exists()


def parameter_layouts(args):
    device = "cpu" if args.backend == "gloo" else "cuda"
    flat = init_device_mesh(device, (4,), mesh_dim_names=("dp",))
    hybrid = DeviceMesh(device, [[0, 2], [1, 3]], mesh_dim_names=("replica", "shard"))
    config = device, flat, hybrid, hybrid["shard"]
    for mixed in (True, False):
        _roundtrip(Path(args.path) / ("mixed" if mixed else "flat"), config, mixed)
    print(f"parameter layouts PASS {args.backend} rank={get_rank()}", flush=True)
