"""Train a tensor-parallel MLP and check every restored shard against dense AdamW."""

from pathlib import Path

import torch
from torch import nn
from torch.distributed import get_rank, get_world_size
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from gdcp.adapters.torch_placements import is_strided, strided_parts

from gdcp import FSDP2Checkpointer
from tests.support.gdcp_fsdp2 import local


def _model(device, mesh=None, tp=False):
    model = nn.Sequential(
        nn.Linear(8, 12, bias=False, device=device),
        nn.Linear(12, 8, bias=False, device=device),
    )
    with torch.no_grad():
        for weight in model.parameters():
            values = torch.arange(weight.numel(), device=device).view_as(weight)
            weight.copy_((values + 1) / 128)
    if mesh is not None:
        if tp:
            parallelize_module(
                model, mesh["tp"], {"0": ColwiseParallel(), "1": RowwiseParallel()}
            )
            dp_mesh = mesh["replica", "dp"] if mesh.ndim == 3 else mesh["dp"]
            fully_shard(model, mesh=dp_mesh)
            _check_placements(model, mesh)
        else:
            fully_shard(model, mesh=mesh)
    return model, torch.optim.AdamW(model.parameters(), lr=0.005, foreach=False)


def _check_placements(model, mesh):
    first, second = model.parameters()
    assert isinstance(first, DTensor) and isinstance(second, DTensor)
    assert first.device_mesh == mesh and second.device_mesh == mesh
    prefix = (Replicate(),) if mesh.ndim == 3 else ()
    dp, tp = first.placements[-2:]
    assert first.placements[:-2] == prefix
    assert is_strided(dp) and strided_parts(dp) == (0, mesh.size(-1))
    assert tp == Shard(0)
    assert second.placements == (*prefix, Shard(0), Shard(1))


def _step(model, opt, step, device, *, dense=False):
    batches = get_world_size() // 2
    inputs = (
        torch.arange(batches * 24, device=device).float().view(batches, 3, 8) / 32
        + step
    )
    batch = inputs.flatten(0, 1) if dense else inputs[get_rank() // 2]
    loss = model(batch).square().mean()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def _expected_shard(value, layer, tp, mesh):
    if not value.ndim:
        return value
    coord = mesh.get_coordinate()
    assert coord is not None
    if tp:
        # Slice the dense oracle explicitly: TP first, then DP; each replica
        # must independently match this slice, even after a cross-layout load.
        value = value.chunk(mesh.shape[-1], dim=layer)[coord[-1]]
        rank, parts = coord[-2], mesh.shape[-2]
    else:
        rank, parts = coord[0], mesh.shape[0]
    width = -(-value.shape[0] // parts)
    return value[rank * width : (rank + 1) * width]


def _check(model, opt, dense, oracle, *, tp, exact=False):
    for i, (p, ref) in enumerate(
        zip(model.parameters(), dense.parameters(), strict=True)
    ):
        assert opt.state[p].keys() == oracle.state[ref].keys()
        actual = {"weight": p, **opt.state[p]}
        expected = {"weight": ref, **oracle.state[ref]}
        for key, value in actual.items():
            torch.testing.assert_close(
                local(value),
                _expected_shard(expected[key], i, tp, p.device_mesh),
                rtol=0 if exact else 2e-5,
                atol=0 if exact else 2e-7,
                msg=lambda msg: f"layer {i} {key}: {msg}",
            )


@torch.no_grad()
def _snapshot(model, opt, device):
    saved, saved_opt = _model(device)
    for p, ref in zip(model.parameters(), saved.parameters(), strict=True):
        ref.copy_(p.full_tensor())
        for key, value in opt.state[p].items():
            saved_opt.state[ref][key] = (
                value.full_tensor() if value.ndim else local(value).clone()
            )
    return saved, saved_opt


def _restore(path, device, mesh, tp, saved, next_step):
    target, restored = _model(device, mesh, tp)
    with torch.no_grad():
        for p in target.parameters():
            local(p).fill_(-8)
    FSDP2Checkpointer.load(path, {"model": target, "optimizer": restored})
    _check(target, restored, *saved, tp=tp, exact=True)
    _step(target, restored, 1, device)
    _check(target, restored, *next_step, tp=tp)


def _roundtrip(root, device, source_layout, targets, initial, next_step):
    source_mesh, source_tp = source_layout
    source, opt = _model(device, source_mesh, source_tp)
    _step(source, opt, 0, device)
    _check(source, opt, *initial, tp=source_tp)
    saved = _snapshot(source, opt, device)
    for fmt in ("safetensors", "dcp", "hf"):
        path = root / fmt
        FSDP2Checkpointer.save(
            path, {"model": source, "optimizer": opt}, checkpoint_format=fmt
        )
        for target_mesh, target_tp in targets:
            _restore(path, device, target_mesh, target_tp, saved, next_step)


def _references(device):
    initial = _model(device)
    next_step = _model(device)
    _step(*initial, step=0, device=device, dense=True)
    for step in range(2):
        _step(*next_step, step=step, device=device, dense=True)
    return initial, next_step


def dp_tp(args):
    device = "cpu" if args.backend == "gloo" else "cuda"
    tp = init_device_mesh(device, (2, 2), mesh_dim_names=("dp", "tp")), True
    flat = init_device_mesh(device, (4,), mesh_dim_names=("world",)), False
    initial, next_step = _references(device)
    root = Path(args.path)
    _roundtrip(root / "dptp", device, tp, [tp, flat], initial, next_step)
    _roundtrip(root / "flat", device, flat, [tp], initial, next_step)
    print(f"DP+TP PASS {args.backend} rank={get_rank()}", flush=True)


def hsdp_tp(args):
    device = "cpu" if args.backend == "gloo" else "cuda"
    world = get_world_size()
    flat = init_device_mesh(device, (world,), mesh_dim_names=("world",)), False
    tp = init_device_mesh(device, (world // 2, 2), mesh_dim_names=("dp", "tp")), True
    shapes = [(2, 2, 2)] if world == 8 else [(1, 2, 2), (2, 1, 2)]
    initial, next_step = _references(device)
    for shape in shapes:
        mesh = init_device_mesh(device, shape, mesh_dim_names=("replica", "dp", "tp"))
        hybrid = mesh, True
        root = Path(args.path) / "x".join(str(size) for size in shape)
        _roundtrip(
            root / "hsdptp", device, hybrid, [hybrid, flat, tp], initial, next_step
        )
        _roundtrip(root / "flat", device, flat, [hybrid], initial, next_step)
        _roundtrip(root / "dptp", device, tp, [hybrid], initial, next_step)
    print(f"HSDP+TP PASS {args.backend} world={world} rank={get_rank()}", flush=True)
