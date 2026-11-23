"""Cross-world training recovery using both public checkpoint implementations."""

from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed import barrier, get_rank, get_world_size
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    ShardPlacementResult,
)
from torch.distributed.tensor import DTensor, Shard

from gdcp import FSDP2Checkpointer
from tests.support.gdcp_dp_tp import _model

_FORMATS = ("native", "bundle", "dcp", "safetensors", "hf")


def _make(config, layout):
    device, flat, mesh = config
    if layout == "tp":
        return _model(device, mesh, tp=True)
    model, _ = _model(device)
    if layout == "parameter":
        # Separate two-rank DP groups carry identical copies of this key.
        submesh = FSDPMeshInfo(mesh["dp"], shard_mesh_dim=0)
        placements = {model[0].weight: ShardPlacementResult(Shard(1), submesh)}
        fully_shard(model, mesh=flat, shard_placement_fn=placements.get)
    else:
        fully_shard(model, mesh=mesh if layout == "hsdp" else flat)
    return model, torch.optim.AdamW(model.parameters(), lr=0.005, foreach=False)


def _step(model, opt, device, step):
    inputs = torch.arange(24, device=device).float().view(3, 8) / 32 + step
    model(inputs).square().mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def _global(t):
    value = t.full_tensor() if isinstance(t, DTensor) else t
    return value.detach().cpu().clone()


def _snapshot(model, opt):
    fields = {}
    for name, p in model.named_parameters():
        fields[name] = _global(p)
        for key, value in opt.state.get(p, {}).items():
            fields[f"{name}.{key}"] = _global(value)
    return fields


def _check(model, opt, reference, *, exact):
    values = _snapshot(model, opt)
    assert values.keys() == reference.keys()
    for key, expected in reference.items():
        torch.testing.assert_close(
            values[key],
            expected,
            rtol=0 if exact else 2e-5,
            atol=0 if exact else 2e-7,
            msg=lambda err: f"{key}: {err}",
        )
    assert opt.param_groups[0]["lr"] == 0.005


def _save(root, layout, config):
    device = config[0]
    model, opt = _make(config, layout)
    _step(model, opt, device, 0)
    reference = _snapshot(model, opt)
    if get_rank() == 0:
        torch.save(reference, root / "reference.pt")
    ms, osd = get_state_dict(model, opt)
    dcp.save(ms, checkpoint_id=root / "native/model")
    dcp.save(osd, checkpoint_id=root / "native/optimizer")
    state = {"model": model, "optimizer": opt}
    dcp.save({"model": ms, "optimizer": osd}, checkpoint_id=root / "bundle")
    for fmt in _FORMATS[2:]:
        FSDP2Checkpointer.save(root / fmt, state, checkpoint_format=fmt)
    _step(model, opt, device, 1)
    continued = _snapshot(model, opt)
    if get_rank() == 0:
        torch.save(continued, root / "continued.pt")
    barrier()


def _load_one(path, config, layout, reference, continued, *, native):
    model, opt = _make(config, layout)
    # Poison both data and hyperparameters before either reader sees the target.
    _step(model, opt, config[0], 7)
    for slots in opt.state.values():
        slots["step"].fill_(91)
    opt.param_groups[0]["lr"] = 0.3
    if native:
        ms, osd = get_state_dict(model, opt)
        dcp.load(ms, checkpoint_id=path / "model")
        dcp.load(osd, checkpoint_id=path / "optimizer")
        set_state_dict(model, opt, model_state_dict=ms, optim_state_dict=osd)
    else:
        FSDP2Checkpointer.load(path, {"model": model, "optimizer": opt})
    _check(model, opt, reference, exact=True)
    _step(model, opt, config[0], 1)
    _check(model, opt, continued, exact=False)


def _load(root, config, layout, *, same_layout=False):
    reference = torch.load(root / "reference.pt", weights_only=True)
    continued = torch.load(root / "continued.pt", weights_only=True)
    for fmt in ("native",) if same_layout else _FORMATS:
        _load_one(root / fmt, config, layout, reference, continued, native=False)
    _load_one(root / "dcp", config, layout, reference, continued, native=True)


def cross_layout(args):
    device = "cpu" if args.backend == "gloo" else "cuda"
    world = get_world_size()
    flat = init_device_mesh(device, (world,), mesh_dim_names=("world",))
    mesh = (
        init_device_mesh(device, (2, 2), mesh_dim_names=("dp", "tp"))
        if world == 4
        else None
    )
    config = device, flat, mesh
    root = Path(args.path)
    flat_root = root / "flat"
    if args.action == "cross_save":
        flat_root.mkdir(parents=True, exist_ok=True)
        _save(flat_root, "flat", config)
    else:
        for layout in ("hsdp", "parameter", "tp"):
            _exchange(root, flat_root, config, layout, world)
    print(
        f"CROSS PASS {args.action} {args.backend} world={world} rank={get_rank()}",
        flush=True,
    )


def _exchange(root, flat_root, config, layout, world):
    path = root / layout
    if world == 4:
        _load(flat_root, config, layout)
        path.mkdir(parents=True, exist_ok=True)
        _save(path, layout, config)
        _load(path, config, layout, same_layout=True)
    else:
        _load(path, config, "flat")


def _lazy_source(device, mesh, partial):
    model, opt = _model(device, mesh)
    if partial:
        model(torch.ones(2, 8, device=device)).sum().backward()
        model.get_parameter("1.weight").grad = None
        opt.step()
        opt.zero_grad(set_to_none=True)
    return model, opt


def _lazy_resume(path, device, mesh, warm, before, after):
    model, opt = _model(device, mesh)
    if warm:
        _step(model, opt, device, 7)
    FSDP2Checkpointer.load(path, {"model": model, "optimizer": opt})
    _check(model, opt, before, exact=True)
    assert all(opt.state.values())
    _step(model, opt, device, 1)
    _check(model, opt, after, exact=True)


def _exercise_lazy_adamw(root, device, mesh, partial):
    source, opt = _lazy_source(device, mesh, partial)
    reference, reference_opt = _lazy_source(device, mesh, partial)
    before = _snapshot(reference, reference_opt)
    _step(reference, reference_opt, device, 1)
    after = _snapshot(reference, reference_opt)
    formats = ["dcp", "safetensors", "hf"]
    for fmt in formats:
        FSDP2Checkpointer.save(
            root / fmt, {"model": source, "optimizer": opt}, checkpoint_format=fmt
        )
        _check(source, opt, before, exact=True)
        assert len(opt.state) == int(partial)
    if partial:
        ms, osd = get_state_dict(source, opt)
        dcp.save({"model": ms, "optimizer": osd}, checkpoint_id=root / "bundle")
        formats.append("bundle")
    for fmt in formats:
        for warm in (False, True):
            _lazy_resume(root / fmt, device, mesh, warm, before, after)


def lazy_adamw(args):
    device = "cpu" if args.backend == "gloo" else "cuda"
    mesh = init_device_mesh(device, (get_world_size(),))
    for partial in (False, True):
        _exercise_lazy_adamw(Path(args.path) / str(partial), device, mesh, partial)
    print(f"lazy AdamW PASS {args.backend} rank={get_rank()}", flush=True)


def cross_hsdp_tp(args):
    # Exercise the native DCP reader and writer as well as all GDCP formats
    # while adding or removing a real replica axis across separate process groups.
    world = get_world_size()
    shape = (2, 2, 2) if world == 8 else (2, 2)
    names = ("replica", "dp", "tp") if world == 8 else ("dp", "tp")
    mesh = init_device_mesh("cpu", shape, mesh_dim_names=names)
    config = "cpu", None, mesh
    root = Path(args.path)
    if args.action == "hsdp_tp_expand":
        _load(root / "2d", config, "tp")
    if args.action == "hsdp_tp_contract":
        _load(root / "3d", config, "tp")
    else:
        path = root / ("3d" if world == 8 else "2d")
        path.mkdir(parents=True, exist_ok=True)
        _save(path, "tp", config)
    print(f"HSDP+TP interchange PASS world={world} rank={get_rank()}", flush=True)
