"""Continue arbitrary optimizer fields on their parameters' existing meshes."""

from pathlib import Path
from typing import Callable, overload
from unittest.mock import patch

import torch
from torch.distributed import get_rank, get_world_size
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    HSDPMeshInfo,
    ShardPlacementResult,
)
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from gdcp import FSDP2Checkpointer
from gdcp.adapters import optimizer_state
from tests.support.gdcp_optimizers import (
    FORMATS,
    LayoutOptimizer,
    _custom_slots,
    _resolved_reduction,
    assert_equal,
    update,
)


def _axis_mesh(parameter):
    mesh = parameter.device_mesh
    return mesh if mesh.ndim == 1 else mesh[mesh.mesh_dim_names[0]]


def _submesh(parameter):
    mesh = parameter.device_mesh
    return mesh if mesh.ndim < 3 else mesh[mesh.mesh_dim_names[:2]]


def _statistic(parameter, mesh):
    return distribute_tensor(
        torch.zeros(16, device=parameter.device),
        mesh,
        [Shard(0)] + [Replicate()] * (mesh.ndim - 1),
        src_data_rank=None,
    )


def _slots(parameter):
    fields = _custom_slots(parameter)
    fields["swapped"] = torch.zeros_like(parameter).transpose(0, 1).contiguous()
    fields["independent"] = _statistic(parameter, parameter.device_mesh)
    fields["axis"] = _statistic(parameter, _axis_mesh(parameter))
    fields["submesh"] = _statistic(parameter, _submesh(parameter))
    return fields


class MeshMomentum(LayoutOptimizer):
    """Flat and transposed momentum both participate in the parameter update."""

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            _update_group(group, self.state)
        return loss


def _update_group(group, state):
    for parameter in group["params"]:
        if parameter.grad is not None:
            _update(parameter, state[parameter], group["lr"])


def _update(parameter, fields, lr):
    if not fields:
        fields.update(_slots(parameter))
    fields["flat"].mul_(0.5).add_(parameter.grad.view(-1))
    fields["swapped"].mul_(0.25).add_(parameter.grad.transpose(0, 1))
    fields["rows"].add_(_resolved_reduction(parameter.grad, 1))
    fields["columns"].add_(_resolved_reduction(parameter.grad, 0))
    fields["replicated"].add_(0.25)
    for field, mesh in (
        ("independent", parameter.device_mesh),
        ("axis", _axis_mesh(parameter)),
        ("submesh", _submesh(parameter)),
    ):
        fields[field].add_(torch.ones_like(_statistic(parameter, mesh)))
    fields["count"] += 1
    parameter.add_(fields["flat"].view(parameter.shape), alpha=-lr)
    parameter.add_(fields["swapped"].transpose(0, 1), alpha=-lr)


def _model(config, layout):
    device, flat, grid, hybrid = config
    model = torch.nn.Sequential(
        *(torch.nn.Linear(8, 8, bias=False, device=device) for _ in range(3))
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(torch.arange(64, device=device).reshape(8, 8) / 128 + index)
    if layout in ("tp", "hsdp_tp"):
        mesh = hybrid if layout == "hsdp_tp" else grid
        parallelize_module(
            model, mesh["tp"], {"0": ColwiseParallel(), "1": RowwiseParallel()}
        )
        fully_shard(
            model,
            mesh=mesh["replica", "dp"] if layout == "hsdp_tp" else mesh["dp"],
        )
    elif layout == "parameter":
        placements = {
            model[0].weight: ShardPlacementResult(
                Shard(1), FSDPMeshInfo(grid["dp"], shard_mesh_dim=0)
            ),
            model[1].weight: ShardPlacementResult(
                Shard(0), HSDPMeshInfo(grid, shard_mesh_dim=1, replicate_mesh_dim=0)
            ),
        }
        fully_shard(model, mesh=flat, shard_placement_fn=placements.get)
    else:
        fully_shard(model, mesh=grid if layout == "hsdp" else flat)
    return model, MeshMomentum(model.parameters())


def _assert_meshes(model, optimizer):
    for name, parameter in model.named_parameters():
        fields = optimizer.state[parameter]
        for field in (
            "flat",
            "swapped",
            "rows",
            "columns",
            "replicated",
            "independent",
        ):
            value = fields[field]
            assert isinstance(value, DTensor)
            assert value.device_mesh is parameter.device_mesh, (name, field)
            assert value.device_mesh == parameter.device_mesh, (name, field)
        assert fields["axis"].device_mesh == _axis_mesh(parameter), name
        assert fields["submesh"].device_mesh == _submesh(parameter), name


def _equivalent_contiguous_mesh(mesh):
    # from_group preserves the existing communication group but gives this
    # equivalent slice contiguous rank storage: torch 2.15 compares its stride.
    groups = mesh.get_all_groups()
    return DeviceMesh.from_group(
        groups[0] if mesh.ndim == 1 else groups,
        mesh.device_type,
        mesh=mesh.mesh.contiguous(),
        mesh_dim_names=mesh.mesh_dim_names,
    )


def _initialize_destination(model, optimizer, *, equivalent=False):
    for parameter in model.parameters():
        fields = optimizer.state[parameter]
        fields.update(_slots(parameter))
        if equivalent:
            value = fields["swapped"]
            fields["swapped"] = DTensor.from_local(
                value.to_local().clone(),
                _equivalent_contiguous_mesh(parameter.device_mesh),
                value.placements,
                run_check=False,
                shape=value.shape,
                stride=value.stride(),
            )


def _restore(
    path,
    config,
    layout,
    source,
    saved,
    *,
    group=None,
    initialize=False,
    equivalent=False,
):
    target, optimizer = _model(config, layout)
    if initialize or equivalent:
        _initialize_destination(target, optimizer, equivalent=equivalent)
    with patch.object(
        optimizer_state, "new_group", wraps=optimizer_state.new_group
    ) as create:
        FSDP2Checkpointer.load(
            path, {"model": target, "optimizer": optimizer}, process_group=group
        )
    assert create.call_count == 0, "matching parameter meshes must reuse their groups"
    _assert_meshes(target, optimizer)
    try:
        assert_equal(target, optimizer, source, saved)
    except AssertionError as exc:
        raise AssertionError(
            f"restore {path} into {layout} initialized={initialize} equivalent={equivalent}: {exc}"
        ) from exc
    update(target, optimizer, 2)
    reference, expected = _model(config, layout)
    for step in range(3):
        update(reference, expected, step)
    assert_equal(target, optimizer, reference, expected)


def _roundtrip(root, config, layout, targets, *, group=None):
    source, saved = _model(config, layout)
    for step in range(2):
        update(source, saved, step)
    for fmt in FORMATS:
        path = root / fmt
        FSDP2Checkpointer.save(
            path,
            {"model": source, "optimizer": saved},
            checkpoint_format=fmt,
            process_group=group,
        )
        for target in targets:
            _restore(
                path,
                config,
                target,
                source,
                saved,
                group=group,
                initialize=target != layout,
            )
        if layout == "tp":
            _restore(path, config, layout, source, saved, equivalent=True)


def optimizer_meshes(args):
    device = "cpu" if args.backend == "gloo" else "cuda"
    world = get_world_size()
    flat = init_device_mesh(device, (world,), mesh_dim_names=("world",))
    grid = init_device_mesh(device, (world // 2, 2), mesh_dim_names=("dp", "tp"))
    hybrid = init_device_mesh(
        device, (2, world // 4, 2), mesh_dim_names=("replica", "dp", "tp")
    )
    config = device, flat, grid, hybrid
    root = Path(args.path)
    for layout in ("flat", "hsdp", "parameter", "tp", "hsdp_tp"):
        targets = (
            [layout, "tp"]
            if layout == "flat"
            else [layout, "flat"]
            if layout == "tp"
            else [layout]
        )
        _roundtrip(root / layout, config, layout, targets)
    # Independent checkpoint jobs use strided DP slices of one root mesh.
    subgroup_config = device, grid["dp"], grid, hybrid
    _roundtrip(
        root / f"subgroup-{get_rank() % 2}",
        subgroup_config,
        "flat",
        ["flat"],
        group=grid["dp"].get_group(),
    )
    print(
        f"optimizer meshes PASS {args.backend} world={world} rank={get_rank()}",
        flush=True,
    )
