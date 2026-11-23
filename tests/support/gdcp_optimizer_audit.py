"""GDCP v1 optimizer ownership corner cases on real Gloo process groups."""

from pathlib import Path
from datetime import timedelta

import torch
from torch.distributed import (
    ProcessGroup,
    barrier,
    destroy_process_group,
    init_process_group,
    new_group,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)

from gdcp import FSDP2Checkpointer
from tests.support.gdcp_optimizers import FORMATS, model_for


def _state(model, value):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
    optimizer.state[model.weight]["statistic"] = value
    return optimizer


def _reject(root, model, optimizer, fmt):
    try:
        FSDP2Checkpointer.save(
            root, {"model": model, "optimizer": optimizer}, checkpoint_format=fmt
        )
    except ValueError as exc:
        assert "state.weight.statistic" in str(exc)
        assert "needs to be a DTensor" in str(exc)
    else:
        raise AssertionError("ambiguous per-rank optimizer bytes were accepted")
    assert not root.exists()


def _plain_cases(root, rank, mesh, fmt):
    model = model_for(mesh)
    value = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    optimizer = _state(model, value)
    path = root / fmt / "identical-parameter-shape"
    FSDP2Checkpointer.save(
        path, {"model": model, "optimizer": optimizer}, checkpoint_format=fmt
    )
    target = model_for(mesh, 10)
    loaded = torch.optim.SGD(target.parameters(), lr=0.5)
    FSDP2Checkpointer.load(path, {"model": target, "optimizer": loaded})
    restored = loaded.state[target.weight]["statistic"]
    assert not isinstance(restored, DTensor)
    torch.testing.assert_close(restored, value, rtol=0, atol=0)

    optimizer.state[model.weight]["statistic"] = value + rank
    _reject(root / fmt / "divergent-parameter-shape", model, optimizer, fmt)
    optimizer.state[model.weight]["statistic"] = torch.ones(
        3, dtype=torch.float64 if rank else torch.float32
    )
    _reject(root / fmt / "divergent-dtype", model, optimizer, fmt)
    optimizer.state[model.weight]["statistic"] = torch.ones(3 + rank)
    _reject(root / fmt / "divergent-shape", model, optimizer, fmt)


def _unsupported_state(root, mesh):
    model = model_for(mesh)
    value = DTensor.from_local(
        torch.ones(3),
        mesh,
        [Partial()],
        run_check=False,
        shape=torch.Size([3]),
        stride=(1,),
    )
    optimizer = _state(model, value)
    try:
        FSDP2Checkpointer.save(root, {"model": model, "optimizer": optimizer})
    except NotImplementedError as exc:
        assert "state.weight.statistic" in str(exc)
        assert "Partial" in str(exc)
    else:
        raise AssertionError("unsupported state placement was accepted")
    assert not root.exists()


def plain_worker(rank, world, rendezvous, root):
    init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        mesh = DeviceMesh("cpu", list(range(world)))
        for fmt in FORMATS:
            _plain_cases(Path(root), rank, mesh, fmt)
        _unsupported_state(Path(root) / "unsupported-placement", mesh)
        print(f"optimizer audit plain rank {rank} PASS", flush=True)
    finally:
        destroy_process_group()


def _submesh_case(root, parameter_mesh, state_mesh, fmt, placement):
    model = model_for(parameter_mesh)
    placements = [placement] + [Replicate()] * (state_mesh.ndim - 1)
    value = distribute_tensor(
        torch.arange(7, dtype=torch.float32), state_mesh, placements, src_data_rank=None
    )
    optimizer = _state(model, value)
    path = root / fmt
    FSDP2Checkpointer.save(
        path, {"model": model, "optimizer": optimizer}, checkpoint_format=fmt
    )
    target = model_for(parameter_mesh, 10)
    loaded = torch.optim.SGD(target.parameters(), lr=0.5)
    FSDP2Checkpointer.load(path, {"model": target, "optimizer": loaded})
    restored = loaded.state[target.weight]["statistic"]
    torch.testing.assert_close(
        restored.device_mesh.mesh, state_mesh.mesh, rtol=0, atol=0
    )
    assert restored.device_mesh.mesh_dim_names == state_mesh.mesh_dim_names
    assert restored.placements == tuple(placements)
    torch.testing.assert_close(restored.to_local(), value.to_local(), rtol=0, atol=0)
    if state_mesh.get_coordinate() is None:
        assert restored.to_local().numel() == 0
        return
    torch.testing.assert_close(
        restored.full_tensor(), torch.arange(7, dtype=torch.float32), rtol=0, atol=0
    )


def submesh_worker(rank, world, rendezvous, root):
    init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        parameter_mesh = DeviceMesh("cpu", list(range(world)))
        meshes = DeviceMesh(
            "cpu",
            torch.arange(world).reshape(2, 2),
            mesh_dim_names=("replica", "state"),
        )
        unnamed = DeviceMesh("cpu", torch.arange(world).reshape(2, 2))
        reverse_group = new_group([1, 0, 3, 2], backend="gloo", sort_ranks=False)
        assert isinstance(reverse_group, ProcessGroup)
        reverse = DeviceMesh.from_group(
            reverse_group, "cpu", mesh=[1, 0, 3, 2], mesh_dim_names=("reverse",)
        )
        for fmt in FORMATS:
            _submesh_format(
                Path(root) / "siblings", parameter_mesh, meshes["state"], fmt
            )
            _submesh_format(Path(root) / "unnamed-2d", parameter_mesh, unnamed, fmt)
            _submesh_format(Path(root) / "reverse", parameter_mesh, reverse, fmt)
        print(f"optimizer audit submesh rank {rank} PASS", flush=True)
    finally:
        destroy_process_group()


def _submesh_format(root, parameter_mesh, state_mesh, fmt):
    _submesh_case(root / "sharded", parameter_mesh, state_mesh, fmt, Shard(0))
    _submesh_case(root / "replicated", parameter_mesh, state_mesh, fmt, Replicate())


def sole_submesh_worker(rank, world, rendezvous, root):
    init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=20),
    )
    try:
        parameter_mesh = DeviceMesh("cpu", list(range(world)))
        state_mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("state",))
        for fmt in FORMATS:
            _submesh_format(Path(root), parameter_mesh, state_mesh, fmt)
        print(f"optimizer sole submesh rank {rank} PASS", flush=True)
    finally:
        destroy_process_group()


def independent_world_worker(rank, world, rendezvous, root, action):
    init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=20),
    )
    try:
        mesh = DeviceMesh("cpu", list(range(world)))
        model = model_for(mesh)
        for fmt in FORMATS:
            _independent_world_format(Path(root) / fmt, model, mesh, action)
        print(
            f"optimizer independent world {world} {action} rank {rank} PASS", flush=True
        )
    finally:
        destroy_process_group()


def _independent_world_format(path, model, mesh, action):
    if action == "save":
        value = distribute_tensor(
            torch.arange(7, dtype=torch.float32), mesh, [Shard(0)]
        )
        FSDP2Checkpointer.save(
            path,
            {"model": model, "optimizer": _state(model, value)},
            checkpoint_format=path.name,
        )
        return
    loaded = torch.optim.SGD(model.parameters(), lr=0.5)
    assert isinstance(model.weight, DTensor)
    before = model.weight.to_local().clone()
    try:
        FSDP2Checkpointer.load(path, {"model": model, "optimizer": loaded})
    except ValueError as exc:
        assert "needs an initialized destination DTensor" in str(exc)
    else:
        raise AssertionError(
            "independent mesh change was accepted without a destination layout"
        )
    assert not loaded.state
    torch.testing.assert_close(model.weight.to_local(), before, rtol=0, atol=0)
    value = distribute_tensor(torch.zeros(7), mesh, [Shard(0)])
    loaded.state[model.weight]["statistic"] = value
    FSDP2Checkpointer.load(path, {"model": model, "optimizer": loaded})
    restored = loaded.state[model.weight]["statistic"]
    torch.testing.assert_close(
        restored.full_tensor(), torch.arange(7, dtype=torch.float32), rtol=0, atol=0
    )


def independent_subgroup_worker(rank, world, rendezvous, root):
    init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=20),
    )
    try:
        grid = DeviceMesh(
            "cpu",
            torch.arange(world).reshape(2, 2),
            mesh_dim_names=("checkpoint", "job"),
        )
        state_meshes = [
            DeviceMesh("cpu", [owner], mesh_dim_names=("state",)) for owner in range(2)
        ]
        mesh = grid["checkpoint"]
        for fmt in FORMATS:
            _independent_subgroup_format(
                Path(root) / f"job-{rank % 2}" / fmt, mesh, state_meshes[rank % 2]
            )
        barrier()
        for fmt in FORMATS:
            _remap_independent_subgroup(
                Path(root) / f"job-{1 - rank % 2}" / fmt, mesh, state_meshes[rank % 2]
            )
        print(f"optimizer independent subgroup rank {rank} PASS", flush=True)
    finally:
        destroy_process_group()


def _independent_subgroup_format(path, mesh, state_mesh):
    from unittest.mock import patch
    from gdcp.adapters import optimizer_state

    source = model_for(mesh)
    value = distribute_tensor(
        torch.arange(7, dtype=torch.float32), state_mesh, [Shard(0)]
    )
    optimizer = _state(source, value)
    optimizer.state[source.weight]["duplicate"] = value.clone()
    checkpoint_group = mesh.get_group()
    FSDP2Checkpointer.save(
        path,
        {"model": source, "optimizer": optimizer},
        checkpoint_format=path.name,
        process_group=checkpoint_group,
    )
    target = model_for(mesh, 10)
    loaded = torch.optim.SGD(target.parameters(), lr=0.5)
    with patch.object(
        optimizer_state, "new_group", wraps=optimizer_state.new_group
    ) as create_group:
        FSDP2Checkpointer.load(
            path, {"model": target, "optimizer": loaded}, process_group=checkpoint_group
        )
    assert create_group.call_count == 1, (
        "independent fields must reuse their load's process group"
    )
    _assert_subgroup_slots(loaded.state[target.weight], state_mesh)


def _assert_subgroup_slots(fields, state_mesh):
    for field in ("statistic", "duplicate"):
        restored = fields[field]
        torch.testing.assert_close(
            restored.device_mesh.mesh, state_mesh.mesh, rtol=0, atol=0
        )
        if state_mesh.get_coordinate() is None:
            assert restored.to_local().numel() == 0
        else:
            torch.testing.assert_close(
                restored.full_tensor(),
                torch.arange(7, dtype=torch.float32),
                rtol=0,
                atol=0,
            )


def _remap_independent_subgroup(path, mesh, state_mesh):
    target = model_for(mesh, 10)
    optimizer = torch.optim.SGD(target.parameters(), lr=0.5)
    FSDP2Checkpointer.load(
        path, {"model": target, "optimizer": optimizer}, process_group=mesh.get_group()
    )
    _assert_subgroup_slots(optimizer.state[target.weight], state_mesh)


def independent_multidim_worker(rank, world, rendezvous, root):
    init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=20),
    )
    try:
        parameter_mesh = DeviceMesh("cpu", list(range(world)))
        state_mesh = DeviceMesh("cpu", torch.arange(world).reshape(2, 2))
        for fmt in FORMATS:
            _multidim_state_format(Path(root) / fmt, parameter_mesh, state_mesh)
        print(f"optimizer independent multidim rank {rank} PASS", flush=True)
    finally:
        destroy_process_group()


def _multidim_state_format(path, parameter_mesh, state_mesh):
    source = model_for(parameter_mesh)
    value = distribute_tensor(
        torch.arange(7, dtype=torch.float32), state_mesh, [Shard(0), Replicate()]
    )
    FSDP2Checkpointer.save(
        path,
        {"model": source, "optimizer": _state(source, value)},
        checkpoint_format=path.name,
    )
    target = model_for(parameter_mesh, 10)
    optimizer = torch.optim.SGD(target.parameters(), lr=0.5)
    FSDP2Checkpointer.load(path, {"model": target, "optimizer": optimizer})
    restored = optimizer.state[target.weight]["statistic"]
    torch.testing.assert_close(
        restored.device_mesh.mesh, state_mesh.mesh, rtol=0, atol=0
    )
    assert restored.device_mesh.mesh_dim_names is None
    assert restored.placements == (Shard(0), Replicate())
    torch.testing.assert_close(
        restored.full_tensor(), torch.arange(7, dtype=torch.float32), rtol=0, atol=0
    )
