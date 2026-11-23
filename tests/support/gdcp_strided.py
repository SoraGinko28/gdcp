"""Real FSDP2 checkpoints with nonrectangular DTensor buffers and optimizer state."""

from pathlib import Path

import torch
from torch import nn
from torch.distributed import get_rank, get_world_size
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard, Partial, distribute_tensor

from gdcp import FSDP2Checkpointer
from gdcp.adapters.torch_placements import strided_shard
from tests.support.gdcp_strided_probe import cases


def _model(device, world, layout):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.arange(64, device=device).float().reshape(8, 8) / 128
            )

    model = Model()
    for name, mesh_shape, placements, shape in cases(4 if layout == "flat" else world):
        dense = torch.arange(torch.tensor(shape).prod().item(), device=device).reshape(
            shape
        )
        if layout == "flat":
            mesh_shape, placements = (world,), (Shard(0),)
            dense = dense.flatten() if name.startswith(("flatten", "view")) else dense
        mesh = DeviceMesh(device, torch.arange(world).reshape(mesh_shape))
        tensor = distribute_tensor(dense, mesh, placements, src_data_rank=None)
        if name.startswith("flatten"):
            tensor = tensor.flatten()
        elif name.startswith("view"):
            tensor = tensor.view(-1)
        model.register_buffer(name, tensor)
    # Add strided dimensions with nontrivial outer and inner strides.
    for name, shape, placements in (
        ("column_strides", (5, 13), (strided_shard(1, split_factor=3),)),
        (
            "nested_strides",
            (37, 3),
            (strided_shard(0, split_factor=3), strided_shard(0, split_factor=2)),
        ),
    ):
        grid = (world,) if len(placements) == 1 else (2, world // 2)
        grid = (world,) if layout == "flat" else grid
        mesh = DeviceMesh(device, torch.arange(world).reshape(grid))
        dense = torch.arange(torch.tensor(shape).prod().item(), device=device).reshape(
            shape
        )
        if layout == "flat":
            mesh = DeviceMesh(device, torch.arange(world))
            placements = (Shard(0),)
        model.register_buffer(
            name, distribute_tensor(dense, mesh, placements, src_data_rank=None)
        )
    fully_shard(model, mesh=DeviceMesh(device, torch.arange(world)))
    return model


def _assert_values(model):
    for name, tensor in model.named_buffers():
        expected = torch.arange(tensor.numel(), device=tensor.device).reshape(
            tensor.shape
        )
        oracle = distribute_tensor(
            expected, tensor.device_mesh, tensor.placements, src_data_rank=None
        )
        torch.testing.assert_close(
            tensor.to_local(),
            oracle.to_local(),
            rtol=0,
            atol=0,
            msg=lambda msg: f"{name}: {msg}",
        )


@torch.no_grad()
def _poison(model):
    for tensor in model.buffers():
        tensor.to_local().fill_(-7)


def _roundtrip(path, model, fmt, device, world):
    FSDP2Checkpointer.save(path, {"model": model}, checkpoint_format=fmt)
    for destination in ("strided", "flat"):
        target = _model(device, world, destination)
        _poison(target)
        FSDP2Checkpointer.load(path, {"model": target})
        _assert_values(target)


def _cross_load(root, device, world):
    targets = ("strided", "flat") if world % 2 == 0 else ("flat",)
    for layout in ("strided", "flat"):
        for fmt in ("safetensors", "dcp", "hf"):
            _load_targets(root / layout / fmt, device, world, targets)


def _load_targets(path, device, world, targets):
    for layout in targets:
        target = _model(device, world, layout)
        _poison(target)
        FSDP2Checkpointer.load(path, {"model": target})
        _assert_values(target)


def _save_or_roundtrip(path, model, fmt, device, world, phase):
    if phase == "save":
        FSDP2Checkpointer.save(path, {"model": model}, checkpoint_format=fmt)
    else:
        _roundtrip(path, model, fmt, device, world)


def strided(args):
    device = "cpu" if args.backend == "gloo" else "cuda"
    world = get_world_size()
    root = Path(args.path)
    phase = getattr(args, "phase", "roundtrip")
    if phase == "load":
        _cross_load(root, device, world)
        print(f"general strided cross load PASS rank={get_rank()}", flush=True)
        return
    for layout in ("strided", "flat"):
        model = _model(device, world, layout)
        _assert_values(model)
        for fmt in ("safetensors", "dcp", "hf"):
            path = root / layout / fmt
            _save_or_roundtrip(path, model, fmt, device, world, phase)
    if phase == "save":
        print(f"general strided cross save PASS rank={get_rank()}", flush=True)
        return
    # A persistently unsupported placement must fail during preparation.
    model = _model(device, world, "flat")
    mesh = DeviceMesh(device, torch.arange(world))
    model.register_buffer(
        "unsupported_partial",
        DTensor.from_local(
            torch.ones(8, device=device), mesh, (Partial(),), run_check=False
        ),
    )
    bad = root / "unsupported"
    try:
        FSDP2Checkpointer.save(bad, {"model": model})
    except (NotImplementedError, RuntimeError) as exc:
        assert "unsupported_partial" in str(exc) and "Partial" in str(exc)
    else:
        raise AssertionError("Partial checkpoint unexpectedly saved")
    assert not bad.exists(), "unsupported placement created checkpoint files"
    print(f"general strided PASS {args.backend} rank={get_rank()}", flush=True)


def _file_worker(rank, world, rendezvous, args):
    from torch.distributed import init_process_group, destroy_process_group

    init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        strided(args)
    finally:
        destroy_process_group()


def main():
    import argparse
    import tempfile
    import torch.multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--backend", default="gloo")
    parser.add_argument(
        "--phase", choices=("roundtrip", "save", "load"), default="roundtrip"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gdcp-strided-store-") as root:
        mp.spawn(
            _file_worker,
            args=(args.world, str(Path(root) / "rendezvous"), args),
            nprocs=args.world,
            join=True,
        )


if __name__ == "__main__":
    main()
