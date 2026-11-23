"""Record real DTensor ownership using a file rendezvous (no torchrun DNS).

Run: python -m tests.support.gdcp_strided_probe --output probe.json
Each record comes from distribute_tensor(arange).to_local() and full_tensor(),
never PyTorch's documented offset helper. A file store works on the Mac where
standalone torchrun's hostname rendezvous hangs.
"""

import argparse
import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.distributed import (
    init_process_group,
    destroy_process_group,
    all_gather_object,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Placement,
    Replicate,
    Shard,
)

from gdcp.adapters.torch_placements import strided_shard, encode_placements


def cases(world):
    s = strided_shard
    layouts: list[
        tuple[str, tuple[int, ...], tuple[Placement, ...], tuple[int, ...]]
    ] = [
        ("dp_tp", (2, world // 2), (s(0, split_factor=world // 2), Shard(0)), (16,)),
        ("strided_alone_even", (world,), (s(0, split_factor=3),), (24,)),
        ("strided_alone_uneven", (world,), (s(0, split_factor=3),), (25,)),
        (
            "nested_strided_uneven",
            (2, world // 2),
            (s(0, split_factor=3), s(0, split_factor=2)),
            (37, 3),
        ),
        ("shard_strided_even", (2, world // 2), (Shard(0), s(0, split_factor=2)), (8,)),
        (
            "shard_strided_uneven",
            (2, world // 2),
            (Shard(0), s(0, split_factor=3)),
            (13,),
        ),
        ("flatten_1d", (world,), (Shard(1),), (2, 8)),
        ("flatten_2d", (2, world // 2), (Shard(0), Shard(1)), (4, 8)),
        ("flatten_3d", (2, 1, world // 2), (Shard(0), Replicate(), Shard(1)), (4, 8)),
    ]
    three_axes = (2, 2, 2) if world == 8 else (2, 1, world // 2)
    layouts.extend(
        [
            (
                "flatten_3d_all_sharded",
                three_axes,
                (Shard(0), Shard(1), Shard(2)),
                (4, 4, 4),
            ),
            (
                "view_3d_all_sharded",
                three_axes,
                (Shard(0), Shard(1), Shard(2)),
                (4, 4, 4),
            ),
        ]
    )
    return layouts


def worker(rank, world, rendezvous, output):
    init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    reports = []
    for name, mesh_shape, placements, shape in cases(world):
        mesh = DeviceMesh(
            "cpu",
            torch.arange(world).reshape(mesh_shape),
            mesh_dim_names=tuple(f"d{i}" for i in range(len(mesh_shape))),
        )
        dense = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
        tensor = distribute_tensor(dense, mesh, placements, src_data_rank=None)
        if name.startswith("flatten"):
            tensor = tensor.flatten()
        elif name.startswith("view"):
            tensor = tensor.view(-1)
        assert isinstance(tensor, DTensor)
        local = tensor.to_local().reshape(-1).tolist()
        full = tensor.full_tensor().reshape(-1).tolist()
        matches = full == dense.reshape(-1).tolist()
        assert matches or name == "nested_strided_uneven", (name, rank, full)
        records = [None] * world
        all_gather_object(
            records,
            {
                "rank": rank,
                "local_shape": list(tensor.to_local().shape),
                "global_elements": local,
            },
        )
        if rank == 0:
            reports.append(
                {
                    "case": name,
                    "mesh": list(mesh_shape),
                    "shape": list(tensor.shape),
                    "placements": encode_placements(tensor.placements),
                    "ranks": records,
                    "full_tensor_arange": matches,
                    "full_tensor_elements": full,
                    "torch_full_tensor_bug": not matches,
                }
            )
    # Real FSDP2 over a TP-sharded parameter, followed by view propagation.
    from torch import nn
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

    mesh = DeviceMesh(
        "cpu", torch.arange(world).reshape(2, world // 2), mesh_dim_names=("dp", "tp")
    )
    model = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(64).reshape(8, 8))
    parallelize_module(model, mesh["tp"], ColwiseParallel())
    fully_shard(model, mesh=mesh["dp"])
    tensor = model.weight.flatten()
    assert isinstance(tensor, DTensor)
    full = tensor.full_tensor().reshape(-1).tolist()
    assert full == list(range(64))
    records = [None] * world
    all_gather_object(
        records,
        {
            "rank": rank,
            "local_shape": list(tensor.to_local().shape),
            "global_elements": tensor.to_local().reshape(-1).tolist(),
        },
    )
    if rank == 0:
        reports.append(
            {
                "case": "flatten_fsdp_tp",
                "mesh": [2, world // 2],
                "shape": list(tensor.shape),
                "placements": encode_placements(tensor.placements),
                "ranks": records,
                "full_tensor_arange": True,
            }
        )
    if rank == 0:
        Path(output).write_text(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "time_pst": datetime.now(timezone(timedelta(hours=-8))).isoformat(),
                    "method": "real Gloo ranks, file rendezvous; distribute_tensor(arange, src_data_rank=None), to_local, full_tensor; fully_shard over ColwiseParallel weight",
                    "cases": reports,
                },
                indent=2,
            )
            + "\n"
        )
    destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--world", type=int, default=4)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gdcp-strided-probe-") as root:
        mp.spawn(
            worker,
            args=(args.world, str(Path(root) / "rendezvous"), args.output),
            nprocs=args.world,
            join=True,
        )


if __name__ == "__main__":
    main()
