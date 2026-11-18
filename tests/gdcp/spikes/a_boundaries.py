"""torchrun --standalone --nproc-per-node=4 --module tests.gdcp.spikes.a_boundaries."""

import os
from math import prod
from pathlib import Path

import torch
from torch.distributed import destroy_process_group, get_rank, get_world_size
from torch.distributed import init_process_group

import gdtensor
import gdtensor.builder as gb
from gdtensor.io import rank_save
from gdtensor_conn.coord import GlooCoordinator
from tests.support.gdcp_spikes import BytesAccessor, model, storage


def main():
    init_process_group("gloo")
    rank, world = get_rank(), get_world_size()
    state = model().state_dict()
    rows, local, origins = [], {}, {}
    for key, tensor in state.items():
        rows.append(gb.NamedDTensor(key, gdtensor.make_gd_tensor(tensor)))
        local[key] = tensor.to_local().detach()
        chunk = (tensor.shape[0] + world - 1) // world
        start = min(rank * chunk, tensor.shape[0])
        rowbytes = prod(tensor.shape[1:]) * tensor.element_size()
        origins[key] = start * rowbytes
        expected = torch.arange(tensor.numel()).reshape(tensor.shape).float()
        assert torch.equal(local[key], expected[start : start + chunk]), key
    table = gb.DTensorTable(rows)
    accessor = BytesAccessor(local, origins)
    compute = gb.ComputeDTensorsView(list(state), table, accessor)
    root = Path(os.environ["GDCP_SPIKE_ROOT"]) / "a"
    view = storage(root, table, world, GlooCoordinator())
    result = rank_save(rank, compute, view)
    assert type(result) is gb.Ok, result
    assert sum(size for _, _, size in accessor.ranges) == sum(
        t.numel() * t.element_size() for t in local.values()
    )
    print(
        f"SPIKE A PASS rank={rank}/{world} shapes={[(k, list(t.shape)) for k, t in local.items()]} ranges={accessor.ranges}",
        flush=True,
    )
    destroy_process_group()


if __name__ == "__main__":
    main()
