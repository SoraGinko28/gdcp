"""Small public-API fixtures shared by GDCP's executable spike scripts."""

from pathlib import Path

import torch
from torch.distributed import get_rank, get_world_size
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

import gdtensor.builder as gb
import gdtensor.tag as gt
from gdtensor.access import DTensorDataAccessor
from gdtensor_conn.formats import make_storage_view


class SpikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(40).reshape(10, 4).float())
        self.tiny = torch.nn.Parameter(torch.arange(6).reshape(2, 3).float())
        self.bias = torch.nn.Parameter(torch.arange(10).float())


def model():
    mesh = init_device_mesh("cpu", (get_world_size(),), mesh_dim_names=("dp",))
    sharded = SpikeModel()
    fully_shard(sharded, mesh=mesh)
    return sharded


def storage(root, compute_table, world, coord=None):
    ids = [nd.global_tensor_id for nd in compute_table.named_d_tensors]
    mesh = gb.ReshapedMesh(
        gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
        ),
        [world],
    )
    tensors = [
        gb.NamedDTensor(
            nd.global_tensor_id,
            gb.NaiveDTensor(
                nd.d_tensor.global_tensor_meta, nd.d_tensor.placements, mesh
            ),
        )
        for nd in compute_table.named_d_tensors
    ]
    full = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(len(ids))),
        gb.ArrangedMesh(gt.WorldSize(world)),
    )
    group = gb.DisjointStorageMeshList([full], ids)
    entries = {
        gb.TensorComponentPos(tid, rank): group for tid in ids for rank in range(world)
    }
    paths = {gb.leader_tensor_component_pos_of(group): "spike.safetensors"}
    layout = gb.PhysicalLayoutTable("safetensors", entries, paths)
    Path(root).mkdir(parents=True, exist_ok=True)
    return make_storage_view(
        ids, gb.DTensorTable(tensors), layout, str(root), coord=coord
    )


class BytesAccessor(DTensorDataAccessor):
    """Check public global ranges against live DTensor local byte ownership."""

    def __init__(self, tensors, origins):
        self.tensors, self.origins = tensors, origins
        self.ranges = []

    def buffer(self, rng):
        data = self.tensors[rng.global_tensor_id].reshape(-1).view(torch.uint8)
        offset = rng.global_tensor_offset - self.origins[rng.global_tensor_id]
        assert 0 <= offset <= data.numel() - rng.size, (get_rank(), rng, offset)
        self.ranges.append((rng.global_tensor_id, rng.global_tensor_offset, rng.size))
        return data[offset : offset + rng.size]

    def read_to_buffer(self, global_range, read_to_buffer):
        data = self.buffer(global_range)
        return read_to_buffer(data.data_ptr(), data.numel())

    def write_from_buffer(self, global_range, write_from_buffer):
        data = self.buffer(global_range)
        return write_from_buffer(data.data_ptr(), data.numel())
