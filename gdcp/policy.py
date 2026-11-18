"""One safetensors file per rank, with replicated tensors stored once."""

from gdtensor.builder import (
    ArrangedMesh,
    DTensorTable,
    NaiveDTensor,
    NamedDTensor,
    Replicate,
    ReshapedMesh,
    Shard,
    Slice,
    SlicedMesh,
    SlotOuterProductMesh,
)
from gdtensor.tag import ConstantOne, WorldSize
from gdtensor_conn.models import build_layout


def storage_mesh(world_size, ranks):
    """Canonical one-dimensional storage mesh for all ranks or one writer."""
    base = ReshapedMesh(
        SlotOuterProductMesh(
            ArrangedMesh(ConstantOne(1)), ArrangedMesh(WorldSize(world_size))
        ),
        [world_size],
    )
    if list(ranks) == list(range(world_size)):
        return base
    if len(ranks) != 1:
        raise ValueError("storage mesh must contain the world or one writer")
    return SlicedMesh(base, [Slice(ranks[0], ranks[0] + 1)])


def _stored_tensor(named, order, world_size):
    tensor = named.d_tensor
    if type(tensor) is not NaiveDTensor or len(tensor.placements) != 1:
        raise ValueError(f"{named.global_tensor_id}: expected a one-dimensional tensor")
    placement = tensor.placements[0]
    if type(placement) is Shard and placement.axis != 0:
        raise ValueError(f"{named.global_tensor_id}: only Shard(0) is supported")
    if type(placement) not in (Shard, Replicate):
        raise ValueError(f"{named.global_tensor_id}: unsupported placement")
    ranks = [order % world_size] if type(placement) is Replicate else range(world_size)
    stored = NaiveDTensor(
        tensor.global_tensor_meta, [placement], storage_mesh(world_size, ranks)
    )
    return NamedDTensor(named.global_tensor_id, stored, named.alias)


class PerRankPolicy:
    def plan(self, component, world_size):
        if world_size < 1:
            raise ValueError("world_size must be positive")
        table = DTensorTable(
            [
                _stored_tensor(named, order, world_size)
                for order, named in enumerate(component.table.named_d_tensors)
            ]
        )
        layout = build_layout(
            table,
            world_size,
            "safetensors",
            lambda order, rank, named: f"__{rank}_0.safetensors",
        )
        return table, layout
