"""The single compatibility boundary for PyTorch's private strided placement.

The layout algorithms use public GDTensor placements. Only this module knows
PyTorch's private spelling; importing it gives an actionable error on a rename.
"""

from torch.distributed.tensor import Placement, Replicate, Shard

try:
    from torch.distributed.tensor.placement_types import _StridedShard
except ImportError as exc:
    raise ImportError(
        "GDCP requires torch.distributed.tensor.placement_types._StridedShard; "
        "compatibility was checked with torch 2.12, 2.14 and 2.15 on macOS; "
        "H20 verification is recorded separately in the GDCP v1 evidence"
    ) from exc


def is_strided(placement):
    return isinstance(placement, _StridedShard)


def strided_parts(placement):
    if not is_strided(placement):
        raise TypeError(f"expected a strided placement, got {placement!r}")
    return int(placement.dim), int(placement.split_factor)


def strided_shard(dim, *, split_factor=1):
    """Construct the private placement for compatibility tests and restore."""
    return _StridedShard(dim, split_factor=split_factor)


def encode_placements(placements):
    result = []
    for placement in placements:
        if is_strided(placement):
            dim, factor = strided_parts(placement)
            result.append(dict(kind="strided_shard", dim=dim, split_factor=factor))
        elif type(placement) is Shard:
            result.append(dict(kind="shard", dim=placement.dim))
        elif isinstance(placement, Replicate):
            result.append(dict(kind="replicate"))
        else:
            raise NotImplementedError(f"unsupported placement {placement!r}")
    return result


def decode_placements(records):
    result: list[Placement] = []
    for record in records:
        kind = record["kind"]
        if kind == "replicate":
            result.append(Replicate())
        elif kind == "shard":
            result.append(Shard(record["dim"]))
        elif kind == "strided_shard":
            result.append(
                strided_shard(record["dim"], split_factor=record["split_factor"])
            )
        else:
            raise ValueError(f"unknown saved torch placement {kind!r}")
    return tuple(result)


def local_shape(shape, mesh, placements):
    """Size only, using run arithmetic instead of per-element offset helpers."""
    sizes = list(shape)
    coordinates = mesh.get_coordinate()
    if coordinates is None:
        return (0,)
    for axis, placement in enumerate(placements):
        if isinstance(placement, Replicate):
            continue
        dim = placement.dim
        length, parts, rank = sizes[dim], mesh.size(axis), coordinates[axis]
        factor = strided_parts(placement)[1] if is_strided(placement) else 1
        first = -(-length // factor)
        full, tail = divmod(length, first) if first else (0, 0)
        chunk = -(-first // parts)
        tail_chunk = -(-tail // parts)
        total = full * max(0, min(chunk, first - rank * chunk))
        total += max(0, min(tail_chunk, tail - rank * tail_chunk))
        sizes[dim] = total
    return tuple(sizes)


local_shape_and_mesh = local_shape
