"""Deterministic checkpoint manifests and explicit storage reconstruction."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from gdtensor.builder import (
    ArrangedMesh,
    DTensorTable,
    DisjointStorageMeshList,
    Err,
    GlobalTensorMeta,
    NaiveDTensor,
    NamedDTensor,
    PhysicalLayoutTable,
    Replicate,
    Shard,
    Slice,
    SlicedMesh,
    SlotOuterProductMesh,
    TensorComponentPos,
    file_path_of,
)
from gdtensor.tag import TensorCount, WorldSize

from .policy import storage_mesh

MANIFEST_NAME = "gdcp_manifest.json"
HF_COMMITTED_NAME = ".gdtensor_committed"
PENDING_NAME = ".gdcp_pending"


def checkpoint_path(path):
    raw = os.fspath(path)
    # pathlib collapses the double slash when the caller passes a Path.
    if raw.lower().startswith("hdfs:/"):
        raise ValueError(
            "hdfs:// checkpoint paths are unsupported; use a shared filesystem path"
        )
    return Path(raw)


@dataclass
class CheckpointDesc:
    manifest: dict

    @property
    def world_size(self):
        return self.manifest["world_size"]

    @property
    def components(self):
        return self.manifest["components"]

    @property
    def rank_state(self):
        return self.manifest["rank_state"]

    @property
    def framework(self):
        return self.manifest["framework"]

    @property
    def global_steps(self):
        return self.manifest["global_steps"]

    def to_dict(self):
        return self.manifest


def _tensor_description(named, world_size, logical_shapes):
    tensor = named.d_tensor
    mesh = tensor.mesh
    ranks = list(range(world_size))
    if type(mesh) is SlicedMesh:
        part = mesh.slices[0]
        if part is not None:
            ranks = ranks[slice(part.start, part.stop)]
    placement = tensor.placements[0]
    meta = tensor.global_tensor_meta
    desc = {
        "id": named.global_tensor_id,
        "shape": list(meta.global_shape),
        "dtype": meta.dtype,
        "itemsize": meta.itemsize,
        "placements": [{"shard": placement.axis}]
        if type(placement) is Shard
        else [{"replicate": None}],
        "mesh": {"shape": [len(ranks)], "ranks": ranks},
    }
    if named.global_tensor_id in logical_shapes:
        desc["logical_shape"] = list(logical_shapes[named.global_tensor_id])
    if named.alias:
        desc["aliases"] = list(named.alias)
    return desc


def describe_component(
    table, layout, world_size, *, objects=False, logical_shapes=None, root=None
):
    files = {}
    orders = {
        named.global_tensor_id: i for i, named in enumerate(table.named_d_tensors)
    }
    positions = sorted(
        layout.tensor_entries, key=lambda pos: (orders[pos.tensor_id], pos.rank)
    )
    for pos in positions:
        path = file_path_of(layout, pos)
        if type(path) is Err:
            raise ValueError(path.value.msg)
        files.setdefault(
            path.value,
            {
                "bytes": 0,
                "slots": [],
                "rectangles": _layout_rectangles(layout.tensor_entries[pos]),
            },
        )["slots"].append([pos.tensor_id, pos.rank])
    if root is not None:
        _record_sizes(files, root)
    return {
        "format": layout.container_format,
        "tensors": [
            _tensor_description(named, world_size, logical_shapes or {})
            for named in table.named_d_tensors
        ],
        "files": files,
        "objects": "objects.json" if objects else None,
    }


def _record_sizes(files, root):
    for name, info in files.items():
        info["bytes"] = (Path(root) / name).stat().st_size


def _layout_rectangles(group):
    return [
        [[part.start, part.stop, part.step] for part in mesh.slices]
        for mesh in group.disjoint_storage_mesh_list
    ]


def _tensor_from_description(info, world_size):
    ranks = info["mesh"]["ranks"]
    if info["mesh"]["shape"] != [len(ranks)]:
        raise ValueError(f"{info['id']}: invalid storage mesh shape")
    if not ranks or any(
        type(rank) is not int or not 0 <= rank < world_size for rank in ranks
    ):
        raise ValueError(f"{info['id']}: invalid storage ranks")
    placements = info["placements"]
    if placements == [{"shard": 0}]:
        placement = Shard(0)
    elif placements == [{"replicate": None}]:
        placement = Replicate()
    else:
        raise ValueError(f"{info['id']}: unsupported storage placements")
    tensor = NaiveDTensor(
        GlobalTensorMeta(info["shape"], info["dtype"], info["itemsize"]),
        [placement],
        storage_mesh(world_size, ranks),
    )
    return NamedDTensor(info["id"], tensor, info.get("aliases", []))


def _layout_file(base, ids, orders, slots, rectangles=None):
    if rectangles is not None:
        return DisjointStorageMeshList(
            [
                SlicedMesh(base, [Slice(*part) for part in rectangle])
                for rectangle in rectangles
            ],
            ids,
        )
    meshes = [
        SlicedMesh(
            base,
            [Slice(orders[tensor_id], orders[tensor_id] + 1), Slice(rank, rank + 1)],
        )
        for tensor_id, rank in slots
    ]
    return DisjointStorageMeshList(meshes, ids)


def rebuild_component(component_desc, world_size):
    """Rebuild saved storage, using manifest slots rather than today's policy."""
    table = DTensorTable(
        [
            _tensor_from_description(info, world_size)
            for info in component_desc["tensors"]
        ]
    )
    ids = [named.global_tensor_id for named in table.named_d_tensors]
    orders = {tensor_id: order for order, tensor_id in enumerate(ids)}
    if len(orders) != len(ids):
        raise ValueError("duplicate tensor id in manifest")
    base = SlotOuterProductMesh(
        ArrangedMesh(TensorCount(len(ids))), ArrangedMesh(WorldSize(world_size))
    )
    entries, paths = {}, {}
    for name, info in component_desc["files"].items():
        _add_layout_file(base, ids, orders, name, info, entries, paths)
    return table, PhysicalLayoutTable(component_desc["format"], entries, paths)


def _add_layout_file(base, ids, orders, name, info, entries, paths):
    _safe_relative(name)
    slots = info["slots"]
    if not slots:
        raise ValueError(f"{name}: file has no slots")
    for tensor_id, _rank in slots:
        if tensor_id not in orders:
            raise ValueError(f"{name}: file slot references missing tensor {tensor_id}")
    group = _layout_file(base, ids, orders, slots, info.get("rectangles"))
    positions = [TensorComponentPos(tensor_id, rank) for tensor_id, rank in slots]
    for pos in positions:
        if pos in entries:
            raise ValueError(f"{pos.tensor_id}: duplicate file slot")
        entries[pos] = group
    leader = min(positions, key=lambda pos: (orders[pos.tensor_id], pos.rank))
    paths[leader] = name


def _safe_relative(name):
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"invalid checkpoint filename {name!r}")


def _completion_marker(root):
    manifest = (root / MANIFEST_NAME).is_file()
    hf = (root / HF_COMMITTED_NAME).is_file()
    if manifest and hf:
        raise ValueError(
            f"ambiguous checkpoint format: {MANIFEST_NAME} and {HF_COMMITTED_NAME}"
        )
    if manifest:
        return MANIFEST_NAME
    if hf:
        return HF_COMMITTED_NAME
    return None


def atomic_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
        entry = os.fstat(file.fileno())
    os.replace(temporary, path)
    return entry.st_dev, entry.st_ino


def json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def write_manifest(path, desc):
    manifest = desc.manifest if isinstance(desc, CheckpointDesc) else desc
    atomic_write(Path(path) / MANIFEST_NAME, json_bytes(manifest))


def read_manifest(path, *, _validate_components=True):
    manifest = json.loads((Path(path) / MANIFEST_NAME).read_bytes())
    if manifest["gdcp_version"] != 1:
        raise ValueError(f"unsupported gdcp_version {manifest['gdcp_version']!r}")
    if type(manifest["world_size"]) is not int or manifest["world_size"] < 1:
        raise ValueError("invalid saved world_size")
    if _validate_components:
        for name, component in manifest["components"].items():
            _safe_relative(name)
            rebuild_component(component, manifest["world_size"])
    for name in manifest["rank_state"]:
        _safe_relative(name)
    return CheckpointDesc(manifest)


def _encode_object(value):
    if isinstance(value, dict):
        return {"dict": [[key, _encode_object(item)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_encode_object(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_encode_object(item) for item in value]}
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"non-JSON optimizer object {type(value).__name__}")


def _decode_object(value):
    if not isinstance(value, dict):
        return value
    if list(value) == ["dict"]:
        return {key: _decode_object(item) for key, item in value["dict"]}
    if list(value) == ["tuple"]:
        return tuple(_decode_object(item) for item in value["tuple"])
    if list(value) == ["list"]:
        return [_decode_object(item) for item in value["list"]]
    raise ValueError("invalid objects.json container")


def write_objects(path, objects):
    atomic_write(
        path, json_bytes({"gdcp_objects_version": 1, "value": _encode_object(objects)})
    )


def read_objects(path):
    document = json.loads(Path(path).read_bytes())
    if document["gdcp_objects_version"] != 1:
        raise ValueError("unsupported objects.json version")
    return _decode_object(document["value"])
