"""Versioned optimizer schemas and lazy buffers, independent of optimizer algorithms."""

from contextlib import nullcontext
from hashlib import sha256
from itertools import combinations, product
from math import prod

import torch
from torch.distributed import (
    ProcessGroup,
    all_gather_object,
    get_backend,
    get_rank,
    new_group,
)
from torch.distributed.tensor import DTensor, Replicate

from gdtensor_conn.coord import NcclCoordinator

SCHEMA_KEY = "_gdcp_optimizer"
SCALARS = (bool, int, float, str, type(None))


def class_name(optimizer):
    cls = type(optimizer)
    return f"{cls.__module__}.{cls.__qualname__}"


def _placement_record(placement):
    # The layout bridge owns the private placement API. Import lazily so plain
    # tensors and legacy checkpoints do not depend on that API.
    from .torch_placements import encode_placements

    return encode_placements([placement])[0]


def _placement_from_record(record):
    from .torch_placements import decode_placements

    return decode_placements([record])[0]


def _dimension_maps(parameter_shape, shape):
    return [
        list(dims)
        for dims in combinations(range(len(parameter_shape)), len(shape))
        if tuple(parameter_shape[d] for d in dims) == tuple(shape)
    ]


def _project_placements(placements, dimensions):
    records = []
    for placement in placements:
        record = _placement_record(placement)
        dim = record.get("dim")
        if dim not in dimensions:
            record = {"kind": "replicate"}
        else:
            record["dim"] = dimensions.index(dim)
        records.append(record)
    return [_placement_from_record(record) for record in records]


def _reshapes_like(tensor, parameter):
    if tensor.numel() != parameter.numel():
        return False
    try:
        viewed = parameter.view(tensor.shape)
    except RuntimeError:
        return False
    assert isinstance(viewed, DTensor)
    return viewed.placements == tensor.placements


def _layout(tensor, parameter):
    if not isinstance(tensor, DTensor):
        return {"kind": "replicate", "plain": True}
    records = [_placement_record(p) for p in tensor.placements]
    own = {
        "kind": "independent",
        "mesh_device_type": tensor.device_mesh.device_type,
        "mesh_shape": list(tensor.device_mesh.mesh.shape),
        "mesh_ranks": tensor.device_mesh.mesh.reshape(-1).tolist(),
        "mesh_dim_names": list(tensor.device_mesh.mesh_dim_names)
        if tensor.device_mesh.mesh_dim_names is not None
        else None,
        "placements": records,
    }
    if not isinstance(parameter, DTensor):
        return (
            {"kind": "replicate"}
            if all(isinstance(p, Replicate) for p in tensor.placements)
            else own
        )
    if tensor.device_mesh != parameter.device_mesh:
        return own
    if tensor.shape == parameter.shape and tensor.placements == parameter.placements:
        return {"kind": "parameter"}
    if _reshapes_like(tensor, parameter):
        return {"kind": "reshape"}
    for dims in _dimension_maps(parameter.shape, tensor.shape):
        if tuple(_project_placements(parameter.placements, dims)) == tensor.placements:
            return {"kind": "project", "dimensions": dims}
    return (
        {"kind": "replicate"}
        if all(isinstance(p, Replicate) for p in tensor.placements)
        else own
    )


def _replicated(shape, dtype, parameter, *, device):
    if not isinstance(parameter, DTensor) or not shape:
        return torch.empty(shape, dtype=dtype, device=device)
    local = torch.empty(shape, dtype=dtype, device=parameter.device)
    return DTensor.from_local(
        local,
        parameter.device_mesh,
        [Replicate()] * parameter.device_mesh.ndim,
        run_check=False,
        shape=torch.Size(shape),
        stride=_stride(shape),
    )


def _stride(shape):
    return tuple(prod(shape[i + 1 :]) for i in range(len(shape)))


def allocate(record, parameter, current=None, ctx=None, mesh_cache=None):
    shape, dtype = record["shape"], getattr(torch, record["dtype"])
    if current is not None and isinstance(current, torch.Tensor):
        return _allocate_current(shape, dtype, current, parameter, mesh_cache)
    device = "cpu" if record.get("device") == "cpu" else parameter.device
    layout = record["layout"]
    kind = layout["kind"]
    if kind == "replicate" and layout.get("plain"):
        return torch.empty(shape, dtype=dtype, device=device)
    if kind == "replicate" or not isinstance(parameter, DTensor):
        return _replicated(shape, dtype, parameter, device=device)
    if kind == "parameter":
        return torch.empty_like(parameter, dtype=dtype)
    if kind == "reshape":
        return torch.empty_like(parameter, dtype=dtype).view(shape)
    if kind == "project":
        dims = layout["dimensions"]
        placements = _project_placements(parameter.placements, dims)
        # distribute_tensor would require a full global allocation on every rank.
        from .torch_placements import local_shape as compute_local_shape

        local_shape = compute_local_shape(shape, parameter.device_mesh, placements)
        return DTensor.from_local(
            torch.empty(local_shape, dtype=dtype, device=parameter.device),
            parameter.device_mesh,
            placements,
            run_check=False,
            shape=torch.Size(shape),
            stride=_stride(shape),
        )
    if kind == "independent":
        from .torch_placements import local_shape as compute_local_shape

        mesh = _independent_mesh(layout, parameter.device_mesh, ctx, mesh_cache)
        placements = [_placement_from_record(p) for p in layout["placements"]]
        local_shape = compute_local_shape(shape, mesh, placements)
        return DTensor.from_local(
            torch.empty(local_shape, dtype=dtype, device=mesh.device_type),
            mesh,
            placements,
            run_check=False,
            shape=torch.Size(shape),
            stride=_stride(shape),
        )
    raise ValueError(f"unsupported optimizer layout recipe {kind!r}")


def _allocate_current(shape, dtype, current, parameter, mesh_cache):
    if list(current.shape) != shape:
        raise ValueError(
            "optimizer destination tensor shape differs from the checkpoint"
        )
    if not isinstance(current, DTensor) or not isinstance(parameter, DTensor):
        return torch.empty_like(current, dtype=dtype)
    mesh = _matching_parameter_mesh(
        current.device_mesh.mesh.reshape(-1).tolist(),
        current.device_mesh.mesh.shape,
        current.device_mesh.mesh_dim_names,
        current.device_mesh.device_type,
        parameter.device_mesh,
        mesh_cache,
    )
    if mesh is None:
        return torch.empty_like(current, dtype=dtype)
    return DTensor.from_local(
        torch.empty_like(current.to_local(), dtype=dtype),
        mesh,
        current.placements,
        run_check=False,
        shape=current.shape,
        stride=current.stride(),
    )


def _independent_mesh(layout, parameter_mesh, ctx, mesh_cache=None):

    groups = layout.get("mesh_groups", [layout["mesh_ranks"]])
    members = (
        parameter_mesh.mesh.reshape(-1).tolist()
        if ctx is None
        else list(ctx.global_ranks or range(ctx.world_size))
    )
    checkpoint_members = layout.get(
        "checkpoint_ranks", sorted(set().union(*map(set, groups)))
    )
    if len(members) != len(checkpoint_members) or any(
        not set(ranks) <= set(checkpoint_members) for ranks in groups
    ):
        raise ValueError(
            "independent optimizer layout needs an initialized destination DTensor when the mesh changes"
        )
    # Rank identities may change while checkpoint-local coordinates stay the
    # same. This mapping is unique and preserves the independent mesh topology.
    rank_map = dict(zip(checkpoint_members, members, strict=True))
    groups = [tuple(rank_map[rank] for rank in ranks) for ranks in groups]
    # A DeviceMesh carries its root layout as well as its visible ranks. In
    # particular, rebuilding a DP axis of DP+TP makes its stride contiguous and
    # DTensor then rejects operations with the parameter. Resolve each parameter
    # before consulting the shared cache: another parameter may have the same
    # ranks but belong to a different root layout.
    for ranks in groups:
        mesh = _matching_parameter_mesh(
            ranks,
            layout["mesh_shape"],
            layout.get("mesh_dim_names"),
            layout.get("mesh_device_type", parameter_mesh.device_type),
            parameter_mesh,
            mesh_cache,
        )
        if mesh is not None:
            return mesh
    # A load owns its cache: process groups must not survive job teardown through
    # a global cache. Every rank traverses the same saved mesh and fiber order.
    key = (
        tuple(map(tuple, groups)),
        tuple(layout["mesh_shape"]),
        tuple(layout.get("mesh_dim_names") or ()),
        layout.get("mesh_device_type", parameter_mesh.device_type),
    )
    cache = {} if mesh_cache is None else mesh_cache
    if key not in cache:
        cache[key] = _independent_mesh_groups(layout, groups, parameter_mesh, ctx)
    return cache[key]


def _matching_parameter_mesh(ranks, shape, names, device_type, parameter_mesh, cache):
    """Reuse the parameter's topology, including named views of its root mesh."""
    root = parameter_mesh._get_root_mesh()
    candidates = [parameter_mesh, root, *root._flatten_mapping.values()]
    # Other parameters/current slots can already hold the requested root slice.
    # Keep those actual objects instead of asking PyTorch for another view.
    if cache is not None:
        candidates.extend(
            mesh
            for mesh in cache.get("parameter_meshes", ())
            if mesh._get_root_mesh() is root
        )
    names = tuple(names) if names is not None else None

    def matches(mesh):
        return (
            mesh.device_type == device_type
            and tuple(mesh.mesh.shape) == tuple(shape)
            and mesh.mesh_dim_names == names
            and mesh.mesh.reshape(-1).tolist() == list(ranks)
        )

    for mesh in candidates:
        if matches(mesh):
            return mesh
    # Slicing reuses the root's layout and communicators; it does not initialize
    # an equivalent standalone DeviceMesh or create any process groups.
    available = set(root.mesh_dim_names or ()) | set(root._flatten_mapping)
    if not names or not set(names) <= available:
        return None
    try:
        mesh = root[names]
    except (KeyError, RuntimeError):
        # An unrelated mesh can reuse names that overlap a flattened axis.
        return None
    if not matches(mesh):
        return None
    if cache is not None:
        cache.setdefault("parameter_meshes", []).append(mesh)
    return mesh


def _independent_mesh_groups(layout, groups, parameter_mesh, ctx):
    coordinator = ctx.coordinator if ctx is not None else None
    device = (
        torch.cuda.device(coordinator.device_index)
        if isinstance(coordinator, NcclCoordinator)
        else nullcontext()
    )
    backend = get_backend(getattr(coordinator, "group", None))
    mesh_device = layout.get("mesh_device_type", parameter_mesh.device_type)
    if mesh_device == "cpu" and "nccl" in backend and "gloo" not in backend:
        backend = "gloo"
    with device:
        meshes = [
            _mesh_from_local_groups(layout, ranks, parameter_mesh.device_type, backend)
            for ranks in groups
        ]
    return next(
        (mesh for mesh in meshes if mesh.get_coordinate() is not None), meshes[0]
    )


def _mesh_from_local_groups(layout, ranks, default_device, backend):
    from torch.distributed.device_mesh import DeviceMesh

    cells = torch.tensor(ranks).reshape(layout["mesh_shape"])
    names = (
        tuple(layout["mesh_dim_names"])
        if layout.get("mesh_dim_names") is not None
        else None
    )
    device_type = layout.get("mesh_device_type", default_device)
    groups = [_mesh_axis_group(cells, axis, backend) for axis in range(cells.ndim)]
    if get_rank() not in ranks:
        return DeviceMesh(device_type, cells, mesh_dim_names=names, _init_backend=False)
    process_groups = [_require_mesh_group(created) for created in groups]
    if cells.ndim == 1:
        return DeviceMesh.from_group(
            process_groups[0], device_type, mesh=cells, mesh_dim_names=names
        )
    mesh = DeviceMesh.from_group(
        process_groups,
        device_type,
        mesh=cells,
        mesh_dim_names=names
        or tuple(f"_gdcp_axis_{axis}" for axis in range(cells.ndim)),
    )
    return _unnamed_mesh(mesh) if names is None else mesh


def _require_mesh_group(created):
    if not isinstance(created, ProcessGroup):
        raise ValueError(
            "independent optimizer mesh is missing a process group for this rank"
        )
    return created


def _unnamed_mesh(mesh):
    # from_group requires names for n-D meshes. Preserve an unnamed source on
    # both the older public attribute and torch 2.15's property-backed spelling.
    fields = vars(mesh)
    fields["_mesh_dim_names" if "_mesh_dim_names" in fields else "mesh_dim_names"] = (
        None
    )
    return mesh


def _mesh_axis_group(cells, axis, backend):
    selected = None
    outer_shape = [size for dim, size in enumerate(cells.shape) if dim != axis]
    for coordinates in product(*(range(size) for size in outer_shape)):
        index: list = list(coordinates)
        index.insert(axis, slice(None))
        ranks = cells[tuple(index)].reshape(-1).tolist()
        created = new_group(
            ranks=ranks,
            backend=backend,
            use_local_synchronization=True,
            sort_ranks=False,
        )
        if get_rank() in ranks:
            selected = created
    return selected


def load_slots(parameters, current, saved, objects, ctx=None):
    slots = {}
    mesh_cache = {
        "parameter_meshes": [
            value.device_mesh
            for value in parameters.values()
            if isinstance(value, DTensor)
        ]
        + [
            value.device_mesh
            for fields in current["state"].values()
            for value in fields.values()
            if isinstance(value, DTensor)
        ]
    }
    for name, fields in saved["fields"].items():
        if name not in parameters:
            raise ValueError(f"optimizer state has no model parameter {name}")
        slots[name] = {
            field: _load_field(
                name, field, record, parameters[name], current, objects, ctx, mesh_cache
            )
            for field, record in fields.items()
        }
    return slots


def _load_field(name, field, record, parameter, current, objects, ctx, mesh_cache):
    if record["kind"] == "tensor":
        return allocate(
            record,
            parameter,
            current["state"].get(name, {}).get(field),
            ctx,
            mesh_cache,
        )
    if record["kind"] == "scalar":
        return objects.get(f"state.{name}.{field}")
    raise ValueError(f"state.{name}.{field}: invalid optimizer field schema")


def verify_plain_replicas(tensors, ctx):
    """Compare ambiguous plain state collectively before a checkpoint can write."""
    reports = {
        key: (
            list(value.shape),
            str(value.dtype),
            sha256(
                memoryview(
                    value.detach()
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                )
            ).hexdigest(),
        )
        for key, value in tensors.items()
    }
    gathered = [reports]
    if ctx.coordinator is not None:
        gathered = [None] * ctx.world_size
        coordinator = ctx.coordinator
        device = (
            torch.cuda.device(coordinator.device_index)
            if isinstance(coordinator, NcclCoordinator)
            else nullcontext()
        )
        with device:
            all_gather_object(gathered, reports, group=coordinator.group)
    keys = set().union(*(report.keys() for report in gathered))
    for key in keys:
        reference = reports.get(key)
        if reference is None or any(
            report.get(key) != reference for report in gathered
        ):
            raise ValueError(
                f"{key}: plain optimizer state differs across ranks; state needs to be a DTensor"
            )
