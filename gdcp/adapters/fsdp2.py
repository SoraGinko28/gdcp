"""Real FSDP2 state dictionaries and per-rank torch serialization."""

import io
from contextlib import ExitStack, nullcontext
from collections import defaultdict
from collections.abc import Callable, Mapping
from functools import cache, partial
from types import MappingProxyType
from math import prod
from typing import cast

import torch
from torch.distributed import (
    ProcessGroup,
    GroupMember,
    all_gather_object,
    get_backend,
    get_rank,
    get_world_size,
    get_process_group_ranks,
    is_initialized,
)
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
)
from torch.distributed.distributed_c10d import group
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import (
    DTensor,
    Replicate as TorchReplicate,
    Shard as TorchShard,
)


from gdtensor.builder import (
    ArrangedMesh,
    DTensorTable,
    Err,
    GlobalTensorMeta,
    NaiveDTensor,
    LiteralMesh,
    ReshapedMesh,
    NamedDTensor,
    Replicate,
    Shard,
    StridedShard,
)
from gdtensor.tag import WorldSize
from gdtensor_conn._geometry import _mesh, owned_intervals
from gdtensor_conn.coord import GlooCoordinator, NcclCoordinator

from .. import _hdfs, core, desc, hf
from ..access import DataAccessorRequest
from gdtensor.access import DTensorDataAccessor
from ..checkpointer import _save_route
from ..state import CheckpointState, Component, RankContext
from .torch_access import TorchAccessor
from .access_checks import owned_digest
from .torch_placements import is_strided, strided_parts
from . import optimizer_state as _optim

_NO_EXTRA_STATE = object()
_DTYPES = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.uint16: "U16",
    torch.uint32: "U32",
    torch.uint64: "U64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}


def _context(process_group=None):
    if process_group == GroupMember.NON_GROUP_MEMBER:
        raise ValueError("current rank is not a member of process_group")
    if process_group is not None and not isinstance(process_group, ProcessGroup):
        raise TypeError("process_group must be a torch.distributed.ProcessGroup")
    if not is_initialized():
        if process_group is not None:
            raise ValueError("process_group requires initialized torch.distributed")
        return RankContext(0, 1)
    backends = get_backend(process_group).lower().split(",")
    if "gloo" in backends or "cpu:gloo" in backends:
        coordinator = GlooCoordinator(process_group)
    elif "nccl" in backends or "cuda:nccl" in backends:
        selected = group.WORLD if process_group is None else process_group
        bound = selected.bound_device_id if selected is not None else None
        device = bound.index if bound is not None else None
        coordinator = NcclCoordinator(
            torch.cuda.current_device() if device is None else device, process_group
        )
    else:
        raise NotImplementedError("process group must provide CPU Gloo or CUDA NCCL")
    members = (
        None if process_group is None else tuple(get_process_group_ranks(process_group))
    )
    return RankContext(
        get_rank(process_group), get_world_size(process_group), coordinator, members
    )


def _unwrap(result):
    if type(result) is Err:
        raise RuntimeError(result.value.msg)
    return result.value


def _collective[T](ctx, action: Callable[[], T]) -> T:
    value, error = None, None
    try:
        value = action()
    except BaseException as exc:
        error = exc
    ok = error is None
    if ctx.coordinator is not None:
        ok = ctx.coordinator.vote(ok)
    if error is not None:
        raise error
    if not ok:
        raise RuntimeError("another rank failed during FSDP2 state preparation")
    return cast(T, value)


def _validate_state(state, *, common_only):
    if not isinstance(state, dict):
        raise TypeError("checkpoint_state must be a dict")
    unknown = state.keys() - {"model", "optimizer", "extra_state"}
    if unknown:
        raise NotImplementedError(f"checkpoint_state key {sorted(unknown)[0]}")
    extra = state.get("extra_state")
    if isinstance(extra, dict) and "loader_ckpt" in extra:
        raise NotImplementedError("loader_ckpt")
    if common_only:
        return
    if "optimizer" in state and "model" not in state:
        raise ValueError("optimizer requires model for parameter names")
    if "model" in state and not isinstance(state["model"], FSDPModule):
        raise TypeError("model must be one FSDP2 fully_shard module")
    if "optimizer" in state and not isinstance(
        state["optimizer"], torch.optim.Optimizer
    ):
        raise TypeError("optimizer must be a torch.optim.Optimizer")


def _ordered_shards(key, tensor):
    # A strided axis chunks the packed index set left-to-right, just as PyTorch
    # does. GDTensor's run geometry handles both this and ordinary shard axes.
    return tensor.device_mesh.mesh, tensor.placements


def _describe_tensor(key, tensor, ctx, replicated_mesh=None):
    if tensor.dtype not in _DTYPES:
        raise NotImplementedError(f"{key}: unsupported real dtype {tensor.dtype}")
    meta = GlobalTensorMeta(
        list(tensor.shape) or [1], _DTYPES[tensor.dtype], tensor.element_size()
    )
    device_mesh = tensor.device_mesh if isinstance(tensor, DTensor) else replicated_mesh
    if device_mesh is None:
        return NaiveDTensor(
            meta, [Replicate()], ArrangedMesh(WorldSize(ctx.world_size))
        )
    if device_mesh.ndim not in (1, 2, 3):
        raise NotImplementedError(
            f"{key}: expected a device mesh with one to three axes"
        )
    ranks, torch_placements = (
        _ordered_shards(key, tensor)
        if isinstance(tensor, DTensor)
        else (device_mesh.mesh, ())
    )
    mesh = _compute_mesh(key, ranks.reshape(-1).tolist(), ranks.shape, ctx)
    if not isinstance(tensor, DTensor):
        # A CPU AdamW step uses the parameter's replica topology for I/O, but
        # the accessor reads and restores the original scalar on its own device.
        return NaiveDTensor(meta, [Replicate() for _ in range(device_mesh.ndim)], mesh)
    placements = [_torch_placement(key, p, tensor.ndim) for p in torch_placements]
    return NaiveDTensor(meta, placements, mesh, torch_d_tensor=tensor)


def _torch_placement(key, placement, ndim):
    if isinstance(placement, TorchReplicate):
        return Replicate()
    if is_strided(placement):
        dim, factor = strided_parts(placement)
        if 0 <= dim < ndim and factor >= 1:
            return StridedShard(dim, factor)
    if type(placement) is TorchShard and 0 <= placement.dim < ndim:
        return Shard(placement.dim)
    raise NotImplementedError(f"{key}: unsupported placement {placement!r}")


def _compute_mesh(key, ranks, shape, ctx):
    expected = ctx.global_ranks or tuple(range(ctx.world_size))
    if len(set(ranks)) != len(ranks) or not set(ranks) <= set(expected):
        raise ValueError(f"{key}: device mesh must match the checkpoint process_group")
    group_ranks = {member: rank for rank, member in enumerate(expected)}
    return _local_mesh(tuple(group_ranks[r] for r in ranks), shape, ctx.world_size)


def _local_mesh(ranks, shape, world_size):
    mesh = (
        ArrangedMesh(WorldSize(world_size))
        if ranks == tuple(range(world_size))
        else LiteralMesh(list(ranks), WorldSize(world_size))
    )
    return ReshapedMesh(mesh, list(shape)) if len(shape) > 1 else mesh


def _submesh_entries(state, world_size):
    @cache
    def subgrid(mesh):
        cells = _mesh(mesh)
        if cells.size < world_size:
            return tuple(cells.ravel().tolist()), cells.shape
        return None

    return {
        (component.name, named.global_tensor_id): (
            named.d_tensor,
            component,
            grid,
        )
        for component in state.components
        for named in component.table.named_d_tensors
        if (grid := subgrid(named.d_tensor.mesh)) is not None
    }


def _submesh_report(entries, rank):
    report = {}
    for key, (_, component, (ranks, _)) in entries.items():
        digest = None
        if rank in ranks:
            digest = owned_digest(component, key[1])
        report[key] = (ranks, digest)
    return report


def _writer_mesh(key, reports, members):
    records = [report[key] for report in reports]
    groups = {ranks for ranks, _ in records}
    first = records[0][0]
    if len(groups) == 1:
        return first
    # Separate submeshes can carry replicas of one checkpoint key. Validate
    # every corresponding shard before allowing one submesh to publish it.
    if sum(map(len, groups)) != len(members) or set().union(*groups) != set(members):
        raise ValueError(f"{key}: parameter meshes must form disjoint groups")
    reference = {}
    for rank, (ranks, digest) in zip(members, records, strict=True):
        if rank not in ranks or len(ranks) != len(first):
            raise ValueError(f"{key}: inconsistent parameter mesh membership")
        coordinate = ranks.index(rank)
        if reference.setdefault(coordinate, digest) != digest:
            raise ValueError(
                f"{key}: same checkpoint key has different values on its parameter meshes"
            )
    return first


def _select_submesh_writers(state, ctx):
    if ctx.world_size == 1 or ctx.coordinator is None:
        return
    entries = _collective(ctx, lambda: _submesh_entries(state, ctx.world_size))
    if ctx.coordinator.vote(not entries):
        return
    local = _collective(ctx, lambda: _submesh_report(entries, ctx.rank))
    reports = [{} for _ in range(ctx.world_size)]
    coordinator = ctx.coordinator
    device = (
        torch.cuda.device(coordinator.device_index)
        if isinstance(coordinator, NcclCoordinator)
        else nullcontext()
    )
    with device:
        all_gather_object(reports, local, group=coordinator.group)
    if any(report.keys() != local.keys() for report in reports):
        raise ValueError("ranks disagree on parameter mesh checkpoint keys")
    members = range(ctx.world_size)
    for key, (layout, _, (_, shape)) in entries.items():
        _optimizer_submesh_metadata(state, key, reports, ctx)
        layout.mesh = _local_mesh(
            _writer_mesh(key, reports, members), shape, ctx.world_size
        )


def _optimizer_submesh_metadata(state, key, reports, ctx):
    component_name, tensor_id = key
    if component_name != "optimizer":
        return
    component = next(
        component for component in state.components if component.name == component_name
    )
    schema = component.objects.get(_optim.SCHEMA_KEY)
    if schema is None:
        return
    fields = schema["fields"]
    names = sorted(fields, key=len, reverse=True)
    name = next(name for name in names if tensor_id.startswith(f"state.{name}."))
    field = tensor_id[len(name) + 7 :]
    record = fields[name][field]["layout"]
    if record["kind"] == "independent":
        record["checkpoint_ranks"] = list(ctx.global_ranks or range(ctx.world_size))
        record["mesh_groups"] = sorted(
            {
                tuple(
                    (ctx.global_ranks or tuple(range(ctx.world_size)))[rank]
                    for rank in ranks
                )
                for report in reports
                for ranks, _ in [report[key]]
            }
        )


def _component(
    name,
    tensors,
    objects,
    ctx,
    scalar_meshes=None,
    *,
    data_accessor=None,
    loading=False,
):
    scalar_meshes = scalar_meshes or {}
    aliases = objects.get("aliases", {}) if name == "model" else {}
    entries = [
        NamedDTensor(
            key,
            _describe_tensor(key, tensor, ctx, scalar_meshes.get(key)),
            [alias for alias, canonical in aliases.items() if canonical == key],
        )
        for key, tensor in tensors.items()
    ]
    intervals = {
        named.global_tensor_id: owned_intervals(named, "compute", ctx.rank)
        for named in entries
        if isinstance(tensors[named.global_tensor_id], DTensor)
    }
    mapping = TorchAccessor(tensors, {}, intervals)
    ownership = {
        named.global_tensor_id: tuple(
            (
                int(start) * named.d_tensor.global_tensor_meta.itemsize,
                int(end) * named.d_tensor.global_tensor_meta.itemsize,
            )
            for start, end in owned_intervals(named, "compute", ctx.rank)
        )
        for named in entries
    }
    request = DataAccessorRequest(
        name,
        "load" if loading else "save",
        MappingProxyType(dict(tensors)),
        MappingProxyType(ownership),
        mapping.local_offset,
    )
    if data_accessor is None:
        for key, tensor in tensors.items():
            if tensor.device.type == "meta":
                raise ValueError(
                    f"{key}: tensor has no readable storage; a data_accessor is required"
                )
        accessor = mapping
    else:
        accessor = data_accessor(request)
        if not isinstance(accessor, DTensorDataAccessor):
            raise TypeError("data_accessor must return a DTensorDataAccessor")
    component = Component(name, DTensorTable(entries), accessor, objects)
    component.data_request = request

    component.logical_shapes = {
        key: [] for key, tensor in tensors.items() if tensor.ndim == 0
    }
    return component


def _optimizer_values(state):
    tensors, objects, paths = {}, {"param_groups": state["param_groups"]}, {}
    if _optim.SCHEMA_KEY in state:
        objects[_optim.SCHEMA_KEY] = state[_optim.SCHEMA_KEY]
    for fqn, fields in state["state"].items():
        _optimizer_fields(fqn, fields, tensors, objects, paths)
    return tensors, objects, paths


def _optimizer_fields(fqn, fields, tensors, objects, paths):
    for field, value in fields.items():
        key = f"state.{fqn}.{field}"
        if key in paths:
            raise ValueError(
                f"optimizer state key collision for {key!r}: {paths[key]!r} and {(fqn, field)!r}"
            )
        paths[key] = fqn, field
        target = tensors if isinstance(value, torch.Tensor) else objects
        target[key] = value


def _model_aliases(model):
    canonical, aliases = {}, {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        first = canonical.setdefault(id(parameter), name)
        if name != first:
            aliases[name] = first
    return aliases


def _model_values(model):
    state = model.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise NotImplementedError("model state_dict contains non-tensor values")
    aliases = _model_aliases(model)
    return {key: tensor for key, tensor in state.items() if key not in aliases}


def _optimizer_tensor(fqn, field, tensor, parameter):
    if isinstance(tensor, DTensor):
        if tensor.ndim == 0 and not all(
            isinstance(p, TorchReplicate) for p in tensor.placements
        ):
            raise ValueError(f"state.{fqn}.{field}: scalar state must be replicated")
        return tensor
    if tensor.ndim == 0:
        return tensor
    if isinstance(parameter, DTensor) and tensor.shape == parameter.to_local().shape:
        return DTensor.from_local(
            tensor,
            parameter.device_mesh,
            parameter.placements,
            run_check=False,
            shape=parameter.shape,
            stride=parameter.stride(),
        )
    return tensor


def _validate_optimizer_tensors(model, state, ctx=None):
    parameters = dict(model.named_parameters())
    for fqn, fields in state["state"].items():
        if fqn not in parameters:
            raise ValueError(f"optimizer state has no model parameter {fqn}")
        state["state"][fqn] = _validate_optimizer_fields(fqn, fields, parameters[fqn])


def _validate_optimizer_fields(fqn, fields, parameter):
    if not isinstance(fields, Mapping):
        raise NotImplementedError(
            f"state.{fqn}: nested optimizer state; expected a flat mapping"
        )
    fields = dict(fields)
    if len({str(field) for field in fields}) != len(fields):
        raise ValueError(f"state.{fqn}: optimizer state key collision")
    for field, value in fields.items():
        if not isinstance(field, str):
            raise TypeError(
                f"state.{fqn}.{field}: optimizer field name must be a string"
            )
        if isinstance(value, torch.Tensor):
            fields[field] = _optimizer_tensor(fqn, field, value, parameter)
        elif not isinstance(value, _optim.SCALARS):
            raise NotImplementedError(f"state.{fqn}.{field}: nested optimizer state")
    return fields


def _restore_objects(state, paths, objects):
    state["param_groups"] = objects["param_groups"]
    for key, value in objects.items():
        if key not in {"param_groups", _optim.SCHEMA_KEY}:
            fqn, field = paths[key]
            state["state"][fqn][field] = value


def _validate_adamw_step(fqn, fields):
    step = fields.get("step")
    if not fields or isinstance(step, torch.Tensor):
        return
    try:
        float(step)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"optimizer state for {fqn} has an invalid step") from exc


def _prepare_optimizer_state(model, optimizer, state, objects, paths):
    """Prepare FSDP2's FQN state without casting or duplicating its tensors.

    Unlike FSDP1, FSDP2 retains the original Parameter objects, so the FQN map
    directly identifies optimizer slots. The normal optimizer loader would
    allocate bf16 copies of every fp32 master and moment before installing
    them, both losing precision and exceeding large-checkpoint memory bounds.
    """
    expected = {"param_groups"} | {
        key
        for key, (fqn, field) in paths.items()
        if not isinstance(state["state"][fqn][field], torch.Tensor)
    }
    if _optim.SCHEMA_KEY in state:
        expected.add(_optim.SCHEMA_KEY)
    if objects.keys() != expected:
        key = sorted(objects.keys() ^ expected)[0]
        raise ValueError(f"optimizer: objects key mismatch for {key}")
    for key in expected - {"param_groups", _optim.SCHEMA_KEY}:
        if not isinstance(objects[key], (bool, int, float, str, type(None))):
            raise ValueError(f"optimizer: unsupported non-tensor object {key}")

    parameters = dict(model.named_parameters())
    groups = _optimizer_groups(
        objects["param_groups"], state["param_groups"], parameters
    )
    _restore_objects(state, paths, objects)
    return _optimizer_installation(optimizer, state, groups, parameters)


def _optimizer_installation(optimizer, state, groups, parameters):
    if type(optimizer).__setstate__ is torch.optim.AdamW.__setstate__:
        for fqn, fields in state["state"].items():
            _validate_adamw_step(fqn, fields)
    slots = defaultdict(
        dict,
        {parameters[fqn]: dict(fields) for fqn, fields in state["state"].items()},
    )
    return {"state": slots, "param_groups": groups}


def _optimizer_group_key(group):
    if not isinstance(group, dict) or not isinstance(group.get("params"), list):
        raise ValueError("optimizer parameter group must contain a params list")
    names = group["params"]
    if any(not isinstance(name, str) for name in names):
        raise ValueError("optimizer parameter group must contain parameter names")
    return tuple(sorted(names))


def _validate_optimizer_groups(saved, current):
    if not isinstance(saved, list):
        raise ValueError("optimizer param_groups must be a list")
    if len(saved) != len(current):
        raise ValueError("optimizer parameter group count does not match")
    saved_keys = {_optimizer_group_key(group) for group in saved}
    current_keys = {_optimizer_group_key(group) for group in current}
    if len(saved_keys) != len(saved) or len(current_keys) != len(current):
        raise ValueError(
            "optimizer parameter group matching is ambiguous: duplicate membership"
        )
    if saved_keys != current_keys:
        raise ValueError("optimizer parameter group members do not match")


def _optimizer_groups(saved, current, parameters):
    _validate_optimizer_groups(saved, current)
    return [
        dict(
            group,
            params=[parameters[name] for name in group["params"]],
            param_names=list(group["params"]),
        )
        for group in saved
    ]


def _named_optimizer_groups(optimizer, names):
    return [
        dict(group, params=[names[parameter] for parameter in group["params"]])
        for group in optimizer.param_groups
    ]


def _install_optimizer_state(model, optimizer, state):
    """Install a state dictionary read by PyTorch's native DCP loader."""
    parameters = dict(model.named_parameters())
    names = {parameter: name for name, parameter in parameters.items()}
    current = _named_optimizer_groups(optimizer, names)
    groups = _optimizer_groups(state["param_groups"], current, parameters)
    _set_optimizer_state(
        optimizer, _optimizer_installation(optimizer, state, groups, parameters)
    )


def _set_optimizer_state(optimizer, state):
    # __setstate__ initializes optimizer internals and may impose class defaults
    # (AdamW forces decoupled_weight_decay). Saved group options are authoritative.
    groups = [dict(settings) for settings in state["param_groups"]]
    optimizer.__setstate__(state)
    optimizer.param_groups = groups


def _optimizer_field_schema(key, value, parameter, ctx):
    if not isinstance(value, torch.Tensor):
        return {"kind": "scalar"}
    try:
        layout = _optim._layout(value, parameter)
    except NotImplementedError as exc:
        raise NotImplementedError(f"{key}: {exc}") from exc
    if layout["kind"] == "independent":
        layout["checkpoint_ranks"] = list(ctx.global_ranks or range(ctx.world_size))
    return {
        "kind": "tensor",
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        # A distributed state's restoration device follows its mesh. Local
        # storage may be supplied by an accessor independently of that mesh.
        "device": value.device_mesh.device_type
        if isinstance(value, DTensor)
        else value.device.type,
        "layout": layout,
    }


def _optimizer_schema(model, state, optimizer_class, ctx):
    parameters = dict(model.named_parameters())
    fields = {
        name: {
            str(field): _optimizer_field_schema(
                f"state.{name}.{field}", value, parameters[name], ctx
            )
            for field, value in values.items()
        }
        for name, values in state["state"].items()
    }
    return {"version": 1, "class": optimizer_class, "fields": fields}


def _expected_optimizer_fields(optimizer, group):
    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        fields = {"step", "exp_avg", "exp_avg_sq"}
        if group.get("amsgrad"):
            fields.add("max_exp_avg_sq")
        return fields
    if isinstance(optimizer, torch.optim.SGD):
        return {"momentum_buffer"} if group.get("momentum", 0) else set()
    if isinstance(optimizer, torch.optim.Adagrad):
        return {"step", "sum"}
    if isinstance(optimizer, torch.optim.RMSprop):
        fields = {"step", "square_avg"}
        if group.get("momentum", 0):
            fields.add("momentum_buffer")
        if group.get("centered"):
            fields.add("grad_avg")
        return fields
    return None


def _validate_optimizer_class(optimizer, current, groups, schema):
    if schema.get("version") != 1 or not isinstance(schema.get("fields"), dict):
        raise ValueError("invalid optimizer layout metadata")
    saved_class, target_class = schema.get("class"), _optim.class_name(optimizer)
    if saved_class == target_class:
        return
    targets = {_optimizer_group_key(g): g for g in current["param_groups"]}
    compatible = all(
        _optimizer_class_group_matches(
            optimizer,
            current,
            saved_group,
            targets[_optimizer_group_key(saved_group)],
            schema,
        )
        for saved_group in groups
    )
    if not compatible:
        raise ValueError(
            f"optimizer class mismatch: saved {saved_class}, target {target_class}; field sets or group option keys differ"
        )


def _optimizer_class_group_matches(optimizer, current, saved, target, schema):
    if saved.keys() - {"param_names"} != target.keys() - {"param_names"}:
        return False
    expected = _expected_optimizer_fields(optimizer, target)
    for name in saved["params"]:
        target_fields = current["state"].get(name)
        saved_fields = set(schema["fields"].get(name, {}))
        wanted = (
            set(target_fields)
            if target_fields
            else (expected if saved_fields else set())
        )
        if wanted is None or saved_fields != wanted:
            return False
    return True


def _native_optimizer_schema(
    parameters, groups, field_keys, tensors, objects, optimizer, *, saved_class=None
):
    # Native DCP has tensor metadata but no GDCP layout recipes. A native
    # parameter-shaped slot follows its parameter; differently shaped native
    # slots need an existing destination DTensor to prescribe a new layout.
    names = sorted(parameters, key=len, reverse=True)
    fields = {}
    for key in field_keys:
        if not key.startswith("state."):
            continue
        name = next((name for name in names if key.startswith(f"state.{name}.")), None)
        if name is None:
            raise ValueError(f"optimizer state has no model parameter for {key}")
        field = key[len(name) + 7 :]
        record = (
            _native_optimizer_tensor_schema(tensors[key], parameters[name])
            if key in tensors
            else {"kind": "scalar"}
        )
        fields.setdefault(name, {})[field] = record
    if saved_class is not None:
        return {"version": 1, "class": saved_class, "fields": fields}
    for saved_group in groups:
        _validate_native_optimizer_fields(optimizer, saved_group, fields)
    # Native checkpoints do not record a class. Group option keys still catch
    # SGD/Adam incompatibility before any destination buffers are touched.
    target_groups = optimizer.param_groups
    if (
        groups
        and target_groups
        and groups[0].keys() - {"params", "param_names"}
        and groups[0].keys() - {"params", "param_names"}
        != target_groups[0].keys() - {"params", "param_names"}
    ):
        raise ValueError(
            f"optimizer class mismatch: saved native optimizer (group keys {sorted(groups[0].keys())}), target {_optim.class_name(optimizer)}"
        )
    return {"version": 1, "class": _optim.class_name(optimizer), "fields": fields}


def _native_optimizer_tensor_schema(info, parameter):
    shape = info.get("logical_shape", info["shape"])
    dtype = info["dtype"]
    dtype = next(
        (str(t).removeprefix("torch.") for t, code in _DTYPES.items() if code == dtype),
        dtype.removeprefix("torch."),
    )
    layout = (
        {"kind": "parameter"}
        if list(parameter.shape) == shape
        else {"kind": "replicate"}
    )
    if shape and prod(shape) == parameter.numel():
        layout = {"kind": "reshape"}
    return {
        "kind": "tensor",
        "shape": shape,
        "dtype": dtype,
        "layout": layout,
        "device": "cpu" if not shape else parameter.device.type,
    }


def _validate_native_optimizer_fields(optimizer, group, fields):
    expected = _expected_optimizer_fields(optimizer, group)
    if expected is None:
        return
    for name in group["params"]:
        missing = expected - fields[name].keys() if name in fields else set()
        if missing:
            raise ValueError(
                f"native DCP missing keys: state.{name}.{sorted(missing)[0]}"
            )


def _validate_native_current_fields(current, field_keys, saved_fields):
    # Entirely absent parameter state is a valid lazy checkpoint and replaces
    # any warmed destination slots. A saved parameter must still contain every
    # destination field, including custom fields unknown to the optimizer class.
    wanted = {
        f"state.{name}.{field}"
        for name, fields in current["state"].items()
        if name in saved_fields
        for field in fields
    }
    missing = wanted - field_keys
    if missing:
        raise ValueError(f"native DCP missing keys: {sorted(missing)[0]}")


def _fresh_adamw_slots(parameter, group, *, amsgrad):
    scalar_dtype = torch.get_default_dtype()
    if group["fused"] or scalar_dtype != torch.float64:
        scalar_dtype = torch.float32
    device = parameter.device if group["capturable"] or group["fused"] else "cpu"
    slots = {
        "step": torch.empty((), dtype=scalar_dtype, device=device),
        "exp_avg": torch.empty_like(parameter),
        "exp_avg_sq": torch.empty_like(parameter),
    }
    if amsgrad:
        slots["max_exp_avg_sq"] = torch.empty_like(parameter)
    return slots


def _load_adamw_slots(parameters, target, groups, field_keys):
    current = target["state"]
    target_groups = {
        _optimizer_group_key(group): group for group in target["param_groups"]
    }
    # Only a parameter with saved state should acquire AdamW slots. Keep
    # all expected fields on active parameters so incomplete records fail.
    active = {
        key[6:].rsplit(".", 1)[0] for key in field_keys if key.startswith("state.")
    }
    # Subclasses can have additional field names containing dots. The
    # live schema supplies their exact parameter/field boundary.
    active.update(
        name
        for name, fields in current.items()
        if any(f"state.{name}.{field}" in field_keys for field in fields)
    )
    slots = {}
    for saved_group in groups:
        settings = {
            **target_groups[_optimizer_group_key(saved_group)],
            **saved_group,
        }
        for name in (p for p in saved_group["params"] if p in active):
            fields = current.get(name)
            slots[name] = (
                dict(fields)
                if fields
                else _fresh_adamw_slots(
                    parameters[name],
                    settings,
                    amsgrad=saved_group.get(
                        "amsgrad", f"state.{name}.max_exp_avg_sq" in field_keys
                    ),
                )
            )
    return slots


class FSDP2Adapter:
    def __init__(self, *, loading=False, common_only=False, data_accessor=None):
        self.loading, self.common_only = loading, common_only
        self.data_accessor = data_accessor
        self.model_state: dict | None = None
        self.optimizer_state: dict | None = None
        self.optimizer_paths = {}
        self.model_aliases = {}
        self.loaded_optimizer = None
        self.loaded_model = None
        self.loaded_extra = _NO_EXTRA_STATE

    def describe(self, checkpoint_state, ctx):
        _validate_state(checkpoint_state, common_only=self.common_only)
        state = CheckpointState()
        if not self.common_only:
            self._components(checkpoint_state, state, ctx)
        if "extra_state" in checkpoint_state or self.common_only:
            state.rank_state["extra_state"] = self._extra(checkpoint_state)
        return state

    def _components(self, checkpoint_state, state, ctx):
        if "model" in checkpoint_state:
            self.model_aliases = _model_aliases(checkpoint_state["model"])
            self.model_state = _model_values(checkpoint_state["model"])
            objects = {"aliases": self.model_aliases} if self.model_aliases else {}
            state.components.append(
                _component(
                    "model",
                    self.model_state,
                    objects,
                    ctx,
                    data_accessor=self.data_accessor,
                    loading=self.loading,
                )
            )
        if "optimizer" in checkpoint_state:
            self._optimizer(checkpoint_state, state, ctx)
        if self.loading:
            state.components.sort(key=lambda component: component.name != "optimizer")

    def _optimizer(self, checkpoint_state, state, ctx):
        model, optimizer = checkpoint_state["model"], checkpoint_state["optimizer"]
        self.optimizer_class = _optim.class_name(optimizer)
        if optimizer.state:
            values = cast(dict, get_optimizer_state_dict(model, optimizer))
            values = {
                **values,
                "state": {
                    name: dict(fields) for name, fields in values["state"].items()
                },
            }
        else:
            # PyTorch materializes an empty optimizer by calling step(). A
            # checkpoint must preserve lazy state and never advance training.
            names = {p: name for name, p in model.named_parameters()}
            values = {
                "state": {},
                "param_groups": _named_optimizer_groups(optimizer, names),
            }
        self.optimizer_state = values
        if self.loading:
            _validate_optimizer_tensors(model, values)
        # Load buffers depend on which parameters had state when saved.
        component = (
            _component("optimizer", {}, {}, ctx)
            if self.loading
            else self._optimizer_component(model, ctx)
        )
        state.components.append(component)

    def prepare_optimizer(
        self,
        checkpoint_state,
        state,
        ctx,
        groups,
        field_keys,
        objects=None,
        tensor_schema=None,
    ):
        assert self.optimizer_state is not None
        _validate_optimizer_groups(groups, self.optimizer_state["param_groups"])
        model = checkpoint_state["model"]
        parameters = dict(model.named_parameters())
        objects = objects or {}
        schema = objects.get(_optim.SCHEMA_KEY)
        optimizer = checkpoint_state["optimizer"]
        if (
            schema is None
            and tensor_schema is not None
            and tensor_schema.get("__native__")
        ):
            schema = _native_optimizer_schema(
                parameters, groups, field_keys, tensor_schema, objects, optimizer
            )
            _validate_native_current_fields(
                self.optimizer_state, field_keys, schema["fields"]
            )
        if (
            schema is None
            and tensor_schema is not None
            and not isinstance(optimizer, torch.optim.AdamW)
        ):
            schema = _native_optimizer_schema(
                parameters,
                groups,
                field_keys,
                tensor_schema,
                objects,
                optimizer,
                saved_class="torch.optim.adamw.AdamW",
            )
        if schema is not None:
            _validate_optimizer_class(
                checkpoint_state["optimizer"], self.optimizer_state, groups, schema
            )
            slots = _optim.load_slots(
                parameters, self.optimizer_state, schema, objects, ctx
            )
        elif isinstance(optimizer, torch.optim.AdamW):
            slots = _load_adamw_slots(
                parameters, self.optimizer_state, groups, field_keys
            )
        else:
            raise ValueError(
                "optimizer class mismatch: saved torch.optim.AdamW (legacy), "
                f"target {_optim.class_name(optimizer)}; legacy checkpoint lacks a generic field schema"
            )
        self.optimizer_state = {"param_groups": groups, "state": slots}
        if _optim.SCHEMA_KEY in objects:
            self.optimizer_state[_optim.SCHEMA_KEY] = schema
        component = self._optimizer_component(model, ctx)
        # Loading always orders the optimizer first.
        state.components[0] = component
        return component

    def _optimizer_component(self, model, ctx):
        assert self.optimizer_state is not None
        _validate_optimizer_tensors(model, self.optimizer_state, ctx)
        if not self.loading:
            self.optimizer_state[_optim.SCHEMA_KEY] = _optimizer_schema(
                model, self.optimizer_state, self.optimizer_class, ctx
            )
        tensors, objects, self.optimizer_paths = _optimizer_values(self.optimizer_state)
        parameters = dict(model.named_parameters())
        scalar_meshes = {
            key: p.device_mesh
            for key, value in tensors.items()
            if value.ndim == 0
            and isinstance(p := parameters[self.optimizer_paths[key][0]], DTensor)
            and p.device_mesh.size() < ctx.world_size
        }
        component = _component(
            "optimizer",
            tensors,
            objects,
            ctx,
            scalar_meshes,
            data_accessor=self.data_accessor,
            loading=self.loading,
        )
        return component

    def _extra(self, checkpoint_state):
        if self.loading:
            return b""
        buffer = io.BytesIO()
        torch.save(checkpoint_state["extra_state"], buffer)
        return buffer.getvalue()

    def prepare_restore(self, checkpoint_state, objects):
        if self.model_state is not None:
            self._prepare_model(objects.get("model", {}))
        if self.optimizer_state is not None:
            self.loaded_optimizer = _prepare_optimizer_state(
                checkpoint_state["model"],
                checkpoint_state["optimizer"],
                self.optimizer_state,
                objects.get("optimizer", {}),
                self.optimizer_paths,
            )

    def prepare_rank_state(self, blobs):
        if "extra_state" in blobs:
            self.loaded_extra = torch.load(
                io.BytesIO(blobs["extra_state"]),
                map_location="cpu",
                weights_only=False,
            )

    def _prepare_model(self, objects):
        assert self.model_state is not None
        complete = dict(self.model_state)
        if objects.keys() - {"aliases"}:
            raise ValueError("model: unexpected checkpoint objects")
        aliases = objects.get("aliases", self.model_aliases)
        if aliases != self.model_aliases:
            raise ValueError("model tied parameter aliases do not match the checkpoint")
        complete.update({alias: complete[name] for alias, name in aliases.items()})
        self.loaded_model = complete

    def restore(self, checkpoint_state, _loaded):
        if self.loaded_optimizer is not None:
            _set_optimizer_state(checkpoint_state["optimizer"], self.loaded_optimizer)
        if self.loaded_model is not None:
            checkpoint_state["model"].load_state_dict(self.loaded_model, strict=True)
        if self.loaded_extra is not _NO_EXTRA_STATE:
            checkpoint_state["extra_state"] = self.loaded_extra


def _finish(state):
    for component in state.components:
        finish = getattr(component.accessor, "finish", None)
        if finish is not None:
            finish()


def save_checkpoint(
    path,
    checkpoint_state,
    global_steps=None,
    *,
    checkpoint_format="safetensors",
    hf_dtype=None,
    max_shard_size=None,
    process_group=None,
    data_accessor=None,
    _preflight=None,
):
    ctx = _context(process_group)

    def prepare():
        if data_accessor is not None and hf_dtype is not None:
            raise ValueError(
                "data_accessor cannot be combined with hf_dtype or fp32 master export"
            )
        if data_accessor is not None and not callable(data_accessor):
            raise TypeError("data_accessor must be a factory")
        if _preflight is not None:
            return _preflight()
        return path, _save_route(checkpoint_format, hf_dtype, max_shard_size)

    root, route = _collective(ctx, prepare)

    def options():
        return {
            **_save_agreement_options(route, max_shard_size, hf_dtype),
            "custom data accessor": data_accessor is not None,
        }

    def describe():
        if route == "HF weights":
            components, keys = ("model",), ()
        else:
            state_keys = checkpoint_state if isinstance(checkpoint_state, dict) else {}
            components = tuple(
                name for name in ("model", "optimizer") if name in state_keys
            )
            keys = ("extra_state",) if "extra_state" in state_keys else ()
        return core._save_request(root, ctx, components, keys, options)

    snapshot, components, keys = _unwrap(core._agree(ctx, describe))

    def materialize():
        if isinstance(root, _hdfs.HdfsPath):
            _hdfs.require_client()
        state, metadata = _prepare_save_state(
            checkpoint_state, snapshot, route, hf_dtype, data_accessor
        )
        state.global_steps = global_steps
        rank_state = dict(state.rank_state)
        core._validate_state(state, rank_state=rank_state)
        if (
            core._component_names(state) != components
            or tuple(sorted(rank_state)) != keys
        ):
            raise ValueError(
                "checkpoint components or rank_state keys changed during state preparation"
            )
        return core._SaveAgreement(
            root, state, ctx, snapshot, rank_state, components
        ), metadata

    agreement, write_metadata = _collective(ctx, materialize)
    _verify_optimizer_replicas(agreement.state, snapshot)
    if components and snapshot.coordinator is not None:
        _select_submesh_writers(agreement.state, snapshot)
        _unwrap(
            core._agree(
                snapshot,
                lambda: (
                    None,
                    {"ordered tensor schema": core._tensor_schema(agreement.state)},
                ),
            )
        )
    if route == "safetensors":
        result = core._save_agreed(root, agreement)
    elif route == "dcp":
        from . import dcp

        result = dcp._save_agreed(root, agreement)
    else:
        result = hf._save_agreed(root, agreement, max_shard_size, write_metadata)
    _unwrap(result)
    if route != "HF weights":
        _collective(ctx, lambda: _finish(agreement.state))


def _verify_optimizer_replicas(state, ctx):
    for component in state.components:
        if component.name != "optimizer":
            continue
        assert component.data_request is not None
        pending = {
            key: value
            for key, value in component.data_request.tensors.items()
            if value.ndim and not isinstance(value, DTensor)
        }
        empty = not pending
        if ctx.coordinator is not None:
            empty = ctx.coordinator.vote(empty)
        if not empty:
            _collective(
                ctx, lambda: _optim.verify_plain_replicas(pending, ctx, component)
            )


def _prepare_save_state(checkpoint_state, ctx, route, dtype, data_accessor=None):
    if route == "HF weights":
        from .hf import _prepare_save

        return _prepare_save(checkpoint_state, dtype, ctx)
    return FSDP2Adapter(data_accessor=data_accessor).describe(
        checkpoint_state, ctx
    ), None


def _save_agreement_options(route, max_shard_size, dtype):
    if route in ("hf", "HF weights"):
        options = hf._save_options(max_shard_size, dtype)
    elif route == "dcp":
        from .dcp import DcpBackend

        options = {"format": DcpBackend.format}
    else:
        options = {"format": "safetensors"}
    return {"dispatch": route, **options}


def _load_format(root, state):
    core._require_not_pending(root)
    marker = desc._completion_marker(root)
    if marker == desc.MANIFEST_NAME:
        if not state.components:
            return "safetensors"
        saved = desc.read_manifest(root, _validate_components=False)
        formats = {item["format"] for item in saved.components.values()}
        return "dcp" if "distcp" in formats else "safetensors"
    if marker == desc.HF_COMMITTED_NAME:
        return "hf"
    native_hf = hf._has_native_files(root)
    native_dcp = (root / ".metadata").is_file() or any(
        (root / name / ".metadata").is_file() for name in ("model", "optimizer")
    )
    if native_hf and native_dcp:
        raise ValueError("ambiguous checkpoint format: native HF and DCP files")
    if native_hf:
        return "hf"
    if native_dcp:
        return "dcp"
    return None


def _latest_step(root):
    for entry in desc.numbered_generations(root.iterdir()):
        if entry.is_symlink() or (entry / desc.PENDING_NAME).exists():
            continue
        if desc._completion_marker(entry) is not None:
            return entry
    raise ValueError(f"no committed global_step_N checkpoint in {root}")


def _load_agreement_options(
    path, state, preflight, validate_groups, common_only, custom_accessor=False
):
    fmt = _load_format(path, state)
    options = {"custom data accessor": custom_accessor}
    if fmt is None and path.is_dir() and not desc.STEP_DIRECTORY.fullmatch(path.name):
        path = _latest_step(path)
        options["generation"] = path.name
        fmt = _load_format(path, state)
    if fmt is None:
        raise ValueError(
            f"missing checkpoint completion marker or native format files in {path}"
        )
    if fmt == "dcp":
        from .dcp import _load_storage_format

        return core._load_options(
            _load_storage_format(path),
            preflight,
            validate_groups,
            dispatch=fmt,
            **options,
        )
    return core._load_options(
        fmt, preflight, validate_groups, optional=common_only, dispatch=fmt, **options
    )


def load_checkpoint(
    path,
    checkpoint_state,
    common_only=False,
    *,
    process_group=None,
    data_accessor=None,
    _preflight=None,
):
    ctx = _context(process_group)
    adapter = FSDP2Adapter(
        loading=True, common_only=common_only, data_accessor=data_accessor
    )

    def prepare():
        if _preflight is not None:
            _preflight()
        return _hdfs.checkpoint_path(path), adapter.describe(checkpoint_state, ctx)

    root, state = _collective(ctx, prepare)

    def preflight(objects):
        adapter.prepare_restore(checkpoint_state, objects)
        return adapter.prepare_rank_state

    optimizer = adapter.optimizer_state
    validate_groups = (
        partial(adapter.prepare_optimizer, checkpoint_state, state, ctx)
        if optimizer is not None
        else None
    )
    with ExitStack() as stack:
        (local, options) = _unwrap(
            core._read_agreement(
                stack,
                root,
                state,
                ctx,
                lambda local: _load_agreement_options(
                    local,
                    state,
                    preflight,
                    validate_groups,
                    common_only,
                    data_accessor is not None,
                ),
                latest=True,
            )
        )
        path = local / options.get("generation", "")
        fmt = options["dispatch"]
        if fmt == "hf":
            result = hf._load_agreed(
                path,
                state,
                ctx,
                preflight=preflight,
                validate_groups=validate_groups,
                optional_rank_state=common_only,
            )
        elif fmt == "dcp":
            from . import dcp

            result = dcp._load_agreed(
                path,
                state,
                ctx,
                options["format"],
                preflight=preflight,
                validate_groups=validate_groups,
            )
        else:
            result = core._load_agreed(
                path,
                state,
                ctx,
                preflight=preflight,
                validate_groups=validate_groups,
                optional_rank_state=common_only,
            )
    if type(result) is Err and isinstance(result.value, core._LoadValidationError):
        raise result.value.cause
    _unwrap(result)
    _collective(ctx, lambda: _finish(state))
    _collective(ctx, lambda: adapter.restore(checkpoint_state, state))
