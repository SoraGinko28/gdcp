"""Real FSDP2 state dictionaries and per-rank torch serialization."""

import io
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

import torch
from torch.distributed import get_backend, get_rank, get_world_size, is_initialized
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
)
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
    NamedDTensor,
    Replicate,
    Shard,
)
from gdtensor.tag import WorldSize
from gdtensor_conn.coord import GlooCoordinator

from .. import core, desc, hf
from ..checkpointer import _SAVE_FORMATS, _save_route
from ..state import CheckpointState, Component, RankContext
from .torch_access import TorchAccessor

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


def _context():
    if not is_initialized():
        return RankContext(0, 1)
    backends = get_backend().lower().split(",")
    if "gloo" not in backends and "cpu:gloo" not in backends:
        raise NotImplementedError(
            "default process group must provide Gloo for CPU tensors "
            "(use gloo or cuda:nccl,cpu:gloo)"
        )
    return RankContext(get_rank(), get_world_size(), GlooCoordinator())


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
    if "optimizer" in state and not isinstance(state["optimizer"], torch.optim.AdamW):
        raise NotImplementedError("optimizer must be torch.optim.AdamW or a subclass")


def _describe_tensor(key, tensor, ctx):
    if tensor.device.type == "meta":
        raise NotImplementedError(f"{key}: meta tensors are not supported")
    if tensor.dtype not in _DTYPES:
        raise NotImplementedError(f"{key}: unsupported real dtype {tensor.dtype}")
    meta = GlobalTensorMeta(
        list(tensor.shape) or [1], _DTYPES[tensor.dtype], tensor.element_size()
    )
    mesh = ArrangedMesh(WorldSize(ctx.world_size))
    if isinstance(tensor, DTensor):
        placement = _dtensor_placement(key, tensor, ctx)
        return NaiveDTensor(meta, [placement], mesh, torch_d_tensor=tensor)
    return NaiveDTensor(meta, [Replicate()], mesh)


def _dtensor_placement(key, tensor, ctx):
    mesh = tensor.device_mesh
    if mesh.ndim != 1 or mesh.mesh.reshape(-1).tolist() != list(range(ctx.world_size)):
        raise NotImplementedError(f"{key}: mesh must be one-dimensional over all ranks")
    placements = tensor.placements
    if len(placements) != 1:
        raise NotImplementedError(f"{key}: unsupported placements {placements}")
    placement = placements[0]
    if isinstance(placement, TorchShard) and placement.dim == 0:
        if tensor.ndim == 0:
            raise NotImplementedError(f"{key}: a scalar cannot use Shard(0)")
        return Shard(0)
    if not isinstance(placement, TorchReplicate):
        raise NotImplementedError(f"{key}: only Shard(0) and Replicate are supported")
    return Replicate()


def _component(name, tensors, objects, ctx):
    aliases = objects.get("aliases", {}) if name == "model" else {}
    entries = [
        NamedDTensor(
            key,
            _describe_tensor(key, tensor, ctx),
            [alias for alias, canonical in aliases.items() if canonical == key],
        )
        for key, tensor in tensors.items()
    ]
    component = Component(
        name,
        DTensorTable(entries),
        TorchAccessor(tensors, ctx.rank, ctx.world_size),
        objects,
    )
    component.logical_shapes = {
        key: [] for key, tensor in tensors.items() if tensor.ndim == 0
    }
    return component


def _optimizer_values(state):
    tensors, objects, paths = {}, {"param_groups": state["param_groups"]}, {}
    for fqn, fields in state["state"].items():
        _optimizer_fields(fqn, fields, tensors, objects, paths)
    return tensors, objects, paths


def _optimizer_fields(fqn, fields, tensors, objects, paths):
    for field, value in fields.items():
        key = f"state.{fqn}.{field}"
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
    if tensor.ndim == 0:
        if isinstance(tensor, DTensor) and not all(
            isinstance(placement, TorchReplicate) for placement in tensor.placements
        ):
            raise ValueError(f"state.{fqn}.{field}: scalar state must be replicated")
        return tensor
    if isinstance(parameter, DTensor) and not isinstance(tensor, DTensor):
        if tensor.shape != parameter.to_local().shape:
            raise ValueError(
                f"state.{fqn}.{field}: expected the parameter's local shard"
            )
        return DTensor.from_local(
            tensor,
            parameter.device_mesh,
            parameter.placements,
            run_check=False,
            shape=parameter.shape,
            stride=parameter.stride(),
        )
    if tensor.shape != parameter.shape:
        raise ValueError(f"state.{fqn}.{field}: state shape must match its parameter")
    if isinstance(parameter, DTensor) and (
        tensor.device_mesh != parameter.device_mesh
        or tensor.placements != parameter.placements
    ):
        raise ValueError(
            f"state.{fqn}.{field}: state sharding must match its parameter"
        )
    return tensor


def _validate_optimizer_tensors(model, state):
    parameters = dict(model.named_parameters())
    for fqn, fields in state["state"].items():
        if fqn not in parameters:
            raise ValueError(f"optimizer state has no model parameter {fqn}")
        _validate_optimizer_fields(fqn, fields, parameters[fqn])


def _validate_optimizer_fields(fqn, fields, parameter):
    for field, value in fields.items():
        if isinstance(value, torch.Tensor):
            fields[field] = _optimizer_tensor(fqn, field, value, parameter)
        elif not isinstance(value, (bool, int, float, str, type(None))):
            raise NotImplementedError(f"state.{fqn}.{field}: nested optimizer state")


def _restore_objects(state, paths, objects):
    state["param_groups"] = objects["param_groups"]
    for key, value in objects.items():
        if key != "param_groups":
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
    if objects.keys() != expected:
        key = sorted(objects.keys() ^ expected)[0]
        raise ValueError(f"optimizer: objects key mismatch for {key}")
    for key in expected - {"param_groups"}:
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
    optimizer.__setstate__(
        _optimizer_installation(optimizer, state, groups, parameters)
    )


def _fresh_optimizer_state(model, optimizer):
    # PyTorch's state-dict helper calls optimizer.step() on an empty optimizer.
    # Use private destination buffers for AdamW's lazy slots during load.
    names = {parameter: name for name, parameter in model.named_parameters()}
    groups = _named_optimizer_groups(optimizer, names)
    state = {}
    if not any(
        parameter.grad is not None
        for group in optimizer.param_groups
        for parameter in group["params"]
    ):
        state = {
            names[parameter]: _fresh_adamw_slots(parameter, group)
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        }
    return {"state": state, "param_groups": groups}


def _fresh_adamw_slots(parameter, group):
    scalar_dtype = torch.get_default_dtype()
    if group["fused"] or scalar_dtype != torch.float64:
        scalar_dtype = torch.float32
    device = parameter.device if group["capturable"] or group["fused"] else "cpu"
    slots = {
        "step": torch.empty((), dtype=scalar_dtype, device=device),
        "exp_avg": torch.empty_like(parameter),
        "exp_avg_sq": torch.empty_like(parameter),
    }
    if group["amsgrad"]:
        slots["max_exp_avg_sq"] = torch.empty_like(parameter)
    return slots


class FSDP2Adapter:
    def __init__(self, *, loading=False, common_only=False):
        self.loading, self.common_only = loading, common_only
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
            state.components.append(_component("model", self.model_state, objects, ctx))
        if "optimizer" in checkpoint_state:
            self._optimizer(checkpoint_state, state, ctx)
        if self.loading:
            state.components.sort(key=lambda component: component.name != "optimizer")

    def _optimizer(self, checkpoint_state, state, ctx):
        model, optimizer = checkpoint_state["model"], checkpoint_state["optimizer"]
        values = (
            _fresh_optimizer_state(model, optimizer)
            if self.loading and not optimizer.state
            else get_optimizer_state_dict(model, optimizer)
        )
        self.optimizer_state = values
        if self.loading:
            self.optimizer_state = {
                "param_groups": values["param_groups"],
                "state": {fqn: dict(fields) for fqn, fields in values["state"].items()},
            }
        _validate_optimizer_tensors(model, self.optimizer_state)
        tensors, objects, self.optimizer_paths = _optimizer_values(self.optimizer_state)
        state.components.append(_component("optimizer", tensors, objects, ctx))

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
            checkpoint_state["optimizer"].__setstate__(self.loaded_optimizer)
        if self.loaded_model is not None:
            checkpoint_state["model"].load_state_dict(self.loaded_model, strict=True)
        if self.loaded_extra is not _NO_EXTRA_STATE:
            checkpoint_state["extra_state"] = self.loaded_extra


def _finish(state):
    for component in state.components:
        component.accessor.finish()


def save_checkpoint(
    path,
    checkpoint_state,
    global_steps=None,
    *,
    checkpoint_format="safetensors",
    hf_dtype=None,
    max_shard_size=None,
    _preflight=None,
):
    ctx = _context()

    def prepare():
        if _preflight is not None:
            return _preflight()
        return path, _save_route(checkpoint_format, hf_dtype, max_shard_size)

    root, route = _collective(ctx, prepare)
    _unwrap(
        core._agree_choice(
            ctx, "checkpoint save route", route, (*_SAVE_FORMATS, "HF weights")
        )
    )
    if route == "HF weights":
        from .hf import _save_hf

        return _save_hf(root, checkpoint_state, hf_dtype, max_shard_size, ctx)
    adapter = FSDP2Adapter()
    state = _collective(ctx, lambda: adapter.describe(checkpoint_state, ctx))
    state.global_steps = global_steps
    if route == "safetensors":
        result = core.save(root, state, ctx)
    elif route == "dcp":
        from . import dcp

        result = dcp.save(root, state, ctx)
    else:
        result = hf.save_state(root, state, ctx, max_shard_size=max_shard_size)
    _unwrap(result)
    _collective(ctx, lambda: _finish(state))


def _load_format(path, state):
    root = Path(path)
    core._require_not_pending(root)
    if (root / desc.MANIFEST_NAME).is_file():
        if not state.components:
            return "safetensors"
        saved = desc.read_manifest(root, _validate_components=False)
        formats = {item["format"] for item in saved.components.values()}
        return "dcp" if "distcp" in formats else "safetensors"
    if (root / hf._COMMITTED).is_file():
        return "hf"
    native_hf = hf._has_native_files(root)
    native_dcp = any(
        (root / name / ".metadata").is_file() for name in ("model", "optimizer")
    )
    if native_hf and native_dcp:
        raise ValueError("ambiguous checkpoint format: native HF and DCP files")
    if native_hf:
        return "hf"
    if native_dcp:
        return "dcp"
    raise ValueError("missing checkpoint completion marker or native format files")


def load_checkpoint(path, checkpoint_state, common_only=False, *, _preflight=None):
    ctx = _context()
    adapter = FSDP2Adapter(loading=True, common_only=common_only)

    def prepare():
        if _preflight is not None:
            _preflight()
        return adapter.describe(checkpoint_state, ctx)

    state = _collective(ctx, prepare)

    def preflight(objects):
        adapter.prepare_restore(checkpoint_state, objects)
        return adapter.prepare_rank_state

    fmt = _unwrap(
        core._collective_choice(
            ctx,
            "checkpoint load format",
            lambda: _load_format(path, state),
            ("safetensors", "dcp", "hf"),
        )
    )
    if fmt == "hf":
        result = hf.load_state(
            path, state, ctx, _preflight=preflight, _optional_rank_state=common_only
        )
    elif fmt == "dcp":
        from . import dcp

        optimizer = adapter.optimizer_state
        validate_groups = (
            partial(_validate_optimizer_groups, current=optimizer["param_groups"])
            if optimizer is not None
            else None
        )
        result = dcp.load(
            path, state, ctx, _preflight=preflight, _validate_groups=validate_groups
        )
    else:
        result = core.load(
            path, state, ctx, _preflight=preflight, _optional_rank_state=common_only
        )
    if type(result) is Err and isinstance(result.value, core._LoadValidationError):
        raise result.value.cause
    _unwrap(result)
    _collective(ctx, lambda: _finish(state))
    _collective(ctx, lambda: adapter.restore(checkpoint_state, state))
