"""Torch-facing HF weight export, with only one converted tensor staged at a time."""

from collections.abc import Mapping
from dataclasses import replace
import json
from typing import Any, cast

import torch
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import DTensorTable, GlobalTensorMeta, NamedDTensor

from ..hf import _load_objects, load_state, read_headers, save_state
from ..state import CheckpointState
from .fsdp2 import (
    _DTYPES as _TORCH_DTYPE_CODES,
    _collective,
    _component,
    _context,
    _model_values,
    _unwrap,
)
from .torch_access import TorchAccessor

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
_FORMAT_DTYPES = {"fp32": ("F32", 4), "bf16": ("BF16", 2)}
_SUPPORTED_CODES = set(_TORCH_DTYPE_CODES.values())


class _CastAccessor(DTensorDataAccessor):
    """Convert local shards lazily, releasing each before visiting the next."""

    def __init__(self, tensors, dtype, ctx):
        self.tensors, self.dtype, self.ctx = tensors, dtype, ctx
        self.current: str | None = None
        self.accessor: TorchAccessor | None = None

    def _get(self, key):
        if key != self.current:
            self.finish()
            tensor = self.tensors[key]
            tensor = (
                tensor.to(dtype=self.dtype) if tensor.is_floating_point() else tensor
            )
            self.current = key
            self.accessor = TorchAccessor(
                {key: tensor}, self.ctx.rank, self.ctx.world_size
            )
        assert self.accessor is not None
        return self.accessor

    def read_to_buffer(self, global_range, read_to_buffer):
        return self._get(global_range.global_tensor_id).read_to_buffer(
            global_range, read_to_buffer
        )

    def write_from_buffer(self, global_range, write_from_buffer):
        raise NotImplementedError("converted accessors are export-only")

    def finish(self):
        if self.accessor is not None:
            self.accessor.finish()
        self.current, self.accessor = None, None


def _dtype(dtype):
    if dtype not in _DTYPES:
        raise ValueError("dtype must be 'fp32' or 'bf16'")
    return _DTYPES[dtype]


def _master_values(model, optimizer, tensors):
    state = cast(dict[str, Any], get_optimizer_state_dict(model, optimizer))
    masters = {
        fqn: fields["master_param"]
        for fqn, fields in state["state"].items()
        if "master_param" in fields
    }
    parameter_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point()
    }
    missing = parameter_names - masters.keys()
    if missing:
        raise ValueError(
            f"fp32 HF export is missing master_param for {sorted(missing)[0]}"
        )
    return tensors | {name: masters[name] for name in parameter_names & tensors.keys()}


def _inputs(model_or_state):
    model, optimizer = None, None
    if (
        isinstance(model_or_state, Mapping)
        and "model" in model_or_state
        and isinstance(model_or_state["model"], torch.nn.Module)
    ):
        model = model_or_state["model"]
        optimizer = model_or_state.get("optimizer")
    elif isinstance(model_or_state, torch.nn.Module):
        model = model_or_state
    if model is not None:
        tensors = _model_values(model)
        if isinstance(tensors, tuple):
            tensors = tensors[0]
        return model, optimizer, tensors
    if not isinstance(model_or_state, Mapping) or any(
        not isinstance(value, torch.Tensor) for value in model_or_state.values()
    ):
        raise TypeError(
            "model_or_state must be a module, tensor mapping, or {'model': module, 'optimizer': optimizer}"
        )
    return None, None, dict(model_or_state)


def _aliases(model):
    aliases, by_id = {}, {}
    if model is None:
        return aliases
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in by_id:
            aliases[name] = by_id[id(parameter)]
        else:
            by_id[id(parameter)] = name
    return aliases


def _weights(model_or_state, dtype, ctx, *, headers=None):
    loading = headers is not None
    model, optimizer, tensors = _inputs(model_or_state)
    aliases = _aliases(model)
    tensors = {name: tensor for name, tensor in tensors.items() if name not in aliases}
    if loading:
        dtype = _load_dtype(headers, tensors, aliases, dtype)
    target_dtype = None if loading and dtype is None else _dtype(dtype)
    if dtype == "fp32" and optimizer is not None:
        tensors = _master_values(model, optimizer, tensors)
    component = _component("model", tensors, {}, ctx)
    entries = []
    for named in component.table.named_d_tensors:
        key = named.global_tensor_id
        alias_names = [
            alias for alias, canonical in aliases.items() if canonical == key
        ]
        described = named.d_tensor
        tensor = tensors[key]
        if loading and tensor.is_floating_point() and tensor.dtype != target_dtype:
            raise ValueError(
                f"HF target dtype mismatch for {key}: expected {target_dtype}"
            )
        if not loading and tensor.is_floating_point():
            code, itemsize = _FORMAT_DTYPES[dtype]
            meta = described.global_tensor_meta
            described = replace(
                described,
                global_tensor_meta=GlobalTensorMeta(
                    list(meta.global_shape), code, itemsize
                ),
            )
        entries.append(NamedDTensor(key, described, alias_names))
    component.table = DTensorTable(entries)
    if not loading:
        component.accessor = _CastAccessor(tensors, target_dtype, ctx)
    return model, CheckpointState(components=[component])


def _write_config(path, model, dtype, ctx):
    from pathlib import Path
    from .. import desc

    if ctx.rank != 0 or model is None or not hasattr(model, "config"):
        return
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(model.config.to_json_string(use_diff=False))
    config["dtype"] = "float32" if dtype == "fp32" else "bfloat16"
    if "torch_dtype" in config:
        config["torch_dtype"] = config["dtype"]
    desc.atomic_write(root / "config.json", desc.json_bytes(config))


def save_hf(path, model_or_state, dtype="fp32", max_shard_size=None):
    ctx = _context()
    state, write_metadata = _collective(
        ctx, lambda: _prepare_save(model_or_state, dtype, ctx)
    )
    _unwrap(
        save_state(
            path,
            state,
            ctx,
            max_shard_size=max_shard_size,
            _write_metadata=write_metadata,
            _dtype_option=dtype,
        )
    )


def _prepare_save(model_or_state, dtype, ctx):
    model, state = _weights(model_or_state, dtype, ctx)
    return state, lambda stage: _write_config(stage, model, dtype, ctx)


def _load_dtype(headers, tensors, aliases, dtype):
    requested = {key: key for key in tensors if key in headers}
    for alias, key in aliases.items():
        if key in tensors and alias in headers:
            requested.setdefault(key, alias)
    for name in requested.values():
        code = headers[name]["dtype"]
        if code not in _SUPPORTED_CODES:
            raise ValueError(f"unsupported dtype {code} for {name}")
    if dtype is not None or not any(
        tensor.is_floating_point() for tensor in tensors.values()
    ):
        return dtype
    floating = {
        headers[name]["dtype"]
        for key, name in requested.items()
        if tensors[key].is_floating_point()
        and headers[name]["dtype"] in ("F32", "BF16")
    }
    if len(floating) != 1:
        raise ValueError("cannot infer HF dtype; pass dtype='fp32' or 'bf16'")
    return "fp32" if floating == {"F32"} else "bf16"


def _model_headers(path):
    headers = read_headers(path)
    _, optimizer_tensors = _load_objects(path, {"model"})
    if not optimizer_tensors:
        return headers
    return {
        key: entry for key, entry in headers.items() if key not in optimizer_tensors
    }


def load_hf(path, model_or_state, dtype=None):
    ctx = _context()
    _model, state = _collective(
        ctx,
        lambda: _weights(model_or_state, dtype, ctx, headers=_model_headers(path)),
    )
    _unwrap(load_state(path, state, ctx, _dtype_option=dtype))
