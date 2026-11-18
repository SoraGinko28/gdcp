"""Torch-facing HF weight export, with only one converted tensor staged at a time."""

from collections.abc import Mapping
from dataclasses import replace
import json
from typing import Any, cast

import torch
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import DTensorTable, GlobalTensorMeta, NamedDTensor

from ..hf import load_state, read_headers, save_state
from ..state import CheckpointState
from .fsdp2 import (
    _collective,
    _component,
    _context,
    _model_values,
    _unwrap,
)
from .torch_access import TorchAccessor

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
_FORMAT_DTYPES = {"fp32": ("F32", 4), "bf16": ("BF16", 2)}


class _CastAccessor(DTensorDataAccessor):
    """Convert local shards lazily, releasing each before visiting the next."""

    def __init__(self, tensors, dtype, ctx):
        self.tensors, self.dtype, self.ctx = tensors, dtype, ctx
        self.current: str | None = None
        self.accessor: TorchAccessor | None = None

    def _get(self, key):
        if key != self.current:
            self.finish()
            tensor = self.tensors[key].to(dtype=self.dtype)
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
    parameter_names = {name for name, _parameter in model.named_parameters()}
    missing = parameter_names - masters.keys()
    if missing:
        raise ValueError(
            f"fp32 HF export is missing master_param for {sorted(missing)[0]}"
        )
    return {name: masters.get(name, tensor) for name, tensor in tensors.items()}


def _inputs(model_or_state, dtype):
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
        if dtype == "fp32" and optimizer is not None:
            tensors = _master_values(model, optimizer, tensors)
        return model, tensors
    if not isinstance(model_or_state, Mapping) or any(
        not isinstance(value, torch.Tensor) for value in model_or_state.values()
    ):
        raise TypeError(
            "model_or_state must be a module, tensor mapping, or {'model': module, 'optimizer': optimizer}"
        )
    return None, dict(model_or_state)


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


def _weights(model_or_state, dtype, ctx, *, loading):
    target_dtype = _dtype(dtype)
    model, tensors = _inputs(model_or_state, dtype)
    aliases = _aliases(model)
    tensors = {name: tensor for name, tensor in tensors.items() if name not in aliases}
    component = _component("model", tensors, {}, ctx)
    entries = []
    for named in component.table.named_d_tensors:
        alias_names = [
            alias
            for alias, canonical in aliases.items()
            if canonical == named.global_tensor_id
        ]
        described = named.d_tensor
        if loading and tensors[named.global_tensor_id].dtype != target_dtype:
            raise ValueError(
                f"HF target dtype mismatch for {named.global_tensor_id}: expected {target_dtype}"
            )
        if not loading:
            code, itemsize = _FORMAT_DTYPES[dtype]
            meta = described.global_tensor_meta
            described = replace(
                described,
                global_tensor_meta=GlobalTensorMeta(
                    list(meta.global_shape), code, itemsize
                ),
            )
        entries.append(NamedDTensor(named.global_tensor_id, described, alias_names))
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
    return _save_hf(path, model_or_state, dtype, max_shard_size, _context())


def _save_hf(path, model_or_state, dtype, max_shard_size, ctx):
    model, state = _collective(
        ctx, lambda: _weights(model_or_state, dtype, ctx, loading=False)
    )
    _unwrap(
        save_state(
            path,
            state,
            ctx,
            max_shard_size=max_shard_size,
            _write_metadata=lambda stage: _write_config(stage, model, dtype, ctx),
        )
    )


def _infer_dtype(path):
    floating = {
        entry["dtype"]
        for entry in read_headers(path).values()
        if entry["dtype"] in ("F32", "BF16")
    }
    if len(floating) != 1:
        raise ValueError("cannot infer HF dtype; pass dtype='fp32' or 'bf16'")
    return "fp32" if floating == {"F32"} else "bf16"


def load_hf(path, model_or_state, dtype=None):
    ctx = _context()
    _model, state = _collective(
        ctx,
        lambda: _weights(
            model_or_state,
            _infer_dtype(path) if dtype is None else dtype,
            ctx,
            loading=True,
        ),
    )
    _unwrap(load_state(path, state, ctx))
