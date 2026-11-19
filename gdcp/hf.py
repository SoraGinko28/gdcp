"""HF-layout safetensors with rank-local GDTensor I/O and native header loading.

This module stays framework neutral. Torch-facing convenience functions import
``adapters.hf`` only when called. No GDCP manifest is needed to read HF files.
"""

import hashlib
import json
import re
import struct
from dataclasses import replace
from functools import partial
from math import prod
from pathlib import Path
from typing import Any

from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import (
    DTensorTable,
    Err,
    GlobalTensorMeta,
    NaiveDTensor,
    NamedDTensor,
    Replicate,
    StorageDTensorsView,
)
from gdtensor_conn._geometry import nonempty_columns
from gdtensor_conn._layout import build_payload_index
from gdtensor_conn.access import StorageDataAccessor
from gdtensor_conn.formats import HF_INDEX_NAME, LoadSessionBase
from gdtensor_conn.iobackend import FileIO
from gdtensor_conn.models import build_layout

from . import core, desc
from .backend import (
    _CoordinatedSaveSession,
    _InterruptIO,
    _begin_session,
    _collective_begin,
    _raise_io_interrupt,
)
from ._transaction import PREFLIGHT
from .policy import PerRankPolicy, storage_mesh
from .state import Component

_SAFETENSORS_DTYPE_BITS = {
    "BOOL": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}
_OBJECTS = "optimizer.json"
_COMMITTED = desc.HF_COMMITTED_NAME
_SINGLE_FILE = "model.safetensors"
# The slash keeps this metadata key outside the component-name namespace.
_OPTIMIZER_TENSORS = "/optimizer_tensors"


def shard_size_bytes(value):
    """Parse HF-style decimal GB/MB/KB or binary GiB/MiB/KiB limits."""
    if value is None:
        return None
    if type(value) is int and value > 0:
        return value
    match = (
        re.fullmatch(
            r"\s*([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)\s*",
            value,
            re.IGNORECASE,
        )
        if isinstance(value, str)
        else None
    )
    if not match:
        raise ValueError(
            "max_shard_size must be a positive byte count or size such as '5GB'"
        )
    unit = match[2].upper()
    power = 0 if unit == "B" else "KMGT".index(unit[0]) + 1
    size = int(float(match[1]) * (1024 if "I" in unit else 1000) ** power)
    if size < 1:
        raise ValueError("max_shard_size must be positive")
    return size


def _file_assignment(table, max_shard_size):
    limit = shard_size_bytes(max_shard_size)
    groups, current, size = [], [], 0
    for named in table.named_d_tensors:
        meta = named.d_tensor.global_tensor_meta
        nbytes = prod(meta.global_shape) * meta.itemsize
        if limit is not None and current and size + nbytes > limit:
            groups.append(current)
            current, size = [], 0
        current.append(named.global_tensor_id)
        size += nbytes
        if limit is not None and size > limit:
            groups.append(current)
            current, size = [], 0
    if current:
        groups.append(current)
    count = len(groups)
    return {
        key: _SINGLE_FILE
        if count == 1
        else f"model-{i + 1:05d}-of-{count:05d}.safetensors"
        for i, group in enumerate(groups)
        for key in group
    }


def _tensor_files(assignment):
    return set(assignment.values()) or {_SINGLE_FILE}


class _CombinedAccessor(DTensorDataAccessor):
    def __init__(self, routes):
        self.routes = routes
        self._active: DTensorDataAccessor | None = None

    def _call(self, method, rng, callback):
        accessor, key = self.routes[rng.global_tensor_id]
        if self._active is not accessor:
            getattr(self._active, "finish", lambda: None)()
            self._active = accessor
        return getattr(accessor, method)(replace(rng, global_tensor_id=key), callback)

    def read_to_buffer(self, global_range, read_to_buffer):
        return self._call("read_to_buffer", global_range, read_to_buffer)

    def write_from_buffer(self, global_range, write_from_buffer):
        return self._call("write_from_buffer", global_range, write_from_buffer)

    def finish(self):
        accessors = {id(accessor): accessor for accessor, _key in self.routes.values()}
        for accessor in accessors.values():
            getattr(accessor, "finish", lambda: None)()
        self._active = None


def _combined_entry(component, named, available):
    prefix = "" if component.name == "model" else component.name + "."
    key = prefix + named.global_tensor_id
    aliases = [prefix + alias for alias in named.alias]
    if available is not None:
        aliases = [alias for alias in aliases if alias in available]
        if key not in available and aliases:
            key = aliases[0]
        if key not in available:
            raise ValueError(f"HF missing key {key}")
    return NamedDTensor(key, named.d_tensor, aliases)


def _combined(state, available=None, *, optimizer_tensors=None):
    entries, routes, logical, objects = [], {}, {}, {}
    other_keys = optimizer_keys = available
    if available is not None and optimizer_tensors is not None:
        other_keys = available - optimizer_tensors
        optimizer_keys = available & optimizer_tensors
    for component in state.components:
        objects[component.name] = component.objects
        keys = optimizer_keys if component.name == "optimizer" else other_keys
        for named in component.table.named_d_tensors:
            entry = _combined_entry(component, named, keys)
            key = entry.global_tensor_id
            entries.append(entry)
            routes[key] = component.accessor, named.global_tensor_id
            logical[key] = component.logical_shapes.get(
                named.global_tensor_id,
                list(named.d_tensor.global_tensor_meta.global_shape),
            )
    if len(entries) != len(routes):
        raise ValueError("duplicate HF tensor name")
    combined = Component("hf", DTensorTable(entries), _CombinedAccessor(routes))
    combined.logical_shapes = logical
    return combined, objects


class _HFSaveSession(_CoordinatedSaveSession):
    def __init__(self, *args, assignment, logical_shapes, **kwargs):
        self.assignment = assignment
        self.logical_shapes = logical_shapes
        super().__init__(*args, **kwargs)

    def my_files(self):
        paths = set(super().my_files())
        if self.rank == 0:
            paths.update(_tensor_files(self.assignment))
        return sorted(paths)

    def _header(self, path):
        if path in self._headers:
            return self._headers[path]
        header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
        offset = 0
        entries = [
            named
            for named in self.table.named_d_tensors
            if self.assignment[named.global_tensor_id] == path
        ]
        for named in entries:
            key = named.global_tensor_id
            meta = named.d_tensor.global_tensor_meta
            size = prod(meta.global_shape) * meta.itemsize
            header[key] = {
                "dtype": meta.dtype,
                "shape": self.logical_shapes.get(key, list(meta.global_shape)),
                "data_offsets": [offset, offset + size],
            }
            offset += size
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        self._headers[path] = struct.pack("<Q", len(raw)) + raw
        return self._headers[path]

    def _open_file(self, path):
        if path in self.index.files:
            return super()._open_file(path)
        header = self._header(path)
        self.data_start[path] = len(header)
        self.io.create_staging(self.abspath(path), len(header))
        if self.rank == 0:
            self.io.pwrite(self.rank, self.abspath(path), 0, header)

    def _leader_files(self):
        files = [
            path
            for path in self.my_files()
            if path in self.index.files
            and self.index.files[path].leader.rank == self.rank
        ]
        if self.rank == 0:
            files += sorted(_tensor_files(self.assignment) - self.index.files.keys())
        return files

    def _after_barrier(self):
        if self.rank != 0:
            return
        files = _tensor_files(self.assignment)
        if len(files) > 1:
            total = sum(
                prod(named.d_tensor.global_tensor_meta.global_shape)
                * named.d_tensor.global_tensor_meta.itemsize
                for named in self.table.named_d_tensors
            )
            desc.atomic_write(
                Path(self.root) / HF_INDEX_NAME,
                desc.json_bytes(
                    {"metadata": {"total_size": total}, "weight_map": self.assignment}
                ),
            )
        # This optional connector marker is for atomic publication only. HF
        # reading itself depends solely on the standard headers and index.
        desc.atomic_write(
            Path(self.root) / _COMMITTED,
            desc.json_bytes(
                {
                    "format": "safetensors",
                    "files": sorted(files),
                }
            ),
        )


class _HFLoadSession(LoadSessionBase):
    def __init__(self, *args, headers, **kwargs):
        self.headers = headers
        super().__init__(*args, **kwargs)

    def _read_file_offsets(self, path):
        for slot in self.index.files[path].slots:
            self.offsets[slot.pos] = self.headers[slot.pos.tensor_id]["offset"]


class HFBackend:
    """Shared-file safetensors storage; no rank assembles global tensors."""

    def __init__(self, assignment, logical_shapes=None, headers=None):
        self.assignment = assignment
        self.logical_shapes = logical_shapes or {}
        self.headers = headers

    def open(self, root, table, layout, coordinator=None):
        io = _InterruptIO(FileIO())
        index = build_payload_index(table, layout)
        ids = [named.global_tensor_id for named in table.named_d_tensors]

        arguments = str(root), index, io
        begin_save = partial(
            _begin_session,
            _HFSaveSession,
            *arguments,
            coord=coordinator,
            assignment=self.assignment,
            logical_shapes=self.logical_shapes,
        )
        begin_load = partial(
            _begin_session, _HFLoadSession, *arguments, headers=self.headers
        )
        return StorageDTensorsView(
            ids,
            table,
            StorageDataAccessor(index, io),
            layout,
            _collective_begin(begin_save, coordinator) if coordinator else begin_save,
            begin_load,
        )


def _group_storage_rows(combined, table):
    # Preserve the policy's stored owners, then keep equal column runs adjacent.
    # Both views need the same header order. Reordering the compute rows can
    # rotate a replicated tensor's designated source writer; every replica has
    # the same value, and its single stored slot is unchanged.
    rows = sorted(
        zip(combined.table.named_d_tensors, table.named_d_tensors, strict=True),
        key=lambda row: tuple(nonempty_columns(row[1])),
    )
    combined = replace(combined, table=DTensorTable([row[0] for row in rows]))
    return combined, DTensorTable([row[1] for row in rows])


def _save_prepare(path, state, ctx, max_shard_size):
    combined, objects = _combined(state)
    for component in state.components:
        if component.name == "optimizer":
            objects[_OPTIMIZER_TENSORS] = sorted(
                f"optimizer.{named.global_tensor_id}"
                for named in component.table.named_d_tensors
            )
    table, _layout = PerRankPolicy().plan(combined, ctx.world_size)
    combined, table = _group_storage_rows(combined, table)
    compute = core._compute_view(combined)
    core._require(compute.check_constraints())
    assignment = _file_assignment(table, max_shard_size)
    layout = build_layout(
        table,
        ctx.world_size,
        "safetensors",
        lambda order, rank, named: assignment[named.global_tensor_id],
    )
    storage = HFBackend(assignment, combined.logical_shapes).open(
        path, table, layout, ctx.coordinator
    )
    core._require(storage.check_constraints())
    return combined, objects, compute, storage


def _save_io(ctx, combined, compute, storage):
    try:
        result = core._rank_save(ctx, compute, storage)
        _raise_io_interrupt(storage)
        return result
    finally:
        _finish_accessor(combined)


def _initialize(transaction, state, ctx, objects, assignment, write_metadata):
    planned = {Path(name) for name in _tensor_files(assignment)}
    planned.update(core._planned_rank_files(state, ctx))
    planned.update({Path(_OBJECTS), Path(_COMMITTED)})
    # A foreign index would take precedence over the newly written single file.
    planned.add(Path(HF_INDEX_NAME))
    if write_metadata is not None:
        planned.add(Path("config.json"))
    transaction.begin(planned)
    if ctx.rank == 0:
        root = transaction.stage
        root.mkdir(parents=True, exist_ok=True)
        desc.atomic_write(root / desc.PENDING_NAME, b"HF save in progress\n")
        core._initialize_directories(root, state.rank_state)
        desc.write_objects(root / _OBJECTS, objects)


def _finish_accessor(combined):
    combined.accessor.finish()


def _save_staged(transaction, state, ctx, max_shard_size, write_metadata):
    path = transaction.stage
    prepared = core._collective(
        ctx, "HF prepare save", lambda: _save_prepare(path, state, ctx, max_shard_size)
    )
    if type(prepared) is Err:
        return prepared
    assert prepared.value is not None
    combined, objects, compute, storage = prepared.value
    result = core._collective(
        ctx,
        PREFLIGHT,
        lambda: _initialize(
            transaction,
            state,
            ctx,
            objects,
            _file_assignment(storage.d_tensor_table, max_shard_size),
            write_metadata,
        ),
    )
    if type(result) is Err:
        return result
    result = core._collective(
        ctx,
        "HF metadata",
        lambda: write_metadata(path) if write_metadata is not None else None,
    )
    if type(result) is Err:
        return result
    result = core._collective(
        ctx, "HF rank state", lambda: core._save_rank_state(path, state, ctx)
    )
    if type(result) is Err:
        return result
    result = core._collective(
        ctx, "HF save", lambda: _save_io(ctx, combined, compute, storage)
    )
    if type(result) is Err:
        return result
    return core._collective(
        ctx,
        "HF publish",
        lambda: _finalize(path, state, ctx),
    )


def _finalize(path, state, ctx):
    if ctx.rank != 0:
        return
    marker = Path(path) / _COMMITTED
    saved = json.loads(marker.read_bytes())
    saved["rank_state"] = core._rank_state_inventory(state, ctx)
    desc.atomic_write(marker, desc.json_bytes(saved))
    (Path(path) / desc.PENDING_NAME).unlink()


def save_state(path, state, ctx, *, max_shard_size=None, _write_metadata=None):
    """Save HF files in a sibling, preserving the old checkpoint on failure."""
    return core._transactional_save(
        path,
        state,
        ctx,
        lambda transaction: _save_staged(
            transaction, state, ctx, max_shard_size, _write_metadata
        ),
    )


def _read_header_file(path):
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path.name}: truncated safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if length > 100_000_000:
            raise ValueError(f"{path.name}: safetensors header exceeds 100 MB")
        raw = file.read(length)
    if len(raw) != length:
        raise ValueError(f"{path.name}: truncated safetensors header")
    return length, json.loads(raw)


def _header_entry(filename, name, entry, length):
    shape, dtype, offsets = entry["shape"], entry["dtype"], entry["data_offsets"]
    if (
        dtype not in _SAFETENSORS_DTYPE_BITS
        or not isinstance(shape, list)
        or any(type(dim) is not int or dim < 0 for dim in shape)
    ):
        raise ValueError(f"{filename}: invalid metadata for {name}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(type(value) is not int for value in offsets)
    ):
        raise ValueError(f"{filename}: invalid offsets for {name}")
    bits = _SAFETENSORS_DTYPE_BITS[dtype]
    nbytes, remainder = divmod(prod(shape) * bits, 8)
    if remainder or offsets[0] < 0 or offsets[1] - offsets[0] != nbytes:
        raise ValueError(f"{filename}: tensor size mismatch for {name}")
    return {
        "shape": shape,
        "dtype": dtype,
        "itemsize": bits // 8 if bits % 8 == 0 else None,
        "file": filename,
        "offset": 8 + length + offsets[0],
    }, (offsets[0], offsets[1], name)


def _add_header_file(root, filename, weight_map, headers):
    desc._safe_relative(filename)
    length, document = _read_header_file(root / filename)
    entries = []
    for name, entry in document.items():
        if name == "__metadata__":
            continue
        if name in headers:
            raise ValueError(f"duplicate HF tensor {name}")
        if weight_map is not None and weight_map.get(name) != filename:
            raise ValueError(f"HF index does not match header for {name}")
        headers[name], span = _header_entry(filename, name, entry, length)
        entries.append(span)
    expected = 0
    for start, end, name in sorted(entries):
        if start != expected:
            raise ValueError(f"{filename}: noncontiguous/overlapping payload at {name}")
        expected = end
    if (root / filename).stat().st_size != 8 + length + expected:
        raise ValueError(f"{filename}: payload size does not match header")


def _has_native_files(root):
    return (root / HF_INDEX_NAME).is_file() or (root / _SINGLE_FILE).is_file()


def _read_completion_marker(root):
    marker = root / _COMMITTED
    if not marker.is_file():
        return {}
    saved = json.loads(marker.read_bytes())
    if saved["format"] != "safetensors" or not isinstance(saved["files"], list):
        raise ValueError("invalid HF completion marker")
    return saved


def read_headers(path):
    """Describe native HF tensor byte ranges, with itemsize=None for packed dtypes."""
    root = Path(path)
    if (root / desc.PENDING_NAME).exists():
        raise ValueError("HF checkpoint has an incomplete save (.gdcp_pending)")
    index_file = root / HF_INDEX_NAME
    weight_map = (
        json.loads(index_file.read_bytes())["weight_map"]
        if index_file.is_file()
        else None
    )
    if weight_map is not None and (not isinstance(weight_map, dict) or not weight_map):
        raise ValueError("HF weight_map must be a nonempty mapping")
    if weight_map is None and not (root / _SINGLE_FILE).is_file():
        # Older GDCP saves with zero tensors only wrote the completion marker.
        if _read_completion_marker(root).get("files") == []:
            return {}
    files = (
        sorted(set(weight_map.values())) if weight_map is not None else [_SINGLE_FILE]
    )
    headers = {}
    for filename in files:
        _add_header_file(root, filename, weight_map, headers)
    if weight_map is not None and set(weight_map) != set(headers):
        raise ValueError("HF index tensor set does not match safetensors headers")
    return headers


def _read_payload_part(file, count):
    data = file.read(count)
    if not data:
        raise ValueError("truncated HF tensor payload")
    return data


def _payload_digest(path, entry):
    remaining = prod(entry["shape"]) * entry["itemsize"]
    digest = hashlib.sha256()
    with (Path(path) / entry["file"]).open("rb") as file:
        file.seek(entry["offset"])
        while remaining:
            data = _read_payload_part(file, min(remaining, 16 * 1024**2))
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def _legacy_optimizer_keys(keys, optimizer):
    if not isinstance(optimizer, dict):
        return set()
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or any(
        not isinstance(group, dict) or not isinstance(group.get("params"), list)
        for group in groups
    ):
        # Generic legacy sidecars provide no parameter names.
        return {key for key in keys if key.startswith("optimizer.")}
    parameters = {
        name for group in groups for name in group["params"] if isinstance(name, str)
    }
    prefixes = {f"optimizer.state.{name}." for name in parameters}
    return {
        key
        for key in keys
        if key.startswith("optimizer.state.")
        and key not in parameters
        and any(
            key[: index + 1] in prefixes
            for index, char in enumerate(key[:-1])
            if char == "."
        )
    }


def _unexpected_keys(keys, components, optimizer_tensors):
    optimizer_keys = keys & optimizer_tensors if "optimizer" in components else set()
    keys -= optimizer_tensors
    if "model" not in components:
        prefixes = tuple(name + "." for name in components - {"optimizer"})
        keys = {key for key in keys if key.startswith(prefixes)}
    return keys | optimizer_keys


def _validate_keys(combined, headers, components, optimizer_tensors, optimizer_objects):
    expected = {named.global_tensor_id for named in combined.table.named_d_tensors}
    aliases = {
        alias: named.global_tensor_id
        for named in combined.table.named_d_tensors
        for alias in named.alias
    }
    unexpected = headers.keys() - expected
    duplicates = []
    for alias in unexpected & aliases.keys():
        canonical = aliases[alias]
        left, right = headers[alias], headers[canonical]
        if any(left[field] != right[field] for field in ("shape", "dtype", "itemsize")):
            raise ValueError(f"HF tied alias {alias} differs from {canonical}")
        duplicates.append((alias, canonical))
    unexpected -= aliases.keys()
    if optimizer_tensors is None:
        optimizer_tensors = _legacy_optimizer_keys(unexpected, optimizer_objects)
    unexpected = _unexpected_keys(unexpected, components, optimizer_tensors)
    if unexpected:
        raise ValueError(f"HF unexpected key {sorted(unexpected)[0]}")
    return duplicates


def _validate_alias_payloads(path, headers, duplicates):
    for alias, canonical in duplicates:
        if _payload_digest(path, headers[alias]) != _payload_digest(
            path, headers[canonical]
        ):
            raise ValueError(f"HF tied alias {alias} differs from {canonical}")


def _load_objects(path, components):
    filename = Path(path) / _OBJECTS
    objects = desc.read_objects(filename) if filename.is_file() else {}
    if not isinstance(objects, dict):
        raise ValueError("HF optimizer.json must contain a mapping")
    if "optimizer" in components and "optimizer" not in objects:
        raise ValueError("HF full-state checkpoint is missing optimizer.json")
    if _OPTIMIZER_TENSORS not in objects:
        return objects, None
    optimizer_tensors = objects.pop(_OPTIMIZER_TENSORS)
    if (
        not isinstance(optimizer_tensors, list)
        or (optimizer_tensors and "optimizer" not in objects)
        or any(
            not isinstance(key, str) or not key.startswith("optimizer.")
            for key in optimizer_tensors
        )
    ):
        raise ValueError("invalid HF optimizer tensor inventory")
    return objects, set(optimizer_tensors)


def _load_prepare(path, state, ctx, validate_groups):
    core._validate_context(ctx)
    core._validate_state(state)
    headers = read_headers(path)
    components = {component.name for component in state.components}
    objects, optimizer_tensors = _load_objects(path, components)
    if validate_groups is not None:
        optimizer = objects["optimizer"]
        if not isinstance(optimizer, dict):
            raise ValueError("HF objects for optimizer must contain a mapping")
        core._validate_load_call(lambda: validate_groups(optimizer.get("param_groups")))
    combined, _objects = _combined(
        state, headers.keys(), optimizer_tensors=optimizer_tensors
    )
    duplicates = _validate_keys(
        combined, headers, components, optimizer_tensors, objects.get("optimizer")
    )
    entries = []
    for named in combined.table.named_d_tensors:
        key, meta = named.global_tensor_id, named.d_tensor.global_tensor_meta
        saved = headers[key]
        if saved["itemsize"] is None:
            raise ValueError(f"unsupported dtype {saved['dtype']} for {key}")
        logical = combined.logical_shapes.get(key, list(meta.global_shape))
        if (
            logical != saved["shape"]
            or meta.dtype != saved["dtype"]
            or meta.itemsize != saved["itemsize"]
        ):
            raise ValueError(f"HF shape/dtype mismatch for {key}")
        entries.append(
            NamedDTensor(
                key,
                NaiveDTensor(
                    GlobalTensorMeta(
                        saved["shape"] or [1], saved["dtype"], saved["itemsize"]
                    ),
                    [Replicate()],
                    storage_mesh(1, [0]),
                ),
            )
        )
    table = DTensorTable(entries)
    assignment = {key: info["file"] for key, info in headers.items()}
    layout = build_layout(
        table,
        1,
        "safetensors",
        lambda order, rank, named: assignment[named.global_tensor_id],
    )
    storage = HFBackend(assignment, headers=headers).open(
        path, table, layout, ctx.coordinator
    )
    compute = core._compute_view(combined)
    core._require(compute.check_constraints())
    core._require(storage.check_constraints())
    return combined, compute, storage, objects, headers, duplicates


def _load_rank_state(path, state, ctx, *, optional):
    root = Path(path)
    saved = _read_completion_marker(root)
    if "rank_state" in saved:
        return core._load_rank_state(
            root, state, saved["rank_state"], ctx, optional=optional
        )
    # Older HF saves did not record the rank files in their completion marker.
    blobs = {}
    for name in state.rank_state:
        filename = root / name / f"{name}_rank_{ctx.rank}.pt"
        if not filename.is_file():
            filename = filename.with_name(f"{name}_rank_0.pt")
        if not optional or filename.is_file():
            blobs[name] = filename.read_bytes()
    return blobs


def _load_rank_only(path, state, ctx, *, optional):
    core._validate_context(ctx)
    core._validate_state(state)
    core._require_not_pending(path)
    root = Path(path)
    if not (root / _COMMITTED).is_file() and not _has_native_files(root):
        raise ValueError("missing HF completion marker or native safetensors files")
    return _load_rank_state(root, state, ctx, optional=optional)


def _validate_load_objects(state, objects, preflight):
    for component in state.components:
        if not isinstance(objects.get(component.name, {}), dict):
            raise ValueError(f"HF objects for {component.name} must contain a mapping")
    if preflight is not None:
        return preflight(objects)


def load_state(
    path,
    state,
    ctx,
    *,
    _preflight=None,
    _validate_groups=None,
    _optional_rank_state=False,
):
    """Load native HF headers directly, including an optional optimizer sidecar."""
    agreed = core._agree_components(state, ctx)
    if type(agreed) is Err:
        return agreed
    if not state.components:
        return core._load_components(
            [],
            ctx,
            state,
            lambda: _load_rank_only(path, state, ctx, optional=_optional_rank_state),
            _preflight,
        )
    prepared = core._collective(
        ctx,
        "HF prepare load",
        lambda: _load_prepare(path, state, ctx, _validate_groups),
    )
    if type(prepared) is Err:
        return prepared
    assert prepared.value is not None
    combined, compute, storage, objects, headers, duplicates = prepared.value

    result = core._load_payloads(
        [(combined.name, compute, storage)],
        ctx,
        lambda: _load_rank_state(path, state, ctx, optional=_optional_rank_state),
        preflight=lambda: _validate_load_objects(state, objects, _preflight),
        validate=lambda: _validate_alias_payloads(path, headers, duplicates),
    )
    if type(result) is Err:
        return result
    return core._collective(
        ctx, "HF load complete", lambda: _restore_state(state, objects, result.value)
    )


def _restore_state(state, objects, blobs):
    for component in state.components:
        if component.name in objects:
            component.objects.clear()
            component.objects.update(objects[component.name])
    state.rank_state.update(blobs)


def save_hf(path, model_or_state, dtype="fp32", max_shard_size=None):
    """Export HF weights; fp32 uses supplied optimizer master weights when present.

    ``model_or_state`` is a module, a tensor mapping, or a checkpoint mapping
    containing ``model`` and optionally ``optimizer``. ``dtype`` accepts
    ``'fp32'`` or ``'bf16'`` and applies only to floating-point tensors.
    Integer and boolean tensors retain their dtypes and values.
    ``max_shard_size=None`` emits one file; otherwise
    a positive byte count or an HF-style limit such as ``'5GB'`` is accepted.
    """
    from .adapters.hf import save_hf as save

    return save(path, model_or_state, dtype=dtype, max_shard_size=max_shard_size)


def load_hf(path, model_or_state, dtype=None):
    """Restore HF weights in place into a module or tensor/checkpoint mapping.

    ``dtype`` optionally selects ``'fp32'`` optimizer master tensors or
    ``'bf16'`` model parameters in a checkpoint mapping. With ``None`` it is
    inferred from the requested floating-point model tensors and aliases.
    Saved and target shapes and dtypes must match exactly.
    """
    from .adapters.hf import load_hf as load

    return load(path, model_or_state, dtype=dtype)
