"""Native, deterministic per-rank DCP records filled by GDTensor byte I/O.

Each tensor chunk is a standard ``torch.save`` ZIP record. ``skip_data`` creates
only its small envelope; GDTensor writes the payload directly into the reserved
record. Fake tensors avoid allocating either the local or global tensor here.
"""

import io
import mmap
import os
import pickle
import struct
import zipfile
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from math import prod
from pathlib import Path
from typing import IO, cast

import numpy as np
import torch
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch._subclasses.fake_tensor import FakeTensorMode

from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import (
    ComputeDTensorsView,
    DTensorTable,
    Err,
    GlobalTensorMeta,
    NaiveDTensor,
    NamedDTensor,
    Replicate,
    Shard,
    Slice,
    SlicedMesh,
    StorageDTensorsView,
    global_tensor_meta_of,
)
from gdtensor.session import StorageLoadSession, StorageSaveSession
from gdtensor_conn._geometry import owned_box
from gdtensor_conn._layout import build_payload_index
from gdtensor_conn._result import catch, invalid, io_error
from gdtensor_conn.access import StorageDataAccessor, call_buffer_callback
from gdtensor_conn.iobackend import FileIO
from gdtensor_conn.models import build_layout

from .. import core, desc
from ..backend import _collective_begin
from ..policy import PerRankPolicy, storage_mesh

_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "U16": torch.uint16,
    "U32": torch.uint32,
    "U64": torch.uint64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
_CODES = {value: key for key, value in _DTYPES.items()}


class _Envelope:
    """Sparse seekable stream storing only the torch ZIP envelope in memory."""

    def __init__(self):
        self.position = 0
        self.length = 0
        self.pieces = []

    def write(self, value):
        self.pieces.append((self.position, bytes(value)))
        self.position += len(value)
        self.length = max(self.length, self.position)
        return len(value)

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        origin = (0, self.position, self.length)[whence]
        self.position = origin + offset
        self.length = max(self.length, self.position)
        return self.position

    def flush(self):
        pass

    def read(self, size=-1):
        size = (
            min(size, self.length - self.position)
            if size >= 0
            else self.length - self.position
        )
        data = bytearray(size)
        for start, value in self.pieces:
            _copy_intersection(data, self.position, start, value)
        self.position += size
        return bytes(data)


def _copy_intersection(data, position, start, value):
    lo, hi = max(position, start), min(position + len(data), start + len(value))
    if lo < hi:
        data[lo - position : hi - position] = value[lo - start : hi - start]


@lru_cache(maxsize=512)
def _envelope(shape, dtype):
    stream = _Envelope()
    with FakeTensorMode():
        tensor = torch.empty(shape, dtype=dtype)
        with torch.serialization.skip_data(materialize_fake_tensors=True):
            torch.save(tensor, cast(IO[bytes], stream))
    offset, size = _payload_record(stream)
    return stream.length, tuple(stream.pieces), offset, size


def _payload_record(stream):
    with zipfile.ZipFile(cast(IO[bytes], stream)) as archive:
        records = [entry for entry in archive.infolist() if "/data/" in entry.filename]
    if len(records) != 1 or records[0].compress_type != zipfile.ZIP_STORED:
        raise ValueError("DCP tensor record must have one uncompressed storage")
    entry = records[0]
    stream.seek(entry.header_offset)
    header = stream.read(30)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    return entry.header_offset + 30 + name_len + extra_len, entry.file_size


class _Window:
    """Read a bounded ZIP subarchive without reading the .distcp file."""

    def __init__(self, handle, start, length):
        self.handle, self.start, self.length = handle, start, length
        self.position = 0

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, self.length)[whence] + offset
        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        size = (
            min(size, self.length - self.position)
            if size >= 0
            else self.length - self.position
        )
        value = os.pread(self.handle.fileno(), size, self.start + self.position)
        self.position += len(value)
        return value


class DcpPolicy:
    def plan(self, component, world_size):
        table, _ = PerRankPolicy().plan(component, world_size)
        layout = build_layout(
            table, world_size, "distcp", lambda order, rank, named: f"__{rank}_0.distcp"
        )
        return table, layout


def _logical_shape(component, named):
    return component.logical_shapes.get(
        named.global_tensor_id, list(global_tensor_meta_of(named).global_shape)
    )


def _slot_geometry(component, named, rank):
    box = owned_box(named, "storage", rank)
    if box is None:
        raise ValueError("DCP slot is outside its storage mesh")
    lo, hi = box
    if _logical_shape(component, named) == []:
        return (), ()
    return tuple(lo), tuple(right - left for left, right in zip(lo, hi))


def _planner_path(component, tensor_id):
    if component.name == "optimizer" and tensor_id.startswith("state."):
        fqn, field = tensor_id[6:].rsplit(".", 1)
        return "state", fqn, field
    return (tensor_id,)


def _objects(component):
    values = {
        key: value for key, value in component.objects.items() if key != "aliases"
    }
    return flatten_state_dict(values)


@dataclass
class _Plan:
    offsets: dict
    records: dict
    sizes: dict
    metadata: Metadata


def _plan(component, table, layout, index):
    offsets, records, sizes, tensors, storage, planner = {}, {}, {}, {}, {}, {}
    for named in table.named_d_tensors:
        _plan_tensor(
            component, named, index, offsets, records, sizes, tensors, storage, planner
        )
    _plan_objects(component, records, sizes, tensors, storage, planner)
    metadata = Metadata(tensors, planner_data=planner, storage_data=storage)
    return _Plan(offsets, records, sizes, metadata)


def _plan_tensor(
    component, named, index, offsets, records, sizes, tensors, storage, planner
):
    tensor_id = named.global_tensor_id
    meta = global_tensor_meta_of(named)
    chunks = []
    for pos, (path, slot) in index.slots.items():
        if pos.tensor_id == tensor_id:
            chunks.append(
                _plan_chunk(
                    component, named, pos, path, slot, offsets, records, sizes, storage
                )
            )
    properties = TensorProperties(dtype=_DTYPES[meta.dtype])
    tensors[tensor_id] = TensorStorageMetadata(
        properties, torch.Size(_logical_shape(component, named)), chunks
    )
    planner[tensor_id] = _planner_path(component, tensor_id)
    _plan_aliases(named, tensors, storage, planner)


def _plan_chunk(component, named, pos, path, slot, offsets, records, sizes, storage):
    chunk_offset, chunk_shape = _slot_geometry(component, named, pos.rank)
    meta = global_tensor_meta_of(named)
    length, pieces, raw_offset, nbytes = _envelope(chunk_shape, _DTYPES[meta.dtype])
    if nbytes != slot.nbytes:
        raise ValueError(f"{pos.tensor_id}: DCP envelope payload size mismatch")
    start = sizes.get(path, 0)
    records.setdefault(path, []).extend(
        (start + offset, data) for offset, data in pieces
    )
    sizes[path] = start + length
    offsets[pos] = start + raw_offset
    index = MetadataIndex(pos.tensor_id, torch.Size(chunk_offset))
    storage[index] = _StorageInfo(path, start, length)
    return ChunkStorageMetadata(torch.Size(chunk_offset), torch.Size(chunk_shape))


def _plan_aliases(named, tensors, storage, planner):
    for alias in named.alias:
        tensors[alias] = tensors[named.global_tensor_id]
        planner[alias] = (alias,)
        copies = [
            (MetadataIndex(alias, index.offset), value)
            for index, value in storage.items()
            if index.fqn == named.global_tensor_id
        ]
        storage.update(copies)


def _plan_objects(component, records, sizes, tensors, storage, planner):
    values, paths = _objects(component)
    for key, value in sorted(values.items()):
        buffer = io.BytesIO()
        torch.save(value, buffer)
        path, start = "__0_0.distcp", sizes.get("__0_0.distcp", 0)
        raw = buffer.getvalue()
        records.setdefault(path, []).append((start, raw))
        sizes[path] = start + len(raw)
        tensors[key] = BytesStorageMetadata()
        storage[MetadataIndex(key)] = _StorageInfo(path, start, len(raw))
        planner[key] = paths[key]


class _Session:
    def __init__(self, rank, root, table, layout, index, io_backend):
        self.rank, self.root, self.table, self.layout = rank, str(root), table, layout
        self.index, self.io, self.state = index, io_backend, "open"
        self.offsets = {}

    def abspath(self, relative):
        return os.path.join(self.root, relative)

    def my_files(self):
        return sorted(
            {
                path
                for pos, (path, slot) in self.index.slots.items()
                if pos.rank == self.rank
            }
        )

    @catch
    def get_file_offset(self, pos):
        if pos not in self.offsets:
            raise invalid(
                "file_offset", f"{pos.tensor_id}:{pos.rank}: DCP payload is absent"
            )
        return self.offsets[pos]

    @catch
    def init_data_accessor(self):
        return StorageDataAccessor(self.index, self.io, session=self)


def _commit_failure(error, message):
    if error is not None and not isinstance(error, Exception):
        raise error
    raise ValueError(f"{message}: {error or 'another rank failed'}")


class _SaveSession(_Session, StorageSaveSession):
    def __init__(self, *args, component, coordinator=None):
        super().__init__(*args)
        if self.rank == 0:
            (Path(self.root) / ".metadata").unlink(missing_ok=True)
        self.coordinator = coordinator
        self.plan = _plan(component, self.table, self.layout, self.index)
        self.offsets = self.plan.offsets
        self._create_records()

    def abspath(self, relative):
        return os.path.join(self.root, relative + ".tmp")

    def my_files(self):
        paths = set(super().my_files())
        if self.rank == 0 and "__0_0.distcp" in self.plan.sizes:
            paths.add("__0_0.distcp")
        return sorted(paths)

    def _create_records(self):
        for path in self.my_files():
            self._create_record_file(path)

    def _create_record_file(self, path):
        self.io.create_staging(self.abspath(path), self.plan.sizes[path])
        for offset, data in self.plan.records[path]:
            self.io.pwrite(self.rank, self.abspath(path), offset, data)

    def _vote(self, ok):
        return ok if self.coordinator is None else self.coordinator.vote(ok)

    def _publish(self):
        for path in self.my_files():
            self.io.sync(self.abspath(path))
            self.io.publish(self.abspath(path), os.path.join(self.root, path))

    @catch
    def commit(self):
        error = self._close_checked()
        if not self._vote(error is None):
            _commit_failure(error, "DCP payload writes aborted")
        error = self._publish_checked()
        if not self._vote(error is None):
            self.state = "unknown"
            _commit_failure(error, "DCP payload publication failed")
        error = self._metadata_checked()
        if not self._vote(error is None):
            self.state = "unknown"
            _commit_failure(error, "DCP metadata publication failed")
        self.state = "committed"

    def _close_checked(self):
        try:
            self.io.close_rank(self.rank, self.root)
        except BaseException as exc:
            return exc
        return None

    def _publish_checked(self):
        try:
            self._publish()
        except BaseException as exc:
            return exc
        return None

    def _metadata_checked(self):
        if self.rank != 0:
            return None
        try:
            desc.atomic_write(
                Path(self.root) / ".metadata",
                pickle.dumps(self.plan.metadata, protocol=4),
            )
        except BaseException as exc:
            return exc
        return None

    @catch
    def abort(self):
        self._vote(False)
        self.io.close_rank(self.rank, self.root)
        for path in self.my_files():
            self.io.discard(self.abspath(path))
        self.state = "aborted"


class _LoadSession(_Session, StorageLoadSession):
    def __init__(self, *args, native_records=None):
        super().__init__(*args)
        self.native_ranges = None
        if native_records is not None:
            self.native_ranges = {
                path: [_native_payload(self.root, *record) for record in records]
                for path, records in native_records.items()
            }
            self.offsets = {
                pos: slot.contiguous_payload_offset
                for pos, (_, slot) in self.index.slots.items()
            }
            return
        metadata = read_metadata(self.root)
        for pos, (path, slot) in self.index.slots.items():
            self.offsets[pos] = _find_offset(
                self.root, self.table, pos, path, slot, metadata
            )

    @catch
    def init_data_accessor(self):
        if self.native_ranges is not None:
            return _NativeAccessor(self, self.native_ranges)
        return StorageDataAccessor(self.index, self.io, session=self)

    @catch
    def finish(self):
        self.io.close_rank(self.rank, self.root)
        self.state = "finished"


def _find_offset(root, table, pos, path, slot, metadata):
    named = next(
        named
        for named in table.named_d_tensors
        if named.global_tensor_id == pos.tensor_id
    )
    box = owned_box(named, "storage", pos.rank)
    if box is None:
        raise ValueError("DCP slot is outside its storage mesh")
    lo, hi = box
    item = _tensor_metadata(metadata, pos.tensor_id)
    meta = global_tensor_meta_of(named)
    if (list(item.size) or [1]) != list(meta.global_shape):
        raise ValueError(f"{pos.tensor_id}: DCP shape mismatch")
    if _CODES.get(item.properties.dtype) != meta.dtype:
        raise ValueError(f"{pos.tensor_id}: DCP dtype mismatch")
    offsets = () if len(item.size) == 0 else tuple(lo)
    sizes = (
        ()
        if len(item.size) == 0
        else tuple(right - left for left, right in zip(lo, hi))
    )
    chunk = next(
        (
            chunk
            for chunk in item.chunks
            if tuple(chunk.offsets) == offsets and tuple(chunk.sizes) == sizes
        ),
        None,
    )
    if chunk is None:
        raise ValueError(
            f"{pos.tensor_id}: DCP chunk geometry cannot represent storage slot {pos.rank}"
        )
    record = metadata.storage_data.get(
        MetadataIndex(pos.tensor_id, torch.Size(offsets))
    )
    if record is None or record.relative_path != path:
        raise ValueError(f"{pos.tensor_id}: DCP storage record does not match {path}")
    return _record_payload(root, record, slot.nbytes, pos.tensor_id)


def _record_payload(root, record, expected, tensor_id):
    path = record.relative_path
    with open(Path(root) / path, "rb") as handle:
        offset, nbytes = _payload_record(_Window(handle, record.offset, record.length))
    if nbytes != expected:
        raise ValueError(
            f"{tensor_id}: DCP payload has {nbytes} bytes, expected {expected}"
        )
    return record.offset + offset


@dataclass(frozen=True)
class _NativePayload:
    start: int
    end: int
    path: str
    offset: int
    shape: tuple[int, ...]
    itemsize: int
    strides: tuple[int, ...] | None
    span: int


def _native_payload(root, tensor_id, start, shape, dtype, record):
    path = os.path.join(root, record.relative_path)
    with open(path, "rb") as handle:
        window = _Window(handle, record.offset, record.length)
        offset, nbytes = _payload_record(window)
        window.seek(0)
        # DCP metadata describes the chunk geometry; torch.save stores its strides.
        with FakeTensorMode():
            tensor = torch.load(
                cast(IO[bytes], window), map_location="cpu", weights_only=True
            )
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.layout != torch.strided
        or tensor.shape != shape
        or tensor.dtype != dtype
    ):
        raise ValueError(f"{tensor_id}: DCP tensor record does not match its metadata")
    itemsize = dtype.itemsize
    storage_offset = tensor.storage_offset() * itemsize
    strides = tuple(stride * itemsize for stride in tensor.stride())
    span = itemsize + sum((dim - 1) * stride for dim, stride in zip(shape, strides))
    if (
        storage_offset < 0
        or any(s < 0 for s in strides)
        or storage_offset + span > nbytes
    ):
        raise ValueError(f"{tensor_id}: DCP tensor view exceeds its serialized storage")
    return _NativePayload(
        start,
        start + prod(shape) * itemsize,
        path,
        record.offset + offset + storage_offset,
        tuple(shape),
        itemsize,
        None if tensor.is_contiguous() else strides,
        span,
    )


def _read_strided(chunk, start, end):
    prefix = chunk.offset % mmap.ALLOCATIONGRANULARITY
    with (
        open(chunk.path, "rb") as handle,
        mmap.mmap(
            handle.fileno(),
            prefix + chunk.span,
            access=mmap.ACCESS_READ,
            offset=chunk.offset - prefix,
        ) as storage,
    ):
        # Treat each element as opaque bytes, including dtypes NumPy cannot represent.
        view = np.ndarray(
            chunk.shape,
            dtype=f"V{chunk.itemsize}",
            buffer=storage,
            offset=prefix,
            strides=chunk.strides,
        )
        return view.flat[start // chunk.itemsize : end // chunk.itemsize].view(np.uint8)


class _NativeAccessor(DTensorDataAccessor):
    """Read each virtual tensor file across its native DCP payload ranges."""

    def __init__(self, session, ranges):
        self.session = session
        self.ranges = ranges

    @catch
    def read_to_buffer(self, global_range, read_to_buffer):
        start = global_range.contiguous_payload_offset
        end = start + global_range.size
        chunks = self.ranges[global_range.file_path]
        first = bisect_right(chunks, start, key=lambda chunk: chunk.end)
        for index in range(first, len(chunks)):
            chunk = chunks[index]
            if chunk.start >= end:
                break
            self._read_chunk(
                chunk, max(start, chunk.start), min(end, chunk.end), read_to_buffer
            )

    def _read_chunk(self, chunk, start, end, callback):
        try:
            data = self._read_payload(chunk, start - chunk.start, end - chunk.start)
        except OSError as exc:
            raise io_error("read", str(exc), chunk.path) from None
        call_buffer_callback(callback, data)

    def _read_payload(self, chunk, start, end):
        if chunk.strides is not None:
            return _read_strided(chunk, start, end)
        return bytearray(
            self.session.io.pread(
                self.session.rank, chunk.path, chunk.offset + start, end - start
            )
        )

    def write_from_buffer(self, global_range, write_from_buffer):
        raise NotImplementedError("native DCP load accessor is read-only")


@dataclass
class DcpBackend:
    components: dict | None = None
    io: FileIO | None = None
    format: str = "distcp"

    def open(
        self, root, storage_table, layout, coordinator=None, *, native_records=None
    ):
        if layout.container_format != "distcp":
            raise ValueError(f"unsupported DCP format {layout.container_format!r}")
        index = build_payload_index(storage_table, layout)
        io_backend = self.io or FileIO()

        @catch
        def begin_save(view, rank):
            assert self.components is not None
            component = self.components[Path(root).name]
            return _SaveSession(
                rank,
                root,
                storage_table,
                layout,
                index,
                io_backend,
                component=component,
                coordinator=coordinator,
            )

        @catch
        def begin_load(view, rank):
            return _LoadSession(
                rank,
                root,
                storage_table,
                layout,
                index,
                io_backend,
                native_records=native_records,
            )

        begin = (
            _collective_begin(begin_save, coordinator) if coordinator else begin_save
        )
        ids = [named.global_tensor_id for named in storage_table.named_d_tensors]
        return StorageDTensorsView(
            ids,
            storage_table,
            StorageDataAccessor(index, io_backend),
            layout,
            begin,
            begin_load,
        )


def read_metadata(root):
    """Read a trusted checkpoint explicitly selected for loading, as PyTorch does."""
    with open(Path(root) / ".metadata", "rb") as handle:
        metadata = pickle.load(handle)
    if not isinstance(metadata, Metadata):
        raise ValueError("invalid native DCP .metadata object")
    if metadata.storage_data is None:
        raise ValueError("native DCP .metadata has no storage_data")
    return metadata


def save(path, state, ctx):
    backend = core._collective(
        ctx,
        "prepare DCP save",
        lambda: DcpBackend(
            {component.name: component for component in state.components}
        ),
    )
    if type(backend) is Err:
        return backend
    return core.save(path, state, ctx, policy=DcpPolicy(), backend=backend.value)


def _native_tensor(tensor_id, item, count, world):
    shape = list(item.size) or [1]
    count = max(count, 1)
    mesh = storage_mesh(world, range(world))
    if count != world:
        mesh = SlicedMesh(mesh, [Slice(0, count)])
    placement = Replicate() if count == 1 else Shard(0)
    dtype = _CODES[item.properties.dtype]
    meta = GlobalTensorMeta(
        shape, dtype, torch.empty((), dtype=item.properties.dtype).element_size()
    )
    return NamedDTensor(tensor_id, NaiveDTensor(meta, [placement], mesh))


def _native_chunks(tensor_id, item):
    shape = list(item.size)
    chunks = sorted(item.chunks, key=lambda chunk: tuple(chunk.offsets))
    error = ValueError(
        f"{tensor_id}: DCP chunk geometry cannot be represented: only contiguous Shard(0) chunks are supported"
    )
    if not shape:
        if len(chunks) != 1 or chunks[0].offsets or chunks[0].sizes:
            raise error
        return chunks
    if prod(shape) == 0 and not chunks:
        return []
    covered, nonempty = 0, []
    for chunk in chunks:
        offsets, sizes = chunk.offsets, chunk.sizes
        if (
            len(offsets) != len(shape)
            or any(offsets[1:])
            or list(sizes[1:]) != shape[1:]
            or len(sizes) != len(shape)
        ):
            raise error
        if sizes[0] == 0:
            continue
        if offsets[0] != covered or sizes[0] < 0 or covered + sizes[0] > shape[0]:
            raise error
        covered += sizes[0]
        nonempty.append(chunk)
    if covered != shape[0]:
        raise error
    return nonempty


def _native_objects(root, metadata, component):
    assert metadata.storage_data is not None
    flat = {}
    for key, item in metadata.state_dict_metadata.items():
        if isinstance(item, BytesStorageMetadata):
            flat[key] = _read_object(root, metadata.storage_data[MetadataIndex(key)])
    if not flat:
        return {
            key: value for key, value in component.objects.items() if key == "aliases"
        }
    tree = unflatten_state_dict(flat, {key: metadata.planner_data[key] for key in flat})
    objects = {key: value for key, value in tree.items() if key != "state"}
    for fqn, fields in tree.get("state", {}).items():
        objects.update(
            {f"state.{fqn}.{field}": value for field, value in fields.items()}
        )
    return objects


def _read_object(root, record):
    desc._safe_relative(record.relative_path)
    if record.length > 16 * 1024**2:
        raise ValueError("DCP non-tensor record exceeds the 16 MiB safety bound")
    with open(Path(root) / record.relative_path, "rb") as handle:
        return torch.load(
            cast(IO[bytes], _Window(handle, record.offset, record.length)),
            weights_only=False,
            map_location="cpu",
        )


def _tensor_metadata(metadata, key):
    item = metadata.state_dict_metadata[key]
    if not isinstance(item, TensorStorageMetadata):
        raise ValueError(f"{key}: native DCP tensor key contains non-tensor metadata")
    return item


def _validate_native_keys(metadata, wanted):
    assert metadata.storage_data is not None
    owners = {key: key for key in wanted}
    owners.update(
        {alias: key for key, named in wanted.items() for alias in named.alias}
    )
    tensor_keys = {
        key
        for key, item in metadata.state_dict_metadata.items()
        if isinstance(item, TensorStorageMetadata)
    }
    unexpected = tensor_keys - owners.keys()
    if unexpected:
        raise ValueError(f"native DCP unexpected key {sorted(unexpected)[0]}")
    records = defaultdict(list)
    for index, record in metadata.storage_data.items():
        if index.fqn in tensor_keys:
            records[index.fqn].append(
                (
                    tuple(index.offset),
                    record.relative_path,
                    record.offset,
                    record.length,
                )
            )
    aliases = {}
    for key in sorted(tensor_keys):
        _validate_native_alias(metadata, key, owners[key], records[key], aliases)


def _validate_native_alias(metadata, key, owner, records, aliases):
    item = _tensor_metadata(metadata, key)
    canonical = _tensor_metadata(metadata, owner)
    if (
        item.size != canonical.size
        or item.properties.dtype != canonical.properties.dtype
    ):
        raise ValueError(f"native DCP tied parameter alias {key} differs from {owner}")
    # GDCP-native aliases point to the same records. Other DCP writers may
    # serialize tied names independently, with no explicit alias information.
    if records and aliases.setdefault(tuple(sorted(records)), owner) != owner:
        raise ValueError(f"native DCP tied parameter aliases differ for {key}")


def read_component(root, component):
    """Build the storage table from native metadata, without a GDCP manifest."""
    metadata = read_metadata(root)
    wanted = {
        named.global_tensor_id: named for named in component.table.named_d_tensors
    }
    items = [
        (key, _tensor_metadata(metadata, key))
        for key in wanted
        if key in metadata.state_dict_metadata
    ]
    if len(items) != len(wanted):
        raise ValueError(
            f"{component.name}: native DCP is missing keys {sorted(wanted.keys() - metadata.state_dict_metadata.keys())}"
        )
    _validate_native_keys(metadata, wanted)
    chunks = {key: _native_chunks(key, item) for key, item in items}
    world = max([len(parts) for parts in chunks.values()] + [1])
    table = DTensorTable(
        [
            replace(
                _native_tensor(key, item, len(chunks[key]), world),
                alias=wanted[key].alias,
            )
            for key, item in items
        ]
    )
    # GDTensor plans over regular slots. One virtual file per tensor puts those
    # slots in row order so byte ranges can cross the actual DCP record boundaries.
    paths = {key: f"tensor-{order}.distcp" for order, (key, _) in enumerate(items)}
    layout = build_layout(
        table,
        world,
        "distcp",
        lambda order, rank, named: paths[named.global_tensor_id],
    )
    records = {
        paths[key]: _native_records(metadata, key, item, chunks[key])
        for key, item in items
    }
    objects = (
        desc.read_objects(Path(root) / "objects.json")
        if (Path(root) / "objects.json").exists()
        else _native_objects(root, metadata, component)
    )
    logical = {key: [] for key, item in items if len(item.size) == 0}
    return table, layout, objects, logical, world, records


def _native_records(metadata, key, item, chunks):
    assert metadata.storage_data is not None
    row_bytes = prod(item.size[1:]) * item.properties.dtype.itemsize
    if row_bytes == 0:
        return []
    records = []
    for chunk in chunks:
        record = metadata.storage_data[MetadataIndex(key, chunk.offsets)]
        desc._safe_relative(record.relative_path)
        start = chunk.offsets[0] if item.size else 0
        records.append(
            (key, start * row_bytes, chunk.sizes, item.properties.dtype, record)
        )
    return records


def _prepare_native(path, state, ctx):
    core._validate_context(ctx)
    core._validate_state(state)
    if state.rank_state:
        key = min(state.rank_state)
        raise ValueError(
            f"native DCP has no GDCP per-rank extra_state (requested {key!r})"
        )
    prepared = []
    for component in state.components:
        root = Path(path) / component.name
        table, layout, objects, logical, world, records = read_component(
            root, component
        )
        saved = desc.describe_component(table, layout, world, logical_shapes=logical)
        core._strict_tensors(component, saved, False)
        compute = ComputeDTensorsView(
            [named.global_tensor_id for named in component.table.named_d_tensors],
            component.table,
            component.accessor,
        )
        storage = DcpBackend().open(
            root, table, layout, ctx.coordinator, native_records=records
        )
        core._require(compute.check_constraints())
        core._require(storage.check_constraints())
        prepared.append((component, compute, storage, objects))
    return prepared


def _has_manifest(path):
    core._require_not_pending(path)
    return (Path(path) / desc.MANIFEST_NAME).exists()


def load(path, state, ctx, *, _preflight=None):
    manifest = core._collective_choice(
        ctx, "DCP load format", lambda: _has_manifest(path), (True, False)
    )
    if type(manifest) is Err:
        return manifest
    if manifest.value:
        return core.load(path, state, ctx, backend=DcpBackend(), _preflight=_preflight)
    agreed = core._agree_components(state, ctx)
    if type(agreed) is Err:
        return agreed
    prepared = core._collective(
        ctx, "prepare native DCP load", lambda: _prepare_native(path, state, ctx)
    )
    if type(prepared) is Err:
        return prepared
    return core._load_components(prepared.value, ctx, {}, _preflight)
