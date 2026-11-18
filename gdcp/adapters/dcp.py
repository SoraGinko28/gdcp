"""Native, deterministic per-rank DCP records filled by GDTensor byte I/O.

Each tensor chunk is a standard ``torch.save`` ZIP record. ``skip_data`` creates
only its small envelope; GDTensor writes the payload directly into the reserved
record. Fake tensors avoid allocating either the local or global tensor here.
"""

import io
import os
import pickle
import struct
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from pathlib import Path
from typing import IO, cast

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
from gdtensor_conn._result import catch, invalid
from gdtensor_conn.access import StorageDataAccessor
from gdtensor_conn.iobackend import FileIO
from gdtensor_conn.models import build_layout

from .. import core, desc
from ..backend import _collective_begin
from ..policy import PerRankPolicy, storage_mesh

_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
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
        self.io.close_rank(self.rank, self.root)
        if not self._vote(True):
            raise ValueError("another rank aborted DCP payload writes")
        error = self._publish_checked()
        if not self._vote(error is None):
            self.state = "unknown"
            _commit_failure(error, "DCP payload publication failed")
        error = self._metadata_checked()
        if not self._vote(error is None):
            self.state = "unknown"
            _commit_failure(error, "DCP metadata publication failed")
        self.state = "committed"

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
        self.io.close_rank(self.rank, self.root)
        self._vote(False)
        for path in self.my_files():
            self.io.discard(self.abspath(path))
        self.state = "aborted"


class _LoadSession(_Session, StorageLoadSession):
    def __init__(self, *args):
        super().__init__(*args)
        metadata = read_metadata(self.root)
        for pos, (path, slot) in self.index.slots.items():
            self.offsets[pos] = _find_offset(
                self.root, self.table, pos, path, slot, metadata
            )

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
    item = metadata.state_dict_metadata[pos.tensor_id]
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
    with open(Path(root) / path, "rb") as handle:
        offset, nbytes = _payload_record(_Window(handle, record.offset, record.length))
    if nbytes != slot.nbytes:
        raise ValueError(
            f"{pos.tensor_id}: DCP payload has {nbytes} bytes, expected {slot.nbytes}"
        )
    return record.offset + offset


@dataclass
class DcpBackend:
    components: dict | None = None
    io: FileIO | None = None
    format: str = "distcp"

    def open(self, root, storage_table, layout, coordinator=None):
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
            return _LoadSession(rank, root, storage_table, layout, index, io_backend)

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
    components = {component.name: component for component in state.components}
    return core.save(
        path, state, ctx, policy=DcpPolicy(), backend=DcpBackend(components)
    )


def _native_tensor(tensor_id, item, world):
    shape = list(item.size) or [1]
    chunks = sorted(item.chunks, key=lambda chunk: tuple(chunk.offsets))
    _validate_chunks(tensor_id, shape, chunks, scalar=len(item.size) == 0)
    count = max(len(chunks), 1)
    mesh = storage_mesh(world, range(world))
    if count != world:
        mesh = SlicedMesh(mesh, [Slice(0, count)])
    placement = Replicate() if count == 1 else Shard(0)
    dtype = _CODES[item.properties.dtype]
    meta = GlobalTensorMeta(
        shape, dtype, torch.empty((), dtype=item.properties.dtype).element_size()
    )
    return NamedDTensor(tensor_id, NaiveDTensor(meta, [placement], mesh))


def _validate_chunks(tensor_id, shape, chunks, scalar=False):
    if prod(shape) == 0 and not chunks:
        return
    count = len(chunks)
    width = (shape[0] + count - 1) // count if count else 0
    for index, chunk in enumerate(chunks):
        lo, sizes = (
            [index * width] + [0] * (len(shape) - 1),
            [min(width, shape[0] - index * width)] + shape[1:],
        )
        if scalar:
            lo, sizes = [], []
        if list(chunk.offsets) != lo or list(chunk.sizes) != sizes:
            raise ValueError(
                f"{tensor_id}: DCP chunk geometry cannot be represented: only canonical contiguous Shard(0) chunks are supported"
            )


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
    world = max([len(item.chunks) for key, item in items] + [1])
    table = DTensorTable([_native_tensor(key, item, world) for key, item in items])
    paths = _native_paths(metadata, table)
    layout = build_layout(
        table,
        world,
        "distcp",
        lambda order, rank, named: paths[(named.global_tensor_id, rank)],
    )
    objects = (
        desc.read_objects(Path(root) / "objects.json")
        if (Path(root) / "objects.json").exists()
        else _native_objects(root, metadata, component)
    )
    logical = {key: [] for key, item in items if len(item.size) == 0}
    return table, layout, objects, logical, world


def _native_paths(metadata, table):
    assert metadata.storage_data is not None
    paths = {}
    for named in table.named_d_tensors:
        chunks = sorted(
            metadata.state_dict_metadata[named.global_tensor_id].chunks,
            key=lambda chunk: tuple(chunk.offsets),
        )
        for rank, chunk in enumerate(chunks):
            record = metadata.storage_data[
                MetadataIndex(named.global_tensor_id, chunk.offsets)
            ]
            desc._safe_relative(record.relative_path)
            paths[(named.global_tensor_id, rank)] = record.relative_path
    return paths


def _prepare_native(path, state, ctx):
    core._validate_context(ctx)
    core._validate_state(state)
    prepared = []
    for component in state.components:
        root = Path(path) / component.name
        table, layout, objects, logical, world = read_component(root, component)
        saved = desc.describe_component(table, layout, world, logical_shapes=logical)
        core._strict_tensors(component, saved, False)
        compute = ComputeDTensorsView(
            [named.global_tensor_id for named in component.table.named_d_tensors],
            component.table,
            component.accessor,
        )
        storage = DcpBackend().open(root, table, layout, ctx.coordinator)
        core._require(compute.check_constraints())
        core._require(storage.check_constraints())
        prepared.append((component, compute, storage, objects))
    return prepared


def load(path, state, ctx):
    pending = core._collective(
        ctx, "prepare DCP load", lambda: core._require_not_pending(path)
    )
    if type(pending) is Err:
        return pending
    if (Path(path) / desc.MANIFEST_NAME).exists():
        return core.load(path, state, ctx, backend=DcpBackend())
    if state.rank_state:
        return core._error(
            "prepare native DCP load",
            ValueError("native DCP has no GDCP per-rank extra_state"),
        )
    prepared = core._collective(
        ctx, "prepare native DCP load", lambda: _prepare_native(path, state, ctx)
    )
    if type(prepared) is Err:
        return prepared
    return core._load_components(prepared.value, ctx)
