"""Read the storage records of a native DCP metadata pickle as data only."""

import pickle
from dataclasses import dataclass
from typing import BinaryIO, NamedTuple

_METADATA = "torch.distributed.checkpoint.metadata"
_FILESYSTEM = "torch.distributed.checkpoint.filesystem"
_GLOBALS = {
    _METADATA: {
        "BytesStorageMetadata",
        "ChunkStorageMetadata",
        "Metadata",
        "MetadataIndex",
        "StorageMeta",
        "TensorProperties",
        "TensorStorageMetadata",
        "_MEM_FORMAT_ENCODING",
    },
    _FILESYSTEM: {"_StorageInfo"},
    "pathlib": {"PosixPath", "WindowsPath", "PurePosixPath", "PureWindowsPath"},
    "torch.serialization": {"_get_layout"},
    "torch": set(
        """Size bcomplex32 bfloat16 bits16 bits1x8 bits2x4 bits4x2 bits8 bool
        complex128 complex32 complex64 float16 float32 float4_e2m1fn_x2 float64
        float8_e4m3fn float8_e4m3fnuz float8_e5m2 float8_e5m2fnuz float8_e8m0fnu
        int1 int2 int3 int4 int5 int6 int7 int8 int16 int32 int64 qint32 qint8
        quint2x4 quint4x2 quint8 uint1 uint2 uint3 uint4 uint5 uint6 uint7 uint8
        uint16 uint32 uint64""".split()
    ),
}


@dataclass(frozen=True, slots=True)
class _Global:
    module: str
    name: str


@dataclass(eq=False, slots=True)
class _Record:
    kind: _Global
    args: tuple
    state: object = None


def _load_record(parser: object, /) -> None:
    assert isinstance(parser, _MetadataParser)
    args = parser.stack.pop()
    kind = parser.stack[-1]
    if type(kind) is not _Global or type(args) is not tuple:
        raise pickle.UnpicklingError("invalid DCP metadata record")
    parser.stack[-1] = _Record(kind, args)


def _load_state(parser: object, /) -> None:
    assert isinstance(parser, _MetadataParser)
    state = parser.stack.pop()
    record = parser.stack[-1]
    if type(record) is not _Record or type(state) not in (dict, tuple):
        raise pickle.UnpicklingError("invalid DCP metadata record state")
    record.state = state


class _MetadataParser(pickle._Unpickler):
    # The Python unpickler initializes this stack in load().
    stack: list[object]

    # Globals stay symbolic, even for torch classes with custom pickle hooks.
    def find_class(self, module, name):
        if name not in _GLOBALS.get(module, ()):
            raise pickle.UnpicklingError(
                f"unsupported DCP metadata global {module}.{name}"
            )
        return _Global(module, name)

    def get_extension(self, code):
        # The pickle extension cache can resolve globals without find_class.
        raise pickle.UnpicklingError(
            "pickle extensions are unsupported in DCP metadata"
        )

    # Replace the opcodes that construct objects or apply their state.
    dispatch = pickle._Unpickler.dispatch.copy()
    dispatch[pickle.NEWOBJ[0]] = dispatch[pickle.REDUCE[0]] = _load_record
    dispatch[pickle.BUILD[0]] = _load_state
    del dispatch[pickle.INST[0]], dispatch[pickle.OBJ[0]], dispatch[pickle.NEWOBJ_EX[0]]


class _StorageRecord(NamedTuple):
    relative_path: str
    offset: int
    length: int


class _MetadataInput:
    def __init__(self, handle: BinaryIO):
        self._handle = handle

    def readline(self) -> bytes:
        return self._handle.readline()

    def read(self, size: int) -> bytes:
        data = self._handle.read(size)
        # The Python unpickler can reach STOP after a short frame read.
        if len(data) != size:
            raise pickle.UnpicklingError("truncated native DCP metadata")
        return data


def _state(record, module, name):
    if (
        type(record) is not _Record
        or record.kind != _Global(module, name)
        or type(record.state) is not dict
    ):
        raise ValueError(f"invalid native DCP {name} record")
    return record.state


def _storage_record(record):
    state = _state(record, _FILESYSTEM, "_StorageInfo")
    path, offset, length = state["relative_path"], state["offset"], state["length"]
    if type(path) is not str or any(
        type(value) is not int or value < 0 for value in (offset, length)
    ):
        raise ValueError("invalid native DCP storage range")
    return _StorageRecord(path, offset, length)


def _read_metadata_state(root):
    with open(root / ".metadata", "rb") as handle:
        metadata = _MetadataParser(_MetadataInput(handle)).load()
    state = _state(metadata, _METADATA, "Metadata")
    if (
        type(state["state_dict_metadata"]) is not dict
        or type(state["storage_data"]) is not dict
    ):
        raise ValueError("invalid native DCP metadata mappings")
    return state


def read_storage_records(root):
    try:
        storage = _read_metadata_state(root)["storage_data"]
        for index in storage:
            _state(index, _METADATA, "MetadataIndex")
        return [_storage_record(record) for record in storage.values()]
    except Exception as exc:
        # Ownership and recovery treat all unreadable or unsupported metadata alike.
        raise ValueError("unrecognized native DCP metadata") from exc
