"""Checkpoint entry ownership from recorded metadata, without reading payloads."""

import json
import os
import struct
from pathlib import Path

from . import desc
from ._dcp_metadata import read_storage_records

_INDEX = "model.safetensors.index.json"
_MARKER = ".gdtensor_committed"
_HF_FIXED = {_INDEX, _MARKER, "model.safetensors", "optimizer.json", "config.json"}
_ITEMSIZE = {
    "BOOL": 1,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


def _json(path):
    return json.loads(path.read_bytes())


def _name(value):
    desc._safe_relative(value)
    return Path(value)


def _rank_files(rank_state):
    files = set()
    for name, info in rank_state.items():
        directory = _name(name)
        files.update(
            directory / _name(info["pattern"].format(rank=rank))
            for rank in info["ranks"]
        )
    return files


def _manifest_files(root):
    saved = desc.read_manifest(root)
    files = {Path(desc.MANIFEST_NAME), Path(desc.PENDING_NAME)}
    directories = {_name(name) for name in saved.components} | {
        _name(name) for name in saved.rank_state
    }
    for name, component in saved.components.items():
        files.update(_component_files(root / name, component, Path(name)))
    files.update(_rank_files(saved.rank_state))
    return files, directories


def _component_files(root, component, prefix):
    files = {prefix / _name(name) for name in component["files"]}
    fixed = {_MARKER, _INDEX} if component["format"] == "safetensors" else {".metadata"}
    files.update(prefix / name for name in fixed if os.path.lexists(root / name))
    if component["objects"] is not None:
        files.add(prefix / _name(component["objects"]))
    return files


def _dcp_files(root):
    records = read_storage_records(root)
    files = {Path(".metadata")}
    if os.path.lexists(root / "objects.json"):
        files.add(Path("objects.json"))
    files.update(_name(record.relative_path) for record in records)
    return files


def _hf_files(root):
    files = {_name(name) for name in _HF_FIXED if os.path.lexists(root / name)}
    if (root / _INDEX).is_file():
        weight_map = _json(root / _INDEX)["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("HF weight_map must be a nonempty mapping")
        files.update(_name(name) for name in weight_map.values())
    marker = _json(root / _MARKER) if (root / _MARKER).is_file() else {}
    if marker:
        if marker.get("format") != "safetensors" or not isinstance(
            marker.get("files"), list
        ):
            raise ValueError("invalid safetensors completion marker")
        files.update(_name(name) for name in marker["files"])
        files.update(_rank_files(marker.get("rank_state", {})))
    return files


def _recorded_inventory(root):
    root = Path(root)
    if (root / desc.MANIFEST_NAME).is_file():
        return _manifest_files(root)
    if (root / _INDEX).is_file() or (root / "model.safetensors").is_file():
        if not (root / _INDEX).is_file():
            _header_complete(root / "model.safetensors")
        return _hf_files(root), set()
    if (root / ".metadata").is_file():
        return _dcp_files(root), set()
    directories = {
        Path(name)
        for name in ("model", "optimizer")
        if (root / name / ".metadata").is_file()
    }
    files = set()
    for directory in directories:
        files.update(directory / name for name in _dcp_files(root / directory))
    return files, directories


def _inventory(root):
    try:
        return _recorded_inventory(root)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ):
        # Metadata with no recognizable checkpoint signature is foreign too.
        # Any new file at the same name must therefore fail collision checking.
        return set(), set()


def owned_files(root):
    """Return exact recorded files plus fixed files of the recognized format.

    Invalid metadata and directories without recognizable checkpoint metadata
    own nothing; every existing entry there must be preserved as foreign.
    """
    return _inventory(root)[0]


def owned_directories(root):
    """Return structural checkpoint directories, never their foreign contents."""
    files, directories = _inventory(root)
    for path in files:
        directories.update(parent for parent in path.parents if parent != Path("."))
    return directories


def _hf_complete(root):
    files = _hf_files(root)
    weight_map = (
        _json(root / _INDEX)["weight_map"] if (root / _INDEX).is_file() else None
    )
    payloads = (
        set(weight_map.values()) if weight_map is not None else {"model.safetensors"}
    )
    actual = {}
    for name in payloads:
        actual.update({key: name for key in _header_complete(root / name)})
    if weight_map is not None and actual != weight_map:
        return False
    marker = _json(root / _MARKER) if (root / _MARKER).is_file() else None
    if marker is not None and set(marker["files"]) != payloads:
        return False
    return all((root / name).is_file() for name in files)


def _header_complete(path):
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("incomplete safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if length > 100_000_000:
            raise ValueError("safetensors header exceeds 100 MB")
        header = json.loads(handle.read(length))
    entries = {name: value for name, value in header.items() if name != "__metadata__"}
    end = max((_entry_end(value) for value in entries.values()), default=0)
    if path.stat().st_size != 8 + length + end:
        raise ValueError("incomplete safetensors file")
    return entries


def _entry_end(value):
    size = _ITEMSIZE[value["dtype"]]
    for dim in value["shape"]:
        if type(dim) is not int or dim < 0:
            raise ValueError("invalid safetensors shape")
        size *= dim
    start, end = value["data_offsets"]
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end - start != size
    ):
        raise ValueError("invalid safetensors byte range")
    return end


def _dcp_complete(root):
    return all(
        (root / _name(record.relative_path)).stat().st_size
        >= record.offset + record.length
        for record in read_storage_records(root)
    )


def _component_complete(root, saved):
    files = saved["files"]
    if not all(
        (root / name).is_file() and (root / name).stat().st_size == info["bytes"]
        for name, info in files.items()
    ):
        return False
    if saved["objects"] is not None and not (root / saved["objects"]).is_file():
        return False
    if not files:
        return True
    if saved["format"] == "safetensors":
        marker = _json(root / _MARKER)
        return set(marker["files"]) == set(files)
    return _dcp_complete(root)


def _manifest_complete(root):
    saved = desc.read_manifest(root)
    if not all(
        _component_complete(root / name, component)
        for name, component in saved.components.items()
    ):
        return False
    return all((root / name).is_file() for name in _rank_files(saved.rank_state))


def complete(root):
    """Conservatively identify a complete generation using metadata and stats."""
    root = Path(root)
    try:
        if not root.is_dir() or (root / desc.PENDING_NAME).exists():
            return False
        return _complete(root)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ):
        return False


def _complete(root):
    if (root / desc.MANIFEST_NAME).is_file():
        return _manifest_complete(root)
    if (root / _INDEX).is_file() or (root / "model.safetensors").is_file():
        return _hf_complete(root)
    if (root / ".metadata").is_file():
        return _dcp_complete(root)
    roots = [
        root / name
        for name in ("model", "optimizer")
        if (root / name / ".metadata").is_file()
    ]
    return bool(roots) and all(_dcp_complete(directory) for directory in roots)
