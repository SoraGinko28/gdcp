"""Checkpoint reader using only JSON, stdlib and the safetensors library."""

import json
from math import prod
from pathlib import Path

import safetensors

_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
}


def independent_reader(root):
    """Reassemble logical tensor shape, dtype and every byte without GDCP/torch."""
    root = Path(root)
    manifest = json.loads((root / "gdcp_manifest.json").read_text())
    result, counts = {}, {}
    for name, component in manifest["components"].items():
        result[name], counts[name] = _read_component(root / name, component)
    return result, counts, manifest


def _read_component(root, component):
    tensors = {spec["id"]: spec for spec in component["tensors"]}
    result = {
        key: bytearray(prod(spec["shape"]) * spec["itemsize"])
        for key, spec in tensors.items()
    }
    counts = {key: bytearray(len(data)) for key, data in result.items()}
    for path in component["files"]:
        entries = dict(safetensors.deserialize((root / path).read_bytes()))
        _copy_entries(entries, tensors, result, counts)
    assert all(all(value == 1 for value in coverage) for coverage in counts.values())
    records = {key: _logical_record(spec, result[key]) for key, spec in tensors.items()}
    return records, counts


def _logical_record(spec, data):
    shape = spec.get("logical_shape", spec["shape"])
    assert prod(shape) == prod(spec["shape"])
    return {"data": data.hex(), "shape": shape, "dtype": _DTYPES[spec["dtype"]]}


def _copy_entries(entries, tensors, result, counts):
    for entry, payload in entries.items():
        key, separator, slot = entry.rpartition("/slot")
        key = key if separator else entry
        spec = tensors[key]
        data = bytes(payload["data"])
        offset, shape = _slot_span(spec, int(slot)) if separator else (0, spec["shape"])
        end = offset + len(data)
        assert payload["shape"] == shape, (entry, payload["shape"], shape)
        assert payload["dtype"] == spec["dtype"]
        assert len(data) == prod(shape) * spec["itemsize"]
        assert 0 <= offset <= end <= len(result[key]), entry
        assert all(value == 0 for value in counts[key][offset:end]), entry
        result[key][offset:end] = data
        counts[key][offset:end] = bytes([1]) * len(data)


def _slot_span(spec, rank):
    placements = spec["placements"]
    if all("replicate" in placement for placement in placements):
        return 0, spec["shape"]
    assert len(placements) == 1 and placements[0] == {"shard": 0}
    ranks = spec["mesh"]["ranks"]
    chunk = (spec["shape"][0] + len(ranks) - 1) // len(ranks)
    start = min(ranks.index(rank) * chunk, spec["shape"][0])
    rows = min(chunk, spec["shape"][0] - start)
    offset = start * prod(spec["shape"][1:]) * spec["itemsize"]
    return offset, [rows, *spec["shape"][1:]]
