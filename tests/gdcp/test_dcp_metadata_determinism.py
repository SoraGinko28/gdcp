"""Equal native DCP metadata publishes identical bytes despite object sharing."""

import pickle

import pytest

torch = pytest.importorskip("torch")

from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)

from gdcp.adapters import dcp


def _copy_string(value):
    return bytearray(value.encode()).decode()


def _metadata(shared, reverse):
    properties = TensorProperties(dtype=torch.float32)
    tensors, planner, storage = {}, {}, {}
    fields = ["exp_avg", "exp_avg_sq", "master_param", "step"]
    for field in reversed(fields) if reverse else fields:
        field = field if shared else _copy_string(field)
        key = f"state.weight.{field}"
        item = TensorStorageMetadata(
            properties if shared else TensorProperties(dtype=torch.float32),
            torch.Size([4]),
            [ChunkStorageMetadata(torch.Size([0]), torch.Size([4]))],
        )
        tensors[key] = item
        planner[key] = ("state", "weight", field)
        storage[MetadataIndex(key, torch.Size([0]))] = _StorageInfo(
            "__0_0.distcp", fields.index(field) * 128, 128
        )
        planner[f"schema.{field}"] = (
            "schema",
            "fields",
            "weight",
            field if shared else _copy_string(field),
        )
    return Metadata(tensors, planner_data=planner, storage_data=storage)


def _publish(root, metadata):
    root.mkdir()
    session = dcp._SaveSession.__new__(dcp._SaveSession)
    session.rank, session.root = 0, str(root)
    session.plan = dcp._Plan({}, {}, {}, metadata)
    assert dcp._SaveSession._metadata_checked(session) is None
    return (root / ".metadata").read_bytes()


@pytest.mark.parametrize("reverse", [False, True])
def test_metadata_bytes_ignore_equal_string_identity_and_mapping_order(
    tmp_path, reverse
):
    first = _metadata(shared=True, reverse=False)
    restored = _metadata(shared=False, reverse=reverse)
    assert first == restored
    assert pickle.dumps(first, protocol=4) != pickle.dumps(restored, protocol=4)
    original = _publish(tmp_path / "original", first)
    returned = _publish(tmp_path / "returned", restored)
    assert original == returned
    assert pickle.loads(original) == first
    native = FileSystemReader(tmp_path / "original").read_metadata()
    assert native.state_dict_metadata == first.state_dict_metadata
    assert native.planner_data == first.planner_data
    assert native.storage_data == first.storage_data
