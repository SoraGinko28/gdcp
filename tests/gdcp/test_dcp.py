"""Genuine per-rank DCP records, deterministic bytes and native interchange."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local

import numpy as np
import pytest

from gdcp import core, desc
from gdcp.state import RankContext
from gdtensor.builder import Err
from gdtensor_conn.coord import ThreadCoordinator
from gdtensor_conn.iobackend import FileIO
from tests.support.gdcp import assert_ok, assert_values, collective, numpy_checkpoint

pytestmark = pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)


def _save(path, state, ctx, **options):
    return core.save(path, state, ctx, **options)


def _dcp_save(path, fixture):
    from gdcp.adapters.dcp import DcpBackend, DcpPolicy

    components = {
        component.name: component for component in fixture.states[0].components
    }
    return collective(
        _save, path, fixture, policy=DcpPolicy(), backend=DcpBackend(components)
    )


def _hashable_files(path):
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in Path(path).rglob("*")
        if file.is_file()
    }


@pytest.mark.parametrize("source,target", [(1, 4), (2, 3), (3, 2), (4, 1)])
def test_dcp_roundtrip_reshard_and_native_reader(tmp_path, source, target):
    torch = pytest.importorskip("torch")
    from torch.distributed.checkpoint import load as native_load
    from gdcp.adapters.dcp import DcpBackend

    original = numpy_checkpoint(source, initialized=True)
    path = tmp_path / "source"
    assert_ok(_dcp_save(path, original))
    restored = numpy_checkpoint(target, initialized=False)
    assert_ok(collective(core.load, path, restored, backend=DcpBackend()))
    assert_values(restored, original)
    dense = _dense_state(original, torch)
    native_load(dense, checkpoint_id=path / "model", no_dist=True)
    _assert_dense(dense, original, torch)


def _dense_state(fixture, torch):
    codes = {"F32": torch.float32, "BF16": torch.bfloat16, "I64": torch.int64}
    return {
        named.global_tensor_id: torch.empty(
            ()
            if named.global_tensor_id == "step"
            else named.d_tensor.global_tensor_meta.global_shape,
            dtype=codes[named.d_tensor.global_tensor_meta.dtype],
        )
        for named in fixture.table.named_d_tensors
    }


def _assert_dense(state, fixture, torch):
    for key, value in state.items():
        expected = fixture.values["model"][key].tobytes()
        actual = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        assert actual == expected, key


def test_dcp_save_twice_and_reshard_return_are_byte_identical(tmp_path):
    from gdcp.adapters.dcp import DcpBackend

    original = numpy_checkpoint(2, initialized=True)
    first, second, final = tmp_path / "first", tmp_path / "second", tmp_path / "final"
    assert_ok(_dcp_save(first, original))
    assert_ok(_dcp_save(second, original))
    assert _hashable_files(first) == _hashable_files(second)
    middle = numpy_checkpoint(3, initialized=False)
    assert_ok(collective(core.load, first, middle, backend=DcpBackend()))
    assert_ok(_dcp_save(tmp_path / "middle", middle))
    returned = numpy_checkpoint(2, initialized=False)
    assert_ok(
        collective(core.load, tmp_path / "middle", returned, backend=DcpBackend())
    )
    assert_ok(_dcp_save(final, returned))
    assert _hashable_files(first) == _hashable_files(final)


def test_native_dcp_load_without_manifest(tmp_path):
    torch = pytest.importorskip("torch")
    from torch.distributed.checkpoint import save as native_save
    from gdcp.adapters.dcp import load

    source = numpy_checkpoint(1, initialized=True)
    dense = _dense_state(source, torch)
    for key, value in dense.items():
        raw = np.frombuffer(
            source.values["model"][key].tobytes(), dtype=np.uint8
        ).copy()
        value.reshape(-1).view(torch.uint8).copy_(torch.from_numpy(raw))
    dense["param_groups"] = source.states[0].components[0].objects["param_groups"]
    native_save(dense, checkpoint_id=tmp_path / "model", no_dist=True)
    restored = numpy_checkpoint(4, initialized=False)

    def _load(path, state, ctx, **kwargs):
        return load(path, state, ctx)

    assert_ok(collective(_load, tmp_path, restored))
    assert_values(restored, source)


@pytest.mark.parametrize("failure", ["rank_state", "lookup", "mixed_formats"])
def test_native_dcp_preflight_failure_votes_on_every_rank(
    tmp_path, monkeypatch, failure
):
    from gdcp.adapters.dcp import load

    assert_ok(_dcp_save(tmp_path, numpy_checkpoint(2, initialized=True)))
    manifest = tmp_path / desc.MANIFEST_NAME
    manifest.unlink()
    target = numpy_checkpoint(2, initialized=False)
    rank_context = local()
    exists = Path.exists

    def rank_local_exists(path):
        if path == manifest and rank_context.rank == 0:
            if failure == "lookup":
                raise OSError("injected manifest lookup failure")
            return failure == "mixed_formats"
        return exists(path)

    if failure == "rank_state":
        target.states[0].rank_state["extra_state"] = b""
    coordinator = ThreadCoordinator(2, timeout=2)
    monkeypatch.setattr(Path, "exists", rank_local_exists)

    def load_rank(rank):
        rank_context.rank = rank
        return load(tmp_path, target.states[rank], RankContext(rank, 2, coordinator))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(load_rank, range(2)))
    assert all(type(result) is Err for result in results), results
    expected = {
        "rank_state": "native DCP has no GDCP per-rank extra_state",
        "lookup": "injected manifest lookup failure",
        "mixed_formats": "disagree on DCP load format",
    }
    assert expected[failure] in results[0].value.msg


def test_dcp_save_close_failure_votes_on_every_rank(tmp_path, monkeypatch):
    from gdcp.adapters.dcp import save

    root = tmp_path / "checkpoint"
    fixture = numpy_checkpoint(2, initialized=True)
    coordinator = ThreadCoordinator(2, timeout=2)
    close = FileIO.close_rank

    def close_rank(io, rank, prefix=None):
        close(io, rank, prefix)
        if rank == 0:
            raise OSError("injected DCP close failure")

    monkeypatch.setattr(FileIO, "close_rank", close_rank)

    def save_rank(rank):
        return save(root, fixture.states[rank], RankContext(rank, 2, coordinator))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save_rank, range(2)))
    assert all(type(result) is Err for result in results), results
    assert "injected DCP close failure" in results[0].value.msg
    assert not root.exists()


def test_invalid_dcp_component_map_votes_on_every_rank(tmp_path):
    from gdcp.adapters.dcp import save

    fixture = numpy_checkpoint(2, initialized=True)
    fixture.states[0].components = [object()]
    coordinator = ThreadCoordinator(2, timeout=2)

    def save_rank(rank):
        return save(
            tmp_path / "checkpoint",
            fixture.states[rank],
            RankContext(rank, 2, coordinator),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save_rank, range(2)))
    assert all(type(result) is Err for result in results), results
    assert "name" in results[0].value.msg
    assert not list(tmp_path.iterdir())


def test_noncanonical_native_chunk_geometry_is_rejected(tmp_path):
    import pickle
    from torch.distributed.checkpoint.metadata import (
        ChunkStorageMetadata,
        TensorStorageMetadata,
    )
    import torch
    from gdcp.adapters.dcp import read_component, read_metadata

    fixture = numpy_checkpoint(2, initialized=True)
    assert_ok(_dcp_save(tmp_path, fixture))
    metadata = read_metadata(tmp_path / "model")
    item = metadata.state_dict_metadata["weight"]
    assert isinstance(item, TensorStorageMetadata)
    item.chunks = [
        ChunkStorageMetadata(torch.Size([0, 0]), torch.Size([10, 2])),
        ChunkStorageMetadata(torch.Size([0, 2]), torch.Size([10, 2])),
    ]
    (tmp_path / "model" / ".metadata").write_bytes(pickle.dumps(metadata))
    with pytest.raises(ValueError, match="chunk geometry cannot be represented"):
        read_component(tmp_path / "model", fixture.states[0].components[0])


def test_tied_aliases_share_one_native_dcp_payload(tmp_path):
    import pickle
    from gdcp.adapters.dcp import read_metadata

    fixture = numpy_checkpoint(2, initialized=True)
    fixture.table.named_d_tensors[0].alias = ["tied.weight"]
    assert_ok(_dcp_save(tmp_path, fixture))
    metadata = read_metadata(tmp_path / "model")
    original = [
        (index.offset, storage)
        for index, storage in metadata.storage_data.items()
        if index.fqn == "weight"
    ]
    aliases = [
        (index.offset, storage)
        for index, storage in metadata.storage_data.items()
        if index.fqn == "tied.weight"
    ]
    assert original == aliases
    assert (
        pickle.dumps(metadata, protocol=4)
        == (tmp_path / "model" / ".metadata").read_bytes()
    )


def _launch(world, action, path):
    import subprocess
    import sys

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        "--module",
        "tests.gdcp.dcp_worker",
        action,
        str(path),
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=240
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout, flush=True)


def test_real_tied_fsdp2_native_dcp_interchange_preserves_fp32_state(tmp_path):
    pytest.importorskip("transformers")
    _launch(2, "save", tmp_path / "gdcp")
    _launch(3, "native-load", tmp_path / "gdcp")
    _launch(2, "native-save", tmp_path / "native")
    _launch(4, "load", tmp_path / "native")


def test_pending_gdcp_save_cannot_be_read_as_native_dcp(tmp_path):
    from gdcp import desc
    from gdcp.adapters.dcp import load

    fixture = numpy_checkpoint(2, initialized=True)
    assert_ok(_dcp_save(tmp_path, fixture))
    (tmp_path / desc.MANIFEST_NAME).unlink()
    (tmp_path / desc.PENDING_NAME).write_bytes(b"incomplete")

    def _load(path, state, ctx, **kwargs):
        return load(path, state, ctx)

    result = collective(_load, tmp_path, fixture)
    assert all("save is pending" in str(item) for item in result)
