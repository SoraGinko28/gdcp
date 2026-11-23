"""GDCP v1 strided ownership and packed torch byte access regression tests."""

import os
from types import SimpleNamespace

import pytest

from gdtensor.builder import (
    ArrangedMesh,
    GlobalTensorMeta,
    NaiveDTensor,
    NamedDTensor,
    StridedShard,
)
from gdtensor.tag import WorldSize
from gdtensor_conn._geometry import owned_intervals


def test_realistic_strided_runs_are_merged():
    value = NamedDTensor(
        "weight",
        NaiveDTensor(
            GlobalTensorMeta([150_000, 4096], "BF16", 2),
            [StridedShard(0, 3)],
            ArrangedMesh(WorldSize(8)),
        ),
    )
    # Three wide contiguous runs per rank, independent of the 150k row count.
    assert [len(owned_intervals(value, "compute", rank)) for rank in range(8)] == [
        3
    ] * 8


def test_packed_accessor_maps_disjoint_runs_and_partial_elements():
    torch = pytest.importorskip("torch")
    from gdcp.adapters.torch_access import TorchAccessor

    values = torch.tensor([0, 1, 4, 5], dtype=torch.int64)
    accessor = TorchAccessor({"x": values}, {}, {"x": [(0, 2), (4, 6)]})
    rng = SimpleNamespace(global_tensor_id="x", global_tensor_offset=4 * 8 + 1, size=7)
    actual = accessor._buffer(rng, "read")
    assert actual.tolist() == values.view(torch.uint8)[17:24].tolist()
    rng.global_tensor_offset, rng.size = 8, 24
    with pytest.raises(ValueError, match="outside the local shard"):
        accessor._buffer(rng, "read")
    accessor.finish()


def test_private_placement_roundtrip():
    pytest.importorskip("torch")
    from gdcp.adapters.torch_placements import (
        decode_placements,
        encode_placements,
        strided_shard,
    )
    from torch.distributed.tensor import Replicate, Shard

    original = (Shard(0), strided_shard(0, split_factor=3), Replicate())
    assert decode_placements(encode_placements(original)) == original


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
def test_real_fsdp2_strided_all_formats(tmp_path, backend):
    if not os.environ.get("GDTENSOR_TORCH_TESTS"):
        pytest.skip("set GDTENSOR_TORCH_TESTS=1")
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < 4 or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("requires four CUDA devices and NCCL")
    from tests.support.gdcp_launch import run_fsdp2

    output = run_fsdp2(4, "strided", tmp_path, "--backend", backend)
    assert output.count("general strided PASS") == 4


@pytest.mark.parametrize("backend", ["gloo", "nccl", "cuda:nccl,cpu:gloo"])
def test_real_fsdp2_strided_cross_world(tmp_path, backend):
    if not os.environ.get("GDTENSOR_TORCH_TESTS"):
        pytest.skip("set GDTENSOR_TORCH_TESTS=1")
    torch = pytest.importorskip("torch")
    if backend != "gloo" and (
        torch.cuda.device_count() < 4 or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("requires four CUDA devices and NCCL")
    from tests.support.gdcp_launch import run_fsdp2

    run_fsdp2(4, "strided_save", tmp_path, "--backend", backend)
    for world in (2, 3):
        run_fsdp2(world, "strided_load", tmp_path, "--backend", backend)
