"""Generic run ownership agrees with independent dense slicing and public I/O."""

from itertools import product

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_conn._geometry import owned_intervals
from tests.support.memory import Memory, storage_view
from tests.support.oracle import ArrayFabric


def _chunk_array(values, axis, parts, coordinate):
    width = (values.shape[axis] + parts - 1) // parts
    selection = [slice(None)] * values.ndim
    selection[axis] = slice(coordinate * width, (coordinate + 1) * width)
    return values[tuple(selection)]


def _dense_placement(values, size, coordinate, placement):
    if type(placement) is gb.Replicate:
        return values
    axis = placement.axis
    if type(placement) is gb.StridedShard:
        pieces = [
            _chunk_array(
                _chunk_array(values, axis, placement.split_factor, part),
                axis,
                size,
                coordinate,
            )
            for part in range(placement.split_factor)
        ]
        return np.concatenate(pieces, axis=axis)
    return _chunk_array(values, axis, size, coordinate)


def _dense(shape, sizes, coordinates, placements):
    values = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    for size, coordinate, placement in zip(sizes, coordinates, placements):
        values = _dense_placement(values, size, coordinate, placement)
    return values.ravel()


def _named(shape, sizes, placements):
    mesh = gb.ReshapedMesh(
        gb.ArrangedMesh(gt.WorldSize(int(np.prod(sizes)))), list(sizes)
    )
    return gb.NamedDTensor(
        "w",
        gb.NaiveDTensor(gb.GlobalTensorMeta(list(shape), "U8", 1), placements, mesh),
    )


@pytest.mark.parametrize(
    "shape,sizes,placements",
    [
        ((11,), (2,), [gb.StridedShard(0, 3)]),
        ((1,), (4,), [gb.StridedShard(0, 11)]),
        ((0,), (2,), [gb.StridedShard(0, 3)]),
        ((19,), (2, 3), [gb.Shard(0), gb.StridedShard(0, 5)]),
        ((19,), (2, 3), [gb.StridedShard(0, 5), gb.Shard(0)]),
        ((7, 11), (2, 3), [gb.StridedShard(1, 3), gb.Shard(0)]),
        (
            (3, 7, 11),
            (2, 2, 2),
            [gb.Shard(1), gb.StridedShard(1, 3), gb.StridedShard(2, 5)],
        ),
        (
            (3, 7, 11),
            (2, 2, 2),
            [gb.StridedShard(2, 5), gb.StridedShard(2, 3), gb.Replicate()],
        ),
    ],
)
def test_runs_match_dense_oracle(shape, sizes, placements):
    named = _named(shape, sizes, placements)
    assert type(named.check_constraints()) is gb.Ok
    for rank, coordinates in enumerate(product(*(range(s) for s in sizes))):
        want = _dense(shape, sizes, coordinates, placements)
        intervals = owned_intervals(named, "compute", rank)
        got = (
            np.concatenate([np.arange(start, stop) for start, stop in intervals])
            if len(intervals)
            else np.array([], dtype=np.int64)
        )
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "shape,sizes,placements",
    [
        ((11,), (2,), [gb.StridedShard(0, 3)]),
        ((19,), (2, 3), [gb.Shard(0), gb.StridedShard(0, 5)]),
        ((7, 11), (2, 3), [gb.StridedShard(1, 3), gb.Shard(0)]),
        (
            (3, 7, 11),
            (2, 2, 2),
            [gb.StridedShard(2, 5), gb.StridedShard(2, 3), gb.Replicate()],
        ),
    ],
)
def test_strided_public_save_and_reshard(shape, sizes, placements):
    named = _named(shape, sizes, placements)
    table = gb.DTensorTable([named])
    points = {
        rank: {"w": _dense(shape, sizes, coordinates, placements)}
        for rank, coordinates in enumerate(product(*(range(size) for size in sizes)))
    }
    count = int(np.prod(shape))
    values = {"w": np.arange(count, dtype=np.uint8).reshape(shape)}
    source = ArrayFabric(table, points, values)
    memory = Memory([0xA5] * count)
    saved = storage_view(memory, [])
    saved.d_tensor_table.named_d_tensors[
        0
    ].d_tensor.global_tensor_meta.global_shape = list(shape)
    for rank in range(len(points)):
        result = gio.rank_save(rank, source.view(rank), saved)
        assert type(result) is gb.Ok, result
    np.testing.assert_array_equal(memory.data, values["w"].ravel())

    target_named = _named(shape, (3,), [gb.Shard(0)])
    target_points = {
        r: {"w": _dense(shape, (3,), (r,), [gb.Shard(0)])} for r in range(3)
    }
    target = ArrayFabric(gb.DTensorTable([target_named]), target_points)
    for rank in range(3):
        result = gio.rank_load(rank, target.view(rank), saved)
        assert type(result) is gb.Ok, result
    target.assert_values(values)
    restored = ArrayFabric(table, points)
    for rank in range(len(points)):
        result = gio.rank_load(rank, restored.view(rank), saved)
        assert type(result) is gb.Ok, result
    restored.assert_values(values)


def test_realistic_weight_uses_runs_not_rows():
    named = _named((150000, 4096), (4, 2), [gb.StridedShard(0, 2), gb.Shard(0)])
    assert [len(owned_intervals(named, "compute", r)) for r in range(8)] == [1] * 8
    named = _named((150000, 4096), (4,), [gb.StridedShard(0, 3)])
    assert [len(owned_intervals(named, "compute", r)) for r in range(4)] == [3] * 4


def test_empty_coordinate_skips_all_identical_virtual_pieces():
    named = _named((150000000000,), (8,), [gb.StridedShard(0, 75000000000)])
    assert len(owned_intervals(named, "compute", 7)) == 0


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_split_factor_must_be_positive_integer(factor):
    assert type(gb.StridedShard(0, factor).check_constraints()) is gb.Err
