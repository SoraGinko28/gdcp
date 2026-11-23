"""Run geometry for union shards, independent of the planner implementation."""

from math import prod

import numpy as np

from gdtensor.builder import Partial, Replicate, Shard, StridedShard
from ._result import invalid

EMPTY = np.empty((0, 2), dtype=np.int64)


def iv_normalize(iv: np.ndarray) -> np.ndarray:
    """Sort, drop empties, merge adjacent / overlapping intervals."""
    if iv.size == 0:
        return EMPTY
    iv = iv[iv[:, 1] > iv[:, 0]]
    if iv.size == 0:
        return EMPTY
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    starts, ends = iv[:, 0], np.maximum.accumulate(iv[:, 1])
    new = np.ones(len(iv), dtype=bool)
    new[1:] = starts[1:] > ends[:-1]
    grp = np.cumsum(new) - 1
    out = np.empty((int(grp[-1]) + 1, 2), dtype=np.int64)
    out[:, 0] = starts[new]
    out[:, 1] = np.maximum.reduceat(ends, np.flatnonzero(new))
    return out


def _chunk(n: int, parts: int, idx: int) -> tuple[int, int]:
    c = -(-n // parts) if n else 0
    s = min(idx * c, n)
    return s, min(s + c, n)


def _append_run(runs, start, end):
    if end <= start:
        return
    if runs and runs[-1][1] == start:
        runs[-1] = runs[-1][0], end
    else:
        runs.append((start, end))


def _select_axis_runs(runs, selections):
    """Map sorted packed-coordinate selections to global runs in linear time."""
    result, index, packed = [], 0, 0
    for start, end in selections:
        while index < len(runs) and packed + runs[index][1] - runs[index][0] <= start:
            packed += runs[index][1] - runs[index][0]
            index += 1
        cursor = start
        while index < len(runs) and cursor < end:
            lo, hi = runs[index]
            stop = min(end, packed + hi - lo)
            _append_run(result, lo + cursor - packed, lo + stop - packed)
            cursor = stop
            completed = cursor == packed + hi - lo
            packed += hi - lo if completed else 0
            index += int(completed)
    return result


def _strided_selections(length, parts, coordinate, split_factor):
    if type(split_factor) is not int or split_factor < 1:
        raise invalid(
            "placements", "StridedShard split_factor must be a positive integer"
        )
    if not length:
        return []
    first_width = -(-length // split_factor)
    # Each first-level piece is a singleton: only coordinate zero owns data.
    # This also avoids a per-element loop when split_factor >= length.
    if first_width == 1:
        return [(0, length)] if coordinate == 0 else []
    if parts == 1:
        return [(0, length)]
    selected = []
    full_count, tail = divmod(length, first_width)
    start, end = _chunk(first_width, parts, coordinate)
    # All full first-level pieces share one shape. Empty coordinates can
    # skip them together even when split_factor is much larger than n.
    if end > start:
        for part in range(full_count):
            base = part * first_width
            _append_run(selected, base + start, base + end)
    if tail:
        base = full_count * first_width
        start, end = _chunk(tail, parts, coordinate)
        _append_run(selected, base + start, base + end)
    return selected


def axis_runs_of(naive, mesh_shape, coords):
    """Owned global axis runs; operations split the packed preceding result.

    Work scales with contiguous runs and virtual pieces, never tensor elements.
    Consecutive pieces are merged before another placement is applied.
    """
    shape = naive.global_tensor_meta.global_shape
    if len(naive.placements) != len(mesh_shape):
        raise invalid("placements", "placements do not match mesh dimensions")
    axes = [[(0, size)] if size else [] for size in shape]
    for placement, size, coordinate in zip(naive.placements, mesh_shape, coords):
        if type(placement) is Partial:
            raise invalid("placements", "Partial placement is not persistable")
        if type(placement) is Replicate:
            continue
        if type(placement) not in (Shard, StridedShard):
            raise invalid("placements", f"unsupported placement {placement!r}")
        axis = placement.axis
        if not 0 <= axis < len(shape):
            raise invalid(
                "placements", f"Shard axis {axis} out of range for shape {shape}"
            )
        runs = axes[axis]
        length = sum(end - start for start, end in runs)
        selected = (
            _strided_selections(length, size, coordinate, placement.split_factor)
            if type(placement) is StridedShard
            else [_chunk(length, size, coordinate)]
        )
        axes[axis] = _select_axis_runs(runs, selected)
    return axes


def axis_runs_to_intervals(shape, axes):
    """Flatten a Cartesian product of sorted axis runs in packed order."""
    if not shape:
        return np.array([[0, 1]], dtype=np.int64)
    if any(not runs for runs in axes):
        return EMPTY
    last = len(shape) - 1
    while last and axes[last] == [(0, shape[last])]:
        last -= 1
    strides = [prod(shape[d + 1 :]) for d in range(last + 1)]
    bases = np.zeros(1, dtype=np.int64)
    for d in range(last):
        points = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in axes[d]])
        bases = (bases[:, None] + points[None, :] * strides[d]).ravel()
    runs = np.asarray(axes[last], dtype=np.int64) * strides[last]
    intervals = (bases[:, None, None] + runs[None, :, :]).reshape(-1, 2)
    return iv_normalize(intervals)
