"""Small independent NumPy state for GDCP's framework-neutral core tests."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

import gdtensor.builder as gb
import gdtensor.tag as gt
from gdcp.backend import SafeTensorsBackend
from gdcp.policy import PerRankPolicy
from gdcp.state import CheckpointState, Component, RankContext
from gdtensor_conn.coord import ThreadCoordinator
from tests.support.oracle import ArrayFabric


SPECS = [
    ("weight", (10, 4), "F32", np.float32, False),
    ("tiny", (2, 3), "F32", np.float32, False),
    ("bias", (10,), "F32", np.float32, False),
    ("bf16", (5, 3), "BF16", np.uint16, False),
    ("empty", (0, 3), "F32", np.float32, False),
    ("buffer", (4,), "I64", np.int64, True),
    ("step", (1,), "F32", np.float32, True),
]


def numpy_table(world):
    entries = []
    for name, shape, dtype, numpy_dtype, replicated in SPECS:
        meta = gb.GlobalTensorMeta(list(shape), dtype, np.dtype(numpy_dtype).itemsize)
        placement = gb.Replicate() if replicated else gb.Shard(0)
        tensor = gb.NaiveDTensor(
            meta, [placement], gb.ArrangedMesh(gt.WorldSize(world))
        )
        entries.append(gb.NamedDTensor(name, tensor))
    return gb.DTensorTable(entries)


def numpy_points(world):
    points = {rank: {} for rank in range(world)}
    for name, shape, _dtype, _numpy_dtype, replicated in SPECS:
        indices = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
        chunk = (shape[0] + world - 1) // world
        for rank in range(world):
            local = (
                indices if replicated else indices[rank * chunk : (rank + 1) * chunk]
            )
            points[rank][name] = local.reshape(-1)
    return points


def numpy_checkpoint(world, *, initialized, names=("model",)):
    fabrics, values = {}, {}
    states = [CheckpointState() for _ in range(world)]
    for offset, name in enumerate(names):
        table = numpy_table(world)
        data = {
            tid: (
                np.arange(int(np.prod(shape)), dtype=dtype) + 100 * order + offset
            ).reshape(shape)
            for order, (tid, shape, _dtype, dtype, _replicated) in enumerate(SPECS)
        }
        fabric = ArrayFabric(table, numpy_points(world), data if initialized else None)
        fabrics[name], values[name] = fabric, data
        _add_components(states, name, fabric)
    return SimpleNamespace(
        table=fabrics[names[-1]].table, states=states, fabrics=fabrics, values=values
    )


def _add_components(states, name, fabric):
    for rank, state in enumerate(states):
        objects = {
            "param_groups": [{"lr": 0.03, "betas": [0.9, 0.999], "params": ["weight"]}]
        }
        component = Component(
            name, fabric.table, fabric.view(rank).data_accessor, objects
        )
        component.logical_shapes = {"step": []}
        state.components.append(component)


def collective(call, root, fixture, *, backend=None, **kwargs):
    world = len(fixture.states)
    coordinator = ThreadCoordinator(world, timeout=30)
    backend = backend or SafeTensorsBackend()
    options = {"backend": backend, **kwargs}
    if call.__name__ == "save":
        options["policy"] = PerRankPolicy()

    def one_rank(rank):
        context = RankContext(rank, world, coordinator)
        return call(root, fixture.states[rank], context, **options)

    with ThreadPoolExecutor(max_workers=world) as executor:
        return list(executor.map(one_rank, range(world)))


def assert_ok(results):
    assert all(type(result) is gb.Ok for result in results), results


def assert_values(fixture, expected):
    for name, fabric in fixture.fabrics.items():
        fabric.assert_values(expected.values[name])
