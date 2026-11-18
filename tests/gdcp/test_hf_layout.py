"""HF row grouping stays compact while preserving native file contents."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file

import gdtensor.builder as gb
from gdtensor.tag import WorldSize
from gdcp import hf
from gdcp.policy import PerRankPolicy
from gdcp.state import CheckpointState, Component, RankContext
from gdtensor_conn.coord import ThreadCoordinator
from tests.support.gdcp import assert_ok, assert_values
from tests.support.oracle import ArrayFabric


def _fixture(count, world, initialized):
    fabrics, values = {}, {}
    states = [CheckpointState() for _ in range(world)]
    for name, fields in (
        ("model", ("weight",)),
        ("optimizer", ("m", "v", "s", "step")),
    ):
        table, data, points = _component(count, world, fields)
        fabric = ArrayFabric(table, points, data if initialized else None)
        fabrics[name], values[name] = fabric, data
        for rank, state in enumerate(states):
            state.components.append(
                Component(name, table, fabric.view(rank).data_accessor)
            )
    return SimpleNamespace(states=states, fabrics=fabrics, values=values)


def _component(count, world, fields):
    entries, data = [], {}
    points = {rank: {} for rank in range(world)}
    mesh = gb.ArrangedMesh(WorldSize(world))
    for index in range(count):
        for field in fields:
            key, scalar = f"tensor{index:04d}.{field}", field == "step"
            shape = [1] if scalar else [12, 2]
            tensor = gb.NaiveDTensor(
                gb.GlobalTensorMeta(shape, "F32", 4),
                [gb.Replicate() if scalar else gb.Shard(0)],
                mesh,
            )
            entries.append(gb.NamedDTensor(key, tensor))
            data[key] = (
                np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) + index
            )
            _points(points, key, shape, scalar, world)
    return gb.DTensorTable(entries), data, points


def _points(points, key, shape, scalar, world):
    indices = np.arange(int(np.prod(shape))).reshape(shape)
    width = (shape[0] + world - 1) // world
    for rank in range(world):
        local = indices if scalar else indices[rank * width : (rank + 1) * width]
        points[rank][key] = local.reshape(-1)


def _collective(call, path, fixture, **kwargs):
    world = len(fixture.states)
    coordinator = ThreadCoordinator(world, timeout=30)

    def rank_call(rank):
        return call(
            path, fixture.states[rank], RankContext(rank, world, coordinator), **kwargs
        )

    with ThreadPoolExecutor(max_workers=world) as executor:
        return list(executor.map(rank_call, range(world)))


@pytest.mark.parametrize("count", [8, 80, 398])
@pytest.mark.parametrize("world", [1, 2, 4, 6])
def test_hf_rectangle_count_is_bounded_by_column_groups(count, world):
    fixture = _fixture(count, world, True)
    ctx = RankContext(0, world, ThreadCoordinator(world) if world > 1 else None)
    combined, _objects, compute, storage = hf._save_prepare(
        "/unused", fixture.states[0], ctx, None
    )
    groups = {
        id(group): group for group in storage.physical_layout.tensor_entries.values()
    }
    assert len(groups) == 1
    # Full-width shards plus at most one run per replicated owner; no term
    # depends on the number of tensor rows.
    assert len(next(iter(groups.values())).disjoint_storage_mesh_list) <= world + 1
    assert compute.tensor_ids == storage.tensor_ids
    assert len(combined.table.named_d_tensors) == count * 5
    scalar_slots = [
        pos.tensor_id
        for pos in storage.physical_layout.tensor_entries
        if pos.tensor_id.endswith(".step")
    ]
    assert len(scalar_slots) == len(set(scalar_slots)) == count


@pytest.mark.parametrize("limit", [None, 500])
def test_grouped_hf_plain_safetensors_and_reshard_are_bitwise(
    tmp_path, monkeypatch, limit
):
    source = _fixture(8, 6, True)
    assert_ok(_collective(hf.save_state, tmp_path, source, max_shard_size=limit))
    actual = {}
    for path in tmp_path.glob("*.safetensors"):
        tensors = load_file(str(path))
        assert not (actual.keys() & tensors.keys())
        actual.update(tensors)
    expected = dict(source.values["model"])
    expected.update(
        {"optimizer." + key: value for key, value in source.values["optimizer"].items()}
    )
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].tobytes() == expected[key].tobytes(), key

    def forbidden_policy(*args, **kwargs):
        raise AssertionError(
            "HF load must reconstruct native headers without save policy"
        )

    monkeypatch.setattr(PerRankPolicy, "plan", forbidden_policy)
    for world in (6, 4):
        target = _fixture(8, world, False)
        assert_ok(_collective(hf.load_state, tmp_path, target))
        assert_values(target, source)
