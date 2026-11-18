"""Shared NumPy checkpoint fixtures and public overwrite operation helpers."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest

from gdcp import core, hf
from gdcp.backend import SafeTensorsBackend
from gdcp.policy import PerRankPolicy
from gdcp.state import RankContext
from gdtensor_conn.coord import ThreadCoordinator
from gdtensor_conn.iobackend import FileIO
from tests.support.gdcp import assert_ok, assert_values, numpy_checkpoint


def _shift(values, shift):
    for value in values.values():
        value += shift


def _fixture(world=2, shift=0):
    fixture = numpy_checkpoint(world, initialized=True, names=("model", "optimizer"))
    for values in fixture.values.values():
        _shift(values, shift)
    for name, fabric in fixture.fabrics.items():
        _fill_fabric(fabric, fixture.values[name])
    for rank, state in enumerate(fixture.states):
        state.rank_state["extra_state"] = f"{shift}-{rank}".encode()
        for component in state.components:
            component.objects["param_groups"][0]["lr"] += shift
    return fixture


def _fill_fabric(fabric, values):
    for rank, tensors in fabric.points.items():
        for key, indices in tensors.items():
            fabric.buffers[rank][key][:] = (
                values[key].reshape(-1)[indices].copy().view(np.uint8).ravel()
            )


def _operation(fmt, loading, fixture):
    if fmt == "hf":
        return (hf.load_state if loading else hf.save_state), {}
    if fmt == "safetensors":
        backend, policy = SafeTensorsBackend(), PerRankPolicy()
    else:
        if not os.environ.get("GDTENSOR_TORCH_TESTS"):
            pytest.skip("set GDTENSOR_TORCH_TESTS=1")
        from gdcp.adapters.dcp import DcpBackend, DcpPolicy

        components = {item.name: item for item in fixture.states[0].components}
        backend, policy = DcpBackend(components), DcpPolicy()
    options: dict[str, Any] = {"backend": backend}
    if not loading:
        options["policy"] = policy
    return (core.load if loading else core.save), options


def _run(fmt, path, fixture, *, loading=False):
    call, options = _operation(fmt, loading, fixture)
    world = len(fixture.states)
    coordinator = ThreadCoordinator(world, timeout=30)

    def rank_call(rank):
        return call(
            path, fixture.states[rank], RankContext(rank, world, coordinator), **options
        )

    with ThreadPoolExecutor(max_workers=world) as executor:
        return list(executor.map(rank_call, range(world)))


def _files(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _fail_rank_one(monkeypatch):
    original = FileIO.pwrite
    failures = []

    def faulty(self, rank, path, offset, data):
        if rank == 1 and offset > 0 and not failures:
            failures.append(str(path))
            raise OSError("injected rank-1 overwrite failure")
        return original(self, rank, path, offset, data)

    monkeypatch.setattr(FileIO, "pwrite", faulty)
    return failures


def _assert_generation(fmt, root, expected, world=3):
    target = _fixture(world, shift=31)
    assert_ok(_run(fmt, root, target, loading=True))
    assert_values(target, expected)
    expected_objects = [item.objects for item in expected.states[0].components]
    assert [item.objects for item in target.states[0].components] == expected_objects
    for rank, state in enumerate(target.states):
        source_rank = rank if rank < len(expected.states) else 0
        assert state.rank_state == expected.states[source_rank].rank_state
