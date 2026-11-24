"""DCP preparation stays local and admits every check before record-file I/O."""

import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gdtensor.builder as gb
import gdtensor.tag as gt
from gdcp import core, policy
from gdcp.adapters import dcp
from gdcp.state import CheckpointState, Component, RankContext
from gdtensor_conn._layout import build_payload_index
from gdtensor_conn.coord import ThreadCoordinator
from gdtensor_conn.iobackend import FileIO
from tests.support.gdcp import assert_ok
from tests.support.oracle import ArrayFabric


class _Slots(dict):
    def __init__(self, slots):
        super().__init__(slots)
        self.visits = 0
        self.calls = 0

    def items(self):
        self.calls += 1
        self.visits += len(self)
        return super().items()


def _fixture(rows=32, world=8, *, aliases=True):
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                f"t{row}",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([world * 2, 3], "F32", 4),
                    [gb.Shard(0)],
                    gb.ArrangedMesh(gt.WorldSize(world)),
                ),
                [f"alias{row}"] if aliases else [],
            )
            for row in range(rows)
        ]
    )
    component = SimpleNamespace(
        name="model", table=table, logical_shapes={}, objects={}
    )
    table, layout = dcp.DcpPolicy().plan(component, world)
    index = build_payload_index(table, layout)
    index.slots = _Slots(index.slots)
    return component, table, layout, index


def _envelope(shape, dtype):
    nbytes = prod(shape) * dtype.itemsize
    return nbytes + 32, ((0, b"head"), (nbytes + 24, b"tail")), 16, nbytes


class _RecordIO:
    def __init__(self):
        self.created = []
        self.written = []

    def create_staging(self, path, size):
        self.created.append((path, size))

    def pwrite(self, rank, path, offset, data):
        self.written.append((rank, path, offset, data))


def _sessions(tmp_path, fixture, world, io, *, indices=None):
    component, table, layout, index = fixture
    coordinator = ThreadCoordinator(world, timeout=5)

    def one(rank):
        try:
            return dcp._SaveSession(
                rank,
                tmp_path,
                table,
                layout,
                index if indices is None else indices[rank],
                io,
                component=component,
                coordinator=coordinator,
            )
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=world) as pool:
        return list(pool.map(one, range(world)))


def test_dcp_policy_builds_one_layout(monkeypatch):
    component, _, _, _ = _fixture()
    calls = []
    original = dcp.build_layout

    def build(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(policy, "build_layout", build)
    monkeypatch.setattr(dcp, "build_layout", build)
    dcp.DcpPolicy().plan(component, 8)
    assert calls == ["distcp"]


def test_dcp_save_indexes_slots_once(monkeypatch):
    fixture = _fixture()
    monkeypatch.setattr(dcp, "_envelope", _envelope)
    dcp._plan(*fixture)
    index = fixture[-1]
    slots = index.slots
    assert isinstance(slots, _Slots)
    assert slots.calls == 1
    assert slots.visits == len(slots)


def _alias_events(rows):
    fixture = _fixture(rows, 4)
    events = 0

    def trace(frame, event, arg) -> Any:
        nonlocal events
        caller = frame.f_back
        if frame.f_code.co_filename == dcp.__file__ and (
            frame.f_code.co_name == "_plan_aliases"
            or (caller is not None and caller.f_code.co_name == "_plan_aliases")
        ):
            events += event == "line"
            return trace
        return None

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        dcp._plan(*fixture)
    finally:
        sys.settrace(previous)
    return events


def test_alias_preparation_does_not_scan_all_previous_tensors(monkeypatch):
    monkeypatch.setattr(dcp, "_envelope", _envelope)
    small, large = _alias_events(16), _alias_events(32)
    assert small > 0
    assert large < small * 2.5, (small, large)


def test_save_records_and_metadata_have_distinct_rank_duties(tmp_path, monkeypatch):
    world = 4
    fixture = _fixture(12, world)
    monkeypatch.setattr(dcp, "_envelope", _envelope)
    expected = dcp._plan(*fixture)
    io = _RecordIO()
    sessions = _sessions(tmp_path, fixture, world, io)
    assert all(isinstance(session, dcp._SaveSession) for session in sessions), sessions
    for rank, session in enumerate(sessions):
        assert isinstance(session, dcp._SaveSession)
        path = f"__{rank}_0.distcp"
        assert session.plan.records == {path: expected.records[path]}
        assert session.my_files() == [path]
        assert {pos.rank for pos in session.offsets} == {rank}
        if rank:
            assert session.plan.metadata is None
        else:
            assert dcp._metadata_bytes(session.plan.metadata) == dcp._metadata_bytes(
                expected.metadata
            )
    assert len(io.created) == world
    assert len(io.written) == sum(len(value) for value in expected.records.values())

    # HSDP can write payload into another logical rank's storage file. Offsets
    # remain correct even when the first requested tensor is the last in a file.
    ordinary = sessions[1]
    assert isinstance(ordinary, dcp._SaveSession)
    remote = [pos for pos in expected.offsets if pos.rank == 2]
    for pos in reversed(remote):
        result = ordinary.get_file_offset(pos)
        assert type(result) is gb.Ok, result
        assert result.value == expected.offsets[pos]
    assert ordinary.plan.records.keys() == {"__1_0.distcp"}
    assert ordinary._record_starts.keys() == {"__2_0.distcp"}
    assert ordinary.offsets.keys() == {
        pos for pos in expected.offsets if pos.rank in (1, 2)
    }


def _corrupt_nonlocal_geometry(original):
    def corrupt(component, named, pos, slot):
        if pos.rank == 3:
            raise ValueError("injected nonlocal geometry failure")
        return original(component, named, pos, slot)

    return corrupt


def _reject_local_preparation(original):
    def reject(component, table, index, paths, named=None):
        if "__1_0.distcp" in paths:
            raise ValueError("injected local preparation failure")
        return original(component, table, index, paths, named)

    return reject


@pytest.mark.parametrize("failure", ["global", "local"])
def test_record_preparation_failure_votes_before_any_file_creation(
    tmp_path, monkeypatch, failure
):
    fixture = _fixture(8, 4)
    monkeypatch.setattr(dcp, "_envelope", _envelope)
    if failure == "global":
        monkeypatch.setattr(
            dcp, "_chunk_envelope", _corrupt_nonlocal_geometry(dcp._chunk_envelope)
        )
    else:
        monkeypatch.setattr(
            dcp, "_local_plan", _reject_local_preparation(dcp._local_plan)
        )
    io = _RecordIO()
    errors = _sessions(tmp_path, fixture, 4, io)
    assert all(isinstance(error, ValueError) for error in errors), errors
    assert not io.created
    assert not io.written
    assert not list(tmp_path.iterdir())


def test_ordinary_rank_rejects_its_malformed_nonlocal_slot(tmp_path, monkeypatch):
    fixture = _fixture(8, 4)
    monkeypatch.setattr(dcp, "_envelope", _envelope)
    index = fixture[-1]
    corrupted = deepcopy(index)
    pos = next(pos for pos in corrupted.slots if pos.rank == 3)
    slot = corrupted.slots[pos][1]
    slot.intervals = slot.intervals.copy()
    slot.intervals[-1, 1] += 1
    io = _RecordIO()
    errors = _sessions(
        tmp_path, fixture, 4, io, indices=[index, corrupted, index, index]
    )
    assert all(isinstance(error, ValueError) for error in errors), errors
    assert "payload size mismatch" in str(errors[1])
    assert not io.created
    assert not io.written


def _baseline_canonical(value, strings):
    """Frozen 6da99a2 encoding oracle; pickle sharing is part of the contract."""
    if type(value) is str:
        return strings.setdefault(value, value)
    if isinstance(value, dict):
        return {
            _baseline_canonical(key, strings): _baseline_canonical(item, strings)
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        }
    if type(value) is torch.Size:
        return torch.Size(value)
    if isinstance(value, tuple):
        return tuple(_baseline_canonical(item, strings) for item in value)
    if isinstance(value, list):
        return [_baseline_canonical(item, strings) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                field.name: _baseline_canonical(getattr(value, field.name), strings)
                for field in fields(value)
                if field.init
            },
        )
    return value


def test_native_metadata_fast_path_preserves_exact_encoding(monkeypatch):
    fixture = _fixture(16, 8)
    monkeypatch.setattr(dcp, "_envelope", _envelope)
    metadata = dcp._plan(*fixture).metadata
    expected = pickle.dumps(_baseline_canonical(metadata, {}), protocol=4)
    calls = []
    original = dcp.dataclass_fields

    def counted(value):
        calls.append(value)
        return original(value)

    dcp._metadata_fields.cache_clear()
    monkeypatch.setattr(dcp, "dataclass_fields", counted)
    assert dcp._metadata_bytes(metadata) == expected
    assert len(calls) <= 7


@pytest.mark.skipif(
    not os.environ.get("GDTENSOR_TORCH_TESTS"), reason="set GDTENSOR_TORCH_TESTS=1"
)
def test_hsdp_rank_zero_writes_two_logical_files_and_native_dcp_reads(tmp_path):
    world = 4
    mesh = gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(world)), [2, 2])
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                f"weight{row}",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([8, 3], "F32", 4),
                    [gb.Replicate(), gb.Shard(0)],
                    mesh,
                ),
            )
            for row in range(4)
        ]
    )
    values = {
        named.global_tensor_id: np.arange(24, dtype=np.float32).reshape(8, 3)
        + 100 * row
        for row, named in enumerate(table.named_d_tensors)
    }
    positions = np.arange(24, dtype=np.int64).reshape(8, 3)
    points = {
        rank: {
            name: positions[(rank % 2) * 4 : (rank % 2 + 1) * 4].reshape(-1)
            for name in values
        }
        for rank in range(world)
    }
    fabric = ArrayFabric(table, points, values)
    writes = []

    class TraceIO(FileIO):
        def pwrite(self, rank, path, offset, data):
            writes.append((rank, Path(path).name))
            return super().pwrite(rank, path, offset, data)

    coordinator = ThreadCoordinator(world, timeout=10)

    def one(rank):
        component = Component("model", table, fabric.view(rank).data_accessor)
        return core.save(
            tmp_path / "checkpoint",
            CheckpointState([component]),
            RankContext(rank, world, coordinator),
            policy=dcp.DcpPolicy(),
            backend=dcp.DcpBackend({"model": component}, TraceIO()),
        )

    with ThreadPoolExecutor(max_workers=world) as pool:
        assert_ok(list(pool.map(one, range(world))))
    assert {path for rank, path in writes if rank == 0} == {
        "__0_0.distcp.tmp",
        "__1_0.distcp.tmp",
    }
    restored = {
        name: torch.empty_like(torch.from_numpy(value))
        for name, value in values.items()
    }
    torch.distributed.checkpoint.load(
        restored, checkpoint_id=tmp_path / "checkpoint" / "model", no_dist=True
    )
    for name, value in restored.items():
        np.testing.assert_array_equal(value.numpy(), values[name])
