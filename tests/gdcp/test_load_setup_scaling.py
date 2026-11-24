"""Load setup indexes once and admits every read before modifying targets."""

from bisect import bisect_right
from contextlib import ExitStack
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdcp import core
from gdcp.backend import SafeTensorsBackend
from gdcp.policy import PerRankPolicy
from gdtensor_conn._layout import build_payload_index
from gdtensor_conn.iobackend import FileIO
from gdtensor_conn.models import build_layout
from tests.support.gdcp import assert_ok, assert_values, collective, numpy_checkpoint
from tests.support.setup_plans import Trace, compute_view, storage_view


class _Visits(list):
    def __init__(self, values):
        super().__init__(values)
        self.visits = 0

    def __iter__(self):
        for value in super().__iter__():
            self.visits += 1
            yield value


def test_dcp_offset_setup_visits_names_and_chunks_once(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from gdcp.adapters import dcp

    fixture = numpy_checkpoint(4, initialized=True)
    component = fixture.states[0].components[0]
    table, layout = dcp.DcpPolicy().plan(component, 4)
    index = build_payload_index(table, layout)
    plan = dcp._plan(component, table, layout, index)
    assert plan is not None
    metadata = plan.metadata
    assert metadata is not None
    names = table.named_d_tensors = _Visits(table.named_d_tensors)
    chunks = []
    for item in metadata.state_dict_metadata.values():
        if isinstance(item, dcp.TensorStorageMetadata):
            item.chunks = _Visits(item.chunks)
            chunks.append(item.chunks)
    records = []
    monkeypatch.setattr(dcp, "read_metadata", lambda root: metadata)

    def payload(root, record, expected, tensor_id):
        records.append((tensor_id, record.relative_path))
        return record.offset + 64

    monkeypatch.setattr(dcp, "_record_payload", payload)
    session = dcp._LoadSession(1, tmp_path, table, layout, index, FileIO())
    requested = [pos for pos in index.slots if pos.rank == 1]
    for pos in requested:
        assert type(session.get_file_offset(pos)) is gb.Ok
    assert names.visits == len(names)
    assert sum(group.visits for group in chunks) == sum(map(len, chunks))
    assert len(records) == len(requested)
    assert set(session.offsets) == set(requested)


@pytest.mark.parametrize("edit", ["dtype", "duplicate_chunk"])
def test_dcp_offset_index_is_rebuilt_and_validates_metadata_edits(
    tmp_path, monkeypatch, edit
):
    torch = pytest.importorskip("torch")
    from gdcp.adapters import dcp

    source = numpy_checkpoint(4, initialized=True)
    component = source.states[0].components[0]
    table, layout = dcp.DcpPolicy().plan(component, 4)
    index = build_payload_index(table, layout)
    plan = dcp._plan(component, table, layout, index)
    assert plan is not None
    metadata = plan.metadata
    assert metadata is not None
    monkeypatch.setattr(dcp, "read_metadata", lambda root: metadata)
    monkeypatch.setattr(dcp, "_record_payload", lambda *args: 64)
    first = dcp._LoadSession(1, tmp_path, table, layout, index, FileIO())
    pos = next(
        pos for pos in index.slots if pos.tensor_id == "weight" and pos.rank == 1
    )
    assert type(first.get_file_offset(pos)) is gb.Ok
    item = metadata.state_dict_metadata["weight"]
    assert isinstance(item, dcp.TensorStorageMetadata)
    if edit == "dtype":
        item.properties.dtype = torch.int32
    else:
        item.chunks.append(item.chunks[0])
    with pytest.raises(ValueError, match="dtype mismatch|duplicate DCP chunk geometry"):
        dcp._LoadSession(1, tmp_path, table, layout, index, FileIO())


class _HeaderIO(FileIO):
    def __init__(self):
        super().__init__()
        self.headers = {}

    def pread(self, rank, path, offset, size):
        if offset == 0 and str(path).endswith(".safetensors"):
            self.headers.setdefault(rank, []).append(Path(path).name)
        return super().pread(rank, path, offset, size)


def test_safetensors_preflight_reads_only_plan_files_except_coordinator(
    tmp_path, monkeypatch
):
    source = numpy_checkpoint(4, initialized=True)
    assert_ok(collective(core.save, tmp_path, source))
    target = numpy_checkpoint(4, initialized=False)
    expected = {}
    original = core.rank_load_files

    def files(rank, *args):
        result = original(rank, *args)
        expected[rank] = result.value
        return result

    monkeypatch.setattr(core, "rank_load_files", files)
    io = _HeaderIO()
    assert_ok(collective(core.load, tmp_path, target, backend=SafeTensorsBackend(io)))
    assert_values(target, source)
    all_files = sorted(path.name for path in (tmp_path / "model").glob("*.safetensors"))
    assert sorted(io.headers[0]) == all_files
    for rank in range(1, 4):
        assert sorted(io.headers[rank]) == expected[rank]
        assert len(io.headers[rank]) == len(set(io.headers[rank]))
    assert len(io.headers[1]) < len(all_files)


def _compute_table(case):
    world = 4
    whole = gb.ArrangedMesh(gt.WorldSize(world))
    square = gb.ReshapedMesh(whole, [2, 2])
    mesh, placements, shape = whole, [gb.Shard(0)], [16, 6]
    if case == "hsdp":
        mesh, placements = square, [gb.Replicate(), gb.Shard(0)]
    elif case == "same":
        mesh, placements = square, [gb.StridedShard(0, 2), gb.Shard(0)]
    elif case == "different":
        mesh, placements = square, [gb.Shard(0), gb.Shard(1)]
    elif case == "strided":
        mesh, placements = square, [gb.Shard(0), gb.StridedShard(0, 2)]
    elif case == "submesh":
        mesh = gb.LiteralMesh([1, 3], gt.WorldSize(world))
    elif case == "uneven":
        shape = [5, 3]
    elif case == "empty_shard":
        shape = [2, 3]
    tensor = gb.NaiveDTensor(gb.GlobalTensorMeta(shape, "F32", 4), placements, mesh)
    return gb.DTensorTable([gb.NamedDTensor("weight", tensor)])


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("saved_world", [3, 4])
@pytest.mark.parametrize(
    "case",
    [
        "fsdp",
        "hsdp",
        "same",
        "different",
        "strided",
        "submesh",
        "uneven",
        "empty_shard",
    ],
)
def test_selected_slots_equal_executed_plan(case, shared, saved_world):
    table = _compute_table(case)
    stored, layout = PerRankPolicy().plan(SimpleNamespace(table=table), saved_world)
    if shared:
        layout = build_layout(
            stored,
            saved_world,
            "safetensors",
            lambda order, rank, named: "shared.safetensors",
        )
    index = build_payload_index(stored, layout)
    for rank in range(4):
        trace = Trace()
        compute = compute_view(table, trace)
        storage = storage_view(stored, layout, trace)
        assert type(gio.rank_load(rank, compute, storage)) is gb.Ok
        reads = [row for row in trace.ranges if row[0] == "storage" and row[-1]]
        expected = {
            index.files[path]
            .slots[bisect_right(index.files[path].starts, offset) - 1]
            .pos
            for _, path, offset, _ in reads
        }
        filenames = sorted({row[1] for row in reads})
        selected = core._load_positions(
            compute, storage, SimpleNamespace(index=index), rank, filenames
        )
        assert set(selected) == expected


def test_custom_session_without_payload_index_preserves_full_preflight(tmp_path):
    table = _compute_table("fsdp")
    stored, layout = PerRankPolicy().plan(SimpleNamespace(table=table), 4)
    trace, checked = Trace(), []
    compute = compute_view(table, trace)
    storage = storage_view(stored, layout, trace)
    session = SimpleNamespace(
        get_file_offset=lambda pos: checked.append(pos) or gb.Ok(0),
        finish=lambda: gb.Ok(None),
    )
    storage.begin_load = lambda view, rank: gb.Ok(session)
    with ExitStack() as stack:
        core._prepare_load_view(
            SimpleNamespace(rank=1), compute, storage, stack, tmp_path
        )
    assert checked == list(layout.tensor_entries)


def test_slot_without_target_name_has_clear_preflight_error():
    table = _compute_table("fsdp")
    stored, layout = PerRankPolicy().plan(SimpleNamespace(table=table), 4)
    index = build_payload_index(stored, layout)
    compute = compute_view(gb.DTensorTable(), Trace())
    storage = storage_view(stored, layout, Trace())
    with pytest.raises(ValueError, match="weight: saved slot has no target tensor"):
        core._load_positions(
            compute, storage, SimpleNamespace(index=index), 1, index.files
        )


def test_shared_file_offset_uses_first_saved_slot_without_validating_other_slots(
    tmp_path,
):
    from gdcp.backend import _LocalSafeTensorsLoadSession
    from gdtensor_conn.formats import COMMIT_MARKER_NAME, safetensors_header

    table = _compute_table("fsdp")
    stored, _ = PerRankPolicy().plan(SimpleNamespace(table=table), 4)
    layout = build_layout(stored, 4, "safetensors", lambda *args: "shared.safetensors")
    index = build_payload_index(stored, layout)
    header = safetensors_header(stored, index, "shared.safetensors")
    (tmp_path / "shared.safetensors").write_bytes(
        header + bytes(index.files["shared.safetensors"].payload_size)
    )
    (tmp_path / COMMIT_MARKER_NAME).write_text(
        json.dumps({"files": ["shared.safetensors"]})
    )
    session = _LocalSafeTensorsLoadSession(
        1, str(tmp_path), stored, layout, index, FileIO(), eager_headers=False
    )
    pos = gb.TensorComponentPos("weight", 1)
    session._prepare_offsets([pos])
    offset = session.get_file_offset(pos)
    assert type(offset) is gb.Ok
    assert offset.value == len(header) + index.slots[pos][1].contiguous_payload_offset
    assert set(session.offsets) == {pos}
    session._finish_preflight([pos])
    assert type(session.get_file_offset(gb.TensorComponentPos("weight", 0))) is gb.Err
    assert type(session.finish()) is gb.Ok


def _save(path, state, ctx, **options):
    return core.save(path, state, ctx, **options)


def _save_dcp(path, source):
    pytest.importorskip("torch")
    from gdcp.adapters.dcp import DcpBackend, DcpPolicy

    components = {
        component.name: component for component in source.states[0].components
    }
    return collective(
        _save, path, source, policy=DcpPolicy(), backend=DcpBackend(components)
    )


def _buffers(fixture):
    return {
        (name, rank, key): value.copy()
        for name, fabric in fixture.fabrics.items()
        for rank, values in fabric.buffers.items()
        for key, value in values.items()
    }


@pytest.mark.parametrize(
    "corruption",
    ["missing_chunk", "shape", "dtype", "record_path", "record_length", "pickle"],
)
def test_corrupt_nonlocal_dcp_metadata_rejects_before_any_target_write(
    tmp_path, corruption
):
    torch = pytest.importorskip("torch")
    from gdcp.adapters import dcp

    source = numpy_checkpoint(4, initialized=True, names=("model", "optimizer"))
    assert_ok(_save_dcp(tmp_path, source))
    root = tmp_path / "optimizer"
    metadata = dcp.read_metadata(root)
    item = metadata.state_dict_metadata["weight"]
    assert isinstance(item, dcp.TensorStorageMetadata)
    assert metadata.storage_data is not None
    if corruption == "missing_chunk":
        item.chunks.pop()
    elif corruption == "shape":
        item.size = torch.Size([11, 4])
    elif corruption == "dtype":
        item.properties.dtype = torch.int32
    elif corruption in ("record_path", "record_length"):
        record = metadata.storage_data[
            dcp.MetadataIndex("weight", item.chunks[-1].offsets)
        ]
        if corruption == "record_path":
            record.relative_path = "__wrong_0.distcp"
        else:
            record.length = 0
    raw = b"invalid pickle" if corruption == "pickle" else dcp._metadata_bytes(metadata)
    (root / ".metadata").write_bytes(raw)
    target = numpy_checkpoint(4, initialized=False, names=("model", "optimizer"))
    before = _buffers(target)
    results = collective(core.load, tmp_path, target, backend=dcp.DcpBackend())
    assert all(type(result) is gb.Err for result in results), results
    for key, value in _buffers(target).items():
        np.testing.assert_array_equal(value, before[key])


def test_corrupt_nonlocal_safetensors_header_rejects_before_any_target_write(tmp_path):
    source = numpy_checkpoint(4, initialized=True, names=("model", "optimizer"))
    assert_ok(collective(core.save, tmp_path, source))
    path = tmp_path / "optimizer" / "__3_0.safetensors"
    raw = path.read_bytes()
    changed = raw.replace(b'"dtype":"F32"', b'"dtype":"I32"', 1)
    assert changed != raw and len(changed) == len(raw)
    path.write_bytes(changed)
    target = numpy_checkpoint(4, initialized=False, names=("model", "optimizer"))
    before = _buffers(target)
    results = collective(core.load, tmp_path, target)
    assert all(type(result) is gb.Err for result in results), results
    for key, value in _buffers(target).items():
        np.testing.assert_array_equal(value, before[key])


class _TrackedTarget(ga.DTensorDataAccessor):
    def __init__(self, delegate, writes):
        self.delegate, self.writes = delegate, writes

    def read_to_buffer(self, global_range, read_to_buffer):
        return self.delegate.read_to_buffer(global_range, read_to_buffer)

    def write_from_buffer(self, global_range, write_from_buffer):
        self.writes.set()
        return self.delegate.write_from_buffer(global_range, write_from_buffer)


@pytest.mark.parametrize("fmt", ["dcp", "safetensors"])
def test_payload_uses_only_sealed_preflight_offsets(tmp_path, monkeypatch, fmt):
    source = numpy_checkpoint(4, initialized=True)
    backend = None
    writes, parsed, sessions = Event(), [], []
    if fmt == "dcp":
        pytest.importorskip("torch")
        from gdcp.adapters import dcp

        assert_ok(_save_dcp(tmp_path, source))
        backend = dcp.DcpBackend()
        session_type = dcp._LoadSession
        parse_owner, parse_name = dcp, "_record_payload"
    else:
        from gdcp.backend import _LocalSafeTensorsLoadSession

        assert_ok(collective(core.save, tmp_path, source))
        session_type = _LocalSafeTensorsLoadSession
        parse_owner, parse_name = session_type, "_read_file_offsets"
    parse = getattr(parse_owner, parse_name)

    def parsing(*args, **kwargs):
        parsed.append(True)
        assert not writes.is_set(), "offset validation started after a target write"
        return parse(*args, **kwargs)

    monkeypatch.setattr(parse_owner, parse_name, parsing)
    finish = session_type._finish_preflight

    def finished(session, positions):
        finish(session, positions)
        assert session.state == "open"
        off_plan = next(pos for pos in session.index.slots if pos not in positions)
        assert type(session.get_file_offset(off_plan)) is gb.Err
        sessions.append((session, set(positions)))

    monkeypatch.setattr(session_type, "_finish_preflight", finished)
    target = numpy_checkpoint(4, initialized=False)
    for state in target.states:
        for component in state.components:
            component.accessor = _TrackedTarget(component.accessor, writes)
    assert_ok(collective(core.load, tmp_path, target, backend=backend))
    assert writes.is_set() and parsed and len(sessions) == 4
    assert_values(target, source)
    count = len(parsed)
    for session, positions in sessions:
        assert set(session.offsets) == positions
        off_plan = next(pos for pos in session.index.slots if pos not in positions)
        assert type(session.get_file_offset(off_plan)) is gb.Err
    assert len(parsed) == count
