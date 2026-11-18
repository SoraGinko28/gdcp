"""Torch-free GDCP policy, manifest, resharding and checkpoint transactions."""

import builtins
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

import gdtensor.builder as gb
from gdcp import core, desc
from gdcp.backend import SafeTensorsBackend
from gdcp.state import CheckpointState
from gdtensor_conn.iobackend import FileIO
from tests.support.gdcp import assert_ok, assert_values, collective, numpy_checkpoint


class FailOnceIO(FileIO):
    def __init__(self, phase="payload"):
        super().__init__()
        self.armed = True
        self.phase = phase

    def pwrite(self, rank, path, offset, data):
        selected = offset == 0 if self.phase == "header" else offset > 0
        if self.armed and rank == 1 and selected:
            self.armed = False
            raise OSError(f"injected rank-1 {self.phase} write failure")
        return super().pwrite(rank, path, offset, data)


class CountingIO(FileIO):
    def __init__(self):
        super().__init__()
        self.reads = []

    def pread(self, rank, path, offset, size):
        self.reads.append(str(path))
        return super().pread(rank, path, offset, size)

    def read_all(self, rank, path):
        self.reads.append(str(path))
        return super().read_all(rank, path)


@pytest.mark.parametrize("target_world", [2, 3])
def test_numpy_collective_roundtrip_and_reshard(tmp_path, target_world):
    source = numpy_checkpoint(2, initialized=True, names=("model", "optimizer"))
    assert_ok(collective(core.save, tmp_path, source))
    target = numpy_checkpoint(
        target_world, initialized=False, names=("optimizer", "model")
    )
    assert_ok(collective(core.load, tmp_path, target))
    assert_values(target, source)
    assert (
        target.states[0].components[0].objects == source.states[0].components[0].objects
    )
    assert (tmp_path / "model" / "__0_0.safetensors").is_file()
    assert (tmp_path / "optimizer" / "objects.json").is_file()


def test_replicated_tensors_have_exactly_one_storage_slot(tmp_path):
    fixture = numpy_checkpoint(3, initialized=True)
    assert_ok(collective(core.save, tmp_path, fixture))
    component = json.loads((tmp_path / desc.MANIFEST_NAME).read_text())["components"][
        "model"
    ]
    for key in ("buffer", "step"):
        occurrences = sum(
            slot[0] == key
            for file in component["files"].values()
            for slot in file["slots"]
        )
        assert occurrences == 1, (key, component)


@pytest.mark.parametrize("phase", ["header", "payload"])
def test_fault_on_one_rank_aborts_every_rank_and_retry_succeeds(tmp_path, phase):
    fixture = numpy_checkpoint(2, initialized=True)
    assert_ok(collective(core.save, tmp_path, fixture))
    failing_io = FailOnceIO(phase)
    results = collective(
        core.save, tmp_path, fixture, backend=SafeTensorsBackend(failing_io)
    )
    assert all(type(result) is gb.Err for result in results), results
    assert (tmp_path / desc.MANIFEST_NAME).exists()
    target = numpy_checkpoint(2, initialized=False)
    assert_ok(collective(core.load, tmp_path, target))
    assert_values(target, fixture)
    assert_ok(
        collective(core.save, tmp_path, fixture, backend=SafeTensorsBackend(failing_io))
    )
    assert_ok(collective(core.load, tmp_path, target))
    assert_values(target, fixture)


def _change_schema(fixture, mismatch):
    tensors = fixture.table.named_d_tensors
    if mismatch == "missing":
        tensors.pop(0)
    elif mismatch == "unexpected":
        tensors.append(gb.NamedDTensor("unexpected.weight", tensors[0].d_tensor))
    elif mismatch == "shape":
        tensors[0].d_tensor.global_tensor_meta.global_shape = [11, 4]
    else:
        tensors[0].d_tensor.global_tensor_meta.dtype = "I32"


def _buffers(fixture):
    return {
        (component, rank, key): data.copy()
        for component, fabric in fixture.fabrics.items()
        for rank, tensors in fabric.buffers.items()
        for key, data in tensors.items()
    }


@pytest.mark.parametrize("mismatch", ["missing", "unexpected", "shape", "dtype"])
def test_strict_preflight_precedes_all_component_reads_and_writes(tmp_path, mismatch):
    source = numpy_checkpoint(2, initialized=True, names=("model", "optimizer"))
    assert_ok(collective(core.save, tmp_path, source))
    target = numpy_checkpoint(2, initialized=False, names=("optimizer", "model"))
    before = _buffers(target)
    _change_schema(target, mismatch)
    counting_io = CountingIO()
    results = collective(
        core.load, tmp_path, target, backend=SafeTensorsBackend(counting_io)
    )
    assert all(type(result) is gb.Err for result in results), results
    expected_key = "unexpected.weight" if mismatch == "unexpected" else "weight"
    assert all(expected_key in str(result.value) for result in results)
    assert counting_io.reads == []
    after = _buffers(target)
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])


def test_model_only_load_from_full_checkpoint(tmp_path):
    source = numpy_checkpoint(2, initialized=True, names=("model", "optimizer"))
    assert_ok(collective(core.save, tmp_path, source))
    target = numpy_checkpoint(3, initialized=False)
    assert_ok(collective(core.load, tmp_path, target))
    assert_values(target, source)


def test_inspect_reads_only_manifest(tmp_path, monkeypatch):
    fixture = numpy_checkpoint(2, initialized=True)
    assert_ok(collective(core.save, tmp_path, fixture))
    original_open, original_io_open = builtins.open, io.open

    def guarded_open(file, *args, **kwargs):
        assert not str(file).endswith(".safetensors"), file
        return original_open(file, *args, **kwargs)

    def guarded_io_open(file, *args, **kwargs):
        assert not str(file).endswith(".safetensors"), file
        return original_io_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    inspected = core.inspect(tmp_path)
    assert type(inspected) is gb.Ok, inspected
    assert inspected.value is not None
    assert inspected.value.world_size == 2
    tensors = {
        item["id"]: item for item in inspected.value.components["model"]["tensors"]
    }
    assert tensors["weight"]["shape"] == [10, 4]
    assert tensors["bf16"]["dtype"] == "BF16"
    assert tensors["step"]["logical_shape"] == []
    assert set(tensors) == {
        item.global_tensor_id for item in fixture.table.named_d_tensors
    }


def test_opaque_rank_state_files_and_rank_zero_fallback(tmp_path, caplog):
    source = numpy_checkpoint(2, initialized=True)
    for rank, state in enumerate(source.states):
        state.rank_state["extra_state"] = f"opaque-rank-{rank}".encode()
    assert_ok(collective(core.save, tmp_path, source))
    target = numpy_checkpoint(3, initialized=False)
    for state in target.states:
        state.rank_state["extra_state"] = b""
    assert_ok(collective(core.load, tmp_path, target))
    assert [state.rank_state["extra_state"] for state in target.states] == [
        b"opaque-rank-0",
        b"opaque-rank-1",
        b"opaque-rank-0",
    ]
    assert "rank 2" in caplog.text and "rank 0" in caplog.text
    assert (
        tmp_path / "extra_state" / "extra_state_rank_0.pt"
    ).read_bytes() == b"opaque-rank-0"


@pytest.mark.parametrize("missing_rank", [0, 1])
def test_missing_rank_state_file_falls_back_or_fails_collectively(
    tmp_path, caplog, missing_rank
):
    source = numpy_checkpoint(2, initialized=True)
    for rank, state in enumerate(source.states):
        state.rank_state["extra_state"] = f"rank-{rank}".encode()
    assert_ok(collective(core.save, tmp_path, source))
    (tmp_path / "extra_state" / f"extra_state_rank_{missing_rank}.pt").unlink()
    target = numpy_checkpoint(2, initialized=False)
    for state in target.states:
        state.rank_state["extra_state"] = b""
    before = _buffers(target)
    results = collective(core.load, tmp_path, target)
    if missing_rank == 1:
        assert_ok(results)
        assert all(
            state.rank_state["extra_state"] == b"rank-0" for state in target.states
        )
        assert "rank 1" in caplog.text and "rank 0" in caplog.text
        return
    assert all(type(result) is gb.Err for result in results), results
    after = _buffers(target)
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])


def test_non_strict_load_is_rejected_before_reading_payload(tmp_path):
    source = numpy_checkpoint(2, initialized=True)
    assert_ok(collective(core.save, tmp_path, source))
    target = numpy_checkpoint(2, initialized=False)
    before = _buffers(target)
    counting_io = CountingIO()
    results = collective(
        core.load,
        tmp_path,
        target,
        backend=SafeTensorsBackend(counting_io),
        strict=False,
    )
    assert all(type(result) is gb.Err for result in results), results
    assert all("strict" in str(result.value) for result in results)
    assert counting_io.reads == []
    after = _buffers(target)
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])


def _keep_empty_only(fixture):
    fixture.table.named_d_tensors = [
        tensor
        for tensor in fixture.table.named_d_tensors
        if tensor.global_tensor_id == "empty"
    ]
    for state in fixture.states:
        state.components[0].objects.clear()


def test_empty_only_component_roundtrip_has_semantics_without_payload_files(tmp_path):
    source = numpy_checkpoint(2, initialized=True)
    _keep_empty_only(source)
    assert_ok(collective(core.save, tmp_path, source))
    inspected = core.inspect(tmp_path)
    assert type(inspected) is gb.Ok, inspected
    assert inspected.value is not None
    component = inspected.value.components["model"]
    assert component["files"] == {}
    assert [(tensor["id"], tensor["shape"]) for tensor in component["tensors"]] == [
        ("empty", [0, 3])
    ]
    target = numpy_checkpoint(3, initialized=False)
    _keep_empty_only(target)
    assert_ok(collective(core.load, tmp_path, target))
    assert all(
        buffers["empty"].nbytes == 0
        for buffers in target.fabrics["model"].buffers.values()
    )
    assert not list(tmp_path.rglob("*.safetensors"))


def test_extra_only_core_roundtrip(tmp_path):
    source = SimpleNamespace(
        states=[
            CheckpointState(rank_state={"extra_state": f"opaque-{rank}".encode()})
            for rank in range(2)
        ]
    )
    assert_ok(collective(core.save, tmp_path, source))
    target = SimpleNamespace(
        states=[CheckpointState(rank_state={"extra_state": b""}) for _ in range(2)]
    )
    assert_ok(collective(core.load, tmp_path, target))
    assert [state.rank_state for state in target.states] == [
        state.rank_state for state in source.states
    ]
    inspected = core.inspect(tmp_path)
    assert type(inspected) is gb.Ok, inspected
    assert inspected.value is not None
    assert inspected.value.components == {}
    assert inspected.value.rank_state["extra_state"]["ranks"] == [0, 1]
    assert not list(tmp_path.rglob("*.safetensors"))
