"""Regression tests for collective checkpoint I/O and directory transactions."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local

import pytest

from gdcp import _transaction, core, desc, hf
from gdcp.backend import SafeTensorsBackend
from gdcp.policy import PerRankPolicy
from gdcp.state import RankContext
from gdtensor.builder import Err
from gdtensor_conn.coord import ThreadCoordinator
from gdtensor_conn.iobackend import FileIO
from tests.support.gdcp import assert_ok
from tests.support.gdcp_overwrite import (
    _assert_generation,
    _files,
    _fixture,
    _operation,
    _run,
)

_PAYLOADS = [
    ("safetensors", "model/__0_0.safetensors"),
    ("dcp", "model/__0_0.distcp"),
    ("hf", "model.safetensors"),
]
_FORMATS = [fmt for fmt, _ in _PAYLOADS]


def _parallel(call, worlds=(2, 2)):
    coordinator = ThreadCoordinator(2, timeout=2)

    def invoke(rank):
        return call(RankContext(rank, worlds[rank], coordinator))

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(invoke, range(2)))


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("names", [("Upper", "upper"), ("é", "e\u0301")])
def test_rank_state_filesystem_aliases_fail_before_payloads(
    tmp_path, monkeypatch, fmt, names
):
    probe = tmp_path / names[0]
    probe.mkdir()
    if not (tmp_path / names[1]).exists():
        pytest.skip("checkpoint filesystem distinguishes these names")
    probe.rmdir()
    root = tmp_path / "checkpoint"
    old, source = _fixture(), _fixture(shift=7)
    assert_ok(_run(fmt, root, old))
    for rank, state in enumerate(source.states):
        state.rank_state = {name: f"{name}-{rank}".encode() for name in names}
    writes = []
    write = core._save_rank_state

    def track_write(*args):
        writes.append(True)
        return write(*args)

    with monkeypatch.context() as patch:
        patch.setattr(core, "_save_rank_state", track_write)
        results = _run(fmt, root, source)
    assert all(type(result) is Err for result in results), results
    assert any(all(name in result.value.msg for name in names) for result in results)
    assert not writes
    _assert_generation(fmt, root, old, world=2)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_rank_state_serialization_failure_votes_before_broadcast(tmp_path, fmt):
    class RankKey(str):
        def __lt__(self, other):
            raise ValueError("cannot order rank keys")

    fixture = _fixture()
    fixture.states[0].rank_state = {RankKey(name): b"" for name in ("one", "two")}
    fixture.states[1].rank_state = {name: b"" for name in ("one", "two")}
    call, options = _operation(fmt, False, fixture)
    root = tmp_path / "checkpoint"
    results = _parallel(
        lambda ctx: call(root, fixture.states[ctx.rank], ctx, **options)
    )
    assert all(type(result) is Err for result in results), results
    assert "cannot order rank keys" in results[0].value.msg
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("phase", ["commit", "abort"])
def test_descriptor_close_failure_coordinates_save(tmp_path, monkeypatch, fmt, phase):
    root = tmp_path / "checkpoint"
    fixture = _fixture()
    call, options = _operation(fmt, False, fixture)
    close, write = FileIO.close_rank, FileIO.pwrite

    def fail_close(io, rank, prefix=None):
        close(io, rank, prefix)
        if rank == 0:
            raise OSError("injected descriptor close failure")

    def fail_write(io, rank, path, offset, data):
        if rank == 0 and offset > 0:
            raise OSError("injected payload write failure")
        return write(io, rank, path, offset, data)

    monkeypatch.setattr(FileIO, "close_rank", fail_close)
    if phase == "abort":
        monkeypatch.setattr(FileIO, "pwrite", fail_write)
    results = _parallel(
        lambda ctx: call(root, fixture.states[ctx.rank], ctx, **options)
    )
    assert all(type(result) is Err for result in results), results
    failure = "descriptor close" if phase == "commit" else "payload write"
    assert f"injected {failure} failure" in results[0].value.msg
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fmt", ["safetensors", "dcp"])
@pytest.mark.parametrize("mismatch", ["missing", "order"])
@pytest.mark.parametrize("loading", [False, True])
def test_component_sequence_agreement_precedes_io(tmp_path, fmt, mismatch, loading):
    root = tmp_path / "checkpoint"
    source, target = _fixture(), _fixture(shift=7)
    if loading:
        assert_ok(_run(fmt, root, source))
    call, options = _operation(fmt, loading, target)
    components = target.states[1].components
    if mismatch == "missing":
        components.pop(0)
    else:
        components.reverse()
    model = target.fabrics["model"].buffers[0]["weight"]
    before = model.tobytes()
    results = _parallel(lambda ctx: call(root, target.states[ctx.rank], ctx, **options))
    assert all(type(result) is Err for result in results), results
    assert all("component" in result.value.msg for result in results)
    assert model.tobytes() == before
    if not loading:
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fmt", _FORMATS)
def test_save_plan_failure_votes_before_opening_sessions(tmp_path, monkeypatch, fmt):
    fixture = _fixture()
    call, options = _operation(fmt, False, fixture)
    root = tmp_path / "checkpoint"
    save = core.rank_save

    def fail_plan(rank, compute, storage):
        if rank == 0:
            return core._error("save plan", ValueError("injected save plan failure"))
        return save(rank, compute, storage)

    monkeypatch.setattr(core, "rank_save", fail_plan)
    results = _parallel(
        lambda ctx: call(root, fixture.states[ctx.rank], ctx, **options)
    )
    assert all(type(result) is Err for result in results), results
    assert "injected save plan failure" in results[0].value.msg
    assert not list(tmp_path.iterdir())


def test_transaction_close_failure_returns_error_on_every_rank(tmp_path, monkeypatch):
    fixture = _fixture()
    close = _transaction.DirectoryTransaction.close
    root = tmp_path / "checkpoint"

    def fail_close(transaction):
        close(transaction)
        if transaction.rank == 0:
            raise OSError("injected transaction close failure")

    monkeypatch.setattr(_transaction.DirectoryTransaction, "close", fail_close)
    results = _parallel(lambda ctx: core.save(root, fixture.states[ctx.rank], ctx))
    assert all(type(result) is Err for result in results), results
    assert "injected transaction close failure" in results[0].value.msg
    assert (root / desc.MANIFEST_NAME).is_file()


@pytest.mark.parametrize("fmt", [fmt for fmt, _ in _PAYLOADS])
@pytest.mark.parametrize("owner", [0, 1])
def test_rank_state_key_mismatch_fails_before_writing(
    tmp_path, monkeypatch, fmt, owner
):
    root = tmp_path / "checkpoint"
    source = _fixture()
    source.states[owner].rank_state["extraé"] = b"opaque"
    writes = []
    begin = _transaction.DirectoryTransaction.begin

    def track_begin(transaction, planned):
        writes.append(transaction.rank)
        return begin(transaction, planned)

    monkeypatch.setattr(_transaction.DirectoryTransaction, "begin", track_begin)
    results = _run(fmt, root, source)
    assert all(type(result) is Err for result in results), results
    assert results[0].value.msg == results[1].value.msg
    assert "extraé" in results[0].value.msg
    assert not writes
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fmt", [fmt for fmt, _ in _PAYLOADS])
def test_rank_state_inventory_and_larger_world_fallback(tmp_path, caplog, fmt):
    root = tmp_path / "checkpoint"
    source = _fixture()
    source.states[0].rank_state = {"second": b"zero", **source.states[0].rank_state}
    source.states[1].rank_state["second"] = b"one"
    assert_ok(_run(fmt, root, source))
    metadata = ".gdtensor_committed" if fmt == "hf" else desc.MANIFEST_NAME
    recorded = json.loads((root / metadata).read_bytes())["rank_state"]
    assert recorded.keys() == source.states[0].rank_state.keys()
    for name, info in recorded.items():
        assert info["ranks"] == [0, 1]
        files = {info["pattern"].format(rank=rank) for rank in info["ranks"]}
        assert {file.name for file in (root / name).iterdir()} == files
    target = _fixture(world=3, shift=7)
    for state in target.states:
        state.rank_state["second"] = b""
    assert_ok(_run(fmt, root, target, loading=True))
    assert [state.rank_state for state in target.states] == [
        source.states[rank].rank_state for rank in (0, 1, 0)
    ]
    assert "extra_state rank 2" in caplog.text


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("worlds", [(3, 2), (1, 2)])
def test_save_agrees_on_world_before_staging(tmp_path, monkeypatch, fmt, worlds):
    source = _fixture()
    call, options = _operation(fmt, False, source)
    for state in source.states:
        state.components.clear()
    root = tmp_path / "checkpoint"
    started = []
    begin = _transaction.DirectoryTransaction.begin

    def track_begin(transaction, planned):
        started.append(transaction.rank)
        return begin(transaction, planned)

    monkeypatch.setattr(_transaction.DirectoryTransaction, "begin", track_begin)
    results = _parallel(
        lambda ctx: call(root, source.states[ctx.rank], ctx, **options), worlds
    )
    assert all(type(result) is Err for result in results), results
    assert results[0].value.msg == results[1].value.msg
    assert "world size" in results[0].value.msg
    assert not started
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize(
    "ranks, conflict", [((0, 1, 1), False), ((0, 1, 1), True), ((0, 1, 0), False)]
)
def test_save_rejects_duplicate_ranks_before_staging(
    tmp_path, monkeypatch, fmt, ranks, conflict
):
    source = _fixture(world=len(ranks))
    call, options = _operation(fmt, False, source)
    for state in source.states:
        state.components.clear()
        state.rank_state.clear()
    if conflict:
        source.states[2].rank_state["extra"] = b"different keys"
    root = tmp_path / "checkpoint"
    started = []
    begin = _transaction.DirectoryTransaction.begin
    coordinator = ThreadCoordinator(len(ranks), timeout=2)

    def track_begin(transaction, planned):
        started.append(transaction.rank)
        return begin(transaction, planned)

    def save_participant(index):
        ctx = RankContext(ranks[index], 2, coordinator)
        return call(root, source.states[index], ctx, **options)

    monkeypatch.setattr(_transaction.DirectoryTransaction, "begin", track_begin)
    with ThreadPoolExecutor(max_workers=len(ranks)) as pool:
        results = list(pool.map(save_participant, range(len(ranks))))
    assert not started
    assert all(type(result) is Err for result in results), results
    assert len({result.value.msg for result in results}) == 1
    assert f"duplicate rank {ranks[-1]}" in results[0].value.msg
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("field", ["rank_state keys", "rank configuration"])
def test_save_rejects_mismatch_restored_by_later_phase(
    tmp_path, monkeypatch, fmt, field
):
    root = tmp_path / "checkpoint"
    old, source = _fixture(), _fixture(shift=7)
    assert_ok(_run(fmt, root, old))
    original_files = _files(root)
    call, options = _operation(fmt, False, source)
    state = source.states[1]
    coordinator = ThreadCoordinator(2, timeout=2)
    contexts = [RankContext(rank, 2, coordinator) for rank in range(2)]
    original_rank_state = dict(state.rank_state)
    observed = []

    def change():
        observed.append("changed")
        if field == "rank_state keys":
            state.rank_state.clear()
        else:
            contexts[1].world_size = 3

    def restore():
        observed.append("restored")
        state.rank_state.update(original_rank_state)
        contexts[1].world_size = 2

    def save_rank(rank):
        rank_options = dict(options)
        if fmt == "hf" and rank == 1:
            rank_options["_write_metadata"] = lambda _path: change()
        return call(root, source.states[rank], contexts[rank], **rank_options)

    model, optimizer = state.components
    if fmt != "hf":
        monkeypatch.setattr(model.accessor, "finish", change, raising=False)
    monkeypatch.setattr(optimizer.accessor, "finish", restore, raising=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save_rank, range(2)))
    assert observed == ["changed"]
    assert all(type(result) is Err for result in results), results
    assert results[0].value.msg == results[1].value.msg
    assert field in results[0].value.msg
    assert _files(root) == original_files
    assert list(tmp_path.iterdir()) == [root]


def test_hf_save_latches_mismatch_between_accessor_finishes(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    old, source = _fixture(), _fixture(shift=7)
    assert_ok(_run("hf", root, old))
    original_files = _files(root)
    state = source.states[1]
    original_rank_state = dict(state.rank_state)
    model, optimizer = state.components
    monkeypatch.setattr(model.accessor, "finish", state.rank_state.clear, raising=False)
    monkeypatch.setattr(
        optimizer.accessor,
        "finish",
        lambda: state.rank_state.update(original_rank_state),
        raising=False,
    )

    results = _run("hf", root, source)
    assert all(type(result) is Err for result in results), results
    assert results[0].value.msg == results[1].value.msg
    assert "rank_state keys" in results[0].value.msg
    assert _files(root) == original_files
    assert list(tmp_path.iterdir()) == [root]


@pytest.mark.parametrize("fmt", _FORMATS)
def test_save_rejects_keys_changed_by_accessor(tmp_path, monkeypatch, fmt):
    root = tmp_path / "checkpoint"
    old, source = _fixture(), _fixture(shift=7)
    assert_ok(_run(fmt, root, old))
    state = source.states[1]
    monkeypatch.setattr(
        state.components[0].accessor, "finish", state.rank_state.clear, raising=False
    )
    results = _run(fmt, root, source)
    assert all(type(result) is Err for result in results), results
    assert results[0].value.msg == results[1].value.msg
    assert "rank_state keys" in results[0].value.msg
    _assert_generation(fmt, root, old, world=2)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_save_uses_agreed_rank_state_bytes(tmp_path, monkeypatch, fmt):
    root = tmp_path / "checkpoint"
    source = _fixture()
    initialize = core._initialize_directories

    def change_values(path, names):
        initialize(path, names)
        for state in source.states:
            state.rank_state["extra_state"] = b"changed after agreement"

    monkeypatch.setattr(core, "_initialize_directories", change_values)
    assert_ok(_run(fmt, root, source))
    for rank in range(2):
        payload = root / "extra_state" / f"extra_state_rank_{rank}.pt"
        assert payload.read_bytes() == f"0-{rank}".encode()


@pytest.mark.parametrize("fmt", _FORMATS)
def test_save_rejects_incomplete_rank_state_inventory(tmp_path, monkeypatch, fmt):
    root = tmp_path / "checkpoint"
    source = _fixture()
    write = desc.atomic_write
    missing = "extra_state_rank_1.pt"

    def discard_rank_file(path, data):
        result = write(path, data)
        if path.name == missing:
            path.unlink()
        return result

    monkeypatch.setattr(desc, "atomic_write", discard_rank_file)
    results = _run(fmt, root, source)
    assert all(type(result) is Err for result in results), results
    assert all(missing in result.value.msg for result in results)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("truth", [False, None])
def test_core_uses_supplied_options_without_testing_truth(tmp_path, truth):
    root = tmp_path / "checkpoint"
    source, target = _fixture(), _fixture(shift=7)
    plans, opens = [], []

    def option_truth(self):
        if truth is None:
            raise OSError("injected option truth-value failure")
        return truth

    class Policy(PerRankPolicy):
        __bool__ = option_truth

        def plan(self, component, world_size):
            plans.append(component.name)
            return super().plan(component, world_size)

    class Backend(SafeTensorsBackend):
        __bool__ = option_truth

        def open(self, root, storage_table, layout, coordinator=None):
            opens.append(root.name)
            return super().open(root, storage_table, layout, coordinator)

    def save(ctx):
        return core.save(
            root,
            source.states[ctx.rank],
            ctx,
            policy=Policy() if ctx.rank == 0 else None,
            backend=Backend() if ctx.rank == 0 else None,
        )

    assert_ok(_parallel(save))
    assert plans == opens == ["model", "optimizer"]
    opens.clear()
    assert_ok(
        _parallel(
            lambda ctx: core.load(
                root,
                target.states[ctx.rank],
                ctx,
                backend=Backend() if ctx.rank == 0 else None,
            )
        )
    )
    assert opens == ["model", "optimizer"]


@pytest.mark.parametrize("loading", [False, True])
def test_core_votes_on_option_preparation_failure(tmp_path, monkeypatch, loading):
    root = tmp_path / "checkpoint"
    source = _fixture()
    if loading:
        assert_ok(_run("safetensors", root, source))
    call = core.load if loading else core.save

    def fail_backend():
        raise OSError("injected option preparation failure")

    monkeypatch.setattr(core, "SafeTensorsBackend", fail_backend)
    results = _parallel(
        lambda ctx: call(
            root,
            source.states[ctx.rank],
            ctx,
            backend=None if ctx.rank == 0 else SafeTensorsBackend(),
        )
    )
    assert all(type(result) is Err for result in results), results
    assert "injected option preparation failure" in results[0].value.msg
    assert root.exists() == loading


@pytest.mark.parametrize("fmt", [fmt for fmt, _ in _PAYLOADS])
@pytest.mark.parametrize("world", [1, 2])
def test_load_rejects_missing_declared_rank_file(tmp_path, fmt, world):
    root = tmp_path / "checkpoint"
    assert_ok(_run(fmt, root, _fixture()))
    filename = "extra_state_rank_1.pt"
    (root / "extra_state" / filename).unlink()
    target = _fixture(world=world, shift=7)
    results = _run(fmt, root, target, loading=True)
    assert all(type(result) is Err for result in results), results
    assert all(filename in result.value.msg for result in results)
    for rank, state in enumerate(target.states):
        assert state.rank_state == {"extra_state": f"7-{rank}".encode()}
    for name, fabric in target.fabrics.items():
        fabric.assert_values(target.values[name])


@pytest.mark.parametrize("fmt", _FORMATS)
def test_extra_missing_worker_rejects_same_world_load(tmp_path, monkeypatch, fmt):
    torch = pytest.importorskip("torch")
    from gdcp.adapters import fsdp2
    from tests.support.gdcp_fsdp2 import assert_missing_extra_state_rejected

    source = tmp_path / "source"
    assert_ok(_run(fmt, source, _fixture()))
    for rank in range(2):
        torch.save(
            {"rank": rank}, source / "extra_state" / f"extra_state_rank_{rank}.pt"
        )
    current = local()
    monkeypatch.setattr(fsdp2, "_context", lambda: current.ctx)

    def check_rank(ctx):
        current.ctx = ctx
        assert ctx.coordinator is not None
        assert_missing_extra_state_rejected(
            source, tmp_path / "missing", ctx.rank, ctx.coordinator
        )

    _parallel(check_rank)


def test_hf_metadata_hook_uses_the_same_vote_on_every_rank(tmp_path):
    source = _fixture()
    root = tmp_path / "checkpoint"
    coordinator = ThreadCoordinator(2, timeout=2)

    def save(rank):
        return hf.save_state(
            root,
            source.states[rank],
            RankContext(rank, 2, coordinator),
            _write_metadata=None if rank else lambda path: None,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert_ok(list(pool.map(save, range(2))))


@pytest.mark.parametrize(("fmt", "name"), _PAYLOADS)
@pytest.mark.parametrize("replace_before_move", [False, True])
def test_obsolete_file_replaced_during_cleanup(
    tmp_path, monkeypatch, caplog, fmt, name, replace_before_move
):
    root = tmp_path / "checkpoint"
    old, new = _fixture(world=1), _fixture(world=1, shift=7)
    assert_ok(_run(fmt, root, old))
    obsolete = _transaction.DirectoryTransaction(root, 0).backup
    old_file = obsolete / name
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"foreign replacement")
    inode = replacement.stat().st_ino
    unlink, rename = Path.unlink, Path.rename
    injected = []

    def replace():
        os.replace(replacement, old_file)
        injected.append(True)

    def unlink_file(path, *args, **kwargs):
        if (
            old_file.parent in path.parents
            and path.name == old_file.name
            and not injected
        ):
            replace()
        return unlink(path, *args, **kwargs)

    def move_file(path, target):
        if path == old_file and replace_before_move and not injected:
            replace()
        return rename(path, target)

    monkeypatch.setattr(_transaction, "_exchange", lambda *_: False)
    monkeypatch.setattr(Path, "unlink", unlink_file)
    monkeypatch.setattr(Path, "rename", move_file)
    assert_ok(_run(fmt, root, new))
    assert injected == [True]
    assert old_file.read_bytes() == b"foreign replacement"
    assert old_file.stat().st_ino == inode
    assert str(obsolete) in caplog.text
    _assert_generation(fmt, root, new, world=1)


@pytest.mark.parametrize("fmt", [fmt for fmt, _ in _PAYLOADS])
def test_published_journal_replaced_during_obsolete_cleanup(
    tmp_path, monkeypatch, caplog, fmt
):
    root = tmp_path / "checkpoint"
    old, new = _fixture(world=1), _fixture(world=1, shift=7)
    assert_ok(_run(fmt, root, old))
    journal = root / _transaction.JOURNAL_NAME
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"foreign journal")
    inode = replacement.stat().st_ino
    clean_old = _transaction._clean_old
    injected = []

    def replace_journal(obsolete, target, *args):
        assert journal.is_file()
        os.replace(replacement, journal)
        injected.append(True)
        return clean_old(obsolete, target, *args)

    monkeypatch.setattr(_transaction, "_clean_old", replace_journal)
    assert_ok(_run(fmt, root, new))
    assert injected == [True]
    assert journal.read_bytes() == b"foreign journal"
    assert journal.stat().st_ino == inode
    assert str(journal) in caplog.text
    _assert_generation(fmt, root, new, world=1)


def test_late_hf_index_stays_in_obsolete_sibling(tmp_path, monkeypatch, caplog):
    root = tmp_path / "checkpoint"
    old, new = _fixture(world=1), _fixture(world=1, shift=7)
    assert_ok(_run("hf", root, old))
    assert (root / "model.safetensors").is_file()
    index = root / hf.HF_INDEX_NAME
    obsolete = _transaction.DirectoryTransaction(root, 0).backup

    def after_scan(_stage, _target):
        index.write_bytes(b"foreign index")
        return False

    monkeypatch.setattr(_transaction, "_exchange", after_scan)
    assert_ok(_run("hf", root, new))
    assert not index.exists()
    assert (obsolete / index.name).read_bytes() == b"foreign index"
    assert str(obsolete) in caplog.text and index.name in caplog.text
    _assert_generation("hf", root, new, world=1)


@pytest.mark.parametrize("fmt", [fmt for fmt, _ in _PAYLOADS])
def test_rollback_collision_preserves_both_logs_and_propagates_interrupt(
    tmp_path, monkeypatch, caplog, fmt
):
    root = tmp_path / "checkpoint"
    old, new = _fixture(), _fixture(shift=7)
    assert_ok(_run(fmt, root, old))
    log = root / "late.log"
    stage = _transaction.DirectoryTransaction(root, 0).stage
    write = FileIO.pwrite
    interrupt = KeyboardInterrupt("during exchange")
    created = []

    def write_with_log(self, rank, path, offset, data):
        if rank == 0 and not created:
            log.write_bytes(b"first log")
            created.append(True)
        return write(self, rank, path, offset, data)

    def abort_exchange(_stage, _target):
        log.write_bytes(b"second log")
        raise interrupt

    call, options = _operation(fmt, False, new)
    coordinator = ThreadCoordinator(2, timeout=30)

    def save(rank):
        try:
            return call(
                root, new.states[rank], RankContext(rank, 2, coordinator), **options
            )
        except (Exception, KeyboardInterrupt) as exc:
            return exc

    with monkeypatch.context() as patch:
        patch.setattr(FileIO, "pwrite", write_with_log)
        patch.setattr(_transaction, "_exchange", abort_exchange)
        with ThreadPoolExecutor(max_workers=2) as pool:
            rank_zero, rank_one = pool.map(save, range(2))
    assert created == [True]
    assert log.read_bytes() == b"second log"
    assert (stage / log.name).read_bytes() == b"first log"
    _assert_generation(fmt, root, old, world=2)
    assert rank_zero is interrupt
    assert type(rank_one) is Err
    assert str(stage) in caplog.text and log.name in caplog.text
