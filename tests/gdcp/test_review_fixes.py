"""Foreign entries survive pathname replacement and blocked rollback."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gdcp import _transaction, hf
from gdcp.state import RankContext
from gdtensor.builder import Err
from gdtensor_conn.coord import ThreadCoordinator
from gdtensor_conn.iobackend import FileIO
from tests.support.gdcp import assert_ok
from tests.support.gdcp_overwrite import _assert_generation, _fixture, _operation, _run

_PAYLOADS = [
    ("safetensors", "model/__0_0.safetensors"),
    ("dcp", "model/__0_0.distcp"),
    ("hf", "model.safetensors"),
]


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
