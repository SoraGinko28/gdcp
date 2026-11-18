"""Directory overwrite ownership, preflight, recovery and symlink regressions."""

import builtins
import json
import os
import pickle
from pathlib import Path

import pytest

from gdcp import _transaction, desc, hf, inspect
from gdcp.state import RankContext
from gdtensor.builder import Err
from gdtensor_conn.iobackend import FileIO
from tests.support.gdcp_overwrite import (
    _assert_generation,
    _fail_rank_one,
    _files,
    _fixture,
    _run,
)
from tests.support.gdcp import assert_ok

_FORMATS = ["safetensors", "dcp", "hf"]


def _foreign(root, fmt):
    (root / "train.log").write_bytes(b"training\x00log\xff\n")
    (root / "log-link").symlink_to("train.log")
    nested = root / "global_step_100"
    fixture = _fixture(world=1, shift=100)
    assert_ok(_run(fmt, nested, fixture))
    return (
        fixture,
        _files(nested),
        (root / "train.log").stat().st_ino,
        nested.stat().st_ino,
    )


def _assert_foreign(root, fmt, foreign):
    fixture, files, inode, nested_inode = foreign
    assert (root / "train.log").read_bytes() == b"training\x00log\xff\n"
    assert (root / "train.log").stat().st_ino == inode
    assert (root / "log-link").is_symlink()
    assert os.readlink(root / "log-link") == "train.log"
    assert (root / "global_step_100").stat().st_ino == nested_inode
    assert _files(root / "global_step_100") == files
    _assert_generation(fmt, root / "global_step_100", fixture, world=1)


def _assert_errors(results, match):
    assert all(type(result) is Err for result in results), results
    assert any(match in result.value.msg for result in results), results


def _assert_only_target(parent):
    assert sorted(item.name for item in parent.iterdir()) == ["checkpoint"]


@pytest.mark.parametrize("fmt", _FORMATS)
def test_foreign_file_checkpoint_directory_and_symlink_survive_success(tmp_path, fmt):
    root = tmp_path / "checkpoint"
    assert_ok(_run(fmt, root, _fixture()))
    foreign = _foreign(root, fmt)
    new = _fixture(shift=7)
    assert_ok(_run(fmt, root, new))
    _assert_foreign(root, fmt, foreign)
    _assert_generation(fmt, root, new)
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_foreign_entries_survive_one_rank_failure(tmp_path, monkeypatch, fmt):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run(fmt, root, old))
    foreign = _foreign(root, fmt)
    before = _files(root)
    failures = _fail_rank_one(monkeypatch)
    _assert_errors(_run(fmt, root, _fixture(shift=7)), "failure")
    assert failures
    assert _files(root) == before
    _assert_foreign(root, fmt, foreign)
    _assert_generation(fmt, root, old)
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_first_save_in_noncheckpoint_directory_keeps_every_entry(tmp_path, fmt):
    root = tmp_path / "checkpoint"
    root.mkdir()
    foreign = _foreign(root, fmt)
    # Payload-looking names alone do not establish checkpoint ownership.
    (root / "__91_0.safetensors").write_bytes(b"foreign payload-like filename")
    new = _fixture()
    assert_ok(_run(fmt, root, new))
    _assert_foreign(root, fmt, foreign)
    assert (
        root / "__91_0.safetensors"
    ).read_bytes() == b"foreign payload-like filename"
    _assert_generation(fmt, root, new)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_unrecorded_entry_inside_component_is_foreign(tmp_path, fmt):
    root = tmp_path / "checkpoint"
    assert_ok(_run(fmt, root, _fixture()))
    directory = root if fmt == "hf" else root / "model"
    foreign = directory / "__99_0.safetensors"
    foreign.write_bytes(b"not in old checkpoint metadata")
    inode = foreign.stat().st_ino
    assert_ok(_run(fmt, root, _fixture(shift=7)))
    assert foreign.read_bytes() == b"not in old checkpoint metadata"
    assert foreign.stat().st_ino == inode


@pytest.mark.parametrize(
    ("fmt", "name"),
    [
        ("safetensors", "model/__0_0.safetensors"),
        ("dcp", "model/__0_0.distcp"),
        ("hf", "model.safetensors"),
    ],
)
def test_foreign_payload_name_collision_fails_before_any_payload(
    tmp_path, monkeypatch, fmt, name
):
    root = tmp_path / "checkpoint"
    collision = root / name
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"foreign bytes must survive")
    before = _files(root)
    writes = []
    original = FileIO.pwrite

    def record_write(self, rank, path, offset, data):
        writes.append(str(path))
        return original(self, rank, path, offset, data)

    monkeypatch.setattr(FileIO, "pwrite", record_write)
    _assert_errors(_run(fmt, root, _fixture()), Path(name).name)
    assert not writes
    assert _files(root) == before
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", ["safetensors", "dcp"])
def test_old_recorded_component_rank_and_object_files_are_removed(tmp_path, fmt):
    root = tmp_path / "checkpoint"
    assert_ok(_run(fmt, root, _fixture()))
    new = _fixture(world=1, shift=7)
    new.states[0].components = new.states[0].components[:1]
    new.states[0].components[0].objects.clear()
    new.states[0].rank_state.clear()
    assert_ok(_run(fmt, root, new))
    assert not (root / "optimizer").exists()
    assert not (root / "extra_state").exists()
    assert not (root / "model" / "objects.json").exists()
    suffix = "safetensors" if fmt == "safetensors" else "distcp"
    assert not (root / "model" / f"__1_0.{suffix}").exists()


def test_save_hf_keeps_tokenizer_and_readme(tmp_path):
    if not os.environ.get("GDTENSOR_TORCH_TESTS"):
        pytest.skip("set GDTENSOR_TORCH_TESTS=1")
    import torch
    from gdcp import load_hf, save_hf

    root = tmp_path / "checkpoint"
    root.mkdir()
    tokenizer, readme = b'{"tokens":["a","b"]}\n', b"# Training export\n"
    (root / "tokenizer.json").write_bytes(tokenizer)
    (root / "README.md").write_bytes(readme)
    model = torch.nn.Linear(4, 2)
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    save_hf(root, model)
    assert (root / "tokenizer.json").read_bytes() == tokenizer
    assert (root / "README.md").read_bytes() == readme
    target = torch.nn.Linear(4, 2)
    load_hf(root, target)
    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in expected.items()
    )


@pytest.mark.parametrize("fmt", _FORMATS)
def test_mount_target_is_rejected_collectively_before_payload(
    tmp_path, monkeypatch, fmt
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "foreign").write_bytes(b"keep")
    original = os.path.ismount
    monkeypatch.setattr(
        os.path, "ismount", lambda path: Path(path) == root or original(path)
    )
    writes = []
    monkeypatch.setattr(FileIO, "pwrite", lambda *args: writes.append(args))
    results = _run(fmt, root, _fixture())
    _assert_errors(results, "subdirectory of the mount")
    assert all("subdirectory of the mount" in result.value.msg for result in results)
    assert not writes
    assert _files(root) == {"foreign": b"keep"}
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("phase", ["payload", "publication"])
def test_keyboard_interrupt_rolls_back_and_next_save_succeeds(
    tmp_path, monkeypatch, fmt, existing, phase
):
    root = tmp_path / "checkpoint"
    old = _fixture(world=1)
    if existing:
        assert_ok(_run(fmt, root, old))
        (root / "foreign").write_bytes(b"unchanged")
    before = _files(root)
    original = (
        FileIO.pwrite
        if phase == "payload"
        else _transaction.DirectoryTransaction.publish
    )
    interrupted = []

    def interrupt(*args):
        result = original(*args)
        writing_payload = phase == "publication" or args[3] > 0
        if writing_payload and not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt("injected transaction interrupt")
        return result

    owner, method = (
        (FileIO, "pwrite")
        if phase == "payload"
        else (_transaction.DirectoryTransaction, "publish")
    )
    with monkeypatch.context() as patch:
        patch.setattr(owner, method, interrupt)
        with pytest.raises(KeyboardInterrupt, match="injected transaction interrupt"):
            _run(fmt, root, _fixture(world=1, shift=7))
    assert interrupted
    assert _files(root) == before
    assert root.exists() is existing
    assert sorted(item.name for item in tmp_path.iterdir()) == (
        ["checkpoint"] if existing else []
    )
    if existing:
        _assert_generation(fmt, root, old, world=1)
    new = _fixture(world=1, shift=7)
    assert_ok(_run(fmt, root, new))
    _assert_generation(fmt, root, new, world=1)
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_pending_stage_is_removed_with_warning_before_retry(tmp_path, caplog, fmt):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run(fmt, root, old))
    transaction = _transaction.DirectoryTransaction(root, 0)
    transaction.stage.mkdir()
    (transaction.stage / desc.PENDING_NAME).write_bytes(b"save interrupted")
    (transaction.stage / "partial-payload").write_bytes(b"unfinished")
    assert_ok(_run(fmt, root, _fixture(shift=7)))
    assert str(transaction.stage) in caplog.text
    assert "remov" in caplog.text.lower()
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_complete_stage_is_refused_unchanged(tmp_path, fmt):
    root = tmp_path / "checkpoint"
    assert_ok(_run(fmt, root, _fixture()))
    stage = _transaction.DirectoryTransaction(root, 0).stage
    assert_ok(_run(fmt, stage, _fixture(shift=7)))
    before = _files(stage)
    _assert_errors(_run(fmt, root, _fixture(shift=11)), str(stage))
    assert _files(stage) == before
    _assert_generation(fmt, stage, _fixture(shift=7))


@pytest.mark.parametrize("fmt", _FORMATS)
def test_backup_with_target_present_is_refused_unchanged(tmp_path, fmt):
    root = tmp_path / "checkpoint"
    assert_ok(_run(fmt, root, _fixture()))
    backup = _transaction.DirectoryTransaction(root, 0).backup
    assert_ok(_run(fmt, backup, _fixture(shift=7)))
    before = _files(backup)
    _assert_errors(_run(fmt, root, _fixture(shift=11)), str(backup))
    assert _files(backup) == before
    _assert_generation(fmt, backup, _fixture(shift=7))


@pytest.mark.parametrize("fmt", _FORMATS)
def test_first_save_and_overwrite_through_directory_symlink(tmp_path, fmt):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "checkpoint"
    link.symlink_to("real", target_is_directory=True)
    old = _fixture()
    assert_ok(_run(fmt, link, old))
    _assert_generation(fmt, link, old)
    _assert_generation(fmt, real, old)
    new = _fixture(shift=7)
    assert_ok(_run(fmt, link, new))
    assert link.is_symlink()
    assert os.readlink(link) == "real"
    _assert_generation(fmt, link, new)
    _assert_generation(fmt, real, new)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["checkpoint", "real"]


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("destination", ["absent", "file"])
def test_invalid_symlink_target_returns_clear_error(tmp_path, fmt, destination):
    referent = tmp_path / destination
    if destination == "file":
        referent.write_bytes(b"not a directory")
    link = tmp_path / "checkpoint"
    link.symlink_to(destination)
    _assert_errors(_run(fmt, link, _fixture()), "symlink")
    assert link.is_symlink()
    assert os.readlink(link) == destination
    assert not (tmp_path / ".checkpoint.gdcp-stage").exists()


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("existing", [False, True])
def test_filesystem_mismatch_is_rejected_collectively_before_payload(
    tmp_path, monkeypatch, fmt, existing
):
    root = tmp_path / "checkpoint"
    if existing:
        root.mkdir()
        (root / "foreign").write_bytes(b"keep")
    before = _files(root)
    checks, writes = [], []

    def reject_device(target, stage):
        checks.append((target, stage))
        raise OSError(
            "staging filesystem differs from target; pass a subdirectory of the mount"
        )

    monkeypatch.setattr(_transaction, "_check_filesystem", reject_device)
    monkeypatch.setattr(FileIO, "pwrite", lambda *args: writes.append(args))
    results = _run(fmt, root, _fixture())
    _assert_errors(results, "subdirectory of the mount")
    assert all("subdirectory of the mount" in result.value.msg for result in results)
    assert len(checks) == 1
    assert not writes
    assert _files(root) == before
    assert root.exists() is existing
    assert sorted(item.name for item in tmp_path.iterdir()) == (
        ["checkpoint"] if existing else []
    )


@pytest.mark.parametrize("fmt", _FORMATS)
def test_complete_backup_without_target_is_restored_before_failed_retry(
    tmp_path, monkeypatch, fmt
):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run(fmt, root, old))
    (root / "foreign").write_bytes(b"keep")
    backup = _transaction.DirectoryTransaction(root, 0).backup
    before = _files(root)
    root.rename(backup)
    failures = _fail_rank_one(monkeypatch)
    _assert_errors(_run(fmt, root, _fixture(shift=7)), "failure")
    assert failures
    assert _files(root) == before
    _assert_generation(fmt, root, old)
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_incomplete_backup_without_target_is_refused_unchanged(tmp_path, fmt):
    root = tmp_path / "checkpoint"
    backup = _transaction.DirectoryTransaction(root, 0).backup
    backup.mkdir()
    (backup / "partial").write_bytes(b"possibly valuable")
    _assert_errors(_run(fmt, root, _fixture()), str(backup))
    assert _files(backup) == {"partial": b"possibly valuable"}
    assert not root.exists()


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("pending", [False, True])
def test_unpublished_stage_carrying_foreign_entries_restores_them_before_retry(
    tmp_path, monkeypatch, caplog, fmt, pending
):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run(fmt, root, old))
    foreign = _foreign(root, fmt)
    stage = _transaction.DirectoryTransaction(root, 0).stage
    stage.mkdir()
    if pending:
        (stage / desc.PENDING_NAME).write_bytes(b"interrupted before publication")
    names = ["train.log", "log-link", "global_step_100"]
    journal = {
        "version": 1,
        "old_identity": _path_identity(root),
        "new_identity": _path_identity(stage),
        "foreign": [
            {"path": name, "identity": _path_identity(root / name)} for name in names
        ],
    }
    desc.atomic_write(stage / ".gdcp_transaction", desc.json_bytes(journal))
    for name in names:
        (root / name).rename(stage / name)
    failures = _fail_rank_one(monkeypatch)
    _assert_errors(_run(fmt, root, _fixture(shift=7)), "failure")
    assert failures
    _assert_foreign(root, fmt, foreign)
    _assert_generation(fmt, root, old)
    assert str(stage) in caplog.text
    _assert_only_target(tmp_path)


def _path_identity(path):
    info = path.lstat()
    return [info.st_dev, info.st_ino]


def test_hf_index_owns_only_recorded_shards_on_success(tmp_path):
    root = tmp_path / "checkpoint"
    ctx = RankContext(0, 1)
    old = _fixture(world=1)
    old.states[0].rank_state.clear()
    assert_ok([hf.save_state(root, old.states[0], ctx, max_shard_size=64)])
    index = root / "model.safetensors.index.json"
    assert index.exists()
    old_shards = {entry["file"] for entry in hf.read_headers(root).values()}
    assert len(old_shards) > 1
    # Standard HF index ownership also works for exports from other writers.
    (root / ".gdtensor_committed").unlink()
    foreign = root / "model-99999-of-99999.safetensors"
    foreign.write_bytes(b"unindexed foreign shard-like name")
    new = _fixture(world=1, shift=7)
    assert_ok([hf.save_state(root, new.states[0], ctx)])
    assert not index.exists()
    assert all(not (root / name).exists() for name in old_shards)
    assert foreign.read_bytes() == b"unindexed foreign shard-like name"
    _assert_generation("hf", root, new, world=1)


def test_native_dcp_metadata_owns_recorded_files_without_gdcp_manifest(tmp_path):
    root = tmp_path / "checkpoint"
    old = _fixture()
    for state in old.states:
        state.rank_state.clear()
    assert_ok(_run("dcp", root, old))
    (root / desc.MANIFEST_NAME).unlink()
    old_rank = root / "model" / "__1_0.distcp"
    assert old_rank.exists()
    foreign = root / "model" / "__99_0.distcp"
    foreign.write_bytes(b"unrecorded foreign DCP-like filename")
    new = _fixture(world=1, shift=7)
    assert_ok(_run("dcp", root, new))
    assert not old_rank.exists()
    assert foreign.read_bytes() == b"unrecorded foreign DCP-like filename"
    _assert_generation("dcp", root, new, world=1)


def _native_dcp_with_unrecognized_planner_data(root):
    if not os.environ.get("GDTENSOR_TORCH_TESTS"):
        pytest.skip("set GDTENSOR_TORCH_TESTS=1")
    from torch.distributed.checkpoint.filesystem import _StorageInfo
    from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex

    root.mkdir(parents=True)
    payload = b"keep these bytes"
    (root / "old.distcp").write_bytes(payload)
    # A range is harmless but requires resolving a global outside DCP's schema.
    metadata = Metadata(
        {},
        planner_data=range(1),
        storage_data={
            MetadataIndex("weight"): _StorageInfo("old.distcp", 0, len(payload))
        },
    )
    (root / ".metadata").write_bytes(pickle.dumps(metadata, protocol=4))


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("component", [False, True], ids=["root", "component"])
def test_unrecognized_native_dcp_metadata_claims_no_entries(tmp_path, fmt, component):
    root = tmp_path / "checkpoint"
    directory = root / "model" if component else root
    _native_dcp_with_unrecognized_planner_data(directory)
    before = _files(root)

    results = _run(fmt, root, _fixture(world=1))

    if component and fmt != "hf":
        _assert_errors(results, "conflicts with foreign entry:")
        assert _files(root) == before
    else:
        assert_ok(results)
        for name, data in before.items():
            assert (root / name).read_bytes() == data


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("failure", ["unreadable", "missing module"])
def test_native_dcp_metadata_read_errors_leave_all_entries_foreign(
    tmp_path, monkeypatch, fmt, failure
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    metadata = root / ".metadata"
    data = pickle.dumps(range, protocol=0).replace(
        b"__builtin__", b"missing_dcp_module"
    )
    metadata.write_bytes(data)
    reads, imports = [], []
    original_open, original_import = builtins.open, builtins.__import__

    def open_file(path, *args, **kwargs):
        if path == metadata:
            reads.append(path)
            # Inject the OS error so the test works with elevated privileges.
            if failure == "unreadable":
                raise PermissionError("unreadable metadata")
        return original_open(path, *args, **kwargs)

    def import_module(name, *args, **kwargs):
        if name == "missing_dcp_module":
            imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_file)
    monkeypatch.setattr(builtins, "__import__", import_module)
    assert type(inspect(root)) is Err  # Native DCP has no GDCP manifest.
    assert not reads
    assert_ok(_run(fmt, root, _fixture(world=1)))
    assert reads
    assert not imports
    assert metadata.read_bytes() == data


@pytest.mark.parametrize("fmt", _FORMATS)
def test_recovery_preserves_unrecognized_native_dcp_backup(tmp_path, fmt):
    root = tmp_path / "checkpoint"
    backup = tmp_path / ".checkpoint.gdcp-backup"
    _native_dcp_with_unrecognized_planner_data(backup)
    before = _files(backup)

    _assert_errors(
        _run(fmt, root, _fixture(world=1)), "unrecognized/incomplete directory"
    )
    assert _files(backup) == before
    assert not root.exists()


def test_native_torch_dcp_metadata_still_establishes_ownership(tmp_path):
    if not os.environ.get("GDTENSOR_TORCH_TESTS"):
        pytest.skip("set GDTENSOR_TORCH_TESTS=1")
    import torch
    from torch.distributed.checkpoint import save as native_save

    root = tmp_path / "checkpoint"
    native_save({"weight": torch.ones(2)}, checkpoint_id=root, no_dist=True)
    old_files = set(_files(root))
    assert ".metadata" in old_files

    assert_ok(_run("hf", root, _fixture(world=1)))
    assert old_files.isdisjoint(_files(root))


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("existing", [False, True])
def test_publication_interrupt_on_one_rank_coordinates_rollback(
    tmp_path, monkeypatch, fmt, existing
):
    root = tmp_path / "checkpoint"
    old = _fixture()
    if existing:
        assert_ok(_run(fmt, root, old))
        (root / "foreign").write_bytes(b"unchanged")
    before = _files(root)
    original = _transaction.DirectoryTransaction.publish

    def interrupt(transaction):
        original(transaction)
        if transaction.rank == 0:
            raise KeyboardInterrupt("injected rank-zero publication interrupt")

    with monkeypatch.context() as patch:
        patch.setattr(_transaction.DirectoryTransaction, "publish", interrupt)
        with pytest.raises(KeyboardInterrupt, match="rank-zero publication interrupt"):
            _run(fmt, root, _fixture(shift=7))
    assert _files(root) == before
    assert root.exists() is existing
    assert sorted(item.name for item in tmp_path.iterdir()) == (
        ["checkpoint"] if existing else []
    )
    if existing:
        _assert_generation(fmt, root, old)
    assert_ok(_run(fmt, root, _fixture(shift=7)))
    _assert_only_target(tmp_path)


def test_hf_rank_state_records_remove_old_files_but_keep_foreign_notes(tmp_path):
    root = tmp_path / "checkpoint"
    assert_ok(_run("hf", root, _fixture()))
    notes = root / "extra_state" / "notes.txt"
    notes.write_bytes(b"foreign rank-state directory notes")
    new = _fixture(world=1, shift=7)
    new.states[0].rank_state.clear()
    assert_ok(_run("hf", root, new))
    assert sorted(item.name for item in (root / "extra_state").iterdir()) == [
        "notes.txt"
    ]
    assert notes.read_bytes() == b"foreign rank-state directory notes"


@pytest.mark.parametrize("shard_size", [None, 64])
def test_malformed_hf_index_is_foreign_and_collision_precedes_payload(
    tmp_path, monkeypatch, shard_size
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    index = root / "model.safetensors.index.json"
    index.write_bytes(b"foreign document, not checkpoint metadata")
    (root / "README.md").write_bytes(b"keep this file")
    before = _files(root)
    writes = []
    monkeypatch.setattr(FileIO, "pwrite", lambda *args: writes.append(args))
    state = _fixture(world=1).states[0]
    _assert_errors(
        [hf.save_state(root, state, RankContext(0, 1), max_shard_size=shard_size)],
        index.name,
    )
    assert not writes
    assert _files(root) == before
    _assert_only_target(tmp_path)


def test_legacy_hf_unrecorded_rank_files_are_foreign_collision(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run("hf", root, old))
    marker = root / ".gdtensor_committed"
    # The pre-fix HF marker listed payload files but no rank-state ownership.
    document = json.loads(marker.read_bytes())
    document.pop("rank_state", None)
    desc.atomic_write(marker, desc.json_bytes(document))
    before = _files(root)
    writes = []
    monkeypatch.setattr(FileIO, "pwrite", lambda *args: writes.append(args))
    _assert_errors(_run("hf", root, _fixture(shift=7)), "extra_state")
    assert not writes
    assert _files(root) == before
    _assert_generation("hf", root, old)
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_complete_stage_with_foreign_journal_restores_entries_then_refuses(
    tmp_path, fmt
):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run(fmt, root, old))
    foreign = _foreign(root, fmt)
    stage = _transaction.DirectoryTransaction(root, 0).stage
    staged = _fixture(shift=7)
    assert_ok(_run(fmt, stage, staged))
    names = ["train.log", "log-link", "global_step_100"]
    journal = {
        "version": 1,
        "old_identity": _path_identity(root),
        "new_identity": _path_identity(stage),
        "foreign": [
            {"path": name, "identity": _path_identity(root / name)} for name in names
        ],
    }
    desc.atomic_write(stage / ".gdcp_transaction", desc.json_bytes(journal))
    staged_files = _files(stage)
    for name in names:
        (root / name).rename(stage / name)
    _assert_errors(_run(fmt, root, _fixture(shift=11)), str(stage))
    _assert_foreign(root, fmt, foreign)
    _assert_generation(fmt, root, old)
    assert _files(stage) == staged_files
    _assert_generation(fmt, stage, staged)


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("phase", ["envelope", "payload_commit", "metadata_commit"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_dcp_session_interrupts_coordinate_and_preserve_old_generation(
    tmp_path, monkeypatch, world, phase, exception
):
    if not os.environ.get("GDTENSOR_TORCH_TESTS"):
        pytest.skip("set GDTENSOR_TORCH_TESTS=1")
    from gdcp.adapters import dcp

    root = tmp_path / "checkpoint"
    old = _fixture(world=world)
    assert_ok(_run("dcp", root, old))
    (root / "foreign").write_bytes(b"unchanged")
    before = _files(root)
    method = "_create_record_file" if phase == "envelope" else "_publish"
    owner = dcp._SaveSession
    if phase == "metadata_commit":
        owner, method = desc, "atomic_write"
    original = getattr(owner, method)
    interrupted = []

    def interrupt(*args):
        result = original(*args)
        selected = (
            Path(args[0]).name == ".metadata"
            if phase == "metadata_commit"
            else args[0].rank == 0
        )
        if selected and not interrupted:
            interrupted.append(True)
            raise exception("injected DCP session interrupt")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(owner, method, interrupt)
        with pytest.raises(exception, match="injected DCP session interrupt"):
            _run("dcp", root, _fixture(world=world, shift=7))
    assert interrupted
    assert _files(root) == before
    _assert_generation("dcp", root, old, world=world)
    _assert_only_target(tmp_path)
    assert_ok(_run("dcp", root, _fixture(world=world, shift=7)))
    _assert_only_target(tmp_path)


def test_linux_mountinfo_decodes_escaped_mountpoint_names(monkeypatch):
    information = (
        r"21 7 0:1 / /check\040point\011line\012back\134slash rw - tmpfs tmpfs rw"
        "\n22 7 0:1 / /ordinary rw - tmpfs tmpfs rw\n"
    )
    original = Path.read_text

    def read_mountinfo(path, *args, **kwargs):
        if path == Path("/proc/self/mountinfo"):
            return information
        return original(path, *args, **kwargs)

    monkeypatch.setattr(_transaction.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", read_mountinfo)
    assert _transaction._mount_points() == {
        Path("/check point\tline\nback\\slash"),
        Path("/ordinary"),
    }


@pytest.mark.parametrize("fmt", _FORMATS)
def test_same_device_bind_mount_is_rejected_from_mountinfo(tmp_path, monkeypatch, fmt):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "foreign").write_bytes(b"keep")
    monkeypatch.setattr(os.path, "ismount", lambda path: False)
    monkeypatch.setattr(_transaction, "_mount_points", lambda: {root})
    writes = []
    monkeypatch.setattr(FileIO, "pwrite", lambda *args: writes.append(args))
    results = _run(fmt, root, _fixture())
    _assert_errors(results, "mount point")
    assert all("subdirectory of the mount" in result.value.msg for result in results)
    assert not writes
    assert _files(root) == {"foreign": b"keep"}
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_first_save_killed_after_publish_recovers_root_journal(tmp_path, fmt):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run(fmt, root, old))
    journal = {
        "version": 1,
        "old_identity": None,
        "new_identity": _path_identity(root),
        "foreign": [],
    }
    desc.atomic_write(root / ".gdcp_transaction", desc.json_bytes(journal))
    new = _fixture(shift=7)
    assert_ok(_run(fmt, root, new))
    assert not (root / ".gdcp_transaction").exists()
    _assert_generation(fmt, root, new)
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("invalid", ["wrong_identity", "invalid_json"])
def test_invalid_root_journal_stays_foreign_and_blocks_payload(
    tmp_path, monkeypatch, fmt, invalid
):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run(fmt, root, old))
    journal = {
        "version": 1,
        "old_identity": None,
        "new_identity": [0, 0],
        "foreign": [],
    }
    data = (
        b"not transaction metadata"
        if invalid == "invalid_json"
        else desc.json_bytes(journal)
    )
    desc.atomic_write(root / ".gdcp_transaction", data)
    before = _files(root)
    writes = []
    monkeypatch.setattr(FileIO, "pwrite", lambda *args: writes.append(args))
    _assert_errors(_run(fmt, root, _fixture(shift=7)), ".gdcp_transaction")
    assert not writes
    assert _files(root) == before
    _assert_generation(fmt, root, old)
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("partial_marker", [False, True])
def test_stage_interrupted_during_initialization_is_removed(
    tmp_path, existing, partial_marker
):
    root = tmp_path / "checkpoint"
    if existing:
        assert_ok(_run("safetensors", root, _fixture()))
    stage = _transaction.DirectoryTransaction(root, 0).stage
    stage.mkdir()
    if partial_marker:
        (stage / (desc.PENDING_NAME + ".tmp")).write_bytes(b"partial marker")
    assert_ok(_run("safetensors", root, _fixture(shift=7)))
    assert not stage.exists()
    _assert_generation("safetensors", root, _fixture(shift=7))


@pytest.mark.parametrize("fmt", _FORMATS)
def test_backup_symlink_to_complete_checkpoint_is_never_moved(tmp_path, fmt):
    source = tmp_path / "source"
    assert_ok(_run(fmt, source, _fixture()))
    root = tmp_path / "checkpoint"
    backup = _transaction.DirectoryTransaction(root, 0).backup
    backup.symlink_to(source, target_is_directory=True)
    before, link = _files(source), os.readlink(backup)
    _assert_errors(_run(fmt, root, _fixture()), str(backup))
    assert not root.exists()
    assert backup.is_symlink() and os.readlink(backup) == link
    assert _files(source) == before


@pytest.mark.parametrize("fmt", _FORMATS)
def test_published_empty_old_stage_is_retained_until_inspected(
    tmp_path, monkeypatch, fmt
):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "foreign").write_bytes(b"foreign-only old target")
    with monkeypatch.context() as patch:
        patch.setattr(_transaction.DirectoryTransaction, "close", lambda self: None)
        assert_ok(_run(fmt, root, _fixture()))
    stage = _transaction.DirectoryTransaction(root, 0).stage
    if not stage.exists():
        # The portable path leaves an unrecognized backup instead of an old stage.
        backup = _transaction.DirectoryTransaction(root, 0).backup
        _assert_errors(_run(fmt, root, _fixture(shift=7)), str(backup))
        assert list(backup.iterdir()) == []
        backup.rmdir()
    else:
        _assert_errors(_run(fmt, root, _fixture(shift=7)), str(stage))
        assert stage.exists() and list(stage.iterdir()) == []
        stage.rmdir()
    assert (root / "foreign").read_bytes() == b"foreign-only old target"
    assert_ok(_run(fmt, root, _fixture(shift=7)))
    assert (root / "foreign").read_bytes() == b"foreign-only old target"


@pytest.mark.parametrize("fmt", ["safetensors", "hf"])
@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("phase", ["payload_publish", "marker_commit"])
@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_connector_publication_interrupt_propagates_after_coordinated_rollback(
    tmp_path, monkeypatch, fmt, world, phase, exception
):
    root = tmp_path / "checkpoint"
    old = _fixture(world=world)
    assert_ok(_run(fmt, root, old))
    (root / "foreign").write_bytes(b"unchanged foreign bytes")
    before = _files(root)
    owner, method = FileIO, "publish"
    path_argument = 2
    if phase == "marker_commit":
        owner, method = (desc, "atomic_write") if fmt == "hf" else (FileIO, "write_all")
        path_argument = 0 if fmt == "hf" else 2
    original = getattr(owner, method)
    interrupted = []

    def interrupt(*args):
        result = original(*args)
        path = Path(args[path_argument])
        selected = (
            path.suffix == ".safetensors"
            if phase == "payload_publish"
            else path.name in (".gdtensor_committed", ".gdtensor_committed.tmp")
        )
        if selected and not interrupted:
            interrupted.append(str(path))
            raise exception("injected connector publication interrupt")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(owner, method, interrupt)
        with pytest.raises(exception, match="injected connector publication interrupt"):
            _run(fmt, root, _fixture(world=world, shift=7))
    assert interrupted
    assert _files(root) == before
    _assert_generation(fmt, root, old, world=world)
    _assert_only_target(tmp_path)
    new = _fixture(world=world, shift=7)
    assert_ok(_run(fmt, root, new))
    _assert_generation(fmt, root, new, world=world)
    assert (root / "foreign").read_bytes() == b"unchanged foreign bytes"
    _assert_only_target(tmp_path)


@pytest.mark.parametrize("fmt", ["safetensors", "dcp"])
def test_foreign_empty_component_directory_collision_precedes_other_payload(
    tmp_path, monkeypatch, fmt
):
    from gdtensor.builder import DTensorTable

    root = tmp_path / "checkpoint"
    (root / "model").mkdir(parents=True)
    (root / "model" / "foreign").write_bytes(b"keep this directory")
    fixture = _fixture()
    for state in fixture.states:
        state.components[0].table = DTensorTable([])
        state.components[0].objects.clear()
    writes = []
    original = FileIO.pwrite

    def record_write(self, rank, path, offset, data):
        writes.append(path)
        return original(self, rank, path, offset, data)

    monkeypatch.setattr(FileIO, "pwrite", record_write)
    _assert_errors(_run(fmt, root, fixture), "model")
    assert not writes
    assert _files(root) == {"model/foreign": b"keep this directory"}
    _assert_only_target(tmp_path)
