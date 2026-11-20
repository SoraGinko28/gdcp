"""Failed replacement never changes the previous checkpoint generation."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gdcp import _transaction, core, hf
from gdcp.state import RankContext
from gdtensor.builder import Err, GDTensorIOError
from tests.support.gdcp import assert_ok
from tests.support.gdcp_overwrite import (
    _assert_generation,
    _fail_rank_one,
    _files,
    _fixture,
    _run,
)


@pytest.mark.parametrize("fmt", ["safetensors", "dcp", "hf"])
def test_rank_failure_overwrite_preserves_old_tensor_objects_and_rank_state(
    tmp_path, monkeypatch, fmt
):
    root = tmp_path / "checkpoint"
    old, new = _fixture(), _fixture(shift=7)
    assert_ok(_run(fmt, root, old))
    original_files = _files(root)
    failures = _fail_rank_one(monkeypatch)
    result = _run(fmt, root, new)
    assert failures
    assert all(type(item) is Err for item in result), result
    assert _files(root) == original_files
    _assert_generation(fmt, root, old)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint"]
    assert_ok(_run(fmt, root, new))
    _assert_generation(fmt, root, new)


@pytest.mark.parametrize("fmt", ["safetensors", "dcp", "hf"])
def test_failed_first_save_leaves_no_checkpoint_or_staging(tmp_path, monkeypatch, fmt):
    root = tmp_path / "checkpoint"
    failures = _fail_rank_one(monkeypatch)
    result = _run(fmt, root, _fixture())
    assert failures
    assert all(type(item) is Err for item in result), result
    assert list(tmp_path.iterdir()) == []


def test_portable_second_rename_failure_restores_previous_path(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run("safetensors", root, old))
    original_files = _files(root)
    original_rename = Path.rename
    failures = []

    def fail_publish(path, target):
        if path.name.endswith(".gdcp-stage") and Path(target) == root and not failures:
            failures.append(path)
            raise OSError("injected portable publication failure")
        return original_rename(path, target)

    monkeypatch.setattr(_transaction, "_exchange", lambda _left, _right: False)
    monkeypatch.setattr(Path, "rename", fail_publish)
    result = _run("safetensors", root, _fixture(shift=7))
    assert failures
    assert all(type(item) is Err for item in result), result
    assert _files(root) == original_files
    _assert_generation("safetensors", root, old)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint"]


def test_failed_publication_vote_rolls_back_complete_new_generation(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run("safetensors", root, old))
    original = core._collective

    def reject_publication(ctx, operation, action):
        if operation == "checkpoint publish" and ctx.rank == 1:
            action = lambda: Err(
                GDTensorIOError(operation=operation, msg="injected vote failure")
            )
        return original(ctx, operation, action)

    monkeypatch.setattr(core, "_collective", reject_publication)
    result = _run("safetensors", root, _fixture(shift=7))
    assert all(type(item) is Err for item in result), result
    _assert_generation("safetensors", root, old)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint"]


@pytest.mark.parametrize("reserved", ["pending-stage", "complete-stage", "backup"])
def test_preexisting_reserved_siblings_follow_recovery_rules(tmp_path, reserved):
    root = tmp_path / "checkpoint"
    assert_ok(_run("safetensors", root, _fixture()))
    transaction = _transaction.DirectoryTransaction(root, 0)
    sibling = transaction.backup if reserved == "backup" else transaction.stage
    if reserved == "complete-stage":
        assert_ok(_run("safetensors", sibling, _fixture(shift=3)))
    else:
        sibling.mkdir()
    (sibling / "unrelated").write_bytes(b"preserve me")
    if reserved == "pending-stage":
        (sibling / core.desc.PENDING_NAME).write_bytes(b"save interrupted")
        assert_ok(_run("safetensors", root, _fixture(shift=7)))
        assert not sibling.exists()
        return
    original_files = _files(sibling)
    result = _run("safetensors", root, _fixture())
    assert all(type(item) is Err for item in result), result
    assert _files(sibling) == original_files


def test_hf_metadata_failure_preserves_previous_files(tmp_path):
    root = tmp_path / "checkpoint"
    old = _fixture(world=1)
    ctx = RankContext(0, 1)
    assert_ok([hf.save_state(root, old.states[0], ctx)])
    original_files = _files(root)

    def fail_metadata(stage):
        (stage / "config.json").write_bytes(b"new incomplete config")
        raise OSError("injected HF config failure")

    result = hf.save_state(
        root, _fixture(world=1, shift=7).states[0], ctx, _write_metadata=fail_metadata
    )
    assert type(result) is Err, result
    assert _files(root) == original_files
    _assert_generation("hf", root, old)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux directory exchange")
def test_linux_exchange_publishes_complete_generation(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    assert_ok(_run("safetensors", root, _fixture()))
    original = _transaction._exchange
    exchanges = []

    def record_exchange(left, right):
        result = original(left, right)
        exchanges.append(result)
        return result

    monkeypatch.setattr(_transaction, "_exchange", record_exchange)
    new = _fixture(shift=7)
    assert_ok(_run("safetensors", root, new))
    _assert_generation("safetensors", root, new)
    assert exchanges == [True]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint"]


def test_weights_only_config_failure_preserves_previous_export(tmp_path):
    if not os.environ.get("GDTENSOR_TORCH_TESTS"):
        pytest.skip("set GDTENSOR_TORCH_TESTS=1")
    import torch
    from gdcp import save_hf

    model = torch.nn.Linear(4, 2)
    setattr(
        model, "config", SimpleNamespace(to_json_string=lambda **kwargs: '{"old":true}')
    )
    root = tmp_path / "checkpoint"
    save_hf(root, model, dtype="fp32")
    original_files = _files(root)

    def fail_config(**kwargs):
        raise OSError("injected model config serialization failure")

    setattr(model, "config", SimpleNamespace(to_json_string=fail_config))
    with pytest.raises(RuntimeError, match="config serialization failure"):
        save_hf(root, model, dtype="fp32")
    assert _files(root) == original_files
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint"]


def test_parent_fsync_failure_after_publication_restores_previous_generation(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkpoint"
    old = _fixture()
    assert_ok(_run("safetensors", root, old))
    original_files = _files(root)
    original = _transaction._sync_directory
    failures = []

    def fail_parent(path):
        if (
            Path(path) == tmp_path
            and (root / core.desc.MANIFEST_NAME).exists()
            and not (
                tmp_path / ".checkpoint.gdcp-stage" / _transaction.JOURNAL_NAME
            ).exists()
            and not failures
        ):
            failures.append(path)
            raise OSError("injected publication parent fsync failure")
        return original(path)

    monkeypatch.setattr(_transaction, "_sync_directory", fail_parent)
    result = _run("safetensors", root, _fixture(shift=7))
    assert failures
    assert all(type(item) is Err for item in result), result
    assert _files(root) == original_files
    _assert_generation("safetensors", root, old)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint"]


def test_obsolete_cleanup_failure_keeps_new_checkpoint_and_old_sibling(
    tmp_path, monkeypatch, caplog
):
    root = tmp_path / "checkpoint"
    old, new = _fixture(), _fixture(shift=7)
    assert_ok(_run("safetensors", root, old))
    original_files = _files(root)
    old_inode = root.stat().st_ino
    transaction = _transaction.DirectoryTransaction(root, 0)
    unlink, rename = Path.unlink, Path.rename
    claims = set()

    def track_claim(path, target):
        moved = rename(path, target)
        if {transaction.stage, transaction.backup}.intersection(path.parents):
            claims.add(moved)
        return moved

    def fail_cleanup(path, *args, **kwargs):
        if path in claims:
            raise OSError("injected obsolete-generation cleanup failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rename", track_claim)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    assert_ok(_run("safetensors", root, new))
    _assert_generation("safetensors", root, new)
    retained = transaction.stage if transaction.stage.exists() else transaction.backup
    assert retained.stat().st_ino == old_inode
    assert _files(retained) == original_files
    assert "retained sibling" in caplog.text


def test_invalid_transaction_path_returns_an_error():
    result = _run("safetensors", Path("/"), _fixture(world=1))
    assert all(type(item) is Err for item in result), result
