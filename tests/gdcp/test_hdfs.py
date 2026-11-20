"""HDFS checkpoint transport. Set GDCP_TEST_HDFS_URI to run against a NameNode."""

import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from gdcp import FSDP2Checkpointer, _hdfs, core, load_hf, save_hf
from gdtensor.builder import Err, Ok
from tests.support.gdcp import assert_ok
from tests.support.gdcp_overwrite import _assert_generation, _fixture, _run


class _DiskClient:
    def __init__(self, root):
        self.root = root

    def path(self, remote):
        return self.root / remote.lstrip("/")

    def mkdirs(self, remote, *, create_parent=False):
        self.path(remote).mkdir(parents=create_parent, exist_ok=True)

    def list_status(self, remote):
        return [
            SimpleNamespace(
                path="/" + p.relative_to(self.root).as_posix(), isdir=p.is_dir()
            )
            for p in self.path(remote).iterdir()
        ]

    def read(self, remote):
        return self.path(remote).open("rb")

    def create(self, remote):
        return self.path(remote).open("xb")

    def rename(self, src, dst, overwrite=False):
        if not overwrite and self.path(dst).exists():
            raise FileExistsError(dst)
        self.path(src).replace(self.path(dst))

    def delete(self, remote, recursive=False):
        path = self.path(remote)
        shutil.rmtree(path) if recursive else path.unlink()


@pytest.fixture
def uri(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setenv("GDCP_HDFS_STAGING_DIR", str(staging))
    cluster = os.environ.get("GDCP_TEST_HDFS_URI")
    if not cluster:
        disk = _DiskClient(tmp_path)
        monkeypatch.setattr(_hdfs, "require_client", lambda: None)
        monkeypatch.setattr(_hdfs, "_client", lambda remote: disk)
        cluster = "hdfs://test-ns/test"
    root = f"{cluster.rstrip('/')}/test-{uuid4().hex}"
    yield root
    assert not list(staging.iterdir())
    if os.environ.get("GDCP_TEST_HDFS_URI"):
        target = _hdfs.checkpoint_path(root)
        assert isinstance(target, _hdfs.HdfsPath)
        _hdfs._client(target).delete(str(target.path), recursive=True)


@pytest.mark.parametrize("fmt", ["safetensors", "dcp", "hf"])
def test_hdfs_collective_roundtrip_and_overwrite(uri, fmt):
    for shift in (1, 7):
        source = _fixture(2, shift)
        assert_ok(_run(fmt, uri, source))
        _assert_generation(fmt, uri, source, world=3)
    if fmt != "hf":
        assert type(core.inspect(uri)) is Ok


@pytest.mark.parametrize("fmt", ["safetensors", "dcp", "hf"])
def test_hdfs_fsdp2_load_latest_and_explicit(uri, fmt):
    pytest.importorskip("torch")
    for step in (10, 2):
        FSDP2Checkpointer.save(
            uri, {"extra_state": step}, global_steps=step, checkpoint_format=fmt
        )
    target = _hdfs.checkpoint_path(uri) / "global_step_20"
    assert isinstance(target, _hdfs.HdfsPath)
    _hdfs._client(target).mkdirs(str(target.path), create_parent=True)
    state = {"extra_state": None}
    FSDP2Checkpointer.load(uri, state)
    assert state["extra_state"] == 10
    FSDP2Checkpointer.load(uri + "/global_step_2", state)
    assert state["extra_state"] == 2


def test_hdfs_hf_weights(uri):
    torch = pytest.importorskip("torch")
    source = {"weight": torch.arange(6, dtype=torch.float32)}
    save_hf(uri, source)
    target = {"weight": torch.zeros(6)}
    load_hf(uri, target)
    assert torch.equal(target["weight"], source["weight"])


def test_hdfs_failed_upload_keeps_checkpoint(uri, monkeypatch):
    source = _fixture()
    assert_ok(_run("safetensors", uri, source))

    def fail_upload(*args):
        raise OSError("upload failed")

    monkeypatch.setattr(_hdfs, "_upload", fail_upload)
    results = _run("safetensors", uri, _fixture(shift=1))
    assert all(type(result) is Err for result in results)
    _assert_generation("safetensors", uri, source)


def test_hdfs_pinned_revision_survives_overwrite(uri):
    source = _fixture()
    assert_ok(_run("safetensors", uri, source))
    with ExitStack() as stack:
        local, first = _hdfs.localize(uri, stack)
        assert_ok(_run("safetensors", uri, _fixture(shift=1)))
        _, second = _hdfs.localize(uri, stack)
        assert first != second
        target = _hdfs.checkpoint_path(first["HDFS revision"])
        assert isinstance(target, _hdfs.HdfsPath)
        assert list(_hdfs._client(target).list_status(str(target.path)))
        _assert_generation("safetensors", local, source)


def test_hdfs_missing_client_does_not_create_local_uri(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def unavailable(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(_hdfs, "import_module", unavailable)
    results = _run("safetensors", "hdfs://unused/path", _fixture())
    assert all(type(result) is Err for result in results)
    assert "hdfs-native" in results[0].value.msg
    assert not list(Path.cwd().iterdir())


@pytest.mark.parametrize("lost_reply", [False, True])
def test_hdfs_publish_failure_is_atomic(uri, monkeypatch, lost_reply):
    before, after = _fixture(), _fixture(shift=10)
    assert_ok(_run("safetensors", uri, before))
    target = _hdfs.checkpoint_path(uri)
    client_type = type(_hdfs._client(target))
    rename = client_type.rename

    def interrupted(client, src, dst, overwrite):
        if lost_reply:
            rename(client, src, dst, overwrite)
        raise OSError("lost publish reply")

    monkeypatch.setattr(client_type, "rename", interrupted)
    results = _run("safetensors", uri, after)
    if lost_reply:
        assert_ok(results)
    else:
        assert all(type(result) is Err for result in results)
    _assert_generation("safetensors", uri, after if lost_reply else before)
