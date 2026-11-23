"""HDFS checkpoint transport. Set GDCP_TEST_HDFS_URI to run against a NameNode."""

import json
import os
import shutil
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from gdcp import FSDP2Checkpointer, _hdfs, core, load_hf, save_hf
from gdcp.policy import PerRankPolicy, storage_mesh
from gdtensor.builder import DTensorTable, Err, Ok
from gdtensor_conn.models import build_layout
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
                path="/" + p.relative_to(self.root).as_posix(),
                isdir=p.is_dir(),
                length=p.stat().st_size,
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
        source = _fixture(1 if fmt == "hf" else 2, shift)
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


def _two_weights(world):
    fixture = _fixture(world)
    for state in fixture.states:
        for component in state.components:
            entries = component.table.named_d_tensors
            component.table = DTensorTable(
                [n for n in entries if n.global_tensor_id in ("weight", "tiny")]
            )
            component.logical_shapes.clear()
    return fixture


def _trace_downloads(monkeypatch):
    downloads = defaultdict(list)
    original = _hdfs._download_file

    def traced(client, remote, local):
        original(client, remote, local)
        directory = next(
            p for p in local.parents if p.name.startswith("gdcp-hdfs-read-")
        )
        downloads[directory].append(local.relative_to(directory).as_posix())

    monkeypatch.setattr(_hdfs, "_download_file", traced)
    return downloads


def _seed_for_read_test(fmt, uri, fixture, tmp_path, **options):
    if fmt != "hf":
        assert_ok(_run(fmt, uri, fixture, **options))
        return
    # Simulate an already published multi-writer HF revision.
    local = tmp_path / "old-hf"
    assert_ok(_run(fmt, local, fixture, **options))
    target = _hdfs.checkpoint_path(uri)
    _hdfs.publish(_hdfs._client(target), target, local)


class _SharedPolicy(PerRankPolicy):
    def plan(self, component, world_size):
        table, _ = super().plan(component, world_size)
        layout = build_layout(
            table, world_size, "safetensors", lambda *args: "shared.safetensors"
        )
        return table, layout


@pytest.mark.parametrize(
    ("fmt", "options"),
    [
        ("hf", {}),
        ("hf", {"max_shard_size": 1}),
        ("safetensors", {"policy": _SharedPolicy()}),
    ],
)
def test_hdfs_shared_file_rejected_before_writes(uri, monkeypatch, fmt, options):
    original = _fixture(1)
    assert_ok(_run(fmt, uri, original))
    target = _hdfs.checkpoint_path(uri)
    assert isinstance(target, _hdfs.HdfsPath)
    client = _hdfs._client(target)
    revision = _hdfs._resolve(client, target.path, False)

    def unexpected_io(*args, **kwargs):
        pytest.fail("shared-file rejection must precede payload writes and upload")

    from gdtensor_conn.iobackend import FileIO

    monkeypatch.setattr(FileIO, "create_staging", unexpected_io)
    monkeypatch.setattr(_hdfs, "_upload", unexpected_io)
    results = _run(fmt, uri, _fixture(2), **options)
    for result in results:
        assert type(result) is Err
        assert "HDFS requires one writer per file" in result.value.msg
        assert ".safetensors" in result.value.msg
    assert _hdfs._resolve(client, target.path, False) == revision
    _assert_generation(fmt, uri, original)


class _RankZeroStoragePolicy(PerRankPolicy):
    def plan(self, component, world_size):
        table, _ = super().plan(component, world_size)
        mesh = storage_mesh(world_size, [0])
        table = DTensorTable(
            [
                replace(n, d_tensor=replace(n.d_tensor, mesh=mesh))
                for n in table.named_d_tensors
            ]
        )
        return table, build_layout(
            table, world_size, "safetensors", lambda *args: "rank0.safetensors"
        )


def test_hdfs_rejects_foreign_payload_writer_even_with_one_storage_rank(uri):
    # Only rank zero owns storage slots, but the compute view has two shard writers.
    original = _two_weights(1)
    assert_ok(_run("safetensors", uri, original))
    results = _run("safetensors", uri, _two_weights(2), policy=_RankZeroStoragePolicy())
    assert all(type(result) is Err for result in results)
    assert "rank0.safetensors: writers are ranks 0 and 1" in results[1].value.msg


def test_hdfs_writer_mapping_must_agree_across_ranks(uri):
    source = _fixture(2)

    class SwappedNames(PerRankPolicy):
        def plan(self, component, world_size):
            table, _ = super().plan(component, world_size)
            swap = any(component is c for c in source.states[1].components)
            layout = build_layout(
                table,
                world_size,
                "safetensors",
                lambda order, rank, named: f"part-{rank ^ swap}.safetensors",
            )
            return table, layout

    assert_ok(_run("safetensors", uri, source))
    results = _run("safetensors", uri, source, policy=SwappedNames())
    assert all(
        type(result) is Err and "HDFS file writers" in result.value.msg
        for result in results
    )
    _assert_generation("safetensors", uri, source)


def test_hdfs_multirank_hf_one_writer_per_file(uri):
    source = _fixture(2)
    for state in source.states:
        state.components = state.components[:1]
        model = state.components[0]
        model.table = DTensorTable(
            [
                n
                for n in model.table.named_d_tensors
                if n.global_tensor_id in ("buffer", "step")
            ]
        )
    assert_ok(_run("hf", uri, source, max_shard_size=1))


@pytest.mark.parametrize("fmt", ["safetensors", "dcp", "hf"])
def test_hdfs_model_only_fetches_intersecting_files(uri, monkeypatch, tmp_path, fmt):
    options = {"max_shard_size": 1} if fmt == "hf" else {}
    _seed_for_read_test(fmt, uri, _two_weights(2), tmp_path, **options)
    target = _two_weights(3)
    for state in target.states:
        state.components = state.components[:1]
        state.rank_state.clear()
    downloads = _trace_downloads(monkeypatch)
    assert_ok(_run(fmt, uri, target, loading=True))
    assert len(downloads) == 3
    payloads = []
    for names in downloads.values():
        assert len(names) == len(set(names))
        assert not any(n.startswith(("extra_state/", "optimizer/")) for n in names)
        payloads.append(
            frozenset(n for n in names if n.endswith((".safetensors", ".distcp")))
        )
    if fmt == "hf":
        target_path = _hdfs.checkpoint_path(uri)
        assert isinstance(target_path, _hdfs.HdfsPath)
        client = _hdfs._client(target_path)
        revision = _hdfs._resolve(client, target_path.path, False)
        with client.read(str(revision / "model.safetensors.index.json")) as stream:
            weights = json.loads(stream.read())["weight_map"]
        both = frozenset([weights["tiny"], weights["weight"]])
        assert Counter(payloads) == Counter(
            [both, both, frozenset([weights["weight"]])]
        )
    else:
        suffix = "distcp" if fmt == "dcp" else "safetensors"
        a, b = (f"model/__{i}_0.{suffix}" for i in range(2))
        assert Counter(payloads) == Counter(map(frozenset, [[a], [a, b], [b]]))


@pytest.mark.parametrize("fmt", ["safetensors", "dcp", "hf"])
def test_hdfs_rank_state_only_downloads_no_tensor_files(
    uri, monkeypatch, tmp_path, fmt
):
    _seed_for_read_test(fmt, uri, _fixture(2), tmp_path)
    target = _fixture(3)
    for state in target.states:
        state.components.clear()
    downloads = _trace_downloads(monkeypatch)
    assert_ok(_run(fmt, uri, target, loading=True))
    marker = ".gdtensor_committed" if fmt == "hf" else "gdcp_manifest.json"
    expected = [(marker, f"extra_state/extra_state_rank_{r}.pt") for r in (0, 1, 0)]
    assert Counter(map(tuple, downloads.values())) == Counter(expected)


def test_hdfs_legacy_hf_headers_and_catalog_validation(uri):
    torch = pytest.importorskip("torch")
    source = {"weight": torch.arange(6, dtype=torch.float32)}
    save_hf(uri, source)
    target = _hdfs.checkpoint_path(uri)
    assert isinstance(target, _hdfs.HdfsPath)
    client = _hdfs._client(target)
    revision = _hdfs._resolve(client, target.path, False)
    catalog = str(revision / _hdfs._HF_HEADERS)
    with client.read(catalog) as stream:
        headers = json.loads(stream.read())
    client.delete(catalog)
    state = {"weight": torch.zeros(6)}
    load_hf(uri, state)
    assert torch.equal(state["weight"], source["weight"])
    headers["model.safetensors"][1]["__metadata__"] = {"format": "changed"}
    with client.create(catalog) as stream:
        stream.write(json.dumps(headers).encode())
    state["weight"].zero_()
    with pytest.raises((RuntimeError, ValueError), match="HF header disagrees"):
        load_hf(uri, state)
    assert not state["weight"].count_nonzero()
