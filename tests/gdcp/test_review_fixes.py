"""Regression tests for collective checkpoint I/O and directory transactions."""

import json
import os
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, local

import pytest

from gdcp import FSDP2Checkpointer, _transaction, core, desc, hf
from gdcp.backend import SafeTensorsBackend
from gdcp.policy import PerRankPolicy
from gdcp.state import RankContext
from gdtensor.builder import DTensorTable, Err, LiteralMesh, Replicate
from gdtensor.tag import WorldSize
from gdtensor_conn.coord import ThreadCoordinator
from gdtensor_conn.iobackend import FileIO
from tests.support import gdcp as fixtures
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
_FORMER_VOTE_NAMES = (".gdcp-save-vote", ".gdcp-load-vote")


def _parallel(call, worlds=(2, 2)):
    coordinator = ThreadCoordinator(2, timeout=2)

    def invoke(rank):
        return call(RankContext(rank, worlds[rank], coordinator))

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(invoke, range(2)))


class _CountingCoordinator(ThreadCoordinator):
    def __init__(self, world, observe=None):
        super().__init__(world, timeout=15)
        self.calls = local()
        self.counts = [0] * world
        self.observe = observe

    def vote(self, ok):
        rank = self.calls.rank
        self.counts[rank] += 1
        if rank == 0 and self.observe is not None:
            self.observe(self.counts[rank])
        return super().vote(ok)


def _counted_operation(fmt, root, fixture, *, loading, observe=None):
    call, options = _operation(fmt, loading, fixture)
    if fmt == "dcp":
        from gdcp.adapters import dcp

        call, options = (dcp.load if loading else dcp.save), {}
    coordinator = _CountingCoordinator(len(fixture.states), observe)

    def invoke(rank):
        coordinator.calls.rank = rank
        ctx = RankContext(rank, len(fixture.states), coordinator)
        return call(root, fixture.states[rank], ctx, **options)

    with ThreadPoolExecutor(max_workers=len(fixture.states)) as pool:
        assert_ok(list(pool.map(invoke, range(len(fixture.states)))))
    assert len(set(coordinator.counts)) == 1
    return coordinator.counts[0]


@pytest.mark.parametrize("fmt", _FORMATS)
def test_load_from_read_only_tree(tmp_path, fmt):
    root = tmp_path / "checkpoint"
    assert_ok(_run(fmt, root, _fixture()))
    original = _files(root)
    modes = {path: path.stat().st_mode for path in [*tmp_path.rglob("*"), tmp_path]}
    try:
        for path in modes:
            path.chmod(0o555 if path.is_dir() else 0o444)
        _assert_generation(fmt, root, _fixture(), world=2)
    finally:
        for path, mode in reversed(list(modes.items())):
            path.chmod(mode)
    assert _files(root) == original
    assert list(tmp_path.iterdir()) == [root]


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("mode", ["shared load", "sibling load", "sibling save"])
def test_independent_jobs_can_overlap(tmp_path, fmt, mode):
    loading = mode != "sibling save"
    first = tmp_path / "first"
    second = first if mode == "shared load" else tmp_path / "second"
    if loading:
        for root in {first, second}:
            assert_ok(_run(fmt, root, _fixture()))
    entered, release = Event(), Event()

    def pause(vote):
        if vote == 6:
            entered.set()
            assert release.wait(10)

    def run_peers():
        jobs = 3 if loading else 1
        with ThreadPoolExecutor(max_workers=jobs) as concurrent:
            peers = [
                concurrent.submit(_run, fmt, second, _fixture(), loading=loading)
                for _ in range(jobs)
            ]
            assert_ok([result for peer in peers for result in peer.result(timeout=8)])

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            _counted_operation, fmt, first, _fixture(), loading=loading, observe=pause
        )
        try:
            assert entered.wait(10)
            run_peers()
        finally:
            release.set()
        pending.result(timeout=8)
    assert set(tmp_path.iterdir()) == {first, second}
    if not loading:
        for root in (first, second):
            _assert_generation(fmt, root, _fixture(), world=2)


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("overwrite", [False, True])
def test_filesystem_entries_during_save_and_load(tmp_path, monkeypatch, fmt, overwrite):
    root = tmp_path / "checkpoint"
    transaction = _transaction.DirectoryTransaction(root, 0)
    stage, backup = transaction.stage, transaction.backup
    foreign = tmp_path / "unrelated"
    foreign.mkdir()
    if overwrite:
        assert_ok(_run(fmt, root, _fixture()))
    # Exercise the portable rename even on systems that support exchange.
    monkeypatch.setattr(_transaction, "_exchange", lambda *_: False)

    def listing(directory):
        return set(directory.rglob("*"))

    def snapshot():
        return set(tmp_path.iterdir()), listing(root), listing(stage), listing(backup)

    def record(samples):
        def observe(_vote):
            samples.append(snapshot())

        return observe

    before = snapshot()
    assert before == (
        {foreign, root} if overwrite else {foreign},
        listing(root),
        set(),
        set(),
    )
    cleanup = []
    rename = Path.rename

    def record_claim(path, target):
        result = rename(path, target)
        if root in path.parents or backup in path.parents:
            cleanup.append((path, result, snapshot()))
        return result

    monkeypatch.setattr(Path, "rename", record_claim)
    saves = []
    _counted_operation(
        fmt, root, _fixture(shift=7), loading=False, observe=record(saves)
    )
    after = snapshot()
    assert after == ({foreign, root}, listing(root), set(), set())
    assert after[1]
    assert any(stage / _transaction.JOURNAL_NAME in sample[2] for sample in saves)
    journal = root / _transaction.JOURNAL_NAME
    published = ({foreign, root}, after[1] | {journal}, set(), set())
    assert any(source == journal for source, _, _ in cleanup)
    assert any(source.is_relative_to(backup) for source, _, _ in cleanup) == overwrite
    for _, claim, (parents, checkpoint, staged, obsolete) in cleanup:
        assert claim in staged
        assert parents <= {foreign, root, stage, backup}
        assert checkpoint in (after[1], published[1])
        assert obsolete <= {backup / name.relative_to(root) for name in before[1]}

    loads = []
    _counted_operation(fmt, root, _fixture(), loading=True, observe=record(loads))
    assert loads and all(sample == after for sample in loads)
    assert snapshot() == after
    obsolete = {backup / name.relative_to(root) for name in before[1]}
    for sample in saves:
        parents, checkpoint, _, old_sibling = sample
        if stage in parents:
            assert parents == before[0] | {stage}
            assert checkpoint == before[1]
            assert not old_sibling
        else:
            assert sample in (before, published, after) or (
                overwrite
                and sample == (published[0] | {backup}, published[1], set(), obsolete)
            )


@pytest.mark.parametrize(
    ("loading", "formats"),
    [
        (False, ("safetensors", "dcp")),
        (False, ("hf", "safetensors")),
        (False, ("hf", "HF weights")),
        (True, ("hf", "safetensors")),
    ],
)
def test_fsdp2_dispatch_shares_initial_agreement(
    tmp_path, monkeypatch, caplog, loading, formats
):
    pytest.importorskip("torch")
    from gdcp.adapters import fsdp2

    roots = (
        [tmp_path / fmt for fmt in formats]
        if loading
        else [tmp_path / "checkpoint"] * 2
    )
    states = [{} if fmt == "HF weights" else {"extra_state": None} for fmt in formats]
    if loading:
        for root, fmt in zip(roots, formats, strict=True):
            FSDP2Checkpointer.save(root, {"extra_state": 7}, checkpoint_format=fmt)
    before = _files(tmp_path)
    initial_states = [dict(state) for state in states]
    current = local()
    monkeypatch.setattr(fsdp2, "_context", lambda: current.ctx)

    def attempt(ctx):
        current.ctx = ctx
        rank = ctx.rank
        fmt = formats[rank]
        call = FSDP2Checkpointer.load if loading else FSDP2Checkpointer.save
        options = {}
        if not loading:
            options = {"checkpoint_format": fmt}
        if fmt == "HF weights":
            options = {"checkpoint_format": "hf", "hf_dtype": "fp32"}
        with pytest.raises(RuntimeError) as error:
            call(roots[rank], states[rank], **options)
        return str(error.value)

    with caplog.at_level("DEBUG", logger=core.__name__):
        leader, peer = _parallel(attempt)
    assert leader == peer
    assert "ranks disagree" in leader and "format options" in leader
    records = [
        record.getMessage() for record in caplog.records if record.name == core.__name__
    ]
    assert len(records) == 2
    for message in records:
        prefix, canonical = message.split(": ", 1)
        rank = int(prefix.rsplit(" ", 1)[1])
        fields = json.loads(canonical)
        assert fields["world size"] == 2
        assert fields["format options"]["dispatch"] == formats[rank]
    assert states == initial_states
    assert _files(tmp_path) == before


def test_hf_save_agrees_on_shard_size_before_staging(tmp_path):
    root = tmp_path / "checkpoint"
    fixture = _fixture()

    def save(ctx):
        return hf.save_state(
            root, fixture.states[ctx.rank], ctx, max_shard_size=(64, 128)[ctx.rank]
        )

    results = _parallel(save)
    assert all(type(result) is Err for result in results), results
    assert results[0].value.msg == results[1].value.msg
    assert "format options" in results[0].value.msg
    assert not list(tmp_path.iterdir())


def _long_names_fixture(monkeypatch):
    monkeypatch.setattr(
        fixtures, "SPECS", [(name * 10, *spec) for name, *spec in fixtures.SPECS]
    )
    fixture = fixtures.numpy_checkpoint(
        2, initialized=True, names=("model" * 10, "optimizer" * 10)
    )
    for rank, state in enumerate(fixture.states):
        state.rank_state = {"extra_state" * 10: bytes([rank])}
    return fixture


@pytest.mark.parametrize("fmt", _FORMATS)
def test_full_state_votes_are_independent_of_names_and_world(
    tmp_path, monkeypatch, fmt
):
    def roundtrip(root, fixture):
        saved = _counted_operation(fmt, root, fixture, loading=False)
        loaded = _counted_operation(fmt, root, fixture, loading=True)
        return saved, loaded

    standard = roundtrip(tmp_path / "baseline", _fixture())
    with monkeypatch.context() as patch:
        longer = roundtrip(tmp_path / ("baseline" * 10), _long_names_fixture(patch))
    more_keys = _fixture()
    for rank, state in enumerate(more_keys.states):
        state.rank_state = {f"extra_stat{key}": bytes([rank]) for key in range(10)}
    more = roundtrip(tmp_path / "morekeys", more_keys)
    larger = roundtrip(tmp_path / "bigworld", _fixture(world=4))
    assert standard == longer == more == larger
    assert (
        standard == {"safetensors": (115, 73), "dcp": (116, 73), "hf": (100, 73)}[fmt]
    )


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("fault", ["hash failure", "different digest", "interrupt"])
@pytest.mark.parametrize("loading", [False, True])
def test_agreement_digest_failure_precedes_payloads(
    tmp_path, monkeypatch, caplog, fmt, fault, loading
):
    root = tmp_path / "checkpoint"
    fixture = _fixture(shift=7)
    if loading:
        assert_ok(_run(fmt, root, _fixture()))
    original = _files(root)
    call, options = _operation(fmt, loading, fixture)
    sha256 = core.sha256
    current = local()
    faults, payload_io = [], []

    def hash_description(data):
        if current.rank != 1:
            return sha256(data)
        faults.append(data)
        if fault == "hash failure":
            raise OSError("injected digest preparation failure")
        if fault == "interrupt":
            raise KeyboardInterrupt("injected digest preparation interrupt")
        return sha256(data + b"changed")

    def run(ctx):
        current.rank = ctx.rank
        try:
            return call(root, fixture.states[ctx.rank], ctx, **options)
        except KeyboardInterrupt as exc:
            return exc

    monkeypatch.setattr(core, "sha256", hash_description)
    io_method = "pread" if loading else "pwrite"
    monkeypatch.setattr(FileIO, io_method, lambda *args: payload_io.append(args))
    with caplog.at_level("DEBUG", logger=core.__name__):
        leader, peer = _parallel(run)
    assert faults and not payload_io
    assert type(leader) is Err
    assert leader.value.operation == "checkpoint rank agreement"
    if fault == "different digest":
        assert type(peer) is Err and leader.value.msg == peer.value.msg
        assert "ranks disagree" in leader.value.msg
        for rank in (0, 1):
            assert (
                f"Checkpoint agreement rank {rank}: {faults[0].decode()}" in caplog.text
            )
    elif fault == "interrupt":
        assert isinstance(peer, KeyboardInterrupt)
    else:
        assert (
            type(peer) is Err
            and "injected digest preparation failure" in peer.value.msg
        )
    assert _files(root) == original
    assert list(tmp_path.iterdir()) == ([root] if loading else [])


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("contents", ["empty", "file", "nested"])
def test_save_load_preserve_unrelated_siblings(tmp_path, fmt, contents):
    root = tmp_path / "checkpoint"
    save_vote, load_vote = [tmp_path / name for name in _FORMER_VOTE_NAMES]
    save_vote.mkdir()
    load_vote.mkdir()
    if contents == "file":
        for sibling in (save_vote, load_vote):
            (sibling / "unrelated").write_bytes(b"foreign file")
    elif contents == "nested":
        for sibling in (save_vote, load_vote):
            (sibling / "nested").mkdir()
            (sibling / "nested" / "unrelated").write_bytes(b"foreign file")
    original = {
        path: (path.stat().st_ino, _files(path)) for path in (save_vote, load_vote)
    }

    assert_ok(_run(fmt, root, _fixture()))
    _assert_generation(fmt, root, _fixture(), world=2)
    assert set(tmp_path.iterdir()) == {root, save_vote, load_vote}
    for path, (inode, files) in original.items():
        assert path.stat().st_ino == inode
        assert _files(path) == files


@pytest.mark.parametrize("fmt", _FORMATS)
def test_unrelated_hidden_directory_can_be_checkpoint_target(tmp_path, fmt):
    for name in _FORMER_VOTE_NAMES:
        root = tmp_path / name
        assert_ok(_run(fmt, root, _fixture()))
    for name in _FORMER_VOTE_NAMES:
        _assert_generation(fmt, tmp_path / name, _fixture(), world=2)
    root = tmp_path / "checkpoint"
    assert_ok(_run(fmt, root, _fixture()))
    _assert_generation(fmt, root, _fixture(), world=2)
    assert {path.name for path in tmp_path.iterdir()} == {
        *_FORMER_VOTE_NAMES,
        root.name,
    }


@pytest.mark.parametrize("fmt", _FORMATS)
def test_save_rejects_different_targets_before_staging(tmp_path, fmt):
    fixture = _fixture()
    call, options = _operation(fmt, False, fixture)

    def save(ctx):
        return call(
            tmp_path / f"checkpoint{ctx.rank}", fixture.states[ctx.rank], ctx, **options
        )

    results = _parallel(save)
    assert all(type(result) is Err for result in results), results
    assert results[0].value.msg == results[1].value.msg
    assert "checkpoint save target" in results[0].value.msg
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("loading", [False, True])
@pytest.mark.parametrize("fault", [OSError, KeyboardInterrupt])
def test_agreement_setup_failure_leaves_no_entries(
    tmp_path, monkeypatch, fmt, loading, fault
):
    root = tmp_path / "checkpoint"
    fixture = _fixture()
    if loading:
        assert_ok(_run(fmt, root, fixture))
    original = _files(tmp_path)
    call, options = _operation(fmt, loading, fixture)
    encode = desc.json_bytes
    current = local()

    def fail_leader_description(fields):
        if current.rank == 0:
            raise fault("injected agreement setup failure")
        return encode(fields)

    def attempt(ctx):
        current.rank = ctx.rank
        try:
            return call(root, fixture.states[ctx.rank], ctx, **options)
        except KeyboardInterrupt as exc:
            return exc

    monkeypatch.setattr(desc, "json_bytes", fail_leader_description)
    leader, peer = _parallel(attempt)
    assert type(leader) is (Err if fault is OSError else fault)
    assert type(peer) is Err
    assert _files(tmp_path) == original
    assert list(tmp_path.iterdir()) == ([root] if loading else [])


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("mode", ["single save", "collective save", "load"])
@pytest.mark.parametrize("empty_file", [False, True])
def test_operations_ignore_unrelated_empty_sibling(tmp_path, fmt, mode, empty_file):
    root = tmp_path / "checkpoint"
    world = 1 if mode == "single save" else 2
    fixture = _fixture(world=world)
    loading = mode == "load"
    if loading:
        assert_ok(_run(fmt, root, fixture))
    sibling = tmp_path / _FORMER_VOTE_NAMES[loading]
    sibling.mkdir()
    if empty_file:
        (sibling / "empty").touch()
    inode = sibling.stat().st_ino
    original = _files(sibling)

    assert_ok(_run(fmt, root, fixture, loading=loading))
    assert set(tmp_path.iterdir()) == {root, sibling}
    assert sibling.stat().st_ino == inode
    assert _files(sibling) == original


def test_agreement_interrupt_does_not_prevent_retry(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    fixture = _fixture()
    interrupt = KeyboardInterrupt("interrupted agreement preparation")
    sha256 = core.sha256
    current = local()

    def hash_description(data):
        if current.rank == 0:
            raise interrupt
        return sha256(data)

    def save(ctx):
        current.rank = ctx.rank
        try:
            return core.save(root, fixture.states[ctx.rank], ctx)
        except KeyboardInterrupt as exc:
            return exc

    with monkeypatch.context() as patch:
        patch.setattr(core, "sha256", hash_description)
        leader, peer = _parallel(save)
    assert leader is interrupt
    assert type(peer) is Err and peer.value.operation == "checkpoint rank agreement"
    assert not list(tmp_path.iterdir())
    assert_ok(_run("safetensors", root, fixture))
    assert list(tmp_path.iterdir()) == [root]


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
    tmp_path, monkeypatch, caplog, fmt, owner
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
    with caplog.at_level("DEBUG", logger=core.__name__):
        results = _run(fmt, root, source)
    assert all(type(result) is Err for result in results), results
    assert results[0].value.msg == results[1].value.msg
    assert "rank_state keys" in results[0].value.msg
    logged = {
        int(record.args[0]): json.loads(record.args[1])
        for record in caplog.records
        if record.msg == "Checkpoint agreement rank %s: %s"
    }
    assert logged[owner]["rank_state keys"] == ["extra_state", "extraé"]
    assert logged[1 - owner]["rank_state keys"] == ["extra_state"]
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
@pytest.mark.parametrize("loading", [False, True])
def test_agreement_rejects_different_keys_with_three_participants(
    tmp_path, fmt, loading
):
    root = tmp_path / "checkpoint"
    source = _fixture(world=3)
    if loading:
        assert_ok(_run(fmt, root, source))
    original = _files(root)
    source.states[2].rank_state["another_key"] = b"different keys"
    call, options = _operation(fmt, loading, source)
    coordinator = ThreadCoordinator(3, timeout=2)

    def operate(rank):
        return call(
            root, source.states[rank], RankContext(rank, 3, coordinator), **options
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(operate, range(3)))
    assert all(type(result) is Err for result in results), results
    assert len({result.value.msg for result in results}) == 1
    assert "rank_state keys" in results[0].value.msg
    assert _files(root) == original
    assert list(tmp_path.iterdir()) == ([root] if loading else [])


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize(
    "field", ["rank_state keys", "rank configuration", "checkpoint component sequence"]
)
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
    original_components = list(state.components)
    observed = []

    def change():
        observed.append("changed")
        if field == "rank_state keys":
            state.rank_state.clear()
        elif field == "rank configuration":
            contexts[1].world_size = 3
        else:
            state.components.reverse()

    def restore():
        observed.append("restored")
        state.rank_state.update(original_rank_state)
        contexts[1].world_size = 2
        state.components[:] = original_components

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
    claims = []

    def replace():
        os.replace(replacement, old_file)
        injected.append(True)

    def unlink_file(path, *args, **kwargs):
        if path in claims and not injected:
            replace()
        return unlink(path, *args, **kwargs)

    def move_file(path, target):
        if path != old_file:
            return rename(path, target)
        if replace_before_move and not injected:
            replace()
        claimed = rename(path, target)
        claims.append(claimed)
        return claimed

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


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize(
    "change",
    [
        "order",
        "id",
        "shape",
        "dtype",
        "itemsize",
        "placement",
        "mesh",
        "alias",
        "logical shape",
    ],
)
def test_save_rejects_tensor_schema_mismatch_before_staging(
    tmp_path, monkeypatch, fmt, change
):
    root = tmp_path / "checkpoint"
    original = _fixture()
    assert_ok(_run(fmt, root, original))
    source = _fixture(shift=7)
    component = source.states[1].components[0]
    entries = deepcopy(component.table.named_d_tensors)
    first = entries[0]
    tensor = first.d_tensor
    if change == "order":
        entries[-1], entries[-2] = entries[-2], entries[-1]
    elif change == "id":
        first.global_tensor_id = "different_weight"
    elif change == "shape":
        tensor.global_tensor_meta.global_shape[0] += 1
    elif change == "dtype":
        tensor.global_tensor_meta.dtype = "I32"
    elif change == "itemsize":
        tensor.global_tensor_meta.itemsize *= 2
    elif change == "placement":
        tensor.placements = [Replicate()]
    elif change == "mesh":
        tensor.mesh = LiteralMesh([1, 0], WorldSize(2))
    elif change == "alias":
        first.alias.append("tied_weight")
    else:
        component.logical_shapes["step"] = [1]
    component.table = DTensorTable(entries)
    before = _files(root)
    monkeypatch.setattr(
        core, "DirectoryTransaction", lambda *args: pytest.fail("opened a transaction")
    )
    results = _run(fmt, root, source)
    assert all(
        type(r) is Err and "ordered tensor schema" in r.value.msg for r in results
    )
    assert results[0].value.msg == results[1].value.msg
    assert _files(root) == before
    assert list(tmp_path.iterdir()) == [root]
    _assert_generation(fmt, root, original)


@pytest.mark.parametrize("fmt", [*_FORMATS, "HF weights"])
def test_facade_checks_materialized_tensor_schema(tmp_path, monkeypatch, fmt):
    pytest.importorskip("torch")
    from gdcp.adapters import fsdp2

    fixture = fixtures.numpy_checkpoint(2, initialized=True)
    component = fixture.states[1].components[0]
    component.table = DTensorTable(list(reversed(component.table.named_d_tensors)))
    current = local()
    monkeypatch.setattr(fsdp2, "_context", lambda: current.ctx)
    monkeypatch.setattr(
        fsdp2,
        "_prepare_save_state",
        lambda state, ctx, route, dtype: (fixture.states[ctx.rank], None),
    )

    def save(ctx):
        current.ctx = ctx
        opts = {"checkpoint_format": fmt}
        if fmt == "HF weights":
            opts = {"checkpoint_format": "hf", "hf_dtype": "fp32"}
        with pytest.raises(RuntimeError, match="ordered tensor schema") as error:
            FSDP2Checkpointer.save(tmp_path / "checkpoint", {"model": object()}, **opts)
        return str(error.value)

    leader, peer = _parallel(save)
    assert leader == peer
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dotted", [False, True])
@pytest.mark.parametrize("kinds", ["objects", "tensors", "mixed"])
def test_optimizer_rejects_flattened_field_collision(dotted, kinds):
    torch = pytest.importorskip("torch")
    from gdcp.adapters.fsdp2 import _optimizer_values

    a = torch.tensor(10) if kinds != "objects" else 10
    b = torch.tensor(20) if kinds == "tensors" else 20
    fields = (
        {"p": {"child.slot": a}, "p.child": {"slot": b}}
        if dotted
        else {"p": {1: a, "1": b}}
    )
    with pytest.raises(ValueError, match="optimizer state key collision") as error:
        _optimizer_values({"param_groups": [], "state": fields})
    names = (
        (("p", "child.slot"), ("p.child", "slot")) if dotted else (("p", 1), ("p", "1"))
    )
    assert all(repr(name) in str(error.value) for name in names)


@pytest.mark.parametrize("fmt", _FORMATS)
def test_collective_save_rejects_mixed_local_and_hdfs_targets(
    tmp_path, monkeypatch, fmt
):
    monkeypatch.chdir(tmp_path)
    fixture = _fixture()
    call, options = _operation(fmt, False, fixture)
    results = _parallel(
        lambda ctx: call(
            ("local", "hdfs://cluster/checkpoint")[ctx.rank],
            fixture.states[ctx.rank],
            ctx,
            **options,
        )
    )
    assert all(type(r) is Err for r in results)
    assert "resolved checkpoint save target" in results[1].value.msg
    assert not list(tmp_path.iterdir())


def test_collapsed_hdfs_path_fails_in_public_checkpoint_apis(tmp_path, monkeypatch):
    path = Path("hdfs://cluster/checkpoint")
    torch = pytest.importorskip("torch")
    from gdcp import inspect, load_hf, save_hf

    monkeypatch.chdir(tmp_path)
    for fmt in _FORMATS:
        with pytest.raises(ValueError, match="hdfs://"):
            FSDP2Checkpointer.save(
                path, {"extra_state": 1}, global_steps=3, checkpoint_format=fmt
            )
    with pytest.raises(ValueError, match="hdfs://"):
        FSDP2Checkpointer.load(path, {"extra_state": None})
    for call in (save_hf, load_hf):
        with pytest.raises((ValueError, RuntimeError), match="hdfs://"):
            call(path, {"weight": torch.ones(2)})
    result = inspect(path)
    assert type(result) is Err and "hdfs://" in result.value.msg
    assert not list(tmp_path.iterdir())
