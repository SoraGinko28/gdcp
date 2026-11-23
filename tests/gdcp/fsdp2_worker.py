"""Distributed torchrun worker for the GDCP FSDP2 acceptance suite."""

import argparse
import io
import json
import logging
import os
import shutil
from inspect import signature
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.distributed import barrier, destroy_process_group, get_rank
from torch.distributed import get_world_size, init_process_group

from gdcp import FSDP2Checkpointer
from gdtensor_conn.coord import GlooCoordinator
from gdtensor_conn.iobackend import FileIO
from tests.support.gdcp_fsdp2 import (
    assert_missing_extra_state_rejected,
    assert_model,
    assert_optimizer,
    assert_rank_failure,
    dense_oracle,
    extra_for,
    local,
    make_model,
    oracle_records,
    raw,
    train_step,
)


def state_for(model, optimizer, rank):
    return {"model": model, "optimizer": optimizer, "extra_state": extra_for(rank)}


def trained(world):
    model, optimizer = make_model(world)
    for step in range(2):
        train_step(model, optimizer, step)
    return model, optimizer


def poisoned(world, *, device="cpu"):
    model, optimizer = make_model(world, target=True, device=device)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            local(tensor).fill_(77)
    return model, optimizer


def check(model, optimizer, steps=2, *, device="cpu"):
    oracle, oracle_optimizer = dense_oracle(steps, device=device)
    assert_model(model, oracle, get_rank(), get_world_size())
    assert_optimizer(
        model, optimizer, oracle, oracle_optimizer, get_rank(), get_world_size()
    )


def save(args):
    model, optimizer = trained(get_world_size())
    check(model, optimizer)
    FSDP2Checkpointer.save(args.path, state_for(model, optimizer, get_rank()))
    if get_rank() == 0:
        expected = oracle_records()
        Path(args.oracle).write_text(json.dumps(expected, sort_keys=True))
    print(
        f"ACCEPTANCE SAVE PASS world={get_world_size()} rank={get_rank()}", flush=True
    )


def load(args):
    model, optimizer = poisoned(get_world_size())
    state = state_for(model, optimizer, get_rank())
    captured = io.StringIO()
    handler = logging.StreamHandler(captured)
    handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(handler)
    FSDP2Checkpointer.load(args.path, state)
    logging.getLogger().removeHandler(handler)
    check(model, optimizer)
    expected_rank = get_rank() if get_rank() < args.source_world else 0
    assert state["extra_state"] == extra_for(expected_rank)
    if get_rank() >= args.source_world:
        assert captured.getvalue(), "extra_state rank-0 fallback must log a warning"
        print(
            f"ACCEPTANCE 9 fallback warning={captured.getvalue().strip()}", flush=True
        )
    if args.continue_step:
        train_step(model, optimizer, 2)
        check(model, optimizer, steps=3)
    if args.output:
        FSDP2Checkpointer.save(args.output, state)
    print(
        f"ACCEPTANCE 1/2/3/9 PASS source={args.source_world} target={get_world_size()} rank={get_rank()} continue={args.continue_step}",
        flush=True,
    )


def atomic(args):
    model, optimizer = trained(get_world_size())
    state = state_for(model, optimizer, get_rank())
    original = FileIO.pwrite
    triggered = []

    def faulty(self, rank, path, offset, data):
        if rank == 1 and offset > 0 and not triggered:
            triggered.append(path)
            raise OSError("GDCP injected rank-1 payload write fault")
        return original(self, rank, path, offset, data)

    with patch.object(FileIO, "pwrite", faulty):
        assert_rank_failure(
            GlooCoordinator(),
            lambda: FSDP2Checkpointer.save(args.output, state),
            "rank-1 write fault",
        )
    assert get_rank() != 1 or triggered
    assert not (Path(args.output) / "gdcp_manifest.json").exists()
    assert_rank_failure(
        GlooCoordinator(),
        lambda: FSDP2Checkpointer.load(args.output, state),
        "uncommitted load",
    )
    FSDP2Checkpointer.save(args.output, state)
    target, target_optimizer = poisoned(get_world_size())
    FSDP2Checkpointer.load(args.output, state_for(target, target_optimizer, get_rank()))
    check(target, target_optimizer)
    print(
        f"ACCEPTANCE 7 PASS rank={get_rank()} fault propagated; no manifest; load rejected; retry restored",
        flush=True,
    )


def _damage(path, variant):
    manifest_path = path / "gdcp_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tensors = manifest["components"]["model"]["tensors"]
    weight = next(tensor for tensor in tensors if tensor["id"] == "weight")
    if variant == "missing":
        tensors.remove(weight)
    elif variant == "unexpected":
        tensors.append(dict(weight, id="unexpected.weight"))
    elif variant == "shape":
        weight["shape"] = [11, 4]
    else:
        weight["dtype"], weight["itemsize"] = "F64", 8
    manifest_path.write_text(json.dumps(manifest))


def _strict_case(args, variant):
    path = Path(args.output) / variant
    if get_rank() == 0:
        shutil.copytree(args.path, path)
        _damage(path, variant)
    barrier()
    model, optimizer = poisoned(get_world_size())
    before = {key: raw(local(tensor)) for key, tensor in model.state_dict().items()}
    touched = []

    def blocked(self, rank, filename, offset, size):
        touched.append(filename)
        raise AssertionError("tensor payload touched during strict preflight")

    with patch.object(FileIO, "pread", blocked):
        error = assert_rank_failure(
            GlooCoordinator(),
            lambda: FSDP2Checkpointer.load(
                path, state_for(model, optimizer, get_rank())
            ),
            variant,
        )
    key = "unexpected.weight" if variant == "unexpected" else "weight"
    assert key in str(error), (variant, error)
    assert not touched, (variant, touched)
    after = {key: raw(local(tensor)) for key, tensor in model.state_dict().items()}
    assert before == after, variant
    print(
        f"ACCEPTANCE 8 PASS variant={variant} rank={get_rank()} key={key} zero tensor reads and unchanged parameters",
        flush=True,
    )


def strict(args):
    for variant in ("missing", "unexpected", "shape", "dtype"):
        _strict_case(args, variant)


def partial(args):
    model, optimizer = trained(get_world_size())
    model_only = str(Path(args.output) / "model-only")
    FSDP2Checkpointer.save(model_only, {"model": model})
    for path in (model_only, args.path):
        target, _unused = poisoned(get_world_size())
        FSDP2Checkpointer.load(path, {"model": target})
        oracle, _unused = dense_oracle()
        assert_model(target, oracle, get_rank(), get_world_size())
    common = {"extra_state": None}
    FSDP2Checkpointer.load(args.path, common, load_common_state_only=True)
    assert common["extra_state"] == extra_for(get_rank())
    print(
        f"ACCEPTANCE 10 PASS rank={get_rank()} model-only roundtrip, model-only from full, common-only",
        flush=True,
    )


def global_steps(args):
    model, optimizer = trained(get_world_size())
    state = state_for(model, optimizer, get_rank())
    root = Path(args.output)
    FSDP2Checkpointer.save(root / "appended", state, global_steps=123)
    nested = root / "appended" / "global_step_123"
    assert (nested / "gdcp_manifest.json").is_file()
    assert not (root / "appended" / "gdcp_manifest.json").exists()
    restored = {"extra_state": None}
    FSDP2Checkpointer.load(root / "appended", restored)
    assert restored["extra_state"] == extra_for(get_rank())
    FSDP2Checkpointer.save(
        root / "direct",
        state,
        global_steps=123,
        ignore_append_global_steps_to_folder=True,
    )
    assert (root / "direct" / "gdcp_manifest.json").is_file()
    assert not (root / "direct" / "global_step_123").exists()
    for path in (nested, root / "direct"):
        assert (
            json.loads((path / "gdcp_manifest.json").read_text())["global_steps"] == 123
        )
    print(
        f"ACCEPTANCE 11 PASS rank={get_rank()} global_step_123 and direct directory",
        flush=True,
    )


def extra_missing(args):
    assert_missing_extra_state_rejected(
        Path(args.path), Path(args.output), get_rank(), GlooCoordinator()
    )
    print(
        f"ACCEPTANCE 9 PASS rank={get_rank()} missing declared file rejected with state unchanged",
        flush=True,
    )


def gpu_save(args):
    rank = get_rank()
    model, optimizer = make_model(get_world_size(), device="cuda")
    for step in range(2):
        train_step(model, optimizer, step)
    state = state_for(model, optimizer, rank)
    formats = ("safetensors", "dcp", "hf")
    for fmt in formats:
        FSDP2Checkpointer.save(Path(args.path) / fmt, state, checkpoint_format=fmt)

    # Attempt to replace each valid generation with different weights, then
    # verify the old generations from a fresh three-rank process group.
    train_step(model, optimizer, 2)
    for fmt in formats:
        _check_failed_overwrite(Path(args.path) / fmt, state, fmt)


def _check_failed_overwrite(path, state, fmt):
    original = FileIO.pwrite
    injected = False

    def faulty(self, writer, filename, offset, data):
        nonlocal injected
        if writer == 1 and offset > 0 and not injected:
            injected = True
            raise OSError("injected rank-1 payload write")
        return original(self, writer, filename, offset, data)

    with patch.object(FileIO, "pwrite", faulty), pytest.raises(RuntimeError):
        FSDP2Checkpointer.save(path, state, checkpoint_format=fmt)
    assert get_rank() != 1 or injected, fmt
    print(f"{fmt}: rank {get_rank()} rejected failed overwrite", flush=True)


def gpu_load(args):
    rank = get_rank()
    for fmt in ("safetensors", "dcp", "hf"):
        model, optimizer = poisoned(get_world_size(), device="cuda")
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}
        FSDP2Checkpointer.load(Path(args.path) / fmt, state)
        assert state["extra_state"] == extra_for(
            rank if rank < args.source_world else 0
        )
        check(model, optimizer, device="cuda")
        train_step(model, optimizer, 2)
        check(model, optimizer, steps=3, device="cuda")
        print(f"{fmt}: rank {rank} restored state and next AdamW step", flush=True)


def subgroups(args):
    from torch.distributed import ProcessGroup, new_group
    from torch.distributed.device_mesh import DeviceMesh

    groups = [new_group(ranks=ranks) for ranks in ([0, 2], [1, 3])]
    team = get_rank() % 2
    pg = groups[team]
    assert isinstance(pg, ProcessGroup)
    device = "cpu" if args.backend == "gloo" else "cuda"
    mesh = DeviceMesh.from_group(pg, device, mesh_dim_names=("dp",))
    root = Path(args.path)
    with pytest.raises(ValueError, match="not a member"):
        FSDP2Checkpointer.load(root, {}, process_group=groups[1 - team])
    model, optimizer = make_model(2, device=device, mesh=mesh)
    for step in range(team + 1):
        train_step(model, optimizer, step)
    state = state_for(model, optimizer, get_rank())
    formats = ("safetensors", "hf", "dcp")
    for fmt in formats if team == 0 else reversed(formats):
        FSDP2Checkpointer.save(
            root / f"team-{team}" / fmt, state, checkpoint_format=fmt, process_group=pg
        )
    # A group-local option failure must not enter any collective in the other group.
    if team == 0:
        with pytest.raises(
            (NotImplementedError, RuntimeError), match="role|another rank"
        ):
            FSDP2Checkpointer.save(
                root / "invalid",
                state,
                role="actor" if get_rank(pg) == 1 else None,
                process_group=pg,
            )
    barrier()
    for fmt in formats:
        _restore_subgroup(root, fmt, 1 - team, pg, mesh, device)
    if "sort_ranks" in signature(new_group).parameters:
        reversed_groups = [
            new_group(ranks=ranks, sort_ranks=False) for ranks in ([2, 0], [3, 1])
        ]
        reverse = reversed_groups[team]
        assert isinstance(reverse, ProcessGroup)
        reversed_path = root / f"reordered-{team}"
        FSDP2Checkpointer.save(reversed_path, state, process_group=reverse)
        restored, _ = make_model(2, target=True, device=device, mesh=mesh)
        into = {"model": restored, "extra_state": None}
        FSDP2Checkpointer.load(reversed_path, into, process_group=pg)
        assert into["extra_state"] == extra_for(2 * (1 - get_rank(pg)) + team)
        reference, _ = dense_oracle(team + 1, device=device)
        assert_model(restored, reference, get_rank(pg), 2)
    # Only one group calls save while the other waits on the default group.
    if team == 1:
        FSDP2Checkpointer.save(root / "solo", state, process_group=pg)
    barrier()
    full_model, _ = make_model(4, target=True, device=device)
    with pytest.raises(ValueError, match="device mesh.*process_group"):
        FSDP2Checkpointer.load(root / "solo", {"model": full_model}, process_group=pg)
    FSDP2Checkpointer.load(root / "solo", {"model": full_model})
    ref, _ = dense_oracle(2, device=device)
    assert_model(full_model, ref, get_rank(), 4)
    print(
        f"subgroup PASS {args.backend} global={get_rank()} local={get_rank(pg)}",
        flush=True,
    )


def _restore_subgroup(root, fmt, source_team, pg, mesh, device):
    model, optimizer = make_model(2, target=True, device=device, mesh=mesh)
    state = {"model": model, "optimizer": optimizer, "extra_state": None}
    FSDP2Checkpointer.load(root / f"team-{source_team}" / fmt, state, process_group=pg)
    rank = get_rank(pg)
    assert state["extra_state"] == extra_for(2 * rank + source_team)
    ref, ref_opt = dense_oracle(source_team + 1, device=device)
    assert_model(model, ref, rank, 2)
    assert_optimizer(model, optimizer, ref, ref_opt, rank, 2)
    train_step(model, optimizer, 3)
    train_step(ref, ref_opt, 3)
    assert_model(model, ref, rank, 2)
    assert_optimizer(model, optimizer, ref, ref_opt, rank, 2)


def hsdp(args):
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

    device = "cpu" if args.backend == "gloo" else "cuda"
    # Shard peers have nonconsecutive global ranks. Row-major rank arithmetic
    # would address different rows for ranks 1 and 2.
    mesh = DeviceMesh(device, [[0, 2], [1, 3]], mesh_dim_names=("replica", "shard"))
    flat = init_device_mesh(device, (4,), mesh_dim_names=("dp",))
    by_replica = init_device_mesh(device, (4, 1), mesh_dim_names=("replica", "shard"))
    by_shard = init_device_mesh(device, (1, 4), mesh_dim_names=("replica", "shard"))
    root = Path(args.path)
    for name, source_mesh, destinations in (
        ("hsdp", mesh, (mesh, flat, by_replica, by_shard)),
        ("fsdp", flat, (mesh,)),
    ):
        _hsdp_roundtrip(root / name, source_mesh, destinations, device)
    if device == "cuda":
        _hsdp_backward(root / "autograd", mesh, by_shard)
    print(f"HSDP PASS {args.backend} rank={get_rank()}", flush=True)


def _hsdp_roundtrip(root, source_mesh, destinations, device):
    model, optimizer = make_model(4, device=device, mesh=source_mesh)
    for step in range(2):
        train_step(model, optimizer, step)
    for fmt in ("safetensors", "dcp", "hf"):
        path = root / fmt
        FSDP2Checkpointer.save(
            path, state_for(model, optimizer, get_rank()), checkpoint_format=fmt
        )
        for target in destinations:
            _restore_hsdp(path, target, device)


def _hsdp_backward(path, source_mesh, target_mesh):
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    def setup(mesh):
        layer = torch.nn.Linear(4, 8, device="cuda")
        fully_shard(layer, mesh=mesh)
        return layer, torch.optim.AdamW(layer.parameters(), foreach=False)

    def update(layer, opt, step):
        # Different gradients per replica exercise HSDP's replica reduction.
        x = torch.full((2, 4), (get_rank() + step + 1) / 8, device="cuda")
        layer(x).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    torch.manual_seed(17)
    source, optimizer = setup(source_mesh)
    update(source, optimizer, 0)
    FSDP2Checkpointer.save(path, {"model": source, "optimizer": optimizer})
    target, restored = setup(target_mesh)
    FSDP2Checkpointer.load(path, {"model": target, "optimizer": restored})
    update(source, optimizer, 1)
    update(target, restored, 1)
    for before, after in zip(source.parameters(), target.parameters(), strict=True):
        assert isinstance(before, DTensor) and isinstance(after, DTensor)
        torch.testing.assert_close(after.full_tensor(), before.full_tensor())
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                restored.state[after][key].full_tensor(),
                optimizer.state[before][key].full_tensor(),
            )


def _restore_hsdp(path, mesh, device):
    model, optimizer = make_model(4, target=True, device=device, mesh=mesh)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            local(tensor).fill_(77)
    state = {"model": model, "optimizer": optimizer, "extra_state": None}
    FSDP2Checkpointer.load(path, state)
    assert state["extra_state"] == extra_for(get_rank())
    reference, reference_optimizer = dense_oracle(2, device=device)
    coordinate = mesh.get_coordinate()
    assert coordinate is not None
    shard, parts = coordinate[-1], mesh.shape[-1]
    assert_model(model, reference, shard, parts)
    assert_optimizer(model, optimizer, reference, reference_optimizer, shard, parts)
    train_step(model, optimizer, 2)
    train_step(reference, reference_optimizer, 2)
    assert_model(model, reference, shard, parts)
    assert_optimizer(model, optimizer, reference, reference_optimizer, shard, parts)


def param_layouts(args):
    from tests.support.gdcp_param_layouts import parameter_layouts

    parameter_layouts(args)


def dp_tp(args):
    from tests.support.gdcp_dp_tp import dp_tp as run

    run(args)


def lazy_adamw(args):
    from tests.support.gdcp_cross_layout import lazy_adamw as run

    run(args)


def hsdp_tp(args):
    from tests.support.gdcp_dp_tp import hsdp_tp as run

    run(args)


def cross_save(args):
    from tests.support.gdcp_cross_layout import cross_layout

    cross_layout(args)


cross_exchange = cross_save


def hsdp_tp_source(args):
    from tests.support.gdcp_cross_layout import cross_hsdp_tp

    cross_hsdp_tp(args)


hsdp_tp_expand = hsdp_tp_contract = hsdp_tp_source


def optimizer_matrix(args):
    from tests.support.gdcp_optimizers import optimizer_matrix as run

    run(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "save",
            "load",
            "atomic",
            "strict",
            "partial",
            "global_steps",
            "extra_missing",
            "subgroups",
            "hsdp",
            "param_layouts",
            "dp_tp",
            "optimizer_matrix",
            "hsdp_tp",
            "lazy_adamw",
            "hsdp_tp_source",
            "hsdp_tp_expand",
            "hsdp_tp_contract",
            "cross_save",
            "cross_exchange",
            "gpu_save",
            "gpu_load",
        ),
    )
    parser.add_argument("path")
    parser.add_argument("--output")
    parser.add_argument("--oracle")
    parser.add_argument("--source-world", type=int, default=2)
    parser.add_argument("--continue-step", action="store_true")
    parser.add_argument(
        "--backend", choices=("nccl", "cuda:nccl,cpu:gloo", "gloo"), default="gloo"
    )
    args = parser.parse_args()
    if args.backend != "gloo":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(args.backend)
    try:
        globals()[args.action](args)
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
