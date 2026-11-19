"""CPU/Gloo torchrun worker for the GDCP v1 acceptance suite."""

import argparse
import io
import json
import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import torch
from torch.distributed import barrier, destroy_process_group, get_rank
from torch.distributed import get_world_size, init_process_group

from gdcp import FSDP2Checkpointer
from gdtensor_conn.coord import GlooCoordinator
from gdtensor_conn.iobackend import FileIO
from tests.support.gdcp_fsdp2 import (
    assert_model,
    assert_optimizer,
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


def poisoned(world):
    model, optimizer = make_model(world, target=True)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            local(tensor).fill_(77)
    return model, optimizer


def check(model, optimizer, steps=2):
    oracle, oracle_optimizer = dense_oracle(steps)
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


def _must_raise(function, name):
    error = None
    try:
        function()
    except Exception as exc:
        error = exc
    assert GlooCoordinator().vote(error is not None), f"every rank must raise: {name}"
    return error


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
        _must_raise(
            lambda: FSDP2Checkpointer.save(args.output, state), "rank-1 write fault"
        )
    assert get_rank() != 1 or triggered
    assert not (Path(args.output) / "gdcp_manifest.json").exists()
    _must_raise(lambda: FSDP2Checkpointer.load(args.output, state), "uncommitted load")
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
        error = _must_raise(
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
    destination = Path(args.output)
    filename = "extra_state_rank_1.pt"
    if get_rank() == 0:
        shutil.copytree(args.path, destination)
        (destination / "extra_state" / filename).unlink()
    barrier()
    sentinel = object()
    common = {"extra_state": sentinel}
    error = _must_raise(
        lambda: FSDP2Checkpointer.load(
            destination, common, load_common_state_only=True
        ),
        "missing declared extra-state file",
    )
    assert isinstance(error, RuntimeError) and filename in str(error), error
    assert common == {"extra_state": sentinel}
    print(
        f"ACCEPTANCE 9 PASS rank={get_rank()} missing declared file rejected with state unchanged",
        flush=True,
    )


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
        ),
    )
    parser.add_argument("path")
    parser.add_argument("--output")
    parser.add_argument("--oracle")
    parser.add_argument("--source-world", type=int, default=2)
    parser.add_argument("--continue-step", action="store_true")
    args = parser.parse_args()
    init_process_group("gloo")
    try:
        globals()[args.action](args)
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
