"""Batch tiny real-architecture checks into a few CPU/Gloo launches."""

import argparse
import json
from pathlib import Path

from torch.distributed import barrier, destroy_process_group, get_rank
from torch.distributed import get_world_size, init_process_group

from gdcp import FSDP2Checkpointer
from scripts.gdcp_v2_worker import atomic_json, check_equal, disk_guard, native_load
from scripts.gdcp_v2_worker import snapshot, timed
from tests.support.gdcp_v2_models import build

MODELS = ("4b", "14b", "24b")
FORMATS = ("safetensors", "dcp", "hf")


def _files(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _save(path, state, info, fmt):
    disk_guard(info["nominal_state_bytes"] + 2_000_000_000)
    return timed(
        lambda: FSDP2Checkpointer.save(
            path,
            state,
            checkpoint_format=fmt,
            max_shard_size=20_000 if fmt == "hf" else None,
        ),
        "cpu",
    )


def _reference(root, key):
    return root / f"reference-{key}.json"


def _source_case(root, key, fmt, rank, world):
    model, optimizer, info = build(key, world, rank, "cpu", tiny=True)
    if key == "4b":
        assert model.lm_head.weight is model.model.embed_tokens.weight
    state = {"model": model, "optimizer": optimizer}
    reference = snapshot(model, optimizer, rank, world)
    path = root / "source" / key / fmt
    duplicate = root / "duplicate" / key / fmt
    seconds = _save(path, state, info, fmt)
    _save(duplicate, state, info, fmt)
    barrier()
    if rank == 0:
        assert _files(path) == _files(duplicate), (key, fmt, "repeat F")
    optimizer.step(synthetic_step=4)
    stepped = snapshot(model, optimizer, rank, world)
    if rank == 0:
        atomic_json(
            _reference(root, key),
            {"reference": reference, "stepped_reference": stepped},
        )
    barrier()
    return {
        "model": key,
        "format": fmt,
        "world": world,
        "F": "pass",
        "save_seconds": seconds,
    }


def _target_case(root, key, fmt, rank, world):
    model, optimizer, info = build(key, world, rank, "cpu", tiny=True, poison=True)
    state = {"model": model, "optimizer": optimizer}
    source = root / "source" / key / fmt
    seconds = timed(lambda: FSDP2Checkpointer.load(source, state), "cpu")
    expected = json.loads(_reference(root, key).read_text())
    check_equal(snapshot(model, optimizer, rank, world), expected["reference"], "V")
    _save(root / f"target-{world}" / key / fmt, state, info, fmt)
    optimizer.step(synthetic_step=4)
    check_equal(
        snapshot(model, optimizer, rank, world), expected["stepped_reference"], "R"
    )
    return {
        "model": key,
        "format": fmt,
        "world": world,
        "V": "pass",
        "R": "pass",
        "local_shapes": "pass",
        "load_seconds": seconds,
    }


def _final_target(root, key, fmt, rank, world, target):
    model, optimizer, info = build(key, world, rank, "cpu", tiny=True, poison=True)
    state = {"model": model, "optimizer": optimizer}
    FSDP2Checkpointer.load(root / f"target-{target}" / key / fmt, state)
    expected = json.loads(_reference(root, key).read_text())
    check_equal(
        snapshot(model, optimizer, rank, world), expected["reference"], "V reverse"
    )
    resave = root / f"final-{target}" / key / fmt
    _save(resave, state, info, fmt)
    barrier()
    if rank == 0:
        assert _files(resave) == _files(root / "source" / key / fmt), (
            key,
            fmt,
            target,
            "reshard F",
        )
    return {
        "model": key,
        "format": fmt,
        "world": world,
        "target": target,
        "V": "pass",
        "F": "pass",
    }


def _native_case(root, key, rank, world):
    model, optimizer, _info = build(key, world, rank, "cpu", tiny=True, poison=True)
    native_load(root / "source" / key / "dcp", model, optimizer)
    expected = json.loads(_reference(root, key).read_text())
    check_equal(
        snapshot(model, optimizer, rank, world), expected["reference"], "native DCP V"
    )
    optimizer.step(synthetic_step=4)
    check_equal(
        snapshot(model, optimizer, rank, world),
        expected["stepped_reference"],
        "native DCP R",
    )
    return {
        "model": key,
        "format": "dcp",
        "world": world,
        "I": "pass",
        "V": "pass",
        "R": "pass",
    }


def _run(args, rank, world):
    records = []
    for key in MODELS:
        _run_model(args, key, rank, world, records)
    if rank == 0:
        atomic_json(
            args.root / f"results-{args.action}-{world}.json",
            {"status": "pass", "checks": records},
        )
    print(
        f"REAL_MODELS_V2_PASS action={args.action} world={world} rank={rank}",
        flush=True,
    )


def _run_model(args, key, rank, world, records):
    for fmt in FORMATS:
        _run_format(args, key, fmt, rank, world, records)
    if args.action == "target" and world == 3:
        records.append(_native_case(args.root, key, rank, world))


def _run_format(args, key, fmt, rank, world, records):
    if args.action == "source":
        records.append(_source_case(args.root, key, fmt, rank, world))
    elif args.action == "target":
        records.append(_target_case(args.root, key, fmt, rank, world))
    else:
        records.extend(
            _final_target(args.root, key, fmt, rank, world, target)
            for target in (1, 3, 4)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("source", "target", "final"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    init_process_group("gloo")
    try:
        _run(args, get_rank(), get_world_size())
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
