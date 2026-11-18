"""CPU/Gloo real-model native-DCP interchange, including tied bf16 weights."""

import argparse
from pathlib import Path

import torch
from torch.distributed import (
    destroy_process_group,
    get_rank,
    get_world_size,
    init_process_group,
)

from gdcp import FSDP2Checkpointer
from scripts.gdcp_v2_worker import native_load, native_save
from tests.support.gdcp_v2_models import build, jsonable, local, state_tensors


def _snapshot(model, optimizer):
    tensors, groups = state_tensors(model, optimizer)
    raw = {
        key: local(value).detach().reshape(-1).view(torch.uint8).numpy().tobytes()
        for key, value in tensors.items()
    }
    return raw, jsonable(groups)


def _poison(model, optimizer):
    tensors, _ = state_tensors(model, optimizer)
    with torch.no_grad():
        for value in tensors.values():
            local(value).fill_(77)
    optimizer.param_groups[0]["lr"] = 0.25


def run(args):
    rank, world = get_rank(), get_world_size()
    model, optimizer, _ = build(args.model, world, rank, "cpu", tiny=True)
    if args.action == "save":
        FSDP2Checkpointer.save(
            args.path, {"model": model, "optimizer": optimizer}, checkpoint_format="dcp"
        )
    elif args.action == "native-save":
        native_save(args.path, model, optimizer)
    else:
        _load(args, model, optimizer)
    if args.model == "4b":
        assert model.lm_head.weight is model.model.embed_tokens.weight
    print(
        f"DCP_REAL_INTEROP_PASS action={args.action} model={args.model} world={world} rank={rank}",
        flush=True,
    )


def _load(args, model, optimizer):
    before = _snapshot(model, optimizer)
    optimizer.step(synthetic_step=4)
    stepped = _snapshot(model, optimizer)
    _poison(model, optimizer)
    if args.action == "native-load":
        native_load(args.path, model, optimizer)
    else:
        FSDP2Checkpointer.load(args.path, {"model": model, "optimizer": optimizer})
    assert _snapshot(model, optimizer) == before, (
        "native interchange changed model/state/groups bytes"
    )
    optimizer.step(synthetic_step=4)
    assert _snapshot(model, optimizer) == stepped, (
        "native interchange changed poststep bytes"
    )
    for fields in optimizer.state.values():
        assert all(
            fields[field].dtype == torch.float32
            for field in ("master_param", "exp_avg", "exp_avg_sq")
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("save", "load", "native-save", "native-load")
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", choices=("4b", "14b", "24b"), default="4b")
    args = parser.parse_args()
    init_process_group("gloo")
    try:
        run(args)
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
