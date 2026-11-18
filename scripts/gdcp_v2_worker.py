"""One resumable v2 distributed check; writes hashes and measured evidence."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

import torch
from torch.distributed import (
    all_gather,
    all_gather_object,
    all_reduce,
    barrier,
    broadcast_object_list,
    destroy_process_group,
    gather,
    get_rank,
    get_world_size,
    init_process_group,
    ReduceOp,
)
from torch.distributed.tensor import DTensor

from gdcp import FSDP2Checkpointer, load_hf, save_hf
from tests.support.gdcp_v2_models import build, jsonable, local, state_tensors


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def tensor_hash(tensor, rank, world):
    host = local(tensor).detach().reshape(-1).view(torch.uint8).cpu()
    if not isinstance(tensor, DTensor):
        return hashlib.sha256(host.numpy().tobytes()).hexdigest()
    # Gather one tensor at a time over CPU Gloo, padding only its local shards.
    size = torch.tensor(host.numel(), dtype=torch.int64)
    sizes = [torch.empty_like(size) for _ in range(world)]
    all_gather(sizes, size)
    maximum = max(int(n) for n in sizes)
    padded = torch.zeros(maximum, dtype=torch.uint8)
    padded[: host.numel()].copy_(host)
    gathered = [torch.empty_like(padded) for _ in range(world)] if rank == 0 else None
    gather(padded, gather_list=gathered, dst=0)
    result = None
    if rank == 0:
        assert gathered is not None
        digest = hashlib.sha256()
        for shard, count in zip(gathered, sizes):
            digest.update(memoryview(shard.numpy())[: int(count)])
        result = digest.hexdigest()
    values = [result]
    broadcast_object_list(values, src=0)
    return values[0]


def verify_shape(name, tensor, rank, world):
    expected = list(tensor.shape)
    if isinstance(tensor, DTensor) and tensor.ndim:
        rows = (tensor.shape[0] + world - 1) // world
        expected[0] = max(0, min(rows, tensor.shape[0] - rank * rows))
    assert list(local(tensor).shape) == expected, (
        name,
        list(local(tensor).shape),
        expected,
    )


def snapshot(model, optimizer, rank, world):
    _assert_master_relation(model, optimizer)
    tensors, groups = state_tensors(model, optimizer)
    return {
        "tensors": _tensor_records(tensors, rank, world),
        "param_groups": jsonable(groups),
    }


def _tensor_records(tensors, rank, world):
    hashes = {}
    for name, tensor in tensors.items():
        verify_shape(name, tensor, rank, world)
        hashes[name] = {
            "sha256": tensor_hash(tensor, rank, world),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return hashes


def _weight_key(name, dtype):
    if dtype == "bf16":
        return name.startswith("model.")
    return name.startswith("optimizer.state.") and name.endswith(".master_param")


def weights_snapshot(model, optimizer, rank, world, dtype):
    """Hash the tensors a weights-only file can restore, without Adam moments."""
    _assert_master_relation(model, optimizer)
    tensors, _groups = state_tensors(model, optimizer)
    selected = {
        name: tensor for name, tensor in tensors.items() if _weight_key(name, dtype)
    }
    expected_dtype = torch.float32 if dtype == "fp32" else torch.bfloat16
    assert selected, "weights snapshot has no tensors"
    assert all(tensor.dtype == expected_dtype for tensor in selected.values()), (
        "weights dtype",
        dtype,
    )
    return {"tensors": _tensor_records(selected, rank, world)}


def timed(action, device):
    barrier()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.monotonic()
    action()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = torch.tensor(time.monotonic() - start, dtype=torch.float64)
    all_reduce(elapsed, op=ReduceOp.MAX)
    return elapsed.item()


def check_equal(actual, expected, label):
    assert actual["param_groups"] == expected["param_groups"], label + " param_groups"
    _check_tensor_records(actual["tensors"], expected["tensors"], label)


def check_weights_equal(actual, expected, dtype):
    selected = {
        name: value
        for name, value in expected["tensors"].items()
        if _weight_key(name, dtype)
    }
    _check_tensor_records(actual["tensors"], selected, "V weights " + dtype)


def _check_tensor_records(actual, expected, label):
    assert actual.keys() == expected.keys(), label + " tensor keys"
    for key, value in actual.items():
        assert value == expected[key], (
            label,
            key,
            value,
            expected[key],
        )


def disk_guard(bytes_to_write):
    # Executed collectively immediately before every save, also inside containers.
    from scripts.gdcp_v2_safety import check_disk

    return check_disk(bytes_to_write)


def save_full(args, state, info):
    disk_guard(info["nominal_state_bytes"] + 2_000_000_000)
    opts = {"checkpoint_format": args.format}
    if args.format == "hf":
        opts["max_shard_size"] = args.max_shard_size
    return timed(lambda: FSDP2Checkpointer.save(args.path, state, **opts), args.device)


def native_save(path, model, optimizer):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    dcp.save(get_model_state_dict(model), checkpoint_id=Path(path) / "model")
    dcp.save(
        get_optimizer_state_dict(model, optimizer),
        checkpoint_id=Path(path) / "optimizer",
    )


def native_load(path, model, optimizer):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
    )

    ms, osd = get_model_state_dict(model), get_optimizer_state_dict(model, optimizer)
    dcp.load(ms, checkpoint_id=Path(path) / "model")
    dcp.load(osd, checkpoint_id=Path(path) / "optimizer")
    from gdcp.adapters.fsdp2 import _install_optimizer_state

    _install_optimizer_state(model, optimizer, osd)
    set_model_state_dict(model, ms)


def run(args, rank, world):
    validate_args(args)
    model, optimizer, info = build(
        args.model,
        world,
        rank,
        args.device,
        tiny=args.tiny,
        poison=args.action in ("load", "native-load") and not args.weights_only,
    )
    if args.weights_only:
        _poison_model(model)
        if args.dtype == "fp32":
            _poison_master_weights(optimizer)
    state = {"model": model, "optimizer": optimizer}
    record: dict[str, Any] = {
        "started_pst": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "model": args.model,
        "tiny": args.tiny,
        "world_size": world,
        "device": args.device,
        "format": args.format,
        "action": args.action,
        "model_info": info,
        "optimizer_step_executed": False,
        "reference_scope": "omitted-unused",
    }
    emit_reference = args.action in ("save", "build") and not args.skip_reference
    if emit_reference:
        before = snapshot(model, optimizer, rank, world)
        record["reference"] = before
        record["reference_scope"] = "full-state"
    if args.action == "save":
        record["seconds"] = save_full(args, state, info)
    elif args.action in ("load", "native-load"):
        _load_check(args, state, info, record, rank, world)
    elif args.action == "native-save":
        disk_guard(info["nominal_state_bytes"] + 2_000_000_000)
        record["seconds"] = timed(
            lambda: native_save(args.path, model, optimizer), args.device
        )
    elif args.action == "export":
        _assert_master_relation(model, optimizer)
        record["master_cast_relation"] = "pass"
        disk_guard(
            info["parameters"] * (4 if args.dtype == "fp32" else 2) + 2_000_000_000
        )
        record["seconds"] = timed(
            lambda: save_hf(
                args.path, state, dtype=args.dtype, max_shard_size=args.max_shard_size
            ),
            args.device,
        )
    if emit_reference:
        optimizer.step(synthetic_step=4)
        record["optimizer_step_executed"] = True
        record["stepped_reference"] = snapshot(model, optimizer, rank, world)
    peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else 0
    peaks = [None] * world
    all_gather_object(peaks, peak)
    record["peak_gpu_allocated_bytes"] = peaks
    assert all(n <= 85 * 1024**3 for n in peaks), ("GPU_MEMORY_LIMIT", peaks)
    record["disk"] = disk_guard(0)
    record["finished_pst"] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
    record["status"] = "pass"
    if "seconds" in record:
        io_bytes = (
            info["parameters"] * (4 if args.dtype == "fp32" else 2)
            if args.action == "export" or args.weights_only
            else info["nominal_state_bytes"]
        )
        record["io_payload_bytes"] = io_bytes
        record["GB_per_second"] = io_bytes / 1e9 / record["seconds"]
    if rank == 0:
        atomic_json(args.result, record)
    print(
        f"GDCP_V2_PASS model={args.model} action={args.action} world={world} rank={rank}",
        flush=True,
    )


def _load_check(args, state, info, record, rank, world):
    model, optimizer = state["model"], state["optimizer"]
    if args.weights_only:
        target = state if args.dtype == "fp32" else model
        record["seconds"] = timed(
            lambda: load_hf(args.path, target, dtype=args.dtype), args.device
        )
        if args.dtype == "fp32":
            _sync_master_parameters(model, optimizer)
    elif args.action == "native-load":
        record["seconds"] = timed(
            lambda: native_load(args.path, model, optimizer), args.device
        )
    else:
        record["seconds"] = timed(
            lambda: FSDP2Checkpointer.load(args.path, state), args.device
        )
    reference = json.loads(Path(args.reference).read_text())
    if args.weights_only:
        after = weights_snapshot(model, optimizer, rank, world, args.dtype)
        check_weights_equal(after, reference["reference"], args.dtype)
        record["V_scope"] = "weights-" + args.dtype
        record["master_cast_relation"] = "pass"
    else:
        after = snapshot(model, optimizer, rank, world)
        check_equal(after, reference["reference"], "V")
        record["V_scope"] = "full-state"
    record["V"], record["local_shapes"] = "pass", "pass"
    if args.output:
        original = args.path
        args.path = args.output
        if args.weights_only:
            disk_guard(
                info["parameters"] * (4 if args.dtype == "fp32" else 2) + 2_000_000_000
            )
            record["resave_seconds"] = timed(
                lambda: save_hf(
                    args.path,
                    state,
                    dtype=args.dtype,
                    max_shard_size=args.max_shard_size,
                ),
                args.device,
            )
        else:
            record["resave_seconds"] = save_full(args, state, info)
        args.path = original
    if args.weights_only:
        record["R"] = "not-applicable-weights-only"
        return
    if args.skip_step_check:
        record["R"] = "not-applicable-return-to-source"
        record["R_coverage"] = (
            "target-world load performs R; this return checks full V and resave file identity"
        )
        return
    optimizer.step(synthetic_step=4)
    record["optimizer_step_executed"] = True
    stepped = snapshot(model, optimizer, rank, world)
    check_equal(stepped, reference["stepped_reference"], "R")
    record["R"] = "pass"


def _poison_model(model):
    with torch.no_grad():
        for parameter in model.parameters():
            local(parameter).fill_(77)


def _poison_master_weights(optimizer):
    with torch.no_grad():
        for fields in optimizer.state.values():
            local(fields["master_param"]).fill_(77)


def _assert_master_relation(model, optimizer):
    for parameter in model.parameters():
        master = optimizer.state[parameter]["master_param"]
        assert torch.equal(local(parameter), local(master).to(torch.bfloat16)), (
            "bf16 != master_param.to(bfloat16)"
        )


def _sync_master_parameters(model, optimizer):
    with torch.no_grad():
        for parameter in model.parameters():
            local(parameter).copy_(
                local(optimizer.state[parameter]["master_param"]).to(torch.bfloat16)
            )


def validate_args(args):
    if args.skip_reference and (args.action != "save" or args.weights_only):
        raise ValueError("--skip-reference is only valid for a full duplicate save")
    if args.skip_step_check and (args.action != "load" or args.weights_only):
        raise ValueError("--skip-step-check is only valid for a full return load")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("4b", "14b", "24b"), required=True)
    parser.add_argument(
        "--action",
        choices=("save", "load", "native-save", "native-load", "export", "build"),
        required=True,
    )
    parser.add_argument("--path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--format", choices=("safetensors", "dcp", "hf"), default="safetensors"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--max-shard-size", type=int, default=None)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-step-check", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group("cuda:nccl,cpu:gloo" if args.device == "cuda" else "gloo")
    try:
        run(args, get_rank(), get_world_size())
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
