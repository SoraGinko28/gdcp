"""Compare default-accessor published bytes and storage IO across source trees.

Run this module separately under the baseline and candidate PYTHONPATH, using
the same --output directory name in separate parents. File rendezvous avoids
the standalone torchrun hostname lookup on macOS. No accessor is supplied.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.multiprocessing as mp
from torch.distributed import barrier, destroy_process_group, init_process_group
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from gdcp import FSDP2Checkpointer
from gdtensor_conn.iobackend import FileIO


def _model(mesh):
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.Linear(16, 8, bias=False)
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(
                torch.arange(parameter.numel()).reshape(parameter.shape) / 128 + index
            )
    model.register_buffer("integer_buffer", torch.arange(3, dtype=torch.int64))
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.125, foreach=False)
    for index, parameter in enumerate(model.parameters()):
        optimizer.state[parameter].update(
            step=torch.tensor(3.0),
            exp_avg=torch.ones_like(parameter) * (index + 0.125),
            exp_avg_sq=torch.ones_like(parameter) * (index + 0.25),
        )
    return model, optimizer


def _record_io(root):
    """Record ordered connector calls plus the checkpoint's audited FS calls."""
    calls, audits, originals = [], [], {}
    active = [True]
    prefix = str(root)

    def normalize(value):
        if isinstance(value, (str, os.PathLike)):
            return os.fspath(value).replace(prefix, "$ROOT")
        if isinstance(value, bytes):
            return {"bytes": len(value), "sha256": sha256(value).hexdigest()}
        if isinstance(value, (bytearray, memoryview)):
            return {"bytes": len(value), "sha256": sha256(value).hexdigest()}
        return value

    for method in (
        "create_staging",
        "sync",
        "publish",
        "discard",
        "read_all",
        "write_all",
        "identity",
        "pwrite",
        "pread",
        "close_rank",
    ):
        original = getattr(FileIO, method)
        originals[method] = original

        def call(self, *args, _method=method, _original=original, **kwargs):
            result = _original(self, *args, **kwargs)
            calls.append([_method, [normalize(arg) for arg in args], kwargs])
            return result

        setattr(FileIO, method, call)

    def audit(event, args):
        if not active[0] or event not in {
            "open",
            "os.mkdir",
            "os.rmdir",
            "os.remove",
            "os.rename",
            "os.scandir",
        }:
            return
        if any(isinstance(a, str) and prefix in a for a in args):
            audits.append([event, [normalize(a) for a in args]])

    sys.addaudithook(audit)

    def finish():
        active[0] = False
        for method, original in originals.items():
            setattr(FileIO, method, original)
        return {
            "connector_calls": calls,
            "filesystem_audit_multiset": sorted(
                Counter(json.dumps(a, sort_keys=True) for a in audits).items()
            ),
        }

    return finish


def worker(rank, world, rendezvous, output, layout):
    init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        shape = (2, 2) if layout == "hsdp" else (world,)
        mesh = DeviceMesh(
            "cpu",
            torch.arange(world).reshape(shape),
            mesh_dim_names=("replica", "shard") if layout == "hsdp" else ("shard",),
        )
        model, optimizer = _model(mesh)
        for fmt in ("safetensors", "dcp", "hf"):
            root = Path(output) / layout / fmt
            root.parent.mkdir(parents=True, exist_ok=True)
            finish = _record_io(root.parent)
            FSDP2Checkpointer.save(
                root, {"model": model, "optimizer": optimizer}, checkpoint_format=fmt
            )
            trace = {"save": finish()}
            target, restored = _model(mesh)
            finish = _record_io(root.parent)
            FSDP2Checkpointer.load(root, {"model": target, "optimizer": restored})
            trace["load"] = finish()
            _assert_restored(model, optimizer, target, restored)
            (root.parent / f"{fmt}-rank-{rank}-io.json").write_text(
                json.dumps(trace, indent=2, sort_keys=True) + "\n"
            )
            barrier()
    finally:
        destroy_process_group()


def _assert_restored(model, optimizer, target, restored):
    for source, actual in zip(model.parameters(), target.parameters(), strict=True):
        assert isinstance(source, DTensor) and isinstance(actual, DTensor)
        torch.testing.assert_close(source.to_local(), actual.to_local(), rtol=0, atol=0)
        _assert_optimizer_state(optimizer.state[source], restored.state[actual])


def _assert_optimizer_state(source, restored):
    for key, value in source.items():
        got = restored[key]
        if isinstance(value, DTensor):
            value, got = value.to_local(), got.to_local()
        torch.testing.assert_close(value, got, rtol=0, atol=0)


def manifest(root):
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def comparison(left, right):
    first, second = manifest(left), manifest(right)
    first.pop("manifest.json", None)
    second.pop("manifest.json", None)
    names = sorted(first.keys() | second.keys())
    changed = [name for name in names if first.get(name) != second.get(name)]
    cases = []
    for layout in ("fsdp", "hsdp"):
        for fmt in ("safetensors", "dcp", "hf"):
            cases.append(_comparison_case(layout, fmt, names, changed))
    return {
        "status": "fail" if changed else "pass",
        "cases": cases,
        "changed": changed,
        "normalization": "output-parent absolute path only",
    }


def _comparison_case(layout, fmt, names, changed):
    files = [name for name in names if name.startswith(f"{layout}/{fmt}/")]
    traces = [name for name in names if name.startswith(f"{layout}/{fmt}-rank-")]
    return {
        "layout": layout,
        "format": fmt,
        "published_files": len(files),
        "rank_traces": len(traces),
        "changed_files": [name for name in files if name in changed],
        "changed_traces": [name for name in traces if name in changed],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    if args.compare:
        result = comparison(*args.compare)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if result["changed"]:
            raise SystemExit(1)
        return
    for layout, world in (("fsdp", 2), ("hsdp", 4)):
        with tempfile.TemporaryDirectory(prefix="gdcp-i11-") as directory:
            mp.spawn(
                worker,
                args=(
                    world,
                    str(Path(directory) / "rendezvous"),
                    str(args.output),
                    layout,
                ),
                nprocs=world,
                join=True,
            )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest(args.output), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
