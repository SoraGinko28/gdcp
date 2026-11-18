"""Standard HF readers/writers against per-tensor v2 reference hashes.

Run this CPU-only utility through the v2 controller. It reads one tensor at a
 time for verification. Native output uses safetensors' own writer and HF Hub's
standard sharding planner, with the optional GDCP optimizer JSON sidecar copied.
"""

import argparse
import gc
import hashlib
import json
import shutil
import time
from datetime import datetime
from math import prod
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from gdcp.hf import read_headers
from scripts.gdcp_v2_safety import check_disk


def now():
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tensor_digest(tensor):
    tensor = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(tensor.numpy())).hexdigest()


def reference_key(key, dtype, references):
    if key.startswith("optimizer."):
        return key
    master = f"optimizer.state.{key}.master_param"
    if dtype == "fp32" and key == "lm_head.weight" and master not in references:
        master = "optimizer.state.model.embed_tokens.weight.master_param"
    return master if dtype == "fp32" and master in references else "model." + key


def standard_tensor(path, name, entry):
    with safe_open(
        str(Path(path) / entry["file"]), framework="pt", device="cpu"
    ) as reader:
        return reader.get_tensor(name)


def validate_layout(path, *, sharded, headers=None):
    """Require the requested layout and exact on-disk/index file coverage.

    Merely setting max_shard_size does not guarantee multiple files: an HF
    writer legitimately emits a single file when the payload is smaller. The
    acceptance harness must reject that case when exercising its sharded cell.
    """
    root = Path(path)
    files = {file.name for file in root.glob("*.safetensors") if file.is_file()}
    index_path = root / "model.safetensors.index.json"
    if sharded:
        _validate_sharded_files(files, index_path)
    elif index_path.exists() or files != {"model.safetensors"}:
        raise ValueError(
            "requested single layout requires only model.safetensors and no index"
        )
    headers = read_headers(path) if headers is None else headers
    return {
        "status": "pass",
        "requested": "sharded" if sharded else "single",
        "actual": "sharded" if index_path.is_file() else "single",
        "files": sorted(files),
        "file_count": len(files),
        "index_file": index_path.name if index_path.is_file() else None,
        "tensor_count": len(headers),
        "indexed_file_coverage": "pass" if sharded else "not-applicable",
    }


def _validate_sharded_files(files, index_path):
    if not index_path.is_file() or len(files) < 2:
        raise ValueError(
            "requested sharded layout requires multiple safetensors files and an index"
        )
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("sharded layout requires a nonempty index weight_map")
    if any(not isinstance(value, str) for value in weight_map.values()):
        raise ValueError("sharded index filenames must be strings")
    if set(weight_map.values()) != files:
        raise ValueError(
            "sharded index file coverage differs from on-disk safetensors files"
        )
    expected = {
        f"model-{number:05d}-of-{len(files):05d}.safetensors"
        for number in range(1, len(files) + 1)
    }
    if files != expected:
        raise ValueError(
            "sharded layout requires complete canonical HF shard filenames"
        )


def verify_payload(args, headers, references):
    checked = {}
    for name, entry in headers.items():
        tensor = standard_tensor(args.path, name, entry)
        key = reference_key(name, args.dtype, references)
        if key not in references:
            raise AssertionError(f"missing reference tensor {key}")
        actual = {
            "sha256": tensor_digest(tensor),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        assert actual == references[key], (name, key, actual, references[key])
        checked[name] = {"reference_key": key, **actual}
        del tensor
    return checked


def verify_transformers(args, references):
    if args.dtype == "full" or args.model != "4b":
        return {
            "status": "header/tensor-checks",
            "reason": "CPU from_pretrained required for weights-only 4b",
        }
    model = AutoModelForCausalLM.from_pretrained(
        args.path, local_files_only=True, dtype="auto", device_map=None
    )
    checked = 0
    for name, parameter in model.state_dict().items():
        key = reference_key(name, args.dtype, references)
        actual = {
            "sha256": tensor_digest(parameter),
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        assert actual == references[key], (
            "transformers",
            name,
            actual,
            references[key],
        )
        checked += 1
    del model
    gc.collect()
    return {
        "status": "pass",
        "checked_state_keys": checked,
        "device": "cpu",
        "loader": "AutoModelForCausalLM.from_pretrained",
    }


def native_assignment(tensors, max_shard_size):
    if max_shard_size is None:
        return {"model.safetensors": list(tensors)}, {}
    split = split_torch_state_dict_into_shards(
        tensors,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )
    if not split.is_sharded:
        return {"model.safetensors": list(tensors)}, {}
    return split.filename_to_tensors, {
        "metadata": split.metadata,
        "weight_map": split.tensor_to_filename,
    }


def copy_sidecars(source, destination):
    for name in ("config.json", "optimizer.json"):
        path = Path(source) / name
        if path.is_file():
            shutil.copyfile(path, destination / name)
    extra = Path(source) / "extra_state"
    if extra.is_dir():
        shutil.copytree(extra, destination / "extra_state", dirs_exist_ok=True)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def byte_comparison(source, native):
    left = {p.name: digest_file(p) for p in Path(source).glob("*.safetensors*")}
    right = {p.name: digest_file(p) for p in Path(native).glob("*.safetensors*")}
    return {
        "byte_identical_to_native_tool": left == right,
        "gdcp_files": left,
        "native_files": right,
        "required": False,
    }


def write_native(args, headers):
    if not args.native_output:
        return {"status": "not-requested"}
    destination = Path(args.native_output)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    tensors = {
        name: standard_tensor(args.path, name, entry) for name, entry in headers.items()
    }
    files, index = native_assignment(tensors, args.max_shard_size)
    records = {}
    for filename, names in files.items():
        payload = sum(
            tensors[name].numel() * tensors[name].element_size() for name in names
        )
        admission = check_disk(payload + 2_000_000)
        start = time.monotonic()
        save_file(
            {name: tensors[name] for name in names},
            str(destination / filename),
            metadata={"format": "pt"},
        )
        records[filename] = {
            "seconds": time.monotonic() - start,
            "payload_bytes": payload,
            "disk_admission": admission,
        }
        print("NATIVE_HF_SAVE", filename, json.dumps(records[filename]), flush=True)
    if index:
        atomic_json(destination / "model.safetensors.index.json", index)
    copy_sidecars(args.path, destination)
    del tensors
    gc.collect()
    layout = validate_layout(destination, sharded=args.max_shard_size is not None)
    return {
        "status": "pass",
        "path": str(destination),
        "files": records,
        "layout": layout,
        **byte_comparison(args.path, destination),
    }


def run(args):
    record = {
        "started_pst": now(),
        "model": args.model,
        "tiny": args.tiny,
        "dtype": args.dtype,
        "path": args.path,
    }
    reference = json.loads(Path(args.reference).read_text())["reference"]
    headers = read_headers(args.path)
    record["layout"] = validate_layout(
        args.path, sharded=args.max_shard_size is not None, headers=headers
    )
    record["tensors"] = verify_payload(args, headers, reference["tensors"])
    record["I_safetensors"] = "pass"
    record["E"] = "pass" if args.dtype in ("fp32", "bf16") else "not-applicable"
    record["transformers"] = verify_transformers(args, reference["tensors"])
    record["native_output"] = write_native(args, headers)
    record["logical_bytes"] = sum(
        prod(value["shape"]) * value["itemsize"] for value in headers.values()
    )
    record["finished_pst"], record["status"] = now(), "pass"
    atomic_json(args.result, record)
    print("GDCP_V2_HF_INTEROP_PASS", args.model, args.dtype, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--model", choices=("4b", "14b", "24b"), required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "full"), required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--native-output")
    parser.add_argument("--max-shard-size", type=int)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        atomic_json(
            args.result, {"status": "fail", "finished_pst": now(), "error": repr(exc)}
        )
        raise


if __name__ == "__main__":
    main()
