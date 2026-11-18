"""Resumable serial H20 controller. Run visibly in the existing h20 tmux."""

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gdcp_v2_safety import (
    GpuGuard,
    GpuSafetyStop,
    check_disk,
    check_host_memory,
    container_pids,
    pst_now,
)

LOG = Path("/data00/iobench/logs/gdcp-v2")
CKPT = Path("/data00/iobench/gdcp-v2-checkpoints")
SOURCE = Path("/data00/iobench/gdcp")
GUARD = GpuGuard()
SOURCE_SHA = None
EXERCISED_SHA = None


class CheckSkipped(RuntimeError):
    """An invalid attempt or unavailable prerequisite cannot count as acceptance."""


class CheckLimit(RuntimeError):
    """Explicit tiny-validation stop after a bounded number of new checks."""


def write_json(path, value):
    if SOURCE_SHA:
        value = dict(value, source_manifest_sha256=SOURCE_SHA)
    if EXERCISED_SHA:
        value = dict(value, exercised_code_sha256=EXERCISED_SHA)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def check_done(result):
    if result.is_file():
        value = json.loads(result.read_text())
        return value.get("status") == "pass" and (
            not SOURCE_SHA
            or (EXERCISED_SHA and value.get("exercised_code_sha256") == EXERCISED_SHA)
            or value.get("source_manifest_sha256") == SOURCE_SHA
        )
    return False


def worker(args, name, model, action, world, device, fmt="safetensors", **options):
    result = LOG / args.stage / model / (name + ".json")
    artifacts = [Path(options[k]) for k in ("output",) if options.get(k)]
    if action in ("save", "native-save", "export") and options.get("path"):
        artifacts.append(Path(options["path"]))
    if check_done(result) and all(p.is_dir() for p in artifacts):
        print("RESUME", result, flush=True)
        return result
    container = "iobench-gdcp-v2-" + ("gpu" if device == "cuda" else "cpu")
    command = [
        "docker",
        "exec",
        "-e",
        "PYTHONPATH=" + str(SOURCE),
        "-e",
        "TZ=America/Los_Angeles",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "HF_HUB_OFFLINE=1",
        container,
        "torchrun",
        "--standalone",
        f"--nproc-per-node={world}",
        "scripts/gdcp_v2_worker.py",
        "--model",
        model,
        "--action",
        action,
        "--device",
        device,
        "--format",
        fmt,
        "--result",
        str(result),
    ]
    if args.stage in ("A", "B"):
        command.append("--tiny")
    for key, value in options.items():
        if value is True:
            command += ["--" + key.replace("_", "-")]
        elif value is not None and value is not False:
            command += ["--" + key.replace("_", "-"), str(value)]
    result.parent.mkdir(parents=True, exist_ok=True)
    if device == "cpu":
        check_host_memory(memory_requirement(args, model))
    attempt = 0
    while True:
        if device == "cuda":
            GUARD.wait_until_clear(job=str(result))
        watcher = (
            GUARD.watch_job(job=str(result)) if device == "cuda" else nullcontext()
        )
        with watcher as observation:
            if observation is not None and observation.error:
                raise GpuSafetyStop(
                    "GPU prelaunch telemetry failed: " + str(observation.error)
                )
            if observation is not None and observation.foreign_seen:
                rc = None
            else:
                print(
                    "COMMAND", json.dumps(command), "attempt", attempt + 1, flush=True
                )
                rc = run_command(command, result)
        if rc is None:
            # Foreign work arrived between admission and the watcher's first sample.
            continue
        attempt += 1
        if observation is not None and (observation.foreign_seen or observation.error):
            archive = result.parent / (name + "-disturbed-attempt" + str(attempt))
            archive.mkdir(parents=True, exist_ok=True)
            for path in (result, result.with_suffix(".log")):
                if path.exists():
                    path.replace(archive / path.name)
            write_json(
                archive / "disturbance.json",
                {
                    "status": "disturbed",
                    "returncode": rc,
                    "command": command,
                    "observations": observation.observations,
                    "error": str(observation.error) if observation.error else None,
                },
            )
            # A disturbed attempt is neither an acceptance pass nor a model failure.
            if observation.error:
                raise GpuSafetyStop("GPU monitoring failed: " + str(observation.error))
            GUARD.wait_until_clear(job=str(result) + " after disturbance")
            for path in artifacts:
                remove(path)
            if attempt >= 2:
                raise CheckSkipped(
                    "Both initial attempt and its one rerun were disturbed; see "
                    + str(archive)
                )
            continue
        passed = (
            result.is_file() and json.loads(result.read_text()).get("status") == "pass"
        )
        if rc or not passed:
            previous = json.loads(result.read_text()) if result.is_file() else {}
            write_json(
                result,
                {
                    **previous,
                    "status": "fail",
                    "returncode": rc,
                    "command": command,
                    "finished_pst": datetime.now(
                        timezone(timedelta(hours=-8))
                    ).isoformat(),
                },
            )
            raise RuntimeError(f"check failed {result}")
        write_json(result, json.loads(result.read_text()))
        return result


def run_command(command, result):
    with result.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        return process.wait()


def remove(path):
    if path.exists():
        shutil.rmtree(path)


def compare_files(args, model, name, first, second):
    record = LOG / args.stage / model / (name + ".json")
    if check_done(record):
        return record
    started = time.monotonic()
    if not first.is_dir() or not second.is_dir():
        raise RuntimeError("comparison checkpoint missing")
    left = {str(p.relative_to(first)): p for p in first.rglob("*") if p.is_file()}
    right = {str(p.relative_to(second)): p for p in second.rglob("*") if p.is_file()}
    failures = (
        ["file names: " + str(sorted(left.keys() ^ right.keys()))]
        if left.keys() != right.keys()
        else []
    )
    hashes = {}
    for key in left.keys() & right.keys():
        a, b = digest_file(left[key]), digest_file(right[key])
        hashes[key] = a
        if a != b:
            failures.append(f"{key}: {a} != {b}")
    status = "fail" if failures else "pass"
    write_json(
        record,
        {
            "status": status,
            "F": status,
            "seconds": time.monotonic() - started,
            "bytes": sum(p.stat().st_size for p in left.values()),
            "files": hashes,
            "errors": failures,
            "finished_pst": datetime.now(timezone(timedelta(hours=-8))).isoformat(),
        },
    )
    assert not failures, (name, failures)
    return record


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        block = stream.read(16 * 1024**2)
        while block:
            h.update(block)
            block = stream.read(16 * 1024**2)
    return h.hexdigest()


def targets(model):
    if model == "4b":
        return [(1, "cuda"), (4, "cuda"), (5, "cuda"), (8, "cpu")]
    if model == "14b":
        return [(3, "cuda"), (4, "cuda"), (5, "cuda"), (8, "cpu")]
    return [(4, "cuda"), (5, "cuda"), (3, "cpu"), (8, "cpu")]


def full_format(args, model, fmt, shard_size=None):
    marker = (
        LOG
        / args.stage
        / model
        / (fmt + ("-sharded" if shard_size else "") + "-complete.json")
    )
    if check_done(marker):
        return
    directory = (
        CKPT / args.stage / model / fmt / ("sharded" if shard_size else "single")
    )
    first, dup, bridge, again = [
        directory / x for x in ("first", "duplicate", "bridge", "again")
    ]
    prefix = fmt + ("-sharded" if shard_size else "")
    ref = worker(
        args,
        prefix + "-save6",
        model,
        "save",
        6,
        "cuda",
        fmt,
        path=first,
        max_shard_size=shard_size,
    )
    worker(
        args,
        prefix + "-duplicate6",
        model,
        "save",
        6,
        "cuda",
        fmt,
        path=dup,
        max_shard_size=shard_size,
        skip_reference=True,
    )
    compare_files(args, model, prefix + "-F-repeat6", first, dup)
    remove(dup)
    if fmt == "dcp":
        dcp_interop(args, model, prefix, first, bridge, ref)
    if fmt == "hf":
        hf_interop(args, model, prefix, first, bridge, ref, "full", shard_size)
        worker(
            args,
            prefix + "-I-native-hf-load",
            model,
            "load",
            4 if model != "24b" else 5,
            "cuda",
            fmt,
            path=bridge,
            reference=ref,
        )
        remove(bridge)
    for world, device in targets(model):
        target_roundtrip(
            args,
            model,
            fmt,
            prefix,
            first,
            bridge,
            again,
            ref,
            world,
            device,
            shard_size,
        )
    reverse = 5 if model == "24b" else 4
    remove(bridge)
    reverse_ref = worker(
        args,
        prefix + f"-reverse-save{reverse}",
        model,
        "save",
        reverse,
        "cuda",
        fmt,
        path=bridge,
        max_shard_size=shard_size,
    )
    worker(
        args,
        prefix + f"-reverse{reverse}-load6",
        model,
        "load",
        6,
        "cuda",
        fmt,
        path=bridge,
        reference=reverse_ref,
    )
    remove(directory)
    write_json(marker, {"status": "pass"})


def target_roundtrip(
    args, model, fmt, prefix, first, bridge, again, ref, world, device, shard_size
):
    remove(bridge)
    remove(again)
    label = prefix + f"-6to{world}-{device}"
    try:
        worker(
            args,
            label,
            model,
            "load",
            world,
            device,
            fmt,
            path=first,
            reference=ref,
            output=bridge,
            max_shard_size=shard_size,
        )
    except GpuSafetyStop:
        raise
    except RuntimeError:
        log = (LOG / args.stage / model / (label + ".log")).read_text()
        if device != "cuda" or not any(
            s in log for s in ("GPU_MEMORY_LIMIT", "CUDA out of memory")
        ):
            raise
        write_json(
            LOG / args.stage / model / (label + "-fallback.json"),
            {
                "status": "pass",
                "reason": "GPU peak exceeds 85 GiB or CUDA OOM",
                "log": str(LOG / args.stage / model / (label + ".log")),
            },
        )
        remove(bridge)
        worker(
            args,
            label + "-fallback-cpu",
            model,
            "load",
            world,
            "cpu",
            fmt,
            path=first,
            reference=ref,
            output=bridge,
            max_shard_size=shard_size,
        )
    worker(
        args,
        label + "-back6",
        model,
        "load",
        6,
        "cuda",
        fmt,
        path=bridge,
        reference=ref,
        output=again,
        max_shard_size=shard_size,
        skip_step_check=True,
    )
    compare_files(args, model, label + "-F-roundtrip", first, again)
    remove(bridge)
    remove(again)


def cpu_stage(args, model):
    for world in range(1, 5):
        for fmt in ("safetensors", "dcp", "hf"):
            base = CKPT / "A" / model / f"{fmt}-{world}"
            ref = worker(
                args, f"{fmt}-save{world}", model, "save", world, "cpu", fmt, path=base
            )
            target = world % 4 + 1
            worker(
                args,
                f"{fmt}-{world}to{target}",
                model,
                "load",
                target,
                "cpu",
                fmt,
                path=base,
                reference=ref,
            )
            remove(base)


def dcp_interop(args, model, prefix, first, native, reference):
    worker(
        args,
        prefix + "-I-native-load",
        model,
        "native-load",
        4 if model != "24b" else 5,
        "cuda",
        "dcp",
        path=first,
        reference=reference,
    )
    remove(native)
    worker(
        args,
        prefix + "-I-native-save",
        model,
        "native-save",
        6,
        "cuda",
        "dcp",
        path=native,
    )
    worker(
        args,
        prefix + "-I-gdcp-load-native",
        model,
        "load",
        4 if model != "24b" else 5,
        "cuda",
        "dcp",
        path=native,
        reference=reference,
    )
    native_comparison(args, model, prefix + "-I-file-comparison", first, native)
    remove(native)


def native_comparison(args, model, label, source, native):
    records = {}
    for component in ("model", "optimizer"):
        names = [
            p.name
            for p in (native / component).iterdir()
            if p.name == ".metadata" or p.suffix == ".distcp"
        ]
        records[component] = {
            name: {
                "gdcp_sha256": digest_file(source / component / name)
                if (source / component / name).is_file()
                else None,
                "native_sha256": digest_file(native / component / name),
            }
            for name in names
        }
    identical = all(
        v["gdcp_sha256"] == v["native_sha256"]
        for c in records.values()
        for v in c.values()
    )
    write_json(
        LOG / args.stage / model / (label + ".json"),
        {
            "status": "pass",
            "byte_identical_to_native_tool": identical,
            "files": records,
        },
    )


def weights_exports(args, model):
    reference = LOG / args.stage / model / "safetensors-save6.json"
    for dtype in ("fp32", "bf16"):
        for sharded in (False, True):
            weights_export(args, model, dtype, sharded, reference)


def weights_export(args, model, dtype, sharded, reference):
    label = "weights-" + dtype + ("-sharded" if sharded else "-single")
    base = CKPT / args.stage / model / label
    first, dup, native, bridge, again = [
        base / x for x in ("first", "duplicate", "native", "bridge", "again")
    ]
    limit = max_shard_size(args, model) if sharded else None
    worker(
        args,
        label + "-export6",
        model,
        "export",
        6,
        "cuda",
        "hf",
        path=first,
        dtype=dtype,
        max_shard_size=limit,
    )
    worker(
        args,
        label + "-duplicate6",
        model,
        "export",
        6,
        "cuda",
        "hf",
        path=dup,
        dtype=dtype,
        max_shard_size=limit,
    )
    compare_files(args, model, label + "-F-repeat", first, dup)
    remove(dup)
    smaller = 1 if model == "4b" else 4 if model == "14b" else 5
    for world, device in ((smaller, "cuda"), (8, "cpu")):
        worker(
            args,
            label + f"-load{world}-{device}",
            model,
            "load",
            world,
            device,
            "hf",
            path=first,
            reference=reference,
            weights_only=True,
            dtype=dtype,
            output=bridge,
            max_shard_size=limit,
        )
        worker(
            args,
            label + f"-back6-from{world}-{device}",
            model,
            "load",
            6,
            "cuda",
            "hf",
            path=bridge,
            output=again,
            reference=reference,
            weights_only=True,
            dtype=dtype,
            max_shard_size=limit,
        )
        compare_files(
            args, model, label + f"-F-roundtrip-{world}-{device}", first, again
        )
        remove(bridge)
        remove(again)
    hf_interop(args, model, label, first, native, reference, dtype, limit)
    for world, device in ((smaller, "cuda"), (8, "cpu")):
        worker(
            args,
            label + f"-native-load{world}-{device}",
            model,
            "load",
            world,
            device,
            "hf",
            path=native,
            reference=reference,
            weights_only=True,
            dtype=dtype,
        )
    remove(base)


def hf_interop(args, model, label, source, native, reference, dtype, limit):
    result = LOG / args.stage / model / (label + "-I-E.json")
    if check_done(result) and (native is None or native.exists()):
        return
    check_host_memory(memory_requirement(args, model))
    check_disk(
        0
        if native is None
        else 14 * 24_000_000_000
        if dtype == "full"
        else (4 if dtype == "fp32" else 2) * 24_000_000_000
    )
    command = [
        "docker",
        "exec",
        "-e",
        "PYTHONPATH=" + str(SOURCE),
        "-e",
        "TZ=America/Los_Angeles",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "HF_HUB_OFFLINE=1",
        "iobench-gdcp-v2-cpu",
        "python",
        "scripts/gdcp_v2_hf_interop.py",
        "--path",
        str(source),
        "--reference",
        str(reference),
        "--model",
        model,
        "--dtype",
        dtype,
        "--result",
        str(result),
    ]
    if native is not None:
        command += ["--native-output", str(native)]
    if args.stage == "B":
        command.append("--tiny")
    if limit:
        command += ["--max-shard-size", str(limit)]
    with result.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        rc = process.wait()
    passed = result.is_file() and json.loads(result.read_text()).get("status") == "pass"
    if rc or not passed:
        write_json(result, {"status": "fail", "returncode": rc, "command": command})
        raise RuntimeError("HF interop failed " + str(result))
    write_json(result, json.loads(result.read_text()))


def model_stage(args, model):
    if args.stage == "A":
        cpu_stage(args, model)
        return
    graph = build_plan(args, model)
    graph.run()


def max_shard_size(args, model):
    return 16384 if args.stage == "B" else 5_000_000_000


def source_manifest():
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".DS_Store",
    }
    files = {
        str(path.relative_to(SOURCE)): digest_file(path)
        for path in sorted(SOURCE.rglob("*"))
        if path.is_file() and not ignored.intersection(path.relative_to(SOURCE).parts)
    }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "sha256": hashlib.sha256(encoded).hexdigest()}


def memory_requirement(args, model):
    if args.stage in ("A", "B"):
        return 1024**3
    # Full state plus a second state/tensor staging allowance, below host RAM.
    parameters = {"4b": 4_100_000_000, "14b": 14_800_000_000, "24b": 24_000_000_000}
    return 2 * 14 * parameters[model] + 64 * 1024**3


def exercised_files(manifest):
    """Payload/value code only; controller and sharing changes do not invalidate data."""
    prefixes = ("gdtensor/", "gdtensor_conn/", "gdcp/", "tests/fixtures/")
    exact = {
        "scripts/gdcp_v2_worker.py",
        "scripts/gdcp_v2_hf_interop.py",
        "tests/support/gdcp_v2_models.py",
        "tests/support/gdcp_fsdp2.py",
    }
    return {
        name: digest
        for name, digest in manifest["files"].items()
        if name.startswith(prefixes) or name in exact
    }


def exercised_digest(manifest):
    encoded = json.dumps(
        exercised_files(manifest), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def import_resume(args, manifest):
    prior = getattr(args, "resume_root", None)
    if not prior:
        return
    old = json.loads((prior / "source-manifest.json").read_text())
    old_files, new_files = exercised_files(old), exercised_files(manifest)
    changed = sorted(
        name
        for name in old_files.keys() | new_files.keys()
        if old_files.get(name) != new_files.get(name)
    )
    record: dict = {
        "resume_root": str(prior),
        "changed_exercised_files": changed,
        "reason": "Controller, GPU sharing and reporting changes do not change checkpoint or value code.",
        "imported": [],
    }
    if changed:
        record["reason"] = (
            "Exercised source changed; prior acceptance checks must rerun."
        )
        print("RESUME_INVALIDATED", json.dumps(record), flush=True)
    else:
        for path in sorted((prior / args.stage).glob("*/*.json")):
            value = json.loads(path.read_text())
            if value.get("status") != "pass":
                continue
            destination = LOG / path.relative_to(prior)
            if destination.exists():
                continue
            value.update(
                evidence_origin=str(path),
                log_origin=str(path.with_suffix(".log")),
                resumed_from=str(prior),
                original_source_manifest_sha256=value.get("source_manifest_sha256"),
            )
            write_json(destination, value)
            record["imported"].append(str(destination))
    write_json(LOG / "resume-import.json", record)


class CheckGraph:
    """Evaluate independent checks even when one dependency branch fails."""

    def __init__(self, args, model):
        self.args, self.model = args, model
        self.checks = {}
        self.active = set()
        self.new_checks = 0

    def add(self, name, runner, dependencies=(), outputs=(), metadata=None, cleanup=()):
        self.checks[name] = dict(
            runner=runner,
            dependencies=list(dependencies),
            outputs=list(outputs),
            metadata=metadata or {},
            cleanup=list(cleanup),
        )
        return name

    def path(self, name):
        return LOG / self.args.stage / self.model / (name + ".json")

    def ensure(self, name, artifacts=False):
        node, record = self.checks[name], self.path(name)
        passed = check_done(record)
        if passed and (not artifacts or all(Path(p).is_dir() for p in node["outputs"])):
            return True
        if record.exists() and not passed:
            value = json.loads(record.read_text())
            if value.get("status") in ("fail", "skipped"):
                return False
        rebuilding = passed
        if name in self.active:
            raise RuntimeError("cyclic check dependency " + name)
        self.active.add(name)
        try:
            need_artifacts = node["metadata"].get("kind") != "format-sentinel"
            missing = [
                dep
                for dep in node["dependencies"]
                if not self.ensure(dep, artifacts=need_artifacts)
            ]
            if missing:
                if not rebuilding:
                    self.finish(
                        name,
                        "skipped",
                        reason="prerequisite failed or unavailable: "
                        + ", ".join(missing),
                    )
                for path in node["cleanup"]:
                    remove(Path(path))
                return False
            maximum = getattr(self.args, "max_new_checks", None)
            if maximum and not rebuilding and self.new_checks >= maximum:
                raise CheckLimit("explicit validation check limit")
            started = time.monotonic()
            saved = record.read_bytes() if rebuilding else None
            original_log = record.with_suffix(".log")
            saved_log = (
                original_log.read_bytes()
                if rebuilding and original_log.exists()
                else None
            )

            def restore_original():
                rebuilt = (
                    LOG / self.args.stage / self.model / "rebuild" / (name + ".json")
                )
                rebuilt.parent.mkdir(parents=True, exist_ok=True)
                if original_log.exists():
                    original_log.replace(rebuilt.with_suffix(".log"))
                if saved_log is not None:
                    original_log.write_bytes(saved_log)
                record.write_bytes(saved)

            try:
                injected = getattr(self.args, "inject_failure", None)
                if injected == name:
                    for path in node["outputs"]:
                        Path(path).mkdir(parents=True, exist_ok=True)
                        (Path(path) / "injected-partial").write_text(
                            "tiny failure fixture\n"
                        )
                    raise RuntimeError("SIMULATED_CHECK_FAILURE on tiny configuration")
                if rebuilding:
                    print(
                        "REBUILD_PREREQUISITE",
                        name,
                        "original acceptance preserved",
                        flush=True,
                    )
                    # A missing artifact is maintenance, not a new acceptance check.
                    record.unlink()
                node["runner"]()
                value = (
                    json.loads(record.read_text())
                    if record.exists()
                    else {"status": "pass"}
                )
                if value.get("status") != "pass":
                    raise RuntimeError("check did not publish a pass result")
                value = dict(node["metadata"], **value)
                value.update(
                    elapsed_seconds=time.monotonic() - started,
                    evidence_path=str(record),
                    log_path=str(record.with_suffix(".log")),
                )
                write_json(record, value)
                if rebuilding:
                    rebuilt = (
                        LOG
                        / self.args.stage
                        / self.model
                        / "rebuild"
                        / (name + ".json")
                    )
                    rebuilt.parent.mkdir(parents=True, exist_ok=True)
                    record.replace(rebuilt)
                    log = record.with_suffix(".log")
                    if log.exists():
                        log.replace(rebuilt.with_suffix(".log"))
                    restore_original()
                else:
                    self.new_checks += 1
                    with (LOG / "sentinels.log").open("a") as stream:
                        stream.write(
                            "__GDCP_CHECK_DONE model=%s check=%s status=pass\n"
                            % (self.model, name)
                        )
                return True
            except (GpuSafetyStop, CheckLimit):
                if rebuilding:
                    restore_original()
                raise
            except Exception as exc:
                status = "skipped" if isinstance(exc, CheckSkipped) else "fail"
                if rebuilding:
                    rebuild = (
                        LOG
                        / self.args.stage
                        / self.model
                        / "rebuild"
                        / (name + ".json")
                    )
                    write_json(
                        rebuild,
                        {"status": status, "error": repr(exc), "maintenance": True},
                    )
                    restore_original()
                else:
                    self.finish(
                        name,
                        status,
                        error=repr(exc),
                        seconds=time.monotonic() - started,
                        traceback=traceback.format_exc(),
                    )
                    self.new_checks += 1
                for path in node["outputs"]:
                    remove(Path(path))
                return False
            finally:
                for path in node["cleanup"]:
                    remove(Path(path))
        finally:
            self.active.remove(name)

    def finish(self, name, status, **fields):
        record = self.path(name)
        prior = json.loads(record.read_text()) if record.exists() else {}
        value = dict(prior, **self.checks[name]["metadata"])
        value.update(
            status=status,
            model=self.model,
            check=name,
            finished_pst=pst_now(),
            evidence_path=str(record),
            log_path=str(record.with_suffix(".log")),
            **fields,
        )
        if not record.with_suffix(".log").exists():
            record.parent.mkdir(parents=True, exist_ok=True)
            record.with_suffix(".log").write_text(
                fields.get(
                    "traceback", fields.get("reason", fields.get("error", status))
                )
                + "\n"
            )
        write_json(record, value)

    def run(self):
        for name in self.checks:
            self.ensure(name)
            if hasattr(self.args, "models"):
                summary(self.args)


def build_plan(args, model):
    graph = CheckGraph(args, model)
    reduced = args.stage == "C" and model != "4b"
    formats = getattr(args, "formats", None)

    def work(
        name,
        action,
        world,
        device,
        fmt,
        dependencies=(),
        cleanup=(),
        source_world=6,
        **options,
    ):
        outputs = [Path(options[k]) for k in ("output",) if options.get(k)]
        if action in ("save", "native-save", "export") and options.get("path"):
            outputs.append(Path(options["path"]))

        def launch():
            try:
                worker(args, name, model, action, world, device, fmt, **options)
            except GpuSafetyStop:
                raise
            except RuntimeError:
                log = graph.path(name).with_suffix(".log")
                text = log.read_text() if log.exists() else ""
                if device != "cuda" or not any(
                    token in text
                    for token in ("GPU_MEMORY_LIMIT", "CUDA out of memory")
                ):
                    raise
                for path in outputs:
                    remove(path)
                # CPU fallback preserves target world and records actual device.
                fallback = worker(
                    args,
                    name + "-fallback-cpu",
                    model,
                    action,
                    world,
                    "cpu",
                    fmt,
                    **options,
                )
                value = json.loads(fallback.read_text())
                value.update(
                    requested_device=device,
                    fallback_reason="85 GiB per-rank rule or CUDA OOM",
                    gpu_failure_log=str(log),
                )
                write_json(graph.path(name), value)

        metadata = dict(
            model=model,
            format=fmt,
            action=action,
            world_size=world,
            device=device,
            check=name,
            source_world_size=source_world
            if action in ("load", "native-load")
            else None,
        )
        return graph.add(name, launch, dependencies, outputs, metadata, cleanup)

    def compare(name, first, second, dependencies, cleanup=()):
        fmt = (
            "dcp"
            if name.startswith("dcp")
            else "safetensors"
            if name.startswith("safetensors")
            else "hf"
        )
        return graph.add(
            name,
            lambda: compare_files(args, model, name, first, second),
            dependencies,
            metadata=dict(
                model=model,
                format=fmt,
                check=name,
                action="file-compare",
                world_size=6,
                device="cpu",
            ),
            cleanup=cleanup,
        )

    def hfcheck(
        label, first, native, reference, dtype, limit, dependencies, cleanup=()
    ):
        return graph.add(
            label + "-I-E",
            lambda: hf_interop(
                args, model, label, first, native, reference, dtype, limit
            ),
            dependencies,
            [native] if native else [],
            dict(
                model=model,
                format="hf",
                check=label + "-I-E",
                action="interop",
                world_size=1,
                device="cpu",
            ),
            cleanup,
        )

    for fmt, sharded in (
        ("safetensors", False),
        ("dcp", False),
        ("hf", False),
        ("hf", True),
    ):
        if reduced and fmt == "hf" and not sharded:
            continue
        prefix = fmt + ("-sharded" if sharded else "")
        if formats and prefix not in formats:
            continue
        limit = max_shard_size(args, model) if sharded else None
        directory = (
            CKPT / args.stage / model / fmt / ("sharded" if sharded else "single")
        )
        first, duplicate, bridge, again = [
            directory / item for item in ("first", "duplicate", "bridge", "again")
        ]
        save = work(
            prefix + "-save6", "save", 6, "cuda", fmt, path=first, max_shard_size=limit
        )
        reference = graph.path(save)
        if not reduced:
            dup = work(
                prefix + "-duplicate6",
                "save",
                6,
                "cuda",
                fmt,
                path=duplicate,
                max_shard_size=limit,
                skip_reference=True,
            )
            compare(prefix + "-F-repeat6", first, duplicate, [save, dup], [duplicate])
        if fmt == "dcp":
            target = 5 if reduced or model == "24b" else 4
            work(
                prefix + "-I-native-load",
                "native-load",
                target,
                "cuda",
                fmt,
                [save],
                path=first,
                reference=reference,
            )
            native_save = work(
                prefix + "-I-native-save", "native-save", 6, "cuda", fmt, path=bridge
            )
            native_load = work(
                prefix + "-I-gdcp-load-native",
                "load",
                target,
                "cuda",
                fmt,
                [save, native_save],
                path=bridge,
                reference=reference,
            )
            graph.add(
                prefix + "-I-file-comparison",
                lambda first=first, bridge=bridge, prefix=prefix: native_comparison(
                    args, model, prefix + "-I-file-comparison", first, bridge
                ),
                [save, native_save, native_load],
                metadata=dict(
                    model=model,
                    format=fmt,
                    check=prefix + "-I-file-comparison",
                    device="cpu",
                    world_size=1,
                ),
                cleanup=[bridge],
            )
        if fmt == "hf":
            native = None if reduced else bridge
            interop = hfcheck(prefix, first, native, reference, "full", limit, [save])
            if not reduced:
                work(
                    prefix + "-I-native-hf-load",
                    "load",
                    4 if model != "24b" else 5,
                    "cuda",
                    fmt,
                    [save, interop],
                    [bridge],
                    path=bridge,
                    reference=reference,
                )
        target_list = (
            [(5, "cuda"), (8, "cpu")]
            if reduced and fmt != "hf"
            else [(5, "cuda")]
            if reduced
            else targets(model)
        )
        for world, device in target_list:
            label = prefix + "-6to%d-%s" % (world, device)
            roundtrip = not reduced or device == "cuda"
            loaded = work(
                label,
                "load",
                world,
                device,
                fmt,
                [save],
                path=first,
                reference=reference,
                output=bridge if roundtrip else None,
                max_shard_size=limit,
            )
            if roundtrip:
                back = work(
                    label + "-back6",
                    "load",
                    6,
                    "cuda",
                    fmt,
                    [save, loaded],
                    path=bridge,
                    reference=reference,
                    output=again,
                    max_shard_size=limit,
                    skip_step_check=True,
                    source_world=world,
                )
                compare(
                    label + "-F-roundtrip", first, again, [save, back], [bridge, again]
                )
        if not reduced:
            reverse = 5 if model == "24b" else 4
            reverse_save = work(
                prefix + "-reverse-save%d" % reverse,
                "save",
                reverse,
                "cuda",
                fmt,
                path=bridge,
                max_shard_size=limit,
            )
            work(
                prefix + "-reverse%d-load6" % reverse,
                "load",
                6,
                "cuda",
                fmt,
                [reverse_save],
                [bridge],
                path=bridge,
                reference=graph.path(reverse_save),
                source_world=reverse,
            )
        # Format cleanup happens after every branch, including failed dependencies.
        graph.add(
            prefix + "-complete",
            lambda: None,
            list(graph.checks)[list(graph.checks).index(save) :],
            metadata=dict(
                model=model,
                format=fmt,
                check=prefix + "-complete",
                kind="format-sentinel",
            ),
            cleanup=[directory],
        )
    reference_name = "weights-reference6"
    reference = graph.path(reference_name)

    def weights_reference():
        for candidate in (
            "safetensors-save6",
            "dcp-save6",
            "hf-save6",
            "hf-sharded-save6",
        ):
            path = graph.path(candidate)
            if check_done(path) and "reference" in json.loads(path.read_text()):
                value = json.loads(path.read_text())
                value.update(reference_origin=str(path), action="reference-reuse")
                write_json(reference, value)
                return
        worker(args, reference_name, model, "build", 6, "cuda")

    if not formats or any(item.startswith("weights-") for item in formats):
        graph.add(
            reference_name,
            weights_reference,
            metadata=dict(
                model=model,
                format="reference",
                check=reference_name,
                world_size=6,
                device="cuda",
            ),
        )
    variants = (
        (("fp32", False), ("bf16", True))
        if reduced
        else (("fp32", False), ("fp32", True), ("bf16", False), ("bf16", True))
    )
    for dtype, sharded in variants:
        label = "weights-" + dtype + ("-sharded" if sharded else "-single")
        if formats and label not in formats:
            continue
        base = CKPT / args.stage / model / label
        first, duplicate, native, bridge, again = [
            base / item for item in ("first", "duplicate", "native", "bridge", "again")
        ]
        limit = max_shard_size(args, model) if sharded else None
        exported = work(
            label + "-export6",
            "export",
            6,
            "cuda",
            "hf",
            path=first,
            dtype=dtype,
            max_shard_size=limit,
        )
        dependencies = [exported] + (
            [reference_name] if reference_name in graph.checks else []
        )
        if not reduced:
            dup = work(
                label + "-duplicate6",
                "export",
                6,
                "cuda",
                "hf",
                path=duplicate,
                dtype=dtype,
                max_shard_size=limit,
            )
            compare(label + "-F-repeat", first, duplicate, [exported, dup], [duplicate])
        interop = hfcheck(
            label,
            first,
            None if reduced else native,
            reference,
            dtype,
            limit,
            dependencies,
        )
        smaller = 1 if model == "4b" else 5
        weight_targets = (
            [(5, "cuda")]
            if reduced and dtype == "fp32"
            else [(8, "cpu")]
            if reduced
            else [(smaller, "cuda"), (8, "cpu")]
        )
        for world, device in weight_targets:
            loaded = work(
                label + "-load%d-%s" % (world, device),
                "load",
                world,
                device,
                "hf",
                dependencies,
                path=first,
                reference=reference,
                weights_only=True,
                dtype=dtype,
                output=bridge if not reduced else None,
                max_shard_size=limit,
            )
            if not reduced:
                back = work(
                    label + "-back6-from%d-%s" % (world, device),
                    "load",
                    6,
                    "cuda",
                    "hf",
                    [loaded],
                    path=bridge,
                    reference=reference,
                    weights_only=True,
                    dtype=dtype,
                    output=again,
                    max_shard_size=limit,
                    source_world=world,
                )
                compare(
                    label + "-F-roundtrip-%d-%s" % (world, device),
                    first,
                    again,
                    [exported, back],
                    [bridge, again],
                )
                work(
                    label + "-native-load%d-%s" % (world, device),
                    "load",
                    world,
                    device,
                    "hf",
                    [interop],
                    path=native,
                    reference=reference,
                    weights_only=True,
                    dtype=dtype,
                )
        graph.add(
            label + "-complete",
            lambda: None,
            list(graph.checks)[list(graph.checks).index(exported) :],
            metadata=dict(
                model=model,
                format="hf",
                check=label + "-complete",
                kind="format-sentinel",
            ),
            cleanup=[base],
        )
    return graph


def summary(args, finished=False, rc=None):
    cells = []
    for model in args.models:
        graph = build_plan(args, model) if args.stage != "A" else None
        paths = (
            {graph.path(name): node["metadata"] for name, node in graph.checks.items()}
            if graph
            else {}
        )
        for path in (LOG / args.stage / model).glob("*.json"):
            paths.setdefault(path, {})
        for path, metadata in paths.items():
            if metadata.get("kind") == "format-sentinel" or path.stem in (
                "model-failure",
                "safety-stop",
            ):
                continue
            value = json.loads(path.read_text()) if path.exists() else {}
            status = value.get("status", "pending")
            if finished and status == "pending":
                status = "skipped"
                value["reason"] = "pipeline stopped before this check"
            checks = {
                tag: value[tag]
                for tag in ("V", "F", "R", "E", "I_safetensors", "local_shapes")
                if tag in value
            }
            cells.append(
                dict(
                    model=model,
                    format=value.get("format", metadata.get("format")),
                    check=path.stem,
                    acceptance=checks,
                    source_world_size=value.get(
                        "source_world_size", metadata.get("source_world_size")
                    ),
                    world_size=value.get("world_size", metadata.get("world_size")),
                    device=value.get("device", metadata.get("device")),
                    status=status,
                    seconds=value.get("seconds", value.get("elapsed_seconds")),
                    bytes=value.get(
                        "io_payload_bytes",
                        value.get("logical_bytes", value.get("bytes")),
                    ),
                    peak_gpu_memory_bytes=value.get("peak_gpu_allocated_bytes"),
                    evidence_path=value.get("evidence_origin", str(path)),
                    result_path=str(path),
                    log_path=value.get(
                        "log_origin",
                        value.get("log_path", str(path.with_suffix(".log"))),
                    ),
                    reason=value.get("reason", value.get("error")),
                    fallback_reason=value.get("fallback_reason"),
                )
            )
    waits = []
    roots = [LOG]
    if getattr(args, "resume_root", None):
        roots.insert(0, args.resume_root)
    for root in roots:
        path = root / "gpu-sharing.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                value = json.loads(line)
                if (
                    value.get("event", "").startswith("wait_")
                    or value.get("event") == "foreign_during_job"
                ):
                    waits.append(dict(value, evidence_path=str(path)))
    counts = {
        status: sum(cell["status"] == status for cell in cells)
        for status in ("pass", "fail", "skipped", "pending")
    }
    record = dict(
        stage=args.stage,
        models=args.models,
        finished=finished,
        returncode=rc,
        updated_pst=pst_now(),
        counts=counts,
        cells=cells,
        gpu_sharing_waits=waits,
    )
    write_json(LOG / "summary.json", record)
    lines = [
        "# GDCP models pipeline",
        "",
        "Updated PST: " + pst_now(),
        "",
        "Status: "
        + ("finished" if finished else "running")
        + "; "
        + json.dumps(counts),
        "",
        "| Model | Format | Check | Worlds | Device | Status | Seconds | Bytes | Peak GPU bytes | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in cells:
        fields = [
            cell["model"],
            cell["format"],
            cell["check"],
            str(cell["source_world_size"]) + " → " + str(cell["world_size"]),
            cell["device"],
            cell["status"],
            cell["seconds"],
            cell["bytes"],
            cell["peak_gpu_memory_bytes"],
            cell["evidence_path"],
        ]
        lines.append(
            "| "
            + " | ".join(str(item) if item is not None else "—" for item in fields)
            + " |"
        )
        if cell["reason"]:
            lines.append(
                "\n"
                + str(cell["model"])
                + "/"
                + str(cell["check"])
                + ": "
                + str(cell["reason"]).replace("\n", " ")
                + "\n"
            )
    lines.extend(
        [
            "",
            "## GPU sharing waits",
            "",
            "All wait samples, process identities, memory, start/end times and disturbances are in summary.json.",
        ]
    )
    temporary = LOG / "SUMMARY.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(LOG / "SUMMARY.md")
    return record


def start_containers(args):
    owned = []
    if not args.manage_containers:
        return owned
    try:
        for mode in ("cpu", "gpu") if args.stage != "A" else ("cpu",):
            name = "iobench-gdcp-v2-" + mode
            probe = subprocess.run(
                ["docker", "inspect", "--format", "{{.Id}}", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            if probe.returncode == 0:
                raise RuntimeError(
                    "container already exists; refusing to claim ownership: " + name
                )
            command = [
                "bash",
                str(SOURCE / "scripts/gdcp_v2_h20_env.sh"),
                name,
                mode,
                str(LOG),
            ]
            environment = dict(
                __import__("os").environ,
                GDCP_GPU_EVIDENCE=str(LOG / "gpu-sharing.jsonl"),
            )
            try:
                subprocess.check_call(command, env=environment)
            except Exception:
                probe = subprocess.run(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        '{{ index .Config.Labels "gdcp.v2.owner" }} {{.Id}}',
                        name,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                if probe.returncode == 0 and probe.stdout.startswith(str(LOG) + " "):
                    owned.append(probe.stdout.split()[-1])
                raise
            identity = subprocess.check_output(
                ["docker", "inspect", "--format", "{{.Id}}", name],
                universal_newlines=True,
            ).strip()
            owned.append(identity)
            write_json(LOG / "containers.json", {"container_ids": owned})
    except Exception:
        for identity in owned:
            subprocess.run(["docker", "rm", "-f", identity], check=False)
        raise
    return owned


def main(argv=None):
    global LOG, GUARD, SOURCE_SHA, EXERCISED_SHA
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("A", "B", "C"), required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("4b", "14b", "24b"),
        default=["4b", "14b", "24b"],
    )
    parser.add_argument("--log-root", type=Path, default=LOG)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--manage-containers", action="store_true")
    parser.add_argument("--formats", nargs="+")
    parser.add_argument("--inject-failure")
    parser.add_argument("--max-new-checks", type=int)
    args = parser.parse_args(argv)
    if args.inject_failure and args.stage != "B":
        parser.error("--inject-failure is allowed only on stage B tiny configurations")
    LOG = args.log_root
    # A shared checkpoint tree admits one controller, even across different log roots.
    CKPT.mkdir(parents=True, exist_ok=True)
    lock = (CKPT / ".pipeline.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another GDCP controller owns the checkpoint tree")
    LOG.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest()
    SOURCE_SHA, EXERCISED_SHA = manifest["sha256"], exercised_digest(manifest)
    manifest_path = LOG / "source-manifest.json"
    if (
        manifest_path.exists()
        and exercised_digest(json.loads(manifest_path.read_text())) != EXERCISED_SHA
    ):
        raise RuntimeError(
            "exercised source changed: use a new --log-root to preserve evidence"
        )
    write_json(manifest_path, manifest)
    import_resume(args, manifest)
    GUARD = GpuGuard(
        LOG / "gpu-sharing.jsonl",
        owned_pids=lambda: container_pids("iobench-gdcp-v2-gpu"),
    )
    owned, rc = [], 0
    summary(args)
    try:
        owned = start_containers(args)
        for model in args.models:
            try:
                check_disk()
                model_stage(args, model)
            except (GpuSafetyStop, CheckLimit):
                raise
            except Exception as exc:
                write_json(
                    LOG / args.stage / model / "model-failure.json",
                    dict(
                        model=model,
                        status="fail",
                        error=repr(exc),
                        traceback=traceback.format_exc(),
                        finished_pst=pst_now(),
                    ),
                )
                rc = 1
            finally:
                remove(CKPT / args.stage / model)
                summary(args)
        result = summary(args)
        if result["counts"]["fail"] or result["counts"]["skipped"]:
            rc = 1
    except CheckLimit as exc:
        print("VALIDATION_LIMIT", str(exc), flush=True)
        rc = 0
    except BaseException as exc:
        rc = 2
        write_json(
            LOG / "pipeline-stop.json",
            dict(
                status="safety-stop" if isinstance(exc, GpuSafetyStop) else "fail",
                error=repr(exc),
                traceback=traceback.format_exc(),
                finished_pst=pst_now(),
            ),
        )
        print("PIPELINE_STOP", repr(exc), flush=True)
    finally:
        for model in args.models:
            remove(CKPT / args.stage / model)
        for identity in owned:
            result = subprocess.run(["docker", "rm", "-f", identity], check=False)
            if result.returncode:
                rc = 2
        summary(args, finished=True, rc=rc)
        with (LOG / "sentinels.log").open("a") as stream:
            stream.write("__GDCP_MODELS_ALL_DONE rc=%d\n" % rc)
        print("__GDCP_MODELS_ALL_DONE rc=%d" % rc, flush=True)
    lock.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
