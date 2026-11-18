"""Admission guards for GDCP v2; host-only, no process control."""

import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path("/data00/iobench/gdcp-v2-checkpoints")
FLOOR = 350_000_000_000
CAP = 1_200_000_000_000
GPU_MAX_WAIT_SECONDS = 12 * 60 * 60
MEMINFO = Path("/proc/meminfo")
EVIDENCE = Path(
    os.environ.get("GDCP_GPU_EVIDENCE", "/data00/iobench/logs/gdcp-v2/gpu-guard.jsonl")
)


class GpuSafetyStop(RuntimeError):
    """Telemetry failure or exhausted sharing allowance; never a model failure."""


class GpuWaitTimeout(GpuSafetyStop):
    """A single uninterrupted wait exceeded its twelve-hour allowance."""


def pst_now():
    return datetime.now(timezone(timedelta(hours=-8))).isoformat()


def check_disk(additional=0):
    usage = shutil.disk_usage("/data00/iobench")
    held = (
        sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
        if ROOT.exists()
        else 0
    )
    record = dict(
        filesystem_path="/data00/iobench",
        free_bytes=usage.free,
        checkpoints_bytes=held,
        additional_bytes=additional,
    )
    if usage.free - additional < FLOOR or held + additional > CAP:
        raise RuntimeError(f"disk admission failed: {record}")
    return record


def check_host_memory(required_bytes, reserve_bytes=650_000_000_000):
    """Admit CPU jobs against available host RAM with an explicit safety reserve."""
    try:
        fields = {
            key: int(value.split()[0]) * 1024
            for line in MEMINFO.read_text().splitlines()
            for key, value in [line.split(":", 1)]
        }
        available = fields["MemAvailable"]
    except Exception as exc:
        raise RuntimeError(
            "cannot establish host memory admission: " + str(exc)
        ) from exc
    record = dict(
        available_bytes=available,
        required_bytes=required_bytes,
        reserve_bytes=reserve_bytes,
    )
    if (
        required_bytes < 0
        or reserve_bytes < 0
        or available < required_bytes + reserve_bytes
    ):
        raise RuntimeError(f"host memory admission failed: {record}")
    return record


def _query(field):
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-" + field, "--format=csv,noheader,nounits"],
            universal_newlines=True,
        )
        return list(csv.reader(output.splitlines(), skipinitialspace=True))
    except Exception as exc:
        raise GpuSafetyStop("GPU telemetry failed: " + str(exc)) from exc


def container_pids(container="iobench-gdcp-v2-gpu"):
    """Read host PIDs only from this task's exact, explicitly owned container."""
    if container != "iobench-gdcp-v2-gpu":
        raise GpuSafetyStop("refusing ownership inference for " + container)
    try:
        output = subprocess.check_output(
            ["docker", "top", container, "-eo", "pid"], universal_newlines=True
        )
        return {int(line.strip()) for line in output.splitlines()[1:] if line.strip()}
    except Exception as exc:
        raise GpuSafetyStop("cannot establish owned host PIDs: " + str(exc)) from exc


def sample_gpu(owned_pids=()):
    """Read physical GPUs 2--7; ownership is explicit, never based on process names."""
    try:
        rows = _query("gpu=index,uuid,memory.used")
        selected = {uuid: int(idx) for idx, uuid, _ in rows if 2 <= int(idx) <= 7}
        memory = {idx: int(used) for idx, _, used in rows if 2 <= int(idx) <= 7}
        if len(selected) != 6 or len(memory) != 6:
            raise ValueError("expected physical GPUs 2,3,4,5,6,7")
        apps = _query("compute-apps=gpu_uuid,pid,process_name,used_memory")
        # Resolve ownership after the app sample so newly spawned task ranks are known.
        owned = set(owned_pids() if callable(owned_pids) else owned_pids)
        processes = [_process(row, selected) for row in apps if row[0] in selected]
    except GpuSafetyStop:
        raise
    except Exception as exc:
        raise GpuSafetyStop("invalid GPU telemetry: " + str(exc)) from exc
    foreign = [p for p in processes if p["pid"] not in owned]
    ours = [p for p in processes if p["pid"] in owned]
    active = {str(p["physical_gpu"]) for p in processes}
    unknown = {idx: n for idx, n in memory.items() if n > 10 and idx not in active}
    return dict(
        observed_pst=pst_now(),
        memory_MiB=memory,
        processes=processes,
        foreign_processes=foreign,
        owned_processes=ours,
        unattributed_memory_MiB=unknown,
        blocked=bool(foreign or unknown),
    )


def _process(row, selected):
    uuid, pid, name, used = row
    return dict(
        physical_gpu=selected[uuid],
        gpu_uuid=uuid,
        pid=int(pid),
        process_name=name,
        used_memory_MiB=int(used) if used.isdigit() else None,
        used_memory_raw=used,
    )


def _admission_sample(sample):
    """Drain our old contexts too; only job monitoring disregards owned PIDs."""
    reasons = []
    if sample.get("foreign_processes"):
        reasons.append("foreign_processes")
    if sample.get("owned_processes"):
        reasons.append("owned_process_drain")
    if sample.get("unattributed_memory_MiB"):
        reasons.append("unattributed_memory")
    if any(n > 10 for n in sample.get("memory_MiB", {}).values()):
        reasons.append("memory_above_idle_threshold")
    blocked = bool(sample.get("blocked") or sample.get("processes") or reasons)
    if blocked and not reasons:
        reasons.append("processes_or_blocked_telemetry")
    return dict(sample, admission_blocked=blocked, admission_wait_reasons=reasons)


def check_gpu():
    """Immediate assertion retained for diagnostics; launches use GpuGuard instead."""
    record = sample_gpu()
    if record["blocked"]:
        raise GpuSafetyStop("GPUs require a sharing wait: " + json.dumps(record))
    return record


class GpuGuard:
    """Wait before launches and observe running jobs, without controlling processes."""

    def __init__(
        self,
        evidence_path=EVIDENCE,
        owned_pids=(),
        sample=None,
        sleep=time.sleep,
        monotonic=time.monotonic,
        wait_interval=60,
        monitor_interval=5,
        max_wait=GPU_MAX_WAIT_SECONDS,
    ):
        self.evidence_path = Path(evidence_path)
        self.owned_pids = owned_pids
        self.sample = sample or (lambda: sample_gpu(self.owned_pids))
        self.sleep = sleep
        self.monotonic = monotonic
        self.wait_interval = wait_interval
        self.monitor_interval = monitor_interval
        self.max_wait = max_wait
        self._lock = threading.Lock()

    def emit(self, event, **fields):
        record = dict(event=event, observed_pst=pst_now(), **fields)
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.evidence_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        print("GPU_GUARD " + json.dumps(record, sort_keys=True), flush=True)
        return record

    def wait_until_clear(self, job):
        start = self.monotonic()
        started_pst = pst_now()
        waiting = False
        while True:
            sample = self._wait_sample(job, started_pst, waiting)
            elapsed = self.monotonic() - start
            if waiting and elapsed > self.max_wait:
                self.emit(
                    "wait_timeout",
                    job=job,
                    wait_start_pst=started_pst,
                    wait_end_pst=pst_now(),
                    elapsed_seconds=elapsed,
                    sample=sample,
                )
                raise GpuWaitTimeout(
                    f"GPU sharing wait exceeded {self.max_wait:g} seconds: " + job
                )
            if not sample["admission_blocked"]:
                self.emit(
                    "wait_end" if waiting else "gpu_admission",
                    job=job,
                    wait_start_pst=started_pst if waiting else None,
                    wait_end_pst=pst_now() if waiting else None,
                    elapsed_seconds=elapsed,
                    sample=sample,
                )
                return sample
            self.emit(
                "wait_sample" if waiting else "wait_start",
                job=job,
                wait_start_pst=started_pst,
                elapsed_seconds=elapsed,
                sample=sample,
            )
            waiting = True
            self.sleep(self.wait_interval)

    def _wait_sample(self, job, started_pst, waiting):
        try:
            return _admission_sample(self.sample())
        except Exception as exc:
            self.emit(
                "telemetry_error",
                job=job,
                wait_start_pst=started_pst if waiting else None,
                wait_end_pst=pst_now() if waiting else None,
                error=str(exc),
            )
            raise GpuSafetyStop("GPU admission telemetry failed: " + str(exc)) from exc

    def watch_job(self, job):
        return GpuJobWatch(self, job)


class GpuJobWatch:
    """A background observer; exit waits for its read-only sampler to finish."""

    def __init__(self, guard, job):
        self.guard = guard
        self.job = job
        self.foreign_seen = False
        self.observations = []
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.guard.emit("job_watch_start", job=self.job)
        self._observe()
        self._thread.start()
        return self

    def _observe(self):
        try:
            sample = self.guard.sample()
            if sample["foreign_processes"]:
                self.foreign_seen = True
                self.observations.append(sample)
                self.guard.emit("foreign_during_job", job=self.job, sample=sample)
        except Exception as exc:
            self.error = exc
            self.guard.emit("telemetry_error", job=self.job, error=str(exc))

    def _run(self):
        while not self._stop.wait(self.guard.monitor_interval):
            self._observe()

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join()
        self._observe()
        self.guard.emit(
            "job_watch_end",
            job=self.job,
            foreign_seen=self.foreign_seen,
            observations=self.observations,
            error=str(self.error) if self.error else None,
        )
        return False


if __name__ == "__main__":
    result: dict = {
        "observed_pst": pst_now(),
        "disk": check_disk(int(sys.argv[2]) if len(sys.argv) > 2 else 0),
    }
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        result["gpu"] = GpuGuard().wait_until_clear(job="cli-gpu-admission")
    print(json.dumps(result, indent=2))
