"""Exercise the GPU sharing protocol with fake telemetry and a virtual clock."""

import json
import subprocess
import threading

import pytest

from scripts import gdcp_v2_safety as safety


class Clock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def observation(foreign=False, unknown=False):
    process = dict(
        physical_gpu=2,
        pid=641389,
        process_name="umm-asr-native",
        used_memory_MiB=384,
    )
    return dict(
        observed_pst=safety.pst_now(),
        memory_MiB={str(i): 384 if i == 2 and foreign else 0 for i in range(2, 8)},
        processes=[process] if foreign else [],
        foreign_processes=[process] if foreign else [],
        owned_processes=[],
        unattributed_memory_MiB={"2": 384} if unknown else {},
        blocked=foreign or unknown,
    )


def guard_for(tmp_path, sample, clock=None, **options):
    clock = clock or Clock()
    return safety.GpuGuard(
        tmp_path / "guard.jsonl",
        sample=sample,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **options,
    )


def evidence(tmp_path):
    return [
        json.loads(line) for line in (tmp_path / "guard.jsonl").read_text().splitlines()
    ]


def test_wait_records_foreign_process_and_clear_end_before_admission(tmp_path):
    samples = iter([observation(True), observation(True), observation()])
    clock = Clock()
    guard = guard_for(tmp_path, lambda: next(samples), clock)
    assert not guard.wait_until_clear("tiny-4b-save6")["blocked"]
    assert clock.sleeps == [60, 60]
    records = evidence(tmp_path)
    assert [r["event"] for r in records] == ["wait_start", "wait_sample", "wait_end"]
    assert records[0]["sample"]["foreign_processes"][0]["pid"] == 641389
    assert records[0]["sample"]["foreign_processes"][0]["used_memory_MiB"] == 384
    assert records[-1]["elapsed_seconds"] == 120
    assert records[-1]["wait_start_pst"] == records[0]["wait_start_pst"]
    assert records[-1]["wait_end_pst"]


def test_wait_allows_exactly_twelve_hours_but_stops_after_limit(tmp_path):
    clock = Clock()
    guard = guard_for(tmp_path, lambda: observation(True), clock)
    with pytest.raises(safety.GpuWaitTimeout, match="exceeded 43200 seconds"):
        guard.wait_until_clear("occupied")
    assert clock.now == 43260
    assert clock.sleeps == [60] * 721
    records = evidence(tmp_path)
    assert records[-2]["event"] == "wait_sample"
    assert records[-1]["event"] == "wait_timeout"
    assert records[-1]["elapsed_seconds"] == 43260
    assert records[-1]["wait_end_pst"]


def test_wait_limit_restarts_after_clear_for_next_job(tmp_path):
    clock = Clock()
    samples = iter([observation(True)] * 60 + [observation()])
    guard = guard_for(tmp_path, lambda: next(samples), clock)
    guard.wait_until_clear("first")
    guard.sample = lambda: observation()
    guard.wait_until_clear("second")
    assert evidence(tmp_path)[-1]["elapsed_seconds"] == 0


def test_clear_observed_after_wait_limit_does_not_authorize_a_launch(tmp_path):
    clock = Clock()
    samples = iter([observation(True)] * 61 + [observation()])
    guard = guard_for(tmp_path, lambda: next(samples), clock, max_wait=3600)
    with pytest.raises(safety.GpuWaitTimeout):
        guard.wait_until_clear("clear-too-late")
    assert clock.now == 3660
    assert evidence(tmp_path)[-1]["event"] == "wait_timeout"
    assert evidence(tmp_path)[-1]["sample"]["blocked"] is False


def test_unattributed_memory_waits_without_claiming_a_foreign_pid(tmp_path):
    samples = iter([observation(unknown=True), observation()])
    guard = guard_for(tmp_path, lambda: next(samples))
    guard.wait_until_clear("unknown-memory")
    assert evidence(tmp_path)[0]["sample"]["foreign_processes"] == []
    assert evidence(tmp_path)[0]["sample"]["unattributed_memory_MiB"] == {"2": 384}


def test_job_monitor_observes_foreign_process_without_interrupting_job(tmp_path):
    sampled = threading.Event()
    calls = []

    def sample():
        calls.append(True)
        foreign = len(calls) > 1
        if foreign:
            sampled.set()
        return observation(foreign)

    guard = guard_for(tmp_path, sample, monitor_interval=0.001)
    with guard.watch_job("running") as watcher:
        assert sampled.wait(2), "background GPU observation did not run"
        completed_on_its_own = True
    assert completed_on_its_own
    assert watcher.foreign_seen
    assert watcher.observations
    assert len(calls) >= 3  # Admission observation, background sample, final sample.
    assert evidence(tmp_path)[-1]["event"] == "job_watch_end"
    assert evidence(tmp_path)[-1]["foreign_seen"] is True


def test_final_job_sample_catches_a_late_arrival(tmp_path):
    samples = iter([observation(), observation(True)])
    guard = guard_for(tmp_path, lambda: next(samples), monitor_interval=60)
    with guard.watch_job("short-job") as watcher:
        pass
    assert watcher.foreign_seen


def test_owned_gpu_context_is_ignored_during_job_but_drained_before_next_launch(
    tmp_path,
):
    owned = observation()
    process = dict(physical_gpu=2, pid=123, process_name="python", used_memory_MiB=1024)
    owned.update(processes=[process], owned_processes=[process], memory_MiB={"2": 1024})
    clock = Clock()
    guard = guard_for(tmp_path, lambda: owned, clock, monitor_interval=60)
    with guard.watch_job("our-running-job") as watcher:
        pass
    assert not watcher.foreign_seen
    assert watcher.error is None
    samples = iter([owned, observation()])
    guard.sample = lambda: next(samples)
    clear = guard.wait_until_clear("next-job")
    assert clock.sleeps == [60]
    assert not clear["admission_blocked"]
    wait_start = next(r for r in evidence(tmp_path) if r["event"] == "wait_start")
    assert wait_start["sample"]["blocked"] is False
    assert wait_start["sample"]["admission_blocked"] is True
    assert "owned_process_drain" in wait_start["sample"]["admission_wait_reasons"]
    assert wait_start["sample"]["owned_processes"][0]["pid"] == 123


def test_owned_process_with_zero_reported_memory_still_drains(tmp_path):
    owned = observation()
    process = dict(physical_gpu=2, pid=123, process_name="python", used_memory_MiB=0)
    owned.update(processes=[process], owned_processes=[process])
    samples = iter([owned, observation()])
    clock = Clock()
    guard = guard_for(tmp_path, lambda: next(samples), clock)
    guard.wait_until_clear("next-job")
    assert clock.sleeps == [60]
    assert evidence(tmp_path)[0]["sample"]["admission_wait_reasons"] == [
        "owned_process_drain"
    ]


def test_sampler_failure_is_a_safety_stop_not_a_model_failure(tmp_path):
    def broken():
        raise RuntimeError("nvidia-smi failed")

    guard = guard_for(tmp_path, broken)
    with pytest.raises(safety.GpuSafetyStop, match="admission telemetry"):
        guard.wait_until_clear("prelaunch")
    with guard.watch_job("running") as watcher:
        pass
    assert isinstance(watcher.error, RuntimeError)
    assert any(r["event"] == "telemetry_error" for r in evidence(tmp_path))


def install_telemetry(monkeypatch, apps="", memories=None):
    memory_values = {} if memories is None else memories
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        if command[1] == "--query-gpu=index,uuid,memory.used":
            return "\n".join(
                f"{i}, gpu-{i}, {memory_values.get(i, 0)}" for i in range(8)
            )
        assert (
            command[1] == "--query-compute-apps=gpu_uuid,pid,process_name,used_memory"
        )
        return apps

    monkeypatch.setattr(safety.subprocess, "check_output", query)
    return calls


def test_sample_ignores_gpus_zero_and_one_and_explicitly_owned_host_pids(monkeypatch):
    calls = install_telemetry(
        monkeypatch,
        "gpu-0, 10, sglang, 80000\ngpu-1, 11, sglang, 80000\n"
        "gpu-2, 12, python, 1024\ngpu-7, 13, umm-asr-native, 384",
        {0: 80000, 1: 80000, 2: 1024, 7: 384},
    )

    def owned():
        assert (
            len(calls) == 2
        )  # Ownership sampled after app query to include new ranks.
        return {12}

    sample = safety.sample_gpu(owned)
    assert sample["blocked"]
    assert {p["pid"] for p in sample["processes"]} == {12, 13}
    assert [p["pid"] for p in sample["foreign_processes"]] == [13]
    assert [p["pid"] for p in sample["owned_processes"]] == [12]
    assert sample["unattributed_memory_MiB"] == {}
    assert set(sample["memory_MiB"]) == {str(i) for i in range(2, 8)}


def test_owned_process_name_does_not_authorize_an_unknown_pid(monkeypatch):
    install_telemetry(monkeypatch, "gpu-2, 1234, torchrun, 384", {2: 384})
    assert safety.sample_gpu({5678})["foreign_processes"][0]["pid"] == 1234


def test_idle_near_zero_memory_and_unattributed_memory(monkeypatch):
    install_telemetry(monkeypatch, memories={2: 10})
    assert not safety.check_gpu()["blocked"]
    install_telemetry(monkeypatch, memories={2: 11})
    assert safety.sample_gpu()["unattributed_memory_MiB"] == {"2": 11}
    with pytest.raises(safety.GpuSafetyStop, match="sharing wait"):
        safety.check_gpu()


@pytest.mark.parametrize("output", ["0, gpu-0, 0", "invalid"])
def test_missing_or_invalid_gpu_telemetry_stops(monkeypatch, output):
    monkeypatch.setattr(safety.subprocess, "check_output", lambda *a, **k: output)
    with pytest.raises(safety.GpuSafetyStop, match="invalid GPU telemetry"):
        safety.sample_gpu()


def test_nvidia_command_failure_stops(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(safety.subprocess, "check_output", fail)
    with pytest.raises(safety.GpuSafetyStop, match="GPU telemetry failed"):
        safety.sample_gpu()


def test_container_ownership_uses_only_exact_task_container_and_host_pids(monkeypatch):
    calls = []

    def top(command, **kwargs):
        calls.append(command)
        return "PID\n 111\n222\n"

    monkeypatch.setattr(safety.subprocess, "check_output", top)
    assert safety.container_pids() == {111, 222}
    assert calls == [["docker", "top", "iobench-gdcp-v2-gpu", "-eo", "pid"]]
    with pytest.raises(safety.GpuSafetyStop, match="refusing ownership"):
        safety.container_pids("unrelated-container")
    assert len(calls) == 1


def test_unavailable_owned_container_is_not_treated_as_foreign_or_idle(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "docker top")

    monkeypatch.setattr(safety.subprocess, "check_output", fail)
    with pytest.raises(safety.GpuSafetyStop, match="cannot establish owned host PIDs"):
        safety.container_pids()


def test_host_memory_admission_preserves_margin(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 2000000000 kB\nMemAvailable: 1000000000 kB\n")
    monkeypatch.setattr(safety, "MEMINFO", meminfo)
    record = safety.check_host_memory(700_000_000_000, reserve_bytes=128_000_000_000)
    assert record == {
        "available_bytes": 1_024_000_000_000,
        "required_bytes": 700_000_000_000,
        "reserve_bytes": 128_000_000_000,
    }
    with pytest.raises(RuntimeError, match="host memory admission failed"):
        safety.check_host_memory(900_000_000_000, reserve_bytes=128_000_000_000)


def test_host_memory_telemetry_error_rejects_launch(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 2000000000 kB\n")
    monkeypatch.setattr(safety, "MEMINFO", meminfo)
    with pytest.raises(RuntimeError, match="cannot establish host memory admission"):
        safety.check_host_memory(1)


def test_host_memory_default_reserves_phase_k_floor(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 1000000000 kB\n")
    monkeypatch.setattr(safety, "MEMINFO", meminfo)
    record = safety.check_host_memory(300_000_000_000)
    assert record["reserve_bytes"] == 650_000_000_000
    with pytest.raises(RuntimeError, match="host memory admission failed"):
        safety.check_host_memory(400_000_000_000)
