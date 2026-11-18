"""GPU launch admission and disturbed-attempt handling without a real GPU job."""

import json
from contextlib import contextmanager
from itertools import repeat
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import gdcp_v2_pipeline as pipeline


class FakeGuard:
    def __init__(self, events, observations, initial=None):
        self.events = events
        self.observations = iter(observations)
        self.initial = iter(initial) if initial is not None else repeat((False, None))

    def wait_until_clear(self, *, job):
        self.events.append(("clear", job))

    @contextmanager
    def watch_job(self, *, job):
        self.events.append(("watch-start", job))
        foreign, error = next(self.observations)
        initial_foreign, initial_error = next(self.initial)
        observation = SimpleNamespace(
            foreign_seen=initial_foreign,
            error=initial_error,
            observations=[],
        )
        try:
            yield observation
        finally:
            observation.foreign_seen = initial_foreign or foreign
            observation.error = initial_error or error
            observation.observations = (
                [{"pid": 42, "process_name": "foreign", "memory_MiB": 384}]
                if observation.foreign_seen
                else []
            )
            self.events.append(("watch-end", job))


def write_fake_artifacts(command, index):
    for flag in ("--output", "--path"):
        if flag in command:
            artifact = Path(command[command.index(flag) + 1])
            artifact.mkdir(parents=True, exist_ok=True)
            (artifact / "payload").write_text(str(index + 1))


@pytest.fixture
def harness(monkeypatch, tmp_path):
    events, launches = [], []
    outcomes = []
    monkeypatch.setattr(pipeline, "LOG", tmp_path / "logs")
    monkeypatch.setattr(pipeline, "SOURCE", tmp_path / "source")
    monkeypatch.setattr(pipeline, "check_host_memory", lambda required: {})

    class FakeProcess:
        def __init__(self, command, **kwargs):
            assert kwargs["stdout"] == pipeline.subprocess.PIPE
            assert kwargs["stderr"] == pipeline.subprocess.STDOUT
            assert len(launches) < len(outcomes), "unexpected extra launch"
            self.index = len(launches)
            self.command = command
            self.returncode = outcomes[self.index]
            self.stdout = [f"attempt {self.index + 1}\n"]
            launches.append(command)
            events.append(("launch", self.index + 1))

        def wait(self):
            result = Path(self.command[self.command.index("--result") + 1])
            result.write_text(
                json.dumps(
                    {
                        "status": "pass" if self.returncode == 0 else "fail",
                        "attempt": self.index + 1,
                    }
                )
            )
            write_fake_artifacts(self.command, self.index)
            events.append(("finished", self.index + 1))
            return self.returncode

    monkeypatch.setattr(pipeline.subprocess, "Popen", FakeProcess)
    return SimpleNamespace(
        events=events,
        launches=launches,
        outcomes=outcomes,
        args=SimpleNamespace(stage="B"),
        output=tmp_path / "checkpoint",
    )


def install_guard(monkeypatch, harness, observations, *, initial=None):
    guard = FakeGuard(harness.events, observations, initial)
    monkeypatch.setattr(pipeline, "GUARD", guard)
    return guard


def run_worker(harness, *, name="check", device="cuda", action="load"):
    result = pipeline.worker(
        harness.args,
        name,
        "4b",
        action,
        6,
        device,
        output=harness.output,
    )
    assert isinstance(result, Path)
    return result


def test_each_gpu_launch_waits_and_is_monitored(monkeypatch, harness):
    install_guard(monkeypatch, harness, [(False, None), (False, None)])
    harness.outcomes.extend([0, 0])
    run_worker(harness, name="first")
    run_worker(harness, name="second")
    assert [event[0] for event in harness.events] == [
        "clear",
        "watch-start",
        "launch",
        "finished",
        "watch-end",
        "clear",
        "watch-start",
        "launch",
        "finished",
        "watch-end",
    ]
    assert all("iobench-gdcp-v2-gpu" in command for command in harness.launches)


@pytest.mark.parametrize("first_returncode", [0, 1])
def test_disturbed_attempt_is_archived_and_rerun_once(
    monkeypatch, harness, first_returncode
):
    install_guard(monkeypatch, harness, [(True, None), (False, None)])
    harness.outcomes.extend([first_returncode, 0])
    result = run_worker(harness)
    assert json.loads(result.read_text())["status"] == "pass"
    assert json.loads(result.read_text())["attempt"] == 2
    archive = result.parent / "check-disturbed-attempt1"
    prior = json.loads((archive / result.name).read_text())
    assert prior["attempt"] == 1
    assert prior["status"] == ("pass" if first_returncode == 0 else "fail")
    assert (archive / "check.log").read_text() == "attempt 1\n"
    disturbance = json.loads((archive / "disturbance.json").read_text())
    assert disturbance["status"] == "disturbed"
    assert disturbance["returncode"] == first_returncode
    assert disturbance["observations"][0]["pid"] == 42
    assert result.with_suffix(".log").read_text() == "attempt 2\n"
    assert (harness.output / "payload").read_text() == "2"
    assert [event[0] for event in harness.events].count("clear") == 3
    assert len(harness.launches) == 2


def test_second_disturbance_is_skipped_without_a_third_launch(monkeypatch, harness):
    install_guard(monkeypatch, harness, [(True, None), (True, None)])
    harness.outcomes.extend([0, 1])
    with pytest.raises(pipeline.CheckSkipped, match="disturb"):
        run_worker(harness)
    root = pipeline.LOG / "B" / "4b"
    assert len(harness.launches) == 2
    for attempt in (1, 2):
        archive = root / f"check-disturbed-attempt{attempt}"
        assert (
            json.loads((archive / "disturbance.json").read_text())["status"]
            == "disturbed"
        )
    assert not (root / "check-disturbed-attempt3").exists()


def test_clean_failure_after_one_disturbance_is_not_rerun(monkeypatch, harness):
    install_guard(monkeypatch, harness, [(True, None), (False, None)])
    harness.outcomes.extend([0, 1])
    with pytest.raises(RuntimeError, match="check failed"):
        run_worker(harness)
    root = pipeline.LOG / "B" / "4b"
    result = json.loads((root / "check.json").read_text())
    assert result["status"] == "fail"
    assert result["attempt"] == 2
    assert len(harness.launches) == 2
    assert (root / "check-disturbed-attempt1" / "disturbance.json").is_file()
    assert not (root / "check-disturbed-attempt2").exists()


def test_clean_failure_is_not_rerun(monkeypatch, harness):
    install_guard(monkeypatch, harness, [(False, None)])
    harness.outcomes.append(1)
    with pytest.raises(RuntimeError, match="check failed"):
        run_worker(harness)
    result = json.loads((pipeline.LOG / "B" / "4b" / "check.json").read_text())
    assert result["status"] == "fail"
    assert result["attempt"] == 1
    assert result["returncode"] == 1
    assert len(harness.launches) == 1


def test_monitor_error_stops_after_current_job_finishes(monkeypatch, harness):
    install_guard(monkeypatch, harness, [(False, RuntimeError("query unavailable"))])
    harness.outcomes.append(0)
    with pytest.raises(pipeline.GpuSafetyStop, match="monitoring failed"):
        run_worker(harness)
    assert [event[0] for event in harness.events] == [
        "clear",
        "watch-start",
        "launch",
        "finished",
        "watch-end",
    ]
    assert len(harness.launches) == 1


def test_admission_stop_prevents_gpu_launch(monkeypatch, harness):
    guard = install_guard(monkeypatch, harness, [])

    def stop(*, job):
        raise pipeline.GpuSafetyStop("foreign occupancy timeout")

    monkeypatch.setattr(guard, "wait_until_clear", stop)
    with pytest.raises(pipeline.GpuSafetyStop, match="timeout"):
        run_worker(harness)
    assert harness.launches == []


def test_initial_foreign_observation_waits_without_consuming_job_rerun(
    monkeypatch, harness
):
    install_guard(
        monkeypatch,
        harness,
        [(False, None), (True, None), (False, None)],
        initial=[(True, None), (False, None), (False, None)],
    )
    harness.outcomes.extend([0, 0])
    result = run_worker(harness)
    assert json.loads(result.read_text())["status"] == "pass"
    event_names = [event[0] for event in harness.events]
    assert event_names[:3] == ["clear", "watch-start", "watch-end"]
    assert event_names.count("launch") == 2
    assert event_names.count("watch-start") == 3
    assert (result.parent / "check-disturbed-attempt1").is_dir()
    assert not (result.parent / "check-disturbed-attempt2").exists()


def test_initial_monitor_error_prevents_gpu_launch(monkeypatch, harness):
    install_guard(
        monkeypatch,
        harness,
        [(False, None)],
        initial=[(False, RuntimeError("initial query unavailable"))],
    )
    with pytest.raises(pipeline.GpuSafetyStop, match="initial query unavailable"):
        run_worker(harness)
    assert harness.launches == []


def test_cpu_worker_does_not_query_gpus(monkeypatch, harness):
    monkeypatch.setattr(pipeline, "GUARD", None)
    harness.outcomes.append(0)
    result = run_worker(harness, device="cpu")
    assert json.loads(result.read_text())["status"] == "pass"
    assert [event[0] for event in harness.events] == ["launch", "finished"]
    assert "iobench-gdcp-v2-cpu" in harness.launches[0]


def test_existing_valid_result_does_not_launch_again(monkeypatch, harness):
    install_guard(monkeypatch, harness, [(False, None)])
    harness.outcomes.append(0)
    first = run_worker(harness)
    assert run_worker(harness) == first
    assert len(harness.launches) == 1


def test_safety_stop_is_not_treated_as_gpu_memory_fallback(monkeypatch, tmp_path):
    def stop(*args, **kwargs):
        raise pipeline.GpuSafetyStop("foreign occupancy timeout")

    monkeypatch.setattr(pipeline, "worker", stop)
    monkeypatch.setattr(pipeline, "LOG", tmp_path / "logs")
    with pytest.raises(pipeline.GpuSafetyStop, match="timeout"):
        pipeline.target_roundtrip(
            SimpleNamespace(stage="B"),
            "4b",
            "safetensors",
            "safetensors",
            tmp_path / "first",
            tmp_path / "bridge",
            tmp_path / "again",
            tmp_path / "reference.json",
            4,
            "cuda",
            None,
        )


@pytest.mark.parametrize("old_sha", [None, "prior-source"])
def test_resume_rejects_pass_from_another_source(monkeypatch, tmp_path, old_sha):
    monkeypatch.setattr(pipeline, "SOURCE_SHA", "final-source")
    result = tmp_path / "check.json"
    result.write_text(json.dumps({"status": "pass", "source_manifest_sha256": old_sha}))
    assert not pipeline.check_done(result)
    pipeline.write_json(result, {"status": "pass"})
    assert pipeline.check_done(result)
    assert json.loads(result.read_text())["source_manifest_sha256"] == "final-source"


def test_source_manifest_ignores_runtime_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "SOURCE", tmp_path)
    source = tmp_path / "module.py"
    source.write_text("initial source\n")
    before = pipeline.source_manifest()
    for name in (".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"):
        cache = tmp_path / name
        cache.mkdir()
        (cache / "generated").write_text("runtime cache\n")
    assert pipeline.source_manifest() == before
    assert set(before["files"]) == {"module.py"}
    source.write_text("modified source\n")
    assert pipeline.source_manifest()["sha256"] != before["sha256"]


def write_check_artifacts(action, options):
    for key in ("path", "output"):
        if options.get(key) and (
            key == "output" or action in ("save", "export", "native-save")
        ):
            root = Path(options[key])
            root.mkdir(parents=True, exist_ok=True)
            (root / "payload").write_bytes(b"same checkpoint bytes")


def fake_check_worker(calls):
    def run(args, name, model, action, world, device, fmt="safetensors", **options):
        calls.append((name, action, world, options))
        write_check_artifacts(action, options)
        return Path("reference.json")

    return run


def test_only_duplicate_full_save_skips_unused_reference(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pipeline, "LOG", tmp_path / "logs")
    monkeypatch.setattr(pipeline, "CKPT", tmp_path / "checkpoints")
    monkeypatch.setattr(pipeline, "worker", fake_check_worker(calls))
    monkeypatch.setattr(pipeline, "targets", lambda model: [])
    pipeline.full_format(SimpleNamespace(stage="C"), "4b", "safetensors")
    saves = [row for row in calls if row[1] == "save"]
    assert [row[0] for row in saves] == [
        "safetensors-save6",
        "safetensors-duplicate6",
        "safetensors-reverse-save4",
    ]
    assert [row[3].get("skip_reference", False) for row in saves] == [
        False,
        True,
        False,
    ]
    reverse_load = calls[-1]
    assert reverse_load[0] == "safetensors-reverse4-load6"
    assert not reverse_load[3].get("skip_step_check", False)


def test_target_keeps_continuation_check_and_return_keeps_values_files(
    monkeypatch, tmp_path
):
    calls = []
    first, bridge, again = [tmp_path / name for name in ("first", "bridge", "again")]
    first.mkdir()
    (first / "payload").write_bytes(b"same checkpoint bytes")
    monkeypatch.setattr(pipeline, "LOG", tmp_path / "logs")
    monkeypatch.setattr(pipeline, "worker", fake_check_worker(calls))
    pipeline.target_roundtrip(
        SimpleNamespace(stage="C"),
        "4b",
        "safetensors",
        "safetensors",
        first,
        bridge,
        again,
        Path("reference.json"),
        4,
        "cuda",
        None,
    )
    assert len(calls) == 2
    target, returned = calls
    assert target[2] == 4 and not target[3].get("skip_step_check", False)
    assert target[3]["reference"] == Path("reference.json")
    assert returned[2] == 6 and returned[3]["skip_step_check"] is True
    assert returned[3]["reference"] == Path("reference.json")
    assert returned[3]["output"] == again
    comparison = pipeline.LOG / "C/4b/safetensors-6to4-cuda-F-roundtrip.json"
    assert json.loads(comparison.read_text())["F"] == "pass"


def graph_harness(monkeypatch, tmp_path, model="4b"):
    monkeypatch.setattr(pipeline, "LOG", tmp_path / "logs")
    monkeypatch.setattr(pipeline, "CKPT", tmp_path / "checkpoints")
    args = SimpleNamespace(stage="C", models=[model])
    return args, pipeline.CheckGraph(args, model)


def check_record(name, model="4b"):
    return pipeline.LOG / "C" / model / (name + ".json")


def passing_runner(name, calls):
    def run():
        calls.append(name)
        record = check_record(name)
        pipeline.write_json(record, {"status": "pass"})
        return record

    return run


def test_failed_check_cleans_outputs_skips_dependents_and_continues(
    monkeypatch, tmp_path
):
    _, graph = graph_harness(monkeypatch, tmp_path)
    calls = []
    broken = pipeline.CKPT / "C/4b/broken"

    def fail():
        calls.append("save")
        broken.mkdir(parents=True)
        (broken / "partial").write_bytes(b"incomplete checkpoint")
        raise RuntimeError("simulated save failure")

    graph.add("save", fail, outputs=(broken,), cleanup=(broken,))
    graph.add("load", passing_runner("load", calls), dependencies=("save",))
    graph.add("unrelated", passing_runner("unrelated", calls))
    graph.run()

    assert calls == ["save", "unrelated"]
    assert not broken.exists()
    failed = json.loads(check_record("save").read_text())
    skipped = json.loads(check_record("load").read_text())
    assert failed["status"] == "fail"
    assert "simulated save failure" in failed["error"]
    assert failed.get("log_path") or failed.get("log")
    assert skipped["status"] == "skipped"
    assert "save" in skipped.get("reason", skipped.get("error", ""))
    assert json.loads(check_record("unrelated").read_text())["status"] == "pass"


def test_failed_independent_check_does_not_block_later_format(monkeypatch, tmp_path):
    _, graph = graph_harness(monkeypatch, tmp_path)
    calls = []

    def fail():
        calls.append("safetensors-load")
        raise RuntimeError("value mismatch")

    graph.add("safetensors-load", fail)
    graph.add(
        "safetensors-return",
        passing_runner("safetensors-return", calls),
        dependencies=("safetensors-load",),
    )
    graph.add("dcp-save", passing_runner("dcp-save", calls))
    graph.run()
    assert calls == ["safetensors-load", "dcp-save"]
    assert (
        json.loads(check_record("safetensors-return").read_text())["status"]
        == "skipped"
    )
    assert json.loads(check_record("dcp-save").read_text())["status"] == "pass"


def test_safety_stop_propagates_instead_of_becoming_model_failure(
    monkeypatch, tmp_path
):
    _, graph = graph_harness(monkeypatch, tmp_path)
    calls = []

    def stop():
        raise pipeline.GpuSafetyStop("disk or telemetry unavailable")

    graph.add("save", stop)
    graph.add("later", passing_runner("later", calls))
    with pytest.raises(pipeline.GpuSafetyStop):
        graph.run()
    assert calls == []


def test_completed_check_is_not_relaunched_when_its_artifacts_are_not_needed(
    monkeypatch, tmp_path
):
    _, graph = graph_harness(monkeypatch, tmp_path)
    calls = []
    missing = pipeline.CKPT / "C/4b/removed"
    pipeline.write_json(check_record("save"), {"status": "pass", "evidence": "prior"})
    graph.add("save", passing_runner("save", calls), outputs=(missing,))
    graph.run()
    assert calls == []
    assert json.loads(check_record("save").read_text())["evidence"] == "prior"


@pytest.mark.parametrize("model", ["14b", "24b"])
def test_large_model_plan_uses_reduced_required_matrix(monkeypatch, tmp_path, model):
    args, _ = graph_harness(monkeypatch, tmp_path, model)
    graph = pipeline.build_plan(args, model)
    names = set(graph.checks)
    assert "safetensors-save6" in names
    assert "safetensors-6to5-cuda" in names
    assert "safetensors-6to5-cuda-F-roundtrip" in names
    assert "safetensors-6to8-cpu" in names
    assert "dcp-I-native-load" in names
    assert "dcp-I-native-save" in names
    assert "dcp-I-gdcp-load-native" in names
    assert "dcp-6to5-cuda-F-roundtrip" in names
    assert "dcp-6to8-cpu" in names
    assert "hf-sharded-save6" in names
    assert "hf-sharded-6to5-cuda-F-roundtrip" in names
    assert "weights-fp32-single-export6" in names
    assert "weights-bf16-sharded-export6" in names
    assert not any("duplicate" in name or "reverse" in name for name in names)
    assert not any(
        name.startswith("hf-") and not name.startswith("hf-sharded-") for name in names
    )
    assert not any(
        name.startswith("weights-fp32-sharded-")
        or name.startswith("weights-bf16-single-")
        for name in names
    )
    assert not any(
        "6to1-" in name or "6to4-" in name or "6to3-" in name for name in names
    )


def test_4b_plan_keeps_original_formats_and_targets(monkeypatch, tmp_path):
    args, _ = graph_harness(monkeypatch, tmp_path)
    names = set(pipeline.build_plan(args, "4b").checks)
    for prefix in ("safetensors", "dcp", "hf", "hf-sharded"):
        assert prefix + "-F-repeat6" in names
        for world, device in ((1, "cuda"), (4, "cuda"), (5, "cuda"), (8, "cpu")):
            assert prefix + f"-6to{world}-{device}" in names
            assert prefix + f"-6to{world}-{device}-F-roundtrip" in names
        assert prefix + "-reverse4-load6" in names
    for dtype in ("fp32", "bf16"):
        for layout in ("single", "sharded"):
            assert f"weights-{dtype}-{layout}-export6" in names


def test_resume_fingerprint_tracks_worker_and_fixture_but_ignores_controller():
    files = {
        "gdcp/core.py": "core-v1",
        "gdtensor_conn/safetensors.py": "connector-v1",
        "scripts/gdcp_v2_worker.py": "worker-v1",
        "scripts/gdcp_v2_hf_interop.py": "interop-v1",
        "scripts/gdcp_v2_pipeline.py": "controller-v1",
        "scripts/gdcp_v2_safety.py": "guard-v1",
        "tests/support/gdcp_v2_models.py": "fixture-v1",
        "tests/fixtures/gdcp-v2/configs/4b/config.json": "config-v1",
        "tests/gdcp/test_v2_pipeline.py": "tests-v1",
    }
    original = pipeline.exercised_files({"files": files})
    for path in (
        "scripts/gdcp_v2_pipeline.py",
        "scripts/gdcp_v2_safety.py",
        "tests/gdcp/test_v2_pipeline.py",
    ):
        changed = dict(files, **{path: "changed"})
        assert pipeline.exercised_files({"files": changed}) == original
    for path in (
        "gdcp/core.py",
        "scripts/gdcp_v2_worker.py",
        "scripts/gdcp_v2_hf_interop.py",
        "tests/support/gdcp_v2_models.py",
        "tests/fixtures/gdcp-v2/configs/4b/config.json",
    ):
        changed = dict(files, **{path: "changed"})
        assert pipeline.exercised_files({"files": changed}) != original


def test_pass_with_matching_exercised_code_survives_controller_source_change(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(pipeline, "SOURCE_SHA", "new-whole-tree")
    monkeypatch.setattr(pipeline, "EXERCISED_SHA", "same-worker-code")
    result = tmp_path / "passed.json"
    result.write_text(
        json.dumps(
            {
                "status": "pass",
                "source_manifest_sha256": "old-whole-tree",
                "exercised_code_sha256": "same-worker-code",
            }
        )
    )
    assert pipeline.check_done(result)
    data = json.loads(result.read_text())
    data["exercised_code_sha256"] = "old-worker-code"
    result.write_text(json.dumps(data))
    assert not pipeline.check_done(result)


def test_format_completion_does_not_rebuild_consumed_checkpoints(monkeypatch, tmp_path):
    _, graph = graph_harness(monkeypatch, tmp_path)
    calls = []
    checkpoint = pipeline.CKPT / "C/4b/full/first"

    def save():
        checkpoint.mkdir(parents=True)
        return passing_runner("save", calls)()

    graph.add("save", save, outputs=(checkpoint,))
    graph.add(
        "consume",
        passing_runner("consume", calls),
        dependencies=("save",),
        cleanup=(checkpoint,),
    )
    graph.add(
        "complete",
        passing_runner("complete", calls),
        dependencies=("save", "consume"),
        metadata={"kind": "format-sentinel"},
    )
    graph.run()
    assert calls == ["save", "consume", "complete"]
    assert not checkpoint.exists()


def test_skipped_format_sentinel_still_cleans_format_checkpoint(monkeypatch, tmp_path):
    _, graph = graph_harness(monkeypatch, tmp_path)
    checkpoint = pipeline.CKPT / "C/4b/full/first"
    checkpoint.mkdir(parents=True)
    (checkpoint / "payload").write_bytes(b"retained prerequisite")

    def fail():
        raise RuntimeError("check mismatch")

    graph.add("bad-load", fail)
    graph.add(
        "complete",
        lambda: None,
        dependencies=("bad-load",),
        metadata={"kind": "format-sentinel"},
        cleanup=(checkpoint.parent,),
    )
    graph.run()
    assert json.loads(check_record("complete").read_text())["status"] == "skipped"
    assert not checkpoint.parent.exists()


def test_memory_fallback_records_actual_cpu_device(monkeypatch, tmp_path):
    args, _ = graph_harness(monkeypatch, tmp_path, "24b")
    args.formats = ["safetensors"]
    launches = []

    def worker(args, name, model, action, world, device, fmt, **options):
        result = check_record(name, model)
        result.parent.mkdir(parents=True, exist_ok=True)
        launches.append((name, device))
        if name == "safetensors-6to5-cuda" and device == "cuda":
            result.with_suffix(".log").write_text("GPU_MEMORY_LIMIT: 86 GiB")
            raise RuntimeError("check failed")
        write_check_artifacts(action, options)
        pipeline.write_json(
            result, {"status": "pass", "device": device, "world_size": world}
        )
        return result

    monkeypatch.setattr(pipeline, "worker", worker)
    graph = pipeline.build_plan(args, "24b")
    assert graph.ensure("safetensors-6to5-cuda")
    result = json.loads(check_record("safetensors-6to5-cuda", "24b").read_text())
    assert result["status"] == "pass"
    assert result["device"] == "cpu"
    assert result["requested_device"] == "cuda"
    assert "85 GiB" in result["fallback_reason"]
    assert launches == [
        ("safetensors-save6", "cuda"),
        ("safetensors-6to5-cuda", "cuda"),
        ("safetensors-6to5-cuda-fallback-cpu", "cpu"),
    ]


def test_weights_checks_reuse_reference_json_without_rebuilding_full_checkpoint(
    monkeypatch, tmp_path
):
    args, _ = graph_harness(monkeypatch, tmp_path, "14b")
    pipeline.write_json(
        check_record("safetensors-save6", "14b"),
        {"status": "pass", "reference": {"saved": "hash"}},
    )
    calls = []

    def worker(args, name, model, action, world, device, fmt, **options):
        calls.append(name)
        write_check_artifacts(action, options)
        result = check_record(name, model)
        pipeline.write_json(result, {"status": "pass"})
        return result

    def interop(args, model, label, source, native, reference, dtype, limit):
        assert json.loads(reference.read_text())["reference"] == {"saved": "hash"}
        pipeline.write_json(check_record(label + "-I-E", model), {"status": "pass"})

    monkeypatch.setattr(pipeline, "worker", worker)
    monkeypatch.setattr(pipeline, "hf_interop", interop)
    graph = pipeline.build_plan(args, "14b")
    assert graph.ensure("weights-fp32-single-I-E")
    assert calls == ["weights-fp32-single-export6"]


def test_summary_preserves_metrics_evidence_and_all_gpu_waits(monkeypatch, tmp_path):
    args, graph = graph_harness(monkeypatch, tmp_path)
    args.resume_root = tmp_path / "prior"
    args.resume_root.mkdir()
    graph.add(
        "passed",
        lambda: None,
        metadata={
            "format": "dcp",
            "source_world_size": 6,
            "world_size": 5,
            "device": "cuda",
        },
    )
    graph.add("failed", lambda: None, metadata={"format": "hf"})
    graph.add("future", lambda: None, metadata={"format": "hf"})
    pipeline.write_json(
        check_record("passed"),
        {
            "status": "pass",
            "seconds": 12.5,
            "io_payload_bytes": 560,
            "peak_gpu_allocated_bytes": [81, 82],
            "V": "pass",
            "R": "pass",
            "evidence_origin": "/prior/pass.json",
            "log_origin": "/prior/pass.log",
        },
    )
    pipeline.write_json(check_record("failed"), {"status": "fail", "error": "mismatch"})
    (args.resume_root / "gpu-sharing.jsonl").write_text(
        json.dumps(
            {
                "event": "wait_start",
                "job": "old",
                "sample": {"foreign_processes": [{"pid": 42}]},
            }
        )
        + "\n"
    )
    (pipeline.LOG / "gpu-sharing.jsonl").write_text(
        "\n".join(
            json.dumps(value)
            for value in (
                {"event": "gpu_admission", "job": "new"},
                {"event": "wait_end", "job": "new", "elapsed_seconds": 120},
                {"event": "foreign_during_job", "job": "new"},
            )
        )
        + "\n"
    )
    monkeypatch.setattr(pipeline, "build_plan", lambda args, model: graph)
    result = pipeline.summary(args)
    cells = {cell["check"]: cell for cell in result["cells"]}
    assert result["counts"] == {"pass": 1, "fail": 1, "pending": 1, "skipped": 0}
    assert cells["passed"]["source_world_size"] == 6
    assert cells["passed"]["world_size"] == 5
    assert cells["passed"]["device"] == "cuda"
    assert cells["passed"]["seconds"] == 12.5
    assert cells["passed"]["bytes"] == 560
    assert cells["passed"]["peak_gpu_memory_bytes"] == [81, 82]
    assert cells["passed"]["evidence_path"] == "/prior/pass.json"
    assert cells["passed"]["log_path"] == "/prior/pass.log"
    assert cells["passed"]["acceptance"] == {"V": "pass", "R": "pass"}
    assert cells["failed"]["reason"] == "mismatch"
    assert [wait["event"] for wait in result["gpu_sharing_waits"]] == [
        "wait_start",
        "wait_end",
        "foreign_during_job",
    ]
    assert "passed" in (pipeline.LOG / "SUMMARY.md").read_text()
    final = pipeline.summary(args, finished=True, rc=1)
    assert final["returncode"] == 1
    assert final["counts"]["pending"] == 0
    assert final["counts"]["skipped"] == 1


@pytest.mark.parametrize("changed", [False, True])
def test_resume_import_preserves_passes_only_when_exercised_code_matches(
    monkeypatch, tmp_path, changed
):
    args, _ = graph_harness(monkeypatch, tmp_path)
    args.resume_root = tmp_path / "prior"
    old = {
        "files": {
            "gdcp/core.py": "old",
            "scripts/gdcp_v2_pipeline.py": "controller-old",
        },
        "sha256": "old-source",
    }
    new = {
        "files": {
            "gdcp/core.py": "new" if changed else "old",
            "scripts/gdcp_v2_pipeline.py": "controller-new",
        },
        "sha256": "new-source",
    }
    args.resume_root.mkdir()
    (args.resume_root / "source-manifest.json").write_text(json.dumps(old))
    prior = args.resume_root / "C/4b/passed.json"
    prior.parent.mkdir(parents=True)
    prior.write_text(
        json.dumps({"status": "pass", "source_manifest_sha256": "old-source"})
    )
    prior.with_suffix(".log").write_text("prior evidence")
    (prior.parent / "failed.json").write_text(json.dumps({"status": "fail"}))
    monkeypatch.setattr(pipeline, "SOURCE_SHA", "new-source")
    monkeypatch.setattr(pipeline, "EXERCISED_SHA", pipeline.exercised_digest(new))
    pipeline.import_resume(args, new)
    imported = check_record("passed")
    assert imported.exists() is not changed
    assert not check_record("failed").exists()
    report = json.loads((pipeline.LOG / "resume-import.json").read_text())
    if changed:
        assert report["changed_exercised_files"] == ["gdcp/core.py"]
        assert "rerun" in report["reason"]
    else:
        value = json.loads(imported.read_text())
        assert value["evidence_origin"] == str(prior)
        assert value["original_source_manifest_sha256"] == "old-source"
        assert pipeline.check_done(imported)


def test_container_setup_refuses_to_claim_existing_container(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "LOG", tmp_path)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="refusing to claim ownership"):
        pipeline.start_containers(SimpleNamespace(stage="C", manage_containers=True))
    assert commands == [
        ["docker", "inspect", "--format", "{{.Id}}", "iobench-gdcp-v2-cpu"]
    ]


def test_partial_container_setup_cleans_only_started_container_id(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(pipeline, "LOG", tmp_path)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if command[1] == "inspect" else 0)

    def setup(command, **kwargs):
        if "gpu" in command:
            raise RuntimeError("GPU container setup failed")

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    monkeypatch.setattr(pipeline.subprocess, "check_call", setup)
    monkeypatch.setattr(
        pipeline.subprocess, "check_output", lambda *a, **k: "owned-cpu-id\n"
    )
    with pytest.raises(RuntimeError, match="setup failed"):
        pipeline.start_containers(SimpleNamespace(stage="C", manage_containers=True))
    assert commands[-1] == ["docker", "rm", "-f", "owned-cpu-id"]
    assert len([command for command in commands if command[1] == "rm"]) == 1


def test_main_continues_next_model_then_cleans_and_writes_completion(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(pipeline, "LOG", tmp_path / "logs")
    monkeypatch.setattr(pipeline, "CKPT", tmp_path / "checkpoints")
    monkeypatch.setattr(
        pipeline, "source_manifest", lambda: {"files": {}, "sha256": "source"}
    )
    monkeypatch.setattr(pipeline, "check_disk", lambda: {})
    monkeypatch.setattr(pipeline, "start_containers", lambda args: ["owned-cpu-id"])
    calls, cleanups = [], []

    def model_stage(args, model):
        calls.append(model)
        checkpoint = pipeline.CKPT / args.stage / model
        checkpoint.mkdir(parents=True)
        (checkpoint / "payload").write_bytes(b"checkpoint")
        if model == "4b":
            raise RuntimeError("one model failed")

    def run(command, **kwargs):
        cleanups.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline, "model_stage", model_stage)
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    rc = pipeline.main(
        [
            "--stage",
            "A",
            "--models",
            "4b",
            "14b",
            "--log-root",
            str(pipeline.LOG),
            "--manage-containers",
        ]
    )
    assert rc == 1
    assert calls == ["4b", "14b"]
    assert not (pipeline.CKPT / "A/4b").exists()
    assert not (pipeline.CKPT / "A/14b").exists()
    assert cleanups == [["docker", "rm", "-f", "owned-cpu-id"]]
    assert (pipeline.LOG / "sentinels.log").read_text().splitlines()[
        -1
    ] == "__GDCP_MODELS_ALL_DONE rc=1"
    summary = json.loads((pipeline.LOG / "summary.json").read_text())
    assert summary["finished"] is True
    assert summary["returncode"] == 1
    assert (pipeline.LOG / "SUMMARY.md").exists()


def test_prerequisite_rebuild_preserves_original_acceptance_and_log(
    monkeypatch, tmp_path
):
    _, graph = graph_harness(monkeypatch, tmp_path)
    checkpoint = pipeline.CKPT / "C/4b/first"
    record = check_record("save")
    pipeline.write_json(
        record, {"status": "pass", "seconds": 9.0, "reference": "original"}
    )
    original = record.read_bytes()
    record.with_suffix(".log").write_text("original acceptance log\n")

    def rebuild():
        checkpoint.mkdir(parents=True)
        pipeline.write_json(
            record, {"status": "pass", "seconds": 11.0, "reference": "regenerated"}
        )
        record.with_suffix(".log").write_text("regeneration log\n")

    graph.add("save", rebuild, outputs=(checkpoint,))
    assert graph.ensure("save", artifacts=True)
    assert checkpoint.is_dir()
    assert record.read_bytes() == original
    assert record.with_suffix(".log").read_text() == "original acceptance log\n"
    rebuilt = record.parent / "rebuild/save.json"
    assert json.loads(rebuilt.read_text())["seconds"] == 11.0
    assert rebuilt.with_suffix(".log").read_text() == "regeneration log\n"
    assert graph.new_checks == 0
