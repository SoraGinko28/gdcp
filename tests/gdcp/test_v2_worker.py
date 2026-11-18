"""Validation scope stays exact while redundant full-state hashes are omitted."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("huggingface_hub")

from scripts import gdcp_v2_worker as worker
from tests.support.gdcp_v2_models import MasterAdamW, build, local


def arguments(tmp_path, **changes):
    values = dict(
        model="4b",
        action="save",
        device="cpu",
        tiny=True,
        weights_only=False,
        dtype="bf16",
        format="safetensors",
        path=str(tmp_path / "checkpoint"),
        output="",
        reference="",
        result=str(tmp_path / "result.json"),
        max_shard_size=None,
        skip_reference=False,
        skip_step_check=False,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def forbidden(*args, **kwargs):
    raise AssertionError("redundant full snapshot or optimizer step executed")


@pytest.fixture
def small_state():
    model = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
    optimizer = MasterAdamW(model, 0, 1)
    return model, optimizer


@pytest.fixture
def process_group(tmp_path):
    torch.distributed.init_process_group(
        "gloo", rank=0, world_size=1, init_method=(tmp_path / "gloo-init").as_uri()
    )
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture
def disk_guard(monkeypatch):
    monkeypatch.setattr(
        worker, "disk_guard", lambda additional=0: {"additional_bytes": additional}
    )


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_weights_hash_only_exported_tensor_fields(monkeypatch, small_state, dtype):
    model, optimizer = small_state
    reference = worker.snapshot(model, optimizer, 0, 1)
    for fields in optimizer.state.values():
        fields["exp_avg"].fill_(99)
        fields["exp_avg_sq"].fill_(99)
        fields["step"].fill_(99)
    hashed, original = [], worker.tensor_hash

    def capture(tensor, rank, world):
        hashed.append(tensor)
        return original(tensor, rank, world)

    monkeypatch.setattr(worker, "tensor_hash", capture)
    actual = worker.weights_snapshot(model, optimizer, 0, 1, dtype)
    worker.check_weights_equal(actual, reference, dtype)
    assert len(hashed) == len(list(model.parameters()))
    assert "param_groups" not in actual
    assert all(worker._weight_key(name, dtype) for name in actual["tensors"])
    assert all(
        value["dtype"] == ("torch.float32" if dtype == "fp32" else "torch.bfloat16")
        for value in actual["tensors"].values()
    )


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_weights_still_detect_bitwise_corruption(small_state, dtype):
    model, optimizer = small_state
    reference = worker.snapshot(model, optimizer, 0, 1)
    first = next(model.parameters())
    with torch.no_grad():
        optimizer.state[first]["master_param"].add_(1)
        worker._sync_master_parameters(model, optimizer)
    actual = worker.weights_snapshot(model, optimizer, 0, 1, dtype)
    with pytest.raises(AssertionError, match="V weights"):
        worker.check_weights_equal(actual, reference, dtype)


def test_weights_still_require_master_cast_relation(small_state):
    model, optimizer = small_state
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(AssertionError, match="bf16 != master_param"):
        worker.weights_snapshot(model, optimizer, 0, 1, "bf16")


def test_fp32_weight_snapshot_rejects_wrong_actual_dtype(small_state):
    model, optimizer = small_state
    first = next(model.parameters())
    optimizer.state[first]["master_param"] = optimizer.state[first][
        "master_param"
    ].double()
    with pytest.raises(AssertionError, match="weights dtype"):
        worker.weights_snapshot(model, optimizer, 0, 1, "fp32")


@pytest.mark.parametrize(
    "changes",
    [
        dict(action="native-save", skip_reference=True),
        dict(action="export", skip_reference=True),
        dict(action="load", skip_reference=True),
        dict(action="build", skip_reference=True),
        dict(action="save", weights_only=True, skip_reference=True),
        dict(action="save", skip_step_check=True),
        dict(action="native-load", skip_step_check=True),
        dict(action="load", weights_only=True, skip_step_check=True),
    ],
)
def test_skip_flags_cannot_disable_required_validation_for_another_action(
    tmp_path, changes
):
    with pytest.raises(ValueError):
        worker.validate_args(arguments(tmp_path, **changes))


@pytest.mark.parametrize("key", ["4b", "14b", "24b"])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("sharded", [False, True])
def test_three_architectures_weights_load_and_resave_exactly_without_optimizer_r(
    monkeypatch, tmp_path, process_group, disk_guard, key, dtype, sharded
):
    model, optimizer, info = build(key, 1, 0, "cpu", tiny=True)
    state = {"model": model, "optimizer": optimizer}
    reference = worker.snapshot(model, optimizer, 0, 1)
    limit = 16384 if sharded else None
    path = tmp_path / "weights"
    worker.save_hf(path, state, dtype=dtype, max_shard_size=limit)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps({"reference": reference}))
    worker._poison_model(model)
    if dtype == "fp32":
        worker._poison_master_weights(optimizer)
    monkeypatch.setattr(worker, "snapshot", forbidden)
    monkeypatch.setattr(optimizer, "step", forbidden)
    args = arguments(
        tmp_path,
        model=key,
        action="load",
        weights_only=True,
        dtype=dtype,
        path=str(path),
        reference=str(reference_path),
        output=str(tmp_path / "again"),
        max_shard_size=limit,
    )
    record = {"optimizer_step_executed": False}
    worker._load_check(args, state, info, record, 0, 1)
    assert (
        record["V"]
        == record["local_shapes"]
        == record["master_cast_relation"]
        == "pass"
    )
    assert record["V_scope"] == "weights-" + dtype
    assert record["R"] == "not-applicable-weights-only"
    assert record["optimizer_step_executed"] is False
    assert "resave_seconds" in record
    original = {
        file.name: file.read_bytes() for file in path.iterdir() if file.is_file()
    }
    returned = {
        file.name: file.read_bytes()
        for file in Path(args.output).iterdir()
        if file.is_file()
    }
    assert original == returned


@pytest.mark.parametrize(
    "fmt,limit", [("safetensors", None), ("dcp", None), ("hf", None), ("hf", 16384)]
)
def test_full_source_and_target_keep_v_r_and_return_keeps_full_v_resave(
    monkeypatch, tmp_path, process_group, disk_guard, fmt, limit
):
    args = arguments(tmp_path, format=fmt, max_shard_size=limit)
    worker.run(args, 0, 1)
    source = json.loads(Path(args.result).read_text())
    assert source["reference_scope"] == "full-state"
    assert source["optimizer_step_executed"] is True
    assert source["reference"]["tensors"] != source["stepped_reference"]["tensors"]
    model, optimizer, info = build("4b", 1, 0, "cpu", tiny=True, poison=True)
    args.action, args.reference = "load", args.result
    target = {"optimizer_step_executed": False}
    worker._load_check(
        args, {"model": model, "optimizer": optimizer}, info, target, 0, 1
    )
    assert target["V"] == target["R"] == "pass"
    assert target["optimizer_step_executed"] is True
    model, optimizer, info = build("4b", 1, 0, "cpu", tiny=True, poison=True)
    monkeypatch.setattr(optimizer, "step", forbidden)
    args.skip_step_check, args.output = True, str(tmp_path / "again")
    returned = {"optimizer_step_executed": False}
    worker._load_check(
        args, {"model": model, "optimizer": optimizer}, info, returned, 0, 1
    )
    assert returned["V"] == returned["local_shapes"] == "pass"
    assert returned["V_scope"] == "full-state"
    assert returned["R"] == "not-applicable-return-to-source"
    assert returned["optimizer_step_executed"] is False
    original = {
        str(file.relative_to(args.path)): file.read_bytes()
        for file in Path(args.path).rglob("*")
        if file.is_file()
    }
    resaved = {
        str(file.relative_to(args.output)): file.read_bytes()
        for file in Path(args.output).rglob("*")
        if file.is_file()
    }
    assert original == resaved


@pytest.mark.parametrize(
    "action,dtype",
    [("save", "bf16"), ("native-save", "bf16"), ("export", "fp32"), ("export", "bf16")],
)
def test_duplicate_native_and_export_jobs_omit_unused_full_references(
    monkeypatch, tmp_path, process_group, disk_guard, action, dtype
):
    model, optimizer, info = build("4b", 1, 0, "cpu", tiny=True)
    monkeypatch.setattr(
        worker, "build", lambda *args, **kwargs: (model, optimizer, info)
    )
    monkeypatch.setattr(worker, "snapshot", forbidden)
    monkeypatch.setattr(optimizer, "step", forbidden)
    args = arguments(
        tmp_path, action=action, dtype=dtype, skip_reference=action == "save"
    )
    worker.run(args, 0, 1)
    record = json.loads(Path(args.result).read_text())
    assert record["status"] == "pass"
    assert record["reference_scope"] == "omitted-unused"
    assert "reference" not in record and "stepped_reference" not in record
    assert record["optimizer_step_executed"] is False
    assert record["peak_gpu_allocated_bytes"] == [0]
    assert record["model_info"]["parameters"] == info["parameters"]
    assert record["seconds"] >= 0 and record["io_payload_bytes"] > 0
    if action == "export":
        assert record["master_cast_relation"] == "pass"


def test_full_return_v_still_detects_corrupted_adam_moment(
    monkeypatch, tmp_path, process_group, disk_guard
):
    args = arguments(tmp_path)
    worker.run(args, 0, 1)
    args.action, args.reference, args.skip_step_check = "load", args.result, True
    model, optimizer, info = build("4b", 1, 0, "cpu", tiny=True, poison=True)
    original_load = worker.FSDP2Checkpointer.load

    def corrupt_after_load(*call_args, **kwargs):
        original_load(*call_args, **kwargs)
        first = next(model.parameters())
        local(optimizer.state[first]["exp_avg"]).add_(1)

    monkeypatch.setattr(worker.FSDP2Checkpointer, "load", corrupt_after_load)
    with pytest.raises(AssertionError, match="V"):
        worker._load_check(
            args, {"model": model, "optimizer": optimizer}, info, {}, 0, 1
        )
