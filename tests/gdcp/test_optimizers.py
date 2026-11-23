"""GDCP v1 arbitrary optimizer and independent state layout regressions."""

import pytest

torch = pytest.importorskip("torch")

from torch.distributed import HashStore, destroy_process_group, init_process_group
from torch.distributed.device_mesh import DeviceMesh
from gdcp.adapters.torch_access import _local_tensor

from gdcp import FSDP2Checkpointer
from gdcp.adapters import fsdp2
from gdcp.state import RankContext
from tests.support.gdcp_optimizers import (
    FORMATS,
    KINDS,
    model_for,
    optimizer_for,
    roundtrip,
    update,
)


@pytest.fixture
def mesh(monkeypatch):
    pytest.importorskip("torch.testing._internal.distributed.fake_pg")
    init_process_group("fake", store=HashStore(), rank=0, world_size=1)
    monkeypatch.setattr(fsdp2, "_context", lambda process_group=None: RankContext(0, 1))
    try:
        yield DeviceMesh("cpu", [0])
    finally:
        destroy_process_group()


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fmt", FORMATS)
def test_optimizer_lazy_roundtrip_and_continuation(mesh, tmp_path, kind, fmt):
    roundtrip(tmp_path, mesh, kind, fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_optimizer_class_mismatch_preserves_everything(mesh, tmp_path, fmt):
    source = model_for(mesh)
    saved = optimizer_for("adam", source)
    update(source, saved, 0)
    FSDP2Checkpointer.save(
        tmp_path, {"model": source, "optimizer": saved}, checkpoint_format=fmt
    )
    target = model_for(mesh, 10)
    optimizer = optimizer_for("sgd", target)
    before = _local_tensor(target.weight).clone()
    with pytest.raises(ValueError, match=r"class mismatch.*Adam.*SGD"):
        FSDP2Checkpointer.load(tmp_path, {"model": target, "optimizer": optimizer})
    assert not optimizer.state
    torch.testing.assert_close(before, _local_tensor(target.weight), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ("sgd", "adam"))
def test_native_pytorch_optimizer_lazy_load(mesh, tmp_path, kind):
    from torch.distributed.checkpoint import save
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
    from tests.support.gdcp_optimizers import assert_equal

    source = model_for(mesh)
    saved = optimizer_for(kind, source)
    update(source, saved, 0)
    save(
        {
            "model": source.state_dict(),
            "optimizer": get_optimizer_state_dict(source, saved),
        },
        checkpoint_id=tmp_path,
        no_dist=True,
    )
    target = model_for(mesh, 10)
    optimizer = optimizer_for(kind, target)
    FSDP2Checkpointer.load(tmp_path, {"model": target, "optimizer": optimizer})
    assert_equal(target, optimizer, source, saved)


def test_real_gloo_optimizer_matrix(tmp_path):
    from torch.multiprocessing import spawn
    from tests.support.gdcp_optimizers import file_worker

    spawn(
        file_worker,
        args=(2, str(tmp_path / "rendezvous"), str(tmp_path / "matrix")),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    "source_world,source_layout,target_world,target_layout",
    [
        (2, "fsdp", 4, "hsdp"),
        (4, "hsdp", 2, "fsdp"),
        (4, "tp", 2, "fsdp"),
        (2, "fsdp", 4, "tp"),
    ],
)
def test_optimizer_state_cross_layout(
    tmp_path, source_world, source_layout, target_world, target_layout
):
    from tests.support.gdcp_optimizers import layout_worker

    torch.multiprocessing.spawn(
        layout_worker,
        args=(
            source_world,
            str(tmp_path / "save-rendezvous"),
            str(tmp_path / "checkpoints"),
            "save",
            source_layout,
        ),
        nprocs=source_world,
        join=True,
    )
    torch.multiprocessing.spawn(
        layout_worker,
        args=(
            target_world,
            str(tmp_path / "load-rendezvous"),
            str(tmp_path / "checkpoints"),
            "load",
            target_layout,
        ),
        nprocs=target_world,
        join=True,
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_compatible_optimizer_subclass_can_restore_parent(mesh, tmp_path, fmt):
    from tests.support.gdcp_optimizers import assert_equal

    class CompatibleAdam(torch.optim.Adam):
        pass

    source = model_for(mesh)
    saved = optimizer_for("adam", source)
    update(source, saved, 0)
    FSDP2Checkpointer.save(
        tmp_path, {"model": source, "optimizer": saved}, checkpoint_format=fmt
    )
    target = model_for(mesh, 10)
    restored = CompatibleAdam(target.parameters(), lr=0.125, foreach=False)
    FSDP2Checkpointer.load(tmp_path, {"model": target, "optimizer": restored})
    assert_equal(target, restored, source, saved)


@pytest.mark.parametrize("fmt", FORMATS)
def test_matching_adam_classes_preserve_saved_group_options(mesh, tmp_path, fmt):
    from tests.support.gdcp_optimizers import assert_equal

    source = model_for(mesh)
    saved = optimizer_for("adam", source)
    update(source, saved, 0)
    target = model_for(mesh, 10)
    loaded = optimizer_for("adamw", target)
    if saved.param_groups[0].keys() != loaded.param_groups[0].keys():
        pytest.skip(
            "this torch version gives Adam and AdamW different group option keys"
        )
    FSDP2Checkpointer.save(
        tmp_path, {"model": source, "optimizer": saved}, checkpoint_format=fmt
    )
    FSDP2Checkpointer.load(tmp_path, {"model": target, "optimizer": loaded})
    assert {
        key: value for key, value in saved.param_groups[0].items() if key != "params"
    } == {
        key: value
        for key, value in loaded.param_groups[0].items()
        if key not in {"params", "param_names"}
    }
    assert_equal(target, loaded, source, saved)


@pytest.mark.parametrize("fmt", FORMATS)
def test_legacy_adamw_checkpoint_loads_compatible_adam(
    mesh, tmp_path, fmt, monkeypatch
):
    from tests.support.gdcp_optimizers import assert_equal

    source = model_for(mesh)
    saved = optimizer_for("adamw", source)
    update(source, saved, 0)
    target = model_for(mesh, 10)
    restored = optimizer_for("adam", target)
    if saved.param_groups[0].keys() != restored.param_groups[0].keys():
        pytest.skip(
            "this torch version gives Adam and AdamW different group option keys"
        )
    original_values = fsdp2._optimizer_values

    def legacy_values(state):
        state = dict(state)
        state.pop("_gdcp_optimizer", None)
        return original_values(state)

    with monkeypatch.context() as patch:
        patch.setattr(fsdp2, "_optimizer_values", legacy_values)
        FSDP2Checkpointer.save(
            tmp_path, {"model": source, "optimizer": saved}, checkpoint_format=fmt
        )
    FSDP2Checkpointer.load(tmp_path, {"model": target, "optimizer": restored})
    assert_equal(target, restored, source, saved)
    expected = {
        key: value
        for key, value in saved.param_groups[0].items()
        if key not in {"params", "param_names"}
    }
    actual = {
        key: value
        for key, value in restored.param_groups[0].items()
        if key not in {"params", "param_names"}
    }
    assert actual == expected
