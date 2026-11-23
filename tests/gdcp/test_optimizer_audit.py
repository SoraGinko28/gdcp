"""Real FSDP2 optimizer ambiguity and independent submesh regressions."""

import time

import pytest

torch = pytest.importorskip("torch")

from tests.support.gdcp_optimizer_audit import plain_worker, submesh_worker


def test_plain_optimizer_state_requires_collective_byte_agreement(tmp_path):
    torch.multiprocessing.spawn(
        plain_worker,
        args=(2, str(tmp_path / "rendezvous"), str(tmp_path / "checkpoints")),
        nprocs=2,
        join=True,
    )


def test_lazy_optimizer_state_on_independent_submesh(tmp_path):
    torch.multiprocessing.spawn(
        submesh_worker,
        args=(4, str(tmp_path / "rendezvous"), str(tmp_path / "checkpoints")),
        nprocs=4,
        join=True,
    )


def _bounded_spawn(worker, world, *arguments):
    context = torch.multiprocessing.spawn(
        worker, args=(world, *arguments), nprocs=world, join=False
    )
    assert context is not None
    deadline = time.monotonic() + 75
    try:
        complete = False
        while time.monotonic() < deadline and not complete:
            complete = context.join(timeout=max(0, deadline - time.monotonic()))
        if not complete:
            pytest.fail("optimizer ownership worker timed out")
    finally:
        _stop_workers(context.processes)


def _stop_workers(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)


def test_lazy_optimizer_state_on_single_submesh(tmp_path):
    from tests.support.gdcp_optimizer_audit import sole_submesh_worker

    _bounded_spawn(
        sole_submesh_worker,
        4,
        str(tmp_path / "rendezvous"),
        str(tmp_path / "checkpoints"),
    )


@pytest.mark.parametrize("source_world,target_world", [(2, 4), (4, 2)])
def test_independent_optimizer_mesh_change_requires_destination_layout(
    tmp_path, source_world, target_world
):
    from tests.support.gdcp_optimizer_audit import independent_world_worker

    _bounded_spawn(
        independent_world_worker,
        source_world,
        str(tmp_path / "save-rendezvous"),
        str(tmp_path / "checkpoints"),
        "save",
    )
    _bounded_spawn(
        independent_world_worker,
        target_world,
        str(tmp_path / "load-rendezvous"),
        str(tmp_path / "checkpoints"),
        "load",
    )


def test_independent_optimizer_mesh_uses_only_its_checkpoint_subgroup(tmp_path):
    from tests.support.gdcp_optimizer_audit import independent_subgroup_worker

    _bounded_spawn(
        independent_subgroup_worker,
        4,
        str(tmp_path / "rendezvous"),
        str(tmp_path / "checkpoints"),
    )


def test_independent_optimizer_state_preserves_unnamed_multidimensional_mesh(tmp_path):
    from tests.support.gdcp_optimizer_audit import independent_multidim_worker

    _bounded_spawn(
        independent_multidim_worker,
        4,
        str(tmp_path / "rendezvous"),
        str(tmp_path / "checkpoints"),
    )
