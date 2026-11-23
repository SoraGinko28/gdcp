"""Real-rank accessor checks, launched with a file store to avoid Mac DNS."""

from pathlib import Path

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from gdcp import FSDP2Checkpointer
from tests.support.gdcp_accessors import MemoryFactory
from tests.support.gdcp_optimizers import FORMATS, model_for


class ReplicaFactory(MemoryFactory):
    def __init__(self, delta, component):
        super().__init__()
        self.delta = delta
        self.component = component

    def __call__(self, request):
        accessor = super().__call__(request)
        if request.component == self.component:
            for value in accessor.storage.values():
                value.add_(self.delta)
        return accessor


def _rejected(path, model, optimizer, fmt, factory):
    state = {"model": model}
    if optimizer is not None:
        state["optimizer"] = optimizer
    try:
        FSDP2Checkpointer.save(
            path, state, checkpoint_format=fmt, data_accessor=factory
        )
    except ValueError as exc:
        assert "weight" in str(exc)
        assert "different values" in str(exc) or "differs across ranks" in str(exc)
    else:
        raise AssertionError("accessor supplied divergent replica bytes were accepted")
    assert not path.exists()


def replica_worker(rank, world, rendezvous, root):
    init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        full = DeviceMesh("cpu", list(range(world)))
        grid = DeviceMesh(
            "cpu",
            torch.arange(world).reshape(2, 2),
            mesh_dim_names=("replica", "shard"),
        )
        for fmt in FORMATS:
            _plain_optimizer(Path(root) / fmt, full, rank, fmt)
            _submesh(Path(root) / fmt, grid["shard"], rank, fmt)
        print(f"accessor replica checks rank {rank} PASS", flush=True)
    finally:
        destroy_process_group()


def _plain_optimizer(root, mesh, rank, fmt):
    model = model_for(mesh)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
    # A genuinely storage-free, ambiguously replicated state must be checked
    # through its accessor. A direct tensor read would fail here.
    optimizer.state[model.weight]["statistic"] = torch.empty(3, device="meta")
    factory = ReplicaFactory(0, "optimizer")
    FSDP2Checkpointer.save(
        root / "plain-accessor-equal",
        {"model": model, "optimizer": optimizer},
        checkpoint_format=fmt,
        data_accessor=factory,
    )
    assert any(
        r.global_tensor_id == "state.weight.statistic"
        for a in factory.accessors
        for r in a.reads
    )
    # The underlying real storage now agrees, while the public accessor's
    # bytes disagree. Hashing the tensor instead would wrongly publish files.
    optimizer.state[model.weight]["statistic"] = torch.zeros(3)
    _rejected(
        root / "plain-accessor-divergent",
        model,
        optimizer,
        fmt,
        ReplicaFactory(rank, "optimizer"),
    )


def _submesh(root, mesh, rank, fmt):
    # Independent submeshes carry different tensor storage. The accessor
    # supplies matching corresponding shards and is the authority for I/O.
    model = model_for(mesh, value=rank // 2)
    FSDP2Checkpointer.save(
        root / "submesh-accessor-equal",
        {"model": model},
        checkpoint_format=fmt,
        data_accessor=ReplicaFactory(0, "model"),
    )
    model = model_for(mesh)
    _rejected(
        root / "submesh-accessor-divergent",
        model,
        None,
        fmt,
        ReplicaFactory(rank // 2, "model"),
    )


def meta_optimizer_worker(rank, world, rendezvous, root):
    init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        mesh = DeviceMesh("cpu", list(range(world)), mesh_dim_names=("fsdp",))
        for fmt in FORMATS:
            source = torch.nn.Linear(4, 8, bias=False, device="meta")
            fully_shard(source, mesh=mesh)
            optimizer = torch.optim.AdamW(source.parameters())
            optimizer.state[source.weight].update(
                step=torch.tensor(5.0),
                exp_avg=torch.empty_like(source.weight),
                exp_avg_sq=torch.empty_like(source.weight),
            )
            optimizer.step = _no_step
            path = Path(root) / fmt
            FSDP2Checkpointer.save(
                path,
                {"model": source, "optimizer": optimizer},
                checkpoint_format=fmt,
                data_accessor=MemoryFactory(),
            )
            target = torch.nn.Linear(4, 8, bias=False, device="meta")
            fully_shard(target, mesh=mesh)
            restored = torch.optim.AdamW(target.parameters())
            restored.step = _no_step
            FSDP2Checkpointer.load(
                path,
                {"model": target, "optimizer": restored},
                data_accessor=MemoryFactory(),
            )
            state = restored.state[target.weight]
            assert isinstance(target.weight, DTensor)
            assert state["step"].device.type == "cpu"
            assert state["step"].item() == 5
            for field in ("exp_avg", "exp_avg_sq"):
                value = state[field]
                assert isinstance(value, DTensor)
                assert value.device.type == "meta"
                assert value.device_mesh is target.weight.device_mesh
                assert value.placements == target.weight.placements
        print(f"accessor meta optimizer rank {rank} PASS", flush=True)
    finally:
        destroy_process_group()


def _no_step(*args, **kwargs):
    raise AssertionError("checkpoint called optimizer.step()")
