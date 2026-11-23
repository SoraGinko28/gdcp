"""GDCP v1 optimizer acceptance cases, with real FSDP2 and a dense oracle."""

from pathlib import Path
from typing import overload, Callable

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from gdcp import FSDP2Checkpointer

KINDS = (
    "sgd",
    "sgd_empty",
    "adam",
    "adamw",
    "adamw_amsgrad",
    "adagrad",
    "rmsprop",
    "custom",
)
FORMATS = ("safetensors", "dcp", "hf")


def dense(value):
    return (value.full_tensor() if isinstance(value, DTensor) else value).cpu()


class LayoutOptimizer(torch.optim.Optimizer):
    """An optimizer whose persistent buffers have several independent shapes."""

    def __init__(self, params, lr=0.125):
        super().__init__(params, dict(lr=lr, custom_option="flat-and-reduced"))

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            _update_custom_group(group, self.state)
        return loss


def _update_custom_group(group, state):
    for parameter in group["params"]:
        _update_custom(parameter, state[parameter], group["lr"])


def _resolved_reduction(parameter, dim):
    value = parameter.sum(dim=dim)
    if not isinstance(value, DTensor):
        return value
    placements = [Replicate() if p.is_partial() else p for p in value.placements]
    return value.redistribute(placements=placements)


def _custom_slots(parameter):
    slots: dict = dict(
        flat=torch.zeros_like(parameter).view(-1),
        rows=torch.zeros_like(_resolved_reduction(parameter, 1)),
        columns=torch.zeros_like(_resolved_reduction(parameter, 0)),
        replicated=torch.arange(3, dtype=torch.float64, device=parameter.device),
        count=0,
        enabled=True,
        label="custom",
        unused=None,
        ratio=0.5,
    )
    if isinstance(parameter, DTensor):
        slots["replicated"] = distribute_tensor(
            slots["replicated"],
            parameter.device_mesh,
            [Replicate()] * parameter.device_mesh.ndim,
        )
    return slots


def _update_custom(parameter, slots, lr):
    if parameter.grad is None:
        return
    if not slots:
        slots.update(_custom_slots(parameter))
    slots["flat"].add_(parameter.grad.view(-1))
    slots["rows"].add_(_resolved_reduction(parameter.grad, 1))
    slots["columns"].add_(_resolved_reduction(parameter.grad, 0))
    slots["replicated"].add_(0.25)
    slots["count"] += 1
    parameter.add_(parameter.grad, alpha=-lr)


def optimizer_for(kind, model):
    params = model.parameters()
    if kind == "sgd":
        return torch.optim.SGD(params, lr=0.125, momentum=0.75)
    if kind == "sgd_empty":
        return torch.optim.SGD(params, lr=0.125)
    if kind == "adam":
        return torch.optim.Adam(params, lr=0.125, foreach=False)
    if kind.startswith("adamw"):
        return torch.optim.AdamW(
            params, lr=0.125, amsgrad=kind.endswith("amsgrad"), foreach=False
        )
    if kind == "adagrad":
        return torch.optim.Adagrad(params, lr=0.125, foreach=False)
    if kind == "rmsprop":
        return torch.optim.RMSprop(
            params, lr=0.125, momentum=0.5, centered=True, foreach=False
        )
    return LayoutOptimizer(params)


def model_for(mesh=None, value=0):
    model = torch.nn.Linear(
        4, 8, bias=False, device=mesh.device_type if mesh is not None else "cpu"
    )
    with torch.no_grad():
        model.weight.copy_(torch.arange(32).reshape(8, 4) / 32 + value)
    if mesh is not None:
        fully_shard(model, mesh=mesh)
    return model


def update(model, optimizer, step):
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter) * ((step + 1) / 8)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def assert_equal(actual, optimizer, expected, expected_optimizer):
    for parameter, reference in zip(
        actual.parameters(), expected.parameters(), strict=True
    ):
        torch.testing.assert_close(dense(parameter), dense(reference), rtol=0, atol=0)
        fields, wanted = (
            optimizer.state.get(parameter, {}),
            expected_optimizer.state.get(reference, {}),
        )
        assert fields.keys() == wanted.keys()
        for field, value in fields.items():
            _assert_field(field, value, wanted[field])


def _assert_field(field, actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(dense(actual), dense(expected), rtol=0, atol=0)
    else:
        assert actual == expected, (field, actual, expected)


def roundtrip(root, mesh, kind, fmt):
    source, oracle = model_for(mesh), model_for()
    saved, oracle_optimizer = optimizer_for(kind, source), optimizer_for(kind, oracle)
    for step in range(2):
        update(source, saved, step)
        update(oracle, oracle_optimizer, step)
    path = Path(root) / kind / fmt
    FSDP2Checkpointer.save(
        path, {"model": source, "optimizer": saved}, checkpoint_format=fmt
    )
    target = model_for(mesh, 10)
    restored = optimizer_for(kind, target)
    target.weight.grad = torch.full_like(target.weight, 0.75)
    gradient = dense(target.weight.grad).clone()
    restored.step = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("load called optimizer.step")
    )
    FSDP2Checkpointer.load(path, {"model": target, "optimizer": restored})
    del restored.step
    torch.testing.assert_close(dense(target.weight.grad), gradient, rtol=0, atol=0)
    assert_equal(target, restored, oracle, oracle_optimizer)
    update(target, restored, 2)
    update(oracle, oracle_optimizer, 2)
    assert_equal(target, restored, oracle, oracle_optimizer)


def file_worker(rank, world, rendezvous, root):
    init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        mesh = DeviceMesh("cpu", list(range(world)))
        _matrix(root, mesh)
        # Ambiguous local values are accepted only when every byte agrees.
        model = model_for(mesh)
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.5)
        optimizer.state[model.weight]["statistic"] = torch.tensor([1.0, 2.0, 3.0])
        FSDP2Checkpointer.save(
            Path(root) / "plain-identical", {"model": model, "optimizer": optimizer}
        )
        optimizer.state[model.weight]["statistic"][0] = rank
        try:
            FSDP2Checkpointer.save(
                Path(root) / "plain-divergent", {"model": model, "optimizer": optimizer}
            )
        except ValueError as exc:
            assert "state.weight.statistic" in str(
                exc
            ) and "needs to be a DTensor" in str(exc)
        else:
            raise AssertionError("divergent plain state was accepted")
        assert not (Path(root) / "plain-divergent").exists()
        print(f"optimizer matrix rank {rank} PASS", flush=True)
    finally:
        destroy_process_group()


def _matrix(root, mesh):
    for kind in KINDS:
        for fmt in FORMATS:
            roundtrip(root, mesh, kind, fmt)


def layout_model(world, layout):
    from torch.distributed.tensor import Shard

    if layout == "fsdp":
        return model_for(DeviceMesh("cpu", list(range(world))))
    ranks = torch.arange(world).reshape(2, world // 2)
    mesh = DeviceMesh(
        "cpu",
        ranks,
        mesh_dim_names=("dp", "tp") if layout == "tp" else ("replica", "dp"),
    )
    model = model_for()
    if layout == "tp":
        model.weight = torch.nn.Parameter(
            distribute_tensor(model.weight.detach(), mesh["tp"], [Shard(1)])
        )
        fully_shard(model, mesh=mesh["dp"])
    else:
        fully_shard(model, mesh=mesh)
    return model


def layout_worker(rank, world, rendezvous, root, action, layout):
    init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        model = layout_model(world, layout)
        optimizer = optimizer_for("custom", model)
        oracle = model_for()
        reference = optimizer_for("custom", oracle)
        for step in range(2):
            update(oracle, reference, step)
        if action == "save":
            update(model, optimizer, 0)
            update(model, optimizer, 1)
            _save_formats(root, model, optimizer)
        else:
            _load_formats(root, model, optimizer, oracle, reference)
        print(
            f"optimizer layout {world} {layout} {action} rank {rank} PASS", flush=True
        )
    finally:
        destroy_process_group()


def _save_formats(root, model, optimizer):
    for fmt in FORMATS:
        FSDP2Checkpointer.save(
            Path(root) / fmt,
            {"model": model, "optimizer": optimizer},
            checkpoint_format=fmt,
        )


def _load_formats(root, model, optimizer, oracle, reference):
    for fmt in FORMATS:
        oracle = model_for()
        reference = optimizer_for("custom", oracle)
        update(oracle, reference, 0)
        update(oracle, reference, 1)
        FSDP2Checkpointer.load(
            Path(root) / fmt, {"model": model, "optimizer": optimizer}
        )
        assert_equal(model, optimizer, oracle, reference)
        update(model, optimizer, 2)
        update(oracle, reference, 2)
        assert_equal(model, optimizer, oracle, reference)


def optimizer_matrix(args):
    from torch.distributed import get_world_size

    device = "cpu" if args.backend == "gloo" else "cuda"
    mesh = DeviceMesh(device, list(range(get_world_size())))
    _matrix(args.path, mesh)
    print("optimizer matrix PASS", flush=True)
