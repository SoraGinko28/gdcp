"""Collective, framework-neutral checkpoint save, load, and inspection."""

import logging
from pathlib import Path
from typing import Any, Callable

from gdtensor import Result
from gdtensor.builder import (
    ComputeDTensorsView,
    Err,
    GDTensorIOError,
    Ok,
    global_tensor_meta_of,
)
from gdtensor.io import rank_load, rank_save

from . import desc
from ._transaction import PREFLIGHT, DirectoryTransaction
from .backend import SafeTensorsBackend, _raise_io_interrupt
from .policy import PerRankPolicy

_LOG = logging.getLogger(__name__)


class _Failure(Exception):
    def __init__(self, error):
        self.error = error
        super().__init__(error.msg)


def _require(result):
    if type(result) is Err:
        raise _Failure(result.value)
    return result.value


def _error(operation, exc):
    if isinstance(exc, _Failure):
        return Err(exc.error)
    return Err(GDTensorIOError(operation=operation, msg=str(exc)))


def _collective(ctx, operation, action: Callable[[], Any]) -> Result[Any]:
    interrupt = None
    try:
        value = action()
        result = value if type(value) in (Ok, Err) else Ok(value)
    except BaseException as exc:
        result = _error(operation, exc)
        if not isinstance(exc, Exception):
            interrupt = exc
    succeeded = type(result) is Ok
    if ctx.coordinator is not None:
        succeeded = ctx.coordinator.vote(succeeded)
    if interrupt is not None:
        raise interrupt
    if not succeeded and type(result) is Ok:
        return _error(
            operation, RuntimeError(f"another rank failed during {operation}")
        )
    return result


def _validate_context(ctx):
    if ctx.world_size < 1 or not 0 <= ctx.rank < ctx.world_size:
        raise ValueError("invalid rank or world_size")
    if ctx.world_size > 1 and ctx.coordinator is None:
        raise ValueError("multiple ranks require a coordinator")


def _validate_state(state):
    names = [component.name for component in state.components]
    if len(set(names)) != len(names):
        raise ValueError("duplicate component name")
    for name in names + list(state.rank_state):
        desc._safe_relative(name)
    for name, data in state.rank_state.items():
        if not isinstance(data, bytes):
            raise TypeError(f"rank_state {name} must contain bytes")


def _compute_view(component):
    return ComputeDTensorsView(
        [named.global_tensor_id for named in component.table.named_d_tensors],
        component.table,
        component.accessor,
    )


def _prepare_save(path, state, ctx, policy, backend):
    _validate_context(ctx)
    _validate_state(state)
    prepared = []
    for component in state.components:
        compute = _compute_view(component)
        _require(compute.check_constraints())
        table, layout = policy.plan(component, ctx.world_size)
        storage = backend.open(
            Path(path) / component.name, table, layout, ctx.coordinator
        )
        _require(storage.check_constraints())
        prepared.append((component, compute, storage))
    return prepared


def _initialize_save(path, state, ctx):
    if ctx.rank != 0:
        return
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    desc.atomic_write(root / desc.PENDING_NAME, b"GDCP save in progress\n")
    for name in [component.name for component in state.components] + list(
        state.rank_state
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def _save_one(ctx, compute, storage):
    # The connector has no session leader for a completely empty component.
    # Its semantic tensor entries still live in GDCP's manifest.
    if not storage.physical_layout.tensor_entries:
        return Ok(None)
    result = rank_save(ctx.rank, compute, storage)
    _raise_io_interrupt(storage)
    if type(result) is Ok:
        getattr(compute.data_accessor, "finish", lambda: None)()
    return result


def _save_objects(path, component, ctx):
    if ctx.rank == 0 and component.objects:
        desc.write_objects(
            Path(path) / component.name / "objects.json", component.objects
        )


def _save_components(path, prepared, ctx):
    for component, compute, storage in prepared:
        result = _collective(
            ctx, f"save {component.name}", lambda: _save_one(ctx, compute, storage)
        )
        if type(result) is Err:
            return result
        result = _collective(
            ctx,
            f"objects {component.name}",
            lambda: _save_objects(path, component, ctx),
        )
        if type(result) is Err:
            return result
    return Ok(None)


def _rank_state_path(root, name, rank):
    return Path(root) / name / f"{name}_rank_{rank}.pt"


def _save_rank_state(path, state, ctx):
    for name, data in state.rank_state.items():
        desc.atomic_write(_rank_state_path(path, name, ctx.rank), data)


def _publish(path, state, prepared, ctx):
    if ctx.rank != 0:
        return
    components = {}
    for component, compute, storage in prepared:
        components[component.name] = desc.describe_component(
            storage.d_tensor_table,
            storage.physical_layout,
            ctx.world_size,
            objects=bool(component.objects),
            logical_shapes=component.logical_shapes,
            root=Path(path) / component.name,
        )
    manifest = {
        "gdcp_version": 1,
        "framework": state.framework,
        "world_size": ctx.world_size,
        "global_steps": state.global_steps,
        "components": components,
        "rank_state": {
            name: {
                "ranks": list(range(ctx.world_size)),
                "pattern": f"{name}_rank_{{rank}}.pt",
            }
            for name in state.rank_state
        },
    }
    desc.write_manifest(path, manifest)
    (Path(path) / desc.PENDING_NAME).unlink()


def _planned_rank_files(state, ctx):
    return {
        Path(name) / f"{name}_rank_{rank}.pt"
        for name in state.rank_state
        for rank in range(ctx.world_size)
    }


def _planned_files(state, prepared, ctx):
    files = {Path(desc.MANIFEST_NAME), Path(desc.PENDING_NAME)}
    files.update(Path(component.name) for component in state.components)
    files.update(_planned_rank_files(state, ctx))
    for component, _compute, storage in prepared:
        prefix = Path(component.name)
        saved = desc.describe_component(
            storage.d_tensor_table, storage.physical_layout, ctx.world_size
        )
        files.update(prefix / name for name in saved["files"])
        if saved["files"]:
            fixed = (
                (".metadata",)
                if saved["format"] != "safetensors"
                else (".gdtensor_committed", "model.safetensors.index.json")
            )
            files.update(prefix / name for name in fixed)
        if component.objects:
            files.add(prefix / "objects.json")
        if component.objects and saved["format"] != "safetensors":
            files.add(prefix / "__0_0.distcp")
    return files


def _initialize_transaction(transaction, state, prepared, ctx):
    planned = _planned_files(state, prepared, ctx) if ctx.rank == 0 else ()
    transaction.begin(planned)
    _initialize_save(transaction.stage, state, ctx)


def _save_staged(transaction, state, ctx, policy, backend):
    path = transaction.stage
    prepared = _collective(
        ctx, "prepare save", lambda: _prepare_save(path, state, ctx, policy, backend)
    )
    if type(prepared) is Err:
        return prepared
    initialized = _collective(
        ctx,
        PREFLIGHT,
        lambda: _initialize_transaction(transaction, state, prepared.value, ctx),
    )
    if type(initialized) is Err:
        return initialized
    result = _save_components(path, prepared.value, ctx)
    if type(result) is Err:
        return result
    result = _collective(ctx, "rank state", lambda: _save_rank_state(path, state, ctx))
    if type(result) is Err:
        return result
    return _collective(
        ctx, "checkpoint commit", lambda: _publish(path, state, prepared.value, ctx)
    )


def _transactional_save(path, ctx, action):
    created = _collective(
        ctx, "prepare transaction", lambda: DirectoryTransaction(path, ctx.rank)
    )
    if type(created) is Err:
        return created
    transaction = created.value
    assert transaction is not None
    try:
        result = action(transaction)
        if type(result) is Ok:
            result = _collective(ctx, "checkpoint publish", transaction.publish)
        if type(result) is Err:
            rollback = _collective(ctx, "checkpoint rollback", transaction.rollback)
            result = rollback if type(rollback) is Err else result
        else:
            transaction.accept()
    except BaseException as exc:
        try:
            _collective(ctx, "checkpoint rollback", transaction.rollback)
        finally:
            transaction.close()
        if not isinstance(exc, Exception):
            raise
        return _error("checkpoint transaction", exc)
    try:
        transaction.close()
    except Exception as exc:
        result = _error("checkpoint recovery", exc)
    return result


def save(path, state, ctx, *, policy=None, backend=None) -> Result[None]:
    """Save to a sibling, then publish collectively without damaging the old save."""
    policy = policy or PerRankPolicy()
    backend = backend or SafeTensorsBackend()
    return _transactional_save(
        path,
        ctx,
        lambda transaction: _save_staged(transaction, state, ctx, policy, backend),
    )


def _require_not_pending(path):
    if (Path(path) / desc.PENDING_NAME).exists():
        raise ValueError("GDCP checkpoint is incomplete: save is pending")


def inspect(path) -> Result[desc.CheckpointDesc]:
    """Read only the JSON manifest, without opening tensor or object files."""
    try:
        _require_not_pending(path)
        return Ok(desc.read_manifest(path))
    except Exception as exc:
        return _error("inspect", exc)


def _strict_tensors(component, saved, strict):
    actual = {
        named.global_tensor_id: named for named in component.table.named_d_tensors
    }
    expected = {info["id"]: info for info in saved["tensors"]}
    missing = actual.keys() - expected.keys()
    unexpected = expected.keys() - actual.keys()
    if missing:
        raise ValueError(f"{component.name}: missing key {sorted(missing)[0]}")
    if strict and unexpected:
        raise ValueError(f"{component.name}: unexpected key {sorted(unexpected)[0]}")
    for tensor_id, named in actual.items():
        _strict_tensor(component, tensor_id, named, expected[tensor_id])


def _strict_tensor(component, tensor_id, named, saved):
    meta = global_tensor_meta_of(named)
    logical = component.logical_shapes.get(tensor_id, list(meta.global_shape))
    if list(meta.global_shape) != saved["shape"] or logical != saved.get(
        "logical_shape", saved["shape"]
    ):
        raise ValueError(f"{component.name}: shape mismatch for {tensor_id}")
    if meta.dtype != saved["dtype"] or meta.itemsize != saved["itemsize"]:
        raise ValueError(f"{component.name}: dtype mismatch for {tensor_id}")


def _load_objects(path, component, saved, strict):
    objects = {}
    if saved["objects"] is not None:
        desc._safe_relative(saved["objects"])
        objects = desc.read_objects(Path(path) / component.name / saved["objects"])
    missing = component.objects.keys() - objects.keys()
    unexpected = objects.keys() - component.objects.keys()
    if missing or (strict and unexpected):
        key = sorted(missing or unexpected)[0]
        raise ValueError(f"{component.name}: objects key mismatch for {key}")
    return objects


def _prepare_load(path, state, ctx, backend, strict):
    _validate_context(ctx)
    _validate_state(state)
    if strict is not True:
        raise NotImplementedError(
            "strict=False is not supported; v1 requires strict=True"
        )
    _require_not_pending(path)
    description = desc.read_manifest(path)
    prepared = []
    for component in state.components:
        item = _prepare_load_component(
            path, component, description, ctx, backend, strict
        )
        prepared.append(item)
    rank_state = _load_rank_state(path, state, description, ctx)
    return prepared, rank_state


def _prepare_load_component(path, component, description, ctx, backend, strict):
    if component.name not in description.components:
        raise ValueError(f"missing component {component.name}")
    saved = description.components[component.name]
    _strict_tensors(component, saved, strict)
    objects = _load_objects(path, component, saved, strict)
    compute = _compute_view(component)
    _require(compute.check_constraints())
    table, layout = desc.rebuild_component(saved, description.world_size)
    storage = backend.open(Path(path) / component.name, table, layout, ctx.coordinator)
    _require(storage.check_constraints())
    return component, compute, storage, objects


def _load_rank_state(path, state, description, ctx):
    loaded = {}
    for name in state.rank_state:
        if name not in description.rank_state:
            raise ValueError(f"missing rank state {name}")
        info = description.rank_state[name]
        loaded[name] = _load_rank_blob(path, name, info, ctx.rank)
    return loaded


def _read_rank_blob(path, name, info, rank):
    filename = info["pattern"].format(rank=rank)
    desc._safe_relative(filename)
    return (Path(path) / name / filename).read_bytes()


def _load_rank_blob(path, name, info, rank):
    if rank == 0:
        return _read_rank_blob(path, name, info, 0)
    if rank in info["ranks"]:
        try:
            return _read_rank_blob(path, name, info, rank)
        except FileNotFoundError:
            pass
    _LOG.warning("%s rank %d is absent; loading rank 0's state", name, rank)
    return _read_rank_blob(path, name, info, 0)


def _load_one(ctx, compute, storage):
    if not storage.physical_layout.tensor_entries:
        return Ok(None)
    result = rank_load(ctx.rank, compute, storage)
    if type(result) is Ok:
        getattr(compute.data_accessor, "finish", lambda: None)()
    return result


def _load_components(prepared, ctx):
    for component, compute, storage, objects in prepared:
        result = _collective(
            ctx, f"load {component.name}", lambda: _load_one(ctx, compute, storage)
        )
        if type(result) is Err:
            return result
        component.objects.clear()
        component.objects.update(objects)
    return Ok(None)


def load(path, state, ctx, *, backend=None, strict=True) -> Result[None]:
    """Restore requested components after all strict metadata checks pass."""
    backend = backend or SafeTensorsBackend()
    prepared = _collective(
        ctx, "prepare load", lambda: _prepare_load(path, state, ctx, backend, strict)
    )
    if type(prepared) is Err:
        return prepared
    assert prepared.value is not None
    components, rank_state = prepared.value
    result = _load_components(components, ctx)
    if type(result) is Err:
        return result
    state.rank_state.update(rank_state)
    return _collective(ctx, "load complete", lambda: None)
