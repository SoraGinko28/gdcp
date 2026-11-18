"""Collective, framework-neutral checkpoint save, load, and inspection."""

import json
import logging
from contextlib import ExitStack
from dataclasses import replace
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


class _LoadValidationError(GDTensorIOError):
    def __init__(self, cause: Exception):
        super().__init__(operation="validate load", msg=str(cause))
        self.cause = cause


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


def _collective_choice(ctx, operation, action, choices):
    selected = _collective(ctx, operation, action)
    if type(selected) is Err:
        return selected
    return _agree_choice(ctx, operation, selected.value, choices)


def _agree_choice(ctx, operation, selected, choices):
    if ctx.coordinator is None:
        return Ok(selected)
    for choice in choices:
        if ctx.coordinator.vote(selected == choice):
            return Ok(selected)
    return _error(operation, ValueError(f"ranks disagree on {operation}"))


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


def _broadcast_names(ctx, operation, source, names):
    coordinator = ctx.coordinator
    assert coordinator is not None
    prepared = _collective(
        ctx,
        operation,
        lambda: json.dumps(names()).encode() if ctx.rank == source else b"",
    )
    if type(prepared) is Err:
        return prepared
    data = prepared.value
    received = bytearray()
    # The coordinator's boolean AND can broadcast one bit at a time from source.
    while coordinator.vote(ctx.rank != source or len(received) < len(data)):
        byte = data[len(received)] if ctx.rank == source else 0
        received.append(
            sum(
                1 << bit
                for bit in range(8)
                if not coordinator.vote(not (byte >> bit & 1))
            )
        )
    return _collective(ctx, operation, lambda: json.loads(received))


def _agree_rank_state(state, ctx):
    coordinator = ctx.coordinator
    if ctx.world_size == 1 or coordinator is None:
        return Ok(None)
    operation = "agree on rank state"
    names = _collective(ctx, operation, lambda: set(state.rank_state))
    if type(names) is Err or coordinator.vote(not names.value):
        return names
    reference = _broadcast_names(ctx, operation, 0, lambda: sorted(names.value))
    if type(reference) is Err:
        return reference
    difference = _collective(ctx, operation, lambda: names.value ^ set(reference.value))
    if type(difference) is Err or coordinator.vote(not difference.value):
        return difference
    for source in range(1, ctx.world_size):
        if not coordinator.vote(ctx.rank != source or not difference.value):
            break
    mismatch = _broadcast_names(ctx, operation, source, lambda: [min(difference.value)])
    if type(mismatch) is Err:
        return mismatch
    (key,) = mismatch.value
    return _error(
        operation,
        ValueError(f"rank_state keys disagree for {key!r} (ranks 0 and {source})"),
    )


def _agree_components(state, ctx):
    coordinator = ctx.coordinator
    if ctx.world_size == 1 or coordinator is None:
        return Ok(None)
    operation = "agree on checkpoint components"
    names = _collective(
        ctx, operation, lambda: [item.name for item in state.components]
    )
    if type(names) is Err or coordinator.vote(not names.value):
        return names
    reference = _broadcast_names(ctx, operation, 0, lambda: names.value)
    if type(reference) is Err:
        return reference

    def compare():
        if names.value != reference.value:
            raise ValueError(
                f"ranks disagree on checkpoint component sequence (rank 0: "
                f"{reference.value!r}, rank {ctx.rank}: {names.value!r})"
            )

    return _collective(ctx, operation, compare)


def _compute_view(component):
    return ComputeDTensorsView(
        [named.global_tensor_id for named in component.table.named_d_tensors],
        component.table,
        component.accessor,
    )


def _prepare_save(path, state, ctx, policy, backend):
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
    names = [component.name for component in state.components] + list(state.rank_state)
    _initialize_directories(root, names)


def _initialize_directories(root, names):
    # Check the staging filesystem itself for case and Unicode aliases. A
    # distinct directory for each spelling also keeps its rank files distinct.
    seen = {}
    for name in dict.fromkeys(names):
        directory = root / name
        directory.mkdir(exist_ok=True)
        info = directory.stat()
        identity = info.st_dev, info.st_ino
        if identity in seen:
            raise ValueError(
                f"checkpoint directories {seen[identity]!r} and {name!r} alias"
            )
        seen[identity] = name


def _rank_save(ctx, compute, storage):
    coordinator = ctx.coordinator
    if coordinator is None:
        return rank_save(ctx.rank, compute, storage)
    begin_save = storage.begin_save
    entered = False

    def begin(view, rank):
        nonlocal entered
        entered = True
        if not coordinator.vote(True):
            return _error(
                "save plan", ValueError("another rank failed to plan its save")
            )
        return begin_save(view, rank)

    try:
        assert begin_save is not None
        return rank_save(ctx.rank, compute, replace(storage, begin_save=begin))
    finally:
        # A rank can fail to build or validate its plan before begin_save.
        if not entered:
            coordinator.vote(False)


def _save_one(ctx, compute, storage):
    # The connector has no session leader for a completely empty component.
    # Its semantic tensor entries still live in GDCP's manifest.
    has_payload = _collective_choice(
        ctx,
        "component payload presence",
        lambda: bool(storage.physical_layout.tensor_entries),
        (True, False),
    )
    if type(has_payload) is Err:
        return has_payload
    if not has_payload.value:
        return Ok(None)
    result = _rank_save(ctx, compute, storage)
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


def _rank_state_inventory(state, ctx):
    return {
        name: {
            "ranks": list(range(ctx.world_size)),
            "pattern": name.replace("{", "{{").replace("}", "}}") + "_rank_{rank}.pt",
        }
        for name in state.rank_state
    }


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
        "rank_state": _rank_state_inventory(state, ctx),
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


def _validate_save(state, ctx):
    _validate_context(ctx)
    _validate_state(state)


def _transactional_save(path, state, ctx, action):
    validated = _collective(ctx, "validate save", lambda: _validate_save(state, ctx))
    if type(validated) is Err:
        return validated
    agreed = _agree_rank_state(state, ctx)
    if type(agreed) is Err:
        return agreed
    agreed = _agree_components(state, ctx)
    if type(agreed) is Err:
        return agreed
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
    except BaseException as exc:
        _collective(ctx, "checkpoint rollback", transaction.rollback)
        if not isinstance(exc, Exception):
            raise
        return _error("checkpoint transaction", exc)
    else:
        if type(result) is Err:
            rollback = _collective(ctx, "checkpoint rollback", transaction.rollback)
            result = rollback if type(rollback) is Err else result
        else:
            transaction.accept()
    finally:
        closed = _collective(ctx, "checkpoint recovery", transaction.close)
    return closed if type(result) is Ok else result


def save(path, state, ctx, *, policy=None, backend=None) -> Result[None]:
    """Save to a sibling, then publish collectively without damaging the old save."""
    policy = policy or PerRankPolicy()
    backend = backend or SafeTensorsBackend()
    return _transactional_save(
        path,
        state,
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
    if sorted(named.alias) != sorted(saved.get("aliases", [])):
        raise _Failure(
            _LoadValidationError(
                ValueError(
                    f"{component.name}: tied parameter aliases differ for {tensor_id}"
                )
            )
        )


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
    rank_state = _load_rank_state(
        path, state, description.rank_state, ctx, saved_world=description.world_size
    )
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
    root = Path(path) / component.name
    for name, info in saved["files"].items():
        size = info.get("bytes")
        if size and (root / name).stat().st_size != size:
            raise ValueError(f"{component.name}: file size mismatch for {name}")
    storage = backend.open(root, table, layout, ctx.coordinator)
    _require(storage.check_constraints())
    return component, compute, storage, objects


def _load_rank_state(path, state, inventory, ctx, *, saved_world=None):
    files = {
        name: _declared_rank_files(path, name, info) for name, info in inventory.items()
    }
    loaded = {}
    for name in state.rank_state:
        if name not in files:
            raise ValueError(f"missing rank state {name}")
        ranks = files[name]
        world = saved_world if saved_world is not None else max(ranks, default=-1) + 1
        loaded[name] = _load_rank_blob(name, ranks, ctx.rank, world)
    return loaded


def _declared_rank_files(path, name, info):
    desc._safe_relative(name)
    files = {}
    for rank in info["ranks"]:
        filename = info["pattern"].format(rank=rank)
        desc._safe_relative(filename)
        file = Path(path) / name / filename
        if not file.is_file():
            raise FileNotFoundError(f"missing declared rank state file {file}")
        files[rank] = file
    return files


def _load_rank_blob(name, files, rank, saved_world):
    if rank in files:
        return files[rank].read_bytes()
    if rank < saved_world or 0 not in files:
        raise ValueError(f"rank state {name} has no declaration for rank {rank}")
    _LOG.warning("%s rank %d is absent; loading rank 0's state", name, rank)
    return files[0].read_bytes()


def _load_one(ctx, compute, storage):
    if not storage.physical_layout.tensor_entries:
        return Ok(None)
    result = rank_load(ctx.rank, compute, storage)
    _raise_io_interrupt(storage)
    if type(result) is Ok:
        getattr(compute.data_accessor, "finish", lambda: None)()
    return result


def _prepare_load_view(ctx, compute, storage, stack):
    # The public executor validates the rank plan before calling begin_load.
    # Stop there on the first pass, then resolve headers without copying bytes.
    planned = GDTensorIOError(operation="prepare load", msg="rank plan validated")
    probe = replace(storage, begin_load=lambda view, rank: Err(planned))
    result = rank_load(ctx.rank, compute, probe)
    if type(result) is not Err or result.value is not planned:
        _require(result)
        raise RuntimeError("rank load skipped plan validation")
    if not storage.physical_layout.tensor_entries:
        return storage
    assert storage.begin_load is not None
    session = _require(storage.begin_load(storage, ctx.rank))
    unused = session

    def close_unused():
        if unused is not None:
            _require(unused.finish())

    def begin_load(view, rank):
        nonlocal unused
        unused = None
        return Ok(session)

    stack.callback(close_unused)
    for pos in storage.physical_layout.tensor_entries:
        _require(session.get_file_offset(pos))
    return replace(storage, begin_load=begin_load)


def _load_payloads(views, ctx, validate):
    stack = ExitStack()
    try:
        checked = _collective(
            ctx, "validate load", lambda: _validate_load(views, ctx, validate, stack)
        )
        result = (
            checked if type(checked) is Err else _copy_load_payloads(checked.value, ctx)
        )
    finally:
        closed = _collective(ctx, "close load", stack.close)
    return closed if type(result) is Ok else result


def _validate_load(views, ctx, validate, stack):
    try:
        validate()
    except Exception as exc:
        raise _Failure(_LoadValidationError(exc)) from exc
    return [
        (name, compute, _prepare_load_view(ctx, compute, storage, stack))
        for name, compute, storage in views
    ]


def _copy_load_payloads(views, ctx):
    for name, compute, storage in views:
        result = _collective(
            ctx, f"load {name}", lambda: _load_one(ctx, compute, storage)
        )
        if type(result) is Err:
            return result
    return Ok(None)


def _load_components(prepared, ctx, rank_state, preflight):
    views = [
        (component.name, compute, storage)
        for component, compute, storage, _ in prepared
    ]
    objects_by_name = {component.name: objects for component, _, _, objects in prepared}

    def validate():
        if preflight is not None:
            preflight(objects_by_name, rank_state)

    result = _load_payloads(views, ctx, validate)
    if type(result) is Err:
        return result
    return _collective(ctx, "restore objects", lambda: _restore_objects(prepared))


def _restore_objects(prepared):
    for component, _, _, objects in prepared:
        component.objects.clear()
        component.objects.update(objects)


def load(
    path, state, ctx, *, backend=None, strict=True, _preflight=None
) -> Result[None]:
    """Restore requested components after all strict metadata checks pass."""
    agreed = _agree_components(state, ctx)
    if type(agreed) is Err:
        return agreed
    backend = backend or SafeTensorsBackend()
    prepared = _collective(
        ctx, "prepare load", lambda: _prepare_load(path, state, ctx, backend, strict)
    )
    if type(prepared) is Err:
        return prepared
    assert prepared.value is not None
    components, rank_state = prepared.value
    result = _load_components(components, ctx, rank_state, _preflight)
    if type(result) is Err:
        return result
    return _collective(
        ctx, "load complete", lambda: state.rank_state.update(rank_state)
    )
