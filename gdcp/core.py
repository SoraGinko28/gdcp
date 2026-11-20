"""Collective, framework-neutral checkpoint save, load, and inspection."""

import logging
import os
import shutil
import tempfile
from uuid import uuid4
from contextlib import ExitStack
from dataclasses import replace
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from gdtensor import Result
from gdtensor.builder import (
    ComputeDTensorsView,
    DTensorTable,
    Err,
    GDTensorIOError,
    Ok,
    Shard,
    global_tensor_meta_of,
)
from gdtensor.io import rank_load, rank_save
from gdtensor_conn._geometry import _mesh, naive_of

from . import _hdfs, desc
from ._transaction import PREFLIGHT, DirectoryTransaction
from .backend import SafeTensorsBackend, _raise_io_interrupt
from .policy import PerRankPolicy
from .state import RankContext

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
    # Dispatch choices are fixed enums with at most four entries.
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


def _validate_state(state, *, rank_state=None):
    rank_state = state.rank_state if rank_state is None else rank_state
    names = [component.name for component in state.components]
    if len(set(names)) != len(names):
        raise ValueError("duplicate component name")
    for name in names + list(rank_state):
        desc._safe_relative(name)
    for name, data in rank_state.items():
        if not isinstance(data, bytes):
            raise TypeError(f"rank_state {name} must contain bytes")


_AGREEMENT = "checkpoint rank agreement"
_DIGEST_BYTES = 8


def _agreement_error(message):
    return _error(_AGREEMENT, ValueError(message))


def _component_names(state):
    return tuple(component.name for component in state.components)


def _agree(ctx, prepare):
    def describe():
        value, fields = prepare()
        canonical = desc.json_bytes(fields)
        digest = int.from_bytes(sha256(canonical).digest()[:_DIGEST_BYTES], "big")
        return value, fields, canonical, digest

    prepared = _collective(ctx, _AGREEMENT, describe)
    if type(prepared) is Err:
        return prepared
    assert prepared.value is not None
    value, fields, canonical, digest = prepared.value
    coordinator = ctx.coordinator
    if coordinator is None:
        return Ok(value)
    reference = 0
    for bit in range(_DIGEST_BYTES * 8):
        mask = 1 << bit
        if coordinator.vote(ctx.rank != 0 or bool(digest & mask)):
            reference |= mask
    if coordinator.vote(digest == reference):
        return Ok(value)
    _LOG.debug("Checkpoint agreement rank %s: %s", ctx.rank, canonical.decode())
    return _agreement_error(f"ranks disagree on {', '.join(fields)}")


def _backend_format(backend):
    return getattr(backend, "format", "safetensors")


def _load_options(fmt, preflight, validate_groups, *, optional=False, **options):
    return {
        "format": fmt,
        "preflight": preflight is not None,
        "validate_optimizer_groups": validate_groups is not None,
        "optional_rank_state": bool(optional),
        **options,
    }


def _agree_load(state, ctx, format_options):
    def prepare():
        _validate_context(ctx)
        options = format_options()
        return options, {
            "world size": ctx.world_size,
            "requested checkpoint component sequence": _component_names(state),
            "rank_state keys": sorted(state.rank_state),
            "format options": options,
        }

    return _agree(ctx, prepare)


def _read_agreement(stack, path, state, ctx, options_for, *, latest=False):
    local = None

    def options():
        nonlocal local
        local, identity = _hdfs.localize(path, stack, latest=latest)
        return {**options_for(local), **identity}

    agreed = _agree_load(state, ctx, options)
    if type(agreed) is Err:
        return agreed
    assert agreed.value is not None
    return Ok((local, agreed.value))


def _compute_view(component, tensor_ids=None):
    table = component.table
    if tensor_ids is not None:
        entries = {named.global_tensor_id: named for named in table.named_d_tensors}
        table = DTensorTable([entries[key] for key in tensor_ids])
    return ComputeDTensorsView(
        [named.global_tensor_id for named in table.named_d_tensors],
        table,
        component.accessor,
    )


def _prepare_save(path, state, agreement, policy, backend):
    ctx = agreement.ctx
    policy = PerRankPolicy() if policy is None else policy
    backend = SafeTensorsBackend() if backend is None else backend
    prepared = []
    for component in state.components:
        compute = _compute_view(component)
        _require(compute.check_constraints())
        table, layout = policy.plan(component, ctx.world_size)
        agreement.observe()
        storage = backend.open(
            Path(path) / component.name, table, layout, ctx.coordinator
        )
        agreement.observe()
        _require(storage.check_constraints())
        prepared.append((component, compute, storage))
    return prepared


def _initialize_save(path, state, ctx, rank_state):
    if ctx.rank != 0:
        return
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    desc.atomic_write(root / desc.PENDING_NAME, b"GDCP save in progress\n")
    names = [component.name for component in state.components] + list(rank_state)
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


def _save_one(agreement, compute, storage):
    ctx = agreement.ctx
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
        agreement.observe()
        getattr(compute.data_accessor, "finish", lambda: None)()
    return result


def _save_objects(path, component, ctx):
    if ctx.rank == 0 and component.objects:
        desc.write_objects(
            Path(path) / component.name / "objects.json", component.objects
        )


def _save_components(path, prepared, agreement):
    ctx = agreement.ctx
    for component, compute, storage in prepared:
        result = agreement.phase(
            f"save {component.name}", lambda: _save_one(agreement, compute, storage)
        )
        if type(result) is Err:
            return result
        result = agreement.phase(
            f"objects {component.name}",
            lambda: _save_objects(path, component, ctx),
        )
        if type(result) is Err:
            return result
    return Ok(None)


def _rank_state_path(root, name, rank):
    return Path(root) / name / f"{name}_rank_{rank}.pt"


def _save_rank_state(path, rank_state, ctx):
    for name, data in rank_state.items():
        desc.atomic_write(_rank_state_path(path, name, ctx.rank), data)


def _rank_state_inventory(rank_state, ctx):
    return {
        name: {
            "ranks": list(range(ctx.world_size)),
            "pattern": name.replace("{", "{{").replace("}", "}}") + "_rank_{rank}.pt",
        }
        for name in rank_state
    }


def _publish(path, state, prepared, ctx, inventory):
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
        "rank_state": inventory,
    }
    desc.write_manifest(path, manifest)
    (Path(path) / desc.PENDING_NAME).unlink()


def _planned_rank_files(rank_state, ctx):
    return {
        _rank_state_path(".", name, rank)
        for name in rank_state
        for rank in range(ctx.world_size)
    }


def _planned_files(state, prepared, ctx, rank_state):
    files = {Path(desc.MANIFEST_NAME), Path(desc.PENDING_NAME)}
    files.update(Path(component.name) for component in state.components)
    files.update(_planned_rank_files(rank_state, ctx))
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


def _initialize_transaction(transaction, state, prepared, ctx, rank_state):
    planned = _planned_files(state, prepared, ctx, rank_state) if ctx.rank == 0 else ()
    transaction.begin(planned)
    _initialize_save(transaction.stage, state, ctx, rank_state)


def _save_staged(transaction, state, agreement, inventory, policy, backend):
    path = transaction.stage
    ctx, rank_state = agreement.ctx, agreement.rank_state
    prepared = agreement.phase(
        "prepare save", lambda: _prepare_save(path, state, agreement, policy, backend)
    )
    if type(prepared) is Err:
        return prepared
    initialized = agreement.phase(
        PREFLIGHT,
        lambda: _initialize_transaction(
            transaction, state, prepared.value, ctx, rank_state
        ),
    )
    if type(initialized) is Err:
        return initialized
    result = _save_components(path, prepared.value, agreement)
    if type(result) is Err:
        return result
    result = agreement.phase(
        "rank state", lambda: _save_rank_state(path, rank_state, ctx)
    )
    if type(result) is Err:
        return result
    return agreement.phase(
        "checkpoint commit",
        lambda: _publish(path, state, prepared.value, ctx, inventory),
    )


def _save_request(path, ctx, components, rank_keys, format_options):
    _validate_context(ctx)
    snapshot = RankContext(ctx.rank, ctx.world_size, ctx.coordinator)
    keys = tuple(sorted(rank_keys))
    target = _hdfs.checkpoint_path(path)
    if isinstance(target, _hdfs.HdfsPath) and not target.path.name:
        raise ValueError(
            "choose a checkpoint directory below the hdfs:// filesystem root"
        )
    return (snapshot, components, keys), {
        "world size": snapshot.world_size,
        "checkpoint component sequence": components,
        "rank_state keys": keys,
        "resolved checkpoint save target": str(target)
        if isinstance(target, _hdfs.HdfsPath)
        else os.path.realpath(target),
        "format options": format_options(),
    }


def _tensor_schema(state):
    # Record each distinct rank grid once, even for thousands of model tensors.
    meshes = {}

    @cache
    def mesh_index(mesh):
        cells = _mesh(mesh)
        grid = (cells.shape, tuple(cells.ravel().tolist()))
        return meshes.setdefault(grid, len(meshes))

    def describe(named, logical_shapes):
        tensor = naive_of(named)
        meta = tensor.global_tensor_meta
        shape = global_tensor_meta_of(named).global_shape
        return {
            "id": named.global_tensor_id,
            "aliases": sorted(named.alias),
            "shape": meta.global_shape,
            "view": shape,
            "logical_shape": logical_shapes.get(named.global_tensor_id, shape),
            "dtype": meta.dtype,
            "itemsize": meta.itemsize,
            "placements": [
                ("Shard", p.axis) if type(p) is Shard else (type(p).__name__,)
                for p in tensor.placements
            ],
            "mesh": mesh_index(tensor.mesh),
        }

    tensors = [
        [
            describe(named, component.logical_shapes)
            for named in component.table.named_d_tensors
        ]
        for component in state.components
    ]
    return {"components": tensors, "meshes": list(meshes)}


def _prepare_save_agreement(path, state, ctx, format_options):
    rank_state, components = dict(state.rank_state), _component_names(state)
    _validate_state(state, rank_state=rank_state)
    (snapshot, _, _), fields = _save_request(
        path, ctx, components, rank_state, format_options
    )
    fields["ordered tensor schema"] = _tensor_schema(state)
    return (snapshot, rank_state, components), fields


def _validate_rank_state_inventory(path, inventory):
    for name, info in inventory.items():
        _declared_rank_files(path, name, info)


class _SaveAgreement:
    def __init__(self, state, context, ctx, rank_state, components):
        self.state = state
        self.context = context
        self.ctx = ctx
        self.rank_state = rank_state
        self.components = components
        self.configuration_changed = False
        self.keys_changed = False
        self.components_changed = False

    def observe(self):
        # Accessors may run at different times on different ranks. Remember
        # changes locally until all ranks reach the next common phase boundary.
        self.configuration_changed |= (self.ctx.rank, self.ctx.world_size) != (
            self.context.rank,
            self.context.world_size,
        )
        self.keys_changed |= self.rank_state.keys() != self.state.rank_state.keys()
        self.components_changed |= self.components != _component_names(self.state)

    def check(self):
        observed = _collective(self.ctx, "check save agreement", self.observe)
        if type(observed) is Err:
            return observed
        coordinator = self.ctx.coordinator
        same_configuration = not self.configuration_changed
        same_keys = not self.keys_changed
        unchanged = same_configuration and same_keys and not self.components_changed
        if coordinator is not None:
            unchanged = coordinator.vote(unchanged)
        if unchanged:
            return Ok(None)
        if coordinator is not None:
            same_configuration = coordinator.vote(same_configuration)
        if not same_configuration:
            return _agreement_error("rank configuration changed during save")
        if coordinator is not None:
            same_keys = coordinator.vote(same_keys)
        name = "checkpoint component sequence" if same_keys else "rank_state keys"
        return _agreement_error(f"{name} changed during save")

    def phase(self, operation, action):
        try:
            result = _collective(self.ctx, operation, action)
        finally:
            checked = self.check()
        return checked if type(checked) is Err else result


def _agree_save(path, state, context, format_options):
    agreed = _agree(
        context, lambda: _prepare_save_agreement(path, state, context, format_options)
    )
    if type(agreed) is Err:
        return agreed
    assert agreed.value is not None
    ctx, rank_state, components = agreed.value
    return Ok(_SaveAgreement(state, context, ctx, rank_state, components))


def _transactional_save(path, agreement, action):
    target = _hdfs.checkpoint_path(path)
    if isinstance(target, _hdfs.HdfsPath):
        return _hdfs_save(target, agreement, action)
    return _local_transaction(target, agreement, action)


def _hdfs_save(target, agreement, action):
    ctx = agreement.ctx

    # All writers must see the same private POSIX transaction directory.
    # Only rank zero generates its name. Independent saves never share it.
    def prepare():
        base = Path(os.environ.get("GDCP_HDFS_STAGING_DIR") or tempfile.gettempdir())
        client = _hdfs._client(target) if ctx.rank == 0 else None
        return base, uuid4().int if ctx.rank == 0 else 0, client

    prepared = _collective(ctx, "prepare HDFS staging", prepare)
    if type(prepared) is Err:
        return prepared
    assert prepared.value is not None
    base, token, client = prepared.value
    if ctx.coordinator is not None:
        token = sum(
            1 << bit
            for bit in range(128)
            if ctx.coordinator.vote(ctx.rank != 0 or bool(token & (1 << bit)))
        )
    work = base / f".gdcp-hdfs-{token:032x}"
    local = work / "checkpoint"

    def create(stack):
        if ctx.rank == 0:
            work.mkdir(mode=0o700)
            stack.callback(_remove_hdfs_stage, work)

    def visible():
        if not work.is_dir():
            raise ValueError(
                "GDCP_HDFS_STAGING_DIR must be visible to every saving rank"
            )

    with ExitStack() as stack:
        made = _collective(ctx, "create HDFS staging", lambda: create(stack))
        if type(made) is Err:
            return made

        checked = _collective(ctx, "check shared HDFS staging", visible)
        if type(checked) is Err:
            return checked
        result = _local_transaction(local, agreement, action)
        if type(result) is Err:
            return result
        return _collective(
            ctx,
            "publish HDFS checkpoint",
            lambda: _hdfs.publish(client, target, local) if ctx.rank == 0 else None,
        )


def _remove_hdfs_stage(work):
    try:
        shutil.rmtree(work)
    except OSError:
        _LOG.warning(
            "Could not remove local HDFS staging directory %s", work, exc_info=True
        )


def _local_transaction(path, agreement, action):
    ctx, rank_state = agreement.ctx, agreement.rank_state
    inventory = agreement.phase(
        "prepare rank state inventory",
        lambda: _rank_state_inventory(rank_state, ctx),
    )
    if type(inventory) is Err:
        return inventory
    created = agreement.phase(
        "prepare transaction", lambda: DirectoryTransaction(path, ctx.rank)
    )
    if type(created) is Err:
        return created
    transaction = created.value
    assert transaction is not None
    try:
        result = action(transaction, agreement, inventory.value)
        if type(result) is Ok:
            result = agreement.phase(
                "validate rank state inventory",
                lambda: _validate_rank_state_inventory(
                    transaction.stage, inventory.value
                ),
            )
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
    agreed = _agree_save(path, state, ctx, lambda: {"format": _backend_format(backend)})
    if type(agreed) is Err:
        return agreed
    return _save_agreed(path, agreed.value, policy=policy, backend=backend)


def _save_agreed(path, agreement, *, policy=None, backend=None):
    return _transactional_save(
        path,
        agreement,
        lambda transaction, agreement, inventory: _save_staged(
            transaction, agreement.state, agreement, inventory, policy, backend
        ),
    )


def _require_not_pending(path):
    if (desc.checkpoint_path(path) / desc.PENDING_NAME).exists():
        raise ValueError("GDCP checkpoint is incomplete: save is pending")


def inspect(path) -> Result[desc.CheckpointDesc]:
    """Read only the JSON manifest, without opening tensor or object files."""
    try:
        with ExitStack() as stack:
            local, _ = _hdfs.localize(path, stack, manifest_only=True)
            _require_not_pending(local)
            return Ok(desc.read_manifest(local))
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


def _load_objects(path, component, saved, strict, validate_groups=None):
    objects = {}
    if saved["objects"] is not None:
        desc._safe_relative(saved["objects"])
        objects = desc.read_objects(Path(path) / component.name / saved["objects"])
    if validate_groups is not None:
        _validate_load_call(lambda: validate_groups(objects.get("param_groups")))
    missing = component.objects.keys() - objects.keys()
    unexpected = objects.keys() - component.objects.keys()
    if missing or (strict and unexpected):
        key = sorted(missing or unexpected)[0]
        raise ValueError(f"{component.name}: objects key mismatch for {key}")
    return objects


def _prepare_load(path, state, ctx, backend, strict, validate_groups):
    backend = SafeTensorsBackend() if backend is None else backend
    _validate_context(ctx)
    _validate_state(state)
    if strict is not True:
        raise NotImplementedError(
            "strict=False is not supported; v1 requires strict=True"
        )
    _require_not_pending(path)
    description = desc.read_manifest(path, _validate_components=bool(state.components))
    optimizer_objects = None
    if validate_groups is not None and "optimizer" in description.components:
        optimizer = next(c for c in state.components if c.name == "optimizer")
        optimizer_objects = _load_objects(
            path,
            optimizer,
            description.components["optimizer"],
            strict,
            validate_groups,
        )
    prepared = []
    for component in state.components:
        objects = optimizer_objects if component.name == "optimizer" else None
        item = _prepare_load_component(
            path, component, description, ctx, backend, strict, objects
        )
        prepared.append(item)
    return prepared, description


def _prepare_load_component(
    path, component, description, ctx, backend, strict, objects
):
    if component.name not in description.components:
        raise ValueError(f"missing component {component.name}")
    saved = description.components[component.name]
    _strict_tensors(component, saved, strict)
    if objects is None:
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
    # The copy planner requires matching row order in both tables.
    if compute.tensor_ids != storage.tensor_ids:
        compute = _compute_view(component, storage.tensor_ids)
    return component, compute, storage, objects


def _load_rank_state(path, state, inventory, ctx, *, saved_world=None, optional=False):
    files = {
        name: _declared_rank_files(path, name, info) for name, info in inventory.items()
    }
    loaded = {}
    for name in state.rank_state:
        ranks = files.get(name)
        if ranks is None and optional:
            continue
        if ranks is None:
            raise ValueError(f"missing rank state {name}")
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


def _load_payloads(views, ctx, load_rank_state, *, preflight=None, validate=None):
    prepare_rank_state = None
    if preflight is not None:
        checked = _collective(
            ctx, "validate load objects", lambda: _validate_load_call(preflight)
        )
        if type(checked) is Err:
            return checked
        prepare_rank_state = checked.value
    rank_state = _collective(ctx, "load rank state", load_rank_state)
    if type(rank_state) is Err:
        return rank_state

    def validate_payloads():
        if prepare_rank_state is not None:
            prepare_rank_state(rank_state.value)
        if validate is not None:
            validate()

    stack = ExitStack()
    try:
        checked = _collective(
            ctx,
            "validate load",
            lambda: _validate_load(views, ctx, validate_payloads, stack),
        )
        result = (
            checked if type(checked) is Err else _copy_load_payloads(checked.value, ctx)
        )
    finally:
        closed = _collective(ctx, "close load", stack.close)
    if type(result) is Err:
        return result
    return closed if type(closed) is Err else rank_state


def _validate_load_call(validate):
    try:
        return validate()
    except Exception as exc:
        raise _Failure(_LoadValidationError(exc)) from exc


def _validate_load(views, ctx, validate, stack):
    _validate_load_call(validate)
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


def _load_components(prepared, ctx, state, load_rank_state, preflight):
    views = [
        (component.name, compute, storage)
        for component, compute, storage, _ in prepared
    ]
    objects = {component.name: objects for component, _, _, objects in prepared}

    result = _load_payloads(
        views,
        ctx,
        load_rank_state,
        preflight=(lambda: preflight(objects)) if preflight is not None else None,
    )
    if type(result) is Err:
        return result

    def restore():
        _restore_objects(prepared)
        state.rank_state.update(result.value)

    return _collective(ctx, "restore objects", restore)


def _restore_objects(prepared):
    for component, _, _, objects in prepared:
        component.objects.clear()
        component.objects.update(objects)


def load(
    path,
    state,
    ctx,
    *,
    backend=None,
    strict=True,
    _preflight=None,
    _validate_groups=None,
    _optional_rank_state=False,
) -> Result[None]:
    """Restore requested components after all strict metadata checks pass."""
    with ExitStack() as stack:
        agreed = _read_agreement(
            stack,
            path,
            state,
            ctx,
            lambda local: _load_options(
                _backend_format(backend),
                _preflight,
                _validate_groups,
                optional=_optional_rank_state,
                strict=strict is True,
            ),
        )
        if type(agreed) is Err:
            return agreed
        assert agreed.value is not None
        local, _ = agreed.value
        return _load_agreed(
            local,
            state,
            ctx,
            backend=backend,
            strict=strict,
            preflight=_preflight,
            validate_groups=_validate_groups,
            optional_rank_state=_optional_rank_state,
        )


def _load_agreed(
    path,
    state,
    ctx,
    *,
    backend=None,
    strict=True,
    preflight=None,
    validate_groups=None,
    optional_rank_state=False,
):
    prepared = _collective(
        ctx,
        "prepare load",
        lambda: _prepare_load(path, state, ctx, backend, strict, validate_groups),
    )
    if type(prepared) is Err:
        return prepared
    assert prepared.value is not None
    components, description = prepared.value

    def load_rank_state():
        return _load_rank_state(
            path,
            state,
            description.rank_state,
            ctx,
            saved_world=description.world_size,
            optional=optional_rank_state,
        )

    return _load_components(components, ctx, state, load_rank_state, preflight)
