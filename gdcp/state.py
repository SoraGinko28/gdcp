"""Framework-neutral checkpoint values and extension protocols."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import DTensorTable, PhysicalLayoutTable, StorageDTensorsView


class Coordinator(Protocol):
    def vote(self, ok: bool) -> bool: ...

    def wait(self) -> None: ...


@dataclass
class Component:
    name: str
    table: DTensorTable
    accessor: DTensorDataAccessor
    objects: dict[str, Any] = field(default_factory=dict)
    # Scalars use a one-element physical tensor because the core requires
    # nonempty shapes. This retains the framework's logical shape.
    logical_shapes: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class CheckpointState:
    components: list[Component] = field(default_factory=list)
    rank_state: dict[str, bytes] = field(default_factory=dict)
    framework: str = "fsdp2"
    global_steps: int | None = None


@dataclass
class RankContext:
    rank: int
    world_size: int
    coordinator: Coordinator | None = None
    # Group-local rank -> global rank. None denotes the default world.
    global_ranks: tuple[int, ...] | None = None

    @property
    def global_rank(self):
        return self.rank if self.global_ranks is None else self.global_ranks[self.rank]


class StoragePolicy(Protocol):
    def plan(
        self, component: Component, world_size: int
    ) -> tuple[DTensorTable, PhysicalLayoutTable]: ...


class Backend(Protocol):
    def open(
        self, root, storage_table, layout, coordinator=None
    ) -> StorageDTensorsView: ...


class Adapter(Protocol):
    def describe(self, checkpoint_state, ctx: RankContext) -> CheckpointState: ...

    def restore(self, checkpoint_state, loaded: CheckpointState) -> None: ...
