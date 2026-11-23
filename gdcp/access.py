"""Public FSDP2 data-accessor preparation contract."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class DataAccessorRequest:
    """One component's rank-local, device-independent byte mapping.

    ``tensors`` maps checkpoint keys to this rank's tensors. ``ownership``
    contains global byte intervals ``(start, end)`` in packed local order,
    including replicated tensors. ``local_offset(key, global_offset, size)``
    translates a range contained in one owned interval to a local byte offset.
    The factory returns a ``gdtensor.access.DTensorDataAccessor``. Its callbacks
    must remain valid for the duration of each synchronous callback invocation.
    """

    component: Literal["model", "optimizer"]
    mode: Literal["save", "load"]
    tensors: Mapping[str, Any]
    ownership: Mapping[str, tuple[tuple[int, int], ...]]
    local_offset: Callable[[str, int, int], int]
