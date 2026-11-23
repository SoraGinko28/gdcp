"""Storage-backed test accessors with no TorchAccessor-specific attributes."""

import ctypes

import torch
from torch.distributed.tensor import DTensor

from gdtensor.access import DTensorDataAccessor


def local_tensor(value):
    return value.to_local() if isinstance(value, DTensor) else value


def values_for(key, tensor):
    """Deterministic bytes supplied independently of a tensor's storage."""
    local = local_tensor(tensor)
    if local.ndim == 0 and local.device.type == "cpu":
        return local.detach().clone()
    return (
        torch.arange(local.numel(), dtype=torch.float64)
        .add(sum(key.encode()) % 17)
        .to(local.dtype)
        .reshape(local.shape)
    )


class MemoryAccessor(DTensorDataAccessor):
    """Exercise only the public GDTensor callback interface, without finish()."""

    def __init__(self, request, storage):
        self.request = request
        self.storage = storage
        self.reads = []
        self.writes = []

    def _buffer(self, rng):
        data = self.storage[rng.global_tensor_id].reshape(-1).view(torch.uint8)
        offset = self.request.local_offset(
            rng.global_tensor_id, rng.global_tensor_offset, rng.size
        )
        assert 0 <= offset <= data.numel() - rng.size
        return data[offset : offset + rng.size]

    def read_to_buffer(self, global_range, read_to_buffer):
        data = self._buffer(global_range)
        self.reads.append(global_range)
        return read_to_buffer(data.data_ptr(), data.numel())

    def write_from_buffer(self, global_range, write_from_buffer):
        data = self._buffer(global_range)
        self.writes.append(global_range)
        result = write_from_buffer(data.data_ptr(), data.numel())
        # Scalar optimizer counters remain readable CPU tensors even when the
        # parameter and its non-scalar state live elsewhere.
        target = self.request.tensors[global_range.global_tensor_id]
        if target.ndim == 0 and target.device.type == "cpu":
            ctypes.memmove(target.data_ptr(), data.data_ptr(), data.numel())
        return result


class MemoryFactory:
    def __init__(self):
        self.accessors = []

    def __call__(self, request):
        storage = {
            key: values_for(key, value)
            if request.mode == "save"
            else torch.zeros_like(local_tensor(value), device="cpu")
            for key, value in request.tensors.items()
        }
        accessor = MemoryAccessor(request, storage)
        self.accessors.append(accessor)
        return accessor
