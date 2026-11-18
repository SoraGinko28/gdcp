"""Access live local torch tensors as raw bytes, including BF16 and scalars."""

from math import prod

import torch
from torch.distributed.tensor import DTensor, Shard

from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import Err, InvalidArgumentError


def _local_tensor(tensor):
    return (
        tensor.to_local().detach() if isinstance(tensor, DTensor) else tensor.detach()
    )


def _origin(tensor, rank, world):
    if not isinstance(tensor, DTensor) or not isinstance(tensor.placements[0], Shard):
        return 0
    # This translates the framework's contiguous local buffer, not a copy plan.
    rows = tensor.shape[0]
    first = min(rank * ((rows + world - 1) // world), rows)
    return first * prod(tensor.shape[1:]) * tensor.element_size()


def _finish_write(key, tensor, host, written):
    if written != tensor.numel() * tensor.element_size():
        raise ValueError(f"{key}: incomplete local tensor load")
    if host.data_ptr() != tensor.data_ptr():
        with torch.no_grad():
            tensor.copy_(host)


class TorchAccessor(DTensorDataAccessor):
    """Stage at most one local tensor, independent of its number of I/O ranges.

    GDTensor's rank plans visit each compute tensor consecutively. On save we
    take one host snapshot; on load a fresh host buffer receives all ranges and
    is copied back once. ``finish`` ends that operation and releases the final
    staging buffer. CPU contiguous tensors need no staging allocation at all.
    """

    def __init__(self, tensors, rank, world_size):
        self.tensors = tensors
        self.origins = {
            key: _origin(value, rank, world_size) for key, value in tensors.items()
        }
        self._key = None
        self._tensor = None
        self._host = None
        self._mode = None
        self._written = 0
        self._finished = set()

    def _release(self):
        if self._key is None:
            return
        if self._mode == "write" and self._written:
            _finish_write(self._key, self._tensor, self._host, self._written)
        self._finished.add(self._key)
        self._key = self._tensor = self._host = self._mode = None
        self._written = 0

    def finish(self):
        """Finish a successful operation and release its last staging tensor."""
        self._release()
        self._finished.clear()

    def _stage(self, key, mode):
        if key == self._key and mode != self._mode:
            raise ValueError("cannot mix save and load on one staging tensor")
        if key == self._key:
            return
        self._release()
        if key in self._finished:
            raise ValueError(f"{key}: tensor ranges must be consecutive")
        tensor = _local_tensor(self.tensors[key])
        if tensor.device.type == "cpu" and tensor.is_contiguous():
            host = tensor
        elif mode == "read":
            host = tensor.contiguous().cpu()
        else:
            host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
        self._key, self._tensor, self._host, self._mode = key, tensor, host, mode

    def _buffer(self, rng, mode):
        self._stage(rng.global_tensor_id, mode)
        assert self._host is not None
        data = self._host.reshape(-1).view(torch.uint8)
        offset = rng.global_tensor_offset - self.origins[rng.global_tensor_id]
        if offset < 0 or offset + rng.size > data.numel():
            raise ValueError(
                f"{rng.global_tensor_id}: range is outside the local tensor"
            )
        return data[offset : offset + rng.size]

    def read_to_buffer(self, global_range, read_to_buffer):
        try:
            data = self._buffer(global_range, "read")
            return read_to_buffer(data.data_ptr(), data.numel())
        except Exception as exc:
            return Err(InvalidArgumentError("global_range", str(exc)))

    def write_from_buffer(self, global_range, write_from_buffer):
        try:
            data = self._buffer(global_range, "write")
            result = write_from_buffer(data.data_ptr(), data.numel())
            self._written += 0 if type(result) is Err else data.numel()
            return result
        except Exception as exc:
            return Err(InvalidArgumentError("global_range", str(exc)))
