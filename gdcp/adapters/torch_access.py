"""Access live local torch tensors as raw bytes, including BF16 and scalars."""

from bisect import bisect_right
from itertools import accumulate
from math import prod

import torch
from torch.distributed.tensor import DTensor

from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import Err, InvalidArgumentError


def _local_tensor(tensor):
    return (
        tensor.to_local().detach() if isinstance(tensor, DTensor) else tensor.detach()
    )


def _shard_spans(tensor, box):
    """Byte stripes, outermost first, for a packed local rectangular shard."""
    if box is None or tensor.ndim == 0:
        return ()
    lo, hi = box
    shape = tensor.shape
    if tuple(end - start for start, end in zip(lo, hi, strict=True)) != tuple(
        tensor.to_local().shape
    ):
        raise ValueError("planned shard shape differs from the local DTensor")
    spans = []
    for dim, (length, start, end) in enumerate(zip(shape, lo, hi, strict=True)):
        if end - start != length:
            stride = prod(shape[dim + 1 :]) * tensor.element_size()
            spans.append((length * stride, start * stride, (end - start) * stride))
    return tuple(spans)


def _local_offset(spans, offset, size):
    for period, start, width in spans:
        block, within = divmod(offset, period)
        relative = within - start
        if relative < 0 or relative + size > width:
            raise ValueError("range is outside the local shard")
        offset = block * width + relative
    return offset


def _finish_write(key, tensor, host, written):
    if written != tensor.numel() * tensor.element_size():
        raise ValueError(f"{key}: incomplete local tensor load")
    if host.data_ptr() != tensor.data_ptr():
        with torch.no_grad():
            tensor.copy_(host)


def _byte_runs(key, tensor, owned):
    width = tensor.element_size()
    starts = [int(start) * width for start, _ in owned]
    ends = [int(end) * width for _, end in owned]
    prefix = list(
        accumulate((end - start for start, end in zip(starts, ends)), initial=0)
    )
    if prefix[-1] != _local_tensor(tensor).numel() * width:
        raise ValueError(f"{key}: planned runs differ from the local DTensor shape")
    return starts, ends, prefix


def _run_offset(runs, offset, size):
    starts, ends, prefix = runs
    index = bisect_right(starts, offset) - 1
    if index < 0 or offset + size > ends[index]:
        raise ValueError("range is outside the local shard")
    return prefix[index] + offset - starts[index]


class TorchAccessor(DTensorDataAccessor):
    """Stage at most one local tensor, independent of its number of I/O ranges.

    GDTensor's rank plans visit each compute tensor consecutively. On save we
    take one host snapshot; on load a fresh host buffer receives all ranges and
    is copied back once. ``finish`` ends that operation and releases the final
    staging buffer. CPU contiguous tensors need no staging allocation at all.
    """

    def __init__(self, tensors, boxes, intervals=None):
        self.tensors = tensors
        self.boxes = boxes
        self.intervals = intervals or {}
        self.spans = {
            key: _shard_spans(value, boxes[key])
            if isinstance(value, DTensor) and key in boxes
            else ()
            for key, value in tensors.items()
        }
        self.runs = {
            key: _byte_runs(key, tensors[key], owned)
            for key, owned in (intervals or {}).items()
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
        if (runs := self.runs.get(rng.global_tensor_id)) is not None:
            offset = _run_offset(runs, rng.global_tensor_offset, rng.size)
        else:
            offset = _local_offset(
                self.spans[rng.global_tensor_id], rng.global_tensor_offset, rng.size
            )
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
