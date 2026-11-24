"""Connector-backed storage views; file formats remain in gdtensor_conn."""

import json
import struct
from dataclasses import dataclass
from functools import partial

from gdtensor import Result
from gdtensor.builder import Err, GDTensorIOError, Ok, StorageDTensorsView
from gdtensor.session import StorageSaveSession
from gdtensor_conn._layout import build_payload_index
from gdtensor_conn._result import catch, invalid, io_error
from gdtensor_conn.access import StorageDataAccessor
from gdtensor_conn.formats import (
    SafeTensorsLoadSession,
    SafeTensorsSaveSession,
    _safetensors_slot_offset,
)
from gdtensor_conn.iobackend import FileIO


class _InterruptIO(FileIO):
    """Remember interrupts that connector commit converts to an error Result."""

    def __init__(self, delegate):
        self.interrupt: BaseException | None = None
        self.tracer = delegate.tracer
        self.fds = delegate.fds
        # Payload access stays bound directly to the original/custom backend.
        self.pwrite = delegate.pwrite
        self.pread = delegate.pread
        self.identity = delegate.identity
        self.read_all = delegate.read_all
        self.create_staging = delegate.create_staging
        self.close_rank = delegate.close_rank
        self.discard = delegate.discard
        self.sync = delegate.sync
        self.publish = delegate.publish
        self.write_all = delegate.write_all


def _remember_io_interrupt(io: FileIO, exc: BaseException) -> None:
    if isinstance(exc, Exception):
        return
    assert isinstance(io, _InterruptIO)
    if io.interrupt is None:
        io.interrupt = exc


def _raise_io_interrupt(storage):
    io = getattr(storage.data_accessor, "io", None)
    if isinstance(io, _InterruptIO) and io.interrupt is not None:
        raise io.interrupt


class _LocalSafeTensorsLoadSession(SafeTensorsLoadSession):
    """Validate selected offsets, then seal the session before payload reads."""

    def __init__(self, *args, **kwargs):
        self._preflight_slots = None
        self._sealed = False
        super().__init__(*args, **kwargs)

    def _prepare_offsets(self, positions):
        self._preflight_slots = {}
        for pos in positions:
            path, slot = self.index.slots[pos]
            self._preflight_slots.setdefault(path, []).append(slot)

    def _read_file_offsets(self, path):
        if self._sealed:
            raise invalid(
                "file_offset", f"{path}: offset was not admitted by preflight"
            )
        full = self.abspath(path)
        try:
            length = struct.unpack("<Q", self.io.pread(self.rank, full, 0, 8))[0]
            header = json.loads(self.io.pread(self.rank, full, 8, length))
        except OSError as exc:
            raise io_error("read", str(exc), full) from None
        # Whole-tensor headers address the first saved slot, even when this
        # rank only reads a later slot of the same shared file.
        first = {}
        for slot in self.index.files[path].slots:
            first.setdefault(slot.tensor_order, slot.contiguous_payload_offset)
        slots = (
            self.index.files[path].slots
            if self._preflight_slots is None
            else self._preflight_slots.get(path, ())
        )
        for slot in slots:
            self.offsets[slot.pos] = (
                8
                + length
                + _safetensors_slot_offset(self.table, slot, header, first, path, full)
            )

    def _finish_preflight(self, positions):
        self.offsets = {pos: self.offsets[pos] for pos in positions}
        self._preflight_slots = None
        self._sealed = True


class _CoordinatedSaveSession(SafeTensorsSaveSession):
    def _attempt(self, action):
        try:
            action()
        except BaseException as exc:
            _remember_io_interrupt(self.io, exc)
            return exc
        return None

    def _close(self):
        self.io.close_rank(self.rank, self.root)

    @catch
    def commit(self):
        if self.state != "open":
            raise invalid("session", f"commit on {self.state} session")
        error = self._attempt(self._close)
        if not self._vote(error is None):
            self.state = "aborted"
            self._discard()
            raise io_error("commit", str(error or "another rank aborted"))
        self.state = "committing"
        error = self._attempt(self._publish)
        published = self._vote(error is None)
        if published:
            error = self._attempt(self._after_barrier)
        if not self._vote(published and error is None):
            self.state = "unknown"
            raise io_error(
                "commit",
                f"commit outcome unknown: {error or 'another rank failed to publish'}",
            )
        self.state = "committed"

    @catch
    def abort(self):
        if self.state in ("committed", "aborted"):
            raise invalid("session", f"abort on {self.state} session")
        error = self._attempt(self._close)
        if self.state != "unknown":
            self._vote(False)
            self._discard()
        self.state = "aborted"
        if error is not None:
            raise io_error("abort", str(error))


def _open_error(message):
    return Err(GDTensorIOError(operation="begin_save", msg=message))


def _abort_opened(result: Result[StorageSaveSession], coordinator):
    if type(result) is Ok:
        session = result.value
        assert session is not None
        session.abort()
    else:
        coordinator.vote(False)


def _collective_begin(begin, coordinator):
    def wrapped(view, rank):
        interrupt = None
        try:
            result = begin(view, rank)
        except BaseException as exc:
            result = _open_error(str(exc))
            interrupt = exc if not isinstance(exc, Exception) else None
        if coordinator.vote(type(result) is Ok):
            return result
        _abort_opened(result, coordinator)
        if interrupt is not None:
            raise interrupt
        return result if type(result) is Err else _open_error("another rank failed")

    return wrapped


@catch
def _begin_session(session_type, root, index, io, view, rank, **options):
    try:
        return session_type(
            rank, root, view.d_tensor_table, view.physical_layout, index, io, **options
        )
    except BaseException:
        io.close_rank(rank, root)
        raise


@dataclass
class SafeTensorsBackend:
    """Thin POSIX safetensors backend, with an optional connector FileIO."""

    io: FileIO | None = None
    format: str = "safetensors"

    def open(self, root, storage_table, layout, coordinator=None):
        if layout.container_format != self.format:
            raise ValueError(f"unsupported format {layout.container_format!r}")
        ids = [named.global_tensor_id for named in storage_table.named_d_tensors]
        io = _InterruptIO(self.io or FileIO())
        index = build_payload_index(storage_table, layout)
        begin_save = partial(
            _begin_session,
            _CoordinatedSaveSession,
            str(root),
            index,
            io,
            coord=coordinator,
        )
        return StorageDTensorsView(
            ids,
            storage_table,
            StorageDataAccessor(index, io),
            layout,
            _collective_begin(begin_save, coordinator) if coordinator else begin_save,
            partial(
                _begin_session,
                _LocalSafeTensorsLoadSession,
                str(root),
                index,
                io,
                eager_headers=False,
            ),
        )
