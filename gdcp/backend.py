"""Connector-backed storage views; file formats remain in gdtensor_conn."""

from dataclasses import dataclass, replace

from gdtensor import Result
from gdtensor.builder import Err, GDTensorIOError, Ok
from gdtensor.session import StorageSaveSession
from gdtensor_conn.formats import make_storage_view
from gdtensor_conn.iobackend import FileIO


class _InterruptIO(FileIO):
    """Remember interrupts that connector commit converts to an error Result."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.interrupt = None
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

    def remember(self, action):
        try:
            return action()
        except BaseException as exc:
            self.interrupt = self.interrupt or (
                exc if not isinstance(exc, Exception) else None
            )
            raise

    def sync(self, path):
        return self.remember(lambda: self.delegate.sync(path))

    def publish(self, temporary_path, final_path):
        return self.remember(lambda: self.delegate.publish(temporary_path, final_path))

    def write_all(self, rank, path, data):
        return self.remember(lambda: self.delegate.write_all(rank, path, data))


def _raise_io_interrupt(storage):
    io = getattr(storage.data_accessor, "io", None)
    if isinstance(io, _InterruptIO) and io.interrupt is not None:
        raise io.interrupt


def _remember_io_interrupt(io, action):
    return io.remember(action) if isinstance(io, _InterruptIO) else action()


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


@dataclass
class SafeTensorsBackend:
    """Thin POSIX safetensors backend, with an optional connector FileIO."""

    io: FileIO | None = None
    format: str = "safetensors"

    def open(self, root, storage_table, layout, coordinator=None):
        if layout.container_format != self.format:
            raise ValueError(f"unsupported format {layout.container_format!r}")
        ids = [named.global_tensor_id for named in storage_table.named_d_tensors]
        view = make_storage_view(
            ids,
            storage_table,
            layout,
            str(root),
            io=_InterruptIO(self.io or FileIO()),
            coord=coordinator,
        )
        if coordinator is None:
            return view
        begin_save = view.begin_save
        if begin_save is None:
            raise ValueError("storage view has no begin_save callback")
        return replace(view, begin_save=_collective_begin(begin_save, coordinator))
