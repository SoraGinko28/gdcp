"""Filesystem records for agreements that the boolean coordinator cannot carry."""

import fcntl
import hashlib
import json
import os
import stat
from contextlib import suppress
from pathlib import Path

from . import desc
from ._transaction import _entry_identity, _identity, _sync_directory

_OWNER = ".owner"
_MAGIC = b"GDCP rank agreement v1\n"


def _record_rank(name, prefix):
    if not name.startswith(prefix):
        return None
    try:
        rank = int(name[len(prefix) :])
    except ValueError:
        return None
    return rank if name == f"{prefix}{rank}" else None


def _is_record(entry):
    return (
        _record_rank(entry.name, "rank-") is not None
        or _record_rank(entry.name, "duplicate-") is not None
    ) and stat.S_ISREG(entry.lstat().st_mode)


def _description_error(other, reference):
    rank, (_, components, keys, target) = other
    _, expected_components, expected_keys, expected_target = reference
    if target != expected_target:
        return f"ranks disagree on checkpoint save target (ranks 0 and {rank})"
    if keys != expected_keys:
        key = min(set(keys) ^ set(expected_keys))
        return f"rank_state keys disagree for {key!r} (ranks 0 and {rank})"
    if components != expected_components:
        return f"ranks disagree on checkpoint component sequence (ranks 0 and {rank})"
    return None


def agreement_records(path, *, loading):
    root = Path(os.path.realpath(path))
    if not loading and not root.name:
        raise ValueError("the filesystem root cannot be a checkpoint")
    name = ".gdcp-load-vote" if loading else ".gdcp-save-vote"
    records = AgreementRecords(root.parent / name)
    if loading:
        return records, None, None
    previous = AgreementRecords(root.with_name(f".{root.name}.gdcp-vote"))
    return records, previous, root.name


class AgreementRecords:
    """A reserved directory, held by rank zero for one agreement phase."""

    def __init__(self, path):
        self.path = path
        self.owner = self.path / _OWNER
        self.fd = None
        self.identity = None
        self.created = False
        self.owned = False
        self.busy = False

    def recover(self):
        if not os.path.lexists(self.path):
            return
        try:
            acquired = self.acquire()
        finally:
            self.close()
        if not acquired:
            raise FileExistsError(self.acquisition_error())

    def acquisition_error(self):
        if self.busy:
            return "checkpoint agreement record is already in use"
        return f"unrecognized agreement directory: {self.path}"

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError:
            return self._acquire_existing()
        self.created = True
        self.identity = _identity(self.path)
        fd = self._open_owner(os.O_CREAT | os.O_EXCL)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._verify_owner(fd)
        self._initialize(fd)
        return True

    def _open_owner(self, create=0):
        fd = os.open(
            self.owner,
            os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | create,
            0o666,
        )
        self.fd = fd
        return fd

    def _acquire_existing(self):
        try:
            self.identity = _identity(self.path)
            fd = self._open_owner()
        except FileNotFoundError:
            return False
        # A creator holds the lock before writing the marker. Checking the
        # marker first keeps a contender from locking a half-initialized owner.
        self._verify_owner(fd)
        if os.pread(fd, len(_MAGIC) + 1, 0) != _MAGIC:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.busy = True
            return False
        self._verify_owner(fd)
        if os.pread(fd, len(_MAGIC) + 1, 0) != _MAGIC:
            return False
        self._initialize(fd)
        return True

    def _verify_owner(self, fd):
        entry = os.fstat(fd)
        if (
            not stat.S_ISREG(entry.st_mode)
            or entry.st_nlink != 1
            or _entry_identity(self.owner) != (entry.st_dev, entry.st_ino)
            or _identity(self.path) != self.identity
        ):
            raise OSError(f"agreement directory changed: {self.path}")

    def _records(self):
        records = [entry for entry in self.path.iterdir() if entry.name != _OWNER]
        foreign = next((entry for entry in records if not _is_record(entry)), None)
        if foreign is not None:
            raise FileExistsError(f"unrecognized agreement entry: {foreign}")
        return records

    def _initialize(self, fd):
        records = self._records()
        if self.created and records:
            raise FileExistsError(f"unrecognized agreement directory: {self.path}")
        if self.created and os.write(fd, _MAGIC) != len(_MAGIC):
            raise OSError(f"short write of agreement owner marker: {self.owner}")
        self.owned = True
        for entry in records:
            entry.unlink()
        os.fsync(fd)
        _sync_directory(self.path)
        _sync_directory(self.path.parent)

    def close(self):
        try:
            self._cleanup()
        finally:
            self._close_owner()

    def _close_owner(self):
        fd = self.fd
        self.fd = None
        if fd is not None:
            os.close(fd)

    def _cleanup(self):
        if not (self.created or self.owned):
            return
        try:
            self._remove(self.fd)
        except (KeyboardInterrupt, SystemExit):
            self._retry_cleanup(self.fd)
            raise

    def _retry_cleanup(self, fd):
        with suppress(OSError, ValueError):
            self._remove(fd)

    def _remove(self, fd):
        if not os.path.lexists(self.path):
            _sync_directory(self.path.parent)
            return
        if self.identity is not None and _identity(self.path) != self.identity:
            raise OSError(f"agreement directory changed: {self.path}")
        if fd is not None and os.path.lexists(self.owner):
            self._remove_owner(fd)
        self.path.rmdir()
        _sync_directory(self.path.parent)

    def _remove_owner(self, fd):
        self._verify_owner(fd)
        records = self._records() if self.owned else ()
        for entry in records:
            entry.unlink()
        self.owner.unlink()

    def claim(self, rank, data):
        try:
            file = (self.path / f"rank-{rank}").open("xb")
        except FileExistsError:
            (self.path / f"duplicate-{rank}").touch()
            _sync_directory(self.path)
            return False
        with file:
            file.write(hashlib.sha256(data).hexdigest().encode() + b"\n")
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        _sync_directory(self.path)
        return True

    def _read(self, rank):
        record = (self.path / f"rank-{rank}").read_bytes()
        digest, _, data = record.partition(b"\n")
        if digest != hashlib.sha256(data).hexdigest().encode():
            raise ValueError(f"invalid checkpoint agreement record for rank {rank}")
        return data

    def compare(self, rank, data, claimed):
        if not claimed:
            return False
        own = self._read(rank)
        if data != own:
            raise ValueError(f"agreement record for rank {rank} changed")
        if rank == 0:
            ids, duplicates = self._participants()
            return self._rank_error(ids, duplicates, json.loads(own)[0]) is None
        return self._read(0) == own

    def _participants(self):
        names = [entry.name for entry in self._records()]
        ids = sorted(
            int(name.removeprefix("rank-"))
            for name in names
            if name.startswith("rank-")
        )
        duplicates = [
            int(name.removeprefix("duplicate-"))
            for name in names
            if name.startswith("duplicate-")
        ]
        return ids, duplicates

    @staticmethod
    def _rank_error(ids, duplicates, world):
        if duplicates:
            return f"duplicate rank {min(duplicates)}"
        if world < 1 or ids[0] < 0 or ids[-1] >= world:
            return "invalid rank or world size"
        missing = next(
            (expected for expected, rank in enumerate(ids) if expected != rank),
            len(ids),
        )
        if missing != world:
            return f"missing rank {missing}"
        return None

    def mismatch(self):
        ids, duplicates = self._participants()
        if duplicates:
            return f"duplicate rank {min(duplicates)}"
        reference = json.loads(self._read(0))
        world = reference[0]
        others = [(rank, json.loads(self._read(rank))) for rank in ids if rank != 0]
        if any(world != other[0] for _, other in others):
            return "ranks disagree on world size"
        rank_error = self._rank_error(ids, duplicates, world)
        if rank_error:
            return rank_error
        errors = (_description_error(other, reference) for other in others)
        return next(
            filter(None, errors), "ranks disagree on the checkpoint agreement record"
        )


def agreement_bytes(ctx, components, rank_state, target):
    return desc.json_bytes([ctx.world_size, components, sorted(rank_state), target])
