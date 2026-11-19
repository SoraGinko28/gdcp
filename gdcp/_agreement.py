"""Filesystem records for agreements that the boolean coordinator cannot carry."""

import errno
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
    rank, (_, components, keys) = other
    _, expected_components, expected_keys = reference
    if keys != expected_keys:
        key = min(set(keys) ^ set(expected_keys))
        return f"rank_state keys disagree for {key!r} (ranks 0 and {rank})"
    if components != expected_components:
        return f"ranks disagree on checkpoint component sequence (ranks 0 and {rank})"
    return None


class AgreementRecords:
    """A reserved sibling, held by rank zero for one agreement phase."""

    def __init__(self, path, *, loading):
        root = Path(os.path.realpath(path))
        if not loading and not root.name:
            raise ValueError("the filesystem root cannot be a checkpoint")
        name = ".gdcp-load-vote" if loading else f".{root.name}.gdcp-vote"
        self.path = root.parent / name
        self.owner = self.path / _OWNER
        self.fd = None
        self.identity = None
        self.owned = False

    def recover(self):
        if not os.path.lexists(self.path):
            return
        try:
            acquired = self.acquire()
        finally:
            self.close()
        if not acquired:
            raise FileExistsError("checkpoint agreement record is already in use")

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.mkdir(exist_ok=True)
        identity = _identity(self.path)
        if not os.path.lexists(self.owner) and any(self.path.iterdir()):
            raise FileExistsError(f"unrecognized agreement directory: {self.path}")
        fd = os.open(
            self.owner,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o666,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except BaseException:
            os.close(fd)
            raise
        self.fd = fd
        self.identity = identity
        self._verify_owner(fd)
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
        marker = os.read(fd, len(_MAGIC) + 1)
        records = self._records()
        if marker != _MAGIC and (marker or records):
            raise FileExistsError(f"unrecognized agreement directory: {self.path}")
        self.owned = True
        for entry in records:
            entry.unlink()
        if not marker and os.write(fd, _MAGIC) != len(_MAGIC):
            raise OSError(f"short write of agreement owner marker: {self.owner}")
        os.fsync(fd)
        _sync_directory(self.path)
        _sync_directory(self.path.parent)

    def close(self):
        fd = self.fd
        if fd is None:
            return
        self.fd = None
        if not self.owned:
            os.close(fd)
            return
        try:
            self._remove(fd)
        except (KeyboardInterrupt, SystemExit):
            self._retry_cleanup(fd)
            raise
        finally:
            os.close(fd)

    def _retry_cleanup(self, fd):
        with suppress(OSError, ValueError):
            self._remove(fd)

    def _remove(self, fd):
        if os.path.lexists(self.owner):
            self._remove_records(fd)
        elif not os.path.lexists(self.path):
            _sync_directory(self.path.parent)
            return
        elif _identity(self.path) != self.identity:
            raise OSError(f"agreement directory changed: {self.path}")
        try:
            self.path.rmdir()
        except OSError as exc:
            self._check_cleanup_race(exc)
        _sync_directory(self.path.parent)

    def _remove_records(self, fd):
        self._verify_owner(fd)
        for entry in self._records():
            entry.unlink()
        self.owner.unlink()

    def _check_cleanup_race(self, exc):
        # Another invocation can claim an empty sibling just after we unlink
        # our owner file. Leave the replacement to the new lock holder.
        if exc.errno != errno.ENOTEMPTY or not os.path.lexists(self.owner):
            raise exc

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


def agreement_bytes(ctx, components, rank_state):
    return desc.json_bytes([ctx.world_size, components, sorted(rank_state)])
