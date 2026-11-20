"""Same-filesystem checkpoint publication and metadata-only foreign-entry moves.

Concurrent readers/writers of one destination are unsupported. The recovery
journal is durable before foreign entries leave the old directory; no foreign
payload is copied or traversed. See docs/GDCP.md for hard-kill recovery windows.
"""

import ctypes
import errno
import json
import logging
import os
import shutil
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

from . import _ownership, desc

_LOG = logging.getLogger(__name__)
JOURNAL_NAME = ".gdcp_transaction"
_RESERVED = {
    Path(JOURNAL_NAME),
    Path(desc.PENDING_NAME),
    Path(desc.MANIFEST_NAME),
    Path(desc.HF_COMMITTED_NAME),
}
PREFLIGHT = (
    "initialize save (check foreign-name collisions, mount point and same filesystem; "
    "pass a subdirectory of the mount)"
)


def _entry_identity(path):
    entry = path.lstat()
    return entry.st_dev, entry.st_ino


def _identity(path):
    identity = _entry_identity(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"checkpoint transaction requires a directory: {path}")
    return identity


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_tree(root):
    # Only checkpoint directories: foreign trees have not been moved here yet.
    for directory, _dirs, _files in os.walk(root, topdown=False):
        _sync_directory(directory)


def _mount_points():
    if sys.platform != "linux":
        return set()
    return _parse_mount_points(Path("/proc/self/mountinfo").read_text())


def _parse_mount_points(mountinfo):
    points = set()
    for line in mountinfo.splitlines():
        name = line.split()[4]
        for escaped, value in (
            (r"\040", " "),
            (r"\011", "\t"),
            (r"\012", "\n"),
            (r"\134", "\\"),
        ):
            name = name.replace(escaped, value)
        points.add(Path(name))
    return points


def _check_filesystem(root, stage):
    reference = root if root.exists() else root.parent
    if os.path.ismount(root) or root in _mount_points():
        raise ValueError(
            f"checkpoint target is a mount point: {root}; pass a subdirectory of the mount"
        )
    if reference.stat().st_dev != stage.parent.stat().st_dev:
        raise ValueError(
            f"checkpoint target and staging location are on different filesystems: "
            f"{root}, {stage}; pass a subdirectory of the mount"
        )


def _linux_rename(source, target, flags):
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, "renameat2"):
        return errno.ENOSYS
    rename = library.renameat2
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), flags) == 0:
        return 0
    return ctypes.get_errno()


def _exchange(left, right):
    """Return False only when the OS/filesystem lacks directory exchange."""
    if sys.platform != "linux":
        return False
    code = _linux_rename(left, right, 2)  # RENAME_EXCHANGE
    if not code:
        return True
    if code in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        return False
    raise OSError(code, os.strerror(code), str(right))


def _rename_exclusive(source, target):
    code = errno.ENOSYS
    if sys.platform == "linux":
        code = _linux_rename(source, target, 1)  # RENAME_NOREPLACE
    elif sys.platform == "darwin":
        rename = getattr(ctypes.CDLL(None, use_errno=True), "renamex_np", None)
        if rename is not None:
            rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            result = rename(os.fsencode(source), os.fsencode(target), 4)  # RENAME_EXCL
            code = ctypes.get_errno() if result else 0
    if code:
        raise OSError(code, os.strerror(code), str(target))


def _snapshot_owned(root):
    files = _ownership.owned_files(root)
    directories = _ownership.owned_directories(root)
    owned = {}
    for name in files | directories:
        try:
            entry = (root / name).lstat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        if (name in directories and stat.S_ISDIR(entry.st_mode)) or (
            name in files
            and (stat.S_ISREG(entry.st_mode) or stat.S_ISLNK(entry.st_mode))
        ):
            owned[name] = (entry.st_dev, entry.st_ino)
    return owned


def _entries(root, owned, relative=Path()):
    for path in (root / relative).iterdir():
        name = relative / path.name
        try:
            entry = path.lstat()
        except FileNotFoundError:
            continue
        if owned.get(name) != (entry.st_dev, entry.st_ino):
            yield name, "foreign"
        elif stat.S_ISDIR(entry.st_mode):
            yield from _entries(root, owned, name)
            yield name, "directory"
        else:
            yield name, "file"


def _foreign_entries(root, owned):
    return [name for name, kind in _entries(root, owned) if kind == "foreign"]


def _salvage_entry(old, root, name, planned):
    _collisions(old, (name,), planned)
    for parent in reversed(name.parents[:-1]):
        directory = root / parent
        directory.mkdir(exist_ok=True)
        _identity(directory)
    source, target = old / name, root / name
    _rename_exclusive(source, target)
    _sync_directory(target.parent)
    _sync_directory(source.parent)


@contextmanager
def _cleanup_workspace(stage, *, occupied=False):
    if occupied:
        workspace = Path(mkdtemp(prefix=".gdcp-cleanup-", dir=stage))
    else:
        workspace = stage
        workspace.mkdir(mode=0o700)
    try:
        yield workspace
    finally:
        if not any(workspace.iterdir()):
            workspace.rmdir()


def _remove_verified(entry, expected, workspace, *, directory=False):
    # Claim the current entry before checking its identity. Deleting under a
    # private name cannot unlink a file recreated at the original pathname.
    claimed = workspace / entry.name
    try:
        entry.rename(claimed)
        if _entry_identity(claimed) != expected:
            raise OSError(f"foreign entry replaced {entry} during cleanup")
        if directory:
            claimed.rmdir()
        else:
            claimed.unlink()
    except BaseException:
        if os.path.lexists(claimed):
            _restore_claimed_entry(claimed, entry)
        raise


def _restore_claimed_entry(claimed, entry):
    try:
        _rename_exclusive(claimed, entry)
    except OSError as exc:
        raise OSError(f"could not restore {entry}; retained {claimed}: {exc}") from exc


def _clean_old(old, root, owned, planned, stage):
    blocked = []
    for name in _foreign_entries(old, owned):
        try:
            _salvage_entry(old, root, name, planned)
        except (OSError, ValueError) as exc:
            blocked.append(f"{name}: {exc}")
    if blocked:
        raise OSError(f"could not move foreign entries: {', '.join(blocked)}")

    # Delete only verified checkpoint files and empty directories. New entries
    # can still arrive in the old directory through an open directory handle.
    with _cleanup_workspace(stage, occupied=old == stage) as workspace:
        _remove_owned(old, owned, workspace)
    old.rmdir()


def _remove_owned(old, owned, workspace):
    for name, kind in _entries(old, owned):
        entry = old / name
        if entry == workspace:
            continue
        if kind == "foreign":
            raise OSError(f"foreign entry appeared during cleanup: {name}")
        _remove_verified(entry, owned[name], workspace, directory=kind == "directory")


def _collisions(root, foreign, planned):
    for name in foreign:
        conflicts = sorted(
            (item for item in planned if name == item or name in item.parents),
            key=lambda item: (-len(item.parts), item),
        )
        if conflicts:
            entry = next(
                (item for item in conflicts if os.path.lexists(root / item)), name
            )
            raise FileExistsError(
                f"new checkpoint conflicts with foreign entry: {entry}"
            )


def _status(path):
    if not os.path.lexists(path):
        return "absent"
    if path.is_symlink() or not path.is_dir():
        return "symlink or non-directory"
    if (path / desc.PENDING_NAME).exists():
        return "pending generation"
    return (
        "complete generation"
        if _ownership.complete(path)
        else "unrecognized/incomplete directory"
    )


class DirectoryTransaction:
    """Rank-zero mutation; all ranks derive the resolved staging pathname."""

    def __init__(self, path, rank):
        self.requested = Path(os.path.abspath(path))
        self.root = Path(os.path.realpath(path))
        if not self.root.name:
            raise ValueError("the filesystem root cannot be a checkpoint")
        self.stage = self.root.with_name(f".{self.root.name}.gdcp-stage")
        self.backup = self.root.with_name(f".{self.root.name}.gdcp-backup")
        self.rank = rank
        self.new_identity = None
        self.old_identity = None
        self.old_owned = {}
        self.journal_identity = None
        self.planned = set()
        self.published = False
        self.exchanged = False
        self.backed_up = False
        self.accepted = False

    def begin(self, planned=()):
        if self.rank != 0:
            return
        self._validate_target()
        self._recover()
        self.planned = {Path(name) for name in planned} | _RESERVED
        foreign = []
        if os.path.lexists(self.root):
            self.old_identity = _identity(self.root)
            self.old_owned = _snapshot_owned(self.root)
            foreign = _foreign_entries(self.root, self.old_owned)
        _collisions(self.root, foreign, self.planned)
        self.root.parent.mkdir(parents=True, exist_ok=True)
        _check_filesystem(self.root, self.stage)
        self.stage.mkdir()
        self.new_identity = _identity(self.stage)
        desc.atomic_write(
            self.stage / desc.PENDING_NAME, b"GDCP transaction initializing\n"
        )
        self._write_journal(())

    def _validate_target(self):
        if self.requested.is_symlink() and not self.requested.exists():
            raise ValueError(f"dangling checkpoint symlink: {self.requested}")
        if os.path.lexists(self.root):
            _identity(self.root)
        # Check before recovery, too: a mounted target must never be mutated.
        if self.root.parent.exists():
            _check_filesystem(self.root, self.stage)

    def _write_journal(self, foreign):
        record = {
            "version": 1,
            "old_identity": self.old_identity,
            "new_identity": self.new_identity,
            "foreign": [
                {"path": str(name), "identity": _entry_identity(self.root / name)}
                for name in foreign
            ],
        }
        self.journal_identity = desc.atomic_write(
            self.stage / JOURNAL_NAME, desc.json_bytes(record)
        )
        _sync_directory(self.stage)
        _sync_directory(self.stage.parent)

    def _refuse_recovery(self, detail=""):
        paths = "; ".join(
            f"{path}: {_status(path)}" for path in (self.root, self.stage, self.backup)
        )
        raise FileExistsError(
            f"checkpoint recovery requires inspection: {detail} {paths}. Restore the desired complete "
            "generation and its foreign entries, then remove only verified obsolete siblings before retrying"
        )

    def _recover_backup(self):
        if (
            os.path.lexists(self.root)
            or self.backup.is_symlink()
            or not _ownership.complete(self.backup)
        ):
            self._refuse_recovery()
        self.backup.rename(self.root)
        _sync_directory(self.root.parent)
        _LOG.warning(
            "restored complete checkpoint backup %s to %s", self.backup, self.root
        )

    def _recover_published_journal(self):
        try:
            self._recover_root_journal()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._refuse_recovery(f"cannot recover {self.root / JOURNAL_NAME}: {exc};")

    def _recover(self):
        if os.path.lexists(self.backup):
            self._recover_backup()
        if not os.path.lexists(self.stage):
            self._recover_published_journal()
            return
        if self.stage.is_symlink() or not self.stage.is_dir():
            self._refuse_recovery()
        journal = self.stage / JOURNAL_NAME
        if not journal.is_file() and os.path.lexists(self.root / JOURNAL_NAME):
            self._recover_empty_cleanup_stage()
            return
        try:
            proven = self._recover_foreign(journal) if journal.is_file() else False
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._refuse_recovery(f"cannot recover {journal}: {exc};")
        pending = (self.stage / desc.PENDING_NAME).is_file()
        initializing = {entry.name for entry in self.stage.iterdir()} <= {
            desc.PENDING_NAME + ".tmp"
        }
        if _ownership.complete(self.stage) or not (proven or pending or initializing):
            self._refuse_recovery()
        shutil.rmtree(self.stage)
        _sync_directory(self.stage.parent)
        _LOG.warning("removed unpublished checkpoint stage %s", self.stage)

    def _recover_empty_cleanup_stage(self):
        if any(self.stage.iterdir()):
            self._refuse_recovery()
        # Publication can recreate an empty stage for journal cleanup.
        self.stage.rmdir()
        self._recover_published_journal()

    def _foreign_record(self, item):
        name = Path(item["path"])
        if name.is_absolute() or ".." in name.parts or name == Path("."):
            self._refuse_recovery()
        return name, tuple(item["identity"])

    def _check_foreign_at_root(self, item):
        name, expected = self._foreign_record(item)
        if _entry_identity(self.root / name) != expected:
            self._refuse_recovery()

    def _recover_root_journal(self):
        journal = self.root / JOURNAL_NAME
        if not journal.is_file() or not _ownership.complete(self.root):
            return
        with journal.open("rb") as file:
            record = json.load(file)
            entry = os.fstat(file.fileno())
        if record["version"] != 1 or tuple(record["new_identity"]) != _identity(
            self.root
        ):
            self._refuse_recovery()
        for item in record["foreign"]:
            self._check_foreign_at_root(item)
        self._remove_journal(journal, (entry.st_dev, entry.st_ino))
        _sync_directory(self.root)
        _LOG.warning("removed published checkpoint transaction journal %s", journal)

    def _recover_foreign(self, journal):
        record = json.loads(journal.read_bytes())
        old = tuple(record["old_identity"]) if record["old_identity"] else None
        actual = _identity(self.root) if os.path.lexists(self.root) else None
        if record["version"] != 1 or old != actual:
            self._refuse_recovery()
        self._check(self.stage, tuple(record["new_identity"]))
        for item in record["foreign"]:
            self._restore_entry(*self._foreign_record(item))
        return True

    def _restore_entry(self, name, expected):
        source, target = self.stage / name, self.root / name
        if os.path.lexists(target) and _entry_identity(target) == expected:
            return
        if (
            os.path.lexists(target)
            or not os.path.lexists(source)
            or _entry_identity(source) != expected
        ):
            self._refuse_recovery(f"cannot restore {source} to {target};")
        target.parent.mkdir(parents=True, exist_ok=True)
        _rename_exclusive(source, target)
        _sync_directory(source.parent)
        _sync_directory(target.parent)

    def _carry_foreign(self):
        foreign = _foreign_entries(self.root, self.old_owned)
        _collisions(self.root, foreign, self.planned)
        self._write_journal(foreign)
        for name in foreign:
            self._move_foreign(name)

    def _move_foreign(self, name):
        source, target = self.root / name, self.stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(target):
            raise FileExistsError(f"checkpoint wrote a foreign entry: {name}")
        source.rename(target)
        _sync_directory(source.parent)
        _sync_directory(target.parent)

    def publish(self):
        if self.rank != 0:
            return
        self._check(self.stage, self.new_identity)
        _sync_tree(self.stage)
        if self.old_identity is not None:
            self._check(self.root, self.old_identity)
            self._carry_foreign()
            self._replace_existing()
        else:
            self._publish_new()
        _sync_directory(self.root.parent)

    def _publish_new(self):
        if os.path.lexists(self.root):
            raise FileExistsError(f"checkpoint appeared during save: {self.root}")
        self.stage.rename(self.root)
        self.published = True

    def _replace_existing(self):
        if os.path.lexists(self.backup):
            raise FileExistsError(
                f"checkpoint backup appeared during save: {self.backup}"
            )
        self.exchanged = _exchange(self.stage, self.root)
        if self.exchanged:
            self.published = True
            return
        self.root.rename(self.backup)
        self.backed_up = True
        self.stage.rename(self.root)
        self.published = True

    def _refresh_publication(self):
        # A BaseException may arrive after a rename but before its flag assignment.
        self.published = (
            self.new_identity is not None
            and self.root.exists()
            and _identity(self.root) == self.new_identity
        )
        self.backed_up = (
            self.old_identity is not None
            and self.backup.exists()
            and _identity(self.backup) == self.old_identity
        )
        self.exchanged = (
            self.published
            and self.stage.exists()
            and _identity(self.stage) == self.old_identity
        )

    def rollback(self):
        if self.rank != 0 or self.new_identity is None:
            return
        self._refresh_publication()
        if self.published:
            self._undo_publication()
        if self.backed_up:
            self._check(self.backup, self.old_identity)
            self.backup.rename(self.root)
            self.backed_up = False
        if (self.stage / JOURNAL_NAME).is_file():
            self._recover_foreign(self.stage / JOURNAL_NAME)
        _sync_directory(self.root.parent)

    def _undo_publication(self):
        self._check(self.root, self.new_identity)
        if self.exchanged:
            self._restore_exchange()
        else:
            self.root.rename(self.stage)
        self.published = False

    def _restore_exchange(self):
        self._check(self.stage, self.old_identity)
        if not _exchange(self.stage, self.root):
            raise OSError("directory exchange became unavailable during rollback")

    def accept(self):
        self.accepted = True

    def close(self):
        if self.rank != 0 or self.new_identity is None:
            return
        if not self.accepted:
            self._finish_aborted()
            return
        obsolete = self.backup if self.backed_up else self.stage
        if self._remove_obsolete(obsolete, self.old_identity):
            self._finish_accepted()

    def _finish_aborted(self):
        try:
            self.rollback()
        except Exception as exc:
            _LOG.warning(
                "checkpoint aborted but retained stage %s: %s", self.stage, exc
            )
        else:
            self._remove_obsolete(self.stage, self.new_identity)

    def _finish_accepted(self):
        journal = self.root / JOURNAL_NAME
        try:
            self._check(self.root, self.new_identity)
            self._remove_journal(journal, self.journal_identity)
            _sync_directory(self.root)
        except Exception as exc:
            _LOG.warning(
                "checkpoint committed but retained transaction journal at %s: %s",
                journal,
                exc,
            )

    def _remove_journal(self, journal, expected):
        if not os.path.lexists(journal):
            return
        with _cleanup_workspace(self.stage) as workspace:
            _remove_verified(journal, expected, workspace)
        _sync_directory(self.stage.parent)

    def _remove_obsolete(self, path, expected):
        if expected is None or not os.path.lexists(path):
            return True
        try:
            self._check(path, expected)
            self._delete_obsolete(path)
            _sync_directory(path.parent)
        except Exception as exc:
            _LOG.warning(
                "checkpoint committed/aborted but retained sibling %s: %s", path, exc
            )
            return False
        return True

    def _delete_obsolete(self, path):
        if self.accepted:
            self._check(self.root, self.new_identity)
            _clean_old(path, self.root, self.old_owned, self.planned, self.stage)
        else:
            shutil.rmtree(path)

    @staticmethod
    def _check(path, expected):
        if expected is None or _identity(path) != expected:
            raise OSError(f"checkpoint directory changed during save: {path}")
