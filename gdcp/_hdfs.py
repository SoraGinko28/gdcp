"""Native HDFS transport for GDCP checkpoint directories."""

import os
import re
import shutil
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from uuid import uuid4

from . import desc

_CURRENT = ".gdcp-current"
_REVISION = re.compile(r"\.gdcp-revision-[0-9a-f]{32}")
_BLOCK = 8 * 1024 * 1024


@dataclass(frozen=True)
class HdfsPath:
    authority: str
    path: PurePosixPath

    def __str__(self):
        return f"hdfs://{self.authority}{self.path}"

    def __truediv__(self, name):
        if PurePosixPath(name).name != name or name in ("", ".", ".."):
            raise ValueError("expected a single HDFS checkpoint directory name")
        return HdfsPath(self.authority, self.path / name)


def checkpoint_path(path):
    if isinstance(path, HdfsPath):
        return path
    raw = os.fspath(path)
    if not raw.lower().startswith("hdfs:"):
        return Path(raw)
    if not raw.lower().startswith("hdfs://"):
        raise ValueError("pass hdfs:// as a string; pathlib lost the URI authority")
    uri = urlsplit(raw)
    remote = PurePosixPath(uri.path)
    if (
        not remote.is_absolute()
        or ".." in remote.parts
        or uri.query
        or uri.fragment
        or uri.username is not None
        or "\x00" in raw
        or uri.path.startswith("//")
    ):
        raise ValueError("invalid hdfs:// checkpoint URI")
    return HdfsPath(uri.netloc, remote)


def require_client():
    try:
        return import_module("hdfs_native")
    except ModuleNotFoundError as exc:
        if exc.name != "hdfs_native":
            raise
        raise RuntimeError("hdfs:// requires the optional hdfs-native client") from exc


def _client(target):
    return require_client().Client(
        f"hdfs://{target.authority}" if target.authority else None
    )


def _revision(client, root):
    try:
        with client.read(str(root / _CURRENT)) as stream:
            value = stream.read(65).decode("ascii")
    except FileNotFoundError:
        return None
    if not _REVISION.fullmatch(value):
        raise ValueError(f"invalid GDCP HDFS revision in {root}")
    return value


def _resolve(client, root, latest):
    revision = _revision(client, root)
    if revision:
        return root / revision
    if latest and not desc.STEP_DIRECTORY.fullmatch(root.name):
        entries = (root / name for name, isdir in _children(client, root) if isdir)
        return _latest_revision(client, entries, root)
    raise FileNotFoundError(f"no committed GDCP HDFS checkpoint in {root}")


def _latest_revision(client, entries, root):
    for entry in desc.numbered_generations(entries):
        revision = _revision(client, entry)
        if revision:
            return entry / revision
    raise FileNotFoundError(f"no committed GDCP HDFS checkpoint in {root}")


def _upload_file(client, path, dst):
    with path.open("rb") as source, client.create(dst) as output:
        shutil.copyfileobj(source, output, _BLOCK)


def _download_file(client, remote, path):
    with client.read(str(remote)) as source, path.open("xb") as output:
        shutil.copyfileobj(source, output, _BLOCK)


def _upload(client, local, remote):
    client.mkdirs(str(remote))
    for path in local.rglob("*"):
        dst = str(remote / path.relative_to(local))
        if path.is_symlink():
            raise ValueError(f"unexpected symlink in checkpoint: {path}")
        if path.is_dir():
            client.mkdirs(dst)
        else:
            _upload_file(client, path, dst)


def publish(client, target, local):
    revision = ".gdcp-revision-" + uuid4().hex
    root = target.path
    data = root / revision
    pending = root / (revision + ".commit")
    client.mkdirs(str(root), create_parent=True)
    # Never replace an unrelated file at the reserved pointer name.
    previous = _revision(client, root)
    try:
        _upload(client, local, data)
        with client.create(str(pending)) as output:
            output.write(revision.encode("ascii"))
    except BaseException:
        with suppress(Exception):
            client.delete(str(data), recursive=True)
        with suppress(Exception):
            client.delete(str(pending))
        raise
    # A lost RPC reply can mean that rename succeeded. Keep the immutable
    # revision in that case so a published pointer always remains readable.
    try:
        client.rename(
            str(pending), str(root / _CURRENT), overwrite=previous is not None
        )
    except Exception:
        # The NameNode may have committed the rename before the connection broke.
        if _revision(client, root) != revision:
            raise


def _children(client, remote):
    for info in client.list_status(str(remote)):
        entry = PurePosixPath(info.path)
        if entry.parent != remote or entry.name in ("", ".", ".."):
            raise ValueError("invalid child in HDFS checkpoint listing")
        yield entry.name, info.isdir


def _download(client, remote, local, manifest_only):
    names = (
        [(desc.MANIFEST_NAME, False)] if manifest_only else _children(client, remote)
    )
    for name, isdir in names:
        path = local / name
        if isdir:
            path.mkdir()
            _download(client, remote / name, path, False)
        else:
            _download_file(client, remote / name, path)


def localize(path, stack, *, latest=False, manifest_only=False):
    target = checkpoint_path(path)
    if not isinstance(target, HdfsPath):
        return target, {}
    client = _client(target)
    remote = _resolve(client, target.path, latest)
    local = Path(stack.enter_context(TemporaryDirectory(prefix="gdcp-hdfs-read-")))
    _download(client, remote, local, manifest_only)
    return local, {"HDFS revision": str(HdfsPath(target.authority, remote))}
