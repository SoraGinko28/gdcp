"""Native HDFS transport for GDCP checkpoint directories."""

import json
import os
import re
import shutil
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from uuid import uuid4

from . import desc

_CURRENT = ".gdcp-current"
_REVISION = re.compile(r"\.gdcp-revision-[0-9a-f]{32}")
_HF_HEADERS = ".gdcp-hf-headers.json"
_ACTIVE: ContextVar["_ReadCache | None"] = ContextVar("gdcp_hdfs_read", default=None)
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
        entries = (root / name for name, info in _children(client, root) if info.isdir)
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
        if (local / desc.HF_COMMITTED_NAME).is_file() and not (
            local / _HF_HEADERS
        ).exists():
            _publish_hf_headers(client, local, data)
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
        yield entry.name, info


def _publish_hf_headers(client, local, remote):
    from .hf import _read_header_file

    files = json.loads((local / desc.HF_COMMITTED_NAME).read_bytes())["files"]
    headers = {}
    for name in files:
        desc._safe_relative(name)
        headers[name] = _read_header_file(local / name)
    with client.create(str(remote / _HF_HEADERS)) as out:
        out.write(json.dumps(headers, separators=(",", ":"), allow_nan=False).encode())


def _list_revision(client, remote, local, sizes):
    for name, info in _children(client, remote):
        path = local / name
        if info.isdir:
            path.mkdir()
            _list_revision(client, remote / name, path, sizes)
        else:
            sizes[path] = info.length


class _ReadCache:
    """One rank's private cache, pinned to one immutable remote revision."""

    def __init__(self, client, remote, root):
        self.client, self.remote, self.root = client, remote, root
        self.sizes = {}
        self.headers = None
        self.fetched = set()
        _list_revision(client, remote, root, self.sizes)

    def fetch(self, path):
        if path in self.fetched:
            return path
        if path not in self.sizes:
            raise FileNotFoundError(path)
        remote = self.remote / path.relative_to(self.root)
        _download_file(self.client, remote, path)
        if path.stat().st_size != self.sizes[path]:
            raise OSError(f"incomplete HDFS download: {remote}")
        self.fetched.add(path)
        return path

    def header(self, path):
        from .hf import _read_header

        catalog = self.root / _HF_HEADERS
        if self.headers is None:
            self.headers = (
                json.loads(self.fetch(catalog).read_bytes())
                if catalog in self.sizes
                else {}
            )
        if catalog in self.sizes or path.name in self.headers:
            return self.headers[path.name]
        # Older revisions have no catalog. Read only header metadata here.
        with self.client.read(str(self.remote / path.relative_to(self.root))) as stream:
            entry = list(_read_header(stream, path))
        self.headers[path.name] = entry
        return entry

    def fetch_payload(self, root, filename):
        desc._safe_relative(filename)
        path = self.fetch(root / filename)
        if root != self.root or self.headers is None:
            return
        from .hf import _read_header_file

        if list(_read_header_file(path)) != self.headers[filename]:
            raise ValueError(f"{filename}: HF header disagrees with HDFS catalog")


def cache_for(path):
    cache = _ACTIVE.get()
    if cache is not None and path is not None and Path(path).is_relative_to(cache.root):
        return cache
    return None


def local_file(path):
    cache = cache_for(path)
    return cache.fetch(path) if cache is not None else path


def is_file(path):
    cache = cache_for(path)
    return path in cache.sizes if cache is not None else path.is_file()


def file_size(path):
    cache = cache_for(path)
    if cache is not None:
        if path not in cache.sizes:
            raise FileNotFoundError(path)
        return cache.sizes[path]
    return path.stat().st_size


def localize(path, stack, *, latest=False, manifest_only=False, selective=False):
    target = checkpoint_path(path)
    if not isinstance(target, HdfsPath):
        return target, {}
    client = _client(target)
    remote = _resolve(client, target.path, latest)
    local = Path(stack.enter_context(TemporaryDirectory(prefix="gdcp-hdfs-read-")))
    if manifest_only:
        _download_file(client, remote / desc.MANIFEST_NAME, local / desc.MANIFEST_NAME)
    else:
        cache = _ReadCache(client, remote, local)
        _stage_metadata(cache, selective, stack)
    return local, {"HDFS revision": str(HdfsPath(target.authority, remote))}


def _stage_metadata(cache, selective, stack):
    root = cache.root
    markers = (desc.MANIFEST_NAME, desc.HF_COMMITTED_NAME)
    if not selective or not any(root / name in cache.sizes for name in markers):
        for file in cache.sizes:
            cache.fetch(file)
        return
    for name in (*markers, desc.PENDING_NAME):
        if root / name in cache.sizes:
            cache.fetch(root / name)
    stack.callback(_ACTIVE.reset, _ACTIVE.set(cache))
