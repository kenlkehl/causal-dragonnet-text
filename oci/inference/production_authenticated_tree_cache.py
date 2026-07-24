"""Process-local authentication for immutable production directory trees.

The first authentication reads and hashes every byte. Repeated authentication
in the same process re-enumerates the complete tree and compares stable
filesystem identities, but does not reread unchanged file content. The cached
authority is deliberately:

* scoped to one PID;
* reset after ``fork``;
* unavailable through pickle or copy;
* never written to disk; and
* poisoned after any observed tree drift.

This is a performance capability, not a replacement for an independent
fresh-process byte authentication. In particular, metadata checks over a
remote filesystem are not cryptographically equivalent to rereading bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

AUTHENTICATED_DIRECTORY_TREE_POLICY = "single_full_hash_process_local_inventory_guard_v1"
AUTHENTICATED_DIRECTORY_TREE_INVENTORY_SCHEMA = (
    "production_authenticated_directory_tree_inventory_v1"
)

_MAX_CACHE_ENTRIES = 8
_BANNED_EXECUTABLE_SUFFIXES = frozenset(
    {
        ".bat",
        ".bin",
        ".cmd",
        ".com",
        ".dll",
        ".dylib",
        ".exe",
        ".jar",
        ".joblib",
        ".pkl",
        ".pickle",
        ".ps1",
        ".pt",
        ".pth",
        ".py",
        ".pyc",
        ".pyo",
        ".sh",
        ".so",
    }
)
_SIGNATURE_FIELDS = (
    "device",
    "inode",
    "mode",
    "link_count",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
)


class AuthenticatedDirectoryTreeDriftError(RuntimeError):
    """Raised when a previously authenticated process-local tree changes."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _legacy_workflow_sha256_json(value: Any) -> str:
    """Match production_all_evidence_workflow._sha byte-for-byte."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _signature_mapping(signature: tuple[int, ...]) -> dict[str, int]:
    return dict(zip(_SIGNATURE_FIELDS, signature, strict=True))


def _canonical_real_root(path: Path | str) -> Path:
    supplied = Path(path)
    if not supplied.is_absolute():
        raise ValueError("authenticated directory tree path must be absolute")
    try:
        state = os.lstat(supplied)
    except OSError as exc:
        raise FileNotFoundError(f"authenticated directory tree does not exist: {supplied}") from exc
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise ValueError("authenticated directory tree root must be one real directory")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied:
        raise ValueError(
            "authenticated directory tree path must be canonical and contain "
            "no symlinked components"
        )
    return resolved


@dataclass(frozen=True)
class _MetadataInventory:
    root_signature: tuple[int, ...]
    directories: tuple[tuple[str, tuple[int, ...]], ...]
    files: tuple[tuple[str, tuple[int, ...]], ...]


def _metadata_inventory(root: Path) -> _MetadataInventory:
    root_state = os.lstat(root)
    if stat.S_ISLNK(root_state.st_mode) or not stat.S_ISDIR(root_state.st_mode):
        raise ValueError(
            "authenticated directory tree root changed into a linked or " "special object"
        )
    directories: list[tuple[str, tuple[int, ...]]] = []
    files: list[tuple[str, tuple[int, ...]]] = []

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current, raw_directories, raw_files in os.walk(
        root,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        current_path = Path(current)
        relative_current = current_path.relative_to(root)
        raw_directories.sort()
        raw_files.sort()
        for name in raw_directories:
            child = current_path / name
            state = os.lstat(child)
            if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
                raise ValueError(
                    "authenticated directory tree contains a linked or " "special directory"
                )
            directories.append(
                (
                    (relative_current / name).as_posix(),
                    _stat_signature(state),
                )
            )
        for name in raw_files:
            child = current_path / name
            state = os.lstat(child)
            if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
                raise ValueError(
                    "authenticated directory tree contains a linked or " "special file"
                )
            files.append(
                (
                    (relative_current / name).as_posix(),
                    _stat_signature(state),
                )
            )
    directories.sort(key=lambda row: row[0])
    files.sort(key=lambda row: row[0])
    if not files:
        raise ValueError("authenticated directory tree cannot be empty")
    return _MetadataInventory(
        root_signature=_stat_signature(root_state),
        directories=tuple(directories),
        files=tuple(files),
    )


@dataclass(frozen=True)
class _AuthenticatedFile:
    relative_path: str
    sha256: str
    size_bytes: int
    signature: tuple[int, ...]
    leading_bytes: bytes = field(repr=False)


def _stable_file_authentication(
    root: Path,
    relative_path: str,
) -> _AuthenticatedFile:
    path = root / relative_path
    before_path = os.lstat(path)
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise ValueError("authenticated directory tree contains a linked or special file")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    leading = b""
    try:
        before_fd = os.fstat(descriptor)
        if _stat_signature(before_fd) != _stat_signature(before_path):
            raise RuntimeError(f"authenticated tree file changed while opening: {relative_path}")
        while block := os.read(descriptor, 1024 * 1024):
            if len(leading) < 16:
                leading += block[: 16 - len(leading)]
            digest.update(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(path)
    signature = _stat_signature(before_path)
    if (
        _stat_signature(before_fd) != signature
        or _stat_signature(after_fd) != signature
        or _stat_signature(after_path) != signature
    ):
        raise RuntimeError(f"authenticated tree file changed while hashing: {relative_path}")
    suffix = path.suffix.casefold()
    mode = int(signature[2])
    if (
        mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        or suffix in _BANNED_EXECUTABLE_SUFFIXES
        or leading.startswith(b"#!")
        or leading.startswith(b"\x7fELF")
        or leading.startswith(b"MZ")
        or leading.startswith((b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"))
    ):
        raise ValueError(
            "authenticated directory tree contains an executable or "
            f"pickle-capable artifact: {relative_path}"
        )
    return _AuthenticatedFile(
        relative_path=relative_path,
        sha256=digest.hexdigest(),
        size_bytes=int(before_path.st_size),
        signature=signature,
        leading_bytes=leading,
    )


@dataclass(frozen=True)
class AuthenticatedDirectoryTreeSnapshot:
    """One nontransferable process-local authenticated tree capability."""

    _owner_pid: int = field(repr=False)
    _root: Path
    _metadata: _MetadataInventory = field(repr=False)
    _files: tuple[_AuthenticatedFile, ...] = field(repr=False)
    _local_model_provenance: Mapping[str, Any] = field(repr=False)
    _workflow_path_identity: Mapping[str, Any] = field(repr=False)
    _workflow_inventory: Mapping[str, Any] = field(repr=False)

    def _require_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "authenticated directory tree capability cannot cross a " "process boundary"
            )

    @property
    def path(self) -> Path:
        self._require_owner()
        return self._root

    def local_model_provenance(self) -> dict[str, Any]:
        self._require_owner()
        return copy.deepcopy(dict(self._local_model_provenance))

    def workflow_path_identity(self) -> dict[str, Any]:
        """Return the historical workflow request wire representation."""

        self._require_owner()
        return copy.deepcopy(dict(self._workflow_path_identity))

    def workflow_inventory_projection(self) -> dict[str, Any]:
        """Return a closed audit projection including metadata signatures."""

        self._require_owner()
        return copy.deepcopy(dict(self._workflow_inventory))

    def __copy__(self) -> None:
        raise TypeError("authenticated directory tree capabilities cannot be copied")

    def __deepcopy__(self, _memo: Any) -> None:
        raise TypeError("authenticated directory tree capabilities cannot be copied")

    def __reduce__(self) -> None:
        raise TypeError("authenticated directory tree capabilities cannot be serialized")

    def __reduce_ex__(self, _protocol: int) -> None:
        raise TypeError("authenticated directory tree capabilities cannot be serialized")


def _full_authentication(root: Path) -> AuthenticatedDirectoryTreeSnapshot:
    before = _metadata_inventory(root)
    authenticated_files = tuple(
        _stable_file_authentication(root, relative_path)
        for relative_path, _signature in before.files
    )
    after = _metadata_inventory(root)
    if (
        before != after
        or tuple((row.relative_path, row.signature) for row in authenticated_files) != before.files
    ):
        raise RuntimeError("authenticated directory tree changed during full authentication")

    builder_directories = [path for path, _signature in before.directories]
    builder_files = [
        {
            "path": row.relative_path,
            "sha256": row.sha256,
            "size_bytes": row.size_bytes,
        }
        for row in authenticated_files
    ]
    builder_body = {
        "directories": builder_directories,
        "files": builder_files,
    }
    local_model_provenance = {
        "path": str(root),
        "tree_sha256": _sha256_json(builder_body),
        "file_count": len(builder_files),
        "directory_count": len(builder_directories),
        "total_file_bytes": sum(int(row["size_bytes"]) for row in builder_files),
    }

    workflow_files = [
        {
            "relative_path": row.relative_path,
            "sha256": row.sha256,
            "size_bytes": row.size_bytes,
        }
        for row in authenticated_files
    ]
    workflow_path_identity = {
        "kind": "directory",
        "path": str(root),
        "file_count": len(workflow_files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in workflow_files),
        "tree_sha256": _legacy_workflow_sha256_json(workflow_files),
        "files": workflow_files,
    }
    workflow_inventory = {
        "schema_version": AUTHENTICATED_DIRECTORY_TREE_INVENTORY_SCHEMA,
        "policy": AUTHENTICATED_DIRECTORY_TREE_POLICY,
        "path": str(root),
        "builder_tree_sha256": local_model_provenance["tree_sha256"],
        "workflow_tree_sha256": workflow_path_identity["tree_sha256"],
        "file_count": len(workflow_files),
        "directory_count": len(builder_directories),
        "total_file_bytes": local_model_provenance["total_file_bytes"],
        "root_signature": _signature_mapping(before.root_signature),
        "directories": [
            {
                "relative_path": relative_path,
                "signature": _signature_mapping(signature),
            }
            for relative_path, signature in before.directories
        ],
        "files": [
            {
                "relative_path": row.relative_path,
                "sha256": row.sha256,
                "size_bytes": row.size_bytes,
                "signature": _signature_mapping(row.signature),
            }
            for row in authenticated_files
        ],
    }
    return AuthenticatedDirectoryTreeSnapshot(
        _owner_pid=os.getpid(),
        _root=root,
        _metadata=before,
        _files=authenticated_files,
        _local_model_provenance=MappingProxyType(local_model_provenance),
        _workflow_path_identity=MappingProxyType(workflow_path_identity),
        _workflow_inventory=MappingProxyType(workflow_inventory),
    )


_CACHE_LOCK = threading.RLock()
_CACHE_PID = os.getpid()
_CACHE: OrderedDict[Path, AuthenticatedDirectoryTreeSnapshot] = OrderedDict()
_POISONED_PATHS: OrderedDict[Path, None] = OrderedDict()


def _reset_cache_for_current_process() -> None:
    global _CACHE_LOCK, _CACHE_PID, _CACHE, _POISONED_PATHS
    _CACHE_LOCK = threading.RLock()
    _CACHE_PID = os.getpid()
    _CACHE = OrderedDict()
    _POISONED_PATHS = OrderedDict()


def _ensure_current_process() -> None:
    if _CACHE_PID != os.getpid():
        _reset_cache_for_current_process()


def clear_authenticated_directory_tree_cache() -> None:
    """Clear all process-local authentication capabilities.

    This is intended for tests and explicit lifecycle teardown. A subsequent
    authentication performs a complete byte reread.
    """

    with _CACHE_LOCK:
        _ensure_current_process()
        _CACHE.clear()
        _POISONED_PATHS.clear()


def _poison_path(root: Path) -> None:
    _CACHE.pop(root, None)
    _POISONED_PATHS[root] = None
    _POISONED_PATHS.move_to_end(root)
    while len(_POISONED_PATHS) > _MAX_CACHE_ENTRIES:
        _POISONED_PATHS.popitem(last=False)


def authenticate_directory_tree(
    path: Path | str,
) -> AuthenticatedDirectoryTreeSnapshot:
    """Authenticate one tree or cheaply recheck an unchanged same-PID tree."""

    with _CACHE_LOCK:
        _ensure_current_process()
        supplied = Path(path)
        if supplied in _POISONED_PATHS:
            _POISONED_PATHS.move_to_end(supplied)
            raise AuthenticatedDirectoryTreeDriftError(
                "authenticated directory tree was previously observed to drift"
            )
        previously_authenticated = supplied in _CACHE
        try:
            root = _canonical_real_root(supplied)
        except BaseException as exc:
            if previously_authenticated:
                _poison_path(supplied)
                raise AuthenticatedDirectoryTreeDriftError(
                    "authenticated directory tree root changed after " "authentication"
                ) from exc
            raise
        if root in _POISONED_PATHS:
            _POISONED_PATHS.move_to_end(root)
            raise AuthenticatedDirectoryTreeDriftError(
                "authenticated directory tree was previously observed to drift"
            )
        cached = _CACHE.get(root)
        if cached is not None:
            try:
                observed = _metadata_inventory(root)
            except BaseException as exc:
                _poison_path(root)
                raise AuthenticatedDirectoryTreeDriftError(
                    "authenticated directory tree changed after authentication"
                ) from exc
            if observed != cached._metadata:
                _poison_path(root)
                raise AuthenticatedDirectoryTreeDriftError(
                    "authenticated directory tree inventory changed after " "authentication"
                )
            _CACHE.move_to_end(root)
            return cached

        snapshot = _full_authentication(root)
        _CACHE[root] = snapshot
        _CACHE.move_to_end(root)
        while len(_CACHE) > _MAX_CACHE_ENTRIES:
            _CACHE.popitem(last=False)
        return snapshot


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_cache_for_current_process)


__all__ = [
    "AUTHENTICATED_DIRECTORY_TREE_INVENTORY_SCHEMA",
    "AUTHENTICATED_DIRECTORY_TREE_POLICY",
    "AuthenticatedDirectoryTreeDriftError",
    "AuthenticatedDirectoryTreeSnapshot",
    "authenticate_directory_tree",
    "clear_authenticated_directory_tree_cache",
]
