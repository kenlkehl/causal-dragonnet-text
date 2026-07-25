"""Immutable, content-addressed source snapshots for long production runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SOURCE_SNAPSHOT_SCHEMA = "production_source_snapshot_v2"
SOURCE_SNAPSHOT_MANIFEST = "source_snapshot_manifest.json"
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "source_repository",
        "files",
        "file_count",
        "python_bytecode_writes_allowed",
        "content_sha256",
        "locator_attestation_sha256",
    }
)
_INVENTORY_FIELDS = frozenset({"relative_path", "sha256", "size_bytes"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_READ_ONLY_FILE_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
_READ_ONLY_DIRECTORY_MODE = (
    stat.S_IRUSR
    | stat.S_IXUSR
    | stat.S_IRGRP
    | stat.S_IXGRP
    | stat.S_IROTH
    | stat.S_IXOTH
)


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


def _scientific_snapshot_body(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the path-neutral source content authenticated scientifically."""

    return {
        "schema_version": value.get("schema_version"),
        "files": value.get("files"),
        "file_count": value.get("file_count"),
        "python_bytecode_writes_allowed": value.get(
            "python_bytecode_writes_allowed"
        ),
    }


def _locator_attestation_body(
    *,
    source_repository: str,
    content_sha256: str,
) -> dict[str, str]:
    """Bind the operational producer locator without changing code identity."""

    return {
        "source_repository": source_repository,
        "content_sha256": content_sha256,
    }


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _directory_signature(path: Path, *, label: str) -> tuple[int, int, int, int, int, int, int]:
    try:
        state = os.lstat(path)
    except OSError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise ValueError(f"{label} must be one real directory")
    return _stat_signature(state)


def _stable_file(path: Path) -> tuple[str, int]:
    try:
        before_path = os.lstat(path)
    except OSError as exc:
        raise FileNotFoundError(f"source snapshot file does not exist: {path}") from exc
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"source snapshot file must be one real regular file: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        before_fd = os.fstat(descriptor)
        if _stat_signature(before_fd) != _stat_signature(before_path):
            raise RuntimeError(f"source file changed while opening: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
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
        raise RuntimeError(f"source file changed while hashing: {path}")
    return digest.hexdigest(), int(after_path.st_size)


def _reject_duplicate_json_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"source snapshot manifest contains duplicate key: {key}")
        result[key] = value
    return result


def _required_source_files(repository_root: Path) -> tuple[Path, ...]:
    required: list[Path] = []
    oci_root = repository_root / "oci"
    scripts_root = repository_root / "scripts"
    if not oci_root.is_dir() or not scripts_root.is_dir():
        raise FileNotFoundError("repository snapshot requires oci/ and scripts/")
    for source_root in (oci_root, scripts_root):
        for path in source_root.rglob("*"):
            state = os.lstat(path)
            if stat.S_ISLNK(state.st_mode):
                raise ValueError(f"source snapshot cannot traverse symlinks: {path}")
            if not stat.S_ISDIR(state.st_mode) and not stat.S_ISREG(state.st_mode):
                raise ValueError(f"source snapshot cannot traverse special files: {path}")
    required.extend(path for path in oci_root.rglob("*.py") if path.is_file())
    required.extend(path for path in scripts_root.rglob("*.py") if path.is_file())
    for relative in ("pyproject.toml", "uv.lock"):
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source snapshot file is absent: {path}")
        required.append(path)
    return tuple(sorted(set(required)))


def _relative_inventory(
    repository_root: Path,
    files: Iterable[Path],
) -> tuple[dict[str, Any], ...]:
    inventory: list[dict[str, Any]] = []
    for source in files:
        if source.is_symlink():
            raise ValueError(f"source snapshot cannot include symlinks: {source}")
        resolved = source.resolve(strict=True)
        try:
            relative = resolved.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(f"source file escapes repository root: {source}") from exc
        digest, size = _stable_file(resolved)
        inventory.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    inventory.sort(key=lambda row: str(row["relative_path"]))
    if len({row["relative_path"] for row in inventory}) != len(inventory):
        raise RuntimeError("source snapshot inventory contains duplicate paths")
    return tuple(inventory)


@dataclass(frozen=True)
class ProductionSourceSnapshot:
    root: Path
    manifest_path: Path
    content_sha256: str
    file_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "content_sha256": self.content_sha256,
            "file_count": self.file_count,
        }


def create_production_source_snapshot(
    *,
    repository_root: Path | str,
    target_dir: Path | str,
) -> ProductionSourceSnapshot:
    """Copy the exact Python/lock state into a fresh content-addressed tree."""

    repository_supplied = Path(repository_root)
    repository = repository_supplied.resolve(strict=True)
    if repository != repository_supplied:
        raise ValueError("source repository path must be canonical")
    target = Path(target_dir)
    if not target.is_absolute():
        raise ValueError("source snapshot target must be absolute")
    if target.is_symlink() or target.exists():
        raise FileExistsError("source snapshot target must be one fresh real path")
    target_parent = target.parent.resolve(strict=True)
    if target_parent != target.parent:
        raise ValueError("source snapshot target parent must be canonical")
    if repository == target or repository in target.parents:
        # A snapshot beneath the repository is allowed, but it must not enter
        # the source inventory recursively. Only explicit source files are read.
        pass
    files = _required_source_files(repository)
    inventory = _relative_inventory(repository, files)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        for row in inventory:
            relative = Path(str(row["relative_path"]))
            source = repository / relative
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination, follow_symlinks=False)
            observed_sha, observed_size = _stable_file(destination)
            if (
                observed_sha != row["sha256"]
                or observed_size != int(row["size_bytes"])
            ):
                raise RuntimeError(f"source snapshot copy changed bytes: {relative}")
        scientific_body = {
            "schema_version": SOURCE_SNAPSHOT_SCHEMA,
            "files": list(inventory),
            "file_count": len(inventory),
            "python_bytecode_writes_allowed": False,
        }
        content_sha256 = _sha256_json(scientific_body)
        source_repository = str(repository)
        manifest = {
            **scientific_body,
            "source_repository": source_repository,
            "content_sha256": content_sha256,
            "locator_attestation_sha256": _sha256_json(
                _locator_attestation_body(
                    source_repository=source_repository,
                    content_sha256=content_sha256,
                )
            ),
        }
        manifest_path = temporary / SOURCE_SNAPSHOT_MANIFEST
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.chmod(_READ_ONLY_FILE_MODE)
            elif path.is_dir():
                path.chmod(_READ_ONLY_DIRECTORY_MODE)
        temporary.chmod(_READ_ONLY_DIRECTORY_MODE)
        if target.is_symlink() or target.exists():
            raise FileExistsError("source snapshot target was populated during publication")
        os.rename(temporary, target)
    except BaseException:
        for directory in [temporary, *temporary.rglob("*")]:
            if directory.is_dir() and not directory.is_symlink():
                try:
                    directory.chmod(stat.S_IRWXU)
                except OSError:
                    pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    snapshot = validate_production_source_snapshot(target)
    return snapshot


def validate_production_source_snapshot(
    snapshot_root: Path | str,
) -> ProductionSourceSnapshot:
    """Reopen and authenticate a source snapshot without importing its code."""

    supplied = Path(snapshot_root)
    if not supplied.is_absolute():
        raise ValueError("source snapshot path must be absolute")
    if supplied.is_symlink() or not supplied.is_dir():
        raise ValueError("source snapshot must be one existing real directory")
    root = supplied.resolve(strict=True)
    if root != supplied:
        raise ValueError("source snapshot path must be canonical")
    root_signature = _directory_signature(root, label="source snapshot root")
    manifest_path = root / SOURCE_SNAPSHOT_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("source snapshot manifest is absent")
    try:
        value = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source snapshot manifest is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("source snapshot manifest must be one object")
    if set(value) != set(_MANIFEST_FIELDS):
        raise ValueError("source snapshot manifest is not a closed schema")
    files = value.get("files")
    scientific_body = _scientific_snapshot_body(value)
    source_repository = value.get("source_repository")
    content_sha256 = value.get("content_sha256")
    locator_attestation_sha256 = value.get("locator_attestation_sha256")
    if (
        scientific_body.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA
        or scientific_body.get("python_bytecode_writes_allowed") is not False
        or not isinstance(source_repository, str)
        or not Path(source_repository).is_absolute()
        or not isinstance(files, list)
        or scientific_body.get("file_count") != len(files)
        or not isinstance(content_sha256, str)
        or _SHA256.fullmatch(content_sha256) is None
        or content_sha256 != _sha256_json(scientific_body)
        or not isinstance(locator_attestation_sha256, str)
        or _SHA256.fullmatch(locator_attestation_sha256) is None
        or locator_attestation_sha256
        != _sha256_json(
            _locator_attestation_body(
                source_repository=source_repository,
                content_sha256=content_sha256,
            )
        )
    ):
        raise ValueError("source snapshot manifest identity is invalid")

    expected_paths = {SOURCE_SNAPSHOT_MANIFEST}
    expected_directories: set[str] = set()
    previous: str | None = None
    for row in files:
        if not isinstance(row, Mapping) or set(row) != set(_INVENTORY_FIELDS):
            raise ValueError("source snapshot file inventory is invalid")
        relative_text = row["relative_path"]
        if not isinstance(relative_text, str) or not relative_text:
            raise ValueError("source snapshot inventory path is invalid")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative_text == SOURCE_SNAPSHOT_MANIFEST
            or (previous is not None and relative_text <= previous)
            or not isinstance(row["sha256"], str)
            or _SHA256.fullmatch(row["sha256"]) is None
            or not isinstance(row["size_bytes"], int)
            or isinstance(row["size_bytes"], bool)
            or row["size_bytes"] < 0
        ):
            raise ValueError("source snapshot inventory order/path is invalid")
        previous = relative_text
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source snapshot file is absent: {relative_text}")
        state = os.lstat(path)
        if (
            not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or stat.S_IMODE(state.st_mode) != _READ_ONLY_FILE_MODE
        ):
            raise ValueError(
                f"source snapshot file is linked, special, or writable: {relative_text}"
            )
        digest, size = _stable_file(path)
        if digest != row["sha256"] or size != int(row["size_bytes"]):
            raise ValueError(f"source snapshot file changed: {relative_text}")
        expected_paths.add(relative.as_posix())
        expected_directories.update(
            parent.as_posix()
            for parent in relative.parents
            if parent != Path(".")
        )

    manifest_state = os.lstat(manifest_path)
    if (
        not stat.S_ISREG(manifest_state.st_mode)
        or int(manifest_state.st_nlink) != 1
        or stat.S_IMODE(manifest_state.st_mode) != _READ_ONLY_FILE_MODE
    ):
        raise ValueError("source snapshot manifest is linked, special, or writable")

    observed_paths: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        state = os.lstat(path)
        if stat.S_ISLNK(state.st_mode):
            raise ValueError("source snapshot contains an unregistered symlink")
        if stat.S_ISREG(state.st_mode):
            observed_paths.add(relative)
        elif stat.S_ISDIR(state.st_mode):
            if stat.S_IMODE(state.st_mode) != _READ_ONLY_DIRECTORY_MODE:
                raise ValueError("source snapshot contains a writable directory")
            observed_directories.add(relative)
        else:
            raise ValueError("source snapshot contains an unregistered special file")
    if observed_paths != expected_paths or observed_directories != expected_directories:
        raise ValueError("source snapshot contains unregistered files or directories")
    if (
        stat.S_IMODE(os.lstat(root).st_mode) != _READ_ONLY_DIRECTORY_MODE
        or _directory_signature(root, label="source snapshot root") != root_signature
    ):
        raise RuntimeError("source snapshot root changed while it was validated")
    return ProductionSourceSnapshot(
        root=root,
        manifest_path=manifest_path,
        content_sha256=content_sha256,
        file_count=len(files),
    )


__all__ = [
    "ProductionSourceSnapshot",
    "SOURCE_SNAPSHOT_MANIFEST",
    "SOURCE_SNAPSHOT_SCHEMA",
    "create_production_source_snapshot",
    "validate_production_source_snapshot",
]
