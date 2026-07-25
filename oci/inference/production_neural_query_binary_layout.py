"""Closed mmap-safe numerical layout for production neural-query artifacts.

Each logical array is written once as an ordinary ``.npy`` payload.  A small
canonical JSON index fixes semantic order, filename, dtype, shape, and both
file- and value-level hashes.  Validation reopens every byte and rejects
missing, extra, reordered, linked, substituted, or mutated payloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PRODUCTION_NEURAL_QUERY_NPY_INDEX_SCHEMA = (
    "production_neural_query_npy_array_index_v1"
)
_ARRAY_NAME = re.compile(r"[a-z][a-z0-9_]*")


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


def _closed_array(value: Any, *, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError(f"{name} cannot contain executable Python objects")
    return array


def numerical_array_sha256(value: Any) -> str:
    """Hash exact numerical/string array semantics independent of its path."""

    array = _closed_array(value, name="neural-query array")
    header = _canonical_json(
        {
            "dtype": array.dtype.str,
            "shape": [int(dimension) for dimension in array.shape],
        }
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _stable_regular_file_sha256(
    path: Path,
    *,
    label: str,
) -> tuple[str, int]:
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be one regular file")
    if before.st_nlink != 1:
        raise ValueError(f"{label} cannot be hard-linked")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    after = path.lstat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or size != after.st_size:
        raise RuntimeError(f"{label} changed while being authenticated")
    return digest.hexdigest(), size


def _atomic_write_new_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace immutable array artifact: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_new_npy(path: Path, array: np.ndarray) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace immutable array artifact: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npy",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_order(
    ordered_names: Sequence[str],
    *,
    available_names: set[str] | None = None,
) -> tuple[str, ...]:
    order = tuple(ordered_names)
    if (
        not order
        or len(order) != len(set(order))
        or any(
            not isinstance(name, str) or _ARRAY_NAME.fullmatch(name) is None
            for name in order
        )
    ):
        raise ValueError("neural-query array order must contain unique safe names")
    if available_names is not None and set(order) != available_names:
        raise ValueError("neural-query array order differs from the exact supplied arrays")
    return order


def write_npy_array_set(
    directory: Path | str,
    arrays: Mapping[str, Any],
    *,
    ordered_names: Sequence[str],
) -> Mapping[str, Any]:
    """Write one new immutable per-array directory and freshly validate it."""

    if not isinstance(arrays, Mapping):
        raise TypeError("neural-query arrays must be one mapping")
    order = _validated_order(
        ordered_names,
        available_names={str(name) for name in arrays},
    )
    root = Path(directory)
    if root.exists() or root.is_symlink():
        raise FileExistsError("neural-query array directory must not already exist")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.parent.is_symlink() or not root.parent.is_dir():
        raise ValueError("neural-query array directory parent must be one real directory")
    root.mkdir(exist_ok=False)

    entries: list[dict[str, Any]] = []
    for position, name in enumerate(order):
        array = _closed_array(arrays[name], name=name)
        filename = f"{position:03d}_{name}.npy"
        path = root / filename
        _atomic_write_new_npy(path, array)
        file_sha256, file_size = _stable_regular_file_sha256(
            path,
            label=f"neural-query array {name}",
        )
        entries.append(
            {
                "name": name,
                "relative_path": filename,
                "dtype": array.dtype.str,
                "shape": [int(dimension) for dimension in array.shape],
                "nbytes": int(array.nbytes),
                "file_size_bytes": int(file_size),
                "file_sha256": file_sha256,
                "content_sha256": numerical_array_sha256(array),
            }
        )
    body = {
        "schema_version": PRODUCTION_NEURAL_QUERY_NPY_INDEX_SCHEMA,
        "array_count": len(order),
        "array_order": list(order),
        "arrays": entries,
        "total_payload_bytes": sum(int(entry["file_size_bytes"]) for entry in entries),
    }
    index = {**body, "content_sha256": _sha256_json(body)}
    _atomic_write_new_bytes(
        root / "index.json",
        (_canonical_json(index) + "\n").encode("utf-8"),
    )
    descriptor, _loaded = validate_npy_array_set(
        root,
        expected_order=order,
    )
    return descriptor


def validate_npy_array_set(
    directory: Path | str,
    *,
    expected_order: Sequence[str],
    expected_inventory: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, np.ndarray]]:
    """Reopen an indexed array set through ``mmap_mode='r'`` and fail closed."""

    order = _validated_order(expected_order)
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("neural-query array artifact must be one real directory")
    index_path = root / "index.json"
    index_sha256, index_size = _stable_regular_file_sha256(
        index_path,
        label="neural-query array index",
    )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("neural-query array index is not valid JSON") from exc
    if not isinstance(index, dict):
        raise ValueError("neural-query array index must be one JSON object")
    expected_fields = {
        "schema_version",
        "array_count",
        "array_order",
        "arrays",
        "total_payload_bytes",
        "content_sha256",
    }
    body = {key: value for key, value in index.items() if key != "content_sha256"}
    entries = index.get("arrays")
    if (
        set(index) != expected_fields
        or index.get("schema_version") != PRODUCTION_NEURAL_QUERY_NPY_INDEX_SCHEMA
        or index.get("content_sha256") != _sha256_json(body)
        or index.get("array_order") != list(order)
        or index.get("array_count") != len(order)
        or not isinstance(entries, list)
        or len(entries) != len(order)
        or [entry.get("name") if isinstance(entry, Mapping) else None for entry in entries]
        != list(order)
    ):
        raise ValueError("neural-query array index has an open, reordered, or invalid schema")

    expected_filenames = {
        "index.json",
        *(
            f"{position:03d}_{name}.npy"
            for position, name in enumerate(order)
        ),
    }
    members = list(root.iterdir())
    if (
        any(member.is_symlink() for member in members)
        or {member.name for member in members} != expected_filenames
        or any(not member.is_file() for member in members)
    ):
        raise ValueError(
            "neural-query array directory has missing, extra, or linked members"
        )

    observed_inventory: dict[str, Any] = {}
    loaded: dict[str, np.ndarray] = {}
    total_payload_bytes = 0
    entry_fields = {
        "name",
        "relative_path",
        "dtype",
        "shape",
        "nbytes",
        "file_size_bytes",
        "file_sha256",
        "content_sha256",
    }
    for position, (name, raw_entry) in enumerate(zip(order, entries, strict=True)):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != entry_fields:
            raise ValueError("neural-query array index entry is open or incomplete")
        expected_filename = f"{position:03d}_{name}.npy"
        if raw_entry.get("name") != name or raw_entry.get("relative_path") != expected_filename:
            raise ValueError("neural-query array index changed semantic order or filenames")
        path = root / expected_filename
        file_sha256, file_size = _stable_regular_file_sha256(
            path,
            label=f"neural-query array {name}",
        )
        if (
            raw_entry.get("file_sha256") != file_sha256
            or raw_entry.get("file_size_bytes") != file_size
        ):
            raise RuntimeError(f"neural-query array {name} changed after emission")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"neural-query array {name} is not a safe NPY payload") from exc
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise ValueError(f"neural-query array {name} is not non-object numerical data")
        observed = {
            "dtype": array.dtype.str,
            "shape": [int(dimension) for dimension in array.shape],
            "content_sha256": numerical_array_sha256(array),
        }
        if (
            raw_entry.get("dtype") != observed["dtype"]
            or raw_entry.get("shape") != observed["shape"]
            or raw_entry.get("nbytes") != int(array.nbytes)
            or raw_entry.get("content_sha256") != observed["content_sha256"]
        ):
            raise RuntimeError(
                f"neural-query array {name} dtype, shape, or content is inconsistent"
            )
        observed_inventory[name] = observed
        loaded[name] = array
        total_payload_bytes += int(file_size)
    if index.get("total_payload_bytes") != total_payload_bytes:
        raise RuntimeError("neural-query array index payload byte count is inconsistent")
    if expected_inventory is not None and observed_inventory != dict(expected_inventory):
        raise RuntimeError("neural-query array inventory differs from its outer manifest")
    if _stable_regular_file_sha256(
        index_path,
        label="neural-query array index",
    ) != (index_sha256, index_size):
        raise RuntimeError("neural-query array index changed while validating")

    descriptor = {
        "schema_version": PRODUCTION_NEURAL_QUERY_NPY_INDEX_SCHEMA,
        "index_file": "index.json",
        "index_sha256": index_sha256,
        "index_size_bytes": index_size,
        "content_sha256": index["content_sha256"],
        "array_order": list(order),
        "array_count": len(order),
        "total_payload_bytes": total_payload_bytes,
        "array_inventory": observed_inventory,
    }
    return descriptor, loaded


__all__ = [
    "PRODUCTION_NEURAL_QUERY_NPY_INDEX_SCHEMA",
    "numerical_array_sha256",
    "validate_npy_array_set",
    "write_npy_array_set",
]
