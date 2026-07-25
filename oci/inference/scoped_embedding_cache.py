"""One authenticated immutable embedding cache with row-scoped views."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


SCOPED_EMBEDDING_CACHE_SCHEMA = "scoped_embedding_cache_v1"
PREPARED_STAGE1_CONTEXT_SCHEMA = "prepared_stage1_context_v1"


def _file_identity(path: Path) -> tuple[str, int, tuple[int, ...]]:
    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ValueError("shared cache payloads must be single-link regular files")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    identities = tuple(
        int(value)
        for value in (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
    )
    for observed in (opened, closed, after):
        if tuple(
            int(value)
            for value in (
                observed.st_dev,
                observed.st_ino,
                observed.st_mode,
                observed.st_nlink,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            )
        ) != identities:
            raise RuntimeError("shared cache payload changed while being authenticated")
    return digest.hexdigest(), int(before.st_size), identities


def _load_row_ids(path: Path, *, column: str) -> tuple[str, ...]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, columns=[column])
        values = frame[column].tolist()
    elif suffix == ".npy":
        array = np.load(path, allow_pickle=False)
        if array.ndim != 1:
            raise ValueError("embedding-cache row-ID NPY must be one-dimensional")
        values = array.tolist()
    else:
        raise ValueError("embedding-cache row IDs must use Parquet or non-object NPY")
    normalized = tuple(str(value) for value in values)
    if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("embedding-cache row IDs must be nonempty and unique")
    return normalized


class SharedEmbeddingCache:
    """Process-local authenticated handle to one immutable dense cache."""

    def __init__(
        self,
        *,
        embedding_path: Path,
        row_ids_path: Path,
        row_id_column: str = "row_id",
    ) -> None:
        self.embedding_path = Path(embedding_path).resolve(strict=True)
        self.row_ids_path = Path(row_ids_path).resolve(strict=True)
        self.row_id_column = str(row_id_column)
        self._file_inventory = MappingProxyType(
            {
                "embeddings": _file_identity(self.embedding_path),
                "row_ids": _file_identity(self.row_ids_path),
            }
        )
        embeddings = np.load(self.embedding_path, mmap_mode="r", allow_pickle=False)
        if embeddings.dtype.hasobject or embeddings.ndim != 2:
            raise ValueError("embedding cache must be one numeric two-dimensional NPY array")
        row_ids = _load_row_ids(self.row_ids_path, column=self.row_id_column)
        if len(row_ids) != int(embeddings.shape[0]):
            raise ValueError("embedding-cache row ID and array row counts differ")
        self._embeddings = embeddings
        self._row_ids = row_ids
        self._row_index = MappingProxyType(
            {row_id: index for index, row_id in enumerate(row_ids)}
        )

    @property
    def identity(self) -> Mapping[str, Any]:
        payload = {
            "schema_version": SCOPED_EMBEDDING_CACHE_SCHEMA,
            "embedding_sha256": self._file_inventory["embeddings"][0],
            "embedding_size_bytes": self._file_inventory["embeddings"][1],
            "row_ids_sha256": self._file_inventory["row_ids"][0],
            "row_ids_size_bytes": self._file_inventory["row_ids"][1],
            "row_count": len(self._row_ids),
            "embedding_dimension": int(self._embeddings.shape[1]),
            "dtype": str(self._embeddings.dtype),
        }
        payload["content_sha256"] = hashlib.sha256(
            repr(sorted(payload.items())).encode("utf-8")
        ).hexdigest()
        return MappingProxyType(payload)

    def assert_unchanged(self) -> None:
        for name, path in (
            ("embeddings", self.embedding_path),
            ("row_ids", self.row_ids_path),
        ):
            if _file_identity(path) != self._file_inventory[name]:
                raise RuntimeError("authenticated embedding cache changed")

    def scoped_view(self, selected_row_ids: Sequence[Any]) -> "ScopedEmbeddingView":
        return ScopedEmbeddingView(self, selected_row_ids)


class ScopedEmbeddingView:
    """Read-only row-limited API supplied to scope fitting code."""

    __slots__ = ("_cache", "_row_ids", "_allowed", "_positions")

    def __init__(
        self,
        cache: SharedEmbeddingCache,
        selected_row_ids: Sequence[Any],
    ) -> None:
        if not isinstance(cache, SharedEmbeddingCache):
            raise TypeError("ScopedEmbeddingView requires an authenticated shared cache")
        row_ids = tuple(str(value) for value in selected_row_ids)
        if not row_ids or len(row_ids) != len(set(row_ids)):
            raise ValueError("scoped embedding row IDs must be nonempty and unique")
        missing = [value for value in row_ids if value not in cache._row_index]
        if missing:
            raise KeyError(f"scoped embedding cache lacks selected rows: {missing[:3]}")
        self._cache = cache
        self._row_ids = row_ids
        self._allowed = frozenset(row_ids)
        self._positions = tuple(cache._row_index[value] for value in row_ids)

    @property
    def row_ids(self) -> tuple[str, ...]:
        return self._row_ids

    @property
    def shape(self) -> tuple[int, int]:
        return len(self._row_ids), int(self._cache._embeddings.shape[1])

    def __len__(self) -> int:
        return len(self._row_ids)

    def _validate_request(self, row_ids: Sequence[Any]) -> tuple[str, ...]:
        requested = tuple(str(value) for value in row_ids)
        forbidden = [value for value in requested if value not in self._allowed]
        if forbidden:
            raise PermissionError(
                "scope attempted to access peer rows outside its fit projection: "
                f"{forbidden[:3]}"
            )
        return requested

    def take(self, row_ids: Sequence[Any] | None = None) -> np.ndarray:
        requested = self._row_ids if row_ids is None else self._validate_request(row_ids)
        positions = [self._cache._row_index[value] for value in requested]
        value = np.asarray(self._cache._embeddings[positions]).copy()
        value.setflags(write=False)
        return value

    def get(self, row_id: Any) -> np.ndarray:
        return self.take((row_id,))[0]

    def iter_batches(
        self,
        *,
        batch_size: int,
    ) -> Iterator[tuple[tuple[str, ...], np.ndarray]]:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self._row_ids), int(batch_size)):
            batch_ids = self._row_ids[start : start + int(batch_size)]
            yield batch_ids, self.take(batch_ids)


@dataclass(frozen=True)
class PreparedStage1Context:
    """Reusable prepared inputs shared by preflight, canary, fit, and validation."""

    context_id: str
    fit_row_ids: tuple[str, ...]
    heldout_row_ids: tuple[str, ...]
    embedding_view: ScopedEmbeddingView
    split_identity: str
    prepared_cohort_identity: str
    configuration_identity: str
    schema_version: str = PREPARED_STAGE1_CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PREPARED_STAGE1_CONTEXT_SCHEMA:
            raise ValueError("unsupported prepared Stage 1 context schema")
        if not self.context_id:
            raise ValueError("prepared Stage 1 context requires an ID")
        if not self.fit_row_ids or len(self.fit_row_ids) != len(set(self.fit_row_ids)):
            raise ValueError("fit row IDs must be nonempty and unique")
        if set(self.fit_row_ids) & set(self.heldout_row_ids):
            raise ValueError("fit and held-out rows must be disjoint")
        if tuple(self.embedding_view.row_ids) != tuple(self.fit_row_ids):
            raise ValueError("prepared context embedding view must expose fit rows only")
        for name, value in (
            ("split_identity", self.split_identity),
            ("prepared_cohort_identity", self.prepared_cohort_identity),
            ("configuration_identity", self.configuration_identity),
        ):
            if len(str(value)) != 64:
                raise ValueError(f"{name} must be one SHA-256 identity")


__all__ = [
    "PREPARED_STAGE1_CONTEXT_SCHEMA",
    "PreparedStage1Context",
    "SCOPED_EMBEDDING_CACHE_SCHEMA",
    "ScopedEmbeddingView",
    "SharedEmbeddingCache",
]
