"""Spawn-worker adapter for the legacy all-source Stage 1 component.

The legacy scientific implementation historically fit every exact and
cumulative scope in one process.  This adapter preserves those native
producers while moving the unit of execution to one canonical scope:

* the parent publishes one immutable text-only cohort plus one fit-label
  projection per physical-fit owner;
* a spawned worker authenticates the descriptor, loads only its fit labels,
  and executes exactly one scope on its assigned device;
* the worker removes all aggregate indexes before sealing its scope-owned
  fragment; and
* the parent authenticates all fragments before publishing the collision-safe
  merge used by the component finalizer; and
* deduplicated logical purposes fail closed until the legacy role-specific
  accumulator is replaced by a role-neutral physical evidence artifact.

No source dataset path or held-out treatment/outcome vector is present in the
worker descriptor.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .production_stage1_legacy_scope_fragments import (
    LEGACY_STAGE1_ROLE_NEUTRAL_BINDING_SET_SCHEMA,
    durably_sync_legacy_stage1_tree,
    merge_legacy_stage1_scope_fragments,
    seal_legacy_stage1_scope_fragment,
    validate_legacy_stage1_fragment_merge,
    validate_legacy_stage1_fragment_merge_from_path,
    validate_legacy_stage1_scope_fragment,
)
from .production_stage1_scope_scheduler import (
    STAGE1_LOGICAL_SCOPE_BINDING_FILENAME,
    Stage1ScopeAssignment,
    Stage1ScopeAttemptStore,
    Stage1ScopeExecutionRequest,
    Stage1ScopePlan,
    Stage1ScopeSpec,
    ValidatedStage1ScopeAttempt,
    derive_stage1_group_seed,
)

LEGACY_STAGE1_SCOPE_DESCRIPTOR_SCHEMA = "production_legacy_stage1_scope_worker_descriptor_v3"
LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_SCHEMA = (
    "production_legacy_stage1_scope_worker_descriptor_set_v3"
)
LEGACY_STAGE1_ONE_SCOPE_AUTHORITY_SCHEMA = "production_legacy_stage1_one_scope_authority_v1"
LEGACY_STAGE1_SCOPE_DESCRIPTOR_RECOVERY_SCHEMA = (
    "production_legacy_stage1_scope_descriptor_recovery_v2"
)
LEGACY_STAGE1_SCOPE_DESCRIPTOR_MANIFEST = "descriptor_manifest.json"
LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_MANIFEST = "descriptor_set_manifest.json"
LEGACY_STAGE1_SCOPE_DESCRIPTOR_RECOVERY_MANIFEST = "recovery_manifest.json"
LEGACY_STAGE1_SCOPE_WORKER_TARGET = (
    "oci.inference.production_stage1_legacy_scope_adapter:" "run_legacy_stage1_scope_worker"
)


class LegacyStage1RoleSpecificDeduplicationError(RuntimeError):
    """A logical alias cannot safely reinterpret a role-specific fragment."""

_ROW_ID = "__production_stage1_scope_row_id_v1__"
_TEXT_FILE = "visible_text.parquet"
_CONFIG_FILE = "effective_config.json"
_AUTHORITY_FILE = "one_scope_authority.json"
_PREFLIGHT_FILE = "cluster_preflight_projection.json"
_FIT_LABEL_FILE = "fit_labels.parquet"
_FRAGMENT_DIRECTORY = "legacy_fragment"
_FRAGMENT_ARTIFACT_DIRECTORY = "artifacts"
_HEX = frozenset("0123456789abcdef")

# These files describe an aggregate component.  A scope worker may use the
# local versions to validate its native output, but they must never enter a
# merge fragment.  The parent reconstructs every one from the authenticated
# logical scope plan after every required physical/logical result authenticates.
LEGACY_STAGE1_AGGREGATE_RELATIVE_PATHS = frozenset(
    {
        "handoff/discovery_contexts.jsonl",
        "handoff/manifest.json",
        "bow_native_family_proof_index.json",
        "htr_native_family_proof_index.json",
        "matched_pair_native_family_proof_index.json",
        "embedding_native_family_proof_index.json",
        "cumulative_legacy_native_family_proof_index.json",
        "cumulative_embedding_native_family_proof_index.json",
        "embedding_cluster_fit_index.json",
        "exact_scope_index.json",
    }
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


def _sha256_file(path: Path) -> tuple[str, int]:
    """Hash one stable, singly-linked regular file without materializing it."""

    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ValueError(
                "scope descriptor inputs must be singly-linked regular files: " f"{path}"
            )
        digest = hashlib.sha256()
        total = 0
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            total += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if tuple(getattr(before, field) for field in identity_fields) != tuple(
        getattr(after, field) for field in identity_fields
    ):
        raise RuntimeError(f"scope descriptor file changed while hashing: {path}")
    if total != int(after.st_size):
        raise RuntimeError(f"scope descriptor file changed length while hashing: {path}")
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or int(named.st_nlink) != 1
        or (int(named.st_dev), int(named.st_ino)) != (int(after.st_dev), int(after.st_ino))
    ):
        raise RuntimeError(f"scope descriptor path was substituted while hashing: {path}")
    return digest.hexdigest(), total


def _read_file_bytes(
    path: Path,
    *,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> tuple[bytes, str, os.stat_result]:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ValueError(f"scope descriptor inputs must be singly-linked regular files: {path}")
        if int(before.st_size) > int(maximum_bytes):
            raise ValueError(f"scope descriptor JSON is unexpectedly large: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError(f"scope descriptor file changed while hashing: {path}")
    payload = b"".join(chunks)
    if len(payload) != int(after.st_size):
        raise RuntimeError(f"scope descriptor file changed length while hashing: {path}")
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(named.st_mode)
        or int(named.st_nlink) != 1
        or (int(named.st_dev), int(named.st_ino)) != (int(after.st_dev), int(after.st_ino))
    ):
        raise RuntimeError(f"scope descriptor path was substituted while hashing: {path}")
    return payload, digest.hexdigest(), after


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be one regular JSON file")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    payload, _digest, _identity = _read_file_bytes(path)
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"{label} contains non-finite constant {value}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one object")
    return value


def _write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    compact: bool = False,
) -> None:
    if compact:
        payload = (_canonical_json(value) + "\n").encode("utf-8")
    else:
        payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable descriptor file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable descriptor file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".parquet",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _file_registration(path: Path, root: Path) -> dict[str, Any]:
    digest, size = _sha256_file(path)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": size,
    }


def _validate_registration(
    root: Path,
    registration: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    if set(registration) != {"relative_path", "sha256", "size_bytes"}:
        raise ValueError(f"{label} registration is not closed")
    relative_text = str(registration["relative_path"])
    relative = Path(relative_text)
    if (
        not relative_text
        or relative.is_absolute()
        or relative.as_posix() != relative_text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} registration escapes its descriptor")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} file is absent")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} file escapes its descriptor") from exc
    digest, size = _sha256_file(path)
    if digest != registration["sha256"] or size != int(registration["size_bytes"]):
        raise ValueError(f"{label} file changed")
    return path


def _closed_tree_inventory(root: Path, *, label: str) -> tuple[set[str], set[str]]:
    """Inventory every entry while rejecting links and special files."""

    root_stat = os.lstat(root)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"{label} root must be one real directory")
    files: set[str] = set()
    directories: set[str] = set()
    pending = [root]
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as iterator:
            entries = list(iterator)
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            observed = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"{label} contains a symbolic link: {relative}")
            if stat.S_ISDIR(observed.st_mode):
                directories.add(relative)
                pending.append(path)
                continue
            if not stat.S_ISREG(observed.st_mode) or int(observed.st_nlink) != 1:
                raise ValueError(f"{label} contains a special or multiply-linked file: {relative}")
            files.add(relative)
    root_after = os.lstat(root)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if tuple(getattr(root_stat, field) for field in identity_fields) != tuple(
        getattr(root_after, field) for field in identity_fields
    ):
        raise RuntimeError(f"{label} root changed during inventory")
    return files, directories


class _RestrictedLogicalIdentityEmbeddingCache:
    """Physical fit-only cache capability with the original logical identity."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        logical_identity: Mapping[str, Any],
        allowed_row_ids: Sequence[int],
    ) -> None:
        from .review_spent_evidence_provider import (
            SpentOnlyFrozenChunkEmbeddingCache,
        )

        self._cache = SpentOnlyFrozenChunkEmbeddingCache(cache_dir)
        self.cache_dir = self._cache.cache_dir
        self._logical_identity = copy.deepcopy(dict(logical_identity))
        self._allowed_row_ids = frozenset(map(int, allowed_row_ids))
        # Construction has already authenticated and privately snapshotted all
        # four cache files.  Retain that identity instead of rehashing a
        # potentially multi-gigabyte matrix on every provider identity call.
        self._physical_identity = copy.deepcopy(dict(self._cache._identity))
        # Bound providers intentionally receive this facade as their cache so
        # proof identities remain equal to the parent's original logical cache.
        self._metadata = self._cache._metadata
        self._embeddings = self._cache._embeddings
        self._offsets = self._cache._offsets

    @property
    def row_count(self) -> int:
        return self._cache.row_count

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._cache.metadata

    def physical_identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._physical_identity)

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._logical_identity)

    def bind_spent(
        self,
        row_ids: Sequence[int],
        texts: Sequence[str],
    ) -> Any:
        requested = tuple(map(int, row_ids))
        if not set(requested).issubset(self._allowed_row_ids):
            raise ValueError("private embedding cache refuses a non-fit row")
        # Reimplement the narrow constructor only to bind this logical facade.
        from .review_spent_evidence_provider import (
            BoundSpentFrozenChunkEmbeddingProvider,
        )

        physical = self._cache.bind_spent(requested, tuple(texts))
        return BoundSpentFrozenChunkEmbeddingProvider(
            cache=self,
            row_ids=physical.row_ids,
            cached_by_row=physical.cached_by_row,
            token_bounded_row_ids=physical.token_bounded_row_ids,
        )


def _write_private_embedding_cache(
    *,
    root: Path,
    prepared: Any,
    scope: Any,
) -> Mapping[str, Any]:
    cache_root = root / "private_embedding_cache"
    cache_root.mkdir(parents=True, exist_ok=False)
    source = prepared.embedding_cache
    allowed = frozenset(map(int, scope.fit_row_ids))
    row_count = int(source.row_count)
    offsets = np.zeros(row_count + 1, dtype=np.int64)
    total_chunks = sum(
        int(source._offsets[row_id + 1]) - int(source._offsets[row_id]) for row_id in allowed
    )
    hidden_size = int(source._embeddings.shape[1])
    matrix_path = cache_root / "chunk_embeddings.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=source._embeddings.dtype,
        shape=(total_chunks, hidden_size),
    )
    cursor = 0
    chunk_rows: list[bytes] = []
    for row_id in range(row_count):
        if row_id in allowed:
            start = int(source._offsets[row_id])
            stop = int(source._offsets[row_id + 1])
            count = stop - start
            matrix[cursor : cursor + count] = source._embeddings[start:stop]
            chunks = list(source._cached_chunks(row_id))
            if len(chunks) != count:
                raise RuntimeError("source cache chunk registry changed")
            cursor += count
        else:
            chunks = []
        offsets[row_id + 1] = cursor
        chunk_rows.append(
            (json.dumps({"chunks": chunks}, ensure_ascii=False) + "\n").encode("utf-8")
        )
    matrix.flush()
    del matrix
    with matrix_path.open("rb") as handle:
        os.fsync(handle.fileno())
    with (cache_root / "offsets.npy").open("xb") as handle:
        np.save(handle, offsets, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    with (cache_root / "chunk_texts.jsonl").open("xb") as handle:
        for row in chunk_rows:
            handle.write(row)
        handle.flush()
        os.fsync(handle.fileno())
    source_metadata = source.metadata
    metadata = {
        key: copy.deepcopy(source_metadata.get(key))
        for key in (
            "num_samples",
            "hidden_size",
            "sentence_model_name",
            "chunk_size_words",
            "chunk_overlap_words",
            "max_chunks",
            "chunk_selection",
            "normalize_embeddings",
            "max_seq_length",
        )
    }
    metadata["private_scope_view"] = {
        "schema_version": "production_stage1_private_embedding_cache_view_v1",
        "scope_id": scope.scope_id,
        "allowed_row_ids_sha256": _sha256_json(list(scope.fit_row_ids)),
        "source_logical_identity_sha256": _sha256_json(dict(prepared.embedding_cache_identity)),
        "nonfit_chunk_bytes_present": False,
    }
    _write_json(cache_root / "metadata.json", metadata)
    physical = _RestrictedLogicalIdentityEmbeddingCache(
        cache_dir=cache_root,
        logical_identity=prepared.embedding_cache_identity,
        allowed_row_ids=scope.fit_row_ids,
    )
    if physical.row_count != row_count or physical.identity() != dict(
        prepared.embedding_cache_identity
    ):
        raise RuntimeError("private cache logical identity binding failed")
    for row_id in set(range(row_count)) - allowed:
        if int(physical._offsets[row_id]) != int(
            physical._offsets[row_id + 1]
        ) or physical._cache._cached_chunks(row_id):
            raise RuntimeError("private cache retained non-fit chunks")
    return {
        "relative_path": cache_root.relative_to(root).as_posix(),
        "allowed_row_ids_sha256": _sha256_json(list(scope.fit_row_ids)),
        "logical_identity": copy.deepcopy(dict(prepared.embedding_cache_identity)),
        "physical_identity": copy.deepcopy(dict(physical.physical_identity())),
        "files": {
            filename: _file_registration(cache_root / filename, root)
            for filename in (
                "metadata.json",
                "chunk_embeddings.npy",
                "offsets.npy",
                "chunk_texts.jsonl",
            )
        },
    }


def _one_scope_authority_from_prepared(
    *,
    prepared: Any,
    scope: Stage1ScopeSpec,
) -> Mapping[str, Any]:
    """Project the canonical plan to exactly one worker-authorized scope."""

    from . import production_stage1_bundle as bundle

    if prepared.stage1_scope_plan.scope(scope.scope_id).as_dict() != scope.as_dict():
        raise ValueError("selected scope differs from the canonical parent plan")
    assignment = prepared.stage1_scope_plan.assignment(scope.scope_id)
    schedule = bundle._canonical_cumulative_spent_schedule(
        prepared.registry,
        initial_training_partitions=(
            prepared.stage1_scope_plan.initial_training_partitions
        ),
    )
    split_scope_fingerprint: str | None
    if scope.scope_kind == "exact_inner":
        split = bundle._canonical_exact_registry_from_wrapper(prepared.registry).inner_split(
            int(scope.outer_fold), int(scope.inner_fold)
        )
        if split.fit_row_ids != scope.fit_row_ids or split.heldout_row_ids != scope.heldout_row_ids:
            raise ValueError("selected exact-inner scope differs from its registry")
        split_scope_fingerprint = split.scope_fingerprint
    elif scope.scope_kind == "cumulative_spent":
        matches = tuple(row for row in schedule.scopes if row.scope_id == scope.scope_id)
        if len(matches) != 1:
            raise ValueError("selected cumulative scope is absent from its schedule")
        selected = matches[0]
        if (
            tuple(selected.spent_row_ids) != scope.fit_row_ids
            or tuple(selected.sealed_row_ids) != scope.heldout_row_ids
            or int(selected.outer_fold) != scope.outer_fold
            or int(selected.context_epoch) != scope.context_epoch
            or int(selected.provider_inner_fold) != scope.provider_inner_fold
        ):
            raise ValueError("selected cumulative scope differs from its schedule")
        split_scope_fingerprint = selected.split_fingerprint
    elif scope.scope_kind == "full_outer":
        split_scope_fingerprint = None
    else:  # pragma: no cover - canonical plan construction rejects this first.
        raise ValueError("selected scope has an unsupported kind")
    body = {
        "schema_version": LEGACY_STAGE1_ONE_SCOPE_AUTHORITY_SCHEMA,
        "stage1_request_sha256": str(prepared.request_sha256),
        "registry_content_sha256": str(prepared.registry_content_sha256),
        "plan_content_sha256": str(prepared.stage1_scope_plan.content_sha256),
        "cumulative_schedule_sha256": str(schedule.schedule_sha256),
        "dataset_row_count": int(len(prepared.modeling_data)),
        "scope": scope.as_dict(),
        "assignment": assignment.as_dict(),
        "split_scope_fingerprint": split_scope_fingerprint,
        "registry_scope_binding_sha256": _sha256_json(
            {
                "registry_content_sha256": str(prepared.registry_content_sha256),
                "scope": scope.as_dict(),
            }
        ),
        "plan_scope_binding_sha256": _sha256_json(
            {
                "plan_content_sha256": str(prepared.stage1_scope_plan.content_sha256),
                "scope": scope.as_dict(),
                "assignment": assignment.as_dict(),
            }
        ),
        "authorized_scope_count": 1,
        "other_scope_definitions_supplied": False,
        "other_scope_row_identities_supplied": False,
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _scope_spec_from_authority(
    value: Any,
    *,
    dataset_row_count: int,
) -> Stage1ScopeSpec:
    if not isinstance(value, Mapping):
        raise ValueError("one-scope authority lacks its selected scope")
    required = {
        "canonical_index",
        "scope_id",
        "scope_kind",
        "outer_fold",
        "inner_fold",
        "context_epoch",
        "provider_inner_fold",
        "fit_row_ids",
        "heldout_row_ids",
        "fit_row_count",
        "heldout_row_count",
        "fit_row_order_fingerprint",
        "heldout_row_order_fingerprint",
        "global_seed",
        "scope_seed",
        "heldout_labels_supplied",
        "scope_sha256",
    }
    if set(value) != required:
        raise ValueError("one-scope authority selected scope is not closed")
    for key in (
        "canonical_index",
        "outer_fold",
        "global_seed",
        "scope_seed",
        "fit_row_count",
        "heldout_row_count",
    ):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int):
            raise ValueError(f"one-scope authority {key} must be an integer")
    optional_integers = ("inner_fold", "context_epoch", "provider_inner_fold")
    for key in optional_integers:
        child = value.get(key)
        if child is not None and (isinstance(child, bool) or not isinstance(child, int)):
            raise ValueError(f"one-scope authority {key} must be null or an integer")
    fit = value.get("fit_row_ids")
    heldout = value.get("heldout_row_ids")
    if (
        not isinstance(fit, list)
        or not isinstance(heldout, list)
        or not fit
        or not heldout
        or any(isinstance(row, bool) or not isinstance(row, int) for row in (*fit, *heldout))
    ):
        raise ValueError("one-scope authority row identities are malformed")
    spec = Stage1ScopeSpec(
        canonical_index=int(value["canonical_index"]),
        scope_id=str(value["scope_id"]),
        scope_kind=str(value["scope_kind"]),
        outer_fold=int(value["outer_fold"]),
        inner_fold=(None if value["inner_fold"] is None else int(value["inner_fold"])),
        context_epoch=(None if value["context_epoch"] is None else int(value["context_epoch"])),
        provider_inner_fold=(
            None if value["provider_inner_fold"] is None else int(value["provider_inner_fold"])
        ),
        fit_row_ids=tuple(map(int, fit)),
        heldout_row_ids=tuple(map(int, heldout)),
        global_seed=int(value["global_seed"]),
        scope_seed=int(value["scope_seed"]),
    )
    if (
        spec.as_dict() != dict(value)
        or spec.canonical_index < 0
        or spec.outer_fold < 1
        or spec.global_seed < 0
        or spec.scope_seed
        != derive_stage1_group_seed(spec.global_seed, spec.fit_row_ids)
        or len(set(spec.fit_row_ids)) != len(spec.fit_row_ids)
        or len(set(spec.heldout_row_ids)) != len(spec.heldout_row_ids)
        or set(spec.fit_row_ids) & set(spec.heldout_row_ids)
        or any(
            row < 0 or row >= int(dataset_row_count)
            for row in (*spec.fit_row_ids, *spec.heldout_row_ids)
        )
    ):
        raise ValueError("one-scope authority selected scope is invalid")
    expected_scope_id: str
    if spec.scope_kind == "full_outer":
        expected_scope_id = f"outer_{spec.outer_fold:03d}_full"
        kind_fields_valid = (
            spec.inner_fold is None
            and spec.context_epoch is None
            and spec.provider_inner_fold is None
            and set(spec.fit_row_ids) | set(spec.heldout_row_ids)
            == set(range(int(dataset_row_count)))
        )
    elif spec.scope_kind == "exact_inner":
        expected_scope_id = f"outer_{spec.outer_fold:03d}_inner_{int(spec.inner_fold or 0):03d}"
        kind_fields_valid = (
            spec.inner_fold is not None
            and spec.inner_fold > 0
            and spec.context_epoch is None
            and spec.provider_inner_fold is None
        )
    elif spec.scope_kind == "cumulative_spent":
        expected_scope_id = (
            f"outer_{spec.outer_fold:03d}_hierarchy_epoch_"
            f"{int(spec.context_epoch if spec.context_epoch is not None else -1):03d}"
        )
        kind_fields_valid = (
            spec.inner_fold is None
            and spec.context_epoch is not None
            and spec.context_epoch >= 0
            and spec.provider_inner_fold == spec.context_epoch + 1
        )
    else:
        raise ValueError("one-scope authority selected scope kind is invalid")
    if spec.scope_id != expected_scope_id or not kind_fields_valid:
        raise ValueError("one-scope authority selected scope semantics changed")
    return spec


def _scope_assignment_from_authority(
    value: Any,
    *,
    scope: Stage1ScopeSpec,
) -> Stage1ScopeAssignment:
    if not isinstance(value, Mapping) or set(value) != {
        "scope_id",
        "gpu_id",
        "execution_rank",
        "fit_row_count",
        "assigned_gpu_load_after",
    }:
        raise ValueError("one-scope authority assignment is not closed")
    for key in ("execution_rank", "fit_row_count", "assigned_gpu_load_after"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int):
            raise ValueError(f"one-scope authority assignment {key} is invalid")
    gpu_id = value.get("gpu_id")
    if gpu_id is not None and (
        isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0
    ):
        raise ValueError("one-scope authority assignment gpu_id is invalid")
    assignment = Stage1ScopeAssignment(
        scope_id=str(value["scope_id"]),
        gpu_id=None if gpu_id is None else int(gpu_id),
        execution_rank=int(value["execution_rank"]),
        fit_row_count=int(value["fit_row_count"]),
        assigned_gpu_load_after=int(value["assigned_gpu_load_after"]),
    )
    if (
        assignment.as_dict() != dict(value)
        or assignment.scope_id != scope.scope_id
        or assignment.execution_rank < 0
        or assignment.fit_row_count != scope.fit_row_count
        or assignment.assigned_gpu_load_after < assignment.fit_row_count
    ):
        raise ValueError("one-scope authority assignment changed")
    return assignment


@dataclass(frozen=True)
class AuthenticatedLegacyStage1ScopeDescriptor:
    root: Path
    manifest: Mapping[str, Any]
    authority: Mapping[str, Any]
    scope: Stage1ScopeSpec
    assignment: Stage1ScopeAssignment
    scope_id: str
    embedding_cache: Any | None = None

    @property
    def manifest_path(self) -> Path:
        return self.root / LEGACY_STAGE1_SCOPE_DESCRIPTOR_MANIFEST

    @property
    def stage1_request_sha256(self) -> str:
        return str(self.manifest["stage1_request_sha256"])

    @property
    def plan_content_sha256(self) -> str:
        return str(self.authority["plan_content_sha256"])

    def worker_parameters(self) -> dict[str, Any]:
        return {
            "descriptor_manifest_path": str(self.manifest_path),
            "stage1_request_sha256": self.stage1_request_sha256,
            "scope_id": self.scope_id,
        }


@dataclass(frozen=True)
class AuthenticatedLegacyStage1ScopeDescriptorSet:
    root: Path
    manifest: Mapping[str, Any]
    descriptors: Mapping[str, AuthenticatedLegacyStage1ScopeDescriptor]

    def worker_parameters_by_scope(self) -> dict[str, Mapping[str, Any]]:
        return {
            scope_id: descriptor.worker_parameters()
            for scope_id, descriptor in self.descriptors.items()
        }


def _visible_text_row_ids(scope: Any) -> tuple[int, ...]:
    if scope.scope_kind == "cumulative_spent":
        return tuple(scope.fit_row_ids)
    visible = set(scope.fit_row_ids) | set(scope.heldout_row_ids)
    return tuple(sorted(visible))


def _scope_preflight_projection(*, prepared: Any, scope: Any) -> Mapping[str, Any]:
    matches = [
        row
        for row in prepared.embedding_cluster_feasibility_audit.get("scopes") or ()
        if isinstance(row, Mapping) and row.get("scope_id") == scope.scope_id
    ]
    if len(matches) != 1:
        raise ValueError(f"full clustered preflight lacks {scope.scope_id}")
    identity = copy.deepcopy(dict(matches[0]["cluster_fit_identity"]))
    body = {
        "schema_version": "production_stage1_private_cluster_preflight_projection_v1",
        "full_preflight_content_sha256": str(
            prepared.embedding_cluster_feasibility_audit["content_sha256"]
        ),
        "full_scope_order_sha256": _sha256_json(
            [item.scope_id for item in prepared.stage1_scope_plan.scopes]
        ),
        "canonical_index": int(scope.canonical_index),
        "scope_id": scope.scope_id,
        "scope_kind": scope.scope_kind,
        "scope_binding_sha256": _sha256_json(
            {
                "registry_content_sha256": prepared.registry_content_sha256,
                "scope": scope.as_dict(),
            }
        ),
        "cluster_fit_identity": identity,
        "cluster_fit_identity_sha256": str(identity["content_sha256"]),
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _descriptor_body_from_prepared(
    *,
    prepared: Any,
    scope: Any,
    files: Mapping[str, Any],
    private_cache: Mapping[str, Any],
) -> dict[str, Any]:
    plan = prepared.stage1_scope_plan
    return {
        "schema_version": LEGACY_STAGE1_SCOPE_DESCRIPTOR_SCHEMA,
        "stage1_request_sha256": str(prepared.request_sha256),
        "registry_content_sha256": str(prepared.registry_content_sha256),
        "plan_content_sha256": str(plan.content_sha256),
        "scope": scope.as_dict(),
        "row_count": int(len(prepared.modeling_data)),
        "columns": {
            "row_id": _ROW_ID,
            "text": str(prepared.config.text_column),
            "treatment": str(prepared.config.treatment_column),
            "outcome": str(prepared.config.outcome_column),
        },
        "files": copy.deepcopy(dict(files)),
        "embedding_cache": copy.deepcopy(dict(private_cache)),
        "htr_model": {
            "path": str(prepared.htr_model_path),
            "tree_sha256": str(prepared.htr_model_sha256),
        },
        "behavior_identity": copy.deepcopy(dict(prepared.behavior_identity)),
        "runtime": {
            "global_seed": int(plan.global_seed),
            "num_workers": int(prepared.options.num_workers),
            "scope_workers_per_gpu": int(prepared.options.scope_workers_per_gpu),
        },
        "visible_text_policy": ("fit_plus_heldout_for_exact_fit_only_for_cumulative_v1"),
        "fit_label_policy": "canonical_fit_rows_only_v1",
        "scope_authority_policy": "one_selected_scope_no_peer_definitions_v1",
        "cluster_preflight_policy": "one_parent_validated_scope_record_v1",
        "heldout_labels_supplied_to_worker": False,
        "full_split_registry_supplied_to_worker": False,
        "full_scope_plan_supplied_to_worker": False,
        "other_scope_row_identities_supplied_to_worker": False,
        "other_scope_preflight_supplied_to_worker": False,
        "source_dataset_path_supplied_to_worker": False,
    }


def _scope_private_effective_config(
    *,
    prepared: Any,
    public_scope_root: Path,
) -> dict[str, Any]:
    """Rebind every runtime data/cache path to the immutable private scope."""

    private_text_path = str(public_scope_root / _TEXT_FILE)
    private_cache_path = str(public_scope_root / "private_embedding_cache")

    def rewrite(value: Any, *, key: str | None = None) -> Any:
        if key == "dataset_path":
            return private_text_path
        if key == "cache_dir":
            return private_cache_path
        if key == "external_corpus_cache_dirs":
            return []
        if isinstance(value, Mapping):
            return {
                str(child_key): rewrite(child_value, key=str(child_key))
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        return copy.deepcopy(value)

    result = rewrite(dict(prepared.request["effective_stage1_config"]))
    if not isinstance(result, dict):
        raise RuntimeError("scope-private effective configuration is malformed")
    forbidden_paths = {
        str(Path(prepared.options.dataset_path).resolve()),
        str(Path(prepared.embedding_cache_path).resolve()),
    }
    serialized = _canonical_json(result)
    if any(forbidden in serialized for forbidden in forbidden_paths):
        raise ValueError("scope-private config still exposes a label cohort or global cache path")
    return result


def _write_scope_descriptor(
    *,
    root: Path,
    public_scope_root: Path,
    prepared: Any,
    scope: Any,
) -> None:
    if root.exists():
        if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
            raise FileExistsError("scope descriptor attempt root must be fresh and empty")
    else:
        root.mkdir(parents=True, exist_ok=False)
    private_config = _scope_private_effective_config(
        prepared=prepared,
        public_scope_root=public_scope_root,
    )
    _write_json(
        root / _CONFIG_FILE,
        private_config,
    )
    _write_json(
        root / _AUTHORITY_FILE,
        _one_scope_authority_from_prepared(prepared=prepared, scope=scope),
    )
    _write_json(
        root / _PREFLIGHT_FILE,
        _scope_preflight_projection(prepared=prepared, scope=scope),
        compact=True,
    )
    visible_ids = list(_visible_text_row_ids(scope))
    text = pd.DataFrame(
        {
            _ROW_ID: np.asarray(visible_ids, dtype=np.int64),
            prepared.config.text_column: prepared.modeling_data.iloc[visible_ids][
                prepared.config.text_column
            ].to_numpy(copy=True),
        }
    )
    _write_parquet(root / _TEXT_FILE, text)
    fit_ids = list(scope.fit_row_ids)
    labels = prepared.modeling_data.iloc[fit_ids][
        [
            prepared.config.treatment_column,
            prepared.config.outcome_column,
        ]
    ].copy()
    labels.insert(0, _ROW_ID, np.asarray(fit_ids, dtype=np.int64))
    _write_parquet(root / _FIT_LABEL_FILE, labels)
    private_cache = _write_private_embedding_cache(
        root=root,
        prepared=prepared,
        scope=scope,
    )
    files = {
        "effective_config": _file_registration(root / _CONFIG_FILE, root),
        "one_scope_authority": _file_registration(root / _AUTHORITY_FILE, root),
        "cluster_preflight_projection": _file_registration(root / _PREFLIGHT_FILE, root),
        "visible_text": _file_registration(root / _TEXT_FILE, root),
        "fit_labels": _file_registration(root / _FIT_LABEL_FILE, root),
    }
    body = _descriptor_body_from_prepared(
        prepared=prepared,
        scope=scope,
        files=files,
        private_cache=private_cache,
    )
    # A scope manifest is its only reusable terminal marker.  All large cache
    # bytes and descriptor inputs must be durable before that marker is
    # published.
    durably_sync_legacy_stage1_tree(root)
    _write_json(
        root / LEGACY_STAGE1_SCOPE_DESCRIPTOR_MANIFEST,
        {**body, "content_sha256": _sha256_json(body)},
    )


def _fsync_directory_path(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"descriptor durability path is not a directory: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _descriptor_recovery_body(*, prepared: Any, descriptor_root: Path) -> dict[str, Any]:
    plan = prepared.stage1_scope_plan
    return {
        "schema_version": LEGACY_STAGE1_SCOPE_DESCRIPTOR_RECOVERY_SCHEMA,
        "stage1_request_sha256": str(prepared.request_sha256),
        "registry_content_sha256": str(prepared.registry_content_sha256),
        "plan_content_sha256": str(plan.content_sha256),
        "descriptor_root": str(descriptor_root),
        "physical_scope_order": [
            scope.scope_id for scope in plan.physical_scopes
        ],
        "physical_scope_count": len(plan.physical_scopes),
        "logical_scope_count": len(plan.scopes),
        "incomplete_attempts_preserved": True,
        "completed_scope_descriptors_reused_byte_for_byte": True,
    }


def _prepare_descriptor_recovery_root(
    *,
    prepared: Any,
    descriptor_root: Path,
) -> Path:
    recovery = descriptor_root.parent / f".{descriptor_root.name}.scope_descriptor_attempts"
    expected_body = _descriptor_recovery_body(
        prepared=prepared,
        descriptor_root=descriptor_root,
    )
    if recovery.exists() or recovery.is_symlink():
        if (
            recovery.is_symlink()
            or not recovery.is_dir()
            or recovery.resolve(strict=True) != recovery
        ):
            raise ValueError("scope descriptor recovery root is invalid")
        manifest = _read_json(
            recovery / LEGACY_STAGE1_SCOPE_DESCRIPTOR_RECOVERY_MANIFEST,
            label="scope descriptor recovery manifest",
        )
        body = {
            key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"
        }
        if (
            set(manifest) != {*expected_body, "content_sha256"}
            or body != expected_body
            or manifest.get("content_sha256") != _sha256_json(body)
        ):
            raise ValueError("scope descriptor recovery belongs to another request")
        allowed = {
            LEGACY_STAGE1_SCOPE_DESCRIPTOR_RECOVERY_MANIFEST,
            *(
                scope.scope_id
                for scope in prepared.stage1_scope_plan.physical_scopes
            ),
        }
        for entry in recovery.iterdir():
            if entry.name not in allowed:
                raise ValueError("scope descriptor recovery contains an unknown scope")
            if entry.name != LEGACY_STAGE1_SCOPE_DESCRIPTOR_RECOVERY_MANIFEST and (
                entry.is_symlink() or not entry.is_dir()
            ):
                raise ValueError("scope descriptor recovery contains an invalid scope entry")
        return recovery

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{recovery.name}.initializing-",
            dir=descriptor_root.parent,
        )
    )
    try:
        _write_json(
            temporary / LEGACY_STAGE1_SCOPE_DESCRIPTOR_RECOVERY_MANIFEST,
            {**expected_body, "content_sha256": _sha256_json(expected_body)},
        )
        durably_sync_legacy_stage1_tree(temporary)
        os.replace(temporary, recovery)
        _fsync_directory_path(descriptor_root.parent)
    except BaseException:
        # An interrupted initialization remains as evidence and cannot be
        # mistaken for the request-bound recovery root.
        raise
    return recovery


def _existing_reusable_descriptor_attempt(
    *,
    prepared: Any,
    scope: Any,
    attempt_parent: Path,
) -> Path | None:
    if not attempt_parent.exists():
        return None
    if attempt_parent.is_symlink() or not attempt_parent.is_dir():
        raise ValueError(f"scope descriptor attempt parent is invalid: {scope.scope_id}")
    reusable: list[Path] = []
    for candidate in sorted(attempt_parent.iterdir()):
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(f"scope descriptor attempt tree is invalid: {scope.scope_id}")
        terminal = candidate / LEGACY_STAGE1_SCOPE_DESCRIPTOR_MANIFEST
        if not terminal.exists():
            # Partial attempts are deliberately preserved but never reused.
            continue
        validate_legacy_stage1_scope_descriptor(
            descriptor_manifest_path=terminal,
            expected_stage1_request_sha256=prepared.request_sha256,
            expected_scope_id=scope.scope_id,
        )
        reusable.append(candidate)
    if len(reusable) > 1:
        raise ValueError(f"multiple sealed descriptor attempts exist: {scope.scope_id}")
    return reusable[0] if reusable else None


def publish_legacy_stage1_scope_descriptor(
    *,
    prepared: Any,
    descriptor_root: Path | str,
) -> AuthenticatedLegacyStage1ScopeDescriptorSet:
    """Atomically publish one private descriptor per physical-fit owner."""

    plan = prepared.stage1_scope_plan
    if not isinstance(plan, Stage1ScopePlan):
        raise TypeError("prepared Stage 1 scope plan has the wrong type")
    root = Path(descriptor_root)
    if not root.is_absolute():
        raise ValueError("scope descriptor root must be absolute")
    if root.is_symlink():
        raise ValueError("scope descriptor root cannot be a symlink")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.parent.is_symlink() or root.parent.resolve(strict=True) != root.parent:
        raise ValueError("scope descriptor parent must be canonical and symlink-free")
    if root.exists() and (root / LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_MANIFEST).exists():
        return validate_legacy_stage1_scope_descriptor_set(
            descriptor_root=root,
            expected_stage1_request_sha256=prepared.request_sha256,
            prepared=prepared,
        )
    root.mkdir(exist_ok=True)
    if root.resolve(strict=True) != root:
        raise ValueError("scope descriptor root must be canonical")
    _fsync_directory_path(root.parent)
    expected_scope_ids = {
        scope.scope_id for scope in plan.physical_scopes
    }
    for entry in root.iterdir():
        if entry.name not in expected_scope_ids or entry.is_symlink() or not entry.is_dir():
            raise ValueError("incomplete descriptor set contains an unexpected public entry")

    recovery = _prepare_descriptor_recovery_root(
        prepared=prepared,
        descriptor_root=root,
    )
    registrations: list[Mapping[str, Any]] = []
    for scope in plan.physical_scopes:
        scope_root = root / scope.scope_id
        if not scope_root.exists():
            attempt_parent = recovery / scope.scope_id
            attempt_parent.mkdir(parents=True, exist_ok=True)
            candidate = _existing_reusable_descriptor_attempt(
                prepared=prepared,
                scope=scope,
                attempt_parent=attempt_parent,
            )
            if candidate is None:
                candidate = Path(
                    tempfile.mkdtemp(
                        prefix="attempt_",
                        dir=attempt_parent,
                    )
                )
                # The candidate is never deleted on failure.  Only a terminal
                # manifest makes it eligible for recovery.
                _write_scope_descriptor(
                    root=candidate,
                    public_scope_root=scope_root,
                    prepared=prepared,
                    scope=scope,
                )
            _fsync_directory_path(attempt_parent)
            os.replace(candidate, scope_root)
            _fsync_directory_path(root)
            _fsync_directory_path(attempt_parent)
        descriptor = validate_legacy_stage1_scope_descriptor(
            descriptor_manifest_path=scope_root / LEGACY_STAGE1_SCOPE_DESCRIPTOR_MANIFEST,
            expected_stage1_request_sha256=prepared.request_sha256,
            expected_scope_id=scope.scope_id,
            prepared=prepared,
        )
        registrations.append(
            {
                "scope_id": scope.scope_id,
                "manifest": _file_registration(descriptor.manifest_path, root),
            }
        )
    body = {
        "schema_version": LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_SCHEMA,
        "stage1_request_sha256": prepared.request_sha256,
        "plan_content_sha256": plan.content_sha256,
        "physical_scope_order": [
            scope.scope_id for scope in plan.physical_scopes
        ],
        "physical_scope_count": len(plan.physical_scopes),
        "logical_scope_count": len(plan.scopes),
        "logical_physical_bindings": plan.as_dict()[
            "logical_physical_bindings"
        ],
        "descriptors": registrations,
        "heldout_labels_shared_between_descriptors": False,
        "full_split_registry_shared_with_workers": False,
        "full_scope_plan_shared_with_workers": False,
        "other_scope_row_identities_shared_with_workers": False,
        "full_cluster_preflight_shared_with_workers": False,
    }
    durably_sync_legacy_stage1_tree(root)
    _write_json(
        root / LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_MANIFEST,
        {**body, "content_sha256": _sha256_json(body)},
    )
    return validate_legacy_stage1_scope_descriptor_set(
        descriptor_root=root,
        expected_stage1_request_sha256=prepared.request_sha256,
        prepared=prepared,
    )


def _read_exact_parquet(
    path: Path,
    *,
    expected_columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or int(before.st_nlink) != 1:
            raise ValueError(f"{label} must be one singly-linked regular file")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            frame = pd.read_parquet(handle)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if tuple(getattr(before, field) for field in identity_fields) != tuple(
        getattr(after, field) for field in identity_fields
    ):
        raise RuntimeError(f"{label} changed while reading")
    if list(frame.columns) != list(expected_columns):
        raise ValueError(f"{label} has hidden, missing, or reordered columns")
    return frame


def validate_legacy_stage1_scope_descriptor(
    *,
    descriptor_manifest_path: Path | str,
    expected_stage1_request_sha256: str,
    expected_scope_id: str | None = None,
    prepared: Any | None = None,
    retain_embedding_cache: bool = False,
) -> AuthenticatedLegacyStage1ScopeDescriptor:
    """Reopen one closed scope-private descriptor."""

    expected_request = _require_sha256(
        expected_stage1_request_sha256,
        label="expected_stage1_request_sha256",
    )
    supplied_manifest = Path(descriptor_manifest_path)
    if supplied_manifest.is_symlink():
        raise ValueError("scope descriptor manifest cannot be a symlink")
    manifest_path = supplied_manifest.absolute()
    root = manifest_path.parent
    if (
        manifest_path.name != LEGACY_STAGE1_SCOPE_DESCRIPTOR_MANIFEST
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise ValueError("scope descriptor manifest path is invalid")
    manifest = _read_json(manifest_path, label="scope descriptor manifest")
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    required = {
        "schema_version",
        "stage1_request_sha256",
        "registry_content_sha256",
        "plan_content_sha256",
        "scope",
        "row_count",
        "columns",
        "files",
        "embedding_cache",
        "htr_model",
        "behavior_identity",
        "runtime",
        "visible_text_policy",
        "fit_label_policy",
        "scope_authority_policy",
        "cluster_preflight_policy",
        "heldout_labels_supplied_to_worker",
        "full_split_registry_supplied_to_worker",
        "full_scope_plan_supplied_to_worker",
        "other_scope_row_identities_supplied_to_worker",
        "other_scope_preflight_supplied_to_worker",
        "source_dataset_path_supplied_to_worker",
        "content_sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("schema_version") != LEGACY_STAGE1_SCOPE_DESCRIPTOR_SCHEMA
        or manifest.get("stage1_request_sha256") != expected_request
        or manifest.get("heldout_labels_supplied_to_worker") is not False
        or manifest.get("full_split_registry_supplied_to_worker") is not False
        or manifest.get("full_scope_plan_supplied_to_worker") is not False
        or manifest.get("other_scope_row_identities_supplied_to_worker") is not False
        or manifest.get("other_scope_preflight_supplied_to_worker") is not False
        or manifest.get("source_dataset_path_supplied_to_worker") is not False
        or manifest.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("scope descriptor manifest has an invalid binding")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "effective_config",
        "one_scope_authority",
        "cluster_preflight_projection",
        "visible_text",
        "fit_labels",
    }:
        raise ValueError("scope descriptor file inventory is incomplete")
    resolved_files = {key: _validate_registration(root, files[key], label=key) for key in files}
    private_cache = manifest.get("embedding_cache")
    if (
        not isinstance(private_cache, Mapping)
        or set(private_cache)
        != {
            "relative_path",
            "allowed_row_ids_sha256",
            "logical_identity",
            "physical_identity",
            "files",
        }
        or private_cache.get("allowed_row_ids_sha256")
        != _sha256_json(list((manifest.get("scope") or {})["fit_row_ids"]))
        or not isinstance(private_cache.get("files"), Mapping)
        or set(private_cache["files"])
        != {
            "metadata.json",
            "chunk_embeddings.npy",
            "offsets.npy",
            "chunk_texts.jsonl",
        }
    ):
        raise ValueError("scope-private embedding cache registration is invalid")
    for filename, registration in private_cache["files"].items():
        path = _validate_registration(
            root,
            registration,
            label=f"private embedding cache {filename}",
        )
        expected_relative = (Path(str(private_cache["relative_path"])) / filename).as_posix()
        if registration["relative_path"] != expected_relative:
            raise ValueError("private embedding cache file layout changed")
    expected_files = {
        LEGACY_STAGE1_SCOPE_DESCRIPTOR_MANIFEST,
        *(str(value["relative_path"]) for value in files.values()),
        *(str(value["relative_path"]) for value in private_cache["files"].values()),
    }
    observed_files, observed_directories = _closed_tree_inventory(
        root,
        label="scope descriptor",
    )
    expected_directories = {
        Path(value).parent.as_posix()
        for value in expected_files
        if Path(value).parent.as_posix() != "."
    }
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("scope descriptor contains unregistered entries")
    config_payload = _read_json(resolved_files["effective_config"], label="effective config")
    authority = _read_json(
        resolved_files["one_scope_authority"],
        label="one-scope authority",
    )
    from ..config import ExperimentConfig

    config = ExperimentConfig.from_dict({"applied_inference": config_payload}).applied_inference
    authority_body = {
        key: copy.deepcopy(value) for key, value in authority.items() if key != "content_sha256"
    }
    authority_fields = {
        "schema_version",
        "stage1_request_sha256",
        "registry_content_sha256",
        "plan_content_sha256",
        "cumulative_schedule_sha256",
        "dataset_row_count",
        "scope",
        "assignment",
        "split_scope_fingerprint",
        "registry_scope_binding_sha256",
        "plan_scope_binding_sha256",
        "authorized_scope_count",
        "other_scope_definitions_supplied",
        "other_scope_row_identities_supplied",
        "content_sha256",
    }
    if (
        set(authority) != authority_fields
        or authority.get("schema_version") != LEGACY_STAGE1_ONE_SCOPE_AUTHORITY_SCHEMA
        or authority.get("stage1_request_sha256") != expected_request
        or authority.get("registry_content_sha256") != manifest["registry_content_sha256"]
        or authority.get("plan_content_sha256") != manifest["plan_content_sha256"]
        or authority.get("dataset_row_count") != int(manifest["row_count"])
        or authority.get("authorized_scope_count") != 1
        or authority.get("other_scope_definitions_supplied") is not False
        or authority.get("other_scope_row_identities_supplied") is not False
        or authority.get("content_sha256") != _sha256_json(authority_body)
    ):
        raise ValueError("one-scope authority has an invalid binding")
    for key in (
        "stage1_request_sha256",
        "registry_content_sha256",
        "plan_content_sha256",
        "cumulative_schedule_sha256",
        "registry_scope_binding_sha256",
        "plan_scope_binding_sha256",
        "content_sha256",
    ):
        _require_sha256(authority.get(key), label=f"one-scope authority {key}")
    row_count = authority.get("dataset_row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
        raise ValueError("one-scope authority dataset_row_count is invalid")
    scope = _scope_spec_from_authority(
        authority.get("scope"),
        dataset_row_count=row_count,
    )
    assignment = _scope_assignment_from_authority(
        authority.get("assignment"),
        scope=scope,
    )
    split_fingerprint = authority.get("split_scope_fingerprint")
    if (
        (scope.scope_kind == "full_outer" and split_fingerprint is not None)
        or (
            scope.scope_kind != "full_outer"
            and _require_sha256(
                split_fingerprint,
                label="one-scope authority split_scope_fingerprint",
            )
            != split_fingerprint
        )
        or authority.get("registry_scope_binding_sha256")
        != _sha256_json(
            {
                "registry_content_sha256": manifest["registry_content_sha256"],
                "scope": scope.as_dict(),
            }
        )
        or authority.get("plan_scope_binding_sha256")
        != _sha256_json(
            {
                "plan_content_sha256": manifest["plan_content_sha256"],
                "scope": scope.as_dict(),
                "assignment": assignment.as_dict(),
            }
        )
    ):
        raise ValueError("one-scope authority selected binding changed")
    scope_id = scope.scope_id
    if manifest.get("scope") != scope.as_dict() or (
        expected_scope_id is not None and scope_id != str(expected_scope_id)
    ):
        raise ValueError("scope descriptor one-scope binding changed")
    physical_cache = _RestrictedLogicalIdentityEmbeddingCache(
        cache_dir=root / str(private_cache["relative_path"]),
        logical_identity=private_cache["logical_identity"],
        allowed_row_ids=scope.fit_row_ids,
    )
    if (
        physical_cache.physical_identity() != private_cache["physical_identity"]
        or physical_cache.identity() != private_cache["logical_identity"]
        or physical_cache.row_count != int(manifest["row_count"])
    ):
        raise ValueError("scope-private embedding cache changed")
    text = _read_exact_parquet(
        resolved_files["visible_text"],
        expected_columns=[_ROW_ID, config.text_column],
        label="visible text projection",
    )
    labels = _read_exact_parquet(
        resolved_files["fit_labels"],
        expected_columns=[
            _ROW_ID,
            config.treatment_column,
            config.outcome_column,
        ],
        label="fit-label projection",
    )
    if (
        text[_ROW_ID].tolist() != list(_visible_text_row_ids(scope))
        or labels[_ROW_ID].tolist() != list(scope.fit_row_ids)
        or text[config.text_column].isna().any()
        or not text[config.text_column].map(lambda value: isinstance(value, str)).all()
        or labels[[config.treatment_column, config.outcome_column]].isna().any().any()
    ):
        raise ValueError("scope-private text or fit-label rows changed")
    for column in (config.treatment_column, config.outcome_column):
        values = labels[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not set(np.unique(values)).issubset({0.0, 1.0}):
            raise ValueError("scope-private fit labels must remain binary")
    projection = _read_json(
        resolved_files["cluster_preflight_projection"],
        label="cluster preflight projection",
    )
    projection_body = {
        key: copy.deepcopy(value) for key, value in projection.items() if key != "content_sha256"
    }
    projection_fields = {
        "schema_version",
        "full_preflight_content_sha256",
        "full_scope_order_sha256",
        "canonical_index",
        "scope_id",
        "scope_kind",
        "scope_binding_sha256",
        "cluster_fit_identity",
        "cluster_fit_identity_sha256",
        "content_sha256",
    }
    if (
        set(projection) != projection_fields
        or projection.get("content_sha256") != _sha256_json(projection_body)
        or projection.get("scope_id") != scope.scope_id
        or projection.get("scope_kind") != scope.scope_kind
        or projection.get("canonical_index") != scope.canonical_index
        or (
            prepared is not None
            and projection.get("full_scope_order_sha256")
            != _sha256_json([item.scope_id for item in prepared.stage1_scope_plan.scopes])
        )
        or projection.get("scope_binding_sha256")
        != _sha256_json(
            {
                "registry_content_sha256": manifest["registry_content_sha256"],
                "scope": scope.as_dict(),
            }
        )
        or projection.get("cluster_fit_identity_sha256")
        != (projection.get("cluster_fit_identity") or {}).get("content_sha256")
    ):
        raise ValueError("scope-private cluster preflight changed")
    for key in (
        "full_preflight_content_sha256",
        "full_scope_order_sha256",
        "scope_binding_sha256",
        "cluster_fit_identity_sha256",
        "content_sha256",
    ):
        _require_sha256(
            projection.get(key),
            label=f"cluster preflight projection {key}",
        )
    if prepared is not None:
        expected_private_config = _scope_private_effective_config(
            prepared=prepared,
            public_scope_root=root,
        )
        if (
            config_payload != expected_private_config
            or text[config.text_column].tolist()
            != prepared.modeling_data.iloc[list(_visible_text_row_ids(scope))][
                prepared.config.text_column
            ].tolist()
            or labels[
                [
                    config.treatment_column,
                    config.outcome_column,
                ]
            ]
            .to_numpy(dtype=float)
            .tolist()
            != prepared.modeling_data.iloc[list(scope.fit_row_ids)][
                [
                    prepared.config.treatment_column,
                    prepared.config.outcome_column,
                ]
            ]
            .to_numpy(dtype=float)
            .tolist()
            or private_cache["logical_identity"] != prepared.embedding_cache_identity
        ):
            raise ValueError("scope-private data differs from parent preparation")
        for row_id in scope.fit_row_ids:
            expected_chunks = prepared.embedding_cache._cached_chunks(row_id)
            if physical_cache._cache._cached_chunks(
                row_id
            ) != expected_chunks or not np.array_equal(
                physical_cache._cache._embeddings[
                    int(physical_cache._cache._offsets[row_id]) : int(
                        physical_cache._cache._offsets[row_id + 1]
                    )
                ],
                prepared.embedding_cache._embeddings[
                    int(prepared.embedding_cache._offsets[row_id]) : int(
                        prepared.embedding_cache._offsets[row_id + 1]
                    )
                ],
            ):
                raise ValueError("scope-private cache differs from parent selected rows")
        expected_body = _descriptor_body_from_prepared(
            prepared=prepared,
            scope=scope,
            files=files,
            private_cache=private_cache,
        )
        if (
            body != expected_body
            or authority
            != _one_scope_authority_from_prepared(
                prepared=prepared,
                scope=scope,
            )
            or projection != _scope_preflight_projection(prepared=prepared, scope=scope)
        ):
            raise ValueError("scope descriptor differs from parent preparation")
    return AuthenticatedLegacyStage1ScopeDescriptor(
        root=root,
        manifest=copy.deepcopy(manifest),
        authority=copy.deepcopy(authority),
        scope=scope,
        assignment=assignment,
        scope_id=scope.scope_id,
        embedding_cache=(physical_cache if retain_embedding_cache else None),
    )


def validate_legacy_stage1_scope_descriptor_set(
    *,
    descriptor_root: Path | str,
    expected_stage1_request_sha256: str,
    prepared: Any | None = None,
) -> AuthenticatedLegacyStage1ScopeDescriptorSet:
    root = Path(descriptor_root).absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("scope descriptor-set root is invalid")
    manifest = _read_json(
        root / LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_MANIFEST,
        label="scope descriptor-set manifest",
    )
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    rows = manifest.get("descriptors")
    required = {
        "schema_version",
        "stage1_request_sha256",
        "plan_content_sha256",
        "physical_scope_order",
        "physical_scope_count",
        "logical_scope_count",
        "logical_physical_bindings",
        "descriptors",
        "heldout_labels_shared_between_descriptors",
        "full_split_registry_shared_with_workers",
        "full_scope_plan_shared_with_workers",
        "other_scope_row_identities_shared_with_workers",
        "full_cluster_preflight_shared_with_workers",
        "content_sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("schema_version") != LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_SCHEMA
        or manifest.get("stage1_request_sha256") != expected_stage1_request_sha256
        or manifest.get("content_sha256") != _sha256_json(body)
        or not isinstance(rows, list)
        or isinstance(manifest.get("physical_scope_count"), bool)
        or not isinstance(manifest.get("physical_scope_count"), int)
        or isinstance(manifest.get("logical_scope_count"), bool)
        or not isinstance(manifest.get("logical_scope_count"), int)
        or int(manifest["physical_scope_count"]) < 1
        or int(manifest["logical_scope_count"])
        < int(manifest["physical_scope_count"])
        or not isinstance(manifest.get("logical_physical_bindings"), list)
        or len(manifest["logical_physical_bindings"])
        != int(manifest["logical_scope_count"])
        or manifest.get("heldout_labels_shared_between_descriptors") is not False
        or manifest.get("full_split_registry_shared_with_workers") is not False
        or manifest.get("full_scope_plan_shared_with_workers") is not False
        or manifest.get("other_scope_row_identities_shared_with_workers") is not False
        or manifest.get("full_cluster_preflight_shared_with_workers") is not False
    ):
        raise ValueError("scope descriptor-set manifest is invalid")
    _require_sha256(
        manifest.get("plan_content_sha256"),
        label="descriptor-set plan_content_sha256",
    )
    descriptors: dict[str, AuthenticatedLegacyStage1ScopeDescriptor] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"scope_id", "manifest"}:
            raise ValueError("descriptor-set row is not closed")
        scope_id = str(row["scope_id"])
        if not scope_id or scope_id in descriptors:
            raise ValueError("descriptor-set scope IDs must be unique and nonempty")
        registration = row["manifest"]
        manifest_path = _validate_registration(
            root, registration, label=f"{scope_id} descriptor manifest"
        )
        descriptors[scope_id] = validate_legacy_stage1_scope_descriptor(
            descriptor_manifest_path=manifest_path,
            expected_stage1_request_sha256=expected_stage1_request_sha256,
            expected_scope_id=scope_id,
            prepared=prepared,
        )
    expected_order = (
        [
            scope.scope_id
            for scope in prepared.stage1_scope_plan.physical_scopes
        ]
        if prepared is not None
        else list(manifest.get("physical_scope_order") or ())
    )
    if (
        list(descriptors) != expected_order
        or manifest.get("physical_scope_order") != expected_order
        or manifest.get("physical_scope_count") != len(descriptors)
        or (
            prepared is not None
            and manifest.get("logical_scope_count")
            != len(prepared.stage1_scope_plan.scopes)
        )
        or (
            prepared is not None
            and manifest.get("logical_physical_bindings")
            != prepared.stage1_scope_plan.as_dict()[
                "logical_physical_bindings"
            ]
        )
        or any(
            descriptor.plan_content_sha256 != str(manifest["plan_content_sha256"])
            for descriptor in descriptors.values()
        )
        or [descriptor.scope.canonical_index for descriptor in descriptors.values()]
        != sorted(descriptor.scope.canonical_index for descriptor in descriptors.values())
        or len({descriptor.scope.canonical_index for descriptor in descriptors.values()})
        != len(descriptors)
    ):
        raise ValueError("scope descriptor-set coverage changed")
    expected_files = {LEGACY_STAGE1_SCOPE_DESCRIPTOR_SET_MANIFEST}
    expected_directories: set[str] = set()
    for scope_id, descriptor in descriptors.items():
        child_files, child_directories = _closed_tree_inventory(
            descriptor.root,
            label=f"{scope_id} descriptor",
        )
        expected_directories.add(scope_id)
        expected_files.update(f"{scope_id}/{relative}" for relative in child_files)
        expected_directories.update(f"{scope_id}/{relative}" for relative in child_directories)
    observed_files, observed_directories = _closed_tree_inventory(
        root,
        label="scope descriptor set",
    )
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("scope descriptor set contains unregistered entries")
    return AuthenticatedLegacyStage1ScopeDescriptorSet(
        root=root,
        manifest=copy.deepcopy(manifest),
        descriptors=descriptors,
    )


def _load_scope_modeling_data(
    *,
    descriptor: AuthenticatedLegacyStage1ScopeDescriptor,
    request: Stage1ScopeExecutionRequest,
) -> tuple[pd.DataFrame, Any]:
    from ..config import ExperimentConfig

    files = descriptor.manifest["files"]
    config_payload = _read_json(
        descriptor.root / str(files["effective_config"]["relative_path"]),
        label="effective config",
    )
    config = ExperimentConfig.from_dict({"applied_inference": config_payload}).applied_inference
    text = _read_exact_parquet(
        descriptor.root / str(files["visible_text"]["relative_path"]),
        expected_columns=[_ROW_ID, config.text_column],
        label="visible text projection",
    )
    labels = _read_exact_parquet(
        descriptor.root / str(files["fit_labels"]["relative_path"]),
        expected_columns=[
            _ROW_ID,
            config.treatment_column,
            config.outcome_column,
        ],
        label="fit-label projection",
    )
    fit_ids = list(map(int, request.scope["fit_row_ids"]))
    row_count = int(descriptor.manifest["row_count"])
    modeling = pd.DataFrame(
        {
            config.text_column: np.full(row_count, "", dtype=object),
            config.treatment_column: np.full(row_count, np.nan, dtype=float),
            config.outcome_column: np.full(row_count, np.nan, dtype=float),
        }
    )
    visible_ids = text[_ROW_ID].to_numpy(dtype=int)
    modeling.loc[visible_ids, config.text_column] = text[config.text_column].to_numpy(copy=True)
    modeling.loc[
        fit_ids,
        [config.treatment_column, config.outcome_column],
    ] = labels[
        [config.treatment_column, config.outcome_column]
    ].to_numpy(dtype=float)
    non_fit = sorted(set(range(row_count)) - set(fit_ids))
    if modeling.iloc[non_fit][[config.treatment_column, config.outcome_column]].notna().any().any():
        raise RuntimeError("held-out labels entered scope worker memory")
    return modeling, config


def _remove_worker_aggregate_files(artifact_root: Path) -> None:
    for relative in LEGACY_STAGE1_AGGREGATE_RELATIVE_PATHS:
        path = artifact_root / relative
        if path.is_symlink():
            raise ValueError("worker aggregate output cannot be a symlink")
        if not path.is_file():
            raise RuntimeError(f"selected legacy worker omitted aggregate staging file: {relative}")
        path.unlink()
    # Aggregate-only directories may remain empty.  Remove those empty
    # directories so a fragment has a minimal, unambiguous tree.
    for path in sorted(
        (item for item in artifact_root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            path.rmdir()
        except OSError:
            pass


def run_legacy_stage1_scope_worker(
    request: Stage1ScopeExecutionRequest,
) -> Mapping[str, Any]:
    """Scheduler target: authenticate inputs, execute, and seal one fragment."""

    if not isinstance(request, Stage1ScopeExecutionRequest):
        raise TypeError("legacy scope worker requires Stage1ScopeExecutionRequest")
    parameters = request.worker_parameters
    if set(parameters) != {
        "descriptor_manifest_path",
        "stage1_request_sha256",
        "scope_id",
    }:
        raise ValueError("legacy scope worker parameters are not closed")
    if parameters["scope_id"] != request.scope_id:
        raise ValueError("legacy scope worker received another scope descriptor")
    descriptor = validate_legacy_stage1_scope_descriptor(
        descriptor_manifest_path=parameters["descriptor_manifest_path"],
        expected_stage1_request_sha256=parameters["stage1_request_sha256"],
        expected_scope_id=request.scope_id,
        retain_embedding_cache=True,
    )
    if (
        request.plan_content_sha256 != descriptor.plan_content_sha256
        or request.scope != descriptor.scope.as_dict()
        or request.assignment != descriptor.assignment.as_dict()
    ):
        raise ValueError("legacy scope request differs from its one-scope authority")
    modeling_data, config = _load_scope_modeling_data(
        descriptor=descriptor,
        request=request,
    )

    from types import SimpleNamespace

    from . import production_stage1_bundle as bundle

    if bundle._source_identity() != descriptor.manifest["behavior_identity"]:
        raise RuntimeError("Stage 1 behavior changed after descriptor publication")
    cache_registration = descriptor.manifest["embedding_cache"]
    cache = descriptor.embedding_cache
    if not isinstance(cache, _RestrictedLogicalIdentityEmbeddingCache):
        raise RuntimeError("scope worker did not retain its authenticated embedding cache")
    if (
        cache.identity() != cache_registration["logical_identity"]
        or cache.physical_identity() != cache_registration["physical_identity"]
    ):
        raise RuntimeError("scope worker embedding cache identity changed")
    htr_model_path = Path(str(descriptor.manifest["htr_model"]["path"])).resolve(strict=True)
    if (
        bundle._directory_tree_sha256(htr_model_path)
        != descriptor.manifest["htr_model"]["tree_sha256"]
    ):
        raise RuntimeError("scope worker HTR model tree changed")
    preflight_projection = _read_json(
        descriptor.root
        / str(descriptor.manifest["files"]["cluster_preflight_projection"]["relative_path"]),
        label="embedding cluster preflight projection",
    )
    preflight = {
        "content_sha256": str(preflight_projection["full_preflight_content_sha256"]),
        "scopes": [
            {
                "scope_id": request.scope_id,
                "cluster_fit_identity": copy.deepcopy(preflight_projection["cluster_fit_identity"]),
            }
        ],
    }
    device = "cpu" if request.gpu_id is None else f"cuda:{int(request.gpu_id)}"
    options = SimpleNamespace(
        seed=int(descriptor.manifest["runtime"]["global_seed"]),
        device=device,
        gpu_ids=(() if request.gpu_id is None else (int(request.gpu_id),)),
        num_workers=int(descriptor.manifest["runtime"]["num_workers"]),
        scope_workers_per_gpu=int(descriptor.manifest["runtime"]["scope_workers_per_gpu"]),
        dataset_path=(
            descriptor.root / str(descriptor.manifest["files"]["visible_text"]["relative_path"])
        ),
    )
    prepared = SimpleNamespace(
        options=options,
        modeling_data=modeling_data,
        config=config,
        htr_model_path=htr_model_path,
        htr_model_sha256=str(descriptor.manifest["htr_model"]["tree_sha256"]),
        embedding_cluster_feasibility_audit=preflight,
        embedding_cache_path=cache.cache_dir,
        embedding_cache=cache,
        embedding_cache_identity=copy.deepcopy(dict(cache_registration["logical_identity"])),
        registry_content_sha256=str(descriptor.manifest["registry_content_sha256"]),
        selected_scope_authority=copy.deepcopy(dict(descriptor.authority)),
        selected_scope_spec=descriptor.scope,
        request_sha256=descriptor.stage1_request_sha256,
    )
    fragment_root = request.payload_dir / _FRAGMENT_DIRECTORY
    artifact_root = fragment_root / _FRAGMENT_ARTIFACT_DIRECTORY
    fragment_root.mkdir(parents=False, exist_ok=False)
    builder = bundle.ProductionStage1BundleBuilder.__new__(bundle.ProductionStage1BundleBuilder)
    accumulator = builder._run_legacy_component(
        artifact_root,
        prepared,
        selected_scope_id=request.scope_id,
    )
    if not isinstance(accumulator, Mapping):
        raise RuntimeError("selected legacy scope returned no accumulator")
    _remove_worker_aggregate_files(artifact_root)
    fragment = seal_legacy_stage1_scope_fragment(
        fragment_root=fragment_root,
        scope_authority=descriptor.scope,
        plan_content_sha256=descriptor.plan_content_sha256,
        scope_id=request.scope_id,
        stage1_request_sha256=descriptor.stage1_request_sha256,
        scope_attempt_request_sha256=request.attempt_request_sha256,
        accumulator=accumulator,
    )
    # Deliberately return only identities.  The accumulator remains inside the
    # fragment and the scheduler seals it in the attempt inventory.
    return {
        "scope_id": request.scope_id,
        "scope_seed": int(request.scope_seed),
        "device": device,
        "fragment_manifest_content_sha256": (fragment.manifest_content_sha256),
        "artifact_count": len(fragment.artifacts),
        "heldout_labels_supplied": False,
    }


def collect_and_merge_legacy_stage1_scope_attempts(
    *,
    prepared: Any,
    attempts: Sequence[ValidatedStage1ScopeAttempt],
    merge_root: Path | str,
    require_production_coverage: bool = True,
) -> Mapping[str, Any]:
    """Authenticate physical attempts and fail closed on role-specific aliases."""

    plan = prepared.stage1_scope_plan
    by_scope = {attempt.scope_id: attempt for attempt in attempts}
    expected = {scope.scope_id for scope in plan.physical_scopes}
    if len(by_scope) != len(attempts) or set(by_scope) != expected:
        raise ValueError(
            "legacy Stage 1 physical attempts have incomplete owner coverage"
        )
    roots = {attempt.attempt_dir.parent.parent for attempt in attempts}
    if len(roots) != 1:
        raise ValueError(
            "legacy Stage 1 physical attempts do not share one bound store"
        )
    attempt_store = Stage1ScopeAttemptStore(next(iter(roots)), plan)
    logical_bindings = attempt_store.validate_logical_bindings()
    if (
        logical_bindings.path.name
        != STAGE1_LOGICAL_SCOPE_BINDING_FILENAME
        or logical_bindings.manifest.get("physical_fit_count")
        != len(plan.physical_scopes)
        or logical_bindings.manifest.get("logical_scope_count")
        != len(plan.scopes)
    ):
        raise ValueError(
            "legacy Stage 1 logical bindings have incomplete coverage"
        )
    reused_groups = [
        (owner, members)
        for owner, members in plan.physical_scope_groups
        if len(members) > 1
    ]
    if reused_groups:
        details = ", ".join(
            (
                f"{owner.scope_id}->"
                + "/".join(member.scope_id for member in members[1:])
                + "["
                + ",".join(sorted({member.scope_kind for member in members}))
                + "]"
            )
            for owner, members in reused_groups
        )
        raise LegacyStage1RoleSpecificDeduplicationError(
            "authenticated physical-fit attempts contain only legacy "
            "role-specific accumulators. Exact-inner workers may transform "
            "held-out text, while cumulative-review views must keep sealed "
            "text unavailable, so their logical evidence bytes cannot be "
            "declared equal. Publication requires a complete "
            f"{LEGACY_STAGE1_ROLE_NEUTRAL_BINDING_SET_SCHEMA} record with "
            "ten fit-side family artifacts and distinct authenticated "
            f"per-purpose logical views ({details})"
        )
    fragment_roots: dict[str, Path] = {}
    attempt_request_hashes: dict[str, str] = {}
    for scope in plan.physical_scopes:
        attempt = by_scope[scope.scope_id]
        request_payload = _read_json(
            attempt.attempt_dir / "attempt_request.json",
            label=f"{scope.scope_id} attempt request",
        )
        request_sha = _require_sha256(
            request_payload.get("attempt_request_sha256"),
            label=f"{scope.scope_id} attempt_request_sha256",
        )
        fragment_root = attempt.attempt_dir / "payload" / _FRAGMENT_DIRECTORY
        validate_legacy_stage1_scope_fragment(
            fragment_root=fragment_root,
            plan=plan,
            scope_id=scope.scope_id,
            stage1_request_sha256=prepared.request_sha256,
            scope_attempt_request_sha256=request_sha,
        )
        fragment_roots[scope.scope_id] = fragment_root
        attempt_request_hashes[scope.scope_id] = request_sha

    destination = Path(merge_root)
    if destination.exists():
        return validate_legacy_stage1_fragment_merge(
            plan=plan,
            stage1_request_sha256=prepared.request_sha256,
            fragment_roots_by_scope=fragment_roots,
            scope_attempt_request_sha256_by_scope=attempt_request_hashes,
            destination_root=destination,
            require_production_coverage=require_production_coverage,
        )
    return merge_legacy_stage1_scope_fragments(
        plan=plan,
        stage1_request_sha256=prepared.request_sha256,
        fragment_roots_by_scope=fragment_roots,
        scope_attempt_request_sha256_by_scope=attempt_request_hashes,
        destination_root=destination,
        require_production_coverage=require_production_coverage,
    )


def _scope_id_from_handoff_row(row: Mapping[str, Any]) -> str:
    outer_fold = int(row["outer_fold"])
    inner_fold = row.get("inner_fold")
    if inner_fold is None:
        return f"outer_{outer_fold:03d}_full"
    return f"outer_{outer_fold:03d}_inner_{int(inner_fold):03d}"


def _validated_merged_accumulators(
    *,
    prepared: Any,
    merge_root: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    manifest = validate_legacy_stage1_fragment_merge_from_path(
        plan=prepared.stage1_scope_plan,
        stage1_request_sha256=prepared.request_sha256,
        destination_root=merge_root,
        require_production_coverage=True,
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("stage1_request_sha256") != prepared.request_sha256
        or manifest.get("plan_content_sha256") != prepared.stage1_scope_plan.content_sha256
        or manifest.get("canonical_scope_order")
        != [scope.scope_id for scope in prepared.stage1_scope_plan.scopes]
        or manifest.get("scope_count") != len(prepared.stage1_scope_plan.scopes)
        or manifest.get("heldout_labels_supplied_to_workers") is not False
    ):
        raise ValueError("legacy fragment merge is not bound to this Stage 1 request")
    accumulator_registration = manifest.get("scope_accumulators")
    if not isinstance(accumulator_registration, Mapping):
        raise ValueError("legacy fragment merge lacks its accumulator registration")
    accumulator_path = _validate_registration(
        merge_root,
        {
            "relative_path": accumulator_registration.get("relative_path"),
            "sha256": accumulator_registration.get("sha256"),
            "size_bytes": accumulator_registration.get("size_bytes"),
        },
        label="legacy fragment merge accumulator",
    )
    accumulators = _read_json(
        accumulator_path,
        label="legacy fragment merge accumulator",
    )
    accumulator_body = {
        key: copy.deepcopy(value) for key, value in accumulators.items() if key != "content_sha256"
    }
    rows = accumulators.get("scopes")
    if (
        not isinstance(rows, list)
        or accumulators.get("content_sha256") != _sha256_json(accumulator_body)
        or accumulators.get("content_sha256") != accumulator_registration.get("content_sha256")
        or accumulators.get("stage1_request_sha256") != prepared.request_sha256
        or accumulators.get("plan_content_sha256") != prepared.stage1_scope_plan.content_sha256
        or accumulators.get("canonical_scope_order")
        != [scope.scope_id for scope in prepared.stage1_scope_plan.scopes]
        or len(rows) != len(prepared.stage1_scope_plan.scopes)
    ):
        raise ValueError("legacy fragment merge accumulators changed")
    payloads: list[Mapping[str, Any]] = []
    expected_payload_fields = {
        "scope_id",
        "scope_kind",
        "handoff_rows",
        "scope_index_rows",
        "native_bow_proof_rows",
        "native_htr_proof_rows",
        "native_matched_pair_proof_rows",
        "native_embedding_proof_rows",
        "cumulative_legacy_registrations",
        "cumulative_embedding_registrations",
        "cumulative_expected_configurations",
        "embedding_cluster_fit_rows",
    }
    for scope, row in zip(prepared.stage1_scope_plan.scopes, rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("scope") != scope.as_dict()
            or not isinstance(row.get("accumulator"), Mapping)
        ):
            raise ValueError("legacy merged accumulator scope order changed")
        wrapper = row["accumulator"]
        payload = wrapper.get("payload")
        if (
            wrapper.get("scope_id") != scope.scope_id
            or wrapper.get("scope_kind") != scope.scope_kind
            or wrapper.get("canonical_index") != scope.canonical_index
            or not isinstance(payload, Mapping)
            or set(payload) != expected_payload_fields
            or payload.get("scope_id") != scope.scope_id
            or payload.get("scope_kind") != scope.scope_kind
        ):
            raise ValueError(f"legacy merged scope accumulator changed: {scope.scope_id}")
        exact_fields = (
            "handoff_rows",
            "scope_index_rows",
            "native_bow_proof_rows",
            "native_htr_proof_rows",
            "native_matched_pair_proof_rows",
            "native_embedding_proof_rows",
        )
        cumulative_fields = (
            "cumulative_legacy_registrations",
            "cumulative_embedding_registrations",
        )
        if scope.scope_kind in {"full_outer", "exact_inner"}:
            if (
                len(payload["handoff_rows"]) != 1
                or len(payload["scope_index_rows"]) != 1
                or _scope_id_from_handoff_row(payload["handoff_rows"][0]) != scope.scope_id
                or payload["scope_index_rows"][0].get("scope_id") != scope.scope_id
                or any(payload[field] for field in cumulative_fields)
                or payload["cumulative_expected_configurations"]
            ):
                raise ValueError(f"exact legacy accumulator has foreign rows: {scope.scope_id}")
            expected_native_count = 1 if scope.scope_kind == "exact_inner" else 0
            for field in exact_fields[2:]:
                if len(payload[field]) != expected_native_count or (
                    expected_native_count and payload[field][0].get("scope_id") != scope.scope_id
                ):
                    raise ValueError(f"legacy native accumulator changed: {scope.scope_id}/{field}")
        else:
            if (
                any(payload[field] for field in exact_fields)
                or len(payload["cumulative_legacy_registrations"]) != 1
                or len(payload["cumulative_embedding_registrations"]) != 1
                or payload["cumulative_legacy_registrations"][0].get("scope_id") != scope.scope_id
                or payload["cumulative_embedding_registrations"][0].get("scope_id")
                != scope.scope_id
                or set(payload["cumulative_expected_configurations"]) != {scope.scope_id}
            ):
                raise ValueError(
                    f"cumulative legacy accumulator has foreign rows: {scope.scope_id}"
                )
        if (
            len(payload["embedding_cluster_fit_rows"]) != 1
            or payload["embedding_cluster_fit_rows"][0].get("scope_id") != scope.scope_id
            or payload["embedding_cluster_fit_rows"][0].get("scope_kind") != scope.scope_kind
        ):
            raise ValueError(f"legacy cluster accumulator changed: {scope.scope_id}")
        payloads.append(copy.deepcopy(dict(payload)))
    return manifest, payloads


def finalize_legacy_stage1_component_from_merge(
    *,
    prepared: Any,
    merge_root: Path | str,
    component_root: Path | str,
) -> Mapping[str, Any]:
    """Reconstruct and validate all aggregate indexes, then publish atomically."""

    from . import production_stage1_bundle as bundle
    from .all_evidence_fusion_runner import load_legacy_full_outer_evidence

    merge = Path(merge_root).resolve(strict=True)
    destination = Path(component_root)
    if not destination.is_absolute():
        raise ValueError("legacy component destination must be absolute")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("legacy component destination must be fresh")
    destination_parent = destination.parent.resolve(strict=True)
    if destination_parent != destination.parent or destination_parent.is_symlink():
        raise ValueError("legacy component destination parent must be canonical")
    manifest, payloads = _validated_merged_accumulators(
        prepared=prepared,
        merge_root=merge,
    )
    copied_files = manifest.get("copied_files")
    if not isinstance(copied_files, list):
        raise ValueError("legacy fragment merge has no copied-file inventory")

    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.finalizing-",
            dir=destination.parent,
        )
    )
    try:
        for registration in copied_files:
            if not isinstance(registration, Mapping):
                raise ValueError("legacy merged file registration is malformed")
            relative = Path(str(registration["relative_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("legacy merged file escapes its component")
            source = merge / relative
            digest, size = _sha256_file(source)
            if digest != registration.get("sha256") or size != int(
                registration.get("size_bytes", -1)
            ):
                raise ValueError(f"legacy merged artifact changed before finalization: {relative}")
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise RuntimeError(f"legacy finalization collision reached copy: {relative}")
            shutil.copyfile(source, target, follow_symlinks=False)
            copied_digest, copied_size = _sha256_file(target)
            if (copied_digest, copied_size) != (digest, size):
                raise RuntimeError(f"legacy artifact changed while finalizing: {relative}")

        handoff_rows = [row for payload in payloads for row in payload["handoff_rows"]]
        scope_index_rows = [row for payload in payloads for row in payload["scope_index_rows"]]
        bow_rows = [row for payload in payloads for row in payload["native_bow_proof_rows"]]
        htr_rows = [row for payload in payloads for row in payload["native_htr_proof_rows"]]
        matched_rows = [
            row for payload in payloads for row in payload["native_matched_pair_proof_rows"]
        ]
        embedding_rows = [
            row for payload in payloads for row in payload["native_embedding_proof_rows"]
        ]
        cumulative_legacy = [
            row for payload in payloads for row in payload["cumulative_legacy_registrations"]
        ]
        cumulative_embedding = [
            row for payload in payloads for row in payload["cumulative_embedding_registrations"]
        ]
        cumulative_configurations = {
            scope_id: configuration
            for payload in payloads
            for scope_id, configuration in payload["cumulative_expected_configurations"].items()
        }
        cluster_rows = [
            row for payload in payloads for row in payload["embedding_cluster_fit_rows"]
        ]
        planned_scopes = tuple(prepared.stage1_scope_plan.scopes)
        exact_inner_count = sum(
            scope.scope_kind == "exact_inner" for scope in planned_scopes
        )
        cumulative_count = sum(
            scope.scope_kind == "cumulative_spent" for scope in planned_scopes
        )
        exact_handoff_count = sum(
            scope.scope_kind in {"full_outer", "exact_inner"}
            for scope in planned_scopes
        )
        if (
            len(handoff_rows) != exact_handoff_count
            or len(scope_index_rows) != exact_handoff_count
            or any(
                len(rows) != exact_inner_count
                for rows in (
                    bow_rows,
                    htr_rows,
                    matched_rows,
                    embedding_rows,
                )
            )
            or len(cumulative_legacy) != cumulative_count
            or len(cumulative_embedding) != cumulative_count
            or len(cumulative_configurations) != cumulative_count
            or len(cluster_rows) != len(planned_scopes)
        ):
            raise RuntimeError(
                "legacy parent finalizer did not receive coverage matching "
                "the authenticated logical scope plan"
            )

        schedule = bundle._canonical_cumulative_spent_schedule(
            prepared.registry,
            initial_training_partitions=(
                prepared.stage1_scope_plan.initial_training_partitions
            ),
        )
        expected_requests: dict[str, Any] = {}
        for scope in schedule.scopes:
            expected_requests[scope.scope_id] = bundle._cumulative_spent_request_from_modeling_data(
                family=bundle.BOW_NUISANCE,
                modeling_data=prepared.modeling_data,
                request_sha256=prepared.request_sha256,
                schedule_sha256=schedule.schedule_sha256,
                scope_id=scope.scope_id,
                outer_fold=scope.outer_fold,
                context_epoch=scope.context_epoch,
                provider_inner_fold=scope.provider_inner_fold,
                split_scope_fingerprint=scope.split_fingerprint,
                spent_row_ids=scope.spent_row_ids,
                sealed_row_ids=scope.sealed_row_ids,
                text_column=prepared.config.text_column,
                treatment_column=prepared.config.treatment_column,
                outcome_column=prepared.config.outcome_column,
            )
        cumulative_index_registration = bundle._write_legacy_cumulative_spent_native_index(
            component_root=temporary,
            index_path=Path("cumulative_legacy_native_family_proof_index.json"),
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=(prepared.registry_content_sha256),
            scope_registrations=cumulative_legacy,
        )
        bundle._validate_legacy_cumulative_spent_native_index(
            component_root=temporary,
            index_registration=cumulative_index_registration,
            expected_requests=expected_requests,
            expected_configuration_by_scope=cumulative_configurations,
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            htr_model_path=prepared.htr_model_path,
            htr_model_sha256=prepared.htr_model_sha256,
            device=prepared.options.device,
        )
        cumulative_embedding_index_registration = bundle._write_cumulative_spent_embedding_index(
            component_root=temporary,
            index_path=Path("cumulative_embedding_native_family_proof_index.json"),
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=(prepared.registry_content_sha256),
            scope_registrations=cumulative_embedding,
        )
        bundle._validate_cumulative_spent_embedding_index(
            component_root=temporary,
            index_registration=cumulative_embedding_index_registration,
            expected_requests=expected_requests,
            request_sha256=prepared.request_sha256,
            schedule_sha256=schedule.schedule_sha256,
            split_registry_content_sha256=prepared.registry_content_sha256,
            embedding_cache=prepared.embedding_cache,
        )

        cluster_by_scope = {str(row["scope_id"]): row for row in cluster_rows}
        cluster_order = [scope.scope_id for scope in prepared.stage1_scope_plan.scopes]
        if len(cluster_by_scope) != len(cluster_rows) or set(cluster_by_scope) != set(
            cluster_order
        ):
            raise ValueError("legacy cluster rows have incomplete coverage")
        ordered_cluster_rows = [cluster_by_scope[scope_id] for scope_id in cluster_order]
        for scope, row in zip(
            prepared.stage1_scope_plan.scopes,
            ordered_cluster_rows,
            strict=True,
        ):
            record_path = bundle._validate_component_native_registration(temporary, row["record"])
            record = _read_json(
                record_path,
                label=f"{scope.scope_id} cluster fit record",
            )
            actual = bundle._validate_embedding_cluster_fit_identity(
                record.get("actual_identity") or {},
                scope_id=scope.scope_id,
                fit_row_ids=scope.fit_row_ids,
            )
            expected = bundle._preflight_cluster_fit_identity(
                prepared,
                scope_id=scope.scope_id,
            )
            if (
                row.get("scope_kind") != scope.scope_kind
                or row.get("identity_sha256") != expected["content_sha256"]
                or actual != expected
                or record.get("scope_id") != scope.scope_id
                or record.get("scope_kind") != scope.scope_kind
                or record.get("preflight_identity_sha256") != expected["content_sha256"]
                or record.get("actual_equals_preflight") is not True
            ):
                raise ValueError(f"legacy cluster fit differs from preflight: {scope.scope_id}")
        cluster_body = {
            "schema_version": (bundle.STAGE1_EMBEDDING_CLUSTER_FIT_INDEX_SCHEMA),
            "request_sha256": prepared.request_sha256,
            "split_registry_content_sha256": (prepared.registry_content_sha256),
            "preflight_audit_content_sha256": (
                prepared.embedding_cluster_feasibility_audit["content_sha256"]
            ),
            "scope_count": len(ordered_cluster_rows),
            "full_outer_scope_count": sum(
                row["scope_kind"] == "full_outer" for row in ordered_cluster_rows
            ),
            "exact_inner_scope_count": sum(
                row["scope_kind"] == "exact_inner" for row in ordered_cluster_rows
            ),
            "cumulative_spent_scope_count": sum(
                row["scope_kind"] == "cumulative_spent" for row in ordered_cluster_rows
            ),
            "scope_order": cluster_order,
            "all_actual_identities_equal_preflight": True,
            "scopes": ordered_cluster_rows,
        }
        cluster_path = temporary / "embedding_cluster_fit_index.json"
        _write_json(
            cluster_path,
            {
                **cluster_body,
                "content_sha256": _sha256_json(cluster_body),
            },
        )

        handoff_rows.sort(
            key=lambda row: (
                int(row["outer_fold"]),
                0 if row["scope"] == "full_outer_train" else int(row["inner_fold"]),
            )
        )
        handoff_dir = temporary / "handoff"
        handoff_dir.mkdir(parents=True, exist_ok=True)
        handoff_path = handoff_dir / "discovery_contexts.jsonl"
        handoff_bytes = b"".join(
            (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            for row in handoff_rows
        )
        with handoff_path.open("xb") as handle:
            handle.write(handoff_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        handoff_sha, _handoff_size = _sha256_file(handoff_path)
        _write_json(
            handoff_dir / "manifest.json",
            {
                "schema_version": ("multi_model_agentic_discovery_handoff_v1"),
                "handoff_file": handoff_path.name,
                "handoff_sha256": handoff_sha,
                "row_count": len(handoff_rows),
                "exact_scope_count": len(handoff_rows),
                "split_registry_content_sha256": (prepared.registry_content_sha256),
                "raw_evidence_sidecar_count": len(scope_index_rows),
                "raw_evidence_sidecars_prompt_visible": False,
                "prompt_compactor_used": False,
                "full_outer_evidence_reused_for_inner": False,
                "heldout_labels_supplied_to_evidence_builder": False,
            },
        )

        native_indexes: dict[str, Mapping[str, Any]] = {}
        for filename, families, rows in (
            (
                "bow_native_family_proof_index.json",
                bundle.PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                bow_rows,
            ),
            (
                "htr_native_family_proof_index.json",
                bundle.PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                htr_rows,
            ),
            (
                "matched_pair_native_family_proof_index.json",
                bundle.PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                matched_rows,
            ),
            (
                "embedding_native_family_proof_index.json",
                bundle.PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                embedding_rows,
            ),
        ):
            index_body = {
                "schema_version": (bundle.STAGE1_NATIVE_FAMILY_PROOF_INDEX_SCHEMA),
                "split_registry_content_sha256": (prepared.registry_content_sha256),
                "registered_families": list(families),
                "exact_inner_scope_count": len(rows),
                "executable_checkpoint_files_retained": False,
                "scopes": rows,
            }
            path = temporary / filename
            _write_json(
                path,
                {
                    **index_body,
                    "content_sha256": _sha256_json(index_body),
                },
            )
            native_indexes[filename] = bundle._component_file_registration(
                path, component_root=temporary
            )
        exact_index = {
            "schema_version": bundle.STAGE1_SCOPE_INDEX_SCHEMA,
            "split_registry_content_sha256": (prepared.registry_content_sha256),
            "registered_native_families": list(
                (
                    *bundle.PRODUCTION_BOW_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                    *bundle.PRODUCTION_HTR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                    *bundle.PRODUCTION_MATCHED_PAIR_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                    *bundle.PRODUCTION_EMBEDDING_REGISTERED_NATIVE_FAMILY_ADAPTERS,
                )
            ),
            "native_bow_family_proof_index": native_indexes["bow_native_family_proof_index.json"],
            "native_htr_family_proof_index": native_indexes["htr_native_family_proof_index.json"],
            "native_matched_pair_family_proof_index": native_indexes[
                "matched_pair_native_family_proof_index.json"
            ],
            "native_embedding_family_proof_index": native_indexes[
                "embedding_native_family_proof_index.json"
            ],
            "native_cumulative_legacy_family_proof_index": (cumulative_index_registration),
            "native_cumulative_embedding_family_proof_index": (
                cumulative_embedding_index_registration
            ),
            "embedding_cluster_fit_index": (
                bundle._component_file_registration(cluster_path, component_root=temporary)
            ),
            "scopes": scope_index_rows,
        }
        _write_json(temporary / "exact_scope_index.json", exact_index)
        if any(
            path.name.lower().endswith(
                (
                    ".joblib",
                    ".pkl",
                    ".pickle",
                    ".pt",
                    ".pth",
                    ".ckpt",
                )
            )
            for path in temporary.rglob("*")
        ):
            raise RuntimeError(
                "executable native serialization entered the finalized legacy component"
            )
        builder = bundle.ProductionStage1BundleBuilder.__new__(bundle.ProductionStage1BundleBuilder)
        builder._validate_legacy_scope_lineage(handoff_path, prepared)
        load_legacy_full_outer_evidence(handoff_path)
        # The published component has no terminal component manifest yet; make
        # every byte and directory entry durable before the atomic rename hands
        # it to the caller for final sealing.
        durably_sync_legacy_stage1_tree(temporary)
        os.replace(temporary, destination)
        destination_parent_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(destination_parent_descriptor)
        finally:
            os.close(destination_parent_descriptor)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    # Reopen at the published path rather than relying on objects that referred
    # to the temporary finalization tree.
    published_handoff = destination / "handoff" / "discovery_contexts.jsonl"
    builder._validate_legacy_scope_lineage(published_handoff, prepared)
    load_legacy_full_outer_evidence(published_handoff)
    planned_scopes = tuple(prepared.stage1_scope_plan.scopes)
    return {
        "component_root": str(destination),
        "scope_count": len(planned_scopes),
        "full_outer_scope_count": sum(
            scope.scope_kind == "full_outer" for scope in planned_scopes
        ),
        "exact_inner_scope_count": sum(
            scope.scope_kind == "exact_inner" for scope in planned_scopes
        ),
        "cumulative_spent_scope_count": sum(
            scope.scope_kind == "cumulative_spent"
            for scope in planned_scopes
        ),
        "heldout_labels_supplied_to_workers": False,
        "aggregate_indexes_emitted_by_parent": True,
    }


__all__ = [
    "AuthenticatedLegacyStage1ScopeDescriptor",
    "LEGACY_STAGE1_AGGREGATE_RELATIVE_PATHS",
    "LEGACY_STAGE1_SCOPE_DESCRIPTOR_MANIFEST",
    "LEGACY_STAGE1_SCOPE_DESCRIPTOR_RECOVERY_MANIFEST",
    "LEGACY_STAGE1_SCOPE_DESCRIPTOR_RECOVERY_SCHEMA",
    "LEGACY_STAGE1_SCOPE_DESCRIPTOR_SCHEMA",
    "LEGACY_STAGE1_ONE_SCOPE_AUTHORITY_SCHEMA",
    "LEGACY_STAGE1_SCOPE_WORKER_TARGET",
    "LegacyStage1RoleSpecificDeduplicationError",
    "collect_and_merge_legacy_stage1_scope_attempts",
    "finalize_legacy_stage1_component_from_merge",
    "publish_legacy_stage1_scope_descriptor",
    "run_legacy_stage1_scope_worker",
    "validate_legacy_stage1_scope_descriptor",
]
