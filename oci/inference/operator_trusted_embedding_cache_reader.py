"""Stat-guarded embedding-cache reuse from an explicit prior trust proof.

This module implements a deliberately narrow research-workflow exception:
an operator may reuse an embedding cache whose bytes were authenticated by a
previous, sealed portable-artifact adoption without hashing or privately
copying those payloads again.  The ordinary cache reader and portable
artifact validators remain unchanged.

The proof builder consumes a live :class:`OperatorTrustedCheckpoint`, so an
arbitrary digest mapping cannot create this capability.  Subsequent readers
open the exact proved files with ``O_NOFOLLOW``, compare retained file
descriptors and paths to the proved stat inventory, and mmap numerical arrays
read-only.  This is not a fresh-byte audit and cannot support global release
certification.
"""

from __future__ import annotations

import copy
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np

from .operator_trusted_checkpoint_adoption import OperatorTrustedCheckpoint
from .portable_identity import identity_sha256
from .review_spent_evidence_provider import (
    SpentOnlyFrozenChunkEmbeddingCache,
    _snapshot_line_spans,
)


OPERATOR_TRUSTED_CACHE_READ_PROOF_SCHEMA = (
    "operator_trusted_frozen_embedding_cache_read_proof_v1"
)
_FILES = (
    "metadata.json",
    "chunk_embeddings.npy",
    "offsets.npy",
    "chunk_texts.jsonl",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROOF_FIELDS = frozenset(
    {
        "schema_version",
        "operator_trust_explicit",
        "artifact_id",
        "artifact_kind",
        "artifact_compatibility_key",
        "artifact_payload_root",
        "prior_adoption_attestation_path",
        "prior_adoption_attestation_sha256",
        "prior_adoption_attestation_size_bytes",
        "prior_adoption_attestation_content_sha256",
        "prior_consumer_request_sha256",
        "prior_full_byte_validation_recorded_at",
        "cache_dir",
        "cache_dir_stat_identity",
        "cache_files",
        "cache_build_identity",
        "provider_identity",
        "legacy_terminal_migration_identity",
        "payload_bytes_reauthenticated",
        "fresh_full_byte_validation_achieved",
        "global_release_certified",
        "scientific_content_sha256",
        "content_sha256",
    }
)
_PROVIDER_FIELDS = frozenset(
    {
        "provider",
        "metadata_sha256",
        "embeddings_sha256",
        "offsets_sha256",
        "chunk_texts_sha256",
        "row_count",
        "chunk_count",
        "cache_snapshot_authentication",
        "chunk_text_storage",
        "embeddings_path_backed",
        "private_snapshot_embedding_mmap",
        "future_row_text_decoded",
        "novel_text_encoding_allowed",
    }
)
_PROVIDER_DIGEST_FIELD = {
    "metadata.json": "metadata_sha256",
    "chunk_embeddings.npy": "embeddings_sha256",
    "offsets.npy": "offsets_sha256",
    "chunk_texts.jsonl": "chunk_texts_sha256",
}


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypeError(f"{label} must be closed finite JSON") from exc


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return text


def _migration_identity(value: Any) -> dict[str, Any]:
    migration = _json_copy(
        value,
        label="operator-trusted cache legacy migration identity",
    )
    if not isinstance(migration, dict):
        raise TypeError(
            "operator-trusted cache legacy migration identity must be one mapping"
        )
    body = {
        key: copy.deepcopy(child)
        for key, child in migration.items()
        if key != "content_sha256"
    }
    if (
        not migration
        or migration.get("schema_version")
        != "legacy_terminal_typed_request_migration_identity_v1"
        or migration.get("phase") != "embedding_cache"
        or migration.get("content_sha256") != identity_sha256(body)
    ):
        raise ValueError(
            "operator-trusted cache legacy migration identity is not sealed"
        )
    return migration


def _scientific_proof_body(
    *,
    artifact_id: str,
    artifact_compatibility_key: str,
    cache_files: list[Mapping[str, Any]],
    cache_build_identity: Mapping[str, Any],
    provider_identity: Mapping[str, Any],
    migration_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": (
            "operator_trusted_frozen_embedding_cache_scientific_binding_v1"
        ),
        "artifact_id": artifact_id,
        "artifact_kind": "embedding_cache",
        "artifact_compatibility_key": artifact_compatibility_key,
        "cache_files": [
            {
                "name": row["name"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
            }
            for row in cache_files
        ],
        "cache_build_identity": copy.deepcopy(dict(cache_build_identity)),
        "provider_identity": copy.deepcopy(dict(provider_identity)),
        "legacy_terminal_migration_identity": copy.deepcopy(
            dict(migration_identity)
        ),
        "payload_bytes_reauthenticated": False,
    }


def _validate_identity_bindings(
    *,
    cache_files: list[Mapping[str, Any]],
    cache_build_identity: Mapping[str, Any],
    provider_identity: Mapping[str, Any],
) -> None:
    if set(provider_identity) != _PROVIDER_FIELDS:
        raise ValueError(
            "operator-trusted cache provider identity is not the closed "
            "frozen-cache schema"
        )
    rows = {str(row["name"]): row for row in cache_files}
    if set(rows) != set(_FILES) or len(rows) != len(cache_files):
        raise ValueError(
            "operator-trusted cache proof must bind exactly four cache files"
        )
    cache_file_identity = cache_build_identity.get("cache_files")
    if (
        not isinstance(cache_file_identity, Mapping)
        or set(cache_file_identity) != set(_FILES)
        or cache_build_identity.get("provider_identity") != provider_identity
    ):
        raise ValueError(
            "operator-trusted cache build identity does not bind the provider "
            "and exact cache files"
        )
    for name in _FILES:
        row = rows[name]
        expected = cache_file_identity[name]
        if (
            not isinstance(expected, Mapping)
            or set(expected) != {"sha256", "size_bytes"}
            or row["sha256"] != expected["sha256"]
            or row["size_bytes"] != expected["size_bytes"]
            or provider_identity[_PROVIDER_DIGEST_FIELD[name]]
            != row["sha256"]
        ):
            raise ValueError(
                f"operator-trusted cache identity differs for {name}"
            )
    if (
        provider_identity["row_count"] != cache_build_identity.get("row_count")
        or provider_identity["chunk_count"]
        != cache_build_identity.get("chunk_count")
    ):
        raise ValueError(
            "operator-trusted cache provider shape differs from build identity"
        )


def build_operator_trusted_cache_read_proof(
    trusted_checkpoint: OperatorTrustedCheckpoint,
    *,
    cache_dir: Path,
    cache_build_identity: Mapping[str, Any],
    provider_identity: Mapping[str, Any],
    migration_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the closed Stage-1 reader proof from one live trusted handle."""

    if not isinstance(trusted_checkpoint, OperatorTrustedCheckpoint):
        raise TypeError(
            "operator-trusted cache proof requires a validated checkpoint handle"
        )
    artifact = trusted_checkpoint.artifact
    if artifact.manifest.get("artifact_kind") != "embedding_cache":
        raise ValueError(
            "operator-trusted cache proof requires an embedding-cache artifact"
        )
    supplied_cache = Path(cache_dir)
    if (
        not supplied_cache.is_absolute()
        or supplied_cache.is_symlink()
        or not supplied_cache.is_dir()
    ):
        raise ValueError(
            "operator-trusted cache directory must be one absolute "
            "non-symlink directory"
        )
    canonical_cache = supplied_cache.resolve(strict=True)
    try:
        cache_relative = canonical_cache.relative_to(artifact.payload_root)
    except ValueError as exc:
        raise ValueError(
            "operator-trusted cache directory lies outside its proved artifact"
        ) from exc
    cache_state = os.lstat(canonical_cache)
    if stat.S_ISLNK(cache_state.st_mode) or not stat.S_ISDIR(cache_state.st_mode):
        raise ValueError(
            "operator-trusted cache directory is not a regular directory"
        )

    observed_names = {
        child.name
        for child in canonical_cache.iterdir()
    }
    if observed_names != set(_FILES):
        raise ValueError(
            "operator-trusted cache directory must contain exactly its four "
            "registered files"
        )
    registrations = {
        row.relative_path: row
        for row in artifact.payloads
    }
    trusted_stats = {
        str(row["relative_path"]): tuple(row["stat_identity"])
        for row in trusted_checkpoint.payload_stat_inventory
    }
    rows: list[dict[str, Any]] = []
    for name in _FILES:
        relative = (cache_relative / name).as_posix()
        registration = registrations.get(relative)
        expected_stat = trusted_stats.get(relative)
        path = canonical_cache / name
        state = os.lstat(path)
        current_stat = _stat_identity(state)
        if (
            registration is None
            or expected_stat is None
            or stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or current_stat != expected_stat
            or int(state.st_size) != registration.size_bytes
        ):
            raise ValueError(
                f"operator-trusted cache file lost proved stat continuity: {name}"
            )
        rows.append(
            {
                "name": name,
                "relative_path": relative,
                "sha256": registration.sha256,
                "size_bytes": registration.size_bytes,
                "stat_identity": list(current_stat),
            }
        )

    build_identity = _json_copy(
        cache_build_identity,
        label="operator-trusted cache build identity",
    )
    provider = _json_copy(
        provider_identity,
        label="operator-trusted cache provider identity",
    )
    if not isinstance(build_identity, dict) or not isinstance(provider, dict):
        raise TypeError(
            "operator-trusted cache identities must be mappings"
        )
    migration = _migration_identity(migration_identity)
    _validate_identity_bindings(
        cache_files=rows,
        cache_build_identity=build_identity,
        provider_identity=provider,
    )
    artifact_id = _require_sha256(
        artifact.artifact_id,
        label="operator-trusted cache artifact ID",
    )
    compatibility_key = _require_sha256(
        artifact.compatibility_key,
        label="operator-trusted cache compatibility key",
    )
    scientific_body = _scientific_proof_body(
        artifact_id=artifact_id,
        artifact_compatibility_key=compatibility_key,
        cache_files=rows,
        cache_build_identity=build_identity,
        provider_identity=provider,
        migration_identity=migration,
    )
    body = {
        "schema_version": OPERATOR_TRUSTED_CACHE_READ_PROOF_SCHEMA,
        "operator_trust_explicit": True,
        "artifact_id": artifact_id,
        "artifact_kind": "embedding_cache",
        "artifact_compatibility_key": compatibility_key,
        "artifact_payload_root": str(artifact.payload_root),
        "prior_adoption_attestation_path": str(
            trusted_checkpoint.prior_attestation_path
        ),
        "prior_adoption_attestation_sha256": (
            trusted_checkpoint.prior_attestation_sha256
        ),
        "prior_adoption_attestation_size_bytes": (
            trusted_checkpoint.prior_attestation_size_bytes
        ),
        "prior_adoption_attestation_content_sha256": (
            trusted_checkpoint.prior_attestation_content_sha256
        ),
        "prior_consumer_request_sha256": (
            trusted_checkpoint.prior_consumer_request_sha256
        ),
        "prior_full_byte_validation_recorded_at": (
            trusted_checkpoint.prior_recorded_at
        ),
        "cache_dir": str(canonical_cache),
        "cache_dir_stat_identity": list(_stat_identity(cache_state)),
        "cache_files": rows,
        "cache_build_identity": build_identity,
        "provider_identity": provider,
        "legacy_terminal_migration_identity": migration,
        "payload_bytes_reauthenticated": False,
        "fresh_full_byte_validation_achieved": False,
        "global_release_certified": False,
        "scientific_content_sha256": identity_sha256(scientific_body),
    }
    return {**body, "content_sha256": identity_sha256(body)}


def validate_operator_trusted_cache_read_proof(
    proof: Mapping[str, Any],
    *,
    cache_dir: Path,
) -> Mapping[str, Any]:
    """Validate the closed proof and current stat inventory without hashing."""

    value = _json_copy(
        proof,
        label="operator-trusted cache read proof",
    )
    if not isinstance(value, dict) or set(value) != _PROOF_FIELDS:
        raise ValueError("operator-trusted cache read proof is not closed")
    body = {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key != "content_sha256"
    }
    if (
        value.get("schema_version") != OPERATOR_TRUSTED_CACHE_READ_PROOF_SCHEMA
        or value.get("operator_trust_explicit") is not True
        or value.get("artifact_kind") != "embedding_cache"
        or value.get("payload_bytes_reauthenticated") is not False
        or value.get("fresh_full_byte_validation_achieved") is not False
        or value.get("global_release_certified") is not False
        or value.get("content_sha256") != identity_sha256(body)
    ):
        raise ValueError("operator-trusted cache read proof is invalid")
    artifact_id = _require_sha256(
        value["artifact_id"],
        label="operator-trusted cache artifact ID",
    )
    compatibility_key = _require_sha256(
        value["artifact_compatibility_key"],
        label="operator-trusted cache compatibility key",
    )
    for field in (
        "prior_adoption_attestation_sha256",
        "prior_adoption_attestation_content_sha256",
        "prior_consumer_request_sha256",
    ):
        _require_sha256(value[field], label=field)
    if (
        not isinstance(value["prior_adoption_attestation_size_bytes"], int)
        or value["prior_adoption_attestation_size_bytes"] <= 0
        or not isinstance(
            value["prior_full_byte_validation_recorded_at"], str
        )
        or not value["prior_full_byte_validation_recorded_at"].strip()
    ):
        raise ValueError(
            "operator-trusted cache prior attestation binding is invalid"
        )
    for field in (
        "artifact_payload_root",
        "prior_adoption_attestation_path",
        "cache_dir",
    ):
        if not isinstance(value[field], str) or not Path(value[field]).is_absolute():
            raise ValueError(
                f"operator-trusted cache operational locator is invalid: {field}"
            )

    supplied_cache = Path(cache_dir)
    if (
        not supplied_cache.is_absolute()
        or supplied_cache.is_symlink()
        or not supplied_cache.is_dir()
    ):
        raise ValueError(
            "operator-trusted cache directory must be one absolute "
            "non-symlink directory"
        )
    canonical_cache = supplied_cache.resolve(strict=True)
    if str(canonical_cache) != value["cache_dir"]:
        raise ValueError(
            "operator-trusted cache directory differs from its sealed proof"
        )
    payload_root = Path(value["artifact_payload_root"])
    if (
        payload_root.is_symlink()
        or payload_root.resolve(strict=True) != payload_root
    ):
        raise ValueError(
            "operator-trusted cache payload root changed or became a symlink"
        )
    try:
        cache_relative = canonical_cache.relative_to(payload_root)
    except ValueError as exc:
        raise ValueError(
            "operator-trusted cache directory lies outside its artifact payload"
        ) from exc
    current_root_stat = _stat_identity(os.lstat(canonical_cache))
    if current_root_stat != tuple(value["cache_dir_stat_identity"]):
        raise ValueError(
            "operator-trusted cache directory stat identity changed"
        )

    raw_rows = value["cache_files"]
    if not isinstance(raw_rows, list) or len(raw_rows) != len(_FILES):
        raise ValueError(
            "operator-trusted cache proof must bind exactly four cache files"
        )
    rows: list[Mapping[str, Any]] = []
    for expected_name, raw in zip(_FILES, raw_rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "name",
            "relative_path",
            "sha256",
            "size_bytes",
            "stat_identity",
        }:
            raise ValueError(
                "operator-trusted cache file proof is not closed"
            )
        row = dict(raw)
        relative = (cache_relative / expected_name).as_posix()
        if (
            row["name"] != expected_name
            or row["relative_path"] != relative
            or _SHA256.fullmatch(str(row["sha256"])) is None
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] <= 0
            or not isinstance(row["stat_identity"], list)
            or len(row["stat_identity"]) != 7
        ):
            raise ValueError(
                f"operator-trusted cache file proof is invalid: {expected_name}"
            )
        path = canonical_cache / expected_name
        state = os.lstat(path)
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or _stat_identity(state) != tuple(row["stat_identity"])
            or int(state.st_size) != row["size_bytes"]
        ):
            raise ValueError(
                f"operator-trusted cache file stat identity changed: {expected_name}"
            )
        rows.append(row)
    if {
        child.name
        for child in canonical_cache.iterdir()
    } != set(_FILES):
        raise ValueError(
            "operator-trusted cache directory gained or lost files"
        )

    build_identity = value["cache_build_identity"]
    provider_identity = value["provider_identity"]
    if not isinstance(build_identity, Mapping) or not isinstance(
        provider_identity, Mapping
    ):
        raise TypeError(
            "operator-trusted cache proof identities must be mappings"
        )
    migration = _migration_identity(
        value["legacy_terminal_migration_identity"]
    )
    _validate_identity_bindings(
        cache_files=rows,
        cache_build_identity=build_identity,
        provider_identity=provider_identity,
    )
    scientific_body = _scientific_proof_body(
        artifact_id=artifact_id,
        artifact_compatibility_key=compatibility_key,
        cache_files=rows,
        cache_build_identity=build_identity,
        provider_identity=provider_identity,
        migration_identity=migration,
    )
    if value["scientific_content_sha256"] != identity_sha256(
        scientific_body
    ):
        raise ValueError(
            "operator-trusted cache scientific binding changed"
        )
    return copy.deepcopy(value)


def _open_readonly_nofollow(
    *,
    root_fd: int,
    name: str,
    expected_stat: tuple[int, ...],
) -> BinaryIO:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError(
            "operator-trusted cache reuse requires POSIX O_NOFOLLOW"
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(name, flags, dir_fd=root_fd)
    try:
        state = os.fstat(descriptor)
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or _stat_identity(state) != expected_stat
        ):
            raise ValueError(
                f"operator-trusted cache descriptor changed: {name}"
            )
        return os.fdopen(descriptor, "rb", closefd=True)
    except BaseException:
        os.close(descriptor)
        raise


def _load_readonly_mmap(handle: BinaryIO, *, name: str) -> np.ndarray:
    try:
        loaded = np.load(
            f"/proc/self/fd/{handle.fileno()}",
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(
            f"operator-trusted cache contains an invalid {name} array"
        ) from exc
    if not isinstance(loaded, np.ndarray):
        raise ValueError(
            f"operator-trusted cache {name} must be one NumPy array"
        )
    loaded.setflags(write=False)
    return loaded


def _validated_authenticated_line_spans(
    value: Sequence[Sequence[int]],
    *,
    row_count: int,
    proved_size: int,
) -> tuple[tuple[int, int], ...]:
    """Validate a caller-authenticated JSONL byte-range index.

    This deliberately validates only the closed structural contract.  The
    caller is responsible for authenticating the index against the already
    proved chunk-text bytes before supplying it.
    """

    if not isinstance(value, (list, tuple)) or len(value) != row_count:
        raise ValueError(
            "authenticated line spans must contain exactly one span per row"
        )
    spans: list[tuple[int, int]] = []
    cursor = 0
    for raw_span in value:
        if (
            not isinstance(raw_span, (list, tuple))
            or len(raw_span) != 2
            or type(raw_span[0]) is not int
            or type(raw_span[1]) is not int
        ):
            raise ValueError(
                "authenticated line spans must contain closed integer pairs"
            )
        start, stop = raw_span
        if start != cursor or stop <= start or stop > proved_size:
            raise ValueError(
                "authenticated line spans must provide contiguous nonempty "
                "coverage"
            )
        spans.append((start, stop))
        cursor = stop
    if cursor != proved_size:
        raise ValueError(
            "authenticated line spans must cover the exact proved file size"
        )
    return tuple(spans)


class OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache(
    SpentOnlyFrozenChunkEmbeddingCache
):
    """Read a proved cache via retained descriptors and read-only mmap.

    The returned provider identity is the original authenticated producer
    identity, because the scientific cache bytes are unchanged.  The separate
    operator-trust proof records that this process performed stat continuity
    checks rather than a fresh full-byte authentication.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        proof: Mapping[str, Any],
        authenticated_line_spans: Sequence[Sequence[int]] | None = None,
    ) -> None:
        canonical_cache = Path(cache_dir)
        validated = validate_operator_trusted_cache_read_proof(
            proof,
            cache_dir=canonical_cache,
        )
        canonical_cache = canonical_cache.resolve(strict=True)
        rows = {
            str(row["name"]): row
            for row in validated["cache_files"]
        }
        supplied_line_spans = (
            None
            if authenticated_line_spans is None
            else _validated_authenticated_line_spans(
                authenticated_line_spans,
                row_count=int(validated["provider_identity"]["row_count"]),
                proved_size=int(rows["chunk_texts.jsonl"]["size_bytes"]),
            )
        )
        if not hasattr(os, "O_NOFOLLOW"):
            raise RuntimeError(
                "operator-trusted cache reuse requires POSIX O_NOFOLLOW"
            )
        root_flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_DIRECTORY"):
            root_flags |= os.O_DIRECTORY
        if hasattr(os, "O_CLOEXEC"):
            root_flags |= os.O_CLOEXEC
        root_fd = os.open(canonical_cache, root_flags)
        expected_root = tuple(validated["cache_dir_stat_identity"])
        try:
            if _stat_identity(os.fstat(root_fd)) != expected_root:
                raise ValueError(
                    "operator-trusted cache directory changed while opening"
                )
            handles = {
                name: _open_readonly_nofollow(
                    root_fd=root_fd,
                    name=name,
                    expected_stat=tuple(rows[name]["stat_identity"]),
                )
                for name in _FILES
            }
        except BaseException:
            os.close(root_fd)
            raise

        self.cache_dir = canonical_cache
        self._cache_root_fd = root_fd
        self._snapshot_files = handles
        self._proof = copy.deepcopy(dict(validated))
        self._proved_file_stats = {
            name: tuple(rows[name]["stat_identity"])
            for name in _FILES
        }
        metadata_size = int(rows["metadata.json"]["size_bytes"])
        metadata_bytes = os.pread(
            handles["metadata.json"].fileno(),
            metadata_size,
            0,
        )
        if len(metadata_bytes) != metadata_size:
            raise RuntimeError(
                "operator-trusted cache metadata ended unexpectedly"
            )
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "operator-trusted cache metadata is invalid JSON"
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError(
                "operator-trusted cache metadata must be one object"
            )
        self._metadata = metadata
        self._embeddings = _load_readonly_mmap(
            handles["chunk_embeddings.npy"],
            name="chunk_embeddings",
        )
        self._offsets = _load_readonly_mmap(
            handles["offsets.npy"],
            name="offsets",
        )
        row_count = int(metadata.get("num_samples", -1))
        hidden_size = int(metadata.get("hidden_size", -1))
        if row_count < 1 or self._embeddings.ndim != 2:
            raise ValueError(
                "operator-trusted cache has invalid metadata or matrix rank"
            )
        if self._offsets.ndim != 1 or len(self._offsets) != row_count + 1:
            raise ValueError(
                "operator-trusted cache offsets do not match row count"
            )
        if not np.issubdtype(self._offsets.dtype, np.integer):
            raise ValueError(
                "operator-trusted cache offsets must be integers"
            )
        if int(self._offsets[-1]) != int(self._embeddings.shape[0]):
            raise ValueError(
                "operator-trusted cache offsets do not span chunk matrix"
            )
        if hidden_size != int(self._embeddings.shape[1]):
            raise ValueError(
                "operator-trusted cache hidden size is inconsistent"
            )
        provider = validated["provider_identity"]
        if (
            int(provider["row_count"]) != row_count
            or int(provider["chunk_count"]) != int(self._embeddings.shape[0])
        ):
            raise ValueError(
                "operator-trusted cache arrays differ from provider shape"
            )
        typed = validated["legacy_terminal_migration_identity"].get(
            "typed_expectation"
        )
        configured_dtype = (
            typed.get("chunk_configuration", {}).get("stored_array_dtype")
            if isinstance(typed, Mapping)
            else None
        )
        if configured_dtype != str(self._embeddings.dtype):
            raise ValueError(
                "operator-trusted cache array dtype differs from migration proof"
            )

        self._chunk_text_snapshot = handles["chunk_texts.jsonl"]
        self._line_spans = (
            supplied_line_spans
            if supplied_line_spans is not None
            else _snapshot_line_spans(
                self._chunk_text_snapshot,
                size=int(rows["chunk_texts.jsonl"]["size_bytes"]),
            )
        )
        if len(self._line_spans) != row_count:
            raise ValueError(
                "operator-trusted chunk-text registry does not match row count"
            )
        self._identity = copy.deepcopy(dict(provider))
        self._assert_proved_files_unchanged()

    @property
    def operator_trusted_read_proof(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._proof)

    def _assert_proved_files_unchanged(self) -> None:
        if _stat_identity(os.fstat(self._cache_root_fd)) != tuple(
            self._proof["cache_dir_stat_identity"]
        ):
            raise RuntimeError(
                "operator-trusted cache directory changed during use"
            )
        for name in _FILES:
            handle = self._snapshot_files[name]
            expected = self._proved_file_stats[name]
            try:
                descriptor_stat = _stat_identity(os.fstat(handle.fileno()))
                path_stat = _stat_identity(os.lstat(self.cache_dir / name))
            except OSError as exc:
                raise RuntimeError(
                    "operator-trusted cache path changed during use: "
                    f"{name}"
                ) from exc
            if descriptor_stat != expected or path_stat != expected:
                raise RuntimeError(
                    "operator-trusted cache stat identity changed during use: "
                    f"{name}"
                )
        if {
            child.name
            for child in self.cache_dir.iterdir()
        } != set(_FILES):
            raise RuntimeError(
                "operator-trusted cache directory inventory changed during use"
            )

    def authenticated_snapshot_identity(self) -> Mapping[str, Any]:
        """Return inherited digests after stat checks, without hashing bytes."""

        self._assert_proved_files_unchanged()
        return copy.deepcopy(self._identity)

    def identity(self) -> Mapping[str, Any]:
        """Return inherited digests after stat checks, without hashing bytes."""

        self._assert_proved_files_unchanged()
        return copy.deepcopy(self._identity)


def cache_build_identity_from_operator_trusted_proof(
    proof: Mapping[str, Any],
    *,
    cache_dir: Path,
) -> Mapping[str, Any]:
    validated = validate_operator_trusted_cache_read_proof(
        proof,
        cache_dir=cache_dir,
    )
    return copy.deepcopy(validated["cache_build_identity"])


__all__ = [
    "OPERATOR_TRUSTED_CACHE_READ_PROOF_SCHEMA",
    "OperatorTrustedSpentOnlyFrozenChunkEmbeddingCache",
    "build_operator_trusted_cache_read_proof",
    "cache_build_identity_from_operator_trusted_proof",
    "validate_operator_trusted_cache_read_proof",
]
