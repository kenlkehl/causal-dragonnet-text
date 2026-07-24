"""Authenticated relocation of a sealed production embedding cache.

The production cache metadata intentionally binds the cache to the exact
prepared Parquet path used when the embeddings were built.  Rewriting that
metadata would destroy the original proof.  This module instead:

* re-authenticates the original cache, prepared cohort, and local model;
* proves that a newly prepared cohort is row-for-row identical;
* byte-copies the original prepared cohort and all four cache files;
* preserves the cache metadata without modification; and
* atomically publishes a closed relocation attestation and terminal manifest.

The cache builder module is deliberately not modified by this feature.  Its
stored code hash remains a valid part of the original cache provenance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pandas as pd
from pandas.testing import assert_frame_equal

from . import production_authenticated_tree_cache as authenticated_tree_module
from .production_authenticated_tree_cache import authenticate_directory_tree
from .production_embedding_cache_builder import (
    validate_published_production_embedding_cache,
)
from .production_text_preparation import PREPARATION_SCHEMA

PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION = "production_embedding_cache_relocator_v2"
PRODUCTION_EMBEDDING_CACHE_RELOCATION_ATTESTATION_SCHEMA = (
    "production_embedding_cache_relocation_attestation_v2"
)
PRODUCTION_EMBEDDING_CACHE_RELOCATION_TERMINAL_SCHEMA = (
    "production_embedding_cache_relocation_terminal_v2"
)
PRODUCTION_EMBEDDING_CACHE_RELOCATION_RESULT_SCHEMA = (
    "production_embedding_cache_relocation_result_v2"
)

RELOCATED_PREPARED_RELATIVE_PATH = Path("prepared") / "modeling_cohort.parquet"
RELOCATED_CACHE_RELATIVE_PATH = Path("embedding_cache")
RELOCATION_ATTESTATION_NAME = "relocation_attestation.json"
RELOCATION_TERMINAL_MANIFEST_NAME = "complete_manifest.json"

_CACHE_FILE_NAMES = (
    "metadata.json",
    "chunk_embeddings.npy",
    "offsets.npy",
    "chunk_texts.jsonl",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REGISTRATION_FIELDS = frozenset({"sha256", "size_bytes"})
_BUILDER_LOCAL_MODEL_PROVENANCE_FIELDS = frozenset(
    {"path", "tree_sha256", "file_count", "directory_count", "total_file_bytes"}
)
_PREPARATION_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "columns",
        "row_count",
        "affected_unit_ids",
        "rows",
        "source",
        "output",
        "non_text_values_unchanged",
        "row_order_unchanged",
        "oracle_columns_decoded_or_materialized",
        "content_sha256",
    }
)
_PREPARATION_POLICY_FIELDS = frozenset(
    {
        "identity",
        "empty_text_policy",
        "repeated_character_policy",
        "repeated_character_threshold",
        "missing_note_marker_sha256",
        "run_marker_sha256",
        "transformations_determined_from_text_only",
    }
)
_PREPARATION_COLUMNS_FIELDS = frozenset({"unit_id", "text", "treatment", "outcome"})
_PREPARATION_FILE_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_PREPARATION_ROW_FIELDS = frozenset(
    {
        "row_position",
        "unit_id",
        "before_text_sha256",
        "after_text_sha256",
        "before_length",
        "after_length",
        "transformations",
    }
)
_EMPTY_TRANSFORMATION_FIELDS = frozenset({"kind", "start", "count"})
_RUN_TRANSFORMATION_FIELDS = frozenset(
    {
        "kind",
        "start",
        "end",
        "count",
        "code_point",
        "unicode_category",
        "run_sha256",
    }
)
_ATTESTATION_FIELDS = frozenset(
    {
        "schema_version",
        "relocator_version",
        "relocator_code_sha256",
        "authenticated_tree_code_sha256",
        "source",
        "fresh_preparation",
        "destination",
        "proofs",
        "content_sha256",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "cache_dir",
        "cache_build_identity",
        "prepared_cohort",
        "preparation_manifest",
        "preparation_content_sha256",
        "prepared_projection_sha256",
        "local_model_path",
        "local_model_tree_sha256",
    }
)
_FRESH_PREPARATION_FIELDS = frozenset(
    {
        "prepared_cohort",
        "preparation_manifest",
        "preparation_content_sha256",
        "prepared_projection_sha256",
    }
)
_DESTINATION_FIELDS = frozenset(
    {
        "root",
        "prepared_cohort",
        "prepared_projection_sha256",
        "cache_dir",
        "cache_files",
        "cache_build_identity",
    }
)
_PROOF_FIELDS = frozenset(
    {
        "source_cache_authenticated",
        "source_preparation_authenticated",
        "fresh_preparation_authenticated",
        "source_and_fresh_rows_equal",
        "source_prepared_bytes_copied_exactly",
        "source_cache_bytes_copied_exactly",
        "cache_metadata_unchanged",
        "local_model_tree_authenticated",
        "local_model_revalidation_policy",
        "symlinks_allowed",
        "hardlinks_allowed",
        "atomic_publication",
    }
)
_TERMINAL_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "relocator_version",
        "relocator_code_sha256",
        "authenticated_tree_code_sha256",
        "root",
        "attestation",
        "artifacts",
        "content_sha256",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "relocator_version",
        "relocator_code_sha256",
        "authenticated_tree_code_sha256",
        "root",
        "cache_dir",
        "prepared_cohort_path",
        "attestation_path",
        "terminal_manifest_path",
        "row_count",
        "prepared_projection_sha256",
        "source_cache_identity_sha256",
        "cache_build_identity",
        "attestation_sha256",
        "terminal_manifest_sha256",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=lambda item: item.item() if hasattr(item, "item") else str(item),
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _directory_signature(
    path: Path,
    *,
    label: str,
) -> tuple[int, int, int, int, int, int]:
    state = os.lstat(path)
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise ValueError(f"{label} must be one real directory")
    return _stat_signature(state)


@dataclass(frozen=True)
class _FileSnapshot:
    sha256: str
    size_bytes: int
    signature: tuple[int, int, int, int, int, int]

    def registration(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}


def _real_canonical_directory(path: Path | str, *, label: str) -> Path:
    supplied = Path(path)
    if not supplied.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    try:
        state = os.lstat(supplied)
    except OSError as exc:
        raise FileNotFoundError(f"{label} does not exist: {supplied}") from exc
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISDIR(state.st_mode):
        raise ValueError(f"{label} must be one real directory")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied:
        raise ValueError(f"{label} cannot contain symlinked or non-canonical components")
    return resolved


def _stable_file_snapshot(path: Path | str, *, label: str) -> tuple[Path, _FileSnapshot]:
    supplied = Path(path)
    if not supplied.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    try:
        before_path = os.lstat(supplied)
    except OSError as exc:
        raise FileNotFoundError(f"{label} does not exist: {supplied}") from exc
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"{label} must be one real regular file")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied:
        raise ValueError(f"{label} cannot contain symlinked or non-canonical components")
    descriptor = os.open(
        supplied,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        before_fd = os.fstat(descriptor)
        if _stat_signature(before_fd) != _stat_signature(before_path):
            raise RuntimeError(f"{label} changed while it was opened")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(supplied)
    signature = _stat_signature(before_path)
    if (
        _stat_signature(before_fd) != signature
        or _stat_signature(after_fd) != signature
        or _stat_signature(after_path) != signature
    ):
        raise RuntimeError(f"{label} changed while it was authenticated")
    return resolved, _FileSnapshot(
        sha256=digest.hexdigest(),
        size_bytes=int(before_path.st_size),
        signature=signature,
    )


def _relocator_code_sha256() -> str:
    _path, snapshot = _stable_file_snapshot(
        Path(__file__).resolve(strict=True),
        label="embedding cache relocator module",
    )
    return snapshot.sha256


def _authenticated_tree_code_sha256() -> str:
    _path, snapshot = _stable_file_snapshot(
        Path(authenticated_tree_module.__file__).resolve(strict=True),
        label="authenticated directory tree module",
    )
    return snapshot.sha256


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key in relocation input: {key}")
        output[key] = value
    return output


def _read_json_snapshot(path: Path | str, *, label: str) -> tuple[dict[str, Any], _FileSnapshot]:
    resolved, before = _stable_file_snapshot(path, label=label)
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    _resolved, after = _stable_file_snapshot(resolved, label=label)
    if before != after:
        raise RuntimeError(f"{label} changed while it was decoded")
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value, after


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        encoded = payload.encode("utf-8")
        cursor = 0
        while cursor < len(encoded):
            cursor += os.write(descriptor, encoded[cursor:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _ordered_projection_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    header = {
        "schema_version": "production_relocated_prepared_projection_v1",
        "columns": list(frame.columns),
        "row_count": len(frame),
        "dtypes": [str(value) for value in frame.dtypes],
    }
    digest.update(_canonical_json(header).encode("utf-8"))
    digest.update(b"\n")
    for row in frame.itertuples(index=False, name=None):
        digest.update(_canonical_json([_json_scalar(value) for value in row]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_registration(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_REGISTRATION_FIELDS):
        raise ValueError(f"{label} must be one closed file registration")
    digest = _require_sha256(value.get("sha256"), label=f"{label}.sha256")
    size = value.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"{label}.size_bytes must be a positive integer")
    return {"sha256": digest, "size_bytes": size}


def _require_manifest_file_identity(
    value: Any,
    *,
    expected_path: Path,
    actual_snapshot: _FileSnapshot,
    label: str,
) -> None:
    registration = _require_registration(
        (
            {"sha256": value.get("sha256"), "size_bytes": value.get("size_bytes")}
            if isinstance(value, Mapping)
            else value
        ),
        label=label,
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != set(_PREPARATION_FILE_FIELDS)
        or value.get("path") != str(expected_path)
        or registration != actual_snapshot.registration()
    ):
        raise ValueError(f"{label} differs from its authenticated file")


def _validate_preparation_manifest(
    *,
    cohort_path: Path,
    manifest_path: Path,
    expected_columns: Mapping[str, str],
) -> tuple[dict[str, Any], _FileSnapshot, _FileSnapshot, pd.DataFrame, str]:
    cohort, cohort_snapshot = _stable_file_snapshot(
        cohort_path,
        label="prepared modeling cohort",
    )
    manifest, manifest_snapshot = _read_json_snapshot(
        manifest_path,
        label="text preparation manifest",
    )
    if set(manifest) != set(_PREPARATION_FIELDS):
        raise ValueError("text preparation manifest is not a closed schema")
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    if manifest.get("schema_version") != PREPARATION_SCHEMA or _require_sha256(
        manifest.get("content_sha256"),
        label="preparation_manifest.content_sha256",
    ) != _sha256_json(body):
        raise ValueError("text preparation manifest content identity is invalid")
    policy = manifest.get("policy")
    columns = manifest.get("columns")
    if (
        not isinstance(policy, Mapping)
        or set(policy) != set(_PREPARATION_POLICY_FIELDS)
        or policy.get("identity") != "neutral_marker_unicode_run_v1"
        or policy.get("empty_text_policy") != "marker"
        or policy.get("repeated_character_policy") != "marker"
        or not isinstance(policy.get("repeated_character_threshold"), int)
        or isinstance(policy.get("repeated_character_threshold"), bool)
        or policy["repeated_character_threshold"] < 1
        or policy.get("transformations_determined_from_text_only") is not True
        or not isinstance(columns, Mapping)
        or set(columns) != set(_PREPARATION_COLUMNS_FIELDS)
        or dict(columns) != dict(expected_columns)
        or manifest.get("non_text_values_unchanged") is not True
        or manifest.get("row_order_unchanged") is not True
        or manifest.get("oracle_columns_decoded_or_materialized") is not False
    ):
        raise ValueError("text preparation manifest changed its production policy")
    _require_sha256(policy.get("missing_note_marker_sha256"), label="missing-note marker")
    _require_sha256(policy.get("run_marker_sha256"), label="unicode-run marker")

    source = manifest.get("source")
    output = manifest.get("output")
    if (
        not isinstance(source, Mapping)
        or set(source) != set(_PREPARATION_FILE_FIELDS)
        or not isinstance(output, Mapping)
        or set(output) != set(_PREPARATION_FILE_FIELDS)
    ):
        raise ValueError("text preparation manifest has invalid file identities")
    source_path = Path(str(source.get("path", "")))
    source_resolved, source_snapshot = _stable_file_snapshot(
        source_path,
        label="preparation source dataset",
    )
    _require_manifest_file_identity(
        source,
        expected_path=source_resolved,
        actual_snapshot=source_snapshot,
        label="preparation source",
    )
    _require_manifest_file_identity(
        output,
        expected_path=cohort,
        actual_snapshot=cohort_snapshot,
        label="preparation output",
    )

    configured_column_names = [
        expected_columns["unit_id"],
        expected_columns["text"],
        expected_columns["treatment"],
        expected_columns["outcome"],
    ]
    try:
        frame = pd.read_parquet(cohort, columns=configured_column_names)
    except Exception as exc:
        raise ValueError("prepared modeling cohort could not be read") from exc
    if list(frame.columns) != configured_column_names or len(frame) < 1:
        raise ValueError("prepared modeling cohort changed its four-column schema")
    if (
        frame[expected_columns["unit_id"]].isna().any()
        or frame[expected_columns["unit_id"]].duplicated().any()
        or frame[expected_columns["text"]].isna().any()
        or not all(isinstance(value, str) for value in frame[expected_columns["text"]])
        or frame[expected_columns["treatment"]].isna().any()
        or set(frame[expected_columns["treatment"]].unique().tolist()) != {0, 1}
        or frame[expected_columns["outcome"]].isna().any()
        or set(frame[expected_columns["outcome"]].unique().tolist()) != {0, 1}
    ):
        raise ValueError("prepared modeling cohort violates its row invariants")
    if manifest.get("row_count") != len(frame):
        raise ValueError("preparation row count differs from prepared cohort")

    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != len(frame):
        raise ValueError("preparation row audit is incomplete")
    affected: list[Any] = []
    ids = frame[expected_columns["unit_id"]].tolist()
    texts = frame[expected_columns["text"]].tolist()
    for position, (row, unit_id, text) in enumerate(zip(rows, ids, texts, strict=True)):
        if not isinstance(row, Mapping) or set(row) != set(_PREPARATION_ROW_FIELDS):
            raise ValueError("preparation row audit is not closed")
        transformations = row.get("transformations")
        if (
            row.get("row_position") != position
            or row.get("unit_id") != _json_scalar(unit_id)
            or row.get("after_length") != len(text)
            or row.get("after_text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest()
            or not isinstance(row.get("before_length"), int)
            or isinstance(row.get("before_length"), bool)
            or row["before_length"] < 0
            or _SHA256.fullmatch(str(row.get("before_text_sha256", ""))) is None
            or not isinstance(transformations, list)
        ):
            raise ValueError(f"preparation row audit is invalid at row {position}")
        for transformation in transformations:
            if not isinstance(transformation, Mapping):
                raise ValueError(f"preparation transformation is invalid at row {position}")
            kind = transformation.get("kind")
            expected_fields = (
                _EMPTY_TRANSFORMATION_FIELDS
                if kind == "empty_text"
                else _RUN_TRANSFORMATION_FIELDS if kind == "unicode_run" else frozenset()
            )
            if set(transformation) != set(expected_fields):
                raise ValueError(f"preparation transformation is not closed at row {position}")
            if kind == "unicode_run":
                _require_sha256(
                    transformation.get("run_sha256"),
                    label=f"row {position} unicode run",
                )
        if transformations:
            affected.append(_json_scalar(unit_id))
    if manifest.get("affected_unit_ids") != affected:
        raise ValueError("preparation affected-unit registry differs from row audits")
    return (
        manifest,
        manifest_snapshot,
        cohort_snapshot,
        frame,
        _ordered_projection_sha256(frame),
    )


def _compare_preparations(
    source_manifest: Mapping[str, Any],
    fresh_manifest: Mapping[str, Any],
    source_frame: pd.DataFrame,
    fresh_frame: pd.DataFrame,
    source_projection_sha256: str,
    fresh_projection_sha256: str,
) -> None:
    comparable_fields = (
        "schema_version",
        "policy",
        "columns",
        "row_count",
        "affected_unit_ids",
        "rows",
        "source",
        "non_text_values_unchanged",
        "row_order_unchanged",
        "oracle_columns_decoded_or_materialized",
    )
    if any(
        source_manifest.get(field_name) != fresh_manifest.get(field_name)
        for field_name in comparable_fields
    ):
        raise ValueError("fresh preparation identity differs from source preparation")
    try:
        assert_frame_equal(
            source_frame,
            fresh_frame,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_frame_type=True,
            check_names=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValueError("fresh prepared cohort is not row-for-row identical to source") from exc
    if source_projection_sha256 != fresh_projection_sha256:
        raise RuntimeError("equal prepared cohorts produced different ordered identities")


def _copy_authenticated_file(
    source: Path,
    destination: Path,
    *,
    label: str,
) -> _FileSnapshot:
    source_resolved, before = _stable_file_snapshot(source, label=label)
    source_descriptor = os.open(
        source_resolved,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    destination_descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        source_state = os.fstat(source_descriptor)
        destination_state = os.fstat(destination_descriptor)
        if (source_state.st_dev, source_state.st_ino) == (
            destination_state.st_dev,
            destination_state.st_ino,
        ):
            raise RuntimeError(f"{label} copy unexpectedly shares an inode with its source")
        while block := os.read(source_descriptor, 1024 * 1024):
            cursor = 0
            while cursor < len(block):
                cursor += os.write(destination_descriptor, block[cursor:])
        os.fsync(destination_descriptor)
    finally:
        os.close(destination_descriptor)
        os.close(source_descriptor)
    _source_after_path, source_after = _stable_file_snapshot(source_resolved, label=label)
    _destination_path, destination_snapshot = _stable_file_snapshot(
        destination,
        label=f"copied {label}",
    )
    if source_after != before or destination_snapshot.registration() != before.registration():
        raise RuntimeError(f"{label} changed or copied incorrectly")
    return destination_snapshot


def _require_distinct_file_objects(
    source: Path,
    destination: Path,
    *,
    label: str,
) -> None:
    source_state = os.lstat(source)
    destination_state = os.lstat(destination)
    if (
        stat.S_ISLNK(source_state.st_mode)
        or not stat.S_ISREG(source_state.st_mode)
        or stat.S_ISLNK(destination_state.st_mode)
        or not stat.S_ISREG(destination_state.st_mode)
    ):
        raise ValueError(f"{label} source and destination must be real regular files")
    if (int(source_state.st_dev), int(source_state.st_ino)) == (
        int(destination_state.st_dev),
        int(destination_state.st_ino),
    ):
        raise ValueError(f"{label} destination cannot be a hard link to its source")


def _require_single_link_regular_file(path: Path, *, label: str) -> None:
    state = os.lstat(path)
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode) or int(state.st_nlink) != 1:
        raise ValueError(f"{label} must be one non-linked regular file")


def _cache_registrations(
    root: Path,
    *,
    require_single_link: bool = False,
) -> dict[str, dict[str, Any]]:
    if set(path.name for path in root.iterdir()) != set(_CACHE_FILE_NAMES):
        raise ValueError("relocated cache must contain exactly the four production cache files")
    output: dict[str, dict[str, Any]] = {}
    for name in _CACHE_FILE_NAMES:
        if require_single_link:
            _require_single_link_regular_file(
                root / name,
                label=f"embedding cache {name}",
            )
        _path, snapshot = _stable_file_snapshot(
            root / name,
            label=f"embedding cache {name}",
        )
        output[name] = snapshot.registration()
    return output


def _registrations_for_relative_paths(
    root: Path,
    relative_paths: Sequence[Path],
    *,
    require_single_link: bool = False,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        if require_single_link:
            _require_single_link_regular_file(
                root / relative,
                label=f"relocation artifact {relative.as_posix()}",
            )
        _path, snapshot = _stable_file_snapshot(
            root / relative,
            label=f"relocation artifact {relative.as_posix()}",
        )
        output[relative.as_posix()] = snapshot.registration()
    return output


def _without_cache_path(identity: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(identity))
    value.pop("cache_path", None)
    return value


def _builder_validated_local_model_provenance(
    *,
    cache_root: Path,
    cache_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover model provenance only after the historical cache validator ran."""

    metadata, metadata_snapshot = _read_json_snapshot(
        cache_root / "metadata.json",
        label="production embedding cache metadata",
    )
    cache_files = cache_identity.get("cache_files")
    if (
        not isinstance(cache_files, Mapping)
        or cache_files.get("metadata.json") != metadata_snapshot.registration()
    ):
        raise RuntimeError("builder-validated cache identity no longer binds its metadata bytes")
    production_provenance = metadata.get("production_provenance")
    local_model = (
        production_provenance.get("local_model")
        if isinstance(production_provenance, Mapping)
        else None
    )
    if not isinstance(local_model, Mapping) or set(local_model) != set(
        _BUILDER_LOCAL_MODEL_PROVENANCE_FIELDS
    ):
        raise ValueError("builder-validated cache has invalid local-model provenance")
    provenance = copy.deepcopy(dict(local_model))
    path = provenance.get("path")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or _require_sha256(
            provenance.get("tree_sha256"),
            label="cache local_model.tree_sha256",
        )
        != cache_identity.get("local_model_tree_sha256")
        or not isinstance(provenance.get("file_count"), int)
        or isinstance(provenance.get("file_count"), bool)
        or provenance["file_count"] < 1
        or not isinstance(provenance.get("directory_count"), int)
        or isinstance(provenance.get("directory_count"), bool)
        or provenance["directory_count"] < 0
        or not isinstance(provenance.get("total_file_bytes"), int)
        or isinstance(provenance.get("total_file_bytes"), bool)
        or provenance["total_file_bytes"] < 0
    ):
        raise ValueError("builder-validated cache has invalid local-model provenance")
    return provenance


def _authenticate_local_model_against_builder_cache(
    *,
    local_model_path: Path | str,
    cache_root: Path,
    cache_identity: Mapping[str, Any],
    expected_model_provenance: Mapping[str, Any] | None = None,
    expected_workflow_inventory: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Bind the shared live-tree capability to historical cache provenance."""

    cache_provenance = _builder_validated_local_model_provenance(
        cache_root=cache_root,
        cache_identity=cache_identity,
    )
    (
        model_path,
        live_provenance,
        workflow_inventory,
    ) = _authenticate_expected_local_model(
        local_model_path=local_model_path,
        expected_model_provenance=expected_model_provenance,
        expected_workflow_inventory=expected_workflow_inventory,
    )
    if live_provenance != cache_provenance:
        raise ValueError(
            "production embedding cache provenance differs from the supplied local model"
        )
    return model_path, live_provenance, workflow_inventory


def _authenticate_expected_local_model(
    *,
    local_model_path: Path | str,
    expected_model_provenance: Mapping[str, Any] | None = None,
    expected_workflow_inventory: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Authenticate or cheaply recheck one process-local tree capability."""

    snapshot = authenticate_directory_tree(local_model_path)
    live_provenance = copy.deepcopy(dict(snapshot.local_model_provenance()))
    workflow_inventory = copy.deepcopy(dict(snapshot.workflow_inventory_projection()))
    if expected_model_provenance is not None and live_provenance != dict(expected_model_provenance):
        raise RuntimeError("local model provenance changed during cache relocation")
    if expected_workflow_inventory is not None and workflow_inventory != dict(
        expected_workflow_inventory
    ):
        raise RuntimeError("local model inventory changed during cache relocation")
    return Path(live_provenance["path"]), live_provenance, workflow_inventory


@dataclass(frozen=True)
class ProductionEmbeddingCacheRelocationOptions:
    """All immutable inputs required to create or revalidate one relocation."""

    source_cache_dir: Path
    source_prepared_cohort_path: Path
    source_preparation_manifest_path: Path
    fresh_prepared_cohort_path: Path
    fresh_preparation_manifest_path: Path
    local_model_path: Path
    target_dir: Path
    unit_id_column: str
    text_column: str
    treatment_column: str
    outcome_column: str
    sentence_model_name: str
    chunk_configuration: Mapping[str, Any]


@dataclass(frozen=True)
class AuthenticatedProductionEmbeddingCacheRelocation:
    """Detached result returned only after complete read-only validation."""

    root: Path
    cache_dir: Path
    prepared_cohort_path: Path
    attestation_path: Path
    terminal_manifest_path: Path
    cache_build_identity: Mapping[str, Any] = field(repr=False)
    _identity: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        identity = copy.deepcopy(dict(self._identity))
        if set(identity) != set(_RESULT_FIELDS):
            raise ValueError("relocated cache result identity is not closed")
        if (
            identity.get("schema_version") != PRODUCTION_EMBEDDING_CACHE_RELOCATION_RESULT_SCHEMA
            or identity.get("relocator_version") != PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION
            or identity.get("relocator_code_sha256") != _relocator_code_sha256()
            or identity.get("authenticated_tree_code_sha256") != _authenticated_tree_code_sha256()
        ):
            raise ValueError("relocated cache result changed its implementation identity")
        for field_name in (
            "relocator_code_sha256",
            "authenticated_tree_code_sha256",
            "prepared_projection_sha256",
            "source_cache_identity_sha256",
            "attestation_sha256",
            "terminal_manifest_sha256",
        ):
            _require_sha256(identity.get(field_name), label=f"result.{field_name}")
        if (
            not isinstance(identity.get("row_count"), int)
            or isinstance(identity.get("row_count"), bool)
            or identity["row_count"] < 1
        ):
            raise ValueError("relocated cache result has an invalid row count")
        object.__setattr__(
            self,
            "cache_build_identity",
            MappingProxyType(copy.deepcopy(dict(self.cache_build_identity))),
        )
        object.__setattr__(self, "_identity", MappingProxyType(identity))

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(dict(self._identity))


def _validated_options(
    options: ProductionEmbeddingCacheRelocationOptions,
    *,
    target_must_exist: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    columns = {
        "unit_id": options.unit_id_column,
        "text": options.text_column,
        "treatment": options.treatment_column,
        "outcome": options.outcome_column,
    }
    if len(set(columns.values())) != 4 or any(
        not isinstance(value, str) or not value.strip() for value in columns.values()
    ):
        raise ValueError("relocation requires four distinct non-empty column names")
    if (
        not isinstance(options.sentence_model_name, str)
        or not options.sentence_model_name.strip()
        or options.sentence_model_name != options.sentence_model_name.strip()
    ):
        raise ValueError("sentence_model_name must be one exact non-empty identifier")
    configuration = copy.deepcopy(dict(options.chunk_configuration))
    target = Path(options.target_dir)
    if not target.is_absolute():
        raise ValueError("relocation target_dir must be an absolute path")
    parent = _real_canonical_directory(target.parent, label="relocation target parent")
    if parent != target.parent:
        raise ValueError("relocation target parent must be canonical")
    if target_must_exist:
        _real_canonical_directory(target, label="relocation target")
    elif target.is_symlink() or target.exists():
        raise FileExistsError("relocation target must be fresh and non-symlink")
    return columns, configuration


def _validate_inputs(
    options: ProductionEmbeddingCacheRelocationOptions,
    columns: Mapping[str, str],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    source_cache = _real_canonical_directory(
        options.source_cache_dir,
        label="source embedding cache",
    )
    (
        source_manifest,
        source_manifest_snapshot,
        source_prepared_snapshot,
        source_frame,
        source_projection,
    ) = _validate_preparation_manifest(
        cohort_path=Path(options.source_prepared_cohort_path),
        manifest_path=Path(options.source_preparation_manifest_path),
        expected_columns=columns,
    )
    (
        fresh_manifest,
        fresh_manifest_snapshot,
        fresh_prepared_snapshot,
        fresh_frame,
        fresh_projection,
    ) = _validate_preparation_manifest(
        cohort_path=Path(options.fresh_prepared_cohort_path),
        manifest_path=Path(options.fresh_preparation_manifest_path),
        expected_columns=columns,
    )
    _compare_preparations(
        source_manifest,
        fresh_manifest,
        source_frame,
        fresh_frame,
        source_projection,
        fresh_projection,
    )
    source_prepared, source_prepared_snapshot_after = _stable_file_snapshot(
        options.source_prepared_cohort_path,
        label="source prepared cohort",
    )
    fresh_prepared, fresh_prepared_snapshot_after = _stable_file_snapshot(
        options.fresh_prepared_cohort_path,
        label="fresh prepared cohort",
    )
    source_preparation_manifest, source_manifest_snapshot_after = _stable_file_snapshot(
        options.source_preparation_manifest_path,
        label="source preparation manifest",
    )
    fresh_preparation_manifest, fresh_manifest_snapshot_after = _stable_file_snapshot(
        options.fresh_preparation_manifest_path,
        label="fresh preparation manifest",
    )
    if (
        source_prepared_snapshot_after != source_prepared_snapshot
        or fresh_prepared_snapshot_after != fresh_prepared_snapshot
        or source_manifest_snapshot_after != source_manifest_snapshot
        or fresh_manifest_snapshot_after != fresh_manifest_snapshot
    ):
        raise RuntimeError("prepared cohort or preparation manifest changed during validation")
    (
        initial_model_path,
        initial_model_provenance,
        initial_model_workflow_inventory,
    ) = _authenticate_expected_local_model(
        local_model_path=options.local_model_path,
    )
    source_identity = validate_published_production_embedding_cache(
        cache_dir=source_cache,
        dataset_path=source_prepared,
        text_column=options.text_column,
        sentence_model_name=options.sentence_model_name,
        chunk_configuration=configuration,
        expected_local_model_path=None,
    )
    (
        model_path,
        model_provenance,
        model_workflow_inventory,
    ) = _authenticate_local_model_against_builder_cache(
        local_model_path=initial_model_path,
        cache_root=source_cache,
        cache_identity=source_identity,
        expected_model_provenance=initial_model_provenance,
        expected_workflow_inventory=initial_model_workflow_inventory,
    )
    source_cache_registrations = _cache_registrations(source_cache)
    if source_cache_registrations != source_identity.get("cache_files"):
        raise RuntimeError("source embedding cache changed after historical validation")
    return {
        "source_cache": source_cache,
        "source_cache_registrations": source_cache_registrations,
        "source_cache_identity": copy.deepcopy(dict(source_identity)),
        "source_prepared": source_prepared,
        "source_prepared_snapshot": source_prepared_snapshot,
        "source_manifest": source_manifest,
        "source_manifest_path": source_preparation_manifest,
        "source_manifest_snapshot": source_manifest_snapshot,
        "source_frame": source_frame,
        "source_projection": source_projection,
        "fresh_prepared": fresh_prepared,
        "fresh_prepared_snapshot": fresh_prepared_snapshot,
        "fresh_manifest": fresh_manifest,
        "fresh_manifest_path": fresh_preparation_manifest,
        "fresh_manifest_snapshot": fresh_manifest_snapshot,
        "fresh_frame": fresh_frame,
        "fresh_projection": fresh_projection,
        "model_path": model_path,
        "model_provenance": model_provenance,
        "model_workflow_inventory": model_workflow_inventory,
    }


def _attestation_body(
    *,
    validated: Mapping[str, Any],
    destination_cache_identity: Mapping[str, Any],
    destination_cache_registrations: Mapping[str, Any],
    target: Path,
    copied_prepared_snapshot: _FileSnapshot,
) -> dict[str, Any]:
    target_prepared = target / RELOCATED_PREPARED_RELATIVE_PATH
    target_cache = target / RELOCATED_CACHE_RELATIVE_PATH
    source_identity = validated["source_cache_identity"]
    return {
        "schema_version": PRODUCTION_EMBEDDING_CACHE_RELOCATION_ATTESTATION_SCHEMA,
        "relocator_version": PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION,
        "relocator_code_sha256": _relocator_code_sha256(),
        "authenticated_tree_code_sha256": _authenticated_tree_code_sha256(),
        "source": {
            "cache_dir": str(validated["source_cache"]),
            "cache_build_identity": copy.deepcopy(source_identity),
            "prepared_cohort": {
                "path": str(validated["source_prepared"]),
                **validated["source_prepared_snapshot"].registration(),
            },
            "preparation_manifest": {
                "path": str(validated["source_manifest_path"]),
                **validated["source_manifest_snapshot"].registration(),
            },
            "preparation_content_sha256": validated["source_manifest"]["content_sha256"],
            "prepared_projection_sha256": validated["source_projection"],
            "local_model_path": str(validated["model_path"]),
            "local_model_tree_sha256": source_identity["local_model_tree_sha256"],
        },
        "fresh_preparation": {
            "prepared_cohort": {
                "path": str(validated["fresh_prepared"]),
                **validated["fresh_prepared_snapshot"].registration(),
            },
            "preparation_manifest": {
                "path": str(validated["fresh_manifest_path"]),
                **validated["fresh_manifest_snapshot"].registration(),
            },
            "preparation_content_sha256": validated["fresh_manifest"]["content_sha256"],
            "prepared_projection_sha256": validated["fresh_projection"],
        },
        "destination": {
            "root": str(target),
            "prepared_cohort": {
                "path": str(target_prepared),
                **copied_prepared_snapshot.registration(),
            },
            "prepared_projection_sha256": validated["source_projection"],
            "cache_dir": str(target_cache),
            "cache_files": copy.deepcopy(dict(destination_cache_registrations)),
            "cache_build_identity": copy.deepcopy(dict(destination_cache_identity)),
        },
        "proofs": {
            "source_cache_authenticated": True,
            "source_preparation_authenticated": True,
            "fresh_preparation_authenticated": True,
            "source_and_fresh_rows_equal": True,
            "source_prepared_bytes_copied_exactly": True,
            "source_cache_bytes_copied_exactly": True,
            "cache_metadata_unchanged": True,
            "local_model_tree_authenticated": True,
            "local_model_revalidation_policy": (
                "single_full_hash_process_local_inventory_guard_v1"
            ),
            "symlinks_allowed": False,
            "hardlinks_allowed": False,
            "atomic_publication": "fresh_temp_sibling_directory_rename_v1",
        },
    }


def _remove_owned_directory(path: Path, *, expected_identity: tuple[int, int]) -> None:
    try:
        state = os.lstat(path)
    except FileNotFoundError:
        return
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISDIR(state.st_mode)
        or (int(state.st_dev), int(state.st_ino)) != expected_identity
    ):
        raise RuntimeError("refusing to remove a substituted relocation directory")
    if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
        raise RuntimeError("safe relocation cleanup is unavailable on this platform")
    shutil.rmtree(path)


def relocate_authenticated_production_embedding_cache(
    options: ProductionEmbeddingCacheRelocationOptions,
) -> AuthenticatedProductionEmbeddingCacheRelocation:
    """Authenticate, copy, attest, and atomically publish one cache relocation."""

    columns, configuration = _validated_options(options, target_must_exist=False)
    validated = _validate_inputs(options, columns, configuration)
    target = Path(options.target_dir)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.relocating-",
            dir=target.parent,
        )
    )
    temporary_state = os.lstat(temporary)
    owned_identity = (int(temporary_state.st_dev), int(temporary_state.st_ino))
    published = False
    try:
        (temporary / RELOCATED_PREPARED_RELATIVE_PATH.parent).mkdir()
        (temporary / RELOCATED_CACHE_RELATIVE_PATH).mkdir()
        _authenticate_expected_local_model(
            local_model_path=validated["model_path"],
            expected_model_provenance=validated["model_provenance"],
            expected_workflow_inventory=validated["model_workflow_inventory"],
        )
        copied_prepared = _copy_authenticated_file(
            validated["source_prepared"],
            temporary / RELOCATED_PREPARED_RELATIVE_PATH,
            label="source prepared cohort",
        )
        for name in _CACHE_FILE_NAMES:
            _copy_authenticated_file(
                validated["source_cache"] / name,
                temporary / RELOCATED_CACHE_RELATIVE_PATH / name,
                label=f"source embedding cache {name}",
            )
        _authenticate_expected_local_model(
            local_model_path=validated["model_path"],
            expected_model_provenance=validated["model_provenance"],
            expected_workflow_inventory=validated["model_workflow_inventory"],
        )
        destination_cache_registrations = _cache_registrations(
            temporary / RELOCATED_CACHE_RELATIVE_PATH,
            require_single_link=True,
        )
        if destination_cache_registrations != validated["source_cache_registrations"]:
            raise RuntimeError("relocated cache bytes differ from source cache bytes")
        if copied_prepared.registration() != validated["source_prepared_snapshot"].registration():
            raise RuntimeError("relocated prepared cohort bytes differ from source")
        _authenticate_expected_local_model(
            local_model_path=validated["model_path"],
            expected_model_provenance=validated["model_provenance"],
            expected_workflow_inventory=validated["model_workflow_inventory"],
        )
        temporary_cache_identity = validate_published_production_embedding_cache(
            cache_dir=temporary / RELOCATED_CACHE_RELATIVE_PATH,
            dataset_path=validated["source_prepared"],
            text_column=options.text_column,
            sentence_model_name=options.sentence_model_name,
            chunk_configuration=configuration,
            expected_local_model_path=None,
        )
        _authenticate_local_model_against_builder_cache(
            local_model_path=validated["model_path"],
            cache_root=temporary / RELOCATED_CACHE_RELATIVE_PATH,
            cache_identity=temporary_cache_identity,
            expected_model_provenance=validated["model_provenance"],
            expected_workflow_inventory=validated["model_workflow_inventory"],
        )
        if _without_cache_path(temporary_cache_identity) != _without_cache_path(
            validated["source_cache_identity"]
        ):
            raise RuntimeError("relocated cache identity differs from source cache identity")
        destination_identity = copy.deepcopy(dict(temporary_cache_identity))
        destination_identity["cache_path"] = str(target / RELOCATED_CACHE_RELATIVE_PATH)
        source_metadata = (validated["source_cache"] / "metadata.json").read_bytes()
        copied_metadata = (temporary / RELOCATED_CACHE_RELATIVE_PATH / "metadata.json").read_bytes()
        if copied_metadata != source_metadata:
            raise RuntimeError("relocated cache metadata was modified")

        attestation_body = _attestation_body(
            validated=validated,
            destination_cache_identity=destination_identity,
            destination_cache_registrations=destination_cache_registrations,
            target=target,
            copied_prepared_snapshot=copied_prepared,
        )
        attestation = {
            **attestation_body,
            "content_sha256": _sha256_json(attestation_body),
        }
        _write_json_new(temporary / RELOCATION_ATTESTATION_NAME, attestation)
        relative_artifacts = (
            RELOCATED_PREPARED_RELATIVE_PATH,
            *(RELOCATED_CACHE_RELATIVE_PATH / name for name in _CACHE_FILE_NAMES),
            Path(RELOCATION_ATTESTATION_NAME),
        )
        artifacts = _registrations_for_relative_paths(
            temporary,
            relative_artifacts,
            require_single_link=True,
        )
        _attestation_path, attestation_snapshot = _stable_file_snapshot(
            temporary / RELOCATION_ATTESTATION_NAME,
            label="relocation attestation",
        )
        terminal_body = {
            "schema_version": PRODUCTION_EMBEDDING_CACHE_RELOCATION_TERMINAL_SCHEMA,
            "status": "complete",
            "relocator_version": PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION,
            "relocator_code_sha256": _relocator_code_sha256(),
            "authenticated_tree_code_sha256": _authenticated_tree_code_sha256(),
            "root": str(target),
            "attestation": {
                "path": str(target / RELOCATION_ATTESTATION_NAME),
                **attestation_snapshot.registration(),
                "content_sha256": attestation["content_sha256"],
            },
            "artifacts": artifacts,
        }
        terminal = {**terminal_body, "content_sha256": _sha256_json(terminal_body)}
        _write_json_new(temporary / RELOCATION_TERMINAL_MANIFEST_NAME, terminal)
        if target.is_symlink() or target.exists():
            raise FileExistsError("relocation target was populated during publication")
        os.rename(temporary, target)
        published = True
        return validate_relocated_production_embedding_cache(options)
    except BaseException:
        cleanup_path = target if published else temporary
        _remove_owned_directory(cleanup_path, expected_identity=owned_identity)
        raise


def _validate_attestation_shape(attestation: Mapping[str, Any]) -> None:
    if set(attestation) != set(_ATTESTATION_FIELDS):
        raise ValueError("relocation attestation is not a closed schema")
    body = {
        key: copy.deepcopy(value) for key, value in attestation.items() if key != "content_sha256"
    }
    if (
        attestation.get("schema_version")
        != PRODUCTION_EMBEDDING_CACHE_RELOCATION_ATTESTATION_SCHEMA
        or attestation.get("relocator_version") != PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION
        or attestation.get("relocator_code_sha256") != _relocator_code_sha256()
        or attestation.get("authenticated_tree_code_sha256") != _authenticated_tree_code_sha256()
        or _require_sha256(
            attestation.get("content_sha256"),
            label="attestation.content_sha256",
        )
        != _sha256_json(body)
        or not isinstance(attestation.get("source"), Mapping)
        or set(attestation["source"]) != set(_SOURCE_FIELDS)
        or not isinstance(attestation.get("fresh_preparation"), Mapping)
        or set(attestation["fresh_preparation"]) != set(_FRESH_PREPARATION_FIELDS)
        or not isinstance(attestation.get("destination"), Mapping)
        or set(attestation["destination"]) != set(_DESTINATION_FIELDS)
        or not isinstance(attestation.get("proofs"), Mapping)
        or set(attestation["proofs"]) != set(_PROOF_FIELDS)
    ):
        raise ValueError("relocation attestation changed its authenticated policy")
    expected_proofs = {
        "source_cache_authenticated": True,
        "source_preparation_authenticated": True,
        "fresh_preparation_authenticated": True,
        "source_and_fresh_rows_equal": True,
        "source_prepared_bytes_copied_exactly": True,
        "source_cache_bytes_copied_exactly": True,
        "cache_metadata_unchanged": True,
        "local_model_tree_authenticated": True,
        "local_model_revalidation_policy": ("single_full_hash_process_local_inventory_guard_v1"),
        "symlinks_allowed": False,
        "hardlinks_allowed": False,
        "atomic_publication": "fresh_temp_sibling_directory_rename_v1",
    }
    if dict(attestation["proofs"]) != expected_proofs:
        raise ValueError("relocation attestation proof policy changed")


def validate_relocated_production_embedding_cache(
    options: ProductionEmbeddingCacheRelocationOptions,
) -> AuthenticatedProductionEmbeddingCacheRelocation:
    """Fresh read-only validation suitable for the Stage 1 consumption boundary."""

    columns, configuration = _validated_options(options, target_must_exist=True)
    target = _real_canonical_directory(options.target_dir, label="relocation target")
    target_signature = _directory_signature(target, label="relocation target")
    expected_root_entries = {
        RELOCATED_PREPARED_RELATIVE_PATH.parent.name,
        RELOCATED_CACHE_RELATIVE_PATH.name,
        RELOCATION_ATTESTATION_NAME,
        RELOCATION_TERMINAL_MANIFEST_NAME,
    }
    if set(path.name for path in target.iterdir()) != expected_root_entries:
        raise ValueError("relocation root contains missing or unregistered entries")
    prepared_root = _real_canonical_directory(
        target / RELOCATED_PREPARED_RELATIVE_PATH.parent,
        label="relocated prepared root",
    )
    prepared_root_signature = _directory_signature(
        prepared_root,
        label="relocated prepared root",
    )
    if set(path.name for path in prepared_root.iterdir()) != {
        RELOCATED_PREPARED_RELATIVE_PATH.name
    }:
        raise ValueError("relocated prepared root is not closed")
    cache_root = _real_canonical_directory(
        target / RELOCATED_CACHE_RELATIVE_PATH,
        label="relocated cache root",
    )
    cache_root_signature = _directory_signature(cache_root, label="relocated cache root")
    actual_cache_registrations = _cache_registrations(
        cache_root,
        require_single_link=True,
    )
    _require_single_link_regular_file(
        target / RELOCATION_ATTESTATION_NAME,
        label="relocation attestation",
    )
    _require_single_link_regular_file(
        target / RELOCATION_TERMINAL_MANIFEST_NAME,
        label="relocation terminal manifest",
    )

    attestation, attestation_snapshot = _read_json_snapshot(
        target / RELOCATION_ATTESTATION_NAME,
        label="relocation attestation",
    )
    _validate_attestation_shape(attestation)
    terminal, terminal_snapshot = _read_json_snapshot(
        target / RELOCATION_TERMINAL_MANIFEST_NAME,
        label="relocation terminal manifest",
    )
    if set(terminal) != set(_TERMINAL_FIELDS):
        raise ValueError("relocation terminal manifest is not a closed schema")
    terminal_body = {
        key: copy.deepcopy(value) for key, value in terminal.items() if key != "content_sha256"
    }
    if (
        terminal.get("schema_version") != PRODUCTION_EMBEDDING_CACHE_RELOCATION_TERMINAL_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("relocator_version") != PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION
        or terminal.get("relocator_code_sha256") != _relocator_code_sha256()
        or terminal.get("authenticated_tree_code_sha256") != _authenticated_tree_code_sha256()
        or terminal.get("root") != str(target)
        or _require_sha256(
            terminal.get("content_sha256"),
            label="terminal.content_sha256",
        )
        != _sha256_json(terminal_body)
    ):
        raise ValueError("relocation terminal manifest identity is invalid")
    terminal_attestation = terminal.get("attestation")
    if (
        not isinstance(terminal_attestation, Mapping)
        or set(terminal_attestation) != {"path", "sha256", "size_bytes", "content_sha256"}
        or terminal_attestation.get("path") != str(target / RELOCATION_ATTESTATION_NAME)
        or _require_registration(
            {
                "sha256": terminal_attestation.get("sha256"),
                "size_bytes": terminal_attestation.get("size_bytes"),
            },
            label="terminal.attestation",
        )
        != attestation_snapshot.registration()
        or terminal_attestation.get("content_sha256") != attestation.get("content_sha256")
    ):
        raise ValueError("terminal manifest does not bind the relocation attestation")
    relative_artifacts = (
        RELOCATED_PREPARED_RELATIVE_PATH,
        *(RELOCATED_CACHE_RELATIVE_PATH / name for name in _CACHE_FILE_NAMES),
        Path(RELOCATION_ATTESTATION_NAME),
    )
    actual_artifacts = _registrations_for_relative_paths(
        target,
        relative_artifacts,
        require_single_link=True,
    )
    if terminal.get("artifacts") != actual_artifacts:
        raise ValueError("relocation artifact bytes differ from terminal manifest")

    validated = _validate_inputs(options, columns, configuration)
    destination = attestation["destination"]
    source = attestation["source"]
    fresh = attestation["fresh_preparation"]
    expected_source_prepared = {
        "path": str(validated["source_prepared"]),
        **validated["source_prepared_snapshot"].registration(),
    }
    expected_source_manifest = {
        "path": str(validated["source_manifest_path"]),
        **validated["source_manifest_snapshot"].registration(),
    }
    expected_fresh_prepared = {
        "path": str(validated["fresh_prepared"]),
        **validated["fresh_prepared_snapshot"].registration(),
    }
    expected_fresh_manifest = {
        "path": str(validated["fresh_manifest_path"]),
        **validated["fresh_manifest_snapshot"].registration(),
    }
    if (
        source.get("cache_dir") != str(validated["source_cache"])
        or source.get("cache_build_identity") != validated["source_cache_identity"]
        or source.get("prepared_cohort") != expected_source_prepared
        or source.get("preparation_manifest") != expected_source_manifest
        or source.get("preparation_content_sha256")
        != validated["source_manifest"]["content_sha256"]
        or source.get("prepared_projection_sha256") != validated["source_projection"]
        or source.get("local_model_path") != str(validated["model_path"])
        or source.get("local_model_tree_sha256")
        != validated["source_cache_identity"]["local_model_tree_sha256"]
        or fresh.get("prepared_cohort") != expected_fresh_prepared
        or fresh.get("preparation_manifest") != expected_fresh_manifest
        or fresh.get("preparation_content_sha256") != validated["fresh_manifest"]["content_sha256"]
        or fresh.get("prepared_projection_sha256") != validated["fresh_projection"]
    ):
        raise ValueError("relocation attestation differs from its authenticated inputs")

    copied_prepared_path, copied_prepared_snapshot = _stable_file_snapshot(
        target / RELOCATED_PREPARED_RELATIVE_PATH,
        label="relocated prepared cohort",
    )
    _require_distinct_file_objects(
        validated["source_prepared"],
        copied_prepared_path,
        label="relocated prepared cohort",
    )
    for name in _CACHE_FILE_NAMES:
        _require_distinct_file_objects(
            validated["source_cache"] / name,
            cache_root / name,
            label=f"relocated cache {name}",
        )
    try:
        copied_frame = pd.read_parquet(
            copied_prepared_path,
            columns=list(validated["source_frame"].columns),
        )
        assert_frame_equal(
            validated["source_frame"],
            copied_frame,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_frame_type=True,
            check_names=True,
            check_exact=True,
            check_like=False,
        )
    except Exception as exc:
        raise ValueError("relocated prepared cohort differs from authenticated source") from exc
    copied_projection = _ordered_projection_sha256(copied_frame)
    if (
        copied_prepared_snapshot.registration()
        != validated["source_prepared_snapshot"].registration()
        or copied_projection != validated["source_projection"]
    ):
        raise ValueError("relocated prepared cohort bytes or rows changed")

    _authenticate_expected_local_model(
        local_model_path=validated["model_path"],
        expected_model_provenance=validated["model_provenance"],
        expected_workflow_inventory=validated["model_workflow_inventory"],
    )
    destination_cache_identity = validate_published_production_embedding_cache(
        cache_dir=cache_root,
        dataset_path=validated["source_prepared"],
        text_column=options.text_column,
        sentence_model_name=options.sentence_model_name,
        chunk_configuration=configuration,
        expected_local_model_path=None,
    )
    _authenticate_local_model_against_builder_cache(
        local_model_path=validated["model_path"],
        cache_root=cache_root,
        cache_identity=destination_cache_identity,
        expected_model_provenance=validated["model_provenance"],
        expected_workflow_inventory=validated["model_workflow_inventory"],
    )
    if (
        actual_cache_registrations != validated["source_cache_registrations"]
        or _without_cache_path(destination_cache_identity)
        != _without_cache_path(validated["source_cache_identity"])
        or destination.get("root") != str(target)
        or destination.get("prepared_cohort")
        != {
            "path": str(copied_prepared_path),
            **copied_prepared_snapshot.registration(),
        }
        or destination.get("prepared_projection_sha256") != copied_projection
        or destination.get("cache_dir") != str(cache_root)
        or destination.get("cache_files") != actual_cache_registrations
        or destination.get("cache_build_identity") != destination_cache_identity
    ):
        raise ValueError("relocated cache differs from its source or attestation")

    result_identity = {
        "schema_version": PRODUCTION_EMBEDDING_CACHE_RELOCATION_RESULT_SCHEMA,
        "relocator_version": PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION,
        "relocator_code_sha256": _relocator_code_sha256(),
        "authenticated_tree_code_sha256": _authenticated_tree_code_sha256(),
        "root": str(target),
        "cache_dir": str(cache_root),
        "prepared_cohort_path": str(copied_prepared_path),
        "attestation_path": str(target / RELOCATION_ATTESTATION_NAME),
        "terminal_manifest_path": str(target / RELOCATION_TERMINAL_MANIFEST_NAME),
        "row_count": len(copied_frame),
        "prepared_projection_sha256": copied_projection,
        "source_cache_identity_sha256": _sha256_json(validated["source_cache_identity"]),
        "cache_build_identity": copy.deepcopy(dict(destination_cache_identity)),
        "attestation_sha256": attestation_snapshot.sha256,
        "terminal_manifest_sha256": terminal_snapshot.sha256,
    }
    if (
        _directory_signature(target, label="relocation target") != target_signature
        or _directory_signature(
            prepared_root,
            label="relocated prepared root",
        )
        != prepared_root_signature
        or _directory_signature(cache_root, label="relocated cache root") != cache_root_signature
        or _cache_registrations(cache_root, require_single_link=True) != actual_cache_registrations
        or _registrations_for_relative_paths(
            target,
            relative_artifacts,
            require_single_link=True,
        )
        != actual_artifacts
        or _stable_file_snapshot(
            target / RELOCATION_TERMINAL_MANIFEST_NAME,
            label="relocation terminal manifest",
        )[1]
        != terminal_snapshot
    ):
        raise RuntimeError("relocation root or artifacts changed while they were validated")
    return AuthenticatedProductionEmbeddingCacheRelocation(
        root=target,
        cache_dir=cache_root,
        prepared_cohort_path=copied_prepared_path,
        attestation_path=target / RELOCATION_ATTESTATION_NAME,
        terminal_manifest_path=target / RELOCATION_TERMINAL_MANIFEST_NAME,
        cache_build_identity=destination_cache_identity,
        _identity=result_identity,
    )


__all__ = [
    "AuthenticatedProductionEmbeddingCacheRelocation",
    "PRODUCTION_EMBEDDING_CACHE_RELOCATION_ATTESTATION_SCHEMA",
    "PRODUCTION_EMBEDDING_CACHE_RELOCATION_RESULT_SCHEMA",
    "PRODUCTION_EMBEDDING_CACHE_RELOCATION_TERMINAL_SCHEMA",
    "PRODUCTION_EMBEDDING_CACHE_RELOCATOR_VERSION",
    "ProductionEmbeddingCacheRelocationOptions",
    "relocate_authenticated_production_embedding_cache",
    "validate_relocated_production_embedding_cache",
]
