"""Portable, lossless clustered-preflight artifact.

Version 1 embedded the complete clustered concept collections in both the
audit JSON and the Stage 1 request JSON.  For a real cohort that duplicated
several gigabytes and forced every consumer to materialize one enormous
mapping.  This version keeps the scientific content unchanged while changing
only its physical representation:

* the Stage 1 request contains one closed, path-neutral audit reference;
* the audit index contains small logical bindings and one compact fit record
  per physical owner;
* each physical owner's lossless concepts occur once in an ordered Parquet
  payload;
* logical aliases name their canonical physical owner and never copy concept
  bytes;
* numerical KMeans/SVD state remains in the existing individual ``.npy``
  state bundle.

The fresh loader hashes every registered byte.  It parses owner concept
payloads only when ``owner_fit_identity``/``logical_scope_record`` is called,
then proves that the reconstructed identity has the exact source collection
and fit roots recorded during preflight.
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
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

PORTABLE_CLUSTER_PREFLIGHT_ARTIFACT_VERSION = "production_stage1_cluster_preflight_artifact_v2"
PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_SCHEMA = "production_stage1_cluster_preflight_manifest_v2"
PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_SCHEMA = "production_stage1_cluster_preflight_audit_index_v2"
PORTABLE_CLUSTER_PREFLIGHT_REFERENCE_SCHEMA = "production_stage1_cluster_preflight_reference_v2"
PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_SCHEMA = (
    "production_stage1_cluster_preflight_scientific_request_v2"
)
PORTABLE_CLUSTER_PREFLIGHT_RESULT_SCHEMA = "production_stage1_cluster_preflight_result_v2"
PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_CONTENT_SCHEMA = (
    "production_stage1_cluster_preflight_scientific_content_v1"
)
PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME = "cluster_preflight_manifest.json"
PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME = "cluster_feasibility_audit_index.json"
PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME = "stage1_preflight_scientific_request.json"
PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY = "owner_concepts"
SUPPORTED_PORTABLE_CLUSTER_PREFLIGHT_PARQUET_COMPRESSIONS = frozenset(
    {
        "none",
        "zstd",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_READ_ONLY_FILE_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
_READ_ONLY_DIRECTORY_MODE = (
    stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
)
_CONCEPT_FIELDS = (
    "raw_cluster_concepts",
    "semantic_cluster_concepts",
    "final_catalog_concepts",
)
_CONCEPT_HASH_FIELDS = {
    "raw_cluster_concepts": "raw_cluster_concepts_sha256",
    "semantic_cluster_concepts": "semantic_cluster_concepts_sha256",
    "final_catalog_concepts": "final_catalog_concepts_sha256",
}
_VIEW_FOR_FIELD = {
    "raw_cluster_concepts": "raw",
    "semantic_cluster_concepts": "semantic",
    "final_catalog_concepts": "final",
}
_FIELD_FOR_VIEW = {value: key for key, value in _VIEW_FOR_FIELD.items()}
_PARQUET_COLUMNS = (
    "view",
    "family",
    "concept_index",
    "payload_json",
    "payload_sha256",
)
_RESULT_FIELDS = {
    "schema_version",
    "artifact_version",
    "artifact_code_sha256",
    "root",
    "manifest_path",
    "audit_path",
    "stage1_request_path",
    "manifest_sha256",
    "audit_sha256",
    "stage1_request_file_sha256",
    "stage1_request_sha256",
    "cluster_audit_content_sha256",
    "portable_audit_reference_content_sha256",
    "compact_audit_index_content_sha256",
    "payload_inventory_content_sha256",
    "physical_storage",
    "scope_count",
    "physical_fit_count",
    "scope_order",
    "physical_scope_order",
    "scope_fit_identity_sha256",
    "path_neutral_scientific_content_sha256",
    "content_sha256",
}


def _canonical_json(value: Any) -> str:
    result = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        result.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("portable preflight values must contain valid UTF-8") from exc
    return result


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_json_streaming(value: Any) -> str:
    """Hash canonical JSON without constructing a cohort-sized string."""

    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256()
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"portable preflight JSON repeats key: {key}")
        output[key] = value
    return output


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        payload = (_canonical_json(dict(value)) + "\n").encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _FileSnapshot:
    sha256: str
    size_bytes: int
    stat_identity: tuple[int, int, int, int, int, int, int]


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _stable_file(
    path: Path,
    *,
    label: str,
    require_read_only: bool,
) -> _FileSnapshot:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise FileNotFoundError(f"{label} is absent: {path}") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ValueError(f"{label} must be one non-linked regular file")
    if require_read_only and stat.S_IMODE(before.st_mode) != _READ_ONLY_FILE_MODE:
        raise ValueError(f"{label} must be read-only")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            after_fd = os.fstat(handle.fileno())
    finally:
        os.close(descriptor)
    after_path = os.lstat(path)
    if _stat_identity(before) != _stat_identity(after_fd) or _stat_identity(
        before
    ) != _stat_identity(after_path):
        raise RuntimeError(f"{label} changed during authentication")
    return _FileSnapshot(
        sha256=digest.hexdigest(),
        size_bytes=int(before.st_size),
        stat_identity=_stat_identity(before),
    )


def _require_unchanged_stat(
    path: Path,
    *,
    snapshot: _FileSnapshot,
    label: str,
) -> None:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise FileNotFoundError(f"{label} is absent after authentication") from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or _stat_identity(observed) != snapshot.stat_identity
    ):
        raise ValueError(f"{label} changed after authentication")


def _artifact_code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _collection_rows(
    fit_identity: Mapping[str, Any],
    *,
    verify_declared_collection_hashes: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, Any]] = {}
    for field_name in _CONCEPT_FIELDS:
        value = fit_identity.get(field_name)
        declared = _require_sha256(
            fit_identity.get(_CONCEPT_HASH_FIELDS[field_name]),
            label=f"{field_name} source identity",
        )
        if verify_declared_collection_hashes and declared != _sha256_json_streaming(value):
            raise ValueError(f"{field_name} differs from its source identity")
        start = len(rows)
        view = _VIEW_FOR_FIELD[field_name]
        if field_name == "final_catalog_concepts":
            if not isinstance(value, Mapping) or not value:
                raise ValueError("final catalog concepts must be one nonempty mapping")
            collection_count = 0
            family_counts: dict[str, int] = {}
            for family in sorted(value):
                concepts = value[family]
                if not isinstance(family, str) or not family:
                    raise ValueError("final catalog concept family is invalid")
                if not isinstance(concepts, list) or not concepts:
                    raise ValueError("final catalog concept family must be nonempty")
                family_counts[family] = len(concepts)
                for concept_index, concept in enumerate(concepts):
                    payload = _canonical_json(concept)
                    rows.append(
                        {
                            "view": view,
                            "family": family,
                            "concept_index": concept_index,
                            "payload_json": payload,
                            "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                        }
                    )
                    collection_count += 1
        else:
            if not isinstance(value, list) or not value:
                raise ValueError(f"{field_name} must be one nonempty list")
            family_counts = {}
            collection_count = len(value)
            for concept_index, concept in enumerate(value):
                if not isinstance(concept, Mapping):
                    raise ValueError(f"{field_name} contains a non-mapping concept")
                family = str(concept.get("contrast_family") or "")
                if not family:
                    raise ValueError(f"{field_name} concept lacks its family")
                family_counts[family] = family_counts.get(family, 0) + 1
                payload = _canonical_json(concept)
                rows.append(
                    {
                        "view": view,
                        "family": family,
                        "concept_index": concept_index,
                        "payload_json": payload,
                        "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    }
                )
        bindings[field_name] = {
            "view": view,
            "row_start": start,
            "row_count": collection_count,
            "family_counts": family_counts,
            "source_collection_sha256": declared,
        }
    scientific_rows = [
        {
            "view": row["view"],
            "family": row["family"],
            "concept_index": row["concept_index"],
            "payload_sha256": row["payload_sha256"],
        }
        for row in rows
    ]
    return rows, bindings, _sha256_json(scientific_rows)


def _compact_fit_record(
    fit_identity: Mapping[str, Any],
    *,
    owner_scope_id: str,
    verify_source_content: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(fit_identity, Mapping):
        raise ValueError("physical clustered preflight lacks its fit identity")
    source_content = _require_sha256(
        fit_identity.get("content_sha256"),
        label=f"{owner_scope_id} fit identity",
    )
    source_body = {key: value for key, value in fit_identity.items() if key != "content_sha256"}
    if verify_source_content and source_content != _sha256_json_streaming(source_body):
        raise ValueError("cluster fit identity differs from its source root")
    rows, collections, payload_content = _collection_rows(
        fit_identity,
        verify_declared_collection_hashes=verify_source_content,
    )
    compact_state = {
        key: copy.deepcopy(value)
        for key, value in fit_identity.items()
        if key
        not in {
            "content_sha256",
            *_CONCEPT_FIELDS,
            *_CONCEPT_HASH_FIELDS.values(),
        }
    }
    body = {
        "physical_owner_scope_id": owner_scope_id,
        "source_fit_identity_content_sha256": source_content,
        "compact_state": compact_state,
        "concept_collections": collections,
        "concept_payload_content_sha256": payload_content,
        "concept_payload_row_count": len(rows),
    }
    return {**body, "content_sha256": _sha256_json(body)}, rows


def _build_compact_index(
    audit: Mapping[str, Any],
    *,
    retain_payload_rows: bool = False,
    verify_source_audit_content: bool = True,
    verify_source_fit_content: bool = True,
    payload_row_consumer: (
        Callable[
            [int, str, Sequence[Mapping[str, Any]], Mapping[str, Any]],
            None,
        ]
        | None
    ) = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if not isinstance(audit, Mapping):
        raise ValueError("portable preflight requires one audit mapping")
    source_audit_content = _require_sha256(
        audit.get("content_sha256"),
        label="source clustered audit",
    )
    source_body = {key: value for key, value in audit.items() if key != "content_sha256"}
    if verify_source_audit_content and source_audit_content != _sha256_json_streaming(source_body):
        raise ValueError("source clustered audit differs from its content root")
    scopes = audit.get("scopes")
    scope_order = audit.get("scope_order")
    physical_order = audit.get("physical_scope_order")
    if (
        not isinstance(scopes, list)
        or not isinstance(scope_order, list)
        or [row.get("scope_id") for row in scopes if isinstance(row, Mapping)] != scope_order
        or len(scope_order) != len(set(scope_order))
        or not isinstance(physical_order, list)
        or not physical_order
        or len(physical_order) != len(set(physical_order))
        or not set(physical_order).issubset(set(scope_order))
    ):
        raise ValueError("source clustered audit has an invalid scope inventory")
    scopes_by_id = {str(row["scope_id"]): row for row in scopes}
    owner_by_scope: dict[str, str] = {}
    logical_members: dict[str, list[str]] = {str(owner): [] for owner in physical_order}
    for scope_id in scope_order:
        scope = scopes_by_id[str(scope_id)]
        binding = scope.get("physical_fit_binding")
        if not isinstance(binding, Mapping):
            raise ValueError("portable clustered scope lacks a physical binding")
        owner = str(binding.get("physical_owner_scope_id") or "")
        if owner not in logical_members or binding.get("logical_scope_id") != scope_id:
            raise ValueError("clustered logical-to-physical binding is invalid")
        owner_by_scope[str(scope_id)] = owner
        logical_members[owner].append(str(scope_id))
    if any(not members or members[0] != owner for owner, members in logical_members.items()):
        raise ValueError("physical owner must be the earliest canonical group member")

    physical_records: list[dict[str, Any]] = []
    payload_rows: dict[str, list[dict[str, Any]]] = {}
    compact_by_owner: dict[str, Mapping[str, Any]] = {}
    for canonical_index, owner in enumerate(physical_order):
        source_identity = scopes_by_id[str(owner)].get("cluster_fit_identity")
        compact_fit, rows = _compact_fit_record(
            source_identity,
            owner_scope_id=str(owner),
            verify_source_content=verify_source_fit_content,
        )
        compact_by_owner[str(owner)] = compact_fit
        if retain_payload_rows:
            payload_rows[str(owner)] = rows
        if payload_row_consumer is not None:
            payload_row_consumer(
                canonical_index,
                str(owner),
                rows,
                compact_fit,
            )
        physical_records.append(
            {
                "canonical_physical_index": canonical_index,
                "physical_owner_scope_id": str(owner),
                "logical_member_scope_ids": logical_members[str(owner)],
                "logical_member_count": len(logical_members[str(owner)]),
                "payload_relative_path": (
                    f"{PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY}/"
                    f"{canonical_index:03d}.parquet"
                ),
                "compact_fit_identity": compact_fit,
            }
        )

    logical_records: list[dict[str, Any]] = []
    for canonical_index, scope_id in enumerate(scope_order):
        source = scopes_by_id[str(scope_id)]
        owner = owner_by_scope[str(scope_id)]
        source_identity = source.get("cluster_fit_identity")
        owner_identity = scopes_by_id[owner].get("cluster_fit_identity")
        if source_identity != owner_identity:
            raise ValueError("logical alias copied or changed its canonical fit identity")
        source_scope_sha = _sha256_json_streaming(source)
        scope_without_identity = {
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key != "cluster_fit_identity"
        }
        binding_body = {
            "physical_owner_scope_id": owner,
            "source_fit_identity_content_sha256": compact_by_owner[owner][
                "source_fit_identity_content_sha256"
            ],
            "physical_fit_record_content_sha256": compact_by_owner[owner]["content_sha256"],
            "reuses_physical_payload": str(scope_id) != owner,
        }
        logical_records.append(
            {
                "canonical_index": canonical_index,
                "scope_id": str(scope_id),
                "scope_without_fit_identity": scope_without_identity,
                "source_scope_record_sha256": source_scope_sha,
                "physical_fit_reference": {
                    **binding_body,
                    "content_sha256": _sha256_json(binding_body),
                },
            }
        )

    audit_header = {
        key: copy.deepcopy(value)
        for key, value in audit.items()
        if key not in {"scopes", "content_sha256"}
    }
    lossless_inventory = [
        {
            "physical_owner_scope_id": row["physical_owner_scope_id"],
            "source_fit_identity_content_sha256": row["compact_fit_identity"][
                "source_fit_identity_content_sha256"
            ],
            "concept_payload_content_sha256": row["compact_fit_identity"][
                "concept_payload_content_sha256"
            ],
        }
        for row in physical_records
    ]
    body = {
        "schema_version": PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_SCHEMA,
        "source_audit_schema_version": audit.get("schema_version"),
        "source_audit_content_sha256": source_audit_content,
        "audit_header": audit_header,
        "logical_scope_count": len(scope_order),
        "physical_fit_count": len(physical_order),
        "deduplicated_logical_scope_count": len(scope_order) - len(physical_order),
        "scope_order": list(scope_order),
        "physical_scope_order": list(physical_order),
        "logical_scopes": logical_records,
        "physical_fits": physical_records,
        "lossless_concept_inventory_sha256": _sha256_json(lossless_inventory),
        "logical_alias_concept_payloads_published": False,
        "concept_serialization": ("ordered_parquet_canonical_json_payload_rows_v1"),
        "numerical_state_serialization": ("separate_authenticated_individual_npy_state_bundle_v1"),
    }
    return {**body, "content_sha256": _sha256_json(body)}, payload_rows


def validate_portable_cluster_preflight_reference(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "artifact_version",
        "source_audit_schema_version",
        "source_audit_content_sha256",
        "audit_index_content_sha256",
        "lossless_concept_inventory_sha256",
        "logical_scope_count",
        "physical_fit_count",
        "deduplicated_logical_scope_count",
        "scope_order_sha256",
        "physical_scope_order_sha256",
        "logical_alias_concept_payloads_published",
        "concept_serialization",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("portable clustered-preflight reference has an invalid schema")
    body = {key: copy.deepcopy(child) for key, child in value.items() if key != "content_sha256"}
    if (
        value.get("schema_version") != PORTABLE_CLUSTER_PREFLIGHT_REFERENCE_SCHEMA
        or value.get("artifact_version") != PORTABLE_CLUSTER_PREFLIGHT_ARTIFACT_VERSION
        or value.get("logical_alias_concept_payloads_published") is not False
        or value.get("concept_serialization") != "ordered_parquet_canonical_json_payload_rows_v1"
        or value.get("content_sha256") != _sha256_json(body)
    ):
        raise ValueError("portable clustered-preflight reference is invalid")
    for key in (
        "source_audit_content_sha256",
        "audit_index_content_sha256",
        "lossless_concept_inventory_sha256",
        "scope_order_sha256",
        "physical_scope_order_sha256",
    ):
        _require_sha256(value.get(key), label=f"preflight reference {key}")
    counts = (
        value.get("logical_scope_count"),
        value.get("physical_fit_count"),
        value.get("deduplicated_logical_scope_count"),
    )
    if (
        any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts)
        or counts[0] < 1
        or counts[1] < 1
        or counts[1] > counts[0]
        or counts[2] != counts[0] - counts[1]
    ):
        raise ValueError("portable clustered-preflight reference counts are invalid")
    return copy.deepcopy(dict(value))


def build_portable_cluster_preflight_reference(
    audit: Mapping[str, Any],
    *,
    verify_source_audit_content: bool = True,
) -> dict[str, Any]:
    index, _payload_rows = _build_compact_index(
        audit,
        verify_source_audit_content=verify_source_audit_content,
        verify_source_fit_content=verify_source_audit_content,
    )
    body = {
        "schema_version": PORTABLE_CLUSTER_PREFLIGHT_REFERENCE_SCHEMA,
        "artifact_version": PORTABLE_CLUSTER_PREFLIGHT_ARTIFACT_VERSION,
        "source_audit_schema_version": index["source_audit_schema_version"],
        "source_audit_content_sha256": index["source_audit_content_sha256"],
        "audit_index_content_sha256": index["content_sha256"],
        "lossless_concept_inventory_sha256": index["lossless_concept_inventory_sha256"],
        "logical_scope_count": index["logical_scope_count"],
        "physical_fit_count": index["physical_fit_count"],
        "deduplicated_logical_scope_count": index["deduplicated_logical_scope_count"],
        "scope_order_sha256": _sha256_json(index["scope_order"]),
        "physical_scope_order_sha256": _sha256_json(index["physical_scope_order"]),
        "logical_alias_concept_payloads_published": False,
        "concept_serialization": index["concept_serialization"],
    }
    return validate_portable_cluster_preflight_reference(
        {**body, "content_sha256": _sha256_json(body)}
    )


def _reference_from_index(index: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": PORTABLE_CLUSTER_PREFLIGHT_REFERENCE_SCHEMA,
        "artifact_version": PORTABLE_CLUSTER_PREFLIGHT_ARTIFACT_VERSION,
        "source_audit_schema_version": index["source_audit_schema_version"],
        "source_audit_content_sha256": index["source_audit_content_sha256"],
        "audit_index_content_sha256": index["content_sha256"],
        "lossless_concept_inventory_sha256": index["lossless_concept_inventory_sha256"],
        "logical_scope_count": index["logical_scope_count"],
        "physical_fit_count": index["physical_fit_count"],
        "deduplicated_logical_scope_count": index["deduplicated_logical_scope_count"],
        "scope_order_sha256": _sha256_json(index["scope_order"]),
        "physical_scope_order_sha256": _sha256_json(index["physical_scope_order"]),
        "logical_alias_concept_payloads_published": False,
        "concept_serialization": index["concept_serialization"],
    }
    return validate_portable_cluster_preflight_reference(
        {**body, "content_sha256": _sha256_json(body)}
    )


def _validate_compact_fit_record(
    value: Any,
    *,
    expected_owner: str,
) -> dict[str, Any]:
    required = {
        "physical_owner_scope_id",
        "source_fit_identity_content_sha256",
        "compact_state",
        "concept_collections",
        "concept_payload_content_sha256",
        "concept_payload_row_count",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("portable physical fit record has an invalid schema")
    body = {key: copy.deepcopy(child) for key, child in value.items() if key != "content_sha256"}
    collections = value.get("concept_collections")
    if (
        value.get("physical_owner_scope_id") != expected_owner
        or value.get("content_sha256") != _sha256_json(body)
        or not isinstance(value.get("compact_state"), Mapping)
        or not isinstance(collections, Mapping)
        or set(collections) != set(_CONCEPT_FIELDS)
        or isinstance(value.get("concept_payload_row_count"), bool)
        or not isinstance(value.get("concept_payload_row_count"), int)
        or value.get("concept_payload_row_count") < 3
    ):
        raise ValueError("portable physical fit record is invalid")
    _require_sha256(
        value.get("source_fit_identity_content_sha256"),
        label="source fit identity",
    )
    _require_sha256(
        value.get("concept_payload_content_sha256"),
        label="concept payload identity",
    )
    expected_start = 0
    for field_name in _CONCEPT_FIELDS:
        binding = collections[field_name]
        required_binding = {
            "view",
            "row_start",
            "row_count",
            "family_counts",
            "source_collection_sha256",
        }
        if (
            not isinstance(binding, Mapping)
            or set(binding) != required_binding
            or binding.get("view") != _VIEW_FOR_FIELD[field_name]
            or binding.get("row_start") != expected_start
            or isinstance(binding.get("row_count"), bool)
            or not isinstance(binding.get("row_count"), int)
            or binding.get("row_count") < 1
            or not isinstance(binding.get("family_counts"), Mapping)
            or not binding["family_counts"]
            or any(
                not isinstance(family, str)
                or not family
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                for family, count in binding["family_counts"].items()
            )
            or sum(binding["family_counts"].values()) != binding["row_count"]
        ):
            raise ValueError("portable concept collection binding is invalid")
        _require_sha256(
            binding.get("source_collection_sha256"),
            label=f"{field_name} collection identity",
        )
        expected_start += int(binding["row_count"])
    if expected_start != value["concept_payload_row_count"]:
        raise ValueError("portable concept row ranges are incomplete")
    return copy.deepcopy(dict(value))


def _validate_compact_index(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "source_audit_schema_version",
        "source_audit_content_sha256",
        "audit_header",
        "logical_scope_count",
        "physical_fit_count",
        "deduplicated_logical_scope_count",
        "scope_order",
        "physical_scope_order",
        "logical_scopes",
        "physical_fits",
        "lossless_concept_inventory_sha256",
        "logical_alias_concept_payloads_published",
        "concept_serialization",
        "numerical_state_serialization",
        "content_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("portable clustered-preflight audit index has an invalid schema")
    body = {key: copy.deepcopy(child) for key, child in value.items() if key != "content_sha256"}
    scope_order = value.get("scope_order")
    physical_order = value.get("physical_scope_order")
    logical = value.get("logical_scopes")
    physical = value.get("physical_fits")
    if (
        value.get("schema_version") != PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_SCHEMA
        or value.get("content_sha256") != _sha256_json(body)
        or not isinstance(value.get("audit_header"), Mapping)
        or not isinstance(scope_order, list)
        or not isinstance(physical_order, list)
        or not isinstance(logical, list)
        or not isinstance(physical, list)
        or len(scope_order) != len(set(scope_order))
        or len(physical_order) != len(set(physical_order))
        or value.get("logical_scope_count") != len(scope_order)
        or value.get("physical_fit_count") != len(physical_order)
        or value.get("deduplicated_logical_scope_count") != len(scope_order) - len(physical_order)
        or len(logical) != len(scope_order)
        or len(physical) != len(physical_order)
        or value.get("logical_alias_concept_payloads_published") is not False
        or value.get("concept_serialization") != "ordered_parquet_canonical_json_payload_rows_v1"
        or value.get("numerical_state_serialization")
        != "separate_authenticated_individual_npy_state_bundle_v1"
    ):
        raise ValueError("portable clustered-preflight audit index is invalid")
    _require_sha256(
        value.get("source_audit_content_sha256"),
        label="source clustered audit",
    )
    _require_sha256(
        value.get("lossless_concept_inventory_sha256"),
        label="lossless concept inventory",
    )

    compact_by_owner: dict[str, dict[str, Any]] = {}
    member_by_scope: dict[str, str] = {}
    lossless_inventory = []
    for position, (expected_owner, row) in enumerate(zip(physical_order, physical, strict=True)):
        required_row = {
            "canonical_physical_index",
            "physical_owner_scope_id",
            "logical_member_scope_ids",
            "logical_member_count",
            "payload_relative_path",
            "compact_fit_identity",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != required_row
            or row.get("canonical_physical_index") != position
            or row.get("physical_owner_scope_id") != expected_owner
            or row.get("payload_relative_path")
            != (f"{PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY}/" f"{position:03d}.parquet")
            or not isinstance(row.get("logical_member_scope_ids"), list)
            or not row["logical_member_scope_ids"]
            or row["logical_member_scope_ids"][0] != expected_owner
            or row.get("logical_member_count") != len(row["logical_member_scope_ids"])
        ):
            raise ValueError("portable physical fit inventory is invalid")
        compact = _validate_compact_fit_record(
            row["compact_fit_identity"],
            expected_owner=str(expected_owner),
        )
        compact_by_owner[str(expected_owner)] = compact
        for scope_id in row["logical_member_scope_ids"]:
            if scope_id in member_by_scope:
                raise ValueError("portable logical scope belongs to two owners")
            member_by_scope[str(scope_id)] = str(expected_owner)
        lossless_inventory.append(
            {
                "physical_owner_scope_id": str(expected_owner),
                "source_fit_identity_content_sha256": compact["source_fit_identity_content_sha256"],
                "concept_payload_content_sha256": compact["concept_payload_content_sha256"],
            }
        )
    if set(member_by_scope) != set(scope_order) or value[
        "lossless_concept_inventory_sha256"
    ] != _sha256_json(lossless_inventory):
        raise ValueError("portable physical fit coverage is incomplete")

    for position, (expected_scope, row) in enumerate(zip(scope_order, logical, strict=True)):
        required_row = {
            "canonical_index",
            "scope_id",
            "scope_without_fit_identity",
            "source_scope_record_sha256",
            "physical_fit_reference",
        }
        reference = row.get("physical_fit_reference") if isinstance(row, Mapping) else None
        owner = member_by_scope.get(str(expected_scope))
        compact = compact_by_owner.get(str(owner))
        if (
            not isinstance(row, Mapping)
            or set(row) != required_row
            or row.get("canonical_index") != position
            or row.get("scope_id") != expected_scope
            or not isinstance(row.get("scope_without_fit_identity"), Mapping)
            or row["scope_without_fit_identity"].get("scope_id") != expected_scope
            or not isinstance(reference, Mapping)
            or set(reference)
            != {
                "physical_owner_scope_id",
                "source_fit_identity_content_sha256",
                "physical_fit_record_content_sha256",
                "reuses_physical_payload",
                "content_sha256",
            }
            or compact is None
        ):
            raise ValueError("portable logical scope inventory is invalid")
        reference_body = {
            key: copy.deepcopy(child) for key, child in reference.items() if key != "content_sha256"
        }
        if (
            reference.get("physical_owner_scope_id") != owner
            or reference.get("source_fit_identity_content_sha256")
            != compact["source_fit_identity_content_sha256"]
            or reference.get("physical_fit_record_content_sha256") != compact["content_sha256"]
            or reference.get("reuses_physical_payload") is not (str(expected_scope) != str(owner))
            or reference.get("content_sha256") != _sha256_json(reference_body)
        ):
            raise ValueError("portable logical scope reference is invalid")
        _require_sha256(
            row.get("source_scope_record_sha256"),
            label=f"{expected_scope} source scope record",
        )
    return copy.deepcopy(dict(value))


def _validate_scientific_audit(
    audit: Mapping[str, Any],
    *,
    config: Any,
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    embedding_cache_identity: Mapping[str, Any],
    initial_training_partitions: int,
) -> dict[str, Any]:
    from .production_stage1_bundle import (
        validate_embedding_cluster_feasibility_audit,
    )

    validated = validate_embedding_cluster_feasibility_audit(
        audit,
        config=config,
        registry=registry,
        registry_content_sha256=registry_content_sha256,
        embedding_cache_identity=embedding_cache_identity,
        initial_training_partitions=initial_training_partitions,
        copy_result=False,
        verify_aggregate_content_hash=False,
    )
    if validated is not audit and validated != audit:
        raise RuntimeError("portable preflight validator changed accepted audit bytes")
    return dict(validated)


def _validate_stage1_request_with_reference(
    request: Mapping[str, Any],
    *,
    expected_reference: Mapping[str, Any],
) -> dict[str, Any]:
    from .production_stage1_bundle import STAGE1_BUNDLE_REQUEST_SCHEMA
    from .production_stage1_hierarchy_contract import (
        validate_production_stage1_hierarchy_request_bindings,
    )

    if not isinstance(request, Mapping):
        raise ValueError("portable preflight Stage 1 request must be one mapping")
    output = copy.deepcopy(dict(request))
    request_sha = _require_sha256(
        output.get("request_sha256"),
        label="portable preflight request",
    )
    body = {key: child for key, child in output.items() if key != "request_sha256"}
    supplied_reference = output.get("embedding_cluster_feasibility_audit")
    if (
        output.get("schema_version") != STAGE1_BUNDLE_REQUEST_SCHEMA
        or request_sha != _sha256_json(body)
        or not isinstance(supplied_reference, Mapping)
        or validate_portable_cluster_preflight_reference(supplied_reference)
        != dict(expected_reference)
    ):
        raise ValueError("portable Stage 1 request is not exactly bound to its audit reference")
    validate_production_stage1_hierarchy_request_bindings(output)
    return output


def _scientific_request_projection(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    # Reuse the existing closed path-neutral projection.  Its clustered field
    # deliberately carries only ``content_sha256`` and therefore works for the
    # v2 reference without embedding any concepts or locators.
    from .production_stage1_cluster_preflight_artifact import (
        stage1_request_scientific_compatibility_projection,
    )

    projection = stage1_request_scientific_compatibility_projection(request)
    body = {
        "schema_version": PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_SCHEMA,
        "stage1_request_scientific_projection": projection,
        "portable_cluster_preflight_reference": copy.deepcopy(
            dict(request["embedding_cluster_feasibility_audit"])
        ),
    }
    return {**body, "content_sha256": _sha256_json(body)}


def _require_parquet_compression(value: Any) -> str:
    if (
        not isinstance(value, str)
        or value not in SUPPORTED_PORTABLE_CLUSTER_PREFLIGHT_PARQUET_COMPRESSIONS
    ):
        supported = ", ".join(sorted(SUPPORTED_PORTABLE_CLUSTER_PREFLIGHT_PARQUET_COMPRESSIONS))
        raise ValueError(
            "portable clustered-preflight parquet_compression must be one "
            f"explicit supported value: {supported}"
        )
    return value


def _physical_storage_metadata(*, parquet_compression: str) -> dict[str, Any]:
    return {
        "owner_concept_payload_format": "parquet",
        "parquet_compression": _require_parquet_compression(parquet_compression),
        "parquet_use_dictionary": False,
        "parquet_write_statistics": False,
        "parquet_data_page_version": "1.0",
    }


def _validate_physical_storage_metadata(value: Any) -> dict[str, Any]:
    required = {
        "owner_concept_payload_format",
        "parquet_compression",
        "parquet_use_dictionary",
        "parquet_write_statistics",
        "parquet_data_page_version",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("owner_concept_payload_format") != "parquet"
        or value.get("parquet_use_dictionary") is not False
        or value.get("parquet_write_statistics") is not False
        or value.get("parquet_data_page_version") != "1.0"
    ):
        raise ValueError("portable clustered-preflight physical storage metadata is invalid")
    compression = _require_parquet_compression(value.get("parquet_compression"))
    return {
        "owner_concept_payload_format": "parquet",
        "parquet_compression": compression,
        "parquet_use_dictionary": False,
        "parquet_write_statistics": False,
        "parquet_data_page_version": "1.0",
    }


def _path_neutral_scientific_content_sha256(
    *,
    index: Mapping[str, Any],
    scientific_request: Mapping[str, Any],
) -> str:
    """Return content identity independent of its physical serialization.

    The ordinary Stage 1 projection binds the portable audit reference.  That
    reference intentionally authenticates the compact-index representation,
    so it is not itself a representation-neutral scientific key.  Replace
    that one binding with the source audit's scientific content root, while
    retaining every other closed Stage 1 scientific compatibility field.
    """

    projection = copy.deepcopy(dict(scientific_request["stage1_request_scientific_projection"]))
    projection.pop("content_sha256", None)
    projection["embedding_cluster_feasibility_audit"] = {
        "source_audit_content_sha256": _require_sha256(
            index.get("source_audit_content_sha256"),
            label="portable preflight source audit scientific content",
        )
    }
    normalized_request_sha256 = _sha256_json(projection)
    body = {
        "schema_version": PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_CONTENT_SCHEMA,
        "source_audit_content_sha256": index["source_audit_content_sha256"],
        "normalized_stage1_request_scientific_content_sha256": (normalized_request_sha256),
    }
    return _sha256_json(body)


def _write_owner_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    parquet_compression: str,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("portable clustered preflight requires pyarrow Parquet support") from exc
    if not rows:
        raise ValueError("owner concept payload cannot be empty")
    table = pa.Table.from_arrays(
        [
            (
                pa.array([str(row[column]) for row in rows], type=pa.large_string())
                if column != "concept_index"
                else pa.array(
                    [int(row[column]) for row in rows],
                    type=pa.int64(),
                )
            )
            for column in _PARQUET_COLUMNS
        ],
        names=list(_PARQUET_COLUMNS),
    )
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"owner concept payload already exists: {path}")
    compression = _require_parquet_compression(parquet_compression)
    pq.write_table(
        table,
        path,
        compression=None if compression == "none" else compression,
        use_dictionary=False,
        write_statistics=False,
        data_page_version="1.0",
    )
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_owner_parquet(
    path: Path,
    *,
    expected_compact_fit: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("portable clustered preflight requires pyarrow Parquet support") from exc
    table = pq.read_table(path)
    expected_schema = pa.schema(
        [
            pa.field("view", pa.large_string(), nullable=True),
            pa.field("family", pa.large_string(), nullable=True),
            pa.field("concept_index", pa.int64(), nullable=True),
            pa.field("payload_json", pa.large_string(), nullable=True),
            pa.field("payload_sha256", pa.large_string(), nullable=True),
        ]
    )
    if table.schema != expected_schema:
        raise ValueError("owner concept Parquet has an invalid closed schema")
    rows = table.to_pylist()
    if len(rows) != expected_compact_fit["concept_payload_row_count"]:
        raise ValueError("owner concept Parquet row count changed")
    scientific_rows: list[dict[str, Any]] = []
    parsed_rows: list[Any] = []
    for row in rows:
        if set(row) != set(_PARQUET_COLUMNS):
            raise ValueError("owner concept Parquet contains an invalid row")
        payload = row["payload_json"]
        if (
            not isinstance(row["view"], str)
            or row["view"] not in _FIELD_FOR_VIEW
            or not isinstance(row["family"], str)
            or not row["family"]
            or isinstance(row["concept_index"], bool)
            or not isinstance(row["concept_index"], int)
            or row["concept_index"] < 0
            or not isinstance(payload, str)
            or hashlib.sha256(payload.encode("utf-8")).hexdigest() != row["payload_sha256"]
        ):
            raise ValueError("owner concept Parquet row is invalid")
        try:
            parsed = json.loads(
                payload,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except json.JSONDecodeError as exc:
            raise ValueError("owner concept payload is not valid JSON") from exc
        if _canonical_json(parsed) != payload:
            raise ValueError("owner concept payload is not canonical JSON")
        scientific_rows.append(
            {
                "view": row["view"],
                "family": row["family"],
                "concept_index": row["concept_index"],
                "payload_sha256": row["payload_sha256"],
            }
        )
        parsed_rows.append(parsed)
    if _sha256_json(scientific_rows) != expected_compact_fit["concept_payload_content_sha256"]:
        raise ValueError("owner concept payload scientific identity changed")

    reconstructed: dict[str, Any] = {}
    bindings = expected_compact_fit["concept_collections"]
    for field_name in _CONCEPT_FIELDS:
        binding = bindings[field_name]
        start = int(binding["row_start"])
        stop = start + int(binding["row_count"])
        selected_meta = scientific_rows[start:stop]
        selected = parsed_rows[start:stop]
        if len(selected) != binding["row_count"] or any(
            row["view"] != binding["view"] for row in selected_meta
        ):
            raise ValueError("owner concept collection range changed")
        if field_name == "final_catalog_concepts":
            collection: dict[str, list[Any]] = {}
            expected_index: dict[str, int] = {}
            for meta, concept in zip(selected_meta, selected, strict=True):
                family = meta["family"]
                index = expected_index.get(family, 0)
                if meta["concept_index"] != index:
                    raise ValueError("final catalog concept order changed")
                expected_index[family] = index + 1
                collection.setdefault(family, []).append(concept)
            if {family: len(concepts) for family, concepts in collection.items()} != dict(
                binding["family_counts"]
            ):
                raise ValueError("final catalog family coverage changed")
        else:
            collection = selected
            if [row["concept_index"] for row in selected_meta] != list(range(len(selected_meta))):
                raise ValueError("owner concept order changed")
            observed_counts: dict[str, int] = {}
            for row in selected_meta:
                observed_counts[row["family"]] = observed_counts.get(row["family"], 0) + 1
            if observed_counts != dict(binding["family_counts"]):
                raise ValueError("owner concept family coverage changed")
        if _sha256_json_streaming(collection) != binding["source_collection_sha256"]:
            raise ValueError("owner concept collection differs from source")
        reconstructed[field_name] = collection

    full_body = copy.deepcopy(dict(expected_compact_fit["compact_state"]))
    for field_name in _CONCEPT_FIELDS:
        full_body[field_name] = reconstructed[field_name]
        full_body[_CONCEPT_HASH_FIELDS[field_name]] = bindings[field_name][
            "source_collection_sha256"
        ]
    if (
        _sha256_json_streaming(full_body)
        != expected_compact_fit["source_fit_identity_content_sha256"]
    ):
        raise ValueError("reconstructed owner fit identity differs from source")
    return {
        **full_body,
        "content_sha256": expected_compact_fit["source_fit_identity_content_sha256"],
    }


def _validate_owner_parquet_storage(
    path: Path,
    *,
    parquet_compression: str,
) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("portable clustered preflight requires pyarrow Parquet support") from exc
    expected = (
        "UNCOMPRESSED"
        if _require_parquet_compression(parquet_compression) == "none"
        else parquet_compression.upper()
    )
    metadata = pq.read_metadata(path)
    if metadata.num_rows < 1 or metadata.num_row_groups < 1:
        raise ValueError("owner concept Parquet physical storage is empty")
    observed = {
        str(metadata.row_group(row_group).column(column).compression).upper()
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    }
    if observed != {expected}:
        raise ValueError(
            "owner concept Parquet physical compression differs from its "
            "authenticated storage metadata"
        )


@dataclass(frozen=True)
class PortableProductionStage1ClusterPreflightArtifact:
    """Freshly authenticated path-neutral preflight with lazy owner payloads."""

    root: Path
    manifest_path: Path
    audit_path: Path
    stage1_request_path: Path
    audit: Mapping[str, Any] = field(repr=False)
    stage1_request: Mapping[str, Any] = field(repr=False)
    reference: Mapping[str, Any] = field(repr=False)
    _identity: Mapping[str, Any] = field(repr=False)
    _payload_snapshots: Mapping[str, _FileSnapshot] = field(repr=False)
    _owner_fit_cache: dict[str, Mapping[str, Any]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _cache_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        index = _validate_compact_index(self.audit)
        reference = validate_portable_cluster_preflight_reference(self.reference)
        if reference != _reference_from_index(index):
            raise ValueError("portable preflight reference differs from its audit index")
        identity = copy.deepcopy(dict(self._identity))
        identity_body = {key: child for key, child in identity.items() if key != "content_sha256"}
        if (
            set(identity) != _RESULT_FIELDS
            or identity.get("schema_version") != PORTABLE_CLUSTER_PREFLIGHT_RESULT_SCHEMA
            or identity.get("artifact_version") != PORTABLE_CLUSTER_PREFLIGHT_ARTIFACT_VERSION
            or identity.get("artifact_code_sha256") != _artifact_code_sha256()
            or identity.get("root") != str(self.root)
            or identity.get("manifest_path") != str(self.manifest_path)
            or identity.get("audit_path") != str(self.audit_path)
            or identity.get("stage1_request_path") != str(self.stage1_request_path)
            or identity.get("portable_audit_reference_content_sha256")
            != reference["content_sha256"]
            or identity.get("compact_audit_index_content_sha256") != index["content_sha256"]
            or _validate_physical_storage_metadata(identity.get("physical_storage"))
            != identity.get("physical_storage")
            or identity.get("content_sha256") != _sha256_json(identity_body)
        ):
            raise ValueError("portable clustered-preflight result identity is invalid")
        object.__setattr__(self, "audit", MappingProxyType(index))
        object.__setattr__(
            self,
            "stage1_request",
            MappingProxyType(copy.deepcopy(dict(self.stage1_request))),
        )
        object.__setattr__(self, "reference", MappingProxyType(reference))
        object.__setattr__(
            self,
            "_identity",
            MappingProxyType(identity),
        )
        object.__setattr__(
            self,
            "_payload_snapshots",
            MappingProxyType(dict(self._payload_snapshots)),
        )

    def identity(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self._identity))

    @property
    def is_portable_v2(self) -> bool:
        return True

    def require_stage1_request(
        self,
        expected_stage1_request: Mapping[str, Any],
    ) -> None:
        validated = _validate_stage1_request_with_reference(
            expected_stage1_request,
            expected_reference=self.reference,
        )
        expected_projection = _scientific_request_projection(validated)
        if expected_projection != dict(self.stage1_request):
            raise ValueError(
                "consumer Stage 1 scientific request differs from sealed "
                "portable clustered preflight"
            )

    def _physical_record(self, owner_scope_id: str) -> Mapping[str, Any]:
        matches = [
            row
            for row in self.audit["physical_fits"]
            if row["physical_owner_scope_id"] == str(owner_scope_id)
        ]
        if len(matches) != 1:
            raise ValueError("portable clustered preflight has no unique physical owner")
        return matches[0]

    def owner_fit_identity(self, owner_scope_id: str) -> dict[str, Any]:
        """Parse and authenticate one physical owner's concepts on demand."""

        owner = str(owner_scope_id)
        with self._cache_lock:
            cached = self._owner_fit_cache.get(owner)
            if cached is not None:
                return copy.deepcopy(dict(cached))
            record = self._physical_record(owner)
            relative = str(record["payload_relative_path"])
            snapshot = self._payload_snapshots.get(relative)
            if snapshot is None:
                raise ValueError("portable owner payload lacks an authenticated byte handle")
            path = self.root / Path(relative)
            _require_unchanged_stat(
                path,
                snapshot=snapshot,
                label=f"owner concept payload {owner}",
            )
            identity = _read_owner_parquet(
                path,
                expected_compact_fit=record["compact_fit_identity"],
            )
            _require_unchanged_stat(
                path,
                snapshot=snapshot,
                label=f"owner concept payload {owner}",
            )
            # Bound resident concept memory to one physical fit.  State-bundle
            # sealing and execution may visit every owner, but they must never
            # recreate the former cohort-wide multi-gigabyte aggregate.
            self._owner_fit_cache.clear()
            self._owner_fit_cache[owner] = copy.deepcopy(identity)
            return copy.deepcopy(identity)

    def logical_scope_record(
        self,
        scope_id: str,
        *,
        include_concepts: bool = True,
    ) -> dict[str, Any]:
        matches = [row for row in self.audit["logical_scopes"] if row["scope_id"] == str(scope_id)]
        if len(matches) != 1:
            raise ValueError("portable clustered preflight has no unique logical scope")
        logical = matches[0]
        output = copy.deepcopy(dict(logical["scope_without_fit_identity"]))
        if not include_concepts:
            output["cluster_fit_reference"] = copy.deepcopy(dict(logical["physical_fit_reference"]))
            return output
        owner = logical["physical_fit_reference"]["physical_owner_scope_id"]
        output["cluster_fit_identity"] = self.owner_fit_identity(owner)
        if _sha256_json_streaming(output) != logical["source_scope_record_sha256"]:
            raise ValueError("reconstructed logical scope differs from source preflight")
        return output

    def source_audit_header(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.audit["audit_header"]))


def _manifest_file_registration(
    *,
    relative_path: str,
    kind: str,
    snapshot: _FileSnapshot,
    scientific_content_sha256: str,
    physical_owner_scope_id: str | None,
) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "kind": kind,
        "size_bytes": snapshot.size_bytes,
        "sha256": snapshot.sha256,
        "scientific_content_sha256": _require_sha256(
            scientific_content_sha256,
            label=f"{relative_path} scientific content",
        ),
        "physical_owner_scope_id": physical_owner_scope_id,
    }


def _manifest_body(
    *,
    index: Mapping[str, Any],
    scientific_request: Mapping[str, Any],
    registrations: Sequence[Mapping[str, Any]],
    physical_storage: Mapping[str, Any],
) -> dict[str, Any]:
    reference = _reference_from_index(index)
    return {
        "schema_version": PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_SCHEMA,
        "status": "complete",
        "artifact_version": PORTABLE_CLUSTER_PREFLIGHT_ARTIFACT_VERSION,
        "artifact_code_sha256": _artifact_code_sha256(),
        "portable_cluster_preflight_reference": reference,
        "audit_index_content_sha256": index["content_sha256"],
        "scientific_request_content_sha256": scientific_request["content_sha256"],
        "logical_scope_count": index["logical_scope_count"],
        "physical_fit_count": index["physical_fit_count"],
        "scope_order": list(index["scope_order"]),
        "physical_scope_order": list(index["physical_scope_order"]),
        "files": [copy.deepcopy(dict(row)) for row in registrations],
        "physical_storage": _validate_physical_storage_metadata(physical_storage),
        "payload_inventory_content_sha256": _sha256_json(
            [
                {
                    "relative_path": row["relative_path"],
                    "kind": row["kind"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                    "scientific_content_sha256": row["scientific_content_sha256"],
                    "physical_owner_scope_id": row["physical_owner_scope_id"],
                }
                for row in registrations
            ]
        ),
        "path_locators_in_scientific_manifest": False,
        "logical_alias_payloads_published": False,
        "hardlinks_allowed": False,
    }


def seal_portable_production_stage1_cluster_preflight_artifact(
    *,
    output_dir: Path | str,
    audit: Mapping[str, Any],
    stage1_request: Mapping[str, Any],
    config: Any,
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    embedding_cache_identity: Mapping[str, Any],
    parquet_compression: str,
) -> PortableProductionStage1ClusterPreflightArtifact:
    """Validate and atomically publish one portable v2 preflight."""

    physical_storage = _physical_storage_metadata(
        parquet_compression=parquet_compression,
    )
    target = Path(output_dir)
    if not target.is_absolute():
        raise ValueError("portable preflight output directory must be absolute")
    if target.exists() or target.is_symlink():
        raise FileExistsError("portable preflight output directory must be fresh")
    parent = target.parent.resolve(strict=True)
    if parent != target.parent:
        raise ValueError("portable preflight output parent must be canonical")
    hierarchy = stage1_request.get("hierarchy_spent_evidence_contract")
    initial_partitions = int(
        (hierarchy.get("initial_spent_partition_count") if isinstance(hierarchy, Mapping) else 0)
        or 0
    )
    validated_audit = _validate_scientific_audit(
        audit,
        config=config,
        registry=registry,
        registry_content_sha256=registry_content_sha256,
        embedding_cache_identity=embedding_cache_identity,
        initial_training_partitions=initial_partitions,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    validated_request: dict[str, Any]
    try:
        payload_root = temporary / PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY
        payload_root.mkdir(exist_ok=False)
        payload_registrations: dict[str, dict[str, Any]] = {}

        def publish_owner_payload(
            canonical_index: int,
            owner: str,
            rows: Sequence[Mapping[str, Any]],
            compact_fit: Mapping[str, Any],
        ) -> None:
            relative = (
                f"{PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY}/" f"{canonical_index:03d}.parquet"
            )
            path = temporary / Path(relative)
            _write_owner_parquet(
                path,
                rows,
                parquet_compression=parquet_compression,
            )
            snapshot = _stable_file(
                path,
                label=f"written owner concept payload {owner}",
                require_read_only=False,
            )
            payload_registrations[owner] = _manifest_file_registration(
                relative_path=relative,
                kind="physical_owner_concepts",
                snapshot=snapshot,
                scientific_content_sha256=compact_fit["concept_payload_content_sha256"],
                physical_owner_scope_id=owner,
            )

        index, _discarded_payload_rows = _build_compact_index(
            validated_audit,
            verify_source_audit_content=False,
            verify_source_fit_content=False,
            payload_row_consumer=publish_owner_payload,
        )
        reference = _reference_from_index(index)
        validated_request = _validate_stage1_request_with_reference(
            stage1_request,
            expected_reference=reference,
        )
        scientific_request = _scientific_request_projection(validated_request)
        if set(payload_registrations) != set(index["physical_scope_order"]):
            raise RuntimeError("portable preflight omitted a physical-owner concept payload")
        registrations: list[dict[str, Any]] = []

        index_path = temporary / PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME
        _write_json_new(index_path, index)
        index_snapshot = _stable_file(
            index_path,
            label="written portable audit index",
            require_read_only=False,
        )
        registrations.append(
            _manifest_file_registration(
                relative_path=PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME,
                kind="audit_index",
                snapshot=index_snapshot,
                scientific_content_sha256=index["content_sha256"],
                physical_owner_scope_id=None,
            )
        )

        request_path = temporary / PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME
        _write_json_new(request_path, scientific_request)
        request_snapshot = _stable_file(
            request_path,
            label="written portable scientific request",
            require_read_only=False,
        )
        registrations.append(
            _manifest_file_registration(
                relative_path=(PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME),
                kind="scientific_request",
                snapshot=request_snapshot,
                scientific_content_sha256=scientific_request["content_sha256"],
                physical_owner_scope_id=None,
            )
        )

        registrations.extend(
            payload_registrations[owner] for owner in index["physical_scope_order"]
        )

        manifest_body = _manifest_body(
            index=index,
            scientific_request=scientific_request,
            registrations=registrations,
            physical_storage=physical_storage,
        )
        manifest = {
            **manifest_body,
            "content_sha256": _sha256_json(manifest_body),
        }
        manifest_path = temporary / PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME
        _write_json_new(manifest_path, manifest)

        for path in temporary.rglob("*"):
            if path.is_file():
                path.chmod(_READ_ONLY_FILE_MODE)
        payload_root.chmod(_READ_ONLY_DIRECTORY_MODE)
        temporary.chmod(_READ_ONLY_DIRECTORY_MODE)
        # Synchronize complete directory inventories once, bottom-up, before
        # the final atomic publication.
        _fsync_directory(payload_root)
        _fsync_directory(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError("portable preflight target was populated during publication")
        os.rename(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.chmod(stat.S_IRWXU)
            for path in temporary.rglob("*"):
                if path.is_dir():
                    path.chmod(stat.S_IRWXU)
                else:
                    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_portable_production_stage1_cluster_preflight_artifact(
        manifest_path=target / PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME,
        config=config,
        registry=registry,
        registry_content_sha256=registry_content_sha256,
        embedding_cache_identity=embedding_cache_identity,
        expected_stage1_request=validated_request,
    )


def _safe_registered_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("portable preflight registration lacks a relative path")
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or candidate.as_posix() != relative
    ):
        raise ValueError("portable preflight registration path is unsafe")
    path = root / candidate
    if path.is_symlink():
        raise ValueError("portable preflight registration is symlinked")
    return path


def load_path_only_portable_production_stage1_cluster_preflight_artifact(
    *,
    manifest_path: Path | str,
    expected_stage1_request: Mapping[str, Any] | None = None,
) -> PortableProductionStage1ClusterPreflightArtifact:
    """Authenticate every registered byte and return lazy owner capabilities.

    All dependency identities needed to validate a sealed v2 artifact are
    carried by its compact index and scientific-request projection.  This
    path-only entry point is therefore suitable for a fresh trust boundary
    such as a serialization benchmark.  The compatibility wrapper below
    retains the older dependency-shaped call surface for ordinary workflow
    consumers.
    """

    supplied = Path(manifest_path)
    if not supplied.is_absolute() or supplied.name != PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME:
        raise ValueError("portable preflight manifest path must be absolute and canonical")
    root = supplied.parent
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
        or stat.S_IMODE(os.lstat(root).st_mode) != _READ_ONLY_DIRECTORY_MODE
    ):
        raise ValueError("portable preflight root is invalid or writable")
    manifest_snapshot = _stable_file(
        supplied,
        label="portable preflight manifest",
        require_read_only=True,
    )
    manifest = _read_json(supplied, label="portable preflight manifest")
    _require_unchanged_stat(
        supplied,
        snapshot=manifest_snapshot,
        label="portable preflight manifest",
    )
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    required_manifest = {
        "schema_version",
        "status",
        "artifact_version",
        "artifact_code_sha256",
        "portable_cluster_preflight_reference",
        "audit_index_content_sha256",
        "scientific_request_content_sha256",
        "logical_scope_count",
        "physical_fit_count",
        "scope_order",
        "physical_scope_order",
        "files",
        "physical_storage",
        "payload_inventory_content_sha256",
        "path_locators_in_scientific_manifest",
        "logical_alias_payloads_published",
        "hardlinks_allowed",
        "content_sha256",
    }
    files = manifest.get("files")
    if (
        set(manifest) != required_manifest
        or manifest.get("schema_version") != PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("artifact_version") != PORTABLE_CLUSTER_PREFLIGHT_ARTIFACT_VERSION
        or manifest.get("artifact_code_sha256") != _artifact_code_sha256()
        or manifest.get("path_locators_in_scientific_manifest") is not False
        or manifest.get("logical_alias_payloads_published") is not False
        or manifest.get("hardlinks_allowed") is not False
        or manifest.get("content_sha256") != _sha256_json(body)
        or not isinstance(files, list)
        or len(files) < 3
    ):
        raise ValueError("portable preflight manifest has an invalid closed envelope")
    reference = validate_portable_cluster_preflight_reference(
        manifest["portable_cluster_preflight_reference"]
    )
    physical_storage = _validate_physical_storage_metadata(manifest["physical_storage"])

    required_registration = {
        "relative_path",
        "kind",
        "size_bytes",
        "sha256",
        "scientific_content_sha256",
        "physical_owner_scope_id",
    }
    registrations_by_path: dict[str, Mapping[str, Any]] = {}
    snapshots_by_path: dict[str, _FileSnapshot] = {}
    inode_keys: set[tuple[int, int]] = set()
    for row in files:
        if not isinstance(row, Mapping) or set(row) != required_registration:
            raise ValueError("portable preflight file registration is malformed")
        relative = str(row.get("relative_path") or "")
        if relative in registrations_by_path:
            raise ValueError("portable preflight file inventory repeats a path")
        path = _safe_registered_path(root, relative)
        snapshot = _stable_file(
            path,
            label=f"portable preflight payload {relative}",
            require_read_only=True,
        )
        if row.get("size_bytes") != snapshot.size_bytes or row.get("sha256") != snapshot.sha256:
            raise ValueError("portable preflight registered bytes changed")
        _require_sha256(
            row.get("scientific_content_sha256"),
            label=f"{relative} scientific content",
        )
        inode_key = (
            snapshot.stat_identity[0],
            snapshot.stat_identity[1],
        )
        if inode_key in inode_keys:
            raise ValueError("portable preflight files may not be hard-linked")
        inode_keys.add(inode_key)
        registrations_by_path[relative] = row
        snapshots_by_path[relative] = snapshot

    expected_tree_files = {
        PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME,
        *registrations_by_path,
    }
    observed_tree_files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("portable preflight tree contains a symlink")
        if path.is_file():
            observed_tree_files.add(relative)
        elif path.is_dir():
            if (
                relative != PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY
                or stat.S_IMODE(os.lstat(path).st_mode) != _READ_ONLY_DIRECTORY_MODE
            ):
                raise ValueError("portable preflight tree contains an unexpected directory")
        else:
            raise ValueError("portable preflight tree contains a non-file entry")
    if observed_tree_files != expected_tree_files:
        raise ValueError("portable preflight artifact tree is not closed")

    audit_registration = registrations_by_path.get(PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME)
    request_registration = registrations_by_path.get(
        PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME
    )
    if (
        audit_registration is None
        or audit_registration.get("kind") != "audit_index"
        or audit_registration.get("physical_owner_scope_id") is not None
        or request_registration is None
        or request_registration.get("kind") != "scientific_request"
        or request_registration.get("physical_owner_scope_id") is not None
    ):
        raise ValueError("portable preflight lacks its compact JSON registrations")
    index_path = root / PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME
    request_path = root / PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME
    index = _validate_compact_index(_read_json(index_path, label="portable preflight audit index"))
    _require_unchanged_stat(
        index_path,
        snapshot=snapshots_by_path[PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME],
        label="portable preflight audit index",
    )
    scientific_request = _read_json(
        request_path,
        label="portable preflight scientific request",
    )
    _require_unchanged_stat(
        request_path,
        snapshot=snapshots_by_path[PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME],
        label="portable preflight scientific request",
    )
    scientific_request_body = {
        key: copy.deepcopy(value)
        for key, value in scientific_request.items()
        if key != "content_sha256"
    }
    if (
        set(scientific_request)
        != {
            "schema_version",
            "stage1_request_scientific_projection",
            "portable_cluster_preflight_reference",
            "content_sha256",
        }
        or scientific_request.get("schema_version")
        != PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_SCHEMA
        or scientific_request.get("content_sha256") != _sha256_json(scientific_request_body)
        or scientific_request.get("portable_cluster_preflight_reference") != reference
        or index["content_sha256"] != manifest["audit_index_content_sha256"]
        or index["content_sha256"] != audit_registration["scientific_content_sha256"]
        or scientific_request["content_sha256"] != manifest["scientific_request_content_sha256"]
        or scientific_request["content_sha256"] != request_registration["scientific_content_sha256"]
        or reference != _reference_from_index(index)
        or manifest["logical_scope_count"] != index["logical_scope_count"]
        or manifest["physical_fit_count"] != index["physical_fit_count"]
        or manifest["scope_order"] != index["scope_order"]
        or manifest["physical_scope_order"] != index["physical_scope_order"]
    ):
        raise ValueError("portable preflight compact JSON bindings changed")

    expected_payload_paths: set[str] = set()
    payload_snapshots: dict[str, _FileSnapshot] = {}
    for row in index["physical_fits"]:
        relative = str(row["payload_relative_path"])
        expected_payload_paths.add(relative)
        registration = registrations_by_path.get(relative)
        owner = row["physical_owner_scope_id"]
        if (
            registration is None
            or registration.get("kind") != "physical_owner_concepts"
            or registration.get("physical_owner_scope_id") != owner
            or registration.get("scientific_content_sha256")
            != row["compact_fit_identity"]["concept_payload_content_sha256"]
        ):
            raise ValueError("portable owner concept registration changed")
        _validate_owner_parquet_storage(
            root / relative,
            parquet_compression=physical_storage["parquet_compression"],
        )
        _require_unchanged_stat(
            root / relative,
            snapshot=snapshots_by_path[relative],
            label=f"portable owner concept payload {owner}",
        )
        payload_snapshots[relative] = snapshots_by_path[relative]
    if set(registrations_by_path) != {
        PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME,
        PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME,
        *expected_payload_paths,
    }:
        raise ValueError("portable preflight contains an unregistered payload")

    expected_inventory = _sha256_json(
        [
            {
                "relative_path": row["relative_path"],
                "kind": row["kind"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "scientific_content_sha256": row["scientific_content_sha256"],
                "physical_owner_scope_id": row["physical_owner_scope_id"],
            }
            for row in files
        ]
    )
    if manifest["payload_inventory_content_sha256"] != expected_inventory:
        raise ValueError("portable preflight payload inventory changed")

    identity_body = {
        "schema_version": PORTABLE_CLUSTER_PREFLIGHT_RESULT_SCHEMA,
        "artifact_version": PORTABLE_CLUSTER_PREFLIGHT_ARTIFACT_VERSION,
        "artifact_code_sha256": _artifact_code_sha256(),
        "root": str(root),
        "manifest_path": str(supplied),
        "audit_path": str(index_path),
        "stage1_request_path": str(request_path),
        "manifest_sha256": manifest_snapshot.sha256,
        "audit_sha256": snapshots_by_path[PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME].sha256,
        "stage1_request_file_sha256": snapshots_by_path[
            PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME
        ].sha256,
        "stage1_request_sha256": scientific_request["content_sha256"],
        "cluster_audit_content_sha256": reference["source_audit_content_sha256"],
        "portable_audit_reference_content_sha256": reference["content_sha256"],
        "compact_audit_index_content_sha256": index["content_sha256"],
        "payload_inventory_content_sha256": manifest["payload_inventory_content_sha256"],
        "physical_storage": physical_storage,
        "scope_count": index["logical_scope_count"],
        "physical_fit_count": index["physical_fit_count"],
        "scope_order": list(index["scope_order"]),
        "physical_scope_order": list(index["physical_scope_order"]),
        "scope_fit_identity_sha256": _sha256_json(
            [
                row["physical_fit_reference"]["source_fit_identity_content_sha256"]
                for row in index["logical_scopes"]
            ]
        ),
        "path_neutral_scientific_content_sha256": (
            _path_neutral_scientific_content_sha256(
                index=index,
                scientific_request=scientific_request,
            )
        ),
    }
    identity = {
        **identity_body,
        "content_sha256": _sha256_json(identity_body),
    }
    artifact = PortableProductionStage1ClusterPreflightArtifact(
        root=root,
        manifest_path=supplied,
        audit_path=index_path,
        stage1_request_path=request_path,
        audit=index,
        stage1_request=scientific_request,
        reference=reference,
        _identity=identity,
        _payload_snapshots=payload_snapshots,
    )
    if expected_stage1_request is not None:
        artifact.require_stage1_request(expected_stage1_request)
    return artifact


def load_portable_production_stage1_cluster_preflight_artifact(
    *,
    manifest_path: Path | str,
    config: Any,
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    embedding_cache_identity: Mapping[str, Any],
    expected_stage1_request: Mapping[str, Any] | None = None,
) -> PortableProductionStage1ClusterPreflightArtifact:
    """Compatibility wrapper for dependency-shaped workflow call sites."""

    del config, registry, registry_content_sha256, embedding_cache_identity
    return load_path_only_portable_production_stage1_cluster_preflight_artifact(
        manifest_path=manifest_path,
        expected_stage1_request=expected_stage1_request,
    )


def transcode_portable_production_stage1_cluster_preflight_artifact(
    *,
    source: PortableProductionStage1ClusterPreflightArtifact,
    output_dir: Path | str,
    parquet_compression: str,
) -> PortableProductionStage1ClusterPreflightArtifact:
    """Publish one scientifically identical replica with a configured codec.

    This operation changes only the physical Parquet serialization.  It
    preserves the compact audit index and scientific request byte-for-byte,
    freshly authenticates the complete output tree, parses every output owner
    payload, and rejects any change to the path-neutral scientific root.
    """

    if not isinstance(
        source,
        PortableProductionStage1ClusterPreflightArtifact,
    ):
        raise TypeError("portable preflight transcoding requires a typed source")
    physical_storage = _physical_storage_metadata(
        parquet_compression=parquet_compression,
    )
    target = Path(output_dir)
    if not target.is_absolute():
        raise ValueError("portable preflight transcode output must be absolute")
    if target.exists() or target.is_symlink():
        raise FileExistsError("portable preflight transcode output must be fresh")
    parent = target.parent.resolve(strict=True)
    if parent != target.parent or not parent.is_dir():
        raise ValueError("portable preflight transcode parent must be canonical")

    index = _validate_compact_index(source.audit)
    scientific_request = copy.deepcopy(dict(source.stage1_request))
    source_identity = source.identity()
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
    )
    try:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "portable clustered preflight requires pyarrow Parquet support"
            ) from exc

        payload_root = temporary / PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY
        payload_root.mkdir(exist_ok=False)
        payload_registrations: dict[str, dict[str, Any]] = {}
        for row in index["physical_fits"]:
            owner = str(row["physical_owner_scope_id"])
            relative = str(row["payload_relative_path"])
            source_snapshot = source._payload_snapshots.get(relative)
            if source_snapshot is None:
                raise ValueError(
                    "portable preflight transcode source payload is unauthenticated"
                )
            source_path = source.root / Path(relative)
            _require_unchanged_stat(
                source_path,
                snapshot=source_snapshot,
                label=f"portable preflight transcode source {owner}",
            )
            table = pq.read_table(source_path)
            rows = table.to_pylist()
            _require_unchanged_stat(
                source_path,
                snapshot=source_snapshot,
                label=f"portable preflight transcode source {owner}",
            )
            output_path = temporary / Path(relative)
            _write_owner_parquet(
                output_path,
                rows,
                parquet_compression=parquet_compression,
            )
            output_snapshot = _stable_file(
                output_path,
                label=f"transcoded owner concept payload {owner}",
                require_read_only=False,
            )
            compact_fit = row["compact_fit_identity"]
            payload_registrations[owner] = _manifest_file_registration(
                relative_path=relative,
                kind="physical_owner_concepts",
                snapshot=output_snapshot,
                scientific_content_sha256=compact_fit[
                    "concept_payload_content_sha256"
                ],
                physical_owner_scope_id=owner,
            )

        registrations: list[dict[str, Any]] = []
        index_path = temporary / PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME
        _write_json_new(index_path, index)
        index_snapshot = _stable_file(
            index_path,
            label="transcoded portable audit index",
            require_read_only=False,
        )
        registrations.append(
            _manifest_file_registration(
                relative_path=PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME,
                kind="audit_index",
                snapshot=index_snapshot,
                scientific_content_sha256=index["content_sha256"],
                physical_owner_scope_id=None,
            )
        )
        request_path = (
            temporary
            / PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME
        )
        _write_json_new(request_path, scientific_request)
        request_snapshot = _stable_file(
            request_path,
            label="transcoded portable scientific request",
            require_read_only=False,
        )
        registrations.append(
            _manifest_file_registration(
                relative_path=(
                    PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME
                ),
                kind="scientific_request",
                snapshot=request_snapshot,
                scientific_content_sha256=scientific_request[
                    "content_sha256"
                ],
                physical_owner_scope_id=None,
            )
        )
        if set(payload_registrations) != set(index["physical_scope_order"]):
            raise RuntimeError(
                "portable preflight transcode omitted a physical owner"
            )
        registrations.extend(
            payload_registrations[owner]
            for owner in index["physical_scope_order"]
        )

        manifest_body = _manifest_body(
            index=index,
            scientific_request=scientific_request,
            registrations=registrations,
            physical_storage=physical_storage,
        )
        manifest = {
            **manifest_body,
            "content_sha256": _sha256_json(manifest_body),
        }
        manifest_path = temporary / PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME
        _write_json_new(manifest_path, manifest)

        for path in temporary.rglob("*"):
            if path.is_file():
                path.chmod(_READ_ONLY_FILE_MODE)
        payload_root.chmod(_READ_ONLY_DIRECTORY_MODE)
        temporary.chmod(_READ_ONLY_DIRECTORY_MODE)
        _fsync_directory(payload_root)
        _fsync_directory(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                "portable preflight transcode target was populated during publication"
            )
        os.rename(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.chmod(stat.S_IRWXU)
            for path in temporary.rglob("*"):
                if path.is_dir():
                    path.chmod(stat.S_IRWXU)
                else:
                    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    replica = (
        load_path_only_portable_production_stage1_cluster_preflight_artifact(
            manifest_path=(
                target / PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME
            ),
        )
    )
    replica_identity = replica.identity()
    for owner in index["physical_scope_order"]:
        replica.owner_fit_identity(str(owner))
    if (
        replica_identity["path_neutral_scientific_content_sha256"]
        != source_identity["path_neutral_scientific_content_sha256"]
        or replica_identity["cluster_audit_content_sha256"]
        != source_identity["cluster_audit_content_sha256"]
        or replica_identity["scope_fit_identity_sha256"]
        != source_identity["scope_fit_identity_sha256"]
        or replica_identity["scope_order"] != source_identity["scope_order"]
        or replica_identity["physical_scope_order"]
        != source_identity["physical_scope_order"]
        or dict(replica.audit) != index
        or dict(replica.stage1_request) != scientific_request
    ):
        raise RuntimeError(
            "portable preflight transcode changed scientific content"
        )
    return replica


__all__ = [
    "PORTABLE_CLUSTER_PREFLIGHT_ARTIFACT_VERSION",
    "PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_NAME",
    "PORTABLE_CLUSTER_PREFLIGHT_AUDIT_INDEX_SCHEMA",
    "PORTABLE_CLUSTER_PREFLIGHT_CONCEPT_DIRECTORY",
    "PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_NAME",
    "PORTABLE_CLUSTER_PREFLIGHT_MANIFEST_SCHEMA",
    "PORTABLE_CLUSTER_PREFLIGHT_REFERENCE_SCHEMA",
    "PORTABLE_CLUSTER_PREFLIGHT_RESULT_SCHEMA",
    "PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_CONTENT_SCHEMA",
    "PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_NAME",
    "PORTABLE_CLUSTER_PREFLIGHT_SCIENTIFIC_REQUEST_SCHEMA",
    "SUPPORTED_PORTABLE_CLUSTER_PREFLIGHT_PARQUET_COMPRESSIONS",
    "PortableProductionStage1ClusterPreflightArtifact",
    "build_portable_cluster_preflight_reference",
    "load_path_only_portable_production_stage1_cluster_preflight_artifact",
    "load_portable_production_stage1_cluster_preflight_artifact",
    "seal_portable_production_stage1_cluster_preflight_artifact",
    "transcode_portable_production_stage1_cluster_preflight_artifact",
    "validate_portable_cluster_preflight_reference",
]
