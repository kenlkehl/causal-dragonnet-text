"""Immutable handoff for the all-scope Stage 1 clustered-embedding preflight.

The scientific preflight is intentionally a separate production phase.  This
module seals its complete 40-scope audit together with the exact Stage 1
request that produced it.  A later supervised modeling process must reopen
these bytes, validate them against its independently reconstructed
config/registry/cache inputs, and require an exact request match.  It may not
silently recompute or substitute a clustered-embedding result.
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
from typing import Any, Iterable, Mapping, Sequence


PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_ARTIFACT_VERSION = (
    "production_stage1_cluster_preflight_artifact_v1"
)
PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_MANIFEST_SCHEMA = (
    "production_stage1_cluster_preflight_manifest_v1"
)
PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_RESULT_SCHEMA = (
    "production_stage1_cluster_preflight_result_v1"
)
CLUSTER_PREFLIGHT_AUDIT_NAME = "cluster_feasibility_audit.json"
CLUSTER_PREFLIGHT_REQUEST_NAME = "stage1_preflight_request.json"
CLUSTER_PREFLIGHT_MANIFEST_NAME = "cluster_preflight_manifest.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "artifact_version",
        "artifact_code_sha256",
        "root",
        "files",
        "bindings",
        "scope_records",
        "content_sha256",
    }
)
_FILES_FIELDS = frozenset({"audit", "stage1_request"})
_FILE_FIELDS = frozenset({"relative_path", "sha256", "size_bytes"})
_BINDING_FIELDS = frozenset(
    {
        "stage1_request_sha256",
        "dataset",
        "source_config",
        "effective_stage1_config_sha256",
        "embedding_cache_sha256",
        "htr_model",
        "split_registry_content_sha256",
        "behavior_identity_sha256",
        "hierarchical_discovery_contract_identity_sha256",
        "architecture_contract_sha256",
        "stage1_scope_plan_sha256",
        "query_config_sha256",
        "runtime_sha256",
        "cluster_audit_content_sha256",
        "cluster_scope_order_sha256",
    }
)
_SCOPE_FIELDS = frozenset(
    {
        "canonical_index",
        "scope_id",
        "scope_kind",
        "outer_fold",
        "inner_fold",
        "context_epoch",
        "provider_inner_fold",
        "fit_row_count",
        "fit_row_order_fingerprint",
        "heldout_row_count",
        "heldout_row_order_fingerprint",
        "scope_record_sha256",
        "cluster_fit_identity_sha256",
    }
)
_RESULT_FIELDS = frozenset(
    {
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
        "scope_count",
        "scope_order",
        "scope_fit_identity_sha256",
        "content_sha256",
    }
)
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
        raise ValueError("cluster preflight values must contain valid UTF-8") from exc
    return result


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"cluster preflight JSON contains duplicate key: {key}")
        output[key] = value
    return output


@dataclass(frozen=True)
class _FileSnapshot:
    sha256: str
    size_bytes: int
    stat_identity: tuple[int, int, int, int, int, int, int]

    def registration(self, *, relative_path: str) -> dict[str, Any]:
        return {
            "relative_path": relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


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


def _stable_file(path: Path, *, label: str) -> _FileSnapshot:
    try:
        before_path = os.lstat(path)
    except OSError as exc:
        raise FileNotFoundError(f"{label} is absent: {path}") from exc
    if (
        stat.S_ISLNK(before_path.st_mode)
        or not stat.S_ISREG(before_path.st_mode)
        or int(before_path.st_nlink) != 1
    ):
        raise ValueError(f"{label} must be one non-linked regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        before_fd = os.fstat(descriptor)
        if _stat_identity(before_fd) != _stat_identity(before_path):
            raise RuntimeError(f"{label} changed while it was opened")
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(path)
    identity = _stat_identity(before_path)
    if (
        _stat_identity(before_fd) != identity
        or _stat_identity(after_fd) != identity
        or _stat_identity(after_path) != identity
    ):
        raise RuntimeError(f"{label} changed while it was authenticated")
    return _FileSnapshot(
        sha256=digest.hexdigest(),
        size_bytes=int(after_path.st_size),
        stat_identity=identity,
    )


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], _FileSnapshot]:
    before = _stable_file(path, label=label)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    after = _stable_file(path, label=label)
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
    ).encode("utf-8")
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
        cursor = 0
        while cursor < len(payload):
            cursor += os.write(descriptor, payload[cursor:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_code_sha256() -> str:
    return _stable_file(
        Path(__file__).resolve(strict=True),
        label="cluster preflight artifact module",
    ).sha256


def _content_identity(value: Any, *, label: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be one content-addressed mapping")
    content = _require_sha256(value.get("content_sha256"), label=f"{label}.content_sha256")
    body = {key: copy.deepcopy(item) for key, item in value.items() if key != "content_sha256"}
    if content != _sha256_json(body):
        raise ValueError(f"{label} content identity is invalid")
    return content


def _validate_stage1_request(
    request: Mapping[str, Any],
    *,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    from .production_stage1_bundle import STAGE1_BUNDLE_REQUEST_SCHEMA
    from .production_stage1_hierarchy_contract import (
        validate_production_stage1_hierarchy_request_bindings,
    )

    output = copy.deepcopy(dict(request))
    request_sha = _require_sha256(
        output.get("request_sha256"),
        label="stage1_request.request_sha256",
    )
    body = {key: item for key, item in output.items() if key != "request_sha256"}
    if (
        output.get("schema_version") != STAGE1_BUNDLE_REQUEST_SCHEMA
        or request_sha != _sha256_json(body)
        or output.get("embedding_cluster_feasibility_audit") != dict(audit)
    ):
        raise ValueError("Stage 1 preflight request is not exactly bound to its audit")
    validate_production_stage1_hierarchy_request_bindings(output)
    return output


def _validate_scientific_audit(
    audit: Mapping[str, Any],
    *,
    config: Any,
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    embedding_cache_identity: Mapping[str, Any],
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
    )
    if validated != dict(audit):
        raise RuntimeError("cluster preflight validator changed accepted audit bytes")
    return copy.deepcopy(dict(validated))


def _scope_records(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    scopes = audit.get("scopes")
    order = audit.get("scope_order")
    if (
        not isinstance(scopes, list)
        or not isinstance(order, list)
        or len(scopes) != len(order)
        or [row.get("scope_id") for row in scopes if isinstance(row, Mapping)] != order
    ):
        raise ValueError("cluster preflight scope inventory is incomplete or reordered")
    records: list[dict[str, Any]] = []
    for canonical_index, scope in enumerate(scopes):
        if not isinstance(scope, Mapping):
            raise ValueError("cluster preflight scope record is invalid")
        fit_identity = scope.get("cluster_fit_identity")
        if not isinstance(fit_identity, Mapping):
            raise ValueError("cluster preflight scope lacks its fitted cluster identity")
        record = {
            "canonical_index": canonical_index,
            "scope_id": scope.get("scope_id"),
            "scope_kind": scope.get("scope_kind"),
            "outer_fold": scope.get("outer_fold"),
            "inner_fold": scope.get("inner_fold"),
            "context_epoch": scope.get("context_epoch"),
            "provider_inner_fold": scope.get("provider_inner_fold"),
            "fit_row_count": scope.get("fit_row_count"),
            "fit_row_order_fingerprint": scope.get("fit_row_order_fingerprint"),
            "heldout_row_count": scope.get("heldout_row_count"),
            "heldout_row_order_fingerprint": scope.get("heldout_row_order_fingerprint"),
            "scope_record_sha256": _sha256_json(dict(scope)),
            "cluster_fit_identity_sha256": _sha256_json(dict(fit_identity)),
        }
        if (
            set(record) != set(_SCOPE_FIELDS)
            or not isinstance(record["scope_id"], str)
            or not record["scope_id"]
            or record["scope_id"] != order[canonical_index]
            or any(
                not isinstance(record[field_name], int)
                or isinstance(record[field_name], bool)
                or record[field_name] < 0
                for field_name in (
                    "canonical_index",
                    "outer_fold",
                    "fit_row_count",
                    "heldout_row_count",
                )
            )
        ):
            raise ValueError("cluster preflight scope binding is invalid")
        _require_sha256(
            record["fit_row_order_fingerprint"],
            label=f"{record['scope_id']}.fit_row_order_fingerprint",
        )
        _require_sha256(
            record["heldout_row_order_fingerprint"],
            label=f"{record['scope_id']}.heldout_row_order_fingerprint",
        )
        records.append(record)
    if len({row["scope_id"] for row in records}) != len(records):
        raise ValueError("cluster preflight scope registry contains duplicates")
    return records


def _request_bindings(
    request: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    required_mappings = {
        name: request.get(name)
        for name in (
            "dataset",
            "source_config",
            "effective_stage1_config",
            "embedding_cache",
            "htr_model",
            "behavior_identity",
            "hierarchical_discovery_contract_identity",
            "architecture_contract",
            "stage1_scope_plan",
            "query_config",
            "runtime",
        )
    }
    if any(not isinstance(value, Mapping) for value in required_mappings.values()):
        raise ValueError("Stage 1 request lacks a required preflight binding")
    bindings = {
        "stage1_request_sha256": request["request_sha256"],
        "dataset": copy.deepcopy(dict(required_mappings["dataset"])),
        "source_config": copy.deepcopy(dict(required_mappings["source_config"])),
        "effective_stage1_config_sha256": _sha256_json(
            required_mappings["effective_stage1_config"]
        ),
        "embedding_cache_sha256": _sha256_json(required_mappings["embedding_cache"]),
        "htr_model": copy.deepcopy(dict(required_mappings["htr_model"])),
        "split_registry_content_sha256": request.get(
            "split_registry_content_sha256"
        ),
        "behavior_identity_sha256": _content_identity(
            required_mappings["behavior_identity"],
            label="behavior_identity",
        ),
        "hierarchical_discovery_contract_identity_sha256": _content_identity(
            required_mappings["hierarchical_discovery_contract_identity"],
            label="hierarchical_discovery_contract_identity",
        ),
        "architecture_contract_sha256": _sha256_json(
            required_mappings["architecture_contract"]
        ),
        "stage1_scope_plan_sha256": _content_identity(
            required_mappings["stage1_scope_plan"],
            label="stage1_scope_plan",
        ),
        "query_config_sha256": _sha256_json(required_mappings["query_config"]),
        "runtime_sha256": _sha256_json(required_mappings["runtime"]),
        "cluster_audit_content_sha256": audit["content_sha256"],
        "cluster_scope_order_sha256": _sha256_json(audit["scope_order"]),
    }
    if set(bindings) != set(_BINDING_FIELDS):
        raise RuntimeError("cluster preflight request binding schema changed")
    for field_name in (
        "stage1_request_sha256",
        "effective_stage1_config_sha256",
        "embedding_cache_sha256",
        "split_registry_content_sha256",
        "behavior_identity_sha256",
        "hierarchical_discovery_contract_identity_sha256",
        "architecture_contract_sha256",
        "stage1_scope_plan_sha256",
        "query_config_sha256",
        "runtime_sha256",
        "cluster_audit_content_sha256",
        "cluster_scope_order_sha256",
    ):
        _require_sha256(bindings[field_name], label=f"bindings.{field_name}")
    return bindings


@dataclass(frozen=True)
class ProductionStage1ClusterPreflightArtifact:
    root: Path
    manifest_path: Path
    audit_path: Path
    stage1_request_path: Path
    audit: Mapping[str, Any] = field(repr=False)
    stage1_request: Mapping[str, Any] = field(repr=False)
    _identity: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        identity = copy.deepcopy(dict(self._identity))
        if (
            set(identity) != set(_RESULT_FIELDS)
            or identity.get("schema_version")
            != PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_RESULT_SCHEMA
            or identity.get("artifact_version")
            != PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_ARTIFACT_VERSION
            or identity.get("artifact_code_sha256") != _artifact_code_sha256()
            or identity.get("content_sha256")
            != _sha256_json(
                {
                    key: value
                    for key, value in identity.items()
                    if key != "content_sha256"
                }
            )
        ):
            raise ValueError("cluster preflight result identity is invalid")
        object.__setattr__(
            self,
            "audit",
            MappingProxyType(copy.deepcopy(dict(self.audit))),
        )
        object.__setattr__(
            self,
            "stage1_request",
            MappingProxyType(copy.deepcopy(dict(self.stage1_request))),
        )
        object.__setattr__(self, "_identity", MappingProxyType(identity))

    def identity(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self._identity))

    def require_stage1_request(
        self,
        expected_stage1_request: Mapping[str, Any],
    ) -> None:
        expected = copy.deepcopy(dict(expected_stage1_request))
        if (
            expected != dict(self.stage1_request)
            or expected.get("request_sha256")
            != self._identity["stage1_request_sha256"]
        ):
            raise ValueError(
                "supervised Stage 1 request differs from sealed cluster preflight request"
            )


def _manifest_body(
    *,
    root: Path,
    audit_snapshot: _FileSnapshot,
    request_snapshot: _FileSnapshot,
    request: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_MANIFEST_SCHEMA,
        "status": "complete",
        "artifact_version": PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_ARTIFACT_VERSION,
        "artifact_code_sha256": _artifact_code_sha256(),
        "root": str(root),
        "files": {
            "audit": audit_snapshot.registration(
                relative_path=CLUSTER_PREFLIGHT_AUDIT_NAME
            ),
            "stage1_request": request_snapshot.registration(
                relative_path=CLUSTER_PREFLIGHT_REQUEST_NAME
            ),
        },
        "bindings": _request_bindings(request, audit),
        "scope_records": _scope_records(audit),
    }


def seal_production_stage1_cluster_preflight_artifact(
    *,
    output_dir: Path | str,
    audit: Mapping[str, Any],
    stage1_request: Mapping[str, Any],
    config: Any,
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    embedding_cache_identity: Mapping[str, Any],
) -> ProductionStage1ClusterPreflightArtifact:
    """Validate and atomically publish one independently reusable preflight."""

    target = Path(output_dir)
    if not target.is_absolute():
        raise ValueError("cluster preflight output directory must be absolute")
    parent = target.parent.resolve(strict=True)
    if parent != target.parent or target.is_symlink() or target.exists():
        raise FileExistsError("cluster preflight output directory must be fresh")
    validated_audit = _validate_scientific_audit(
        audit,
        config=config,
        registry=registry,
        registry_content_sha256=registry_content_sha256,
        embedding_cache_identity=embedding_cache_identity,
    )
    validated_request = _validate_stage1_request(
        stage1_request,
        audit=validated_audit,
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        audit_path = temporary / CLUSTER_PREFLIGHT_AUDIT_NAME
        request_path = temporary / CLUSTER_PREFLIGHT_REQUEST_NAME
        _write_json_new(audit_path, validated_audit)
        _write_json_new(request_path, validated_request)
        audit_snapshot = _stable_file(audit_path, label="written cluster audit")
        request_snapshot = _stable_file(
            request_path,
            label="written Stage 1 preflight request",
        )
        body = _manifest_body(
            root=target,
            audit_snapshot=audit_snapshot,
            request_snapshot=request_snapshot,
            request=validated_request,
            audit=validated_audit,
        )
        manifest = {**body, "content_sha256": _sha256_json(body)}
        _write_json_new(temporary / CLUSTER_PREFLIGHT_MANIFEST_NAME, manifest)
        for path in temporary.iterdir():
            path.chmod(_READ_ONLY_FILE_MODE)
        temporary.chmod(_READ_ONLY_DIRECTORY_MODE)
        if target.is_symlink() or target.exists():
            raise FileExistsError("cluster preflight target was populated during publication")
        os.rename(temporary, target)
        parent_descriptor = os.open(
            target.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
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
    return load_production_stage1_cluster_preflight_artifact(
        manifest_path=target / CLUSTER_PREFLIGHT_MANIFEST_NAME,
        config=config,
        registry=registry,
        registry_content_sha256=registry_content_sha256,
        embedding_cache_identity=embedding_cache_identity,
        expected_stage1_request=validated_request,
    )


def _validate_registration(
    value: Any,
    *,
    expected_relative_path: str,
    snapshot: _FileSnapshot,
    label: str,
) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(_FILE_FIELDS)
        or value.get("relative_path") != expected_relative_path
        or value.get("sha256") != snapshot.sha256
        or value.get("size_bytes") != snapshot.size_bytes
    ):
        raise ValueError(f"{label} registration differs from its file")


def load_production_stage1_cluster_preflight_artifact(
    *,
    manifest_path: Path | str,
    config: Any,
    registry: Mapping[str, Any],
    registry_content_sha256: str,
    embedding_cache_identity: Mapping[str, Any],
    expected_stage1_request: Mapping[str, Any] | None = None,
) -> ProductionStage1ClusterPreflightArtifact:
    """Reopen and authenticate a sealed preflight without recomputing fits."""

    supplied_manifest = Path(manifest_path)
    if not supplied_manifest.is_absolute():
        raise ValueError("cluster preflight manifest path must be absolute")
    root = supplied_manifest.parent
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
        or supplied_manifest.name != CLUSTER_PREFLIGHT_MANIFEST_NAME
    ):
        raise ValueError("cluster preflight root or manifest path is invalid")
    observed_names = {path.name for path in root.iterdir()}
    expected_names = {
        CLUSTER_PREFLIGHT_AUDIT_NAME,
        CLUSTER_PREFLIGHT_REQUEST_NAME,
        CLUSTER_PREFLIGHT_MANIFEST_NAME,
    }
    if observed_names != expected_names:
        raise ValueError("cluster preflight artifact tree is not closed")
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("cluster preflight artifact contains a non-file")
        if stat.S_IMODE(os.lstat(path).st_mode) != _READ_ONLY_FILE_MODE:
            raise ValueError("cluster preflight artifact contains a writable file")
    if stat.S_IMODE(os.lstat(root).st_mode) != _READ_ONLY_DIRECTORY_MODE:
        raise ValueError("cluster preflight artifact root is writable")

    manifest, manifest_snapshot = _read_json(
        supplied_manifest,
        label="cluster preflight manifest",
    )
    body = {key: copy.deepcopy(value) for key, value in manifest.items() if key != "content_sha256"}
    files = manifest.get("files")
    if (
        set(manifest) != set(_MANIFEST_FIELDS)
        or manifest.get("schema_version")
        != PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("artifact_version")
        != PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_ARTIFACT_VERSION
        or manifest.get("artifact_code_sha256") != _artifact_code_sha256()
        or manifest.get("root") != str(root)
        or manifest.get("content_sha256") != _sha256_json(body)
        or not isinstance(files, Mapping)
        or set(files) != set(_FILES_FIELDS)
        or not isinstance(manifest.get("bindings"), Mapping)
        or set(manifest["bindings"]) != set(_BINDING_FIELDS)
        or not isinstance(manifest.get("scope_records"), list)
    ):
        raise ValueError("cluster preflight manifest has an invalid closed envelope")

    audit_path = root / CLUSTER_PREFLIGHT_AUDIT_NAME
    request_path = root / CLUSTER_PREFLIGHT_REQUEST_NAME
    audit, audit_snapshot = _read_json(audit_path, label="cluster preflight audit")
    request, request_snapshot = _read_json(
        request_path,
        label="Stage 1 preflight request",
    )
    _validate_registration(
        files["audit"],
        expected_relative_path=CLUSTER_PREFLIGHT_AUDIT_NAME,
        snapshot=audit_snapshot,
        label="cluster preflight audit",
    )
    _validate_registration(
        files["stage1_request"],
        expected_relative_path=CLUSTER_PREFLIGHT_REQUEST_NAME,
        snapshot=request_snapshot,
        label="Stage 1 preflight request",
    )
    validated_audit = _validate_scientific_audit(
        audit,
        config=config,
        registry=registry,
        registry_content_sha256=registry_content_sha256,
        embedding_cache_identity=embedding_cache_identity,
    )
    validated_request = _validate_stage1_request(
        request,
        audit=validated_audit,
    )
    expected_bindings = _request_bindings(validated_request, validated_audit)
    expected_scopes = _scope_records(validated_audit)
    if (
        manifest["bindings"] != expected_bindings
        or manifest["scope_records"] != expected_scopes
    ):
        raise ValueError("cluster preflight bindings or scope order changed")
    if expected_stage1_request is not None and validated_request != dict(
        expected_stage1_request
    ):
        raise ValueError("cluster preflight request differs from supervised Stage 1 request")

    fit_identities = [
        row["cluster_fit_identity_sha256"] for row in expected_scopes
    ]
    identity_body = {
        "schema_version": PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_RESULT_SCHEMA,
        "artifact_version": PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_ARTIFACT_VERSION,
        "artifact_code_sha256": _artifact_code_sha256(),
        "root": str(root),
        "manifest_path": str(supplied_manifest),
        "audit_path": str(audit_path),
        "stage1_request_path": str(request_path),
        "manifest_sha256": manifest_snapshot.sha256,
        "audit_sha256": audit_snapshot.sha256,
        "stage1_request_file_sha256": request_snapshot.sha256,
        "stage1_request_sha256": validated_request["request_sha256"],
        "cluster_audit_content_sha256": validated_audit["content_sha256"],
        "scope_count": len(expected_scopes),
        "scope_order": [row["scope_id"] for row in expected_scopes],
        "scope_fit_identity_sha256": _sha256_json(fit_identities),
    }
    identity = {
        **identity_body,
        "content_sha256": _sha256_json(identity_body),
    }
    artifact = ProductionStage1ClusterPreflightArtifact(
        root=root,
        manifest_path=supplied_manifest,
        audit_path=audit_path,
        stage1_request_path=request_path,
        audit=validated_audit,
        stage1_request=validated_request,
        _identity=identity,
    )
    if expected_stage1_request is not None:
        artifact.require_stage1_request(expected_stage1_request)
    return artifact


__all__ = [
    "CLUSTER_PREFLIGHT_AUDIT_NAME",
    "CLUSTER_PREFLIGHT_MANIFEST_NAME",
    "CLUSTER_PREFLIGHT_REQUEST_NAME",
    "PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_ARTIFACT_VERSION",
    "PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_MANIFEST_SCHEMA",
    "PRODUCTION_STAGE1_CLUSTER_PREFLIGHT_RESULT_SCHEMA",
    "ProductionStage1ClusterPreflightArtifact",
    "load_production_stage1_cluster_preflight_artifact",
    "seal_production_stage1_cluster_preflight_artifact",
]
