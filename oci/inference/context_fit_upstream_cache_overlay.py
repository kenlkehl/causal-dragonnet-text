"""Authenticated read-only reuse of complete context-fit upstream bundles.

The gate and final upstream producers already write content-addressed caches,
but benchmark runs intentionally start in fresh output directories.  This
module is the narrow bridge between those two properties.  A caller registers
an externally SHA-256-authenticated cache *index*. Every distinct indexed
file is read once into an immutable byte snapshot during authentication.
Historical paths are never used as writable cache roots.

Only complete top-level bundles are eligible.  Backend work directories and
executable checkpoints are deliberately outside the index schema.  On an
exact current binding hit, the snapshots are atomically materialized in the
fresh output-local cache and the original producer performs its normal cache
authentication.  Gate bundles are materialized only from ``bind_fold``;
final bundles are materialized only from ``produce``.

The companion run manifest matters for historical gate bundles.  The gate
cache identity predates recursive backend runtime attestation, while the final
producer identity in the same immutable input manifest records it.  Reuse is
therefore accepted only when the run manifest links the exact gate identity to
the exact recursively attested final backend used by the current process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np

from .all_evidence_post_extraction_review import ObservableCausalRows
from .context_fit_upstream_gate_provider import (
    CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION,
    ContextFitUpstreamGateProvider,
    _context_folds as _gate_context_folds,
    _exact_texts as _gate_exact_texts,
    _integer_rows as _gate_integer_rows,
    _positive_int as _gate_positive_int,
)
from .final_context_fit_upstream_bank import (
    FINAL_CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA,
    FinalContextFitUpstreamProducer,
    _exact_texts as _final_exact_texts,
    _expected_fit_rows_by_position,
    _finite_vector as _final_finite_vector,
    _fold_ids as _final_fold_ids,
    _integer_rows as _final_integer_rows,
    _positive_int as _final_positive_int,
)

CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION = "context_fit_upstream_cache_index_v1"
CONTEXT_FIT_GATE_CACHE_OVERLAY_ID = "authenticated_context_fit_gate_cache_overlay_v1"
FINAL_CONTEXT_FIT_CACHE_OVERLAY_ID = "authenticated_final_context_fit_cache_overlay_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INDEX_FIELDS = frozenset({"schema_version", "entries", "content_sha256"})
_INDEX_ENTRY_FIELDS = frozenset(
    {
        "kind",
        "cache_manifest_path",
        "cache_manifest_sha256",
        "cache_files",
        "run_manifest_path",
        "run_manifest_sha256",
    }
)
_RUN_MANIFEST_FIELDS = frozenset({"schema_version", "body", "content_sha256"})
_GATE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "cache_key",
        "binding",
        "context_row_ids",
        "context_inner_fold_ids",
        "gate_row_ids",
        "source_names",
        "source_kinds",
        "source_values_file",
        "source_values_sha256",
        "source_context_values_file",
        "source_context_values_sha256",
        "feature_names",
        "feature_kinds",
        "feature_roles",
        "feature_values_file",
        "feature_values_sha256",
        "feature_context_values_file",
        "feature_context_values_sha256",
        "content_sha256",
    }
)
_GATE_FILES = (
    "calibrated_sources.npy",
    "features.npy",
    "calibrated_sources_context_oof.npy",
    "features_context_oof.npy",
)
_FINAL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "cache_key",
        "binding",
        "calibrated_sources",
        "raw_features",
        "matrix_files",
        "content_sha256",
    }
)
_FINAL_FILES = (
    "calibrated_source_train_oof.npy",
    "calibrated_source_outer_heldout.npy",
    "raw_feature_train_oof.npy",
    "raw_feature_outer_heldout.npy",
)
_FINAL_MATRIX_KEYS = frozenset(
    {
        "source_train_oof",
        "source_outer_heldout",
        "feature_train_oof",
        "feature_outer_heldout",
    }
)
_KINDS = frozenset({"review_gate", "final_upstream"})


class ContextFitCacheAuthenticationError(RuntimeError):
    """A registered cache index or one of its snapshots is unauthenticated."""


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


def _required_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip()
    if not _SHA256.fullmatch(digest):
        raise ContextFitCacheAuthenticationError(f"{label} must be one lowercase SHA-256")
    return digest


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON field {key!r}")
        output[key] = value
    return output


def _parse_json_snapshot(snapshot: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_closed_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContextFitCacheAuthenticationError(
            f"{label} is not closed finite UTF-8 JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise ContextFitCacheAuthenticationError(f"{label} root must be an object")
    return value


def _read_snapshot(path: Path, *, expected_sha256: str, label: str) -> bytes:
    try:
        snapshot = path.read_bytes()
    except OSError as exc:
        raise ContextFitCacheAuthenticationError(f"{label} is unreadable: {path}") from exc
    if hashlib.sha256(snapshot).hexdigest() != expected_sha256:
        raise ContextFitCacheAuthenticationError(f"{label} SHA-256 mismatch: {path}")
    return snapshot


class _SnapshotRegistry:
    """Read each resolved source path once and reject conflicting declarations."""

    def __init__(self) -> None:
        self._digest_by_path: dict[Path, str] = {}
        self._snapshot_by_path: dict[Path, bytes] = {}

    def read(self, path: Path, *, expected_sha256: str, label: str) -> bytes:
        prior = self._digest_by_path.get(path)
        if prior is not None and prior != expected_sha256:
            raise ContextFitCacheAuthenticationError(
                f"{label} path was registered with conflicting SHA-256 digests: {path}"
            )
        snapshot = self._snapshot_by_path.get(path)
        if snapshot is None:
            snapshot = _read_snapshot(path, expected_sha256=expected_sha256, label=label)
            self._digest_by_path[path] = expected_sha256
            self._snapshot_by_path[path] = snapshot
        return snapshot


def _resolve_index_path(index_path: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ContextFitCacheAuthenticationError(f"{label} must be a non-empty path")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = index_path.parent / candidate
    return candidate.resolve()


def _closed_hash_wrapper(snapshot: bytes, *, label: str) -> Mapping[str, Any]:
    raw = _parse_json_snapshot(snapshot, label=label)
    if set(raw) != _RUN_MANIFEST_FIELDS:
        raise ContextFitCacheAuthenticationError(f"{label} has an unsupported closed schema")
    body = raw["body"]
    if not isinstance(body, Mapping) or raw["content_sha256"] != _sha256_json(body):
        raise ContextFitCacheAuthenticationError(f"{label} content hash mismatch")
    if raw["schema_version"] != body.get("runner_schema_version"):
        raise ContextFitCacheAuthenticationError(f"{label} schema does not match its runner schema")
    return raw


def _manifest_identity_record(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"identity", "identity_sha256"}:
        raise ContextFitCacheAuthenticationError(f"{label} identity record is malformed")
    identity = value["identity"]
    if not isinstance(identity, Mapping) or value["identity_sha256"] != _sha256_json(identity):
        raise ContextFitCacheAuthenticationError(f"{label} identity hash mismatch")
    return identity


def _unwrap_gate_identity(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    if identity.get("provider") == CONTEXT_FIT_GATE_CACHE_OVERLAY_ID:
        delegate = identity.get("delegate_provider_identity")
        if not isinstance(delegate, Mapping):
            raise ContextFitCacheAuthenticationError(
                "historical gate overlay identity lacks a delegate identity"
            )
        return delegate
    return identity


def _unwrap_final_identity(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    if identity.get("producer") == FINAL_CONTEXT_FIT_CACHE_OVERLAY_ID:
        delegate = identity.get("delegate_producer_identity")
        if not isinstance(delegate, Mapping):
            raise ContextFitCacheAuthenticationError(
                "historical final overlay identity lacks a delegate identity"
            )
        return delegate
    return identity


@dataclass(frozen=True)
class _RunAttestation:
    run_manifest_path: Path
    run_manifest_sha256: str
    runner_schema_version: str
    _gate_provider_identity_json: str = field(repr=False)
    _final_producer_identity_json: str = field(repr=False)

    @property
    def gate_provider_identity(self) -> Mapping[str, Any]:
        return json.loads(self._gate_provider_identity_json)

    @property
    def final_producer_identity(self) -> Mapping[str, Any]:
        return json.loads(self._final_producer_identity_json)

    def identity(self) -> dict[str, Any]:
        return {
            "run_manifest_path": str(self.run_manifest_path),
            "run_manifest_sha256": self.run_manifest_sha256,
            "runner_schema_version": self.runner_schema_version,
            "gate_provider_identity_sha256": _sha256_json(self.gate_provider_identity),
            "final_producer_identity_sha256": _sha256_json(self.final_producer_identity),
        }


def _authenticate_run_manifest(
    *, path: Path, expected_sha256: str, snapshot: bytes
) -> _RunAttestation:
    raw = _closed_hash_wrapper(snapshot, label="companion immutable run-input manifest")
    body = raw["body"]
    providers = body.get("post_extraction_review_providers")
    final_inputs = body.get("final_upstream_model_inputs")
    if not isinstance(providers, Mapping) or not isinstance(final_inputs, Mapping):
        raise ContextFitCacheAuthenticationError(
            "companion run manifest lacks upstream provider attestations"
        )
    gate_source = _manifest_identity_record(
        providers.get("calibrated_gate_sources"), label="historical gate source provider"
    )
    gate_features = _manifest_identity_record(
        providers.get("role_aware_gate_feature_banks"),
        label="historical gate feature provider",
    )
    if gate_source != gate_features:
        raise ContextFitCacheAuthenticationError(
            "historical gate source and feature providers do not share one identity"
        )
    final_identity = _manifest_identity_record(
        final_inputs.get("producer"), label="historical final upstream producer"
    )
    gate_delegate = _unwrap_gate_identity(gate_source)
    final_delegate = _unwrap_final_identity(final_identity)
    gate_backend = gate_delegate.get("backend")
    final_backend = final_delegate.get("backend_identity")
    runtime = final_delegate.get("backend_runtime_attestation")
    if gate_backend != final_backend or not isinstance(runtime, Mapping):
        raise ContextFitCacheAuthenticationError(
            "companion run manifest does not link gate identity to a runtime-attested backend"
        )
    return _RunAttestation(
        run_manifest_path=path,
        run_manifest_sha256=expected_sha256,
        runner_schema_version=str(raw["schema_version"]),
        _gate_provider_identity_json=_canonical_json(gate_delegate),
        _final_producer_identity_json=_canonical_json(final_delegate),
    )


@dataclass(frozen=True)
class _FileSnapshot:
    filename: str
    sha256: str
    snapshot: bytes = field(repr=False)

    def identity(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "byte_count": len(self.snapshot),
        }


@dataclass(frozen=True)
class AuthenticatedContextFitCacheSource:
    """One complete immutable gate or final cache snapshot."""

    kind: str
    index_path: Path
    index_sha256: str
    cache_manifest_path: Path
    cache_manifest_sha256: str
    cache_key: str
    run_attestation: _RunAttestation
    files: tuple[_FileSnapshot, ...] = field(repr=False)
    manifest_snapshot: bytes = field(repr=False)
    _binding_json: str = field(repr=False)

    @property
    def binding(self) -> Mapping[str, Any]:
        return json.loads(self._binding_json)

    @property
    def snapshots_by_filename(self) -> Mapping[str, bytes]:
        return {row.filename: row.snapshot for row in self.files}

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "index_path": str(self.index_path),
            "index_sha256": self.index_sha256,
            "cache_manifest_path": str(self.cache_manifest_path),
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "cache_key": self.cache_key,
            "binding_sha256": _sha256_json(self.binding),
            "files": [row.identity() for row in self.files],
            "run_attestation": self.run_attestation.identity(),
        }


def _cache_file_map(value: Any, *, expected: Sequence[str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ContextFitCacheAuthenticationError(
            "cache index entry must explicitly hash every canonical matrix file"
        )
    return {
        str(filename): _required_sha256(value[filename], label=f"cache_files.{filename}")
        for filename in expected
    }


def _authenticate_gate_manifest(
    raw: Mapping[str, Any], *, path: Path, file_hashes: Mapping[str, str]
) -> tuple[str, Mapping[str, Any], Mapping[str, str]]:
    if set(raw) != _GATE_MANIFEST_FIELDS:
        raise ContextFitCacheAuthenticationError("gate cache manifest has a wrong closed schema")
    if raw["schema_version"] != CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION:
        raise ContextFitCacheAuthenticationError("gate cache manifest schema is unsupported")
    content = {key: raw[key] for key in raw if key != "content_sha256"}
    if raw["content_sha256"] != _sha256_json(content):
        raise ContextFitCacheAuthenticationError("gate cache manifest content hash mismatch")
    cache_key = _required_sha256(raw["cache_key"], label="gate cache key")
    binding = raw["binding"]
    if not isinstance(binding, Mapping) or _sha256_json(binding) != cache_key:
        raise ContextFitCacheAuthenticationError("gate cache binding does not match its key")
    if path.name != "manifest.json" or path.parent.name != cache_key:
        raise ContextFitCacheAuthenticationError("gate cache manifest path is not canonical")
    declared = {
        str(raw["source_values_file"]): str(raw["source_values_sha256"]),
        str(raw["feature_values_file"]): str(raw["feature_values_sha256"]),
        str(raw["source_context_values_file"]): str(raw["source_context_values_sha256"]),
        str(raw["feature_context_values_file"]): str(raw["feature_context_values_sha256"]),
    }
    if set(declared) != set(_GATE_FILES):
        raise ContextFitCacheAuthenticationError("gate cache filenames are not canonical")
    if any(
        _required_sha256(value, label="gate matrix SHA-256") != file_hashes[name]
        for name, value in declared.items()
    ):
        raise ContextFitCacheAuthenticationError("gate index and manifest matrix hashes differ")
    return cache_key, binding, declared


def _authenticate_final_manifest(
    raw: Mapping[str, Any], *, path: Path, file_hashes: Mapping[str, str]
) -> tuple[str, Mapping[str, Any], Mapping[str, str]]:
    if set(raw) != _FINAL_MANIFEST_FIELDS:
        raise ContextFitCacheAuthenticationError("final cache manifest has a wrong closed schema")
    if raw["schema_version"] != FINAL_CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA:
        raise ContextFitCacheAuthenticationError("final cache manifest schema is unsupported")
    content = {key: raw[key] for key in raw if key != "content_sha256"}
    if raw["content_sha256"] != _sha256_json(content):
        raise ContextFitCacheAuthenticationError("final cache manifest content hash mismatch")
    cache_key = _required_sha256(raw["cache_key"], label="final cache key")
    binding = raw["binding"]
    if not isinstance(binding, Mapping) or _sha256_json(binding) != cache_key:
        raise ContextFitCacheAuthenticationError("final cache binding does not match its key")
    if path.name != "manifest.json" or path.parent.name != cache_key:
        raise ContextFitCacheAuthenticationError("final cache manifest path is not canonical")
    matrices = raw["matrix_files"]
    if not isinstance(matrices, Mapping) or set(matrices) != _FINAL_MATRIX_KEYS:
        raise ContextFitCacheAuthenticationError("final cache matrix records are malformed")
    declared: dict[str, str] = {}
    for record in matrices.values():
        if not isinstance(record, Mapping) or set(record) != {"filename", "sha256"}:
            raise ContextFitCacheAuthenticationError("final cache matrix record is malformed")
        filename = str(record["filename"])
        if filename in declared:
            raise ContextFitCacheAuthenticationError("final cache has duplicate matrix files")
        declared[filename] = _required_sha256(record["sha256"], label="final matrix SHA-256")
    if set(declared) != set(_FINAL_FILES) or any(
        declared[name] != file_hashes[name] for name in _FINAL_FILES
    ):
        raise ContextFitCacheAuthenticationError("final index and manifest matrix hashes differ")
    return cache_key, binding, declared


def _authenticate_index_entry(
    *,
    index_path: Path,
    index_sha256: str,
    entry: Mapping[str, Any],
    run_attestation_cache: dict[tuple[Path, str], _RunAttestation],
    snapshots: _SnapshotRegistry,
) -> AuthenticatedContextFitCacheSource:
    if set(entry) != _INDEX_ENTRY_FIELDS:
        raise ContextFitCacheAuthenticationError("cache index entry has a wrong closed schema")
    kind = str(entry["kind"])
    if kind not in _KINDS:
        raise ContextFitCacheAuthenticationError("cache index entry kind is unsupported")
    expected_files = _GATE_FILES if kind == "review_gate" else _FINAL_FILES
    file_hashes = _cache_file_map(entry["cache_files"], expected=expected_files)
    manifest_path = _resolve_index_path(
        index_path, entry["cache_manifest_path"], label="cache_manifest_path"
    )
    manifest_sha = _required_sha256(entry["cache_manifest_sha256"], label="cache_manifest_sha256")
    manifest_snapshot = snapshots.read(
        manifest_path, expected_sha256=manifest_sha, label=f"{kind} cache manifest"
    )
    manifest = _parse_json_snapshot(manifest_snapshot, label=f"{kind} cache manifest")
    if kind == "review_gate":
        cache_key, binding, declared_files = _authenticate_gate_manifest(
            manifest, path=manifest_path, file_hashes=file_hashes
        )
    else:
        cache_key, binding, declared_files = _authenticate_final_manifest(
            manifest, path=manifest_path, file_hashes=file_hashes
        )
    file_snapshots: list[_FileSnapshot] = []
    for filename in expected_files:
        snapshot = snapshots.read(
            manifest_path.parent / filename,
            expected_sha256=file_hashes[filename],
            label=f"{kind} cache matrix {filename}",
        )
        if not snapshot.startswith(b"\x93NUMPY"):
            raise ContextFitCacheAuthenticationError(
                f"{kind} cache matrix {filename} is not a NumPy array"
            )
        if declared_files[filename] != hashlib.sha256(snapshot).hexdigest():
            raise ContextFitCacheAuthenticationError(
                f"{kind} cache matrix {filename} changed during authentication"
            )
        file_snapshots.append(
            _FileSnapshot(filename=filename, sha256=file_hashes[filename], snapshot=snapshot)
        )
    run_path = _resolve_index_path(
        index_path, entry["run_manifest_path"], label="run_manifest_path"
    )
    run_sha = _required_sha256(entry["run_manifest_sha256"], label="run_manifest_sha256")
    run_key = (run_path, run_sha)
    run_attestation = run_attestation_cache.get(run_key)
    if run_attestation is None:
        run_snapshot = snapshots.read(
            run_path,
            expected_sha256=run_sha,
            label="companion immutable run-input manifest",
        )
        run_attestation = _authenticate_run_manifest(
            path=run_path, expected_sha256=run_sha, snapshot=run_snapshot
        )
        run_attestation_cache[run_key] = run_attestation
    if kind == "review_gate":
        if binding.get("provider_identity") != run_attestation.gate_provider_identity:
            raise ContextFitCacheAuthenticationError(
                "gate cache binding does not match its companion run provider identity"
            )
    elif binding.get("producer_identity_sha256") != _sha256_json(
        run_attestation.final_producer_identity
    ):
        raise ContextFitCacheAuthenticationError(
            "final cache binding does not match its companion run producer identity"
        )
    return AuthenticatedContextFitCacheSource(
        kind=kind,
        index_path=index_path,
        index_sha256=index_sha256,
        cache_manifest_path=manifest_path,
        cache_manifest_sha256=manifest_sha,
        cache_key=cache_key,
        run_attestation=run_attestation,
        files=tuple(file_snapshots),
        manifest_snapshot=manifest_snapshot,
        _binding_json=_canonical_json(binding),
    )


def authenticate_context_fit_cache_index_registrations(
    registrations: Sequence[str],
) -> tuple[AuthenticatedContextFitCacheSource, ...]:
    """Authenticate repeatable mandatory ``INDEX_PATH::SHA256`` registrations."""

    sources: list[AuthenticatedContextFitCacheSource] = []
    seen_indexes: set[Path] = set()
    seen_keys: set[tuple[str, str]] = set()
    run_attestation_cache: dict[tuple[Path, str], _RunAttestation] = {}
    snapshots = _SnapshotRegistry()
    for raw_registration in registrations:
        raw_path, separator, raw_digest = str(raw_registration).strip().rpartition("::")
        if not separator or not raw_path.strip() or not raw_digest.strip():
            raise ContextFitCacheAuthenticationError(
                "--read-only-context-fit-cache-index must use INDEX_PATH::SHA256"
            )
        index_sha = _required_sha256(raw_digest, label="registered cache index SHA-256")
        index_path = Path(raw_path).expanduser().resolve()
        if index_path in seen_indexes:
            raise ContextFitCacheAuthenticationError(f"duplicate cache index: {index_path}")
        seen_indexes.add(index_path)
        snapshot = snapshots.read(
            index_path, expected_sha256=index_sha, label="context-fit cache index"
        )
        raw = _parse_json_snapshot(snapshot, label="context-fit cache index")
        if (
            set(raw) != _INDEX_FIELDS
            or raw["schema_version"] != CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION
        ):
            raise ContextFitCacheAuthenticationError(
                "context-fit cache index schema is unsupported"
            )
        content = {key: raw[key] for key in raw if key != "content_sha256"}
        if raw["content_sha256"] != _sha256_json(content):
            raise ContextFitCacheAuthenticationError(
                "context-fit cache index content hash mismatch"
            )
        entries = raw["entries"]
        if not isinstance(entries, list) or not entries:
            raise ContextFitCacheAuthenticationError("context-fit cache index must be non-empty")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ContextFitCacheAuthenticationError(
                    "context-fit cache entry must be an object"
                )
            source = _authenticate_index_entry(
                index_path=index_path,
                index_sha256=index_sha,
                entry=entry,
                run_attestation_cache=run_attestation_cache,
                snapshots=snapshots,
            )
            key = (source.kind, source.cache_key)
            if key in seen_keys:
                raise ContextFitCacheAuthenticationError(
                    f"duplicate indexed {source.kind} cache key: {source.cache_key}"
                )
            seen_keys.add(key)
            sources.append(source)
    return tuple(sources)


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _materialize_bundle(*, source: AuthenticatedContextFitCacheSource, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError("fresh context-fit cache unexpectedly contains a hit target")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        snapshots = source.snapshots_by_filename
        for filename, snapshot in snapshots.items():
            path = temporary / filename
            with path.open("xb") as handle:
                handle.write(snapshot)
                handle.flush()
                os.fsync(handle.fileno())
        manifest = temporary / "manifest.json"
        with manifest.open("xb") as handle:
            handle.write(source.manifest_snapshot)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()


def _verify_materialized(source: AuthenticatedContextFitCacheSource, destination: Path) -> None:
    expected = {row.filename: row.sha256 for row in source.files}
    expected["manifest.json"] = source.cache_manifest_sha256
    for filename, digest in expected.items():
        try:
            current = (destination / filename).read_bytes()
        except OSError as exc:
            raise RuntimeError("materialized context-fit cache became unavailable") from exc
        if hashlib.sha256(current).hexdigest() != digest:
            raise RuntimeError("materialized context-fit cache bytes changed during the run")


class AuthenticatedContextFitGateCacheOverlay:
    """Exact-hit overlay for complete untouched-review-gate bundles."""

    def __init__(
        self,
        *,
        provider: ContextFitUpstreamGateProvider,
        runtime_producer: FinalContextFitUpstreamProducer,
        sources: Sequence[AuthenticatedContextFitCacheSource],
        output_root: Path | str,
    ) -> None:
        if type(provider) is not ContextFitUpstreamGateProvider:
            raise TypeError("gate cache overlay requires the exact current gate provider")
        if type(runtime_producer) is not FinalContextFitUpstreamProducer:
            raise TypeError("gate cache overlay requires the exact current final producer")
        self.provider = provider
        self.runtime_producer = runtime_producer
        self.sources = tuple(row for row in sources if row.kind == "review_gate")
        if not self.sources:
            raise ValueError("gate cache overlay requires at least one gate source")
        self.output_root = Path(output_root).resolve()
        self.cache_dir = Path(provider.cache_dir).resolve()
        if self.cache_dir.parent != self.output_root:
            raise ValueError("gate writable cache must be a direct child of fresh output")
        if self.cache_dir.exists() and any(self.cache_dir.iterdir()):
            raise ValueError("gate writable cache must be nonexistent or empty")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._provider_identity = json.loads(_canonical_json(provider.identity()))
        self._runtime_producer_identity = json.loads(_canonical_json(runtime_producer.identity()))
        if self._provider_identity.get("backend") != self._runtime_producer_identity.get(
            "backend_identity"
        ):
            raise ContextFitCacheAuthenticationError(
                "current gate provider and runtime producer do not share one backend identity"
            )
        by_key: dict[str, AuthenticatedContextFitCacheSource] = {}
        ineligible: list[AuthenticatedContextFitCacheSource] = []
        for source in self.sources:
            if (
                source.run_attestation.gate_provider_identity != self._provider_identity
                or source.run_attestation.final_producer_identity != self._runtime_producer_identity
            ):
                ineligible.append(source)
                continue
            by_key[source.cache_key] = source
        self._sources_by_key = by_key
        self._materialized: set[str] = set()
        self._lock = threading.Lock()
        self._code_sha256 = _module_sha256()
        self._identity = {
            "provider": CONTEXT_FIT_GATE_CACHE_OVERLAY_ID,
            "overlay_code_sha256": self._code_sha256,
            "delegate_provider_identity": copy.deepcopy(self._provider_identity),
            "delegate_provider_identity_sha256": _sha256_json(self._provider_identity),
            "runtime_producer_identity_sha256": _sha256_json(self._runtime_producer_identity),
            "read_only_sources": [source.identity() for source in self.sources],
            "eligible_source_cache_keys": sorted(by_key),
            "ineligible_identity_source_count": len(ineligible),
            "ineligible_identity_source_hashes": sorted(
                source.cache_manifest_sha256 for source in ineligible
            ),
            "historical_source_writes_allowed": False,
            "materialization_boundary": "bind_fold_after_proposal_and_quality_guard",
            "matrix_values_decoded_during_registration": False,
        }

    def _assert_current(self) -> None:
        if _module_sha256() != self._code_sha256:
            raise RuntimeError("gate cache overlay code changed after authentication")
        if json.loads(_canonical_json(self.provider.identity())) != self._provider_identity:
            raise RuntimeError("gate delegate identity changed after authentication")
        if json.loads(_canonical_json(self.runtime_producer.identity())) != (
            self._runtime_producer_identity
        ):
            raise RuntimeError("gate runtime attestation changed after authentication")

    def identity(self) -> Mapping[str, Any]:
        self._assert_current()
        return copy.deepcopy(self._identity)

    def bind_fold(
        self,
        *,
        outer_fold: int,
        context: ObservableCausalRows,
        context_texts: Sequence[str],
        gate_texts: Sequence[str],
        exact_gate_row_ids: Sequence[int],
    ):
        self._assert_current()
        fold = _gate_positive_int(outer_fold, name="outer_fold")
        if not isinstance(context, ObservableCausalRows):
            raise TypeError("context must be ObservableCausalRows")
        context_ids = _gate_integer_rows(context.row_ids, name="context.row_ids")
        if context.inner_fold_ids is None:
            raise ValueError("context.inner_fold_ids are required")
        folds = _gate_context_folds(context.inner_fold_ids, length=len(context_ids))
        gate_ids = _gate_integer_rows(exact_gate_row_ids, name="exact_gate_row_ids")
        if set(context_ids) & set(gate_ids):
            raise ValueError("review context and gate rows must be disjoint")
        context_exact = _gate_exact_texts(
            context_texts, name="context_texts", length=len(context_ids)
        )
        gate_exact = _gate_exact_texts(gate_texts, name="gate_texts", length=len(gate_ids))
        binding = self.provider._binding(
            outer_fold=fold,
            context=context,
            gate_row_ids=gate_ids,
            context_texts=context_exact,
            gate_texts=gate_exact,
            context_inner_fold_ids=folds,
        )
        cache_key = _sha256_json(binding)
        source = self._sources_by_key.get(cache_key)
        if source is not None:
            if source.binding != binding:
                raise ContextFitCacheAuthenticationError(
                    "gate cache hash matched but exact binding differed"
                )
            destination = self.cache_dir / cache_key
            with self._lock:
                if cache_key in self._materialized:
                    _verify_materialized(source, destination)
                else:
                    _materialize_bundle(source=source, destination=destination)
                    _verify_materialized(source, destination)
                    self._materialized.add(cache_key)
        return self.provider.bind_fold(
            outer_fold=outer_fold,
            context=context,
            context_texts=context_texts,
            gate_texts=gate_texts,
            exact_gate_row_ids=exact_gate_row_ids,
        )


class AuthenticatedFinalContextFitCacheOverlay:
    """Exact-hit overlay for complete post-freeze final upstream bundles."""

    def __init__(
        self,
        *,
        producer: FinalContextFitUpstreamProducer,
        sources: Sequence[AuthenticatedContextFitCacheSource],
        output_root: Path | str,
    ) -> None:
        if type(producer) is not FinalContextFitUpstreamProducer:
            raise TypeError("final cache overlay requires the exact current final producer")
        self.producer = producer
        self.sources = tuple(row for row in sources if row.kind == "final_upstream")
        if not self.sources:
            raise ValueError("final cache overlay requires at least one final source")
        self.output_root = Path(output_root).resolve()
        self.cache_dir = Path(producer.cache_dir).resolve()
        if self.cache_dir.parent != self.output_root:
            raise ValueError("final writable cache must be a direct child of fresh output")
        if self.cache_dir.exists() and any(self.cache_dir.iterdir()):
            raise ValueError("final writable cache must be nonexistent or empty")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._producer_identity = json.loads(_canonical_json(producer.identity()))
        self._producer_identity_sha256 = _sha256_json(self._producer_identity)
        by_key: dict[str, AuthenticatedContextFitCacheSource] = {}
        ineligible: list[AuthenticatedContextFitCacheSource] = []
        for source in self.sources:
            if source.run_attestation.final_producer_identity != self._producer_identity:
                ineligible.append(source)
                continue
            by_key[source.cache_key] = source
        self._sources_by_key = by_key
        self._materialized: set[str] = set()
        self._lock = threading.Lock()
        self._code_sha256 = _module_sha256()
        self._identity = {
            "producer": FINAL_CONTEXT_FIT_CACHE_OVERLAY_ID,
            "overlay_code_sha256": self._code_sha256,
            "delegate_producer_identity": copy.deepcopy(self._producer_identity),
            "delegate_producer_identity_sha256": self._producer_identity_sha256,
            "package_producer_identity_sha256": self._producer_identity_sha256,
            "read_only_sources": [source.identity() for source in self.sources],
            "eligible_source_cache_keys": sorted(by_key),
            "ineligible_identity_source_count": len(ineligible),
            "ineligible_identity_source_hashes": sorted(
                source.cache_manifest_sha256 for source in ineligible
            ),
            "historical_source_writes_allowed": False,
            "materialization_boundary": "produce_after_registry_freeze",
            "matrix_values_decoded_during_registration": False,
        }

    def _assert_current(self) -> None:
        if _module_sha256() != self._code_sha256:
            raise RuntimeError("final cache overlay code changed after authentication")
        if json.loads(_canonical_json(self.producer.identity())) != self._producer_identity:
            raise RuntimeError("final delegate identity changed after authentication")

    def identity(self) -> Mapping[str, Any]:
        self._assert_current()
        return copy.deepcopy(self._identity)

    def authenticated_package_producer_identity_sha256(self) -> str:
        self._assert_current()
        return self._producer_identity_sha256

    def produce(
        self,
        *,
        outer_fold: int,
        outer_train_row_ids: Sequence[Any],
        outer_train_texts: Sequence[Any],
        outer_train_treatment: Sequence[Any],
        outer_train_outcome: Sequence[Any],
        outer_heldout_row_ids: Sequence[Any],
        outer_heldout_texts: Sequence[Any],
        meta_inner_fold_ids: Sequence[Any],
    ):
        self._assert_current()
        fold = _final_positive_int(outer_fold, name="outer_fold")
        train_rows = _final_integer_rows(outer_train_row_ids, name="outer_train_row_ids")
        heldout_rows = _final_integer_rows(outer_heldout_row_ids, name="outer_heldout_row_ids")
        if set(train_rows) & set(heldout_rows):
            raise ValueError("outer train and outer heldout rows must be disjoint")
        train_texts = _final_exact_texts(
            outer_train_texts, name="outer_train_texts", length=len(train_rows)
        )
        heldout_texts = _final_exact_texts(
            outer_heldout_texts, name="outer_heldout_texts", length=len(heldout_rows)
        )
        treatment = _final_finite_vector(
            outer_train_treatment, name="outer_train_treatment", length=len(train_rows)
        )
        if set(np.unique(treatment).tolist()) != {0.0, 1.0}:
            raise ValueError("outer_train_treatment must contain binary 0/1 values")
        outcome = _final_finite_vector(
            outer_train_outcome, name="outer_train_outcome", length=len(train_rows)
        )
        folds = _final_fold_ids(meta_inner_fold_ids, length=len(train_rows))
        _expected_fit_rows_by_position(train_rows, folds)
        binding = self.producer._binding(
            outer_fold=fold,
            train_row_ids=train_rows,
            train_texts=train_texts,
            train_treatment=treatment,
            train_outcome=outcome,
            heldout_row_ids=heldout_rows,
            heldout_texts=heldout_texts,
            meta_inner_fold_ids=folds,
        )
        cache_key = _sha256_json(binding)
        source = self._sources_by_key.get(cache_key)
        if source is not None:
            if source.binding != binding:
                raise ContextFitCacheAuthenticationError(
                    "final cache hash matched but exact binding differed"
                )
            destination = self.cache_dir / "artifacts" / cache_key
            with self._lock:
                if cache_key in self._materialized:
                    _verify_materialized(source, destination)
                else:
                    _materialize_bundle(source=source, destination=destination)
                    _verify_materialized(source, destination)
                    self._materialized.add(cache_key)
        return self.producer.produce(
            outer_fold=outer_fold,
            outer_train_row_ids=outer_train_row_ids,
            outer_train_texts=outer_train_texts,
            outer_train_treatment=outer_train_treatment,
            outer_train_outcome=outer_train_outcome,
            outer_heldout_row_ids=outer_heldout_row_ids,
            outer_heldout_texts=outer_heldout_texts,
            meta_inner_fold_ids=meta_inner_fold_ids,
        )


__all__ = [
    "AuthenticatedContextFitCacheSource",
    "AuthenticatedContextFitGateCacheOverlay",
    "AuthenticatedFinalContextFitCacheOverlay",
    "CONTEXT_FIT_CACHE_INDEX_SCHEMA_VERSION",
    "CONTEXT_FIT_GATE_CACHE_OVERLAY_ID",
    "ContextFitCacheAuthenticationError",
    "FINAL_CONTEXT_FIT_CACHE_OVERLAY_ID",
    "authenticate_context_fit_cache_index_registrations",
]
