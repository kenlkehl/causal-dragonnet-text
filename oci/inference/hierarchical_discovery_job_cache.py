"""Authenticated immutable cache for validated hierarchical discovery jobs.

The cache deliberately sits *after* the hierarchy's semantic validators.  A
remote response is persisted only after validation succeeds, and a replayed
response is run through the same validator again before it is returned.  Each
entry is content addressed by the exact job, runner identity, hierarchy
precommit, validator implementation, cache schema/code identity, and validated
response.

The cache root is a machine-local envelope.  Its absolute path is nevertheless
part of :meth:`AuthenticatedHierarchicalDiscoveryJobCache.identity`, allowing
the approval wrapper to bind the exact envelope before any lookup occurs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .all_evidence_discovery_interfaces import canonical_json, content_sha256

HIERARCHICAL_DISCOVERY_JOB_CACHE_VERSION = "authenticated_hierarchical_discovery_job_cache_v3"
HIERARCHICAL_DISCOVERY_JOB_CACHE_IDENTITY_VERSION = (
    "authenticated_hierarchical_discovery_job_cache_identity_v3"
)
HIERARCHICAL_DISCOVERY_JOB_CACHE_LOOKUP_VERSION = (
    "authenticated_hierarchical_discovery_job_cache_lookup_v1"
)
HIERARCHICAL_DISCOVERY_JOB_CACHE_ENTRY_VERSION = (
    "authenticated_hierarchical_discovery_job_cache_entry_v3"
)
HIERARCHICAL_DISCOVERY_JOB_CACHE_HIT_VERSION = (
    "authenticated_hierarchical_discovery_job_cache_hit_v3"
)
HIERARCHICAL_DISCOVERY_CACHE_RESPONSE_TRACE_VERSION = (
    "authenticated_cache_response_attempt_trace_v2"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID = re.compile(r"job_[0-9a-f]{64}\Z")
_ENTRY_NAME = re.compile(r"entry_([0-9a-f]{64})\.json\Z")
_ENTRY_KEYS = frozenset(
    {
        "schema_version",
        "lookup_identity",
        "lookup_sha256",
        "wire_response",
        "wire_response_sha256",
        "validated_response",
        "validated_response_sha256",
        "response_attempt_trace",
        "response_attempt_trace_sha256",
        "entry_sha256",
    }
)
_LOOKUP_KEYS = frozenset(
    {
        "schema_version",
        "cache_identity_sha256",
        "hierarchy_inner_precommit_sha256",
        "runner_identity",
        "runner_identity_sha256",
        "validator_code_sha256",
        "job",
        "job_id",
    }
)
_CACHE_HIT_KEYS = frozenset(
    {
        "schema_version",
        "record_type",
        "job_id",
        "job_kind",
        "job_sha256",
        "runner_identity_sha256",
        "hierarchy_inner_precommit_sha256",
        "validator_code_sha256",
        "cache_identity_sha256",
        "cache_lookup_sha256",
        "cache_entry_sha256",
        "wire_response",
        "wire_response_sha256",
        "validated_response_sha256",
        "response_attempt_trace_sha256",
        "outcome",
        "record_sha256",
    }
)


def _clone(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"cached JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> None:
    raise ValueError(f"cached JSON contains non-finite constant {token!r}")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("cached JSON contains a non-finite number")
    return value


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("cached entry is not valid UTF-8") from exc
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_finite_float,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("cached entry is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise TypeError("cached entry must be one JSON object")
    canonical = canonical_json(parsed).encode("utf-8")
    if raw != canonical:
        raise ValueError("cached entry bytes differ from their canonical authenticated form")
    return parsed


def _closed_mapping(
    value: Any,
    *,
    keys: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one JSON object")
    if set(value) != keys:
        raise ValueError(f"{label} has an unexpected closed schema")
    return value


def _implementation_file_sha256() -> str:
    return _sha256_bytes(Path(__file__).read_bytes())


def _validate_response_trace_bindings(
    trace: Any,
    *,
    job: Mapping[str, Any],
    wire_response_sha256: str,
    validated_response_sha256: str,
) -> Mapping[str, Any]:
    """Bind both the received wire object and its normalized projection."""

    if not isinstance(trace, Mapping):
        raise TypeError("cached response_attempt_trace must be one JSON object")
    if trace.get("logical_job_id") != job.get("job_id"):
        raise ValueError("cached response-attempt trace cites another logical job")
    trace_sha256 = _require_sha256(
        trace.get("trace_sha256"),
        label="cached response-attempt trace_sha256",
    )
    trace_body = {key: value for key, value in trace.items() if key != "trace_sha256"}
    if trace_sha256 != content_sha256(trace_body):
        raise ValueError("cached response-attempt trace_sha256 changed")

    if trace.get("mode") == "single_validated_response":
        row = _closed_mapping(
            trace,
            keys=frozenset(
                {
                    "schema_version",
                    "mode",
                    "logical_job_id",
                    "job_sha256",
                    "wire_response_sha256",
                    "validated_response_sha256",
                    "trace_sha256",
                }
            ),
            label="single-response cache trace",
        )
        if row["schema_version"] != HIERARCHICAL_DISCOVERY_CACHE_RESPONSE_TRACE_VERSION:
            raise ValueError("single-response cache trace has the wrong schema version")
        if row["job_sha256"] != content_sha256(job):
            raise ValueError("single-response cache trace cites another job")
        if row["wire_response_sha256"] != wire_response_sha256:
            raise ValueError("cached wire response differs from its cache trace")
        if row["validated_response_sha256"] != validated_response_sha256:
            raise ValueError("cached normalized response differs from its cache trace")
        return row

    attempts = trace.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("cached response-attempt trace has no recognized authenticated mode")
    final_attempt = attempts[-1]
    if not isinstance(final_attempt, Mapping):
        raise TypeError("cached final response attempt must be one JSON object")
    if final_attempt.get("raw_response_projection_sha256") != wire_response_sha256:
        raise ValueError("cached wire response differs from its response-attempt trace")
    if final_attempt.get("normalized_validated_response_sha256") != validated_response_sha256:
        raise ValueError("cached validated response differs from its response-attempt trace")
    return trace


def _normalized_runner_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    identity = _clone(value)
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("runner identity must be one non-empty JSON object")
    declared = _require_sha256(
        identity.get("identity_sha256"),
        label="runner identity_sha256",
    )
    body = {key: row for key, row in identity.items() if key != "identity_sha256"}
    if declared != content_sha256(body):
        raise ValueError("runner identity_sha256 does not authenticate runner identity")
    return dict(identity)


def _job_dictionary(job: Any) -> dict[str, Any]:
    if not callable(getattr(job, "as_dict", None)):
        raise TypeError("cache job must be a DiscoveryJsonJob")
    post_init = getattr(job, "__post_init__", None)
    if callable(post_init):
        post_init()
    job_id = getattr(job, "job_id", None)
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise ValueError("cache job_id must identify one DiscoveryJsonJob")
    value = _clone(job.as_dict())
    if not isinstance(value, Mapping) or value.get("job_id") != job_id:
        raise ValueError("cache job dictionary differs from its job_id")
    return dict(value)


@dataclass(frozen=True)
class HierarchicalDiscoveryJobCacheConfig:
    """Closed immutable-cache policy bound into offline approval."""

    max_entry_bytes: int = 32_000_000
    file_mode: int = 0o600
    directory_mode: int = 0o700

    def __post_init__(self) -> None:
        if isinstance(self.max_entry_bytes, bool) or not isinstance(self.max_entry_bytes, int):
            raise TypeError("max_entry_bytes must be an integer")
        if self.max_entry_bytes < 1:
            raise ValueError("max_entry_bytes must be positive")
        for label, value in (
            ("file_mode", self.file_mode),
            ("directory_mode", self.directory_mode),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an integer")
            if value < 0 or value > 0o777:
                raise ValueError(f"{label} must be a permission mode from 0 through 0o777")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_entry_bytes": self.max_entry_bytes,
            "file_mode": self.file_mode,
            "directory_mode": self.directory_mode,
            "write_policy": "exclusive_create_never_overwrite",
            "replay_policy": "strict_bytes_then_same_semantic_validator",
            "symlink_policy": "reject_cache_root_namespace_and_entry_symlinks",
        }


@dataclass(frozen=True)
class AuthenticatedCacheReplay:
    """One successfully revalidated cache hit."""

    wire_response: dict[str, Any]
    validated_response: dict[str, Any]
    response_attempt_trace: dict[str, Any]
    execution_metadata: dict[str, Any]


class AuthenticatedHierarchicalDiscoveryJobCache:
    """Immutable one-entry namespace for each exact hierarchy job context."""

    def __init__(
        self,
        *,
        root: str | os.PathLike[str],
        config: HierarchicalDiscoveryJobCacheConfig | None = None,
    ) -> None:
        raw_root = os.fspath(root)
        if not isinstance(raw_root, str) or not raw_root:
            raise ValueError("cache root must be one non-empty path")
        self.root = Path(os.path.abspath(raw_root))
        self.config = config or HierarchicalDiscoveryJobCacheConfig()
        if not isinstance(self.config, HierarchicalDiscoveryJobCacheConfig):
            raise TypeError("config must be HierarchicalDiscoveryJobCacheConfig")
        self._active_inner_precommit_sha256: str | None = None
        self._active_runner_identity_json: str | None = None
        self._execution_metadata: list[dict[str, Any]] = []

    def identity(self) -> dict[str, Any]:
        body = {
            "schema_version": HIERARCHICAL_DISCOVERY_JOB_CACHE_IDENTITY_VERSION,
            "cache_version": HIERARCHICAL_DISCOVERY_JOB_CACHE_VERSION,
            "mode": "read_write_immutable",
            "root_envelope": {
                "kind": "machine_local_absolute_path",
                "absolute_path": str(self.root),
            },
            "config": self.config.as_dict(),
            "implementation_file_sha256": _implementation_file_sha256(),
            "entry_schema_version": HIERARCHICAL_DISCOVERY_JOB_CACHE_ENTRY_VERSION,
            "hit_metadata_schema_version": HIERARCHICAL_DISCOVERY_JOB_CACHE_HIT_VERSION,
            "response_attempt_trace_policy": (
                "raw_wire_and_normalized_validated_hashes_with_exact_repair_sequence_v2"
            ),
        }
        return {**body, "identity_sha256": content_sha256(body)}

    @property
    def execution_metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(_clone(row) for row in self._execution_metadata)

    def begin_execution(
        self,
        *,
        hierarchy_inner_precommit_sha256: str,
        runner_identity: Mapping[str, Any],
    ) -> None:
        """Begin one approved execution without looking up any job entry."""

        inner_precommit_sha256 = _require_sha256(
            hierarchy_inner_precommit_sha256,
            label="hierarchy_inner_precommit_sha256",
        )
        identity = _normalized_runner_identity(runner_identity)
        self._active_inner_precommit_sha256 = inner_precommit_sha256
        self._active_runner_identity_json = canonical_json(identity)
        self._execution_metadata = []

    def _assert_active(
        self,
        *,
        hierarchy_inner_precommit_sha256: str,
        runner_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        inner = _require_sha256(
            hierarchy_inner_precommit_sha256,
            label="hierarchy_inner_precommit_sha256",
        )
        identity = _normalized_runner_identity(runner_identity)
        if self._active_inner_precommit_sha256 != inner:
            raise ValueError("cache execution context has a different hierarchy precommit")
        if self._active_runner_identity_json != canonical_json(identity):
            raise ValueError("cache execution context has a different runner identity")
        return identity

    def _lookup_identity(
        self,
        *,
        job: Any,
        hierarchy_inner_precommit_sha256: str,
        runner_identity: Mapping[str, Any],
        validator_code_sha256: str,
    ) -> dict[str, Any]:
        identity = self._assert_active(
            hierarchy_inner_precommit_sha256=hierarchy_inner_precommit_sha256,
            runner_identity=runner_identity,
        )
        validator_sha = _require_sha256(
            validator_code_sha256,
            label="validator_code_sha256",
        )
        job_dict = _job_dictionary(job)
        return {
            "schema_version": HIERARCHICAL_DISCOVERY_JOB_CACHE_LOOKUP_VERSION,
            "cache_identity_sha256": self.identity()["identity_sha256"],
            "hierarchy_inner_precommit_sha256": hierarchy_inner_precommit_sha256,
            "runner_identity": identity,
            "runner_identity_sha256": identity["identity_sha256"],
            "validator_code_sha256": validator_sha,
            "job": job_dict,
            "job_id": job_dict["job_id"],
        }

    def _secure_root(self, *, create: bool) -> None:
        try:
            root_stat = self.root.lstat()
        except FileNotFoundError:
            if not create:
                return
            self.root.mkdir(parents=True, mode=self.config.directory_mode, exist_ok=True)
            root_stat = self.root.lstat()
        if stat.S_ISLNK(root_stat.st_mode):
            raise ValueError("cache root cannot be a symlink")
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError("cache root must be a directory")
        current = Path(self.root.anchor)
        for part in self.root.parts[1:]:
            current /= part
            try:
                component_stat = current.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(component_stat.st_mode):
                raise ValueError("cache root path cannot traverse a symlink")

    def _namespace(self, lookup_sha256: str) -> Path:
        _require_sha256(lookup_sha256, label="cache lookup_sha256")
        return self.root / lookup_sha256

    def _entry_paths(self, *, namespace: Path) -> tuple[Path, ...]:
        try:
            namespace_stat = namespace.lstat()
        except FileNotFoundError:
            return ()
        if stat.S_ISLNK(namespace_stat.st_mode):
            raise ValueError("cache namespace cannot be a symlink")
        if not stat.S_ISDIR(namespace_stat.st_mode):
            raise ValueError("cache namespace must be a directory")
        rows: list[Path] = []
        with os.scandir(namespace) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError("cache entry cannot be a symlink")
                if _ENTRY_NAME.fullmatch(entry.name) is None or not entry.is_file(
                    follow_symlinks=False
                ):
                    raise ValueError("cache namespace contains an unexpected entry")
                rows.append(namespace / entry.name)
        if len(rows) > 1:
            raise ValueError("immutable cache namespace contains multiple entries")
        return tuple(rows)

    def _read_entry(
        self,
        *,
        path: Path,
        expected_lookup: Mapping[str, Any],
    ) -> dict[str, Any]:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("cache entry must be a regular file")
            if metadata.st_size < 1 or metadata.st_size > self.config.max_entry_bytes:
                raise ValueError("cache entry violates its authenticated byte bound")
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1_048_576))
                if not chunk:
                    raise ValueError("cache entry was truncated during authentication")
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise ValueError("cache entry grew during authentication")
        finally:
            os.close(descriptor)
        raw = b"".join(chunks)
        entry = _strict_json_object(raw)
        _closed_mapping(entry, keys=_ENTRY_KEYS, label="cache entry")
        lookup = _closed_mapping(
            entry["lookup_identity"],
            keys=_LOOKUP_KEYS,
            label="cache lookup identity",
        )
        if canonical_json(lookup) != canonical_json(expected_lookup):
            raise ValueError("cached entry identity differs from the exact requested job context")
        lookup_sha256 = _require_sha256(
            entry["lookup_sha256"],
            label="cache entry lookup_sha256",
        )
        if lookup_sha256 != content_sha256(expected_lookup):
            raise ValueError("cached lookup_sha256 does not authenticate its identity")
        wire_response = entry["wire_response"]
        if not isinstance(wire_response, Mapping):
            raise TypeError("cached wire_response must be one JSON object")
        wire_response_sha256 = _require_sha256(
            entry["wire_response_sha256"],
            label="cached wire_response_sha256",
        )
        if wire_response_sha256 != content_sha256(wire_response):
            raise ValueError("cached wire_response_sha256 changed")
        response = entry["validated_response"]
        if not isinstance(response, Mapping):
            raise TypeError("cached validated_response must be one JSON object")
        response_sha256 = _require_sha256(
            entry["validated_response_sha256"],
            label="cached validated_response_sha256",
        )
        if response_sha256 != content_sha256(response):
            raise ValueError("cached validated_response_sha256 changed")
        response_attempt_trace = entry["response_attempt_trace"]
        if not isinstance(response_attempt_trace, Mapping):
            raise TypeError("cached response_attempt_trace must be one JSON object")
        response_attempt_trace_sha256 = _require_sha256(
            entry["response_attempt_trace_sha256"],
            label="cached response_attempt_trace_sha256",
        )
        if response_attempt_trace_sha256 != content_sha256(response_attempt_trace):
            raise ValueError("cached response_attempt_trace_sha256 changed")
        _validate_response_trace_bindings(
            response_attempt_trace,
            job=lookup["job"],
            wire_response_sha256=wire_response_sha256,
            validated_response_sha256=response_sha256,
        )
        entry_sha256 = _require_sha256(
            entry["entry_sha256"],
            label="cache entry_sha256",
        )
        body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        if entry_sha256 != content_sha256(body):
            raise ValueError("cache entry_sha256 does not authenticate the entry")
        matched_name = _ENTRY_NAME.fullmatch(path.name)
        if matched_name is None or matched_name.group(1) != entry_sha256:
            raise ValueError("cache entry filename does not authenticate its response")
        return entry

    def replay_validated(
        self,
        *,
        job: Any,
        hierarchy_inner_precommit_sha256: str,
        runner_identity: Mapping[str, Any],
        validator_code_sha256: str,
        validator: Callable[[Any], Mapping[str, Any]],
    ) -> AuthenticatedCacheReplay | None:
        """Return one same-validator replay, or ``None`` on a genuine miss."""

        if not callable(validator):
            raise TypeError("cache validator must be callable")
        lookup = self._lookup_identity(
            job=job,
            hierarchy_inner_precommit_sha256=hierarchy_inner_precommit_sha256,
            runner_identity=runner_identity,
            validator_code_sha256=validator_code_sha256,
        )
        lookup_sha256 = content_sha256(lookup)
        self._secure_root(create=False)
        paths = self._entry_paths(namespace=self._namespace(lookup_sha256))
        if not paths:
            return None
        entry = self._read_entry(path=paths[0], expected_lookup=lookup)
        validated = validator(_clone(entry["wire_response"]))
        if not isinstance(validated, Mapping):
            raise TypeError("job validator must return one JSON object on cache replay")
        detached = _clone(validated)
        if canonical_json(detached) != canonical_json(entry["validated_response"]):
            raise ValueError("same semantic validator changed the cached validated response")
        job_dict = lookup["job"]
        record_body = {
            "schema_version": HIERARCHICAL_DISCOVERY_JOB_CACHE_HIT_VERSION,
            "record_type": "authenticated_cache_hit",
            "job_id": job_dict["job_id"],
            "job_kind": job_dict["job_kind"],
            "job_sha256": content_sha256(job_dict),
            "runner_identity_sha256": lookup["runner_identity_sha256"],
            "hierarchy_inner_precommit_sha256": hierarchy_inner_precommit_sha256,
            "validator_code_sha256": validator_code_sha256,
            "cache_identity_sha256": lookup["cache_identity_sha256"],
            "cache_lookup_sha256": lookup_sha256,
            "cache_entry_sha256": entry["entry_sha256"],
            "wire_response": _clone(entry["wire_response"]),
            "wire_response_sha256": entry["wire_response_sha256"],
            "validated_response_sha256": entry["validated_response_sha256"],
            "response_attempt_trace_sha256": entry["response_attempt_trace_sha256"],
            "outcome": "cache_hit",
        }
        metadata = {**record_body, "record_sha256": content_sha256(record_body)}
        _closed_mapping(metadata, keys=_CACHE_HIT_KEYS, label="cache-hit metadata")
        self._execution_metadata.append(_clone(metadata))
        return AuthenticatedCacheReplay(
            wire_response=_clone(entry["wire_response"]),
            validated_response=detached,
            response_attempt_trace=_clone(entry["response_attempt_trace"]),
            execution_metadata=_clone(metadata),
        )

    def store_validated(
        self,
        *,
        job: Any,
        hierarchy_inner_precommit_sha256: str,
        runner_identity: Mapping[str, Any],
        validator_code_sha256: str,
        wire_response: Mapping[str, Any] | None = None,
        validated_response: Mapping[str, Any],
        response_attempt_trace: Mapping[str, Any] | None = None,
    ) -> str:
        """Exclusively create an immutable entry after semantic validation."""

        response = _clone(validated_response)
        if not isinstance(response, Mapping):
            raise TypeError("validated_response must be one JSON object")
        wire = _clone(response if wire_response is None else wire_response)
        if not isinstance(wire, Mapping):
            raise TypeError("wire_response must be one JSON object")
        if response_attempt_trace is None:
            job_dict = _job_dictionary(job)
            trace_body = {
                "schema_version": HIERARCHICAL_DISCOVERY_CACHE_RESPONSE_TRACE_VERSION,
                "mode": "single_validated_response",
                "logical_job_id": job_dict["job_id"],
                "job_sha256": content_sha256(job_dict),
                "wire_response_sha256": content_sha256(wire),
                "validated_response_sha256": content_sha256(response),
            }
            trace = {**trace_body, "trace_sha256": content_sha256(trace_body)}
        else:
            trace = _clone(response_attempt_trace)
            if not isinstance(trace, Mapping) or not trace:
                raise TypeError("response_attempt_trace must be one non-empty JSON object")
        lookup = self._lookup_identity(
            job=job,
            hierarchy_inner_precommit_sha256=hierarchy_inner_precommit_sha256,
            runner_identity=runner_identity,
            validator_code_sha256=validator_code_sha256,
        )
        _validate_response_trace_bindings(
            trace,
            job=lookup["job"],
            wire_response_sha256=content_sha256(wire),
            validated_response_sha256=content_sha256(response),
        )
        lookup_sha256 = content_sha256(lookup)
        body = {
            "schema_version": HIERARCHICAL_DISCOVERY_JOB_CACHE_ENTRY_VERSION,
            "lookup_identity": lookup,
            "lookup_sha256": lookup_sha256,
            "wire_response": wire,
            "wire_response_sha256": content_sha256(wire),
            "validated_response": response,
            "validated_response_sha256": content_sha256(response),
            "response_attempt_trace": trace,
            "response_attempt_trace_sha256": content_sha256(trace),
        }
        entry = {**body, "entry_sha256": content_sha256(body)}
        encoded = canonical_json(entry).encode("utf-8")
        if len(encoded) > self.config.max_entry_bytes:
            raise ValueError("validated cache entry exceeds its authenticated byte bound")

        self._secure_root(create=True)
        namespace = self._namespace(lookup_sha256)
        try:
            namespace.mkdir(mode=self.config.directory_mode)
        except FileExistsError:
            pass
        existing = self._entry_paths(namespace=namespace)
        if existing:
            observed = self._read_entry(path=existing[0], expected_lookup=lookup)
            if canonical_json(observed) != canonical_json(entry):
                raise ValueError("immutable cache entry differs from the validated response")
            return entry["entry_sha256"]

        path = namespace / f"entry_{entry['entry_sha256']}.json"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags, self.config.file_mode)
            offset = 0
            while offset < len(encoded):
                offset += os.write(descriptor, encoded[offset:])
            os.fsync(descriptor)
        except FileExistsError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
        observed_paths = self._entry_paths(namespace=namespace)
        if len(observed_paths) != 1:
            raise ValueError("immutable cache create did not yield exactly one entry")
        observed = self._read_entry(path=observed_paths[0], expected_lookup=lookup)
        if canonical_json(observed) != canonical_json(entry):
            raise ValueError("immutable cache replay differs from the validated response")
        return entry["entry_sha256"]


__all__ = [
    "HIERARCHICAL_DISCOVERY_CACHE_RESPONSE_TRACE_VERSION",
    "HIERARCHICAL_DISCOVERY_JOB_CACHE_ENTRY_VERSION",
    "HIERARCHICAL_DISCOVERY_JOB_CACHE_HIT_VERSION",
    "HIERARCHICAL_DISCOVERY_JOB_CACHE_IDENTITY_VERSION",
    "HIERARCHICAL_DISCOVERY_JOB_CACHE_LOOKUP_VERSION",
    "HIERARCHICAL_DISCOVERY_JOB_CACHE_VERSION",
    "AuthenticatedCacheReplay",
    "AuthenticatedHierarchicalDiscoveryJobCache",
    "HierarchicalDiscoveryJobCacheConfig",
]
