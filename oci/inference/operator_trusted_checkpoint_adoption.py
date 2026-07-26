"""Explicit transitive checkpoint trust without rereading large payload bytes.

This module is an operational exception path.  It never changes the ordinary
portable-artifact validator or its full-byte adoption policy.  An operator may
instead name an immutable v3 adoption attestation that proves the exact
artifact was fully authenticated by an earlier request.  We authenticate the
small controls again, require filesystem-stat continuity since that earlier
attestation, and issue a new request-bound record that clearly says the payload
bytes were *not* reauthenticated.

The resulting handle is appropriate for a trusted research-tool process.  It
cannot satisfy a fresh-byte terminal audit or global release certification.
"""

from __future__ import annotations

import calendar
import copy
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import portable_artifacts as _portable
from .portable_identity import identity_sha256


OPERATOR_TRUSTED_ADOPTION_ATTESTATION = (
    "portable_checkpoint_operator_trusted_transitive_adoption_attestation_v1"
)
OPERATOR_TRUSTED_VALIDATION_POLICY = (
    "operator_explicit_prior_full_byte_attestation_and_stat_continuity_"
    "no_payload_read_v1"
)
_PRIOR_FULL_BYTE_POLICY = (
    "fresh_full_byte_and_exact_control_inventory_no_force_v3"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIOR_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "producer_artifact_id",
        "producer_artifact_kind",
        "producer_compatibility_key",
        "producer_content_root",
        "producer_locator",
        "producer_manifest_sha256",
        "producer_manifest_size_bytes",
        "producer_locator_sha256",
        "producer_locator_size_bytes",
        "producer_operational_phase_binding_content_sha256",
        "consumer_request_sha256",
        "validated_upstream_artifact_ids",
        "validation_policy",
        "content_sha256",
        "recorded_at",
    }
)


@dataclass(frozen=True)
class OperatorTrustedCheckpoint:
    """Stat-guarded artifact handle derived from one prior full-byte audit."""

    artifact: _portable.ValidatedPortableArtifact
    prior_attestation_path: Path
    prior_attestation_sha256: str
    prior_attestation_size_bytes: int
    prior_attestation_content_sha256: str
    prior_consumer_request_sha256: str
    prior_recorded_at: str
    payload_stat_inventory: tuple[Mapping[str, Any], ...]


def _timestamp_ns(value: Any) -> int:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("prior adoption attestation recorded_at is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "prior adoption attestation recorded_at is invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("prior adoption attestation recorded_at must be timezone-aware")
    utc = parsed.astimezone(timezone.utc)
    return (
        calendar.timegm(utc.utctimetuple()) * 1_000_000_000
        + utc.microsecond * 1_000
    )


def _validate_prior_attestation(
    path: Path,
) -> tuple[dict[str, Any], str, int, int, int]:
    source = Path(path)
    if source.is_symlink():
        raise ValueError("prior adoption attestation cannot be a symlink")
    source = source.resolve(strict=True)
    digest, size, identity = _portable._safe_file_hash_with_identity(
        source,
        label="prior full-byte checkpoint adoption attestation",
    )
    value = _portable._strict_json_bytes(
        _portable._safe_read(
            source,
            label="prior full-byte checkpoint adoption attestation",
        ),
        label="prior full-byte checkpoint adoption attestation",
    )
    body = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "content_sha256"
    }
    upstream = value.get("validated_upstream_artifact_ids")
    phase_binding = value.get(
        "producer_operational_phase_binding_content_sha256"
    )
    if (
        set(value) != _PRIOR_REQUIRED_FIELDS
        or value.get("schema_version")
        != _portable.PORTABLE_ADOPTION_ATTESTATION
        or value.get("validation_policy") != _PRIOR_FULL_BYTE_POLICY
        or value.get("content_sha256") != identity_sha256(body)
        or any(
            _SHA256.fullmatch(str(value.get(field, ""))) is None
            for field in (
                "producer_artifact_id",
                "producer_compatibility_key",
                "producer_content_root",
                "producer_manifest_sha256",
                "producer_locator_sha256",
                "consumer_request_sha256",
                "content_sha256",
            )
        )
        or value.get("producer_artifact_id")
        != value.get("producer_content_root")
        or value.get("producer_artifact_kind")
        not in _portable.CHECKPOINT_ARTIFACT_KINDS
        or not isinstance(value.get("producer_locator"), str)
        or not Path(str(value["producer_locator"])).is_absolute()
        or not isinstance(value.get("producer_manifest_size_bytes"), int)
        or int(value["producer_manifest_size_bytes"]) <= 0
        or not isinstance(value.get("producer_locator_size_bytes"), int)
        or int(value["producer_locator_size_bytes"]) <= 0
        or (
            phase_binding is not None
            and _SHA256.fullmatch(str(phase_binding)) is None
        )
        or not isinstance(upstream, list)
        or len(upstream) != len(set(map(str, upstream)))
        or any(_SHA256.fullmatch(str(item)) is None for item in upstream)
    ):
        raise ValueError("prior full-byte checkpoint adoption attestation is invalid")
    return (
        value,
        digest,
        size,
        _timestamp_ns(value["recorded_at"]),
        int(identity[6]),
    )


def _source_root(source: Path) -> Path:
    candidate = Path(source)
    root = (
        candidate.parent
        if candidate.name == _portable.MANIFEST_NAME
        else candidate
    )
    if root.is_symlink():
        raise ValueError("operator-trusted artifact root cannot be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(
            "operator-trusted artifact source must be a directory or manifest"
        )
    return root


def _payload_stat_cache(
    *,
    manifest: Mapping[str, Any],
    payload_root: Path,
    prior_recorded_at_ns: int,
    prior_attestation_ctime_ns: int,
) -> tuple[
    dict[str, tuple[tuple[int, ...], str, int]],
    tuple[Mapping[str, Any], ...],
]:
    raw_rows = manifest.get("payloads")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("operator-trusted artifact payload inventory is empty")
    registrations = tuple(
        _portable.PayloadRegistration(**dict(row))
        for row in raw_rows
    )
    cache: dict[str, tuple[tuple[int, ...], str, int]] = {}
    inventory: list[Mapping[str, Any]] = []
    for registration in registrations:
        relative = registration.relative_path
        path = payload_root / relative
        boundaries = _portable._safe_path_boundaries(
            root=payload_root,
            relative_path=relative,
            label=f"operator-trusted artifact payload {relative}",
        )
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise ValueError(
                "operator-trusted artifact payload path must be lexical and symlink-free"
            )
        state = os.lstat(path)
        identity = _portable._stat_identity(state)
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or int(state.st_nlink) != 1
            or int(state.st_size) != registration.size_bytes
            or boundaries.get(str(path)) != identity
        ):
            raise ValueError(
                f"operator-trusted artifact payload metadata changed: {relative}"
            )
        if (
            int(state.st_mtime_ns) > prior_recorded_at_ns
            or int(state.st_ctime_ns) > prior_recorded_at_ns
            or int(state.st_ctime_ns)
            >= prior_attestation_ctime_ns
        ):
            raise ValueError(
                "operator-trusted artifact payload has a filesystem change "
                f"newer than its prior full-byte attestation: {relative}"
            )
        cache[str(resolved)] = (
            identity,
            registration.sha256,
            registration.size_bytes,
        )
        inventory.append(
            {
                "relative_path": relative,
                "size_bytes": registration.size_bytes,
                "stat_identity": list(identity),
            }
        )
    return cache, tuple(inventory)


def validate_operator_trusted_portable_artifact(
    *,
    source: Path,
    prior_attestation_path: Path,
    expected_kind: str | None = None,
    expected_compatibility_key: str | None = None,
    expected_upstream_artifact_ids: Sequence[str] | None = None,
) -> OperatorTrustedCheckpoint:
    """Validate controls and stat continuity without reading payload contents."""

    (
        prior,
        prior_sha256,
        prior_size,
        prior_recorded_at_ns,
        prior_attestation_ctime_ns,
    ) = (
        _validate_prior_attestation(Path(prior_attestation_path))
    )
    root = _source_root(Path(source))
    manifest_path = root / _portable.MANIFEST_NAME
    locator_path = root / _portable.LOCATOR_NAME
    if str(locator_path.resolve(strict=True)) != prior["producer_locator"]:
        raise ValueError(
            "operator-trusted artifact locator differs from its prior attestation"
        )
    manifest_sha256, manifest_size, _manifest_identity = (
        _portable._safe_file_hash_with_identity(
            manifest_path,
            label="operator-trusted artifact manifest",
        )
    )
    locator_sha256, locator_size, _locator_identity = (
        _portable._safe_file_hash_with_identity(
            locator_path,
            label="operator-trusted artifact locator",
        )
    )
    if (
        manifest_sha256 != prior["producer_manifest_sha256"]
        or manifest_size != prior["producer_manifest_size_bytes"]
        or locator_sha256 != prior["producer_locator_sha256"]
        or locator_size != prior["producer_locator_size_bytes"]
    ):
        raise ValueError(
            "operator-trusted artifact controls differ from the prior "
            "full-byte attestation"
        )
    manifest = _portable._strict_json_bytes(
        _portable._safe_read(
            manifest_path,
            label="operator-trusted artifact manifest",
        ),
        label="operator-trusted artifact manifest",
    )
    locator = _portable._strict_json_bytes(
        _portable._safe_read(
            locator_path,
            label="operator-trusted artifact locator",
        ),
        label="operator-trusted artifact locator",
    )
    raw_payload_root = locator.get("payload_root")
    if not isinstance(raw_payload_root, str):
        raise ValueError("operator-trusted artifact payload locator is invalid")
    supplied_payload_root = Path(raw_payload_root)
    if supplied_payload_root.is_symlink():
        raise ValueError("operator-trusted artifact payload root cannot be a symlink")
    payload_root = supplied_payload_root.resolve(strict=True)
    cache, stat_inventory = _payload_stat_cache(
        manifest=manifest,
        payload_root=payload_root,
        prior_recorded_at_ns=prior_recorded_at_ns,
        prior_attestation_ctime_ns=prior_attestation_ctime_ns,
    )
    artifact = _portable.validate_portable_artifact(
        root,
        expected_kind=expected_kind,
        expected_compatibility_key=expected_compatibility_key,
        expected_upstream_artifact_ids=expected_upstream_artifact_ids,
        payload_authentication_cache=cache,
    )
    if (
        artifact.artifact_id != prior["producer_artifact_id"]
        or artifact.manifest["artifact_kind"]
        != prior["producer_artifact_kind"]
        or artifact.compatibility_key
        != prior["producer_compatibility_key"]
        or artifact.manifest["content_root"]
        != prior["producer_content_root"]
        or list(artifact.manifest["upstream_artifact_ids"])
        != prior["validated_upstream_artifact_ids"]
        or (
            None
            if artifact.phase_binding is None
            else artifact.phase_binding.get("content_sha256")
        )
        != prior["producer_operational_phase_binding_content_sha256"]
    ):
        raise ValueError(
            "operator-trusted artifact identity differs from its prior "
            "full-byte attestation"
        )
    return OperatorTrustedCheckpoint(
        artifact=artifact,
        prior_attestation_path=Path(prior_attestation_path).resolve(strict=True),
        prior_attestation_sha256=prior_sha256,
        prior_attestation_size_bytes=prior_size,
        prior_attestation_content_sha256=str(prior["content_sha256"]),
        prior_consumer_request_sha256=str(prior["consumer_request_sha256"]),
        prior_recorded_at=str(prior["recorded_at"]),
        payload_stat_inventory=stat_inventory,
    )


def _stable_adoption_body(
    *,
    trusted: OperatorTrustedCheckpoint,
    consumer_request_sha256: str,
) -> dict[str, Any]:
    artifact = trusted.artifact
    manifest_sha256, manifest_size, _manifest_identity = (
        _portable._safe_file_hash_with_identity(
            artifact.manifest_path,
            label="operator-trusted artifact manifest for adoption",
        )
    )
    locator_sha256, locator_size, _locator_identity = (
        _portable._safe_file_hash_with_identity(
            artifact.locator_path,
            label="operator-trusted artifact locator for adoption",
        )
    )
    stat_rows = [
        copy.deepcopy(dict(row))
        for row in trusted.payload_stat_inventory
    ]
    stat_body = {
        "schema_version": "operator_trusted_payload_stat_inventory_v1",
        "payloads": stat_rows,
    }
    phase_binding = artifact.phase_binding
    return {
        "schema_version": OPERATOR_TRUSTED_ADOPTION_ATTESTATION,
        "producer_artifact_id": artifact.artifact_id,
        "producer_artifact_kind": artifact.manifest["artifact_kind"],
        "producer_compatibility_key": artifact.compatibility_key,
        "producer_content_root": artifact.manifest["content_root"],
        "producer_locator": str(artifact.locator_path),
        "producer_manifest_sha256": manifest_sha256,
        "producer_manifest_size_bytes": manifest_size,
        "producer_locator_sha256": locator_sha256,
        "producer_locator_size_bytes": locator_size,
        "producer_operational_phase_binding_content_sha256": (
            None
            if phase_binding is None
            else phase_binding.get("content_sha256")
        ),
        "consumer_request_sha256": consumer_request_sha256,
        "validated_upstream_artifact_ids": list(
            artifact.manifest["upstream_artifact_ids"]
        ),
        "validation_policy": OPERATOR_TRUSTED_VALIDATION_POLICY,
        "operator_trust_explicit": True,
        "prior_adoption_attestation_path": str(
            trusted.prior_attestation_path
        ),
        "prior_adoption_attestation_sha256": (
            trusted.prior_attestation_sha256
        ),
        "prior_adoption_attestation_size_bytes": (
            trusted.prior_attestation_size_bytes
        ),
        "prior_adoption_attestation_content_sha256": (
            trusted.prior_attestation_content_sha256
        ),
        "prior_consumer_request_sha256": (
            trusted.prior_consumer_request_sha256
        ),
        "prior_full_byte_validation_policy": _PRIOR_FULL_BYTE_POLICY,
        "prior_full_byte_validation_recorded_at": (
            trusted.prior_recorded_at
        ),
        "payload_stat_inventory": stat_body,
        "payload_stat_inventory_content_sha256": identity_sha256(
            stat_body
        ),
        "payload_bytes_reauthenticated": False,
        "fresh_full_byte_validation_achieved": False,
        "global_release_certified": False,
    }


def adopt_checkpoint_from_prior_full_byte_attestation(
    *,
    source: Path,
    prior_attestation_path: Path,
    attestation_root: Path,
    consumer_request_sha256: str,
    expected_kind: str | None = None,
    expected_compatibility_key: str | None = None,
    expected_upstream_artifact_ids: Sequence[str] | None = None,
    trusted_checkpoint: OperatorTrustedCheckpoint | None = None,
) -> Mapping[str, Any]:
    """Publish an explicit request-bound transitive trust attestation."""

    if _SHA256.fullmatch(str(consumer_request_sha256)) is None:
        raise ValueError("consumer request identity must be one lowercase SHA-256")
    trusted = (
        validate_operator_trusted_portable_artifact(
            source=source,
            prior_attestation_path=prior_attestation_path,
            expected_kind=expected_kind,
            expected_compatibility_key=expected_compatibility_key,
            expected_upstream_artifact_ids=expected_upstream_artifact_ids,
        )
        if trusted_checkpoint is None
        else trusted_checkpoint
    )
    requested_root = _source_root(Path(source))
    if (
        requested_root != trusted.artifact.root
        or Path(prior_attestation_path).resolve(strict=True)
        != trusted.prior_attestation_path
    ):
        raise ValueError(
            "operator-trusted checkpoint handle does not match its sources"
        )
    revalidated = validate_operator_trusted_portable_artifact(
        source=requested_root,
        prior_attestation_path=trusted.prior_attestation_path,
        expected_kind=expected_kind,
        expected_compatibility_key=expected_compatibility_key,
        expected_upstream_artifact_ids=expected_upstream_artifact_ids,
    )
    if (
        revalidated.artifact.artifact_id
        != trusted.artifact.artifact_id
        or revalidated.payload_stat_inventory
        != trusted.payload_stat_inventory
    ):
        raise RuntimeError(
            "operator-trusted checkpoint changed before adoption"
        )
    stable_body = _stable_adoption_body(
        trusted=revalidated,
        consumer_request_sha256=consumer_request_sha256,
    )
    target_root = Path(attestation_root)
    if target_root.exists() and (
        target_root.is_symlink() or not target_root.is_dir()
    ):
        raise ValueError(
            "operator-trusted adoption root must be a symlink-free directory"
        )
    target_root.mkdir(parents=True, exist_ok=True)
    target = (
        target_root
        / f"{revalidated.artifact.artifact_id}.adoption.json"
    )
    if target.exists():
        return validate_operator_trusted_checkpoint_adoption(
            attestation_path=target,
            source=requested_root,
            prior_attestation_path=revalidated.prior_attestation_path,
            consumer_request_sha256=consumer_request_sha256,
            expected_kind=expected_kind,
            expected_compatibility_key=expected_compatibility_key,
            expected_upstream_artifact_ids=expected_upstream_artifact_ids,
        )
    body = {**stable_body, "recorded_at": _portable._utc_now()}
    attestation = {
        **body,
        "content_sha256": identity_sha256(body),
    }
    _portable._atomic_json_new(target, attestation)
    reopened = _portable._strict_json_bytes(
        _portable._safe_read(
            target,
            label="new operator-trusted checkpoint adoption attestation",
        ),
        label="new operator-trusted checkpoint adoption attestation",
    )
    if reopened != attestation:
        raise RuntimeError(
            "operator-trusted checkpoint adoption attestation changed "
            "after publication"
        )
    return reopened


def validate_operator_trusted_checkpoint_adoption(
    *,
    attestation_path: Path,
    source: Path,
    prior_attestation_path: Path,
    consumer_request_sha256: str,
    expected_kind: str | None = None,
    expected_compatibility_key: str | None = None,
    expected_upstream_artifact_ids: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    """Revalidate a transitive adoption using controls and stat continuity."""

    if _SHA256.fullmatch(str(consumer_request_sha256)) is None:
        raise ValueError("consumer request identity must be one lowercase SHA-256")
    trusted = validate_operator_trusted_portable_artifact(
        source=source,
        prior_attestation_path=prior_attestation_path,
        expected_kind=expected_kind,
        expected_compatibility_key=expected_compatibility_key,
        expected_upstream_artifact_ids=expected_upstream_artifact_ids,
    )
    value = _portable._strict_json_bytes(
        _portable._safe_read(
            Path(attestation_path),
            label="operator-trusted checkpoint adoption attestation",
        ),
        label="operator-trusted checkpoint adoption attestation",
    )
    expected_stable = _stable_adoption_body(
        trusted=trusted,
        consumer_request_sha256=consumer_request_sha256,
    )
    body = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "content_sha256"
    }
    observed_stable = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"recorded_at", "content_sha256"}
    }
    if (
        observed_stable != expected_stable
        or set(value)
        != {*expected_stable, "recorded_at", "content_sha256"}
        or not isinstance(value.get("recorded_at"), str)
        or not str(value["recorded_at"]).strip()
        or value.get("content_sha256") != identity_sha256(body)
    ):
        raise ValueError(
            "operator-trusted checkpoint adoption attestation is invalid"
        )
    return value


__all__ = [
    "OPERATOR_TRUSTED_ADOPTION_ATTESTATION",
    "OPERATOR_TRUSTED_VALIDATION_POLICY",
    "OperatorTrustedCheckpoint",
    "adopt_checkpoint_from_prior_full_byte_attestation",
    "validate_operator_trusted_checkpoint_adoption",
    "validate_operator_trusted_portable_artifact",
]
