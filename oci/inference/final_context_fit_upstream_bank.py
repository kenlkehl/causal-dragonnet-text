"""Authenticated final outer-fold banks from context-fitted upstream models.

The adaptive post-extraction reviewer and the final outer-fold predictor have
different honesty requirements.  Review gates need recursively spent-only
models.  Once the registry is frozen, the final predictor instead needs one
complete meta-inner OOF bank on outer train and one model fitted on all outer
train for the label-free outer heldout rows.

This module is the strict bridge from :class:`ContextFitUpstreamBackend` to
that final representation.  Its public production method has no heldout
treatment or outcome argument.  For every precommitted meta-inner fold it
passes only the complementary rows' observable labels to the backend, then
passes the complete outer train for the final heldout transformation.  Source
and raw-feature schemas must be byte-for-byte stable across every fit.

Inputs, backend identity/runtime code, output matrices, and the deterministic
fit-row lineage are content addressed.  Existing cache entries are always
re-authenticated and are never silently repaired after tampering.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import marshal
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .context_fit_upstream_gate_provider import (
    ContextFitUpstreamBackend,
    ContextFitUpstreamPrediction,
)
from .fold_honest_r_stack import FitRowProvenance

FINAL_CONTEXT_FIT_UPSTREAM_PRODUCER_ID = "final_context_fit_upstream_producer_v1"
FINAL_CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA = "final_context_fit_upstream_cache_v1"

_FORBIDDEN = ("true", "oracle", "ground_truth")
_SHA256_LENGTH = 64
_ROLES = frozenset(
    {
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        UNCALIBRATED_EFFECT_MODIFIER_ROLE,
    }
)
_MATRIX_FILENAMES = MappingProxyType(
    {
        "source_train_oof": "calibrated_source_train_oof.npy",
        "source_outer_heldout": "calibrated_source_outer_heldout.npy",
        "feature_train_oof": "raw_feature_train_oof.npy",
        "feature_outer_heldout": "raw_feature_outer_heldout.npy",
    }
)
_MANIFEST_FIELDS = frozenset(
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
_SOURCE_FIELDS = frozenset({"names", "kinds", "content_sha256"})
_FEATURE_FIELDS = frozenset({"names", "kinds", "roles", "content_sha256"})
_FILE_RECORD_FIELDS = frozenset({"filename", "sha256"})


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _valid_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != _SHA256_LENGTH:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from exc
    if normalized != str(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _closed_json(value: Any, *, path: str) -> Any:
    """Normalize identity metadata while rejecting benchmark truth channels."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key or any(token in key.lower() for token in _FORBIDDEN):
                raise ValueError(f"{path} contains a forbidden or empty identity field")
            if key in result:
                raise ValueError(f"{path} contains colliding identity fields")
            result[key] = _closed_json(raw_value, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_closed_json(item, path=f"{path}[]") for item in value]
    if isinstance(value, np.generic):
        return _closed_json(value.item(), path=path)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if any(token in value.lower() for token in _FORBIDDEN):
            raise ValueError(f"{path} contains a forbidden identity value")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite identity metadata")
        return value
    raise TypeError(f"{path} must contain closed JSON-compatible metadata")


def _integer_rows(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of integer row IDs")
    result: list[int] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain canonical integer row IDs")
        row_id = int(value)
        if row_id < 0:
            raise ValueError(f"{name} cannot contain negative row IDs")
        result.append(row_id)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return tuple(result)


def _exact_texts(values: Sequence[Any], *, name: str, length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of exact strings")
    result = tuple(values)
    if len(result) != int(length) or not all(isinstance(value, str) for value in result):
        raise ValueError(f"{name} must contain exactly {length} strings")
    return result


def _finite_vector(values: Sequence[Any], *, name: str, length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) != int(length):
        raise ValueError(f"{name} must be one-dimensional with length {length}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    result = vector.copy()
    result.setflags(write=False)
    return result


def _finite_matrix(values: Any, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite matrix with shape {shape}")
    result = matrix.copy()
    result.setflags(write=False)
    return result


def _safe_names(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(str(value).strip() for value in values)
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique non-empty strings")
    if any(any(token in value.lower() for token in _FORBIDDEN) for value in result):
        raise ValueError(f"{name} contains forbidden benchmark metadata")
    return result


def _aligned_metadata(values: Sequence[Any], *, name: str, length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(str(value).strip() for value in values)
    if len(result) != int(length) or any(not value for value in result):
        raise ValueError(f"{name} must align with {length} columns")
    if any(any(token in value.lower() for token in _FORBIDDEN) for value in result):
        raise ValueError(f"{name} contains forbidden benchmark metadata")
    return result


def _fold_ids(values: Sequence[Any], *, length: int) -> tuple[Hashable, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("meta_inner_fold_ids must be a sequence")
    result: list[Hashable] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, str)):
            raise TypeError("meta_inner_fold_ids must contain integers or strings")
        if isinstance(value, str):
            normalized: Hashable = value.strip()
            if not normalized or any(token in normalized.lower() for token in _FORBIDDEN):
                raise ValueError("meta_inner_fold_ids contains an invalid string fold ID")
        else:
            normalized = int(value)
            if normalized < 1:
                raise ValueError("integer meta_inner_fold_ids must be positive")
        result.append(normalized)
    if len(result) != int(length):
        raise ValueError(f"meta_inner_fold_ids must have length {length}")
    if len(set(result)) < 2:
        raise ValueError("meta_inner_fold_ids must contain at least two folds")
    return tuple(result)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _float_hex_sha256(values: np.ndarray) -> str:
    vector = np.asarray(values, dtype=float)
    return _sha256_json([float(value).hex() for value in vector])


def _matrix_content_sha256(values: np.ndarray) -> str:
    matrix = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    header = _canonical_json({"dtype": "<f8", "shape": list(matrix.shape), "order": "C"}).encode(
        "utf-8"
    )
    return hashlib.sha256(header + b"\0" + matrix.tobytes(order="C")).hexdigest()


def _lineage_payload(lineage: FitRowProvenance, *, active: set[int] | None = None) -> Any:
    if not isinstance(lineage, FitRowProvenance):
        raise TypeError("lineage entries must be FitRowProvenance")
    stack = set() if active is None else active
    identity = id(lineage)
    if identity in stack:
        raise ValueError("fit-row lineage contains a cycle")
    stack.add(identity)
    try:
        normalized_rows: list[int] = []
        for value in lineage.fit_row_ids:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError("fit-row lineage must contain canonical integer row IDs")
            row_id = int(value)
            if row_id < 0:
                raise ValueError("fit-row lineage cannot contain negative row IDs")
            normalized_rows.append(row_id)
        fit_rows = sorted(normalized_rows)
        return {
            "fit_row_ids": fit_rows,
            "upstream": [_lineage_payload(parent, active=stack) for parent in lineage.upstream],
        }
    finally:
        stack.remove(identity)


def _provenance_payload(values: tuple[tuple[FitRowProvenance, ...], ...]) -> list[list[Any]]:
    return [[_lineage_payload(lineage) for lineage in row] for row in values]


def _expected_fit_rows_by_position(
    train_row_ids: tuple[int, ...], meta_fold_ids: tuple[Hashable, ...]
) -> tuple[frozenset[int], ...]:
    fit_by_fold = {
        fold_id: frozenset(
            row_id
            for row_id, candidate_fold in zip(train_row_ids, meta_fold_ids)
            if candidate_fold != fold_id
        )
        for fold_id in dict.fromkeys(meta_fold_ids)
    }
    if any(not values for values in fit_by_fold.values()):
        raise ValueError("every meta-inner fold must have a non-empty complementary fit")
    return tuple(fit_by_fold[fold_id] for fold_id in meta_fold_ids)


def _validate_lineage_matrix(
    values: Sequence[Sequence[FitRowProvenance]],
    *,
    name: str,
    rows: int,
    columns: int,
    expected_fit_rows: Sequence[frozenset[int]],
) -> tuple[tuple[FitRowProvenance, ...], ...]:
    matrix = tuple(tuple(row) for row in values)
    if len(matrix) != int(rows) or any(len(row) != int(columns) for row in matrix):
        raise ValueError(f"{name} must have provenance shape {(rows, columns)}")
    if len(expected_fit_rows) != int(rows):
        raise ValueError(f"{name} expected-fit-row specification is misaligned")
    for row_index, (row, expected) in enumerate(zip(matrix, expected_fit_rows)):
        for lineage in row:
            if not isinstance(lineage, FitRowProvenance):
                raise TypeError(f"{name} entries must be FitRowProvenance")
            recursive = lineage.recursive_fit_row_ids()
            if recursive != expected:
                raise ValueError(
                    f"{name} row {row_index} does not carry its exact recursive fit rows"
                )
    return matrix


def _bank_digest(
    *,
    bank_type: str,
    train_row_ids: tuple[int, ...],
    heldout_row_ids: tuple[int, ...],
    meta_fold_ids: tuple[Hashable, ...],
    names: tuple[str, ...],
    kinds: tuple[str, ...],
    roles: tuple[str, ...] | None,
    train_values: np.ndarray,
    heldout_values: np.ndarray,
    train_provenance: tuple[tuple[FitRowProvenance, ...], ...],
    heldout_provenance: tuple[tuple[FitRowProvenance, ...], ...],
) -> str:
    payload: dict[str, Any] = {
        "bank_type": bank_type,
        "train_row_ids": list(train_row_ids),
        "heldout_row_ids": list(heldout_row_ids),
        "meta_inner_fold_ids": list(meta_fold_ids),
        "names": list(names),
        "kinds": list(kinds),
        "train_values_sha256": _matrix_content_sha256(train_values),
        "heldout_values_sha256": _matrix_content_sha256(heldout_values),
        "train_provenance": _provenance_payload(train_provenance),
        "heldout_provenance": _provenance_payload(heldout_provenance),
    }
    if roles is not None:
        payload["roles"] = list(roles)
    return _sha256_json(payload)


@dataclass(frozen=True)
class AuthenticatedCalibratedTauBank:
    """Meta-inner OOF and outer-heldout calibrated treatment-effect sources."""

    train_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    meta_inner_fold_ids: tuple[Hashable, ...]
    source_names: tuple[str, ...]
    source_kinds: tuple[str, ...]
    train_oof_values: np.ndarray = field(repr=False)
    outer_heldout_values: np.ndarray = field(repr=False)
    train_oof_fit_row_provenance: tuple[tuple[FitRowProvenance, ...], ...] = field(repr=False)
    outer_heldout_fit_row_provenance: tuple[tuple[FitRowProvenance, ...], ...] = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        train_rows = _integer_rows(self.train_row_ids, name="source_bank.train_row_ids")
        heldout_rows = _integer_rows(self.heldout_row_ids, name="source_bank.heldout_row_ids")
        if set(train_rows) & set(heldout_rows):
            raise ValueError("source-bank train and heldout rows must be disjoint")
        folds = _fold_ids(self.meta_inner_fold_ids, length=len(train_rows))
        names = _safe_names(self.source_names, name="source_bank.source_names")
        kinds = _aligned_metadata(
            self.source_kinds, name="source_bank.source_kinds", length=len(names)
        )
        train_values = _finite_matrix(
            self.train_oof_values,
            name="source_bank.train_oof_values",
            shape=(len(train_rows), len(names)),
        )
        heldout_values = _finite_matrix(
            self.outer_heldout_values,
            name="source_bank.outer_heldout_values",
            shape=(len(heldout_rows), len(names)),
        )
        train_expected = _expected_fit_rows_by_position(train_rows, folds)
        heldout_expected = tuple(frozenset(train_rows) for _ in heldout_rows)
        train_provenance = _validate_lineage_matrix(
            self.train_oof_fit_row_provenance,
            name="source_bank.train_oof_fit_row_provenance",
            rows=len(train_rows),
            columns=len(names),
            expected_fit_rows=train_expected,
        )
        heldout_provenance = _validate_lineage_matrix(
            self.outer_heldout_fit_row_provenance,
            name="source_bank.outer_heldout_fit_row_provenance",
            rows=len(heldout_rows),
            columns=len(names),
            expected_fit_rows=heldout_expected,
        )
        digest = _bank_digest(
            bank_type="calibrated_tau_sources",
            train_row_ids=train_rows,
            heldout_row_ids=heldout_rows,
            meta_fold_ids=folds,
            names=names,
            kinds=kinds,
            roles=None,
            train_values=train_values,
            heldout_values=heldout_values,
            train_provenance=train_provenance,
            heldout_provenance=heldout_provenance,
        )
        if _valid_sha256(self.content_sha256, name="source_bank.content_sha256") != digest:
            raise ValueError("source-bank authenticated content SHA-256 mismatch")
        object.__setattr__(self, "train_row_ids", train_rows)
        object.__setattr__(self, "heldout_row_ids", heldout_rows)
        object.__setattr__(self, "meta_inner_fold_ids", folds)
        object.__setattr__(self, "source_names", names)
        object.__setattr__(self, "source_kinds", kinds)
        object.__setattr__(self, "train_oof_values", train_values)
        object.__setattr__(self, "outer_heldout_values", heldout_values)
        object.__setattr__(self, "train_oof_fit_row_provenance", train_provenance)
        object.__setattr__(self, "outer_heldout_fit_row_provenance", heldout_provenance)
        object.__setattr__(self, "content_sha256", digest)

    def verify_authenticated_content(self) -> None:
        digest = _bank_digest(
            bank_type="calibrated_tau_sources",
            train_row_ids=self.train_row_ids,
            heldout_row_ids=self.heldout_row_ids,
            meta_fold_ids=self.meta_inner_fold_ids,
            names=self.source_names,
            kinds=self.source_kinds,
            roles=None,
            train_values=self.train_oof_values,
            heldout_values=self.outer_heldout_values,
            train_provenance=self.train_oof_fit_row_provenance,
            heldout_provenance=self.outer_heldout_fit_row_provenance,
        )
        if digest != self.content_sha256:
            raise ValueError("source-bank in-memory authenticated content was modified")


@dataclass(frozen=True)
class AuthenticatedRoleAwareFeatureBank:
    """Meta-inner OOF and outer-heldout uncalibrated role-aware raw features."""

    train_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    meta_inner_fold_ids: tuple[Hashable, ...]
    feature_names: tuple[str, ...]
    feature_kinds: tuple[str, ...]
    consumer_roles: tuple[str, ...]
    train_oof_values: np.ndarray = field(repr=False)
    outer_heldout_values: np.ndarray = field(repr=False)
    train_oof_fit_row_provenance: tuple[tuple[FitRowProvenance, ...], ...] = field(repr=False)
    outer_heldout_fit_row_provenance: tuple[tuple[FitRowProvenance, ...], ...] = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        train_rows = _integer_rows(self.train_row_ids, name="feature_bank.train_row_ids")
        heldout_rows = _integer_rows(self.heldout_row_ids, name="feature_bank.heldout_row_ids")
        if set(train_rows) & set(heldout_rows):
            raise ValueError("feature-bank train and heldout rows must be disjoint")
        folds = _fold_ids(self.meta_inner_fold_ids, length=len(train_rows))
        names = _safe_names(self.feature_names, name="feature_bank.feature_names")
        kinds = _aligned_metadata(
            self.feature_kinds, name="feature_bank.feature_kinds", length=len(names)
        )
        roles = _aligned_metadata(
            self.consumer_roles, name="feature_bank.consumer_roles", length=len(names)
        )
        if set(roles) - _ROLES:
            raise ValueError("feature_bank.consumer_roles contains an unsupported role")
        train_values = _finite_matrix(
            self.train_oof_values,
            name="feature_bank.train_oof_values",
            shape=(len(train_rows), len(names)),
        )
        heldout_values = _finite_matrix(
            self.outer_heldout_values,
            name="feature_bank.outer_heldout_values",
            shape=(len(heldout_rows), len(names)),
        )
        train_expected = _expected_fit_rows_by_position(train_rows, folds)
        heldout_expected = tuple(frozenset(train_rows) for _ in heldout_rows)
        train_provenance = _validate_lineage_matrix(
            self.train_oof_fit_row_provenance,
            name="feature_bank.train_oof_fit_row_provenance",
            rows=len(train_rows),
            columns=len(names),
            expected_fit_rows=train_expected,
        )
        heldout_provenance = _validate_lineage_matrix(
            self.outer_heldout_fit_row_provenance,
            name="feature_bank.outer_heldout_fit_row_provenance",
            rows=len(heldout_rows),
            columns=len(names),
            expected_fit_rows=heldout_expected,
        )
        digest = _bank_digest(
            bank_type="role_aware_raw_features",
            train_row_ids=train_rows,
            heldout_row_ids=heldout_rows,
            meta_fold_ids=folds,
            names=names,
            kinds=kinds,
            roles=roles,
            train_values=train_values,
            heldout_values=heldout_values,
            train_provenance=train_provenance,
            heldout_provenance=heldout_provenance,
        )
        if _valid_sha256(self.content_sha256, name="feature_bank.content_sha256") != digest:
            raise ValueError("feature-bank authenticated content SHA-256 mismatch")
        object.__setattr__(self, "train_row_ids", train_rows)
        object.__setattr__(self, "heldout_row_ids", heldout_rows)
        object.__setattr__(self, "meta_inner_fold_ids", folds)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_kinds", kinds)
        object.__setattr__(self, "consumer_roles", roles)
        object.__setattr__(self, "train_oof_values", train_values)
        object.__setattr__(self, "outer_heldout_values", heldout_values)
        object.__setattr__(self, "train_oof_fit_row_provenance", train_provenance)
        object.__setattr__(self, "outer_heldout_fit_row_provenance", heldout_provenance)
        object.__setattr__(self, "content_sha256", digest)

    def verify_authenticated_content(self) -> None:
        digest = _bank_digest(
            bank_type="role_aware_raw_features",
            train_row_ids=self.train_row_ids,
            heldout_row_ids=self.heldout_row_ids,
            meta_fold_ids=self.meta_inner_fold_ids,
            names=self.feature_names,
            kinds=self.feature_kinds,
            roles=self.consumer_roles,
            train_values=self.train_oof_values,
            heldout_values=self.outer_heldout_values,
            train_provenance=self.train_oof_fit_row_provenance,
            heldout_provenance=self.outer_heldout_fit_row_provenance,
        )
        if digest != self.content_sha256:
            raise ValueError("feature-bank in-memory authenticated content was modified")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("final upstream cache manifest is unreadable") from exc
    if not isinstance(payload, Mapping) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("final upstream cache manifest does not match its closed schema")
    if payload["schema_version"] != FINAL_CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA:
        raise ValueError("unsupported final upstream cache schema")
    content = {key: value for key, value in payload.items() if key != "content_sha256"}
    if payload["content_sha256"] != _sha256_json(content):
        raise ValueError("final upstream cache manifest content SHA-256 mismatch")
    return payload


@dataclass(frozen=True)
class AuthenticatedFinalContextFitUpstreamBank:
    """One complete, on-disk-authenticated final outer-fold upstream package."""

    outer_fold: int
    cache_key: str
    producer_identity_sha256: str
    manifest_path: Path
    manifest_sha256: str
    calibrated_sources: AuthenticatedCalibratedTauBank
    raw_features: AuthenticatedRoleAwareFeatureBank

    def __post_init__(self) -> None:
        fold = _positive_int(self.outer_fold, name="outer_fold")
        cache_key = _valid_sha256(self.cache_key, name="cache_key")
        producer_sha = _valid_sha256(self.producer_identity_sha256, name="producer_identity_sha256")
        manifest_sha = _valid_sha256(self.manifest_sha256, name="manifest_sha256")
        path = Path(self.manifest_path).resolve(strict=True)
        if not path.is_file() or path.name != "manifest.json" or path.parent.name != cache_key:
            raise ValueError("manifest_path is not the canonical content-addressed manifest")
        if not isinstance(self.calibrated_sources, AuthenticatedCalibratedTauBank):
            raise TypeError("calibrated_sources has the wrong authenticated bank type")
        if not isinstance(self.raw_features, AuthenticatedRoleAwareFeatureBank):
            raise TypeError("raw_features has the wrong authenticated bank type")
        source = self.calibrated_sources
        features = self.raw_features
        if (
            source.train_row_ids != features.train_row_ids
            or source.heldout_row_ids != features.heldout_row_ids
            or source.meta_inner_fold_ids != features.meta_inner_fold_ids
        ):
            raise ValueError("source and raw-feature banks do not share the exact split")
        object.__setattr__(self, "outer_fold", fold)
        object.__setattr__(self, "cache_key", cache_key)
        object.__setattr__(self, "producer_identity_sha256", producer_sha)
        object.__setattr__(self, "manifest_path", path)
        object.__setattr__(self, "manifest_sha256", manifest_sha)

    def verify_authenticated_content(self) -> None:
        """Re-read every persisted byte and reject any manifest/matrix change."""

        self.calibrated_sources.verify_authenticated_content()
        self.raw_features.verify_authenticated_content()
        if _sha256_file(self.manifest_path) != self.manifest_sha256:
            raise ValueError("final upstream cache manifest bytes were modified")
        payload = _read_manifest(self.manifest_path)
        if payload["cache_key"] != self.cache_key:
            raise ValueError("final upstream cache key changed")
        binding = payload["binding"]
        if not isinstance(binding, Mapping):
            raise TypeError("final upstream cache binding must be an object")
        if _sha256_json(binding) != self.cache_key:
            raise ValueError("final upstream cache binding does not match its key")
        if binding.get("outer_fold") != self.outer_fold:
            raise ValueError("final upstream cache outer fold changed")
        if binding.get("producer_identity_sha256") != self.producer_identity_sha256:
            raise ValueError("final upstream producer identity changed")
        source_record = payload["calibrated_sources"]
        feature_record = payload["raw_features"]
        if not isinstance(source_record, Mapping) or set(source_record) != _SOURCE_FIELDS:
            raise ValueError("calibrated-source manifest record has an invalid schema")
        if not isinstance(feature_record, Mapping) or set(feature_record) != _FEATURE_FIELDS:
            raise ValueError("raw-feature manifest record has an invalid schema")
        if source_record != {
            "names": list(self.calibrated_sources.source_names),
            "kinds": list(self.calibrated_sources.source_kinds),
            "content_sha256": self.calibrated_sources.content_sha256,
        }:
            raise ValueError("calibrated-source manifest metadata changed")
        if feature_record != {
            "names": list(self.raw_features.feature_names),
            "kinds": list(self.raw_features.feature_kinds),
            "roles": list(self.raw_features.consumer_roles),
            "content_sha256": self.raw_features.content_sha256,
        }:
            raise ValueError("raw-feature manifest metadata changed")

        matrices = payload["matrix_files"]
        if not isinstance(matrices, Mapping) or set(matrices) != set(_MATRIX_FILENAMES):
            raise ValueError("final upstream matrix manifest has an invalid schema")
        expected_arrays = {
            "source_train_oof": self.calibrated_sources.train_oof_values,
            "source_outer_heldout": self.calibrated_sources.outer_heldout_values,
            "feature_train_oof": self.raw_features.train_oof_values,
            "feature_outer_heldout": self.raw_features.outer_heldout_values,
        }
        for name, filename in _MATRIX_FILENAMES.items():
            record = matrices[name]
            if not isinstance(record, Mapping) or set(record) != _FILE_RECORD_FIELDS:
                raise ValueError(f"matrix record {name!r} has an invalid schema")
            if record["filename"] != filename:
                raise ValueError(f"matrix record {name!r} changed its canonical filename")
            path = self.manifest_path.parent / filename
            if not path.is_file() or _sha256_file(path) != record["sha256"]:
                raise ValueError(f"matrix record {name!r} failed SHA-256 authentication")
            with path.open("rb") as handle:
                values = np.load(handle, allow_pickle=False)
            expected = np.asarray(expected_arrays[name], dtype=float)
            if values.shape != expected.shape or not np.array_equal(values, expected):
                raise ValueError(f"matrix record {name!r} differs from the authenticated bank")


def _unwrap_callable(value: Any) -> Any:
    if isinstance(value, (staticmethod, classmethod)):
        return value.__func__
    return value


def _callable_code_sha256(owner: type[Any], name: str) -> str:
    try:
        value = _unwrap_callable(inspect.getattr_static(owner, name))
    except AttributeError as exc:
        raise TypeError(f"backend class does not define {name}()") from exc
    code = getattr(value, "__code__", None)
    if code is None:
        raise TypeError(f"backend {name}() must be a Python implementation")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _backend_runtime_attestation(backend: ContextFitUpstreamBackend) -> Mapping[str, Any]:
    owner = type(backend)
    if "identity" in vars(backend) or "fit_predict" in vars(backend):
        raise TypeError("backend has unauthenticated per-instance method overrides")
    source_file = inspect.getsourcefile(owner)
    if not source_file:
        raise TypeError("backend class must come from an authenticated Python source file")
    path = Path(source_file).resolve(strict=True)
    attestation: dict[str, Any] = {
        "class_module": owner.__module__,
        "class_qualname": owner.__qualname__,
        "module_file_sha256": _sha256_file(path),
        "identity_code_sha256": _callable_code_sha256(owner, "identity"),
        "fit_predict_code_sha256": _callable_code_sha256(owner, "fit_predict"),
    }
    members = getattr(backend, "backends", None)
    if members is not None:
        if isinstance(members, (str, bytes, Mapping)):
            raise TypeError("composite backend members have an invalid runtime shape")
        member_tuple = tuple(members)
        if not member_tuple:
            raise ValueError("composite backend has no members")
        attestation["members"] = [_backend_runtime_attestation(member) for member in member_tuple]
    return attestation


@dataclass(frozen=True)
class _OutputSchema:
    source_names: tuple[str, ...]
    source_kinds: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_kinds: tuple[str, ...]
    feature_roles: tuple[str, ...]

    @classmethod
    def from_prediction(cls, prediction: ContextFitUpstreamPrediction) -> "_OutputSchema":
        return cls(
            source_names=prediction.calibrated_source_names,
            source_kinds=prediction.calibrated_source_kinds,
            feature_names=prediction.feature_names,
            feature_kinds=prediction.feature_kinds,
            feature_roles=prediction.feature_roles,
        )


class FinalContextFitUpstreamProducer:
    """Produce one strict final all-evidence numerical/raw-feature package."""

    def __init__(self, cache_dir: Path | str, *, backend: ContextFitUpstreamBackend) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        if not callable(getattr(backend, "identity", None)) or not callable(
            getattr(backend, "fit_predict", None)
        ):
            raise TypeError("backend must implement identity() and fit_predict()")
        self.backend = backend
        self._backend_identity = _closed_json(backend.identity(), path="backend.identity")
        self._runtime_attestation = _closed_json(
            _backend_runtime_attestation(backend), path="backend.runtime_attestation"
        )
        self._producer_identity = {
            "producer": FINAL_CONTEXT_FIT_UPSTREAM_PRODUCER_ID,
            "producer_code_sha256": _module_sha256(),
            "backend_identity": self._backend_identity,
            "backend_runtime_attestation": self._runtime_attestation,
            "precommitted_meta_inner_folds_required": True,
            "heldout_labels_accepted": False,
            "postfreeze_only": True,
            "calibrated_sources_required": True,
            "role_aware_raw_features_required": True,
        }

    def _assert_backend_stable(self) -> None:
        identity = _closed_json(self.backend.identity(), path="backend.identity")
        runtime = _closed_json(
            _backend_runtime_attestation(self.backend), path="backend.runtime_attestation"
        )
        if identity != self._backend_identity:
            raise ValueError("upstream backend identity changed during final-bank production")
        if runtime != self._runtime_attestation:
            raise TypeError("upstream backend runtime implementation changed")

    def identity(self) -> Mapping[str, Any]:
        self._assert_backend_stable()
        return copy.deepcopy(self._producer_identity)

    def _binding(
        self,
        *,
        outer_fold: int,
        train_row_ids: tuple[int, ...],
        train_texts: tuple[str, ...],
        train_treatment: np.ndarray,
        train_outcome: np.ndarray,
        heldout_row_ids: tuple[int, ...],
        heldout_texts: tuple[str, ...],
        meta_inner_fold_ids: tuple[Hashable, ...],
    ) -> Mapping[str, Any]:
        identity_sha = _sha256_json(self._producer_identity)
        return {
            "producer_identity_sha256": identity_sha,
            "outer_fold": outer_fold,
            "outer_train_row_ids": list(train_row_ids),
            "outer_train_text_sha256": _sha256_json(list(train_texts)),
            "outer_train_treatment_sha256": _float_hex_sha256(train_treatment),
            "outer_train_outcome_sha256": _float_hex_sha256(train_outcome),
            "outer_heldout_row_ids": list(heldout_row_ids),
            "outer_heldout_text_sha256": _sha256_json(list(heldout_texts)),
            "meta_inner_fold_ids": list(meta_inner_fold_ids),
            "outer_heldout_labels_accepted": False,
        }

    def _call_backend(
        self,
        *,
        outer_fold: int,
        context_row_ids: tuple[int, ...],
        context_texts: tuple[str, ...],
        context_treatment: np.ndarray,
        context_outcome: np.ndarray,
        prediction_row_ids: tuple[int, ...],
        prediction_texts: tuple[str, ...],
        work_dir: Path,
    ) -> ContextFitUpstreamPrediction:
        self._assert_backend_stable()
        if set(context_row_ids) & set(prediction_row_ids):
            raise ValueError("backend context and prediction rows must be disjoint")
        work_dir.mkdir(parents=True, exist_ok=True)
        treatment_copy = np.asarray(context_treatment, dtype=float).copy()
        outcome_copy = np.asarray(context_outcome, dtype=float).copy()
        treatment_copy.setflags(write=False)
        outcome_copy.setflags(write=False)
        prediction = self.backend.fit_predict(
            outer_fold=outer_fold,
            context_row_ids=context_row_ids,
            context_texts=context_texts,
            context_treatment=treatment_copy,
            context_outcome=outcome_copy,
            gate_row_ids=prediction_row_ids,
            gate_texts=prediction_texts,
            work_dir=work_dir,
        )
        self._assert_backend_stable()
        if type(prediction) is not ContextFitUpstreamPrediction:
            raise TypeError("backend must return the exact authenticated prediction type")
        if prediction.gate_row_ids != prediction_row_ids:
            raise ValueError("backend changed prediction row identity or order")
        if not prediction.calibrated_source_names:
            raise ValueError("final upstream backend produced no calibrated tau sources")
        if not prediction.feature_names:
            raise ValueError("final upstream backend produced no role-aware raw features")
        return prediction

    @staticmethod
    def _lineages(
        *,
        train_row_ids: tuple[int, ...],
        heldout_row_ids: tuple[int, ...],
        meta_inner_fold_ids: tuple[Hashable, ...],
        columns: int,
    ) -> tuple[
        tuple[tuple[FitRowProvenance, ...], ...],
        tuple[tuple[FitRowProvenance, ...], ...],
    ]:
        lineages_by_fold = {
            fold_id: FitRowProvenance(
                fit_row_ids=frozenset(
                    row_id
                    for row_id, candidate_fold in zip(train_row_ids, meta_inner_fold_ids)
                    if candidate_fold != fold_id
                )
            )
            for fold_id in dict.fromkeys(meta_inner_fold_ids)
        }
        train = tuple(
            tuple(lineages_by_fold[fold_id] for _ in range(columns))
            for fold_id in meta_inner_fold_ids
        )
        full = FitRowProvenance(fit_row_ids=frozenset(train_row_ids))
        heldout = tuple(tuple(full for _ in range(columns)) for _ in heldout_row_ids)
        return train, heldout

    @staticmethod
    def _make_banks(
        *,
        train_row_ids: tuple[int, ...],
        heldout_row_ids: tuple[int, ...],
        meta_inner_fold_ids: tuple[Hashable, ...],
        schema: _OutputSchema,
        source_train: np.ndarray,
        source_heldout: np.ndarray,
        feature_train: np.ndarray,
        feature_heldout: np.ndarray,
    ) -> tuple[AuthenticatedCalibratedTauBank, AuthenticatedRoleAwareFeatureBank]:
        source_train_lineage, source_heldout_lineage = FinalContextFitUpstreamProducer._lineages(
            train_row_ids=train_row_ids,
            heldout_row_ids=heldout_row_ids,
            meta_inner_fold_ids=meta_inner_fold_ids,
            columns=len(schema.source_names),
        )
        feature_train_lineage, feature_heldout_lineage = FinalContextFitUpstreamProducer._lineages(
            train_row_ids=train_row_ids,
            heldout_row_ids=heldout_row_ids,
            meta_inner_fold_ids=meta_inner_fold_ids,
            columns=len(schema.feature_names),
        )
        source_digest = _bank_digest(
            bank_type="calibrated_tau_sources",
            train_row_ids=train_row_ids,
            heldout_row_ids=heldout_row_ids,
            meta_fold_ids=meta_inner_fold_ids,
            names=schema.source_names,
            kinds=schema.source_kinds,
            roles=None,
            train_values=source_train,
            heldout_values=source_heldout,
            train_provenance=source_train_lineage,
            heldout_provenance=source_heldout_lineage,
        )
        feature_digest = _bank_digest(
            bank_type="role_aware_raw_features",
            train_row_ids=train_row_ids,
            heldout_row_ids=heldout_row_ids,
            meta_fold_ids=meta_inner_fold_ids,
            names=schema.feature_names,
            kinds=schema.feature_kinds,
            roles=schema.feature_roles,
            train_values=feature_train,
            heldout_values=feature_heldout,
            train_provenance=feature_train_lineage,
            heldout_provenance=feature_heldout_lineage,
        )
        source = AuthenticatedCalibratedTauBank(
            train_row_ids=train_row_ids,
            heldout_row_ids=heldout_row_ids,
            meta_inner_fold_ids=meta_inner_fold_ids,
            source_names=schema.source_names,
            source_kinds=schema.source_kinds,
            train_oof_values=source_train,
            outer_heldout_values=source_heldout,
            train_oof_fit_row_provenance=source_train_lineage,
            outer_heldout_fit_row_provenance=source_heldout_lineage,
            content_sha256=source_digest,
        )
        features = AuthenticatedRoleAwareFeatureBank(
            train_row_ids=train_row_ids,
            heldout_row_ids=heldout_row_ids,
            meta_inner_fold_ids=meta_inner_fold_ids,
            feature_names=schema.feature_names,
            feature_kinds=schema.feature_kinds,
            consumer_roles=schema.feature_roles,
            train_oof_values=feature_train,
            outer_heldout_values=feature_heldout,
            train_oof_fit_row_provenance=feature_train_lineage,
            outer_heldout_fit_row_provenance=feature_heldout_lineage,
            content_sha256=feature_digest,
        )
        return source, features

    def _load_package(
        self,
        *,
        manifest_path: Path,
        cache_key: str,
        binding: Mapping[str, Any],
        train_row_ids: tuple[int, ...],
        heldout_row_ids: tuple[int, ...],
        meta_inner_fold_ids: tuple[Hashable, ...],
    ) -> AuthenticatedFinalContextFitUpstreamBank:
        payload = _read_manifest(manifest_path)
        if payload["cache_key"] != cache_key or payload["binding"] != binding:
            raise ValueError("final upstream cache binding mismatch")
        source_record = payload["calibrated_sources"]
        feature_record = payload["raw_features"]
        if not isinstance(source_record, Mapping) or set(source_record) != _SOURCE_FIELDS:
            raise ValueError("calibrated-source cache metadata has an invalid schema")
        if not isinstance(feature_record, Mapping) or set(feature_record) != _FEATURE_FIELDS:
            raise ValueError("raw-feature cache metadata has an invalid schema")
        schema = _OutputSchema(
            source_names=_safe_names(source_record["names"], name="cached source names"),
            source_kinds=_aligned_metadata(
                source_record["kinds"],
                name="cached source kinds",
                length=len(source_record["names"]),
            ),
            feature_names=_safe_names(feature_record["names"], name="cached feature names"),
            feature_kinds=_aligned_metadata(
                feature_record["kinds"],
                name="cached feature kinds",
                length=len(feature_record["names"]),
            ),
            feature_roles=_aligned_metadata(
                feature_record["roles"],
                name="cached feature roles",
                length=len(feature_record["names"]),
            ),
        )
        matrices = payload["matrix_files"]
        if not isinstance(matrices, Mapping) or set(matrices) != set(_MATRIX_FILENAMES):
            raise ValueError("final upstream matrix cache has an invalid schema")
        loaded: dict[str, np.ndarray] = {}
        for name, filename in _MATRIX_FILENAMES.items():
            record = matrices[name]
            if not isinstance(record, Mapping) or set(record) != _FILE_RECORD_FIELDS:
                raise ValueError(f"cached matrix record {name!r} has an invalid schema")
            if record["filename"] != filename:
                raise ValueError(f"cached matrix record {name!r} changed filename")
            path = manifest_path.parent / filename
            if not path.is_file() or _sha256_file(path) != record["sha256"]:
                raise ValueError(f"cached matrix record {name!r} failed authentication")
            with path.open("rb") as handle:
                loaded[name] = np.load(handle, allow_pickle=False)
        source, features = self._make_banks(
            train_row_ids=train_row_ids,
            heldout_row_ids=heldout_row_ids,
            meta_inner_fold_ids=meta_inner_fold_ids,
            schema=schema,
            source_train=loaded["source_train_oof"],
            source_heldout=loaded["source_outer_heldout"],
            feature_train=loaded["feature_train_oof"],
            feature_heldout=loaded["feature_outer_heldout"],
        )
        if source.content_sha256 != source_record["content_sha256"]:
            raise ValueError("cached calibrated-source content digest changed")
        if features.content_sha256 != feature_record["content_sha256"]:
            raise ValueError("cached raw-feature content digest changed")
        package = AuthenticatedFinalContextFitUpstreamBank(
            outer_fold=int(binding["outer_fold"]),
            cache_key=cache_key,
            producer_identity_sha256=str(binding["producer_identity_sha256"]),
            manifest_path=manifest_path,
            manifest_sha256=_sha256_file(manifest_path),
            calibrated_sources=source,
            raw_features=features,
        )
        package.verify_authenticated_content()
        return package

    def _write_package(
        self,
        *,
        artifact_dir: Path,
        cache_key: str,
        binding: Mapping[str, Any],
        source: AuthenticatedCalibratedTauBank,
        features: AuthenticatedRoleAwareFeatureBank,
    ) -> AuthenticatedFinalContextFitUpstreamBank:
        if artifact_dir.exists():
            raise FileExistsError(
                "refusing to overwrite an incomplete or unauthenticated final upstream cache"
            )
        artifact_dir.mkdir(parents=True, exist_ok=False)
        arrays = {
            "source_train_oof": source.train_oof_values,
            "source_outer_heldout": source.outer_heldout_values,
            "feature_train_oof": features.train_oof_values,
            "feature_outer_heldout": features.outer_heldout_values,
        }
        records: dict[str, Any] = {}
        try:
            for name, filename in _MATRIX_FILENAMES.items():
                destination = artifact_dir / filename
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=artifact_dir, prefix=f".{filename}.", delete=False
                ) as handle:
                    np.save(handle, np.asarray(arrays[name], dtype=np.float64), allow_pickle=False)
                    temporary = Path(handle.name)
                try:
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                records[name] = {
                    "filename": filename,
                    "sha256": _sha256_file(destination),
                }
            content = {
                "schema_version": FINAL_CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA,
                "cache_key": cache_key,
                "binding": binding,
                "calibrated_sources": {
                    "names": list(source.source_names),
                    "kinds": list(source.source_kinds),
                    "content_sha256": source.content_sha256,
                },
                "raw_features": {
                    "names": list(features.feature_names),
                    "kinds": list(features.feature_kinds),
                    "roles": list(features.consumer_roles),
                    "content_sha256": features.content_sha256,
                },
                "matrix_files": records,
            }
            payload = {**content, "content_sha256": _sha256_json(content)}
            manifest_path = artifact_dir / "manifest.json"
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=artifact_dir,
                prefix=".manifest.",
                delete=False,
            ) as handle:
                handle.write(_canonical_json(payload) + "\n")
                temporary_manifest = Path(handle.name)
            try:
                temporary_manifest.replace(manifest_path)
            finally:
                temporary_manifest.unlink(missing_ok=True)
        except Exception:
            # Preserve a failed directory as evidence.  A retry must fail closed
            # instead of blessing a mixture of old and newly generated bytes.
            raise
        package = AuthenticatedFinalContextFitUpstreamBank(
            outer_fold=int(binding["outer_fold"]),
            cache_key=cache_key,
            producer_identity_sha256=str(binding["producer_identity_sha256"]),
            manifest_path=manifest_path,
            manifest_sha256=_sha256_file(manifest_path),
            calibrated_sources=source,
            raw_features=features,
        )
        package.verify_authenticated_content()
        return package

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
    ) -> AuthenticatedFinalContextFitUpstreamBank:
        """Produce strict OOF/final banks without accepting heldout labels."""

        self._assert_backend_stable()
        fold = _positive_int(outer_fold, name="outer_fold")
        train_rows = _integer_rows(outer_train_row_ids, name="outer_train_row_ids")
        heldout_rows = _integer_rows(outer_heldout_row_ids, name="outer_heldout_row_ids")
        if set(train_rows) & set(heldout_rows):
            raise ValueError("outer train and outer heldout row IDs must be disjoint")
        train_texts = _exact_texts(
            outer_train_texts, name="outer_train_texts", length=len(train_rows)
        )
        heldout_texts = _exact_texts(
            outer_heldout_texts, name="outer_heldout_texts", length=len(heldout_rows)
        )
        treatment = _finite_vector(
            outer_train_treatment,
            name="outer_train_treatment",
            length=len(train_rows),
        )
        if set(np.unique(treatment).tolist()) != {0.0, 1.0}:
            raise ValueError("outer_train_treatment must contain binary 0/1 values")
        outcome = _finite_vector(
            outer_train_outcome, name="outer_train_outcome", length=len(train_rows)
        )
        folds = _fold_ids(meta_inner_fold_ids, length=len(train_rows))
        _expected_fit_rows_by_position(train_rows, folds)
        binding = self._binding(
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
        artifact_dir = self.cache_dir / "artifacts" / cache_key
        manifest_path = artifact_dir / "manifest.json"
        if artifact_dir.exists():
            if not manifest_path.is_file():
                raise ValueError("final upstream cache directory has no authenticated manifest")
            return self._load_package(
                manifest_path=manifest_path,
                cache_key=cache_key,
                binding=binding,
                train_row_ids=train_rows,
                heldout_row_ids=heldout_rows,
                meta_inner_fold_ids=folds,
            )

        unique_folds = tuple(dict.fromkeys(folds))
        schema: _OutputSchema | None = None
        source_train: np.ndarray | None = None
        feature_train: np.ndarray | None = None
        for fold_index, fold_id in enumerate(unique_folds, start=1):
            fit_positions = [index for index, value in enumerate(folds) if value != fold_id]
            prediction_positions = [index for index, value in enumerate(folds) if value == fold_id]
            prediction = self._call_backend(
                outer_fold=fold,
                context_row_ids=tuple(train_rows[index] for index in fit_positions),
                context_texts=tuple(train_texts[index] for index in fit_positions),
                context_treatment=treatment[fit_positions],
                context_outcome=outcome[fit_positions],
                prediction_row_ids=tuple(train_rows[index] for index in prediction_positions),
                prediction_texts=tuple(train_texts[index] for index in prediction_positions),
                work_dir=self.cache_dir
                / "backend_work"
                / cache_key
                / f"meta_fold_{fold_index:03d}",
            )
            candidate_schema = _OutputSchema.from_prediction(prediction)
            if schema is None:
                schema = candidate_schema
                source_train = np.full(
                    (len(train_rows), len(schema.source_names)), np.nan, dtype=float
                )
                feature_train = np.full(
                    (len(train_rows), len(schema.feature_names)), np.nan, dtype=float
                )
            elif candidate_schema != schema:
                raise ValueError(
                    "upstream source names/kinds or feature names/kinds/roles changed "
                    "across meta-inner fits"
                )
            assert source_train is not None and feature_train is not None
            source_train[prediction_positions, :] = prediction.calibrated_source_values
            feature_train[prediction_positions, :] = prediction.feature_values

        if schema is None or source_train is None or feature_train is None:
            raise RuntimeError("final upstream producer completed no meta-inner fits")
        if not np.isfinite(source_train).all() or not np.isfinite(feature_train).all():
            raise RuntimeError("final upstream OOF matrices are incomplete")
        heldout_prediction = self._call_backend(
            outer_fold=fold,
            context_row_ids=train_rows,
            context_texts=train_texts,
            context_treatment=treatment,
            context_outcome=outcome,
            prediction_row_ids=heldout_rows,
            prediction_texts=heldout_texts,
            work_dir=self.cache_dir / "backend_work" / cache_key / "full_outer_train",
        )
        if _OutputSchema.from_prediction(heldout_prediction) != schema:
            raise ValueError(
                "upstream source names/kinds or feature names/kinds/roles changed on "
                "the full-outer-train fit"
            )
        source, features = self._make_banks(
            train_row_ids=train_rows,
            heldout_row_ids=heldout_rows,
            meta_inner_fold_ids=folds,
            schema=schema,
            source_train=source_train,
            source_heldout=heldout_prediction.calibrated_source_values,
            feature_train=feature_train,
            feature_heldout=heldout_prediction.feature_values,
        )
        return self._write_package(
            artifact_dir=artifact_dir,
            cache_key=cache_key,
            binding=binding,
            source=source,
            features=features,
        )


__all__ = [
    "FINAL_CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA",
    "FINAL_CONTEXT_FIT_UPSTREAM_PRODUCER_ID",
    "AuthenticatedCalibratedTauBank",
    "AuthenticatedFinalContextFitUpstreamBank",
    "AuthenticatedRoleAwareFeatureBank",
    "FinalContextFitUpstreamProducer",
]
