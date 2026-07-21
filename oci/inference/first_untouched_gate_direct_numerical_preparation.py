"""Prepare the direct numerical channel for the first untouched review gate.

This module is deliberately upstream of feature discovery.  It fits the shared
context-only Stage-1 provider on the initial spent rows, predicts the first
gate from text and row IDs only, authenticates the resulting cache, and writes
the coordinate-preserving direct-numerical manifest.  Gate treatment and
outcome are absent from the public API and therefore cannot cross this
boundary.

The returned bound provider is the exact object that later consumers use for
gate source and feature-bank views.  Rebinding the fold is neither necessary
nor allowed by this helper.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    TFIDF_SEMANTIC_RETRIEVAL,
)
from .all_evidence_post_extraction_review import ObservableCausalRows
from .context_fit_upstream_gate_provider import (
    CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION,
    BoundContextFitUpstreamGateProvider,
    ContextFitUpstreamGateProvider,
)
from .context_fit_upstream_cache_overlay import (
    AuthenticatedContextFitGateCacheOverlay,
)
from .coordinate_preserving_context_fit_upstream_backend import (
    COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID,
)
from .direct_upstream_numerical_manifest import (
    SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON,
    PersistedDirectNumericalManifest,
    build_direct_upstream_numerical_manifest,
    canonical_json,
    content_sha256,
    load_authenticated_numerical_bank_snapshot,
    write_direct_upstream_numerical_manifest,
)
from .lossless_stage1_evidence_catalog import (
    RoleNeutralEvidenceCatalog,
    validate_role_neutral_catalog,
)

FIRST_UNTOUCHED_GATE_PREPARATION_VERSION = "first_untouched_gate_direct_numerical_preparation_v1"
FIRST_UNTOUCHED_GATE_PREPARATION_AUDIT_VERSION = (
    "first_untouched_gate_direct_numerical_preparation_audit_v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CACHE_BINDING_KEYS = frozenset(
    {
        "provider_identity",
        "outer_fold",
        "context_row_ids_sha256",
        "context_text_sha256",
        "context_treatment_sha256",
        "context_outcome_sha256",
        "context_inner_fold_assignment_sha256",
        "gate_row_ids_sha256",
        "gate_text_sha256",
        "context_row_count",
        "gate_row_count",
        "gate_labels_in_binding",
        "gate_labels_exposed_to_backend",
        "context_values_cross_fitted_by_exact_inner_fold",
    }
)
_BOUND_IDENTITY_KEYS = frozenset(
    {
        "provider",
        "outer_fold",
        "gate_row_ids_sha256",
        "parent_identity_sha256",
        "cache_manifest_sha256",
    }
)
_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "preparation_version",
        "implementation_file_sha256",
        "outer_fold",
        "bounds",
        "initial_spent_binding",
        "first_untouched_gate_binding",
        "semantic_catalog_binding",
        "upstream_cache_binding",
        "direct_numerical_manifest_binding",
        "assurances",
    }
)
_INITIAL_SPENT_AUDIT_KEYS = frozenset(
    {
        "row_count",
        "row_ids_sha256",
        "text_sha256",
        "treatment_sha256",
        "outcome_sha256",
        "inner_fold_assignment_sha256",
        "placeholder_extraction_columns",
    }
)
_GATE_AUDIT_KEYS = frozenset(
    {
        "row_count",
        "row_ids_sha256",
        "text_sha256",
        "treatment_accepted",
        "outcome_accepted",
        "labels_in_cache_binding",
        "labels_exposed_to_backend",
    }
)
_CATALOG_AUDIT_KEYS = frozenset(
    {
        "catalog_sha256",
        "scope",
        "inner_fold",
        "split_fingerprint",
        "atom_count",
        "family_bindings",
        "all_active_architectures_bound",
    }
)
_FAMILY_SEMANTIC_AUDIT_KEYS = frozenset(
    {"source_family", "semantic_atom_count", "semantic_atom_ids_sha256"}
)
_UPSTREAM_AUDIT_KEYS = frozenset(
    {
        "provider_identity_sha256",
        "bind_provider_kind",
        "bind_provider_identity",
        "bind_provider_identity_sha256",
        "raw_delegate_provider_identity",
        "raw_delegate_provider_identity_sha256",
        "authenticated_overlay_used",
        "coordinate_preserving_backend",
        "bind_fold_invocation_count",
        "bound_provider_identity_sha256",
        "source_cache_schema",
        "source_cache_key",
        "source_manifest_sha256",
        "producer_identity_sha256",
        "stable_output_schema_sha256",
        "shared_lineage_sha256",
        "lineage_scope",
        "source_cache_materialized_before_discovery",
        "bound_provider_returned_for_gate_view_reuse",
    }
)
_DIRECT_AUDIT_KEYS = frozenset(
    {
        "content_sha256",
        "file_sha256",
        "canonical_filename",
        "signal_count",
        "family_signal_counts",
        "semantic_retrieval_numerical_zero_reason",
        "other_numerical_zero_reasons_present",
        "coordinate_to_semantic_atom_linkage",
        "concept_grounding_allowed",
    }
)
_FAMILY_SIGNAL_AUDIT_KEYS = frozenset({"source_family", "signal_count"})
_ASSURANCE_AUDIT_KEYS = frozenset(
    {
        "gate_treatment_parameter_exists",
        "gate_outcome_parameter_exists",
        "gate_labels_used_for_fit_or_cache",
        "initial_spent_labels_used_for_context_fit_only",
        "provider_bound_exactly_once",
        "later_gate_views_reuse_returned_bound_provider",
        "raw_matrix_values_exposed_to_discovery",
        "direct_coordinate_metadata_exposed_to_discovery",
    }
)
_MATRIX_FILE_FIELDS = (
    "source_values_file",
    "source_context_values_file",
    "feature_values_file",
    "feature_context_values_file",
)


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _closed_clone(value: Any, *, label: str) -> Any:
    try:
        encoded = canonical_json(value)
        result = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain finite closed JSON") from exc
    if isinstance(result, Mapping):
        return dict(result)
    return result


def _exact_integer_rows(values: Sequence[Any], *, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{label} must be a sequence of integer row IDs")
    rows: list[int] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{label} must contain canonical integer row IDs")
        normalized = int(value)
        if normalized < 0:
            raise ValueError(f"{label} cannot contain negative row IDs")
        rows.append(normalized)
    if not rows or len(rows) != len(set(rows)):
        raise ValueError(f"{label} must be non-empty and unique")
    return tuple(rows)


def _exact_texts(values: Sequence[Any], *, label: str, length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{label} must be a sequence of exact strings")
    result = tuple(values)
    if len(result) != int(length) or any(not isinstance(value, str) for value in result):
        raise ValueError(f"{label} must contain exactly {length} strings")
    try:
        for value in result:
            value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must contain valid UTF-8 text") from exc
    return result


def _exact_inner_folds(values: Sequence[Any], *, length: int) -> tuple[Hashable, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("initial_spent_inner_fold_ids must be a sequence")
    result: list[Hashable] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, str)):
            raise TypeError("initial_spent_inner_fold_ids must contain integer or string IDs")
        normalized: Hashable = int(value) if isinstance(value, (int, np.integer)) else value
        if isinstance(normalized, str) and not normalized:
            raise ValueError("initial_spent_inner_fold_ids cannot contain empty strings")
        result.append(normalized)
    if len(result) != int(length) or len(set(result)) < 2:
        raise ValueError("initial_spent_inner_fold_ids must define at least two aligned folds")
    return tuple(result)


def _float_hex_sha256(values: np.ndarray) -> str:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("observable vectors must be finite and one-dimensional")
    return content_sha256([float(value).hex() for value in vector])


def _text_bytes(*groups: Sequence[str]) -> int:
    return sum(len(value.encode("utf-8")) for group in groups for value in group)


def _json_object_no_duplicates(payload: bytes, *, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite JSON: {value}")

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return result


@dataclass(frozen=True)
class FirstUntouchedGatePreparationBounds:
    """Fixed resource limits applied before unbounded cache parsing."""

    max_initial_spent_rows: int = 1_000_000
    max_first_gate_rows: int = 1_000_000
    max_total_text_utf8_bytes: int = 1_073_741_824
    max_catalog_atoms: int = 100_000
    max_source_manifest_bytes: int = 16_777_216
    max_direct_numerical_signals: int = 16_384
    max_single_matrix_file_bytes: int = 1_073_741_824
    max_total_matrix_file_bytes: int = 4_294_967_296

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            _positive_integer(value, label=name)
        if self.max_single_matrix_file_bytes > self.max_total_matrix_file_bytes:
            raise ValueError(
                "max_single_matrix_file_bytes cannot exceed max_total_matrix_file_bytes"
            )

    def as_dict(self) -> dict[str, int]:
        return {
            "max_initial_spent_rows": self.max_initial_spent_rows,
            "max_first_gate_rows": self.max_first_gate_rows,
            "max_total_text_utf8_bytes": self.max_total_text_utf8_bytes,
            "max_catalog_atoms": self.max_catalog_atoms,
            "max_source_manifest_bytes": self.max_source_manifest_bytes,
            "max_direct_numerical_signals": self.max_direct_numerical_signals,
            "max_single_matrix_file_bytes": self.max_single_matrix_file_bytes,
            "max_total_matrix_file_bytes": self.max_total_matrix_file_bytes,
        }


def _provider_identity(provider: ContextFitUpstreamGateProvider) -> dict[str, Any]:
    identity = _closed_clone(provider.identity(), label="provider identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("provider identity must be one non-empty JSON object")
    backend = identity.get("backend")
    if not isinstance(backend, Mapping) or backend.get("backend") != (
        COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID
    ):
        raise ValueError(
            "first-gate direct numerical preparation requires the "
            "coordinate-preserving v3 backend"
        )
    return identity


def _raw_provider_and_wrapper_identity(
    provider: ContextFitUpstreamGateProvider | AuthenticatedContextFitGateCacheOverlay,
) -> tuple[ContextFitUpstreamGateProvider, dict[str, Any] | None]:
    """Return the exact raw delegate plus an optional immutable overlay identity."""

    if isinstance(provider, ContextFitUpstreamGateProvider):
        return provider, None
    if not isinstance(provider, AuthenticatedContextFitGateCacheOverlay):
        raise TypeError(
            "provider must be the shared ContextFitUpstreamGateProvider or its "
            "authenticated read-only cache overlay"
        )
    if provider.hierarchical_first_gate_preparation is not True:
        raise ValueError(
            "authenticated gate overlay must explicitly allow label-free "
            "hierarchical first-gate preparation"
        )
    raw = provider.provider
    if not isinstance(raw, ContextFitUpstreamGateProvider):
        raise TypeError("authenticated gate overlay has the wrong raw delegate")
    wrapper_identity = _closed_clone(provider.identity(), label="gate overlay identity")
    if not isinstance(wrapper_identity, dict) or not wrapper_identity:
        raise ValueError("gate overlay identity must be one non-empty JSON object")
    return raw, wrapper_identity


def _semantic_atom_bindings(
    catalog: RoleNeutralEvidenceCatalog,
) -> dict[str, tuple[str, ...]]:
    bindings = {
        family: tuple(atom.evidence_id for atom in catalog.family_atoms(family))
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    if set(bindings) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET:
        raise RuntimeError("semantic catalog family binding is incomplete")
    if any(not values for values in bindings.values()):
        missing = [family for family, values in bindings.items() if not values]
        raise ValueError(
            "first-gate semantic catalog must contain every active architecture: "
            f"missing={missing}"
        )
    flattened = [evidence_id for values in bindings.values() for evidence_id in values]
    if len(flattened) != len(set(flattened)) or set(flattened) != {
        atom.evidence_id for atom in catalog.atoms
    }:
        raise ValueError("semantic family bindings are not an exact catalog partition")
    return bindings


def _expected_cache_binding(
    *,
    provider_identity: Mapping[str, Any],
    outer_fold: int,
    context: ObservableCausalRows,
    context_texts: tuple[str, ...],
    gate_row_ids: tuple[int, ...],
    gate_texts: tuple[str, ...],
    inner_fold_ids: tuple[Hashable, ...],
) -> dict[str, Any]:
    return {
        "provider_identity": _closed_clone(provider_identity, label="provider identity"),
        "outer_fold": outer_fold,
        "context_row_ids_sha256": content_sha256(list(context.row_ids)),
        "context_text_sha256": content_sha256(list(context_texts)),
        "context_treatment_sha256": _float_hex_sha256(context.treatment),
        "context_outcome_sha256": _float_hex_sha256(context.outcome),
        "context_inner_fold_assignment_sha256": content_sha256(
            {
                "row_ids": list(context.row_ids),
                "inner_fold_ids": list(inner_fold_ids),
            }
        ),
        "gate_row_ids_sha256": content_sha256(list(gate_row_ids)),
        "gate_text_sha256": content_sha256(list(gate_texts)),
        "context_row_count": len(context.row_ids),
        "gate_row_count": len(gate_row_ids),
        "gate_labels_in_binding": False,
        "gate_labels_exposed_to_backend": False,
        "context_values_cross_fitted_by_exact_inner_fold": True,
    }


def _authenticate_bound_identity(
    bound: BoundContextFitUpstreamGateProvider,
    *,
    outer_fold: int,
    gate_row_ids: tuple[int, ...],
    provider_identity_sha256: str,
) -> dict[str, Any]:
    identity = _closed_clone(bound.identity(), label="bound provider identity")
    if not isinstance(identity, dict) or set(identity) != _BOUND_IDENTITY_KEYS:
        raise ValueError("bound provider identity has an unexpected closed schema")
    if identity["outer_fold"] != outer_fold:
        raise ValueError("bound provider changed the outer fold")
    if identity["gate_row_ids_sha256"] != content_sha256(list(gate_row_ids)):
        raise ValueError("bound provider changed first-gate row identity/order")
    if identity["parent_identity_sha256"] != provider_identity_sha256:
        raise ValueError("bound provider parent identity differs from the shared provider")
    _require_sha256(identity["cache_manifest_sha256"], label="cache_manifest_sha256")
    return identity


def _authenticate_cache_payload(
    *,
    path: Path,
    expected_manifest_sha256: str,
    expected_binding: Mapping[str, Any],
    context_row_ids: tuple[int, ...],
    inner_fold_ids: tuple[Hashable, ...],
    gate_row_ids: tuple[int, ...],
    bounds: FirstUntouchedGatePreparationBounds,
) -> dict[str, Any]:
    size = path.stat().st_size
    if size < 1 or size > bounds.max_source_manifest_bytes:
        raise ValueError("bound upstream source manifest exceeds its byte bound")
    payload_bytes = path.read_bytes()
    if _sha256_bytes(payload_bytes) != expected_manifest_sha256:
        raise ValueError("bound upstream source manifest failed byte authentication")
    payload = _json_object_no_duplicates(payload_bytes, label="bound upstream source manifest")
    if payload.get("schema_version") != CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION:
        raise ValueError("first-gate preparation requires the context-fit gate cache schema")
    binding = payload.get("binding")
    if not isinstance(binding, Mapping) or set(binding) != _CACHE_BINDING_KEYS:
        raise ValueError("bound upstream cache binding has an unexpected closed schema")
    if _closed_clone(binding, label="cache binding") != _closed_clone(
        expected_binding, label="expected cache binding"
    ):
        raise ValueError("bound upstream cache binding differs from exact preparation inputs")
    if payload.get("cache_key") != content_sha256(binding):
        raise ValueError("bound upstream cache key does not authenticate its binding")
    if tuple(payload.get("context_row_ids") or ()) != context_row_ids:
        raise ValueError("bound upstream cache changed initial-spent row identity/order")
    if tuple(payload.get("context_inner_fold_ids") or ()) != inner_fold_ids:
        raise ValueError("bound upstream cache changed inner-fold identity/order")
    if tuple(payload.get("gate_row_ids") or ()) != gate_row_ids:
        raise ValueError("bound upstream cache changed first-gate row identity/order")

    total_matrix_bytes = 0
    for field_name in _MATRIX_FILE_FIELDS:
        filename = payload.get(field_name)
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("bound upstream cache contains a non-canonical matrix filename")
        matrix_path = path.parent / filename
        if not matrix_path.is_file():
            raise ValueError("bound upstream cache matrix is missing")
        matrix_bytes = matrix_path.stat().st_size
        if matrix_bytes < 1 or matrix_bytes > bounds.max_single_matrix_file_bytes:
            raise ValueError("bound upstream cache matrix exceeds its per-file byte bound")
        total_matrix_bytes += matrix_bytes
    if total_matrix_bytes > bounds.max_total_matrix_file_bytes:
        raise ValueError("bound upstream cache matrices exceed their total byte bound")

    source_count = len(payload.get("source_names") or ())
    feature_count = len(payload.get("feature_names") or ())
    if source_count + feature_count > bounds.max_direct_numerical_signals:
        raise ValueError("bound upstream numerical schema exceeds its signal-count bound")
    return payload


@dataclass(frozen=True)
class PreparedFirstUntouchedGateDirectNumerical:
    """Authenticated preparation result with a reusable bound provider."""

    bound_provider: BoundContextFitUpstreamGateProvider = field(repr=False)
    persisted_manifest: PersistedDirectNumericalManifest
    audit_sha256: str
    _audit_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bound_provider, BoundContextFitUpstreamGateProvider):
            raise TypeError("bound_provider must be BoundContextFitUpstreamGateProvider")
        if not isinstance(self.persisted_manifest, PersistedDirectNumericalManifest):
            raise TypeError("persisted_manifest must be PersistedDirectNumericalManifest")
        _require_sha256(self.audit_sha256, label="audit_sha256")
        self.verify()

    @property
    def audit(self) -> dict[str, Any]:
        result = _json_object_no_duplicates(
            self._audit_json.encode("utf-8"), label="first-gate preparation audit"
        )
        return _closed_clone(result, label="first-gate preparation audit")

    def verify(self) -> None:
        audit = self.audit
        if set(audit) != _AUDIT_KEYS:
            raise ValueError("first-gate preparation audit has an unexpected closed schema")
        if audit.get("schema_version") != FIRST_UNTOUCHED_GATE_PREPARATION_AUDIT_VERSION:
            raise ValueError("first-gate preparation audit schema changed")
        if audit.get("preparation_version") != FIRST_UNTOUCHED_GATE_PREPARATION_VERSION:
            raise ValueError("first-gate preparation implementation version changed")
        if audit.get("implementation_file_sha256") != _module_sha256():
            raise ValueError("first-gate preparation implementation bytes changed")
        if content_sha256(audit) != self.audit_sha256:
            raise ValueError("first-gate preparation audit SHA-256 mismatch")

        self.persisted_manifest.verify()
        direct = self.persisted_manifest.manifest
        initial_binding = audit.get("initial_spent_binding")
        direct_binding = audit.get("direct_numerical_manifest_binding")
        upstream_binding = audit.get("upstream_cache_binding")
        gate_binding = audit.get("first_untouched_gate_binding")
        catalog_binding = audit.get("semantic_catalog_binding")
        bounds_binding = audit.get("bounds")
        assurances = audit.get("assurances")
        if not all(
            isinstance(value, Mapping)
            for value in (
                initial_binding,
                direct_binding,
                upstream_binding,
                gate_binding,
                catalog_binding,
                bounds_binding,
                assurances,
            )
        ):
            raise ValueError("first-gate preparation audit contains malformed bindings")
        closed_bindings = (
            (initial_binding, _INITIAL_SPENT_AUDIT_KEYS, "initial-spent"),
            (gate_binding, _GATE_AUDIT_KEYS, "first-gate"),
            (catalog_binding, _CATALOG_AUDIT_KEYS, "semantic catalog"),
            (upstream_binding, _UPSTREAM_AUDIT_KEYS, "upstream cache"),
            (direct_binding, _DIRECT_AUDIT_KEYS, "direct manifest"),
            (assurances, _ASSURANCE_AUDIT_KEYS, "assurances"),
        )
        for binding, expected_keys, label in closed_bindings:
            if set(binding) != expected_keys:
                raise ValueError(f"{label} audit binding has an unexpected closed schema")
        if set(bounds_binding) != set(FirstUntouchedGatePreparationBounds().as_dict()):
            raise ValueError("preparation bounds audit has an unexpected closed schema")
        FirstUntouchedGatePreparationBounds(**bounds_binding)
        if initial_binding.get("placeholder_extraction_columns") != ["_oci_row_id"]:
            raise ValueError("initial-spent extraction is not placeholder-only")
        if any(
            gate_binding.get(key) is not False
            for key in (
                "treatment_accepted",
                "outcome_accepted",
                "labels_in_cache_binding",
                "labels_exposed_to_backend",
            )
        ):
            raise ValueError("first-gate audit claims that a gate-label channel exists")
        if (
            catalog_binding.get("scope") != "inner_train"
            or catalog_binding.get("inner_fold") is None
        ):
            raise ValueError("semantic catalog audit is not spent-only inner_train evidence")
        family_bindings = catalog_binding.get("family_bindings")
        if not isinstance(family_bindings, list) or len(family_bindings) != len(
            ACTIVE_STAGE1_CONCEPT_FAMILIES
        ):
            raise ValueError("semantic catalog audit does not bind exactly ten architectures")
        for expected_family, family_binding in zip(ACTIVE_STAGE1_CONCEPT_FAMILIES, family_bindings):
            if not isinstance(family_binding, Mapping) or set(family_binding) != (
                _FAMILY_SEMANTIC_AUDIT_KEYS
            ):
                raise ValueError("semantic family audit has an unexpected closed schema")
            if family_binding.get("source_family") != expected_family:
                raise ValueError("semantic family audit changed architecture order")
            coverage = direct.family(expected_family)
            if family_binding.get("semantic_atom_count") != len(
                coverage.semantic_atom_ids
            ) or family_binding.get("semantic_atom_ids_sha256") != content_sha256(
                list(coverage.semantic_atom_ids)
            ):
                raise ValueError("semantic family audit differs from direct manifest")
        if catalog_binding.get("all_active_architectures_bound") is not True:
            raise ValueError("semantic catalog audit does not bind all architectures")
        if (
            upstream_binding.get("coordinate_preserving_backend")
            != (COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID)
            or upstream_binding.get("bind_fold_invocation_count") != 1
        ):
            raise ValueError("upstream audit changed the v3 single-bind policy")
        raw_identity = upstream_binding.get("raw_delegate_provider_identity")
        bind_identity = upstream_binding.get("bind_provider_identity")
        if not isinstance(raw_identity, Mapping) or not isinstance(bind_identity, Mapping):
            raise ValueError("upstream audit lost its bind/delegate identities")
        raw_identity_sha256 = content_sha256(raw_identity)
        bind_identity_sha256 = content_sha256(bind_identity)
        if (
            upstream_binding.get("raw_delegate_provider_identity_sha256") != raw_identity_sha256
            or upstream_binding.get("provider_identity_sha256") != raw_identity_sha256
            or upstream_binding.get("bind_provider_identity_sha256") != bind_identity_sha256
        ):
            raise ValueError("upstream bind/delegate identity authentication changed")
        bind_kind = upstream_binding.get("bind_provider_kind")
        overlay_used = upstream_binding.get("authenticated_overlay_used")
        if bind_kind == "raw_context_fit_upstream_gate_provider":
            if overlay_used is not False or bind_identity != raw_identity:
                raise ValueError("raw first-gate provider audit is inconsistent")
        elif bind_kind == "authenticated_context_fit_gate_cache_overlay":
            if overlay_used is not True:
                raise ValueError("authenticated overlay first-gate audit is inconsistent")
            if bind_identity.get("delegate_provider_identity") != raw_identity:
                raise ValueError("authenticated overlay audit cites a different delegate")
            if bind_identity.get("delegate_provider_identity_sha256") != (raw_identity_sha256):
                raise ValueError("authenticated overlay delegate digest changed")
            if (
                bind_identity.get(
                    "hierarchical_deferred_first_gate_materialization_allowed"
                )
                is not True
            ):
                raise ValueError("authenticated overlay audit lacks first-gate authorization")
        else:
            raise ValueError("upstream bind provider kind is unsupported")
        if (
            upstream_binding.get("source_cache_materialized_before_discovery") is not True
            or upstream_binding.get("bound_provider_returned_for_gate_view_reuse") is not True
        ):
            raise ValueError("upstream audit changed the preparation/reuse policy")
        if direct_binding.get("canonical_filename") != ("direct_upstream_numerical_manifest.json"):
            raise ValueError("direct manifest audit changed the canonical filename")
        family_signal_counts = direct_binding.get("family_signal_counts")
        if not isinstance(family_signal_counts, list) or len(family_signal_counts) != len(
            ACTIVE_STAGE1_CONCEPT_FAMILIES
        ):
            raise ValueError("direct manifest audit has incomplete family signal counts")
        for coverage, row in zip(direct.family_coverage, family_signal_counts):
            if not isinstance(row, Mapping) or set(row) != _FAMILY_SIGNAL_AUDIT_KEYS:
                raise ValueError("family signal audit has an unexpected closed schema")
            if row != {
                "source_family": coverage.source_family,
                "signal_count": len(coverage.coordinate_ids),
            }:
                raise ValueError("family signal audit differs from direct manifest")
        if direct_binding.get("signal_count") != direct.signal_count:
            raise ValueError("direct signal count differs from preparation audit")
        if direct_binding.get("semantic_retrieval_numerical_zero_reason") != (
            SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON
        ):
            raise ValueError("semantic-retrieval zero reason changed in preparation audit")
        if any(
            direct_binding.get(key) is not False
            for key in (
                "other_numerical_zero_reasons_present",
                "coordinate_to_semantic_atom_linkage",
                "concept_grounding_allowed",
            )
        ):
            raise ValueError("direct manifest audit claims a forbidden linkage")
        expected_assurances = {
            "gate_treatment_parameter_exists": False,
            "gate_outcome_parameter_exists": False,
            "gate_labels_used_for_fit_or_cache": False,
            "initial_spent_labels_used_for_context_fit_only": True,
            "provider_bound_exactly_once": True,
            "later_gate_views_reuse_returned_bound_provider": True,
            "raw_matrix_values_exposed_to_discovery": False,
            "direct_coordinate_metadata_exposed_to_discovery": False,
        }
        if assurances != expected_assurances:
            raise ValueError("first-gate preparation assurances changed")
        if direct_binding.get("content_sha256") != direct.content_sha256:
            raise ValueError("direct manifest content differs from preparation audit")
        if direct_binding.get("file_sha256") != self.persisted_manifest.file_sha256:
            raise ValueError("persisted direct manifest differs from preparation audit")
        if direct.semantic_catalog_sha256 != catalog_binding.get("catalog_sha256"):
            raise ValueError("direct manifest semantic catalog binding changed")
        if direct.source_manifest_sha256 != upstream_binding.get("source_manifest_sha256"):
            raise ValueError("direct manifest upstream source binding changed")
        for field_name, expected in (
            ("source_cache_schema", direct.source_cache_schema),
            ("source_cache_key", direct.source_cache_key),
            ("producer_identity_sha256", direct.producer_identity_sha256),
            ("stable_output_schema_sha256", direct.stable_output_schema_sha256),
            ("shared_lineage_sha256", direct.shared_lineage_sha256),
            ("lineage_scope", direct.lineage_scope),
        ):
            if upstream_binding.get(field_name) != expected:
                raise ValueError(f"direct manifest {field_name} differs from preparation audit")

        bound_identity = _closed_clone(
            self.bound_provider.identity(), label="bound provider identity"
        )
        if content_sha256(bound_identity) != upstream_binding.get("bound_provider_identity_sha256"):
            raise ValueError("bound provider identity changed after preparation")
        if self.bound_provider.outer_fold != audit.get("outer_fold"):
            raise ValueError("bound provider outer fold changed after preparation")
        if content_sha256(list(self.bound_provider.exact_gate_row_ids)) != gate_binding.get(
            "row_ids_sha256"
        ):
            raise ValueError("bound provider gate rows changed after preparation")
        source_path = self.bound_provider.authenticated_cache_manifest_path
        if _sha256_file(source_path) != upstream_binding.get("source_manifest_sha256"):
            raise ValueError("bound upstream source manifest changed after preparation")


def prepare_first_untouched_gate_direct_numerical(
    *,
    outer_fold: int,
    initial_spent_row_ids: Sequence[int],
    initial_spent_texts: Sequence[str],
    initial_spent_treatment: Sequence[float],
    initial_spent_outcome: Sequence[float],
    initial_spent_inner_fold_ids: Sequence[Hashable],
    first_gate_row_ids: Sequence[int],
    first_gate_texts: Sequence[str],
    catalog: RoleNeutralEvidenceCatalog,
    provider: ContextFitUpstreamGateProvider | AuthenticatedContextFitGateCacheOverlay,
    destination: Path | str,
    bounds: FirstUntouchedGatePreparationBounds | None = None,
) -> PreparedFirstUntouchedGateDirectNumerical:
    """Fit/cache one first gate and persist its v3 direct numerical manifest.

    Only initial-spent treatment and outcome are accepted.  The absence of
    first-gate label parameters is intentional and security relevant.
    """

    fold = _positive_integer(outer_fold, label="outer_fold")
    limits = bounds or FirstUntouchedGatePreparationBounds()
    if not isinstance(limits, FirstUntouchedGatePreparationBounds):
        raise TypeError("bounds must be FirstUntouchedGatePreparationBounds")
    context_ids = _exact_integer_rows(initial_spent_row_ids, label="initial_spent_row_ids")
    gate_ids = _exact_integer_rows(first_gate_row_ids, label="first_gate_row_ids")
    if set(context_ids) & set(gate_ids):
        raise ValueError("initial-spent and first-gate rows must be disjoint")
    if len(context_ids) > limits.max_initial_spent_rows:
        raise ValueError("initial-spent row count exceeds its preparation bound")
    if len(gate_ids) > limits.max_first_gate_rows:
        raise ValueError("first-gate row count exceeds its preparation bound")
    context_texts = _exact_texts(
        initial_spent_texts, label="initial_spent_texts", length=len(context_ids)
    )
    gate_texts = _exact_texts(first_gate_texts, label="first_gate_texts", length=len(gate_ids))
    if _text_bytes(context_texts, gate_texts) > limits.max_total_text_utf8_bytes:
        raise ValueError("spent and first-gate text exceeds its preparation byte bound")
    inner_folds = _exact_inner_folds(initial_spent_inner_fold_ids, length=len(context_ids))

    if not isinstance(catalog, RoleNeutralEvidenceCatalog):
        raise TypeError("catalog must be RoleNeutralEvidenceCatalog")
    catalog_json_before = canonical_json(catalog.as_dict())
    validate_role_neutral_catalog(catalog)
    if catalog.outer_fold != fold:
        raise ValueError("semantic catalog outer fold differs from preparation outer fold")
    if catalog.scope != "inner_train" or catalog.inner_fold is None:
        raise ValueError(
            "first untouched-gate preparation requires an inner_train spent-only catalog"
        )
    if len(catalog.atoms) > limits.max_catalog_atoms:
        raise ValueError("semantic catalog atom count exceeds its preparation bound")
    semantic_bindings = _semantic_atom_bindings(catalog)

    raw_provider, wrapper_identity_before = _raw_provider_and_wrapper_identity(provider)
    provider_identity_before = _provider_identity(raw_provider)
    provider_identity_sha256 = content_sha256(provider_identity_before)
    bind_provider_identity = (
        provider_identity_before if wrapper_identity_before is None else wrapper_identity_before
    )
    bind_provider_kind = (
        "raw_context_fit_upstream_gate_provider"
        if wrapper_identity_before is None
        else "authenticated_context_fit_gate_cache_overlay"
    )

    destination_path = Path(destination).resolve()
    if destination_path.name != "direct_upstream_numerical_manifest.json":
        raise ValueError("destination must use the canonical direct manifest filename")

    context = ObservableCausalRows(
        row_ids=context_ids,
        extracted=pd.DataFrame({"_oci_row_id": context_ids}),
        treatment=np.asarray(initial_spent_treatment, dtype=float),
        outcome=np.asarray(initial_spent_outcome, dtype=float),
        inner_fold_ids=inner_folds,
    )
    if tuple(context.extracted.columns) != ("_oci_row_id",):
        raise RuntimeError("first-gate context extraction is not placeholder-only")
    if tuple(map(int, context.extracted["_oci_row_id"].tolist())) != context_ids:
        raise RuntimeError("first-gate placeholder extraction changed row identity/order")
    expected_cache_binding = _expected_cache_binding(
        provider_identity=provider_identity_before,
        outer_fold=fold,
        context=context,
        context_texts=context_texts,
        gate_row_ids=gate_ids,
        gate_texts=gate_texts,
        inner_fold_ids=inner_folds,
    )

    # Exactly one bind.  All later source/feature views reuse this returned
    # bound provider and never refit or reopen the first gate.
    bound = provider.bind_fold(
        outer_fold=fold,
        context=context,
        context_texts=context_texts,
        gate_texts=gate_texts,
        exact_gate_row_ids=gate_ids,
    )
    if not isinstance(bound, BoundContextFitUpstreamGateProvider):
        raise TypeError("shared provider returned the wrong bound-provider type")

    provider_identity_after_bind = _provider_identity(raw_provider)
    if provider_identity_after_bind != provider_identity_before:
        raise ValueError("shared upstream provider identity changed during first-gate bind")
    if (
        wrapper_identity_before is not None
        and _closed_clone(provider.identity(), label="gate overlay identity")
        != wrapper_identity_before
    ):
        raise ValueError("authenticated gate overlay identity changed during first-gate bind")
    bound_identity = _authenticate_bound_identity(
        bound,
        outer_fold=fold,
        gate_row_ids=gate_ids,
        provider_identity_sha256=provider_identity_sha256,
    )
    source_path = bound.authenticated_cache_manifest_path
    source_manifest_sha256 = bound_identity["cache_manifest_sha256"]
    payload = _authenticate_cache_payload(
        path=source_path,
        expected_manifest_sha256=source_manifest_sha256,
        expected_binding=expected_cache_binding,
        context_row_ids=context_ids,
        inner_fold_ids=inner_folds,
        gate_row_ids=gate_ids,
        bounds=limits,
    )

    snapshot = load_authenticated_numerical_bank_snapshot(source_path)
    if snapshot.source_cache_schema != CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION:
        raise ValueError("authenticated first-gate snapshot has the wrong cache schema")
    if snapshot.source_manifest_sha256 != source_manifest_sha256:
        raise ValueError("authenticated snapshot differs from the bound source manifest")
    if snapshot.source_cache_key != payload.get("cache_key"):
        raise ValueError("authenticated snapshot changed the bound source cache key")
    if len(snapshot.calibrated_source_names) + len(snapshot.raw_feature_names) > (
        limits.max_direct_numerical_signals
    ):
        raise ValueError("authenticated numerical snapshot exceeds its signal-count bound")

    direct_manifest = build_direct_upstream_numerical_manifest(
        snapshot,
        semantic_catalog_sha256=catalog.catalog_sha256,
        semantic_atom_ids_by_family=semantic_bindings,
        numerical_zero_reasons={TFIDF_SEMANTIC_RETRIEVAL: SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON},
    )
    if direct_manifest.signal_count > limits.max_direct_numerical_signals:
        raise ValueError("direct numerical manifest exceeds its signal-count bound")
    zero_reason_families = {
        row.source_family for row in direct_manifest.family_coverage if row.numerical_zero_reason
    }
    if zero_reason_families != {TFIDF_SEMANTIC_RETRIEVAL}:
        raise ValueError("direct numerical zero reasons differ from the sole approved exception")
    if direct_manifest.family(TFIDF_SEMANTIC_RETRIEVAL).numerical_zero_reason != (
        SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON
    ):
        raise ValueError("semantic-retrieval numerical zero reason changed")
    if tuple(row.source_family for row in direct_manifest.family_coverage) != (
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    ):
        raise RuntimeError("direct numerical manifest changed active architecture order")
    for family, evidence_ids in semantic_bindings.items():
        if direct_manifest.family(family).semantic_atom_ids != evidence_ids:
            raise ValueError("direct manifest changed one semantic family atom binding")

    persisted = write_direct_upstream_numerical_manifest(direct_manifest, destination_path)
    persisted.verify()
    if bound.authenticated_cache_manifest_path != source_path:
        raise ValueError("bound source manifest path changed during direct-manifest writing")
    if _provider_identity(raw_provider) != provider_identity_before:
        raise ValueError("shared upstream provider identity changed during manifest writing")
    if (
        wrapper_identity_before is not None
        and _closed_clone(provider.identity(), label="gate overlay identity")
        != wrapper_identity_before
    ):
        raise ValueError("authenticated gate overlay identity changed during manifest writing")
    validate_role_neutral_catalog(catalog)
    if canonical_json(catalog.as_dict()) != catalog_json_before:
        raise ValueError("semantic catalog changed during first-gate preparation")

    audit = {
        "schema_version": FIRST_UNTOUCHED_GATE_PREPARATION_AUDIT_VERSION,
        "preparation_version": FIRST_UNTOUCHED_GATE_PREPARATION_VERSION,
        "implementation_file_sha256": _module_sha256(),
        "outer_fold": fold,
        "bounds": limits.as_dict(),
        "initial_spent_binding": {
            "row_count": len(context_ids),
            "row_ids_sha256": content_sha256(list(context_ids)),
            "text_sha256": content_sha256(list(context_texts)),
            "treatment_sha256": _float_hex_sha256(context.treatment),
            "outcome_sha256": _float_hex_sha256(context.outcome),
            "inner_fold_assignment_sha256": expected_cache_binding[
                "context_inner_fold_assignment_sha256"
            ],
            "placeholder_extraction_columns": ["_oci_row_id"],
        },
        "first_untouched_gate_binding": {
            "row_count": len(gate_ids),
            "row_ids_sha256": content_sha256(list(gate_ids)),
            "text_sha256": content_sha256(list(gate_texts)),
            "treatment_accepted": False,
            "outcome_accepted": False,
            "labels_in_cache_binding": False,
            "labels_exposed_to_backend": False,
        },
        "semantic_catalog_binding": {
            "catalog_sha256": catalog.catalog_sha256,
            "scope": catalog.scope,
            "inner_fold": catalog.inner_fold,
            "split_fingerprint": catalog.split_fingerprint,
            "atom_count": len(catalog.atoms),
            "family_bindings": [
                {
                    "source_family": family,
                    "semantic_atom_count": len(semantic_bindings[family]),
                    "semantic_atom_ids_sha256": content_sha256(list(semantic_bindings[family])),
                }
                for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
            ],
            "all_active_architectures_bound": True,
        },
        "upstream_cache_binding": {
            "provider_identity_sha256": provider_identity_sha256,
            "bind_provider_kind": bind_provider_kind,
            "bind_provider_identity": bind_provider_identity,
            "bind_provider_identity_sha256": content_sha256(bind_provider_identity),
            "raw_delegate_provider_identity": provider_identity_before,
            "raw_delegate_provider_identity_sha256": provider_identity_sha256,
            "authenticated_overlay_used": wrapper_identity_before is not None,
            "coordinate_preserving_backend": (
                COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID
            ),
            "bind_fold_invocation_count": 1,
            "bound_provider_identity_sha256": content_sha256(bound_identity),
            "source_cache_schema": snapshot.source_cache_schema,
            "source_cache_key": snapshot.source_cache_key,
            "source_manifest_sha256": snapshot.source_manifest_sha256,
            "producer_identity_sha256": snapshot.producer_identity_sha256,
            "stable_output_schema_sha256": snapshot.stable_output_schema_sha256,
            "shared_lineage_sha256": snapshot.shared_lineage_sha256,
            "lineage_scope": snapshot.lineage_scope,
            "source_cache_materialized_before_discovery": True,
            "bound_provider_returned_for_gate_view_reuse": True,
        },
        "direct_numerical_manifest_binding": {
            "content_sha256": direct_manifest.content_sha256,
            "file_sha256": persisted.file_sha256,
            "canonical_filename": persisted.path.name,
            "signal_count": direct_manifest.signal_count,
            "family_signal_counts": [
                {
                    "source_family": row.source_family,
                    "signal_count": len(row.coordinate_ids),
                }
                for row in direct_manifest.family_coverage
            ],
            "semantic_retrieval_numerical_zero_reason": (SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON),
            "other_numerical_zero_reasons_present": False,
            "coordinate_to_semantic_atom_linkage": False,
            "concept_grounding_allowed": False,
        },
        "assurances": {
            "gate_treatment_parameter_exists": False,
            "gate_outcome_parameter_exists": False,
            "gate_labels_used_for_fit_or_cache": False,
            "initial_spent_labels_used_for_context_fit_only": True,
            "provider_bound_exactly_once": True,
            "later_gate_views_reuse_returned_bound_provider": True,
            "raw_matrix_values_exposed_to_discovery": False,
            "direct_coordinate_metadata_exposed_to_discovery": False,
        },
    }
    normalized_audit = _closed_clone(audit, label="first-gate preparation audit")
    audit_json = canonical_json(normalized_audit)
    return PreparedFirstUntouchedGateDirectNumerical(
        bound_provider=bound,
        persisted_manifest=persisted,
        audit_sha256=content_sha256(normalized_audit),
        _audit_json=audit_json,
    )


__all__ = [
    "FIRST_UNTOUCHED_GATE_PREPARATION_AUDIT_VERSION",
    "FIRST_UNTOUCHED_GATE_PREPARATION_VERSION",
    "FirstUntouchedGatePreparationBounds",
    "PreparedFirstUntouchedGateDirectNumerical",
    "prepare_first_untouched_gate_direct_numerical",
]
