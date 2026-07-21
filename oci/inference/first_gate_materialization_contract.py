"""Pre-fit contract for deferred first-gate numerical materialization.

The hierarchical discovery approval boundary does not need a realized numerical
bank.  It does, however, need to bind everything that is allowed to determine
that bank.  This module creates that structural intent without looking in the
provider cache, fitting a model, decoding a matrix, or accepting gate labels.

After approval and proposal freeze, ``verify_realization`` authenticates a
realized :class:`DirectUpstreamNumericalManifest` against the intent.  Matrix
and per-column value hashes are necessarily realization-time facts; they are
attested then rather than represented by pre-fit placeholder hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import all_evidence_post_extraction_review as _review_module
from . import context_fit_upstream_cache_overlay as _overlay_module
from . import context_fit_upstream_gate_provider as _gate_provider_module
from . import coordinate_preserving_context_fit_upstream_backend as _coordinate_module
from . import direct_upstream_numerical_manifest as _direct_module
from . import lossless_stage1_evidence_catalog as _catalog_module
from .all_evidence_discovery_interfaces import (
    ACTIVE_STAGE1_CONCEPT_FAMILIES,
    ACTIVE_STAGE1_CONCEPT_FAMILY_SET,
    HETEROGENEITY_AXIS,
    TFIDF_SEMANTIC_RETRIEVAL,
    canonical_json,
    content_sha256,
)
from .all_evidence_post_extraction_review import ObservableCausalRows
from .context_fit_upstream_cache_overlay import AuthenticatedContextFitGateCacheOverlay
from .context_fit_upstream_gate_provider import (
    CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION,
    BoundContextFitUpstreamGateProvider,
    ContextFitUpstreamGateProvider,
)
from .coordinate_preserving_context_fit_upstream_backend import (
    COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID,
    CoordinatePreservingContextFitUpstreamBackend,
    CoordinatePreservingUpstreamSchemaConfig,
)
from .direct_upstream_numerical_manifest import (
    CALIBRATED_SOURCES_BLOCK,
    CONTEXT_OOF_SCOPE,
    DirectUpstreamNumericalManifest,
    EFFECT_REGRESSION_COVARIATE_ROLE,
    EXACT_PRECOMMITTED_ALIGNMENT,
    NESTED_CALIBRATED_STATUS,
    PREDICTION_SCOPE,
    RAW_FEATURES_BLOCK,
    SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON,
    UNCALIBRATED_BASIS_STATUS,
)
from .lossless_stage1_evidence_catalog import (
    RoleNeutralEvidenceCatalog,
    validate_role_neutral_catalog,
)

FIRST_GATE_MATERIALIZATION_INTENT_SCHEMA_VERSION = "first_gate_materialization_intent_v1"
FIRST_GATE_MATERIALIZATION_REALIZATION_ATTESTATION_SCHEMA_VERSION = (
    "first_gate_materialization_realization_attestation_v1"
)
FIRST_GATE_LINEAGE_SCOPE = (
    "exact_inner_fold_oof_context_rows_and_complete_spent_context_fit_" "for_label_free_gate_rows"
)
FIRST_GATE_DEFERRED_MATERIALIZATION_BOUNDARY = (
    "after_exact_approval_and_review_proposal_freeze_before_first_gate_evaluation_v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BODY_KEYS = frozenset(
    {
        "contract_version",
        "implementation_file_sha256",
        "outer_fold",
        "code_identities",
        "bind_provider",
        "raw_delegate_provider",
        "exact_cache_binding",
        "source_cache_key",
        "input_bindings",
        "semantic_catalog",
        "coordinate_schema",
        "materialization_boundary",
        "assurances",
    }
)
_INTENT_KEYS = frozenset({"schema_version", "body", "content_sha256"})
_ATTESTATION_KEYS = frozenset({"schema_version", "body", "content_sha256"})
_ATTESTATION_BODY_KEYS = frozenset(
    {
        "intent_content_sha256",
        "direct_manifest_content_sha256",
        "source_manifest_sha256",
        "source_cache_key",
        "stable_output_schema_sha256",
        "shared_lineage_sha256",
        "coordinate_identity_sequence_sha256",
        "matrix_bindings_sha256",
        "preparation_audit_sha256",
        "bound_provider_identity_sha256",
        "preparation_audit_verified",
        "bound_source_manifest_bytes_verified",
        "source_matrix_and_column_values_reauthenticated",
        "unknown_pre_fit_matrix_and_value_hashes_now_authenticated",
        "exact_intent_match",
    }
)


def _module_sha256(module: ModuleType) -> str:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str):
        raise RuntimeError(f"module {module.__name__!r} has no source file")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def _this_module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closed_clone(value: Any, *, label: str) -> Any:
    try:
        encoded = canonical_json(value)
        result = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain finite closed JSON") from exc
    return result


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


def _code_identities(*, overlay_used: bool) -> list[dict[str, str]]:
    rows: list[tuple[str, ModuleType | None]] = [
        ("first_gate_materialization_contract", None),
        ("context_fit_upstream_gate_provider", _gate_provider_module),
        ("coordinate_preserving_context_fit_upstream_backend", _coordinate_module),
        ("direct_upstream_numerical_manifest", _direct_module),
        ("lossless_stage1_evidence_catalog", _catalog_module),
        ("observable_causal_rows", _review_module),
    ]
    if overlay_used:
        rows.append(("authenticated_context_fit_gate_cache_overlay", _overlay_module))
    return [
        {
            "component": component,
            "implementation_file_sha256": (
                _this_module_sha256() if module is None else _module_sha256(module)
            ),
        }
        for component, module in rows
    ]


def _provider_contract(
    provider: ContextFitUpstreamGateProvider | AuthenticatedContextFitGateCacheOverlay,
) -> tuple[
    ContextFitUpstreamGateProvider,
    str,
    dict[str, Any],
    dict[str, Any],
]:
    if type(provider) is ContextFitUpstreamGateProvider:
        raw = provider
        kind = "raw_context_fit_upstream_gate_provider"
        bind_identity = _closed_clone(provider.identity(), label="provider identity")
    elif type(provider) is AuthenticatedContextFitGateCacheOverlay:
        if provider.hierarchical_first_gate_preparation is not True:
            raise ValueError(
                "authenticated cache overlay must explicitly allow hierarchical "
                "first-gate materialization"
            )
        if type(provider.provider) is not ContextFitUpstreamGateProvider:
            raise TypeError("authenticated cache overlay has the wrong raw delegate")
        raw = provider.provider
        kind = "authenticated_context_fit_gate_cache_overlay"
        bind_identity = _closed_clone(provider.identity(), label="overlay identity")
    else:
        raise TypeError(
            "provider must be the exact ContextFitUpstreamGateProvider or exact "
            "AuthenticatedContextFitGateCacheOverlay"
        )
    if type(raw.backend) is not CoordinatePreservingContextFitUpstreamBackend:
        raise TypeError("raw provider must use the exact coordinate-preserving backend")
    if type(raw.backend.config) is not CoordinatePreservingUpstreamSchemaConfig:
        raise TypeError("coordinate-preserving backend has the wrong config type")
    raw_identity = _closed_clone(raw.identity(), label="raw provider identity")
    if not isinstance(raw_identity, dict) or not raw_identity:
        raise ValueError("raw provider identity must be one non-empty object")
    backend_identity = raw_identity.get("backend")
    if not isinstance(backend_identity, Mapping) or backend_identity.get("backend") != (
        COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID
    ):
        raise ValueError("raw provider identity does not bind the v3 coordinate backend")
    if raw_identity.get("provider_code_sha256") != _module_sha256(_gate_provider_module):
        raise ValueError("raw provider code identity differs from current implementation")
    config_identity = _closed_clone(
        raw.backend.config.identity(), label="coordinate config identity"
    )
    if backend_identity.get("config") != config_identity:
        raise ValueError("provider backend identity differs from its exact coordinate config")
    return raw, kind, bind_identity, raw_identity


def _semantic_bindings(catalog: RoleNeutralEvidenceCatalog) -> dict[str, tuple[str, ...]]:
    bindings = {
        family: tuple(atom.evidence_id for atom in catalog.family_atoms(family))
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    }
    if set(bindings) != ACTIVE_STAGE1_CONCEPT_FAMILY_SET or any(
        not values for values in bindings.values()
    ):
        raise ValueError("semantic catalog must contain every active Stage-1 architecture")
    flattened = [item for values in bindings.values() for item in values]
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


def _coordinate_identity(
    *,
    matrix_block: str,
    column_index: int,
    coordinate_name: str,
    source_family: str,
    source_kind: str,
    producer_subarchitecture: str,
    consumer_role: str,
    observable_axis: str,
    calibration_status: str,
    statistic_kind: str,
    statistic_rank: int | None,
    statistic_width: int,
    alignment_mode: str,
    source_coordinate_identity_preserved: bool,
) -> dict[str, Any]:
    fields = {
        "matrix_block": matrix_block,
        "column_index": column_index,
        "coordinate_name": coordinate_name,
        "source_family": source_family,
        "source_kind": source_kind,
        "producer_subarchitecture": producer_subarchitecture,
        "consumer_role": consumer_role,
        "observable_axes": [observable_axis],
        "calibration_status": calibration_status,
        "statistic_kind": statistic_kind,
        "statistic_rank": statistic_rank,
        "statistic_width": statistic_width,
        "alignment_mode": alignment_mode,
        "output_coordinate_identity_stable": True,
        "source_coordinate_identity_preserved": source_coordinate_identity_preserved,
        "concept_grounding_allowed": False,
    }
    identity_sha256 = content_sha256(fields)
    return {
        "coordinate_id": (f"num.{matrix_block}.{column_index:04d}.{identity_sha256[:12]}"),
        "coordinate_identity_sha256": identity_sha256,
        "identity_fields": fields,
    }


def _coordinate_rows(
    *,
    provider_identity: Mapping[str, Any],
    calibrated_sources: Sequence[Mapping[str, Any]],
    raw_features: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    source_names = tuple(str(row["name"]) for row in calibrated_sources)
    source_kinds = tuple(str(row["kind"]) for row in calibrated_sources)
    feature_names = tuple(str(row["name"]) for row in raw_features)
    feature_kinds = tuple(str(row["kind"]) for row in raw_features)
    feature_roles = tuple(str(row["consumer_role"]) for row in raw_features)
    raw_semantics = _direct_module._validate_gate_stable_schema_config(
        provider_identity,
        source_names=source_names,
        source_kinds=source_kinds,
        feature_names=feature_names,
        feature_kinds=feature_kinds,
        feature_roles=feature_roles,
    )
    rows: list[dict[str, Any]] = []
    for index, (name, kind) in enumerate(zip(source_names, source_kinds)):
        family = _direct_module._CALIBRATED_KIND_FAMILY.get(kind)
        if family is None:
            raise ValueError(f"unsupported calibrated source kind: {kind!r}")
        rows.append(
            _coordinate_identity(
                matrix_block=CALIBRATED_SOURCES_BLOCK,
                column_index=index,
                coordinate_name=name,
                source_family=family,
                source_kind=kind,
                producer_subarchitecture=_direct_module._calibrated_producer(name, kind),
                consumer_role=EFFECT_REGRESSION_COVARIATE_ROLE,
                observable_axis=HETEROGENEITY_AXIS,
                calibration_status=NESTED_CALIBRATED_STATUS,
                statistic_kind="direct_prediction",
                statistic_rank=None,
                statistic_width=1,
                alignment_mode=EXACT_PRECOMMITTED_ALIGNMENT,
                source_coordinate_identity_preserved=True,
            )
        )
    for index, semantic in enumerate(raw_semantics):
        family = _direct_module._RAW_KIND_FAMILY.get(semantic.source_kind)
        axis = _direct_module._ROLE_AXIS.get(semantic.consumer_role)
        if family is None or axis is None:
            raise ValueError("raw coordinate has an unsupported family or consumer role")
        rows.append(
            _coordinate_identity(
                matrix_block=RAW_FEATURES_BLOCK,
                column_index=index,
                coordinate_name=semantic.coordinate_name,
                source_family=family,
                source_kind=semantic.source_kind,
                producer_subarchitecture=semantic.producer_subarchitecture,
                consumer_role=semantic.consumer_role,
                observable_axis=axis,
                calibration_status=UNCALIBRATED_BASIS_STATUS,
                statistic_kind=semantic.statistic_kind,
                statistic_rank=semantic.statistic_rank,
                statistic_width=semantic.statistic_width,
                alignment_mode=semantic.alignment_mode,
                source_coordinate_identity_preserved=(
                    semantic.source_coordinate_identity_preserved
                ),
            )
        )
    stable_output_sha256 = _direct_module._stable_output_schema_sha256(
        source_names=source_names,
        source_kinds=source_kinds,
        feature_names=feature_names,
        feature_kinds=feature_kinds,
        feature_roles=feature_roles,
    )
    return rows, stable_output_sha256


def _family_rows(
    *,
    coordinates: Sequence[Mapping[str, Any]],
    semantic_bindings: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in ACTIVE_STAGE1_CONCEPT_FAMILIES:
        members = [row for row in coordinates if row["identity_fields"]["source_family"] == family]
        coordinate_ids = [str(row["coordinate_id"]) for row in members]
        kinds = list(dict.fromkeys(str(row["identity_fields"]["source_kind"]) for row in members))
        zero_reason = (
            SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON if family == TFIDF_SEMANTIC_RETRIEVAL else ""
        )
        if family == TFIDF_SEMANTIC_RETRIEVAL:
            if coordinate_ids:
                raise ValueError("semantic retrieval unexpectedly has an independent coordinate")
        elif not coordinate_ids:
            raise ValueError(f"active numerical architecture has no coordinates: {family}")
        atom_ids = tuple(str(value) for value in semantic_bindings[family])
        rows.append(
            {
                "source_family": family,
                "coordinate_ids": coordinate_ids,
                "source_kinds": kinds,
                "semantic_atom_ids": list(atom_ids),
                "semantic_atom_ids_sha256": content_sha256(list(atom_ids)),
                "numerical_zero_reason": zero_reason,
                "coordinate_to_semantic_atom_linkage": False,
            }
        )
    return rows


def _expected_lineage(cache_binding: Mapping[str, Any], cache_key: str) -> str:
    return content_sha256(
        {
            "source_cache_key": cache_key,
            "context_row_ids_sha256": cache_binding["context_row_ids_sha256"],
            "context_inner_fold_assignment_sha256": cache_binding[
                "context_inner_fold_assignment_sha256"
            ],
            "gate_row_ids_sha256": cache_binding["gate_row_ids_sha256"],
            "context_values_cross_fitted_by_exact_inner_fold": cache_binding[
                "context_values_cross_fitted_by_exact_inner_fold"
            ],
            "gate_labels_exposed_to_backend": cache_binding["gate_labels_exposed_to_backend"],
        }
    )


def _verify_body(body: Mapping[str, Any]) -> None:
    if set(body) != _BODY_KEYS:
        raise ValueError("materialization intent body has an unexpected closed schema")
    if body.get("contract_version") != FIRST_GATE_MATERIALIZATION_INTENT_SCHEMA_VERSION:
        raise ValueError("materialization intent contract version changed")
    if body.get("implementation_file_sha256") != _this_module_sha256():
        raise ValueError("materialization intent implementation changed")
    _positive_integer(body.get("outer_fold"), label="outer_fold")
    bind_provider = body.get("bind_provider")
    raw_provider = body.get("raw_delegate_provider")
    if not isinstance(bind_provider, Mapping) or not isinstance(raw_provider, Mapping):
        raise ValueError("materialization intent provider bindings are malformed")
    overlay_used = bind_provider.get("kind") == "authenticated_context_fit_gate_cache_overlay"
    if bind_provider.get("kind") not in {
        "raw_context_fit_upstream_gate_provider",
        "authenticated_context_fit_gate_cache_overlay",
    }:
        raise ValueError("materialization intent bind-provider kind is unsupported")
    if bind_provider.get("authenticated_overlay_used") is not overlay_used:
        raise ValueError("materialization intent overlay flag is inconsistent")
    for record, label in ((bind_provider, "bind provider"), (raw_provider, "raw provider")):
        identity = record.get("identity")
        if not isinstance(identity, Mapping) or record.get("identity_sha256") != content_sha256(
            identity
        ):
            raise ValueError(f"{label} identity binding is inconsistent")
    code_rows = body.get("code_identities")
    if code_rows != _code_identities(overlay_used=overlay_used):
        raise ValueError("materialization intent code identities changed")
    cache_binding = body.get("exact_cache_binding")
    if not isinstance(cache_binding, Mapping):
        raise ValueError("materialization intent exact cache binding is malformed")
    if cache_binding.get("provider_identity") != raw_provider.get("identity"):
        raise ValueError("exact cache binding changed raw provider identity")
    if cache_binding.get("outer_fold") != body.get("outer_fold"):
        raise ValueError("exact cache binding changed outer fold")
    if (
        cache_binding.get("gate_labels_in_binding") is not False
        or cache_binding.get("gate_labels_exposed_to_backend") is not False
    ):
        raise ValueError("materialization intent permits gate-label access")
    cache_key = body.get("source_cache_key")
    if cache_key != content_sha256(cache_binding):
        raise ValueError("materialization intent cache key differs from exact binding")
    _require_sha256(cache_key, label="source_cache_key")
    inputs = body.get("input_bindings")
    if not isinstance(inputs, Mapping):
        raise ValueError("materialization intent input bindings are malformed")
    spent = inputs.get("initial_spent")
    gate = inputs.get("first_untouched_gate")
    if not isinstance(spent, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("materialization intent spent/gate bindings are malformed")
    expected_input_pairs = (
        (spent, "row_ids_sha256", "context_row_ids_sha256"),
        (spent, "text_sha256", "context_text_sha256"),
        (spent, "treatment_sha256", "context_treatment_sha256"),
        (spent, "outcome_sha256", "context_outcome_sha256"),
        (spent, "inner_fold_assignment_sha256", "context_inner_fold_assignment_sha256"),
        (gate, "row_ids_sha256", "gate_row_ids_sha256"),
        (gate, "text_sha256", "gate_text_sha256"),
    )
    for record, record_key, cache_key_name in expected_input_pairs:
        if record.get(record_key) != cache_binding.get(cache_key_name):
            raise ValueError("materialization intent input hash differs from cache binding")
    if spent.get("row_count") != cache_binding.get("context_row_count") or gate.get(
        "row_count"
    ) != cache_binding.get("gate_row_count"):
        raise ValueError("materialization intent row counts differ from cache binding")
    if gate.get("treatment_accepted") is not False or gate.get("outcome_accepted") is not False:
        raise ValueError("materialization intent accepts gate labels")

    catalog = body.get("semantic_catalog")
    coordinate_schema = body.get("coordinate_schema")
    if not isinstance(catalog, Mapping) or not isinstance(coordinate_schema, Mapping):
        raise ValueError("materialization intent catalog/coordinate schema is malformed")
    families = catalog.get("family_bindings")
    if not isinstance(families, list) or [row.get("source_family") for row in families] != list(
        ACTIVE_STAGE1_CONCEPT_FAMILIES
    ):
        raise ValueError("materialization intent catalog family order changed")
    semantic_bindings: dict[str, tuple[str, ...]] = {}
    for row in families:
        atom_ids = tuple(row.get("semantic_atom_ids") or ())
        if not atom_ids or row.get("semantic_atom_ids_sha256") != content_sha256(list(atom_ids)):
            raise ValueError("materialization intent semantic family binding is invalid")
        semantic_bindings[str(row["source_family"])] = atom_ids
    flattened = [value for values in semantic_bindings.values() for value in values]
    if len(flattened) != len(set(flattened)) or len(flattened) != catalog.get("atom_count"):
        raise ValueError("materialization intent semantic atom partition is invalid")
    raw_identity = raw_provider["identity"]
    backend_identity = raw_identity.get("backend")
    if not isinstance(backend_identity, Mapping) or backend_identity.get("backend") != (
        COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID
    ):
        raise ValueError("materialization intent does not bind the coordinate backend")
    config_identity = coordinate_schema.get("config_identity")
    if config_identity != backend_identity.get("config") or coordinate_schema.get(
        "config_identity_sha256"
    ) != content_sha256(config_identity):
        raise ValueError("materialization intent coordinate config binding changed")
    calibrated = coordinate_schema.get("calibrated_sources")
    raw = coordinate_schema.get("raw_features")
    if not isinstance(calibrated, list) or not isinstance(raw, list):
        raise ValueError("materialization intent output schema is malformed")
    expected_coordinates, stable_sha = _coordinate_rows(
        provider_identity=raw_identity,
        calibrated_sources=calibrated,
        raw_features=raw,
    )
    if coordinate_schema.get("coordinates") != expected_coordinates:
        raise ValueError("materialization intent structural coordinates changed")
    if coordinate_schema.get("stable_output_schema_sha256") != stable_sha:
        raise ValueError("materialization intent stable output schema changed")
    expected_families = _family_rows(
        coordinates=expected_coordinates, semantic_bindings=semantic_bindings
    )
    if coordinate_schema.get("family_coverage") != expected_families:
        raise ValueError("materialization intent coordinate family coverage changed")
    calibrated_count = sum(
        row["identity_fields"]["matrix_block"] == CALIBRATED_SOURCES_BLOCK
        for row in expected_coordinates
    )
    raw_count = len(expected_coordinates) - calibrated_count
    if coordinate_schema.get("matrix_shape_contract") != {
        CONTEXT_OOF_SCOPE: {
            CALIBRATED_SOURCES_BLOCK: [spent["row_count"], calibrated_count],
            RAW_FEATURES_BLOCK: [spent["row_count"], raw_count],
        },
        PREDICTION_SCOPE: {
            CALIBRATED_SOURCES_BLOCK: [gate["row_count"], calibrated_count],
            RAW_FEATURES_BLOCK: [gate["row_count"], raw_count],
        },
    }:
        raise ValueError("materialization intent matrix shape contract changed")
    if (
        coordinate_schema.get("expected_shared_lineage_sha256")
        != _expected_lineage(cache_binding, cache_key)
        or coordinate_schema.get("lineage_scope") != FIRST_GATE_LINEAGE_SCOPE
    ):
        raise ValueError("materialization intent lineage contract changed")
    if body.get("materialization_boundary") != {
        "boundary": FIRST_GATE_DEFERRED_MATERIALIZATION_BOUNDARY,
        "exact_approval_required_before_materialization": True,
        "review_proposal_frozen_before_materialization": True,
        "realization_verified_before_first_gate_evaluation": True,
    }:
        raise ValueError("materialization intent execution boundary changed")
    if body.get("assurances") != {
        "intent_creation_calls_bind_fold": False,
        "intent_creation_fits_backend": False,
        "intent_creation_looks_up_cache": False,
        "intent_creation_reads_matrix_values": False,
        "gate_treatment_parameter_exists": False,
        "gate_outcome_parameter_exists": False,
        "gate_labels_in_binding": False,
        "gate_labels_exposed_to_backend": False,
        "unknown_matrix_and_value_hashes_deferred_to_realization": True,
        "placeholder_matrix_or_value_hashes_used": False,
        "coordinate_to_semantic_atom_linkage": False,
        "concept_grounding_allowed": False,
    }:
        raise ValueError("materialization intent assurances changed")


@dataclass(frozen=True)
class FirstGateMaterializationRealizationAttestation:
    """Closed attestation produced after a realized manifest passes the intent."""

    _body_json: str = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        body = _closed_clone(json.loads(self._body_json), label="realization attestation body")
        if not isinstance(body, dict):
            raise ValueError("realization attestation body must be one object")
        if content_sha256(body) != _require_sha256(
            self.content_sha256, label="realization attestation content_sha256"
        ):
            raise ValueError("realization attestation content SHA-256 mismatch")
        if set(body) != _ATTESTATION_BODY_KEYS:
            raise ValueError("realization attestation body has an unexpected closed schema")
        for name in (
            "intent_content_sha256",
            "direct_manifest_content_sha256",
            "source_manifest_sha256",
            "source_cache_key",
            "stable_output_schema_sha256",
            "shared_lineage_sha256",
            "coordinate_identity_sequence_sha256",
            "matrix_bindings_sha256",
        ):
            _require_sha256(body.get(name), label=f"realization attestation {name}")
        for name in ("preparation_audit_sha256", "bound_provider_identity_sha256"):
            if body.get(name):
                _require_sha256(body[name], label=f"realization attestation {name}")
        for name in (
            "preparation_audit_verified",
            "bound_source_manifest_bytes_verified",
            "source_matrix_and_column_values_reauthenticated",
            "unknown_pre_fit_matrix_and_value_hashes_now_authenticated",
            "exact_intent_match",
        ):
            if not isinstance(body.get(name), bool):
                raise ValueError(f"realization attestation {name} must be a boolean")
        if body["preparation_audit_verified"] is not bool(body["preparation_audit_sha256"]):
            raise ValueError("realization attestation audit verification flag is inconsistent")
        bound_verified = bool(body["bound_provider_identity_sha256"])
        if body["bound_source_manifest_bytes_verified"] is not bound_verified:
            raise ValueError("realization attestation bound-source flag is inconsistent")
        if body["source_matrix_and_column_values_reauthenticated"] is not bound_verified:
            raise ValueError("realization attestation matrix reauthentication flag is inconsistent")
        if body["unknown_pre_fit_matrix_and_value_hashes_now_authenticated"] is not (
            bound_verified
        ):
            raise ValueError("realization attestation value-authentication flag is inconsistent")
        if body["exact_intent_match"] is not True:
            raise ValueError("realization attestation does not assert an exact intent match")
        object.__setattr__(self, "_body_json", canonical_json(body))

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self._body_json)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (FIRST_GATE_MATERIALIZATION_REALIZATION_ATTESTATION_SCHEMA_VERSION),
            "body": self.body,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "FirstGateMaterializationRealizationAttestation":
        normalized = _closed_clone(payload, label="realization attestation")
        if not isinstance(normalized, dict) or set(normalized) != _ATTESTATION_KEYS:
            raise ValueError("realization attestation has an unexpected closed schema")
        if normalized.get("schema_version") != (
            FIRST_GATE_MATERIALIZATION_REALIZATION_ATTESTATION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported realization attestation schema")
        return cls(
            _body_json=canonical_json(normalized["body"]),
            content_sha256=normalized["content_sha256"],
        )


@dataclass(frozen=True)
class FirstGateMaterializationIntent:
    """Immutable pre-fit intent binding inputs and stable output coordinates."""

    _body_json: str = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        body = _closed_clone(json.loads(self._body_json), label="materialization intent body")
        if not isinstance(body, dict):
            raise ValueError("materialization intent body must be one object")
        if content_sha256(body) != _require_sha256(
            self.content_sha256, label="materialization intent content_sha256"
        ):
            raise ValueError("materialization intent content SHA-256 mismatch")
        _verify_body(body)
        object.__setattr__(self, "_body_json", canonical_json(body))

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self._body_json)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FIRST_GATE_MATERIALIZATION_INTENT_SCHEMA_VERSION,
            "body": self.body,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FirstGateMaterializationIntent":
        normalized = _closed_clone(payload, label="materialization intent")
        if not isinstance(normalized, dict) or set(normalized) != _INTENT_KEYS:
            raise ValueError("materialization intent has an unexpected closed schema")
        if normalized.get("schema_version") != FIRST_GATE_MATERIALIZATION_INTENT_SCHEMA_VERSION:
            raise ValueError("unsupported materialization intent schema")
        return cls(
            _body_json=canonical_json(normalized["body"]),
            content_sha256=normalized["content_sha256"],
        )

    def verify(self) -> None:
        body = self.body
        if content_sha256(body) != self.content_sha256:
            raise ValueError("materialization intent changed after construction")
        _verify_body(body)

    def verify_realization(
        self,
        manifest: DirectUpstreamNumericalManifest,
        *,
        preparation_audit: Mapping[str, Any] | None = None,
        bound_provider: BoundContextFitUpstreamGateProvider | None = None,
    ) -> FirstGateMaterializationRealizationAttestation:
        """Verify realized hashes and coordinates, then return a closed attestation."""

        self.verify()
        if type(manifest) is not DirectUpstreamNumericalManifest:
            raise TypeError("manifest must be the exact DirectUpstreamNumericalManifest type")
        body = self.body
        raw_provider = body["raw_delegate_provider"]
        schema = body["coordinate_schema"]
        expected = {
            "source_cache_schema": CONTEXT_FIT_UPSTREAM_CACHE_SCHEMA_VERSION,
            "source_cache_key": body["source_cache_key"],
            "producer_identity_sha256": raw_provider["identity_sha256"],
            "stable_output_schema_sha256": schema["stable_output_schema_sha256"],
            "semantic_catalog_sha256": body["semantic_catalog"]["catalog_sha256"],
            "shared_lineage_sha256": schema["expected_shared_lineage_sha256"],
            "lineage_scope": schema["lineage_scope"],
        }
        for field_name, expected_value in expected.items():
            if getattr(manifest, field_name) != expected_value:
                raise ValueError(f"realized direct manifest changed {field_name}")
        expected_coordinates = schema["coordinates"]
        actual_coordinates = [
            {
                "coordinate_id": row.coordinate_id,
                "coordinate_identity_sha256": row.coordinate_identity_sha256,
                "identity_fields": row.coordinate_identity_fields(),
            }
            for row in manifest.coordinates
        ]
        if actual_coordinates != expected_coordinates:
            raise ValueError("realized direct manifest changed structural coordinates")
        expected_families = schema["family_coverage"]
        actual_families = [
            {
                "source_family": row.source_family,
                "coordinate_ids": list(row.coordinate_ids),
                "source_kinds": list(row.source_kinds),
                "semantic_atom_ids": list(row.semantic_atom_ids),
                "semantic_atom_ids_sha256": row.semantic_atom_ids_sha256,
                "numerical_zero_reason": row.numerical_zero_reason,
                "coordinate_to_semantic_atom_linkage": False,
            }
            for row in manifest.family_coverage
        ]
        if actual_families != expected_families:
            raise ValueError("realized direct manifest changed architecture coverage")
        matrix_shapes = schema["matrix_shape_contract"]
        for matrix in manifest.matrices:
            expected_shape = matrix_shapes[matrix.row_scope][matrix.matrix_block]
            if list(matrix.shape) != expected_shape:
                raise ValueError("realized direct manifest changed an expected matrix shape")
        for coordinate in manifest.coordinates:
            if (
                coordinate.source_cache_key != manifest.source_cache_key
                or coordinate.shared_lineage_sha256 != manifest.shared_lineage_sha256
                or coordinate.lineage_scope != manifest.lineage_scope
            ):
                raise ValueError("realized coordinate changed its cache/lineage binding")

        audit_sha256 = ""
        if preparation_audit is not None:
            audit = _closed_clone(preparation_audit, label="preparation audit")
            if not isinstance(audit, dict):
                raise ValueError("preparation audit must be one object")
            _verify_preparation_audit(intent=body, manifest=manifest, audit=audit)
            audit_sha256 = content_sha256(audit)

        bound_identity_sha256 = ""
        realized_values_reauthenticated = False
        if bound_provider is not None:
            if type(bound_provider) is not BoundContextFitUpstreamGateProvider:
                raise TypeError(
                    "bound_provider must be the exact BoundContextFitUpstreamGateProvider type"
                )
            bound_identity = _closed_clone(
                bound_provider.identity(), label="bound provider identity"
            )
            if bound_provider.outer_fold != body["outer_fold"]:
                raise ValueError("bound provider changed outer fold")
            if (
                bound_identity.get("gate_row_ids_sha256")
                != body["input_bindings"]["first_untouched_gate"]["row_ids_sha256"]
            ):
                raise ValueError("bound provider changed first-gate row identity")
            if bound_identity.get("parent_identity_sha256") != raw_provider["identity_sha256"]:
                raise ValueError("bound provider changed raw delegate identity")
            if bound_identity.get("cache_manifest_sha256") != manifest.source_manifest_sha256:
                raise ValueError("bound provider and direct manifest source bytes differ")
            source_path = bound_provider.authenticated_cache_manifest_path
            if _file_sha256(source_path) != manifest.source_manifest_sha256:
                raise ValueError("bound provider source manifest changed")
            snapshot = _direct_module.load_authenticated_numerical_bank_snapshot(source_path)
            semantic_bindings = {
                row["source_family"]: tuple(row["semantic_atom_ids"])
                for row in body["semantic_catalog"]["family_bindings"]
            }
            rebuilt = _direct_module.build_direct_upstream_numerical_manifest(
                snapshot,
                semantic_catalog_sha256=body["semantic_catalog"]["catalog_sha256"],
                semantic_atom_ids_by_family=semantic_bindings,
                numerical_zero_reasons={
                    TFIDF_SEMANTIC_RETRIEVAL: SEMANTIC_RETRIEVAL_NUMERICAL_ZERO_REASON
                },
            )
            if rebuilt.as_dict() != manifest.as_dict():
                raise ValueError(
                    "realized direct manifest differs from reauthenticated source matrices"
                )
            bound_identity_sha256 = content_sha256(bound_identity)
            realized_values_reauthenticated = True

        attestation_body = {
            "intent_content_sha256": self.content_sha256,
            "direct_manifest_content_sha256": manifest.content_sha256,
            "source_manifest_sha256": manifest.source_manifest_sha256,
            "source_cache_key": manifest.source_cache_key,
            "stable_output_schema_sha256": manifest.stable_output_schema_sha256,
            "shared_lineage_sha256": manifest.shared_lineage_sha256,
            "coordinate_identity_sequence_sha256": content_sha256(
                [row["coordinate_identity_sha256"] for row in actual_coordinates]
            ),
            "matrix_bindings_sha256": content_sha256([row.as_dict() for row in manifest.matrices]),
            "preparation_audit_sha256": audit_sha256,
            "bound_provider_identity_sha256": bound_identity_sha256,
            "preparation_audit_verified": preparation_audit is not None,
            "bound_source_manifest_bytes_verified": bound_provider is not None,
            "source_matrix_and_column_values_reauthenticated": realized_values_reauthenticated,
            "unknown_pre_fit_matrix_and_value_hashes_now_authenticated": (
                realized_values_reauthenticated
            ),
            "exact_intent_match": True,
        }
        return FirstGateMaterializationRealizationAttestation(
            _body_json=canonical_json(attestation_body),
            content_sha256=content_sha256(attestation_body),
        )


def _verify_preparation_audit(
    *,
    intent: Mapping[str, Any],
    manifest: DirectUpstreamNumericalManifest,
    audit: Mapping[str, Any],
) -> None:
    if audit.get("outer_fold") != intent["outer_fold"]:
        raise ValueError("preparation audit changed outer fold")
    spent = audit.get("initial_spent_binding")
    gate = audit.get("first_untouched_gate_binding")
    catalog = audit.get("semantic_catalog_binding")
    upstream = audit.get("upstream_cache_binding")
    direct = audit.get("direct_numerical_manifest_binding")
    assurances = audit.get("assurances")
    if not all(
        isinstance(row, Mapping) for row in (spent, gate, catalog, upstream, direct, assurances)
    ):
        raise ValueError("preparation audit is missing a required binding")
    expected_spent = intent["input_bindings"]["initial_spent"]
    for name in (
        "row_count",
        "row_ids_sha256",
        "text_sha256",
        "treatment_sha256",
        "outcome_sha256",
        "inner_fold_assignment_sha256",
    ):
        if spent.get(name) != expected_spent[name]:
            raise ValueError(f"preparation audit changed initial-spent {name}")
    expected_gate = intent["input_bindings"]["first_untouched_gate"]
    for name in ("row_count", "row_ids_sha256", "text_sha256"):
        if gate.get(name) != expected_gate[name]:
            raise ValueError(f"preparation audit changed first-gate {name}")
    if any(
        gate.get(name) is not False
        for name in (
            "treatment_accepted",
            "outcome_accepted",
            "labels_in_cache_binding",
            "labels_exposed_to_backend",
        )
    ):
        raise ValueError("preparation audit permits first-gate label access")
    expected_catalog = intent["semantic_catalog"]
    for name in ("catalog_sha256", "scope", "inner_fold", "split_fingerprint", "atom_count"):
        if catalog.get(name) != expected_catalog[name]:
            raise ValueError(f"preparation audit changed semantic catalog {name}")
    expected_family_audit = [
        {
            "source_family": row["source_family"],
            "semantic_atom_count": len(row["semantic_atom_ids"]),
            "semantic_atom_ids_sha256": row["semantic_atom_ids_sha256"],
        }
        for row in expected_catalog["family_bindings"]
    ]
    if catalog.get("family_bindings") != expected_family_audit:
        raise ValueError("preparation audit changed semantic family bindings")
    expected_upstream = {
        "provider_identity_sha256": intent["raw_delegate_provider"]["identity_sha256"],
        "bind_provider_kind": intent["bind_provider"]["kind"],
        "bind_provider_identity_sha256": intent["bind_provider"]["identity_sha256"],
        "raw_delegate_provider_identity_sha256": intent["raw_delegate_provider"]["identity_sha256"],
        "source_cache_schema": manifest.source_cache_schema,
        "source_cache_key": manifest.source_cache_key,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "producer_identity_sha256": manifest.producer_identity_sha256,
        "stable_output_schema_sha256": manifest.stable_output_schema_sha256,
        "shared_lineage_sha256": manifest.shared_lineage_sha256,
        "lineage_scope": manifest.lineage_scope,
    }
    for name, value in expected_upstream.items():
        if upstream.get(name) != value:
            raise ValueError(f"preparation audit changed upstream {name}")
    if direct.get("content_sha256") != manifest.content_sha256:
        raise ValueError("preparation audit changed direct manifest content")
    if direct.get("signal_count") != manifest.signal_count:
        raise ValueError("preparation audit changed direct signal count")
    if assurances.get("gate_labels_used_for_fit_or_cache") is not False:
        raise ValueError("preparation audit permits gate labels")
    if "materialization_intent_sha256" in audit and audit.get(
        "materialization_intent_sha256"
    ) != content_sha256(intent):
        raise ValueError("preparation audit changed materialization intent binding")


def prepare_first_gate_materialization_intent(
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
) -> FirstGateMaterializationIntent:
    """Create an exact first-gate materialization intent without materializing it.

    The absence of gate treatment/outcome parameters is security relevant.  The
    function calls neither ``bind_fold`` nor ``fit_predict`` and does not inspect
    the provider cache directory.
    """

    fold = _positive_integer(outer_fold, label="outer_fold")
    context_ids = _exact_integer_rows(initial_spent_row_ids, label="initial_spent_row_ids")
    gate_ids = _exact_integer_rows(first_gate_row_ids, label="first_gate_row_ids")
    if set(context_ids) & set(gate_ids):
        raise ValueError("initial-spent and first-gate rows must be disjoint")
    context_texts = _exact_texts(
        initial_spent_texts, label="initial_spent_texts", length=len(context_ids)
    )
    gate_texts = _exact_texts(first_gate_texts, label="first_gate_texts", length=len(gate_ids))
    inner_folds = _exact_inner_folds(initial_spent_inner_fold_ids, length=len(context_ids))

    if not isinstance(catalog, RoleNeutralEvidenceCatalog):
        raise TypeError("catalog must be RoleNeutralEvidenceCatalog")
    catalog_before = canonical_json(catalog.as_dict())
    validate_role_neutral_catalog(catalog)
    if catalog.outer_fold != fold:
        raise ValueError("semantic catalog outer fold differs from intent outer fold")
    if catalog.scope != "inner_train" or catalog.inner_fold is None:
        raise ValueError("first-gate intent requires an inner_train spent-only catalog")
    semantic_bindings = _semantic_bindings(catalog)

    raw, bind_kind, bind_identity, raw_identity = _provider_contract(provider)
    raw_identity_before = canonical_json(raw_identity)
    bind_identity_before = canonical_json(bind_identity)
    config = raw.backend.config
    context = ObservableCausalRows(
        row_ids=context_ids,
        extracted=pd.DataFrame({"_oci_row_id": context_ids}),
        treatment=np.asarray(initial_spent_treatment, dtype=float),
        outcome=np.asarray(initial_spent_outcome, dtype=float),
        inner_fold_ids=inner_folds,
    )
    expected_binding = _expected_cache_binding(
        provider_identity=raw_identity,
        outer_fold=fold,
        context=context,
        context_texts=context_texts,
        gate_row_ids=gate_ids,
        gate_texts=gate_texts,
        inner_fold_ids=inner_folds,
    )
    provider_binding = raw._binding(
        outer_fold=fold,
        context=context,
        gate_row_ids=gate_ids,
        context_texts=context_texts,
        gate_texts=gate_texts,
        context_inner_fold_ids=inner_folds,
    )
    if _closed_clone(provider_binding, label="provider cache binding") != expected_binding:
        raise ValueError("raw provider pure cache binding differs from exact intent inputs")
    source_cache_key = content_sha256(expected_binding)

    calibrated_sources = [
        {"name": str(row.output_name), "kind": str(row.source_kind)}
        for row in config.calibrated_sources
    ]
    raw_features = [
        {"name": name, "kind": kind, "consumer_role": role}
        for name, kind, role in config.raw_output_schema()
    ]
    coordinates, stable_schema_sha256 = _coordinate_rows(
        provider_identity=raw_identity,
        calibrated_sources=calibrated_sources,
        raw_features=raw_features,
    )
    family_coverage = _family_rows(coordinates=coordinates, semantic_bindings=semantic_bindings)
    calibrated_count = len(calibrated_sources)
    raw_count = len(raw_features)
    catalog_family_rows = [
        {
            "source_family": family,
            "semantic_atom_ids": list(semantic_bindings[family]),
            "semantic_atom_ids_sha256": content_sha256(list(semantic_bindings[family])),
        }
        for family in ACTIVE_STAGE1_CONCEPT_FAMILIES
    ]
    config_identity = _closed_clone(config.identity(), label="coordinate config identity")
    body = {
        "contract_version": FIRST_GATE_MATERIALIZATION_INTENT_SCHEMA_VERSION,
        "implementation_file_sha256": _this_module_sha256(),
        "outer_fold": fold,
        "code_identities": _code_identities(
            overlay_used=bind_kind == "authenticated_context_fit_gate_cache_overlay"
        ),
        "bind_provider": {
            "kind": bind_kind,
            "identity": bind_identity,
            "identity_sha256": content_sha256(bind_identity),
            "authenticated_overlay_used": bind_kind
            == "authenticated_context_fit_gate_cache_overlay",
        },
        "raw_delegate_provider": {
            "identity": raw_identity,
            "identity_sha256": content_sha256(raw_identity),
        },
        "exact_cache_binding": expected_binding,
        "source_cache_key": source_cache_key,
        "input_bindings": {
            "initial_spent": {
                "row_count": len(context_ids),
                "row_ids_sha256": expected_binding["context_row_ids_sha256"],
                "text_sha256": expected_binding["context_text_sha256"],
                "treatment_sha256": expected_binding["context_treatment_sha256"],
                "outcome_sha256": expected_binding["context_outcome_sha256"],
                "inner_fold_assignment_sha256": expected_binding[
                    "context_inner_fold_assignment_sha256"
                ],
                "placeholder_extraction_columns": ["_oci_row_id"],
            },
            "first_untouched_gate": {
                "row_count": len(gate_ids),
                "row_ids_sha256": expected_binding["gate_row_ids_sha256"],
                "text_sha256": expected_binding["gate_text_sha256"],
                "treatment_accepted": False,
                "outcome_accepted": False,
            },
        },
        "semantic_catalog": {
            "catalog_sha256": catalog.catalog_sha256,
            "scope": catalog.scope,
            "inner_fold": catalog.inner_fold,
            "split_fingerprint": catalog.split_fingerprint,
            "atom_count": len(catalog.atoms),
            "family_bindings": catalog_family_rows,
            "all_active_stage1_architectures_bound": True,
        },
        "coordinate_schema": {
            "backend": COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID,
            "config_identity": config_identity,
            "config_identity_sha256": content_sha256(config_identity),
            "calibrated_sources": calibrated_sources,
            "raw_features": raw_features,
            "stable_output_schema_sha256": stable_schema_sha256,
            "coordinates": coordinates,
            "family_coverage": family_coverage,
            "matrix_shape_contract": {
                CONTEXT_OOF_SCOPE: {
                    CALIBRATED_SOURCES_BLOCK: [len(context_ids), calibrated_count],
                    RAW_FEATURES_BLOCK: [len(context_ids), raw_count],
                },
                PREDICTION_SCOPE: {
                    CALIBRATED_SOURCES_BLOCK: [len(gate_ids), calibrated_count],
                    RAW_FEATURES_BLOCK: [len(gate_ids), raw_count],
                },
            },
            "expected_shared_lineage_sha256": _expected_lineage(expected_binding, source_cache_key),
            "lineage_scope": FIRST_GATE_LINEAGE_SCOPE,
        },
        "materialization_boundary": {
            "boundary": FIRST_GATE_DEFERRED_MATERIALIZATION_BOUNDARY,
            "exact_approval_required_before_materialization": True,
            "review_proposal_frozen_before_materialization": True,
            "realization_verified_before_first_gate_evaluation": True,
        },
        "assurances": {
            "intent_creation_calls_bind_fold": False,
            "intent_creation_fits_backend": False,
            "intent_creation_looks_up_cache": False,
            "intent_creation_reads_matrix_values": False,
            "gate_treatment_parameter_exists": False,
            "gate_outcome_parameter_exists": False,
            "gate_labels_in_binding": False,
            "gate_labels_exposed_to_backend": False,
            "unknown_matrix_and_value_hashes_deferred_to_realization": True,
            "placeholder_matrix_or_value_hashes_used": False,
            "coordinate_to_semantic_atom_linkage": False,
            "concept_grounding_allowed": False,
        },
    }
    if canonical_json(raw.identity()) != raw_identity_before:
        raise ValueError("raw provider identity changed while creating materialization intent")
    if canonical_json(provider.identity()) != bind_identity_before:
        raise ValueError("bind provider identity changed while creating materialization intent")
    validate_role_neutral_catalog(catalog)
    if canonical_json(catalog.as_dict()) != catalog_before:
        raise ValueError("semantic catalog changed while creating materialization intent")
    normalized = _closed_clone(body, label="materialization intent body")
    return FirstGateMaterializationIntent(
        _body_json=canonical_json(normalized),
        content_sha256=content_sha256(normalized),
    )


__all__ = [
    "FIRST_GATE_DEFERRED_MATERIALIZATION_BOUNDARY",
    "FIRST_GATE_LINEAGE_SCOPE",
    "FIRST_GATE_MATERIALIZATION_INTENT_SCHEMA_VERSION",
    "FIRST_GATE_MATERIALIZATION_REALIZATION_ATTESTATION_SCHEMA_VERSION",
    "FirstGateMaterializationIntent",
    "FirstGateMaterializationRealizationAttestation",
    "prepare_first_gate_materialization_intent",
]
