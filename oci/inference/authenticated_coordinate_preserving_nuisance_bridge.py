"""Fail-closed nuisance derivation from exact, coordinate-preserved Stage-1 outputs.

The historical stable upstream wrapper reduced each BoW/HTR nuisance family to
row-wise order statistics.  A coordinate-preserving successor can instead
retain the six configured BoW predictions and the singleton HTR prediction for
each nuisance target.  This module is the estimator-facing bridge for that
successor; it does not alter or weaken the historical bridge.

Names are not treated as semantic proof.  A caller must supply two values that
were frozen before package production:

* the exact ``FinalContextFitUpstreamProducer.identity()`` SHA-256; and
* the coordinate contract SHA-256 returned by
  :func:`coordinate_preserving_nuisance_contract_sha256`.

The package, live producer graph, exact coordinate indices/names/kinds/roles,
ordered float64 arithmetic, source lineages, and resulting sealed nuisance
extension are all bound into one derivation digest.  There is no package-only
or name-only entry point.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
)
from .coordinate_preserving_context_fit_upstream_backend import (
    COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID,
    CoordinatePreservingContextFitUpstreamBackend,
    CoordinatePreservingUpstreamSchemaConfig,
    PrecommittedNamedRawCoordinate,
    PrecommittedVolatileRawFeatureFamily,
)
from .final_context_fit_r_stack_adapter import (
    EXACT_OUTCOME_PREDICTION,
    EXACT_PROPENSITY_PREDICTION,
    SealedExactNuisanceBankExtension,
)
from .final_context_fit_upstream_bank import (
    AuthenticatedFinalContextFitUpstreamBank,
    FinalContextFitUpstreamProducer,
)
from .fold_honest_r_stack import FitRowProvenance

AUTHENTICATED_COORDINATE_PRESERVING_NUISANCE_BRIDGE_ID = (
    "authenticated_coordinate_preserving_stage1_nuisance_bridge_v3"
)
AUTHENTICATED_COORDINATE_PRESERVING_NUISANCE_DERIVATION_SCHEMA = (
    "authenticated_coordinate_preserving_stage1_nuisance_derivation_v3"
)
COORDINATE_PRESERVING_NUISANCE_CONTRACT_SCHEMA = "coordinate_preserving_stage1_nuisance_contract_v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_VIEW = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_FORBIDDEN = re.compile(r"(?:oracle|ground_truth|true_ite|true_cate|true_effect)", re.I)
_BOW_KIND = "bow_nuisance"
_HTR_KIND = "htr_nuisance"
_BOW_COUNT = 6

_OUTPUTS = (
    (
        "bow_equal_mean_propensity_prediction",
        "bow_nuisance_equal_mean_from_six_exact_coordinates",
        EXACT_PROPENSITY_PREDICTION,
        _BOW_KIND,
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        "equal_mean_six",
    ),
    (
        "htr_propensity_prediction",
        "htr_nuisance_singleton_exact_coordinate",
        EXACT_PROPENSITY_PREDICTION,
        _HTR_KIND,
        PROPENSITY_NUISANCE_FEATURE_ROLE,
        "singleton",
    ),
    (
        "bow_equal_mean_outcome_prediction",
        "bow_nuisance_equal_mean_from_six_exact_coordinates",
        EXACT_OUTCOME_PREDICTION,
        _BOW_KIND,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        "equal_mean_six",
    ),
    (
        "htr_outcome_prediction",
        "htr_nuisance_singleton_exact_coordinate",
        EXACT_OUTCOME_PREDICTION,
        _HTR_KIND,
        OUTCOME_NUISANCE_FEATURE_ROLE,
        "singleton",
    ),
)

# Detect runtime monkeypatches at the two authentication boundaries used here.
_AUTHENTICATED_PRODUCER_IDENTITY = FinalContextFitUpstreamProducer.identity
_AUTHENTICATED_PACKAGE_VERIFY = (
    AuthenticatedFinalContextFitUpstreamBank.verify_authenticated_content
)
_AUTHENTICATED_COORDINATE_BACKEND_IDENTITY = CoordinatePreservingContextFitUpstreamBackend.identity
_AUTHENTICATED_COORDINATE_BACKEND_FIT_PREDICT = (
    CoordinatePreservingContextFitUpstreamBackend.fit_predict
)
_AUTHENTICATED_COORDINATE_BACKEND_STABLE_RAW = (
    CoordinatePreservingContextFitUpstreamBackend._stable_raw_features
)


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


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_thaw(item) for item in value]
    return value


def _valid_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return value


def _view_names(values: Sequence[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("bow_view_names must be a sequence")
    result = tuple(str(value).strip() for value in values)
    if len(result) != _BOW_COUNT:
        raise ValueError("coordinate-preserving nuisance contract requires exactly six BoW views")
    if len(result) != len(set(result)):
        raise ValueError("bow_view_names must be unique")
    if any(
        not value or _SAFE_VIEW.fullmatch(value) is None or _FORBIDDEN.search(value)
        for value in result
    ):
        raise ValueError("bow_view_names contains an invalid or forbidden view name")
    return result


def _coordinate_name(*, kind: str, role: str, view_name: str | None = None) -> str:
    suffix = "propensity" if role == PROPENSITY_NUISANCE_FEATURE_ROLE else "outcome"
    target = "treatment" if role == PROPENSITY_NUISANCE_FEATURE_ROLE else "outcome"
    if kind == _BOW_KIND:
        assert view_name is not None
        return f"stage1_raw__bow__{view_name}__{target}_pred__as_{suffix}"
    if kind == _HTR_KIND:
        return f"stage1_raw__htr__nuisance__{target}_pred__as_{suffix}"
    raise AssertionError("closed nuisance coordinate kind changed")


def coordinate_preserving_nuisance_schema(
    bow_view_names: Sequence[Any],
) -> tuple[Mapping[str, str], ...]:
    """Return the exact raw coordinates a stable successor must preserve.

    The tuple order is the required within-family order: six configured BoW
    views followed by the singleton HTR coordinate, first for treatment and
    then for outcome.
    """

    views = _view_names(bow_view_names)
    rows: list[Mapping[str, str]] = []
    for role in (PROPENSITY_NUISANCE_FEATURE_ROLE, OUTCOME_NUISANCE_FEATURE_ROLE):
        rows.extend(
            MappingProxyType(
                {
                    "feature_name": _coordinate_name(kind=_BOW_KIND, role=role, view_name=view),
                    "feature_kind": _BOW_KIND,
                    "consumer_role": role,
                }
            )
            for view in views
        )
        rows.append(
            MappingProxyType(
                {
                    "feature_name": _coordinate_name(kind=_HTR_KIND, role=role),
                    "feature_kind": _HTR_KIND,
                    "consumer_role": role,
                }
            )
        )
    return tuple(rows)


def _contract_payload(bow_view_names: tuple[str, ...]) -> Mapping[str, Any]:
    one_sixth = float(np.float64(1.0) / np.float64(_BOW_COUNT))
    return {
        "schema_version": COORDINATE_PRESERVING_NUISANCE_CONTRACT_SCHEMA,
        "bow_view_names": list(bow_view_names),
        "coordinates": [
            dict(item) for item in coordinate_preserving_nuisance_schema(bow_view_names)
        ],
        "outputs": [
            {
                "output_name": name,
                "output_kind": kind,
                "semantic": semantic,
                "source_kind": source_kind,
                "consumer_role": role,
                "arithmetic": (
                    {
                        "operation": "ordered_float64_sum_then_multiply",
                        "source_count": _BOW_COUNT,
                        "weight_float64_hex": one_sixth.hex(),
                    }
                    if operation == "equal_mean_six"
                    else {
                        "operation": "exact_float64_singleton_copy",
                        "source_count": 1,
                        "weight_float64_hex": float(1.0).hex(),
                    }
                ),
            }
            for name, kind, semantic, source_kind, role, operation in _OUTPUTS
        ],
        "coordinate_names_are_semantic_proof": False,
        "precommitted_producer_identity_required": True,
        "exact_coordinate_preserving_backend_required": True,
        "complete_family_membership_required": True,
    }


def coordinate_preserving_nuisance_contract_sha256(
    bow_view_names: Sequence[Any],
) -> str:
    """Hash the exact coordinate membership, order, and arithmetic contract."""

    views = _view_names(bow_view_names)
    return _sha256_json(_contract_payload(views))


def precommit_runtime_producer_identity_sha256(
    runtime_producer: FinalContextFitUpstreamProducer,
) -> str:
    """Hash a reviewed live producer graph before it produces a package."""

    if type(runtime_producer) is not FinalContextFitUpstreamProducer:
        raise TypeError("runtime_producer must be the exact FinalContextFitUpstreamProducer")
    if inspect.getattr_static(FinalContextFitUpstreamProducer, "identity") is not (
        _AUTHENTICATED_PRODUCER_IDENTITY
    ):
        raise TypeError("FinalContextFitUpstreamProducer.identity runtime changed")
    return _sha256_json(runtime_producer.identity())


def _coordinate_runtime_identity_sha256(
    runtime_producer: FinalContextFitUpstreamProducer,
    *,
    bow_view_names: tuple[str, ...],
) -> tuple[str, str]:
    backend = runtime_producer.backend
    if type(backend) is not CoordinatePreservingContextFitUpstreamBackend:
        raise TypeError("runtime producer must use the exact coordinate-preserving v3 backend")
    for name, expected in (
        ("identity", _AUTHENTICATED_COORDINATE_BACKEND_IDENTITY),
        ("fit_predict", _AUTHENTICATED_COORDINATE_BACKEND_FIT_PREDICT),
        ("_stable_raw_features", _AUTHENTICATED_COORDINATE_BACKEND_STABLE_RAW),
    ):
        if inspect.getattr_static(CoordinatePreservingContextFitUpstreamBackend, name) is not (
            expected
        ):
            raise TypeError(f"coordinate-preserving backend runtime changed: {name}")
    config = backend.config
    if type(config) is not CoordinatePreservingUpstreamSchemaConfig:
        raise TypeError("coordinate-preserving backend config has the wrong closed type")
    if not all(
        type(item) is PrecommittedNamedRawCoordinate for item in config.named_raw_coordinates
    ) or not all(
        type(item) is PrecommittedVolatileRawFeatureFamily for item in config.volatile_raw_families
    ):
        raise TypeError("coordinate-preserving config contains an open raw schema type")
    expected = tuple(
        (
            item["feature_name"],
            item["feature_kind"],
            item["consumer_role"],
            item["feature_name"],
            True,
        )
        for item in coordinate_preserving_nuisance_schema(bow_view_names)
    )
    target_keys = {
        (_BOW_KIND, PROPENSITY_NUISANCE_FEATURE_ROLE),
        (_HTR_KIND, PROPENSITY_NUISANCE_FEATURE_ROLE),
        (_BOW_KIND, OUTCOME_NUISANCE_FEATURE_ROLE),
        (_HTR_KIND, OUTCOME_NUISANCE_FEATURE_ROLE),
    }
    actual = tuple(
        (
            item.child_name,
            item.source_kind,
            item.consumer_role,
            str(item.output_name),
            item.required,
        )
        for item in config.named_raw_coordinates
        if (item.source_kind, item.consumer_role) in target_keys
    )
    if actual != expected:
        raise ValueError(
            "coordinate-preserving backend did not precommit the exact fourteen "
            "BoW/HTR nuisance coordinates"
        )
    if any(item.key in target_keys for item in config.volatile_raw_families):
        raise ValueError(
            "BoW/HTR nuisance coordinates cannot also enter a volatile family reduction"
        )
    identity = backend.identity()
    if identity.get("backend") != COORDINATE_PRESERVING_CONTEXT_FIT_UPSTREAM_BACKEND_ID:
        raise ValueError("coordinate-preserving backend has the wrong implementation ID")
    return _sha256_json(identity), _sha256_json(config.identity())


def _lineage_payload(lineage: FitRowProvenance, *, active: set[int] | None = None) -> Any:
    if not isinstance(lineage, FitRowProvenance):
        raise TypeError("source lineage entries must be FitRowProvenance")
    stack = set() if active is None else active
    identity = id(lineage)
    if identity in stack:
        raise ValueError("source lineage contains a cycle")
    stack.add(identity)
    try:
        rows = sorted(int(value) for value in lineage.fit_row_ids)
        if any(value < 0 for value in rows):
            raise ValueError("source lineage contains a negative row ID")
        return {
            "fit_row_ids": rows,
            "upstream": [_lineage_payload(item, active=stack) for item in lineage.upstream],
        }
    finally:
        stack.remove(identity)


def _selected_lineage_sha256(
    values: tuple[tuple[FitRowProvenance, ...], ...], indices: tuple[int, ...]
) -> str:
    return _sha256_json([[_lineage_payload(row[index]) for index in indices] for row in values])


def _combined_lineage(
    values: tuple[tuple[FitRowProvenance, ...], ...], indices: tuple[int, ...]
) -> tuple[FitRowProvenance, ...]:
    if len(indices) == 1:
        return tuple(row[indices[0]] for row in values)
    return tuple(
        FitRowProvenance(
            fit_row_ids=frozenset(),
            upstream=tuple(row[index] for index in indices),
        )
        for row in values
    )


def _ordered_equal_mean(matrix: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    if len(indices) != _BOW_COUNT:
        raise AssertionError("BoW equal mean lost its six-coordinate contract")
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    for index in indices:
        result = np.add(result, np.asarray(matrix[:, index], dtype=np.float64))
    return np.multiply(result, np.float64(1.0) / np.float64(_BOW_COUNT))


def _assert_coordinate_probabilities(
    matrix: np.ndarray,
    *,
    indices: tuple[int, ...],
    semantic: str,
    scope: str,
) -> None:
    values = np.asarray(matrix[:, indices], dtype=float)
    if semantic == EXACT_PROPENSITY_PREDICTION:
        if np.any(values <= 0.0) or np.any(values >= 1.0):
            raise ValueError(
                f"{scope} source propensity coordinates must each be strictly inside (0, 1)"
            )
    elif semantic == EXACT_OUTCOME_PREDICTION:
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(f"{scope} source outcome coordinates must each be inside [0, 1]")
    else:
        raise AssertionError("closed nuisance semantic changed")


def _runtime_identity_sha256(
    package: AuthenticatedFinalContextFitUpstreamBank,
    runtime_producer: FinalContextFitUpstreamProducer,
    *,
    precommitted_sha256: str,
    bow_view_names: tuple[str, ...],
) -> tuple[str, str, str]:
    if type(package) is not AuthenticatedFinalContextFitUpstreamBank:
        raise TypeError("package must be the exact authenticated final upstream type")
    if type(runtime_producer) is not FinalContextFitUpstreamProducer:
        raise TypeError("runtime_producer must be the exact FinalContextFitUpstreamProducer")
    if (
        inspect.getattr_static(
            AuthenticatedFinalContextFitUpstreamBank, "verify_authenticated_content"
        )
        is not _AUTHENTICATED_PACKAGE_VERIFY
    ):
        raise TypeError("authenticated package verification runtime changed")
    expected = _valid_sha256(precommitted_sha256, name="precommitted_producer_identity_sha256")
    package.verify_authenticated_content()
    actual = precommit_runtime_producer_identity_sha256(runtime_producer)
    if actual != expected or package.producer_identity_sha256 != expected:
        raise ValueError(
            "live producer, package producer, and precommitted producer identities must match"
        )
    backend_sha, schema_sha = _coordinate_runtime_identity_sha256(
        runtime_producer,
        bow_view_names=bow_view_names,
    )
    return actual, backend_sha, schema_sha


def _derive_material(
    package: AuthenticatedFinalContextFitUpstreamBank,
    *,
    bow_view_names: tuple[str, ...],
) -> tuple[tuple[Mapping[str, Any], ...], SealedExactNuisanceBankExtension]:
    raw = package.raw_features
    names = raw.feature_names
    kinds = raw.feature_kinds
    roles = raw.consumer_roles
    required_schema = coordinate_preserving_nuisance_schema(bow_view_names)

    required_by_group: dict[tuple[str, str], tuple[str, ...]] = {}
    for source_kind, role in (
        (_BOW_KIND, PROPENSITY_NUISANCE_FEATURE_ROLE),
        (_HTR_KIND, PROPENSITY_NUISANCE_FEATURE_ROLE),
        (_BOW_KIND, OUTCOME_NUISANCE_FEATURE_ROLE),
        (_HTR_KIND, OUTCOME_NUISANCE_FEATURE_ROLE),
    ):
        required_by_group[(source_kind, role)] = tuple(
            item["feature_name"]
            for item in required_schema
            if item["feature_kind"] == source_kind and item["consumer_role"] == role
        )

    indices_by_group: dict[tuple[str, str], tuple[int, ...]] = {}
    for key, expected_names in required_by_group.items():
        observed_indices = tuple(
            index for index, candidate in enumerate(zip(kinds, roles)) if candidate == key
        )
        observed_names = tuple(names[index] for index in observed_indices)
        if observed_names != expected_names:
            raise ValueError(
                "coordinate-preserving raw bank does not contain the exact complete ordered "
                f"nuisance family {key[0]} ({key[1]})"
            )
        indices_by_group[key] = observed_indices

    train_columns: list[np.ndarray] = []
    heldout_columns: list[np.ndarray] = []
    train_lineages: list[tuple[FitRowProvenance, ...]] = []
    heldout_lineages: list[tuple[FitRowProvenance, ...]] = []
    records: list[Mapping[str, Any]] = []
    one_sixth = float(np.float64(1.0) / np.float64(_BOW_COUNT))

    for output_name, output_kind, semantic, source_kind, role, operation in _OUTPUTS:
        indices = indices_by_group[(source_kind, role)]
        _assert_coordinate_probabilities(
            raw.train_oof_values,
            indices=indices,
            semantic=semantic,
            scope="train_oof",
        )
        _assert_coordinate_probabilities(
            raw.outer_heldout_values,
            indices=indices,
            semantic=semantic,
            scope="outer_heldout",
        )
        if operation == "equal_mean_six":
            train_value = _ordered_equal_mean(raw.train_oof_values, indices)
            heldout_value = _ordered_equal_mean(raw.outer_heldout_values, indices)
            arithmetic = {
                "operation": "ordered_float64_sum_then_multiply",
                "source_count": _BOW_COUNT,
                "weight_float64_hex": one_sixth.hex(),
            }
        else:
            if len(indices) != 1:
                raise ValueError("HTR nuisance family must be one exact singleton coordinate")
            train_value = np.asarray(raw.train_oof_values[:, indices[0]], dtype=np.float64).copy()
            heldout_value = np.asarray(
                raw.outer_heldout_values[:, indices[0]], dtype=np.float64
            ).copy()
            arithmetic = {
                "operation": "exact_float64_singleton_copy",
                "source_count": 1,
                "weight_float64_hex": float(1.0).hex(),
            }
        train_columns.append(train_value)
        heldout_columns.append(heldout_value)
        train_lineages.append(_combined_lineage(raw.train_oof_fit_row_provenance, indices))
        heldout_lineages.append(_combined_lineage(raw.outer_heldout_fit_row_provenance, indices))
        records.append(
            MappingProxyType(
                {
                    "output_name": output_name,
                    "output_kind": output_kind,
                    "semantic": semantic,
                    "source_coordinates": [
                        {
                            "raw_column_index": index,
                            "feature_name": names[index],
                            "feature_kind": kinds[index],
                            "consumer_role": roles[index],
                        }
                        for index in indices
                    ],
                    "arithmetic": arithmetic,
                    "train_source_lineage_sha256": _selected_lineage_sha256(
                        raw.train_oof_fit_row_provenance, indices
                    ),
                    "heldout_source_lineage_sha256": _selected_lineage_sha256(
                        raw.outer_heldout_fit_row_provenance, indices
                    ),
                }
            )
        )

    nuisance = SealedExactNuisanceBankExtension.seal_for_package(
        package,
        prediction_names=tuple(item[0] for item in _OUTPUTS),
        prediction_kinds=tuple(item[1] for item in _OUTPUTS),
        prediction_semantics=tuple(item[2] for item in _OUTPUTS),
        train_oof_values=np.column_stack(train_columns),
        outer_heldout_values=np.column_stack(heldout_columns),
        train_oof_fit_row_provenance=tuple(zip(*train_lineages)),
        outer_heldout_fit_row_provenance=tuple(zip(*heldout_lineages)),
    )
    return tuple(records), nuisance


def _derivation_digest(
    *,
    package_cache_key: str,
    package_manifest_sha256: str,
    raw_bank_content_sha256: str,
    producer_identity_sha256: str,
    coordinate_backend_identity_sha256: str,
    coordinate_schema_identity_sha256: str,
    contract_sha256: str,
    bow_view_names: tuple[str, ...],
    output_records: Sequence[Mapping[str, Any]],
    nuisance_content_sha256: str,
) -> str:
    return _sha256_json(
        {
            "schema_version": (AUTHENTICATED_COORDINATE_PRESERVING_NUISANCE_DERIVATION_SCHEMA),
            "bridge": AUTHENTICATED_COORDINATE_PRESERVING_NUISANCE_BRIDGE_ID,
            "package_cache_key": package_cache_key,
            "package_manifest_sha256": package_manifest_sha256,
            "raw_bank_content_sha256": raw_bank_content_sha256,
            "producer_identity_sha256": producer_identity_sha256,
            "coordinate_backend_identity_sha256": coordinate_backend_identity_sha256,
            "coordinate_schema_identity_sha256": coordinate_schema_identity_sha256,
            "coordinate_contract_sha256": contract_sha256,
            "bow_view_names": list(bow_view_names),
            "outputs": [_deep_thaw(item) for item in output_records],
            "nuisance_content_sha256": nuisance_content_sha256,
        }
    )


def _normalize_output_records(
    values: Sequence[Mapping[str, Any]],
    *,
    bow_view_names: tuple[str, ...],
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("output_records must be a sequence")
    records = tuple(values)
    if len(records) != len(_OUTPUTS) or not all(isinstance(item, Mapping) for item in records):
        raise ValueError("output_records must contain exactly four mappings")
    exact_fields = {
        "output_name",
        "output_kind",
        "semantic",
        "source_coordinates",
        "arithmetic",
        "train_source_lineage_sha256",
        "heldout_source_lineage_sha256",
    }
    exact_coordinate_fields = {
        "raw_column_index",
        "feature_name",
        "feature_kind",
        "consumer_role",
    }
    schema = coordinate_preserving_nuisance_schema(bow_view_names)
    used_indices: set[int] = set()
    normalized: list[Mapping[str, Any]] = []
    one_sixth = float(np.float64(1.0) / np.float64(_BOW_COUNT))
    for record, expected_output in zip(records, _OUTPUTS):
        if set(record) != exact_fields:
            raise ValueError("one output record does not match its closed schema")
        output_name, output_kind, semantic, source_kind, role, operation = expected_output
        if (
            record["output_name"] != output_name
            or record["output_kind"] != output_kind
            or record["semantic"] != semantic
        ):
            raise ValueError("output_records changed the four-output contract")
        raw_coordinates = record["source_coordinates"]
        if isinstance(raw_coordinates, (str, bytes, Mapping)):
            raise TypeError("source_coordinates must be a sequence")
        coordinates = tuple(raw_coordinates)
        expected_coordinates = tuple(
            item
            for item in schema
            if item["feature_kind"] == source_kind and item["consumer_role"] == role
        )
        if len(coordinates) != len(expected_coordinates):
            raise ValueError("source_coordinates changed exact family membership")
        prior_index = -1
        normalized_coordinates: list[Mapping[str, Any]] = []
        for coordinate, expected_coordinate in zip(coordinates, expected_coordinates):
            if not isinstance(coordinate, Mapping) or set(coordinate) != exact_coordinate_fields:
                raise ValueError("one source coordinate does not match its closed schema")
            raw_index = coordinate["raw_column_index"]
            if (
                isinstance(raw_index, (bool, np.bool_))
                or not isinstance(raw_index, (int, np.integer))
                or int(raw_index) < 0
            ):
                raise TypeError("raw_column_index must be a non-negative integer")
            index = int(raw_index)
            if index <= prior_index or index in used_indices:
                raise ValueError("source coordinate indices must be ordered and globally unique")
            prior_index = index
            used_indices.add(index)
            expected_metadata = {
                "feature_name": expected_coordinate["feature_name"],
                "feature_kind": expected_coordinate["feature_kind"],
                "consumer_role": expected_coordinate["consumer_role"],
            }
            if any(coordinate[key] != value for key, value in expected_metadata.items()):
                raise ValueError("source coordinate metadata changed the exact contract")
            normalized_coordinates.append(
                {
                    "raw_column_index": index,
                    **expected_metadata,
                }
            )
        expected_arithmetic = (
            {
                "operation": "ordered_float64_sum_then_multiply",
                "source_count": _BOW_COUNT,
                "weight_float64_hex": one_sixth.hex(),
            }
            if operation == "equal_mean_six"
            else {
                "operation": "exact_float64_singleton_copy",
                "source_count": 1,
                "weight_float64_hex": float(1.0).hex(),
            }
        )
        if not isinstance(record["arithmetic"], Mapping) or dict(record["arithmetic"]) != (
            expected_arithmetic
        ):
            raise ValueError("output arithmetic changed the exact contract")
        train_lineage_sha = _valid_sha256(
            record["train_source_lineage_sha256"],
            name="train_source_lineage_sha256",
        )
        heldout_lineage_sha = _valid_sha256(
            record["heldout_source_lineage_sha256"],
            name="heldout_source_lineage_sha256",
        )
        normalized.append(
            _deep_freeze(
                {
                    "output_name": output_name,
                    "output_kind": output_kind,
                    "semantic": semantic,
                    "source_coordinates": normalized_coordinates,
                    "arithmetic": expected_arithmetic,
                    "train_source_lineage_sha256": train_lineage_sha,
                    "heldout_source_lineage_sha256": heldout_lineage_sha,
                }
            )
        )
    return tuple(normalized)


@dataclass(frozen=True)
class AuthenticatedCoordinatePreservingNuisanceDerivation:
    """Four exact nuisances derived from fourteen authenticated coordinates."""

    package_cache_key: str
    package_manifest_sha256: str
    raw_bank_content_sha256: str
    producer_identity_sha256: str
    coordinate_backend_identity_sha256: str
    coordinate_schema_identity_sha256: str
    coordinate_contract_sha256: str
    bow_view_names: tuple[str, ...]
    output_records: tuple[Mapping[str, Any], ...]
    nuisance: SealedExactNuisanceBankExtension = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "package_cache_key",
            "package_manifest_sha256",
            "raw_bank_content_sha256",
            "producer_identity_sha256",
            "coordinate_backend_identity_sha256",
            "coordinate_schema_identity_sha256",
            "coordinate_contract_sha256",
            "content_sha256",
        ):
            object.__setattr__(self, name, _valid_sha256(getattr(self, name), name=name))
        views = _view_names(self.bow_view_names)
        if self.coordinate_contract_sha256 != (
            coordinate_preserving_nuisance_contract_sha256(views)
        ):
            raise ValueError("coordinate contract SHA-256 does not match its exact schema")
        if type(self.nuisance) is not SealedExactNuisanceBankExtension:
            raise TypeError("nuisance must use the exact sealed nuisance-extension type")
        records = _normalize_output_records(self.output_records, bow_view_names=views)
        if len(records) != len(_OUTPUTS):
            raise ValueError("output_records must contain exactly four nuisance derivations")
        object.__setattr__(self, "bow_view_names", views)
        object.__setattr__(self, "output_records", records)
        expected_digest = _derivation_digest(
            package_cache_key=self.package_cache_key,
            package_manifest_sha256=self.package_manifest_sha256,
            raw_bank_content_sha256=self.raw_bank_content_sha256,
            producer_identity_sha256=self.producer_identity_sha256,
            coordinate_backend_identity_sha256=self.coordinate_backend_identity_sha256,
            coordinate_schema_identity_sha256=self.coordinate_schema_identity_sha256,
            contract_sha256=self.coordinate_contract_sha256,
            bow_view_names=views,
            output_records=records,
            nuisance_content_sha256=self.nuisance.content_sha256,
        )
        if self.content_sha256 != expected_digest:
            raise ValueError("coordinate-preserving nuisance derivation digest mismatch")

    def audit_record(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": (AUTHENTICATED_COORDINATE_PRESERVING_NUISANCE_DERIVATION_SCHEMA),
                "bridge": AUTHENTICATED_COORDINATE_PRESERVING_NUISANCE_BRIDGE_ID,
                "package_cache_key": self.package_cache_key,
                "package_manifest_sha256": self.package_manifest_sha256,
                "raw_bank_content_sha256": self.raw_bank_content_sha256,
                "producer_identity_sha256": self.producer_identity_sha256,
                "coordinate_backend_identity_sha256": self.coordinate_backend_identity_sha256,
                "coordinate_schema_identity_sha256": self.coordinate_schema_identity_sha256,
                "coordinate_contract_sha256": self.coordinate_contract_sha256,
                "bow_view_names": list(self.bow_view_names),
                "outputs": [_deep_thaw(item) for item in self.output_records],
                "nuisance_content_sha256": self.nuisance.content_sha256,
                "semantic_inference_from_feature_names": False,
                "producer_identity_precommitted": True,
                "coordinate_contract_precommitted": True,
                "coordinate_preserving_backend_runtime_authenticated": True,
                "complete_bow_and_htr_family_membership_proved": True,
                "bow_htr_nuisance_volatile_reduction_allowed": False,
                "source_lineages_bound": True,
                "package_only_derivation_supported": False,
            }
        )

    def verify_authenticated_content(
        self,
        package: AuthenticatedFinalContextFitUpstreamBank,
        *,
        runtime_producer: FinalContextFitUpstreamProducer,
    ) -> None:
        producer_sha, backend_sha, schema_sha = _runtime_identity_sha256(
            package,
            runtime_producer,
            precommitted_sha256=self.producer_identity_sha256,
            bow_view_names=self.bow_view_names,
        )
        if (
            backend_sha != self.coordinate_backend_identity_sha256
            or schema_sha != self.coordinate_schema_identity_sha256
        ):
            raise ValueError("coordinate-preserving backend or schema identity changed")
        if (
            package.cache_key != self.package_cache_key
            or package.manifest_sha256 != self.package_manifest_sha256
            or package.raw_features.content_sha256 != self.raw_bank_content_sha256
        ):
            raise ValueError("coordinate nuisance derivation is bound to a different package")
        records, expected_nuisance = _derive_material(
            package,
            bow_view_names=self.bow_view_names,
        )
        if [_deep_thaw(item) for item in records] != [
            _deep_thaw(item) for item in self.output_records
        ]:
            raise ValueError("coordinate indices, metadata, arithmetic, or lineage changed")
        if expected_nuisance.content_sha256 != self.nuisance.content_sha256:
            raise ValueError("derived exact nuisance values or provenance changed")
        self.nuisance.validate_parent(package)
        expected_digest = _derivation_digest(
            package_cache_key=package.cache_key,
            package_manifest_sha256=package.manifest_sha256,
            raw_bank_content_sha256=package.raw_features.content_sha256,
            producer_identity_sha256=producer_sha,
            coordinate_backend_identity_sha256=backend_sha,
            coordinate_schema_identity_sha256=schema_sha,
            contract_sha256=self.coordinate_contract_sha256,
            bow_view_names=self.bow_view_names,
            output_records=records,
            nuisance_content_sha256=expected_nuisance.content_sha256,
        )
        if expected_digest != self.content_sha256:
            raise ValueError("coordinate-preserving nuisance derivation digest changed")


def derive_exact_nuisance_from_coordinate_preserved_stage1(
    package: AuthenticatedFinalContextFitUpstreamBank,
    *,
    runtime_producer: FinalContextFitUpstreamProducer,
    bow_view_names: Sequence[Any],
    precommitted_producer_identity_sha256: str,
    precommitted_coordinate_contract_sha256: str,
) -> AuthenticatedCoordinatePreservingNuisanceDerivation:
    """Derive BoW means and HTR singletons from exact named coordinates."""

    views = _view_names(bow_view_names)
    contract_sha = _valid_sha256(
        precommitted_coordinate_contract_sha256,
        name="precommitted_coordinate_contract_sha256",
    )
    if contract_sha != coordinate_preserving_nuisance_contract_sha256(views):
        raise ValueError("precommitted coordinate contract does not match bow_view_names")
    producer_sha, backend_sha, schema_sha = _runtime_identity_sha256(
        package,
        runtime_producer,
        precommitted_sha256=precommitted_producer_identity_sha256,
        bow_view_names=views,
    )
    records, nuisance = _derive_material(package, bow_view_names=views)
    digest = _derivation_digest(
        package_cache_key=package.cache_key,
        package_manifest_sha256=package.manifest_sha256,
        raw_bank_content_sha256=package.raw_features.content_sha256,
        producer_identity_sha256=producer_sha,
        coordinate_backend_identity_sha256=backend_sha,
        coordinate_schema_identity_sha256=schema_sha,
        contract_sha256=contract_sha,
        bow_view_names=views,
        output_records=records,
        nuisance_content_sha256=nuisance.content_sha256,
    )
    result = AuthenticatedCoordinatePreservingNuisanceDerivation(
        package_cache_key=package.cache_key,
        package_manifest_sha256=package.manifest_sha256,
        raw_bank_content_sha256=package.raw_features.content_sha256,
        producer_identity_sha256=producer_sha,
        coordinate_backend_identity_sha256=backend_sha,
        coordinate_schema_identity_sha256=schema_sha,
        coordinate_contract_sha256=contract_sha,
        bow_view_names=views,
        output_records=records,
        nuisance=nuisance,
        content_sha256=digest,
    )
    result.verify_authenticated_content(package, runtime_producer=runtime_producer)
    return result


__all__ = [
    "AUTHENTICATED_COORDINATE_PRESERVING_NUISANCE_BRIDGE_ID",
    "AUTHENTICATED_COORDINATE_PRESERVING_NUISANCE_DERIVATION_SCHEMA",
    "COORDINATE_PRESERVING_NUISANCE_CONTRACT_SCHEMA",
    "AuthenticatedCoordinatePreservingNuisanceDerivation",
    "coordinate_preserving_nuisance_contract_sha256",
    "coordinate_preserving_nuisance_schema",
    "derive_exact_nuisance_from_coordinate_preserved_stage1",
    "precommit_runtime_producer_identity_sha256",
]
