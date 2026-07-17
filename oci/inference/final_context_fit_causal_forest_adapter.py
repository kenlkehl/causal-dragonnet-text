"""Outer-honest causal-forest routing for final context-fit evidence.

The final upstream package contains two semantically different banks:

* calibrated treatment-effect predictions, which are effect-modifier inputs;
* uncalibrated role-aware raw features, which are routed according to their
  declared modifier or nuisance-basis role.

This module performs one final fit on the complete outer-train OOF matrix and
one prediction on the label-free outer-heldout full-fit transform.  That use is
outer-honest.  It deliberately cannot emit meta-inner OOF forest predictions:
the assembled train bank contains one gate transform per row, but does not
contain the complement-only, context-inner-OOF feature matrix required to fit
each meta-inner forest without recursive target-fold leakage.

Exact nuisance predictions remain distinct from raw nuisance bases.  When
provided, they enter the existing CausalForestDML path as control covariates;
the adapter does not claim that ``CausalForestHead`` consumes them as fixed
nuisance functions.  A future residualized low-level forest may consume them
directly behind a separately authenticated backend.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import marshal
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .all_evidence_post_extraction_review import (
    OUTCOME_NUISANCE_FEATURE_ROLE,
    PROPENSITY_NUISANCE_FEATURE_ROLE,
    UNCALIBRATED_EFFECT_MODIFIER_ROLE,
)
from .final_context_fit_r_stack_adapter import SealedExactNuisanceBankExtension
from .final_context_fit_upstream_bank import AuthenticatedFinalContextFitUpstreamBank
from .fold_honest_r_stack import FitRowProvenance

FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID = (
    "strict_outer_honest_final_context_fit_causal_forest_v2"
)
FINAL_FOREST_EXPLICIT_BLOCK_SCHEMA = "sealed_final_forest_explicit_block_v1"
FINAL_FOREST_TAU_SCHEMA = "sealed_outer_heldout_final_forest_tau_v2"

_FORBIDDEN = ("true", "oracle", "ground_truth")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class NestedFinalForestFeaturesRequired(RuntimeError):
    """Raised when current row-wise OOF inputs are asked to prove forest OOF tau."""


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


def _valid_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized != str(value) or _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return normalized


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _row_ids(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    result: list[int] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain canonical integer row IDs")
        row_id = int(value)
        if row_id < 0:
            raise ValueError(f"{name} cannot contain negative row IDs")
        result.append(row_id)
    normalized = tuple(result)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be non-empty and unique")
    return normalized


def _safe_names(values: Sequence[Any], *, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(str(value).strip() for value in tuple(values))
    if (not allow_empty and not result) or any(not value for value in result):
        raise ValueError(f"{name} contains an empty name")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique names")
    if any(any(token in value.lower() for token in _FORBIDDEN) for value in result):
        raise ValueError(f"{name} contains forbidden benchmark metadata")
    return result


def _finite_matrix(values: Any, *, name: str, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite matrix with shape {shape}")
    result = np.array(matrix, dtype=float, copy=True, order="C")
    result.setflags(write=False)
    return result


def _finite_vector(values: Any, *, name: str, length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) != int(length) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector with length {length}")
    result = vector.copy()
    result.setflags(write=False)
    return result


def _matrix_sha256(values: np.ndarray) -> str:
    matrix = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    header = _canonical_json({"dtype": "<f8", "shape": list(matrix.shape), "order": "C"}).encode(
        "utf-8"
    )
    return hashlib.sha256(header + b"\0" + matrix.tobytes(order="C")).hexdigest()


def _closed_identity(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key or any(token in key.lower() for token in _FORBIDDEN):
                raise ValueError(f"{path} contains a forbidden or empty field")
            if key in result:
                raise ValueError(f"{path} contains colliding fields")
            result[key] = _closed_identity(raw_value, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_closed_identity(item, path=f"{path}[]") for item in value]
    if isinstance(value, np.generic):
        return _closed_identity(value.item(), path=path)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if any(token in value.lower() for token in _FORBIDDEN):
            raise ValueError(f"{path} contains forbidden benchmark metadata")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite metadata")
        return value
    raise TypeError(f"{path} must contain closed JSON-compatible metadata")


def _explicit_digest(
    *,
    parent_cache_key: str,
    train_row_ids: tuple[int, ...],
    heldout_row_ids: tuple[int, ...],
    effect_names: tuple[str, ...],
    control_names: tuple[str, ...],
    effect_train: np.ndarray,
    effect_heldout: np.ndarray,
    control_train: np.ndarray,
    control_heldout: np.ndarray,
) -> str:
    return _sha256_json(
        {
            "schema_version": FINAL_FOREST_EXPLICIT_BLOCK_SCHEMA,
            "parent_cache_key": parent_cache_key,
            "train_row_ids": list(train_row_ids),
            "heldout_row_ids": list(heldout_row_ids),
            "effect_names": list(effect_names),
            "control_names": list(control_names),
            "effect_train_sha256": _matrix_sha256(effect_train),
            "effect_heldout_sha256": _matrix_sha256(effect_heldout),
            "control_train_sha256": _matrix_sha256(control_train),
            "control_heldout_sha256": _matrix_sha256(control_heldout),
        }
    )


@dataclass(frozen=True)
class SealedFinalForestExplicitBlock:
    """Label-free explicit modifier/control matrices bound to one package split."""

    parent_cache_key: str
    train_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    effect_names: tuple[str, ...]
    control_names: tuple[str, ...]
    effect_train_values: np.ndarray = field(repr=False)
    effect_heldout_values: np.ndarray = field(repr=False)
    control_train_values: np.ndarray = field(repr=False)
    control_heldout_values: np.ndarray = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        parent = _valid_sha256(self.parent_cache_key, name="parent_cache_key")
        train_rows = _row_ids(self.train_row_ids, name="train_row_ids")
        heldout_rows = _row_ids(self.heldout_row_ids, name="heldout_row_ids")
        if set(train_rows) & set(heldout_rows):
            raise ValueError("explicit train and heldout rows must be disjoint")
        effect_names = _safe_names(self.effect_names, name="effect_names", allow_empty=True)
        control_names = _safe_names(self.control_names, name="control_names", allow_empty=True)
        # A contract may legitimately be both a confounder and an effect
        # modifier.  Preserve the same encoded column in both matrices; the
        # downstream opaque X/W namespaces remain disjoint.
        effect_train = _finite_matrix(
            self.effect_train_values,
            name="effect_train_values",
            shape=(len(train_rows), len(effect_names)),
        )
        effect_heldout = _finite_matrix(
            self.effect_heldout_values,
            name="effect_heldout_values",
            shape=(len(heldout_rows), len(effect_names)),
        )
        control_train = _finite_matrix(
            self.control_train_values,
            name="control_train_values",
            shape=(len(train_rows), len(control_names)),
        )
        control_heldout = _finite_matrix(
            self.control_heldout_values,
            name="control_heldout_values",
            shape=(len(heldout_rows), len(control_names)),
        )
        digest = _explicit_digest(
            parent_cache_key=parent,
            train_row_ids=train_rows,
            heldout_row_ids=heldout_rows,
            effect_names=effect_names,
            control_names=control_names,
            effect_train=effect_train,
            effect_heldout=effect_heldout,
            control_train=control_train,
            control_heldout=control_heldout,
        )
        if _valid_sha256(self.content_sha256, name="content_sha256") != digest:
            raise ValueError("explicit forest block content SHA-256 mismatch")
        object.__setattr__(self, "parent_cache_key", parent)
        object.__setattr__(self, "train_row_ids", train_rows)
        object.__setattr__(self, "heldout_row_ids", heldout_rows)
        object.__setattr__(self, "effect_names", effect_names)
        object.__setattr__(self, "control_names", control_names)
        object.__setattr__(self, "effect_train_values", effect_train)
        object.__setattr__(self, "effect_heldout_values", effect_heldout)
        object.__setattr__(self, "control_train_values", control_train)
        object.__setattr__(self, "control_heldout_values", control_heldout)
        object.__setattr__(self, "content_sha256", digest)

    @classmethod
    def seal_for_package(
        cls,
        package: AuthenticatedFinalContextFitUpstreamBank,
        *,
        effect_names: Sequence[Any],
        control_names: Sequence[Any],
        effect_train_values: Any,
        effect_heldout_values: Any,
        control_train_values: Any,
        control_heldout_values: Any,
    ) -> "SealedFinalForestExplicitBlock":
        if type(package) is not AuthenticatedFinalContextFitUpstreamBank:
            raise TypeError("package must be the exact authenticated final upstream type")
        package.verify_authenticated_content()
        source = package.calibrated_sources
        effect = _safe_names(effect_names, name="effect_names", allow_empty=True)
        control = _safe_names(control_names, name="control_names", allow_empty=True)
        effect_train = _finite_matrix(
            effect_train_values,
            name="effect_train_values",
            shape=(len(source.train_row_ids), len(effect)),
        )
        effect_heldout = _finite_matrix(
            effect_heldout_values,
            name="effect_heldout_values",
            shape=(len(source.heldout_row_ids), len(effect)),
        )
        control_train = _finite_matrix(
            control_train_values,
            name="control_train_values",
            shape=(len(source.train_row_ids), len(control)),
        )
        control_heldout = _finite_matrix(
            control_heldout_values,
            name="control_heldout_values",
            shape=(len(source.heldout_row_ids), len(control)),
        )
        digest = _explicit_digest(
            parent_cache_key=package.cache_key,
            train_row_ids=source.train_row_ids,
            heldout_row_ids=source.heldout_row_ids,
            effect_names=effect,
            control_names=control,
            effect_train=effect_train,
            effect_heldout=effect_heldout,
            control_train=control_train,
            control_heldout=control_heldout,
        )
        result = cls(
            parent_cache_key=package.cache_key,
            train_row_ids=source.train_row_ids,
            heldout_row_ids=source.heldout_row_ids,
            effect_names=effect,
            control_names=control,
            effect_train_values=effect_train,
            effect_heldout_values=effect_heldout,
            control_train_values=control_train,
            control_heldout_values=control_heldout,
            content_sha256=digest,
        )
        package.verify_authenticated_content()
        return result

    def validate_parent(self, package: AuthenticatedFinalContextFitUpstreamBank) -> None:
        package.verify_authenticated_content()
        source = package.calibrated_sources
        if (
            self.parent_cache_key != package.cache_key
            or self.train_row_ids != source.train_row_ids
            or self.heldout_row_ids != source.heldout_row_ids
        ):
            raise ValueError("explicit forest block is not bound to this package split")
        self.verify_authenticated_content()

    def verify_authenticated_content(self) -> None:
        digest = _explicit_digest(
            parent_cache_key=self.parent_cache_key,
            train_row_ids=self.train_row_ids,
            heldout_row_ids=self.heldout_row_ids,
            effect_names=self.effect_names,
            control_names=self.control_names,
            effect_train=self.effect_train_values,
            effect_heldout=self.effect_heldout_values,
            control_train=self.control_train_values,
            control_heldout=self.control_heldout_values,
        )
        if digest != self.content_sha256:
            raise ValueError("explicit forest block in-memory content was modified")


@dataclass(frozen=True)
class FinalCausalForestDesign:
    """Copied, role-routed matrices for one final outer-heldout forest fit."""

    train_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    effect_names: tuple[str, ...]
    control_names: tuple[str, ...]
    effect_train_values: np.ndarray = field(repr=False)
    effect_heldout_values: np.ndarray = field(repr=False)
    control_train_values: np.ndarray = field(repr=False)
    control_heldout_values: np.ndarray = field(repr=False)
    routing_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        train_rows = _row_ids(self.train_row_ids, name="design.train_row_ids")
        heldout_rows = _row_ids(self.heldout_row_ids, name="design.heldout_row_ids")
        effect_names = _safe_names(self.effect_names, name="design.effect_names")
        control_names = _safe_names(self.control_names, name="design.control_names")
        if set(effect_names) & set(control_names):
            raise ValueError("forest effect and control namespaces overlap")
        effect_train = _finite_matrix(
            self.effect_train_values,
            name="design.effect_train_values",
            shape=(len(train_rows), len(effect_names)),
        )
        effect_heldout = _finite_matrix(
            self.effect_heldout_values,
            name="design.effect_heldout_values",
            shape=(len(heldout_rows), len(effect_names)),
        )
        control_train = _finite_matrix(
            self.control_train_values,
            name="design.control_train_values",
            shape=(len(train_rows), len(control_names)),
        )
        control_heldout = _finite_matrix(
            self.control_heldout_values,
            name="design.control_heldout_values",
            shape=(len(heldout_rows), len(control_names)),
        )
        audit = _closed_identity(self.routing_audit, path="routing_audit")
        object.__setattr__(self, "train_row_ids", train_rows)
        object.__setattr__(self, "heldout_row_ids", heldout_rows)
        object.__setattr__(self, "effect_names", effect_names)
        object.__setattr__(self, "control_names", control_names)
        object.__setattr__(self, "effect_train_values", effect_train)
        object.__setattr__(self, "effect_heldout_values", effect_heldout)
        object.__setattr__(self, "control_train_values", control_train)
        object.__setattr__(self, "control_heldout_values", control_heldout)
        object.__setattr__(self, "routing_audit", MappingProxyType(dict(audit)))


def prepare_final_causal_forest_design(
    package: AuthenticatedFinalContextFitUpstreamBank,
    *,
    exact_nuisance: SealedExactNuisanceBankExtension,
    explicit_features: SealedFinalForestExplicitBlock,
) -> FinalCausalForestDesign:
    """Authenticate and route one package without consuming any labels."""

    if type(package) is not AuthenticatedFinalContextFitUpstreamBank:
        raise TypeError("package must be the exact authenticated final upstream type")
    if type(exact_nuisance) is not SealedExactNuisanceBankExtension:
        raise TypeError("exact_nuisance must use the sealed exact-nuisance type")
    if type(explicit_features) is not SealedFinalForestExplicitBlock:
        raise TypeError("explicit_features must use the sealed explicit block type")
    package.verify_authenticated_content()
    exact_nuisance.validate_parent(package)
    explicit_features.validate_parent(package)
    source = package.calibrated_sources
    raw = package.raw_features
    effect_raw_indices = tuple(
        index
        for index, role in enumerate(raw.consumer_roles)
        if role == UNCALIBRATED_EFFECT_MODIFIER_ROLE
    )
    control_raw_indices = tuple(
        index
        for index, role in enumerate(raw.consumer_roles)
        if role in {PROPENSITY_NUISANCE_FEATURE_ROLE, OUTCOME_NUISANCE_FEATURE_ROLE}
    )

    source_effect_names = tuple(
        f"upstream_calibrated_tau_{index + 1:03d}" for index in range(len(source.source_names))
    )
    raw_effect_names = tuple(
        f"upstream_raw_modifier_{position + 1:03d}" for position in range(len(effect_raw_indices))
    )
    raw_control_names = tuple(
        f"upstream_raw_control_{position + 1:03d}" for position in range(len(control_raw_indices))
    )
    nuisance_control_names = tuple(
        f"upstream_exact_nuisance_{index + 1:03d}"
        for index in range(len(exact_nuisance.prediction_names))
    )
    explicit_effect_names = tuple(
        f"explicit_modifier_{index + 1:03d}" for index in range(len(explicit_features.effect_names))
    )
    explicit_control_names = tuple(
        f"explicit_control_{index + 1:03d}" for index in range(len(explicit_features.control_names))
    )

    effect_train = np.column_stack(
        (
            source.train_oof_values,
            raw.train_oof_values[:, effect_raw_indices],
            explicit_features.effect_train_values,
        )
    )
    effect_heldout = np.column_stack(
        (
            source.outer_heldout_values,
            raw.outer_heldout_values[:, effect_raw_indices],
            explicit_features.effect_heldout_values,
        )
    )
    control_train = np.column_stack(
        (
            raw.train_oof_values[:, control_raw_indices],
            exact_nuisance.train_oof_values,
            explicit_features.control_train_values,
        )
    )
    control_heldout = np.column_stack(
        (
            raw.outer_heldout_values[:, control_raw_indices],
            exact_nuisance.outer_heldout_values,
            explicit_features.control_heldout_values,
        )
    )
    effect_names = (
        *source_effect_names,
        *raw_effect_names,
        *explicit_effect_names,
    )
    control_names = (
        *raw_control_names,
        *nuisance_control_names,
        *explicit_control_names,
    )
    routing_audit = {
        "schema": "final_causal_forest_role_routing_v1",
        "package_cache_key": package.cache_key,
        "package_manifest_sha256": package.manifest_sha256,
        "source_bank_content_sha256": source.content_sha256,
        "raw_bank_content_sha256": raw.content_sha256,
        "exact_nuisance_content_sha256": exact_nuisance.content_sha256,
        "explicit_block_content_sha256": explicit_features.content_sha256,
        "effect_columns": {
            "calibrated_tau_count": len(source.source_names),
            "raw_modifier_count": len(effect_raw_indices),
            "explicit_modifier_count": len(explicit_features.effect_names),
        },
        "control_columns": {
            "raw_nuisance_basis_count": len(control_raw_indices),
            "exact_nuisance_prediction_count": len(exact_nuisance.prediction_names),
            "explicit_control_count": len(explicit_features.control_names),
        },
        "calibrated_tau_routed_as_effect_modifiers": True,
        "raw_modifier_features_relabelled_as_calibrated_tau": False,
        "raw_nuisance_bases_relabelled_as_exact_predictions": False,
        "exact_nuisance_routed_as_fixed_causal_forest_nuisance": False,
        "exact_nuisance_routed_as_control_covariates": True,
        "outer_train_values_are_meta_inner_oof": True,
        "outer_heldout_values_are_full_outer_train_transforms": True,
        "safe_for_single_final_outer_heldout_forest_fit": True,
        "safe_for_meta_inner_forest_oof_generation": False,
    }
    # Re-authenticate after every estimator-facing value has been copied.
    package.verify_authenticated_content()
    exact_nuisance.verify_authenticated_content()
    explicit_features.verify_authenticated_content()
    return FinalCausalForestDesign(
        train_row_ids=source.train_row_ids,
        heldout_row_ids=source.heldout_row_ids,
        effect_names=effect_names,
        control_names=control_names,
        effect_train_values=effect_train,
        effect_heldout_values=effect_heldout,
        control_train_values=control_train,
        control_heldout_values=control_heldout,
        routing_audit=routing_audit,
    )


@runtime_checkable
class FinalCausalForestBackend(Protocol):
    """Narrow label-safe backend for one final heldout forest prediction."""

    def identity(self) -> Mapping[str, Any]: ...

    def fit_predict(
        self,
        *,
        effect_train: np.ndarray,
        control_train: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        effect_heldout: np.ndarray,
        control_heldout: np.ndarray,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class FixedCausalForestHeadBackend:
    """Precommitted wrapper preserving the prior working forest defaults.

    EconML's tuning step consumes outer-train labels only.  It remains enabled
    here because that was part of the recovered working causal-forest path and
    does not expose the outer-heldout labels.
    """

    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 10
    max_features: str | float | int = "sqrt"
    honest: bool = True
    inference: bool = True
    tune_model: bool = True
    random_state: int = 42

    def __post_init__(self) -> None:
        trees = _positive_int(self.n_estimators, name="n_estimators")
        if trees % 4 != 0:
            raise ValueError("n_estimators must be divisible by four")
        if self.max_depth is not None:
            _positive_int(self.max_depth, name="max_depth")
        _positive_int(self.min_samples_leaf, name="min_samples_leaf")
        for name, value in (
            ("honest", self.honest),
            ("inference", self.inference),
            ("tune_model", self.tune_model),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be boolean")
        if not self.honest:
            raise ValueError("final causal forest must keep honest tree splitting enabled")
        if isinstance(self.random_state, (bool, np.bool_)) or not isinstance(
            self.random_state, (int, np.integer)
        ):
            raise TypeError("random_state must be an integer")

    def identity(self) -> Mapping[str, Any]:
        return {
            "backend": "repository_causal_forest_prior_working_path_v2",
            "n_estimators": int(self.n_estimators),
            "max_depth": self.max_depth,
            "min_samples_leaf": int(self.min_samples_leaf),
            "max_features": self.max_features,
            "honest": self.honest,
            "inference": self.inference,
            "tune_model": self.tune_model,
            "random_state": int(self.random_state),
            "exact_nuisance_used_as_fixed_internal_predictions": False,
            "tuning_labels": "outer_train_only",
            "outer_heldout_labels_accepted": False,
            "repository_runtime": dict(_repository_causal_forest_runtime_attestation()),
        }

    def fit_predict(
        self,
        *,
        effect_train: np.ndarray,
        control_train: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        effect_heldout: np.ndarray,
        control_heldout: np.ndarray,
    ) -> np.ndarray:
        from ..models import causal_forest_head as causal_forest_head_module

        runtime_before = _repository_causal_forest_runtime_attestation()
        if not runtime_before["econml_import_available"]:
            raise RuntimeError(
                "the fixed final causal forest cannot import EconML under the current "
                "runtime; repair the locked environment before fitting"
            )
        CausalForestHead = causal_forest_head_module.CausalForestHead

        model = CausalForestHead(
            n_estimators=int(self.n_estimators),
            max_depth=self.max_depth,
            min_samples_leaf=int(self.min_samples_leaf),
            max_features=self.max_features,
            honest=self.honest,
            inference=self.inference,
            random_state=int(self.random_state),
            tune_model=self.tune_model,
        )
        model.fit(
            X=np.asarray(effect_train, dtype=float),
            T=np.asarray(treatment, dtype=float),
            Y=np.asarray(outcome, dtype=float),
            W=np.asarray(control_train, dtype=float),
        )
        model_fit_audit = model.fit_audit()
        runtime_after = _repository_causal_forest_runtime_attestation()
        if runtime_after != runtime_before:
            raise RuntimeError("causal-forest implementation changed during model fitting")
        object.__setattr__(
            self,
            "_last_fit_audit",
            MappingProxyType(
                {
                    **copy.deepcopy(model_fit_audit),
                    "outer_train_labels_only": True,
                    "outer_heldout_labels_accepted": False,
                    "repository_runtime": copy.deepcopy(dict(runtime_after)),
                }
            ),
        )
        result = model.predict(np.asarray(effect_heldout, dtype=float), return_ci=False)
        if not isinstance(result, Mapping) or "tau_pred" not in result:
            raise TypeError("CausalForestHead returned no tau_pred vector")
        return np.asarray(result["tau_pred"], dtype=float)

    def fit_audit(self) -> Mapping[str, Any]:
        if not hasattr(self, "_last_fit_audit"):
            raise RuntimeError("fixed causal-forest backend has not completed fitting")
        return copy.deepcopy(dict(self._last_fit_audit))


def _unwrap_callable(value: Any) -> Any:
    if isinstance(value, (staticmethod, classmethod)):
        return value.__func__
    return value


def _method_code_sha256(owner: type[Any], name: str) -> str:
    value = _unwrap_callable(inspect.getattr_static(owner, name))
    code = getattr(value, "__code__", None)
    if code is None:
        raise TypeError(f"forest backend {name} must be implemented in Python")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _function_code_sha256(value: Any, *, name: str) -> str:
    function = _unwrap_callable(value)
    code = getattr(function, "__code__", None)
    if code is None:
        raise TypeError(f"causal-forest runtime function {name} must be implemented in Python")
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _optional_class_method_code(owner: type[Any], name: str) -> str | None:
    try:
        value = _unwrap_callable(inspect.getattr_static(owner, name))
    except AttributeError:
        return None
    code = getattr(value, "__code__", None)
    if code is None:
        return None
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


def _repository_causal_forest_runtime_attestation() -> Mapping[str, Any]:
    """Bind the dynamically imported forest implementation and dependencies."""

    from ..models import causal_forest_head as module

    module_path = Path(module.__file__).resolve(strict=True)
    head = module.CausalForestHead
    result: dict[str, Any] = {
        "causal_forest_head_module_sha256": _sha256_file(module_path),
        "causal_forest_head_class_module": head.__module__,
        "causal_forest_head_class_qualname": head.__qualname__,
        "causal_forest_head_init_code_sha256": _method_code_sha256(head, "__init__"),
        "causal_forest_head_create_model_code_sha256": _method_code_sha256(head, "_create_model"),
        "causal_forest_head_fit_code_sha256": _method_code_sha256(head, "fit"),
        "causal_forest_head_predict_code_sha256": _method_code_sha256(head, "predict"),
        "causal_forest_head_fit_audit_code_sha256": _method_code_sha256(head, "fit_audit"),
        "tune_wrapper_code_sha256": _function_code_sha256(
            module.tune_causal_forest_model,
            name="tune_causal_forest_model",
        ),
        "econml_distribution_version": _installed_version("econml"),
        "sklearn_distribution_version": _installed_version("scikit-learn"),
        "numpy_distribution_version": _installed_version("numpy"),
        "python_version": sys.version.split()[0],
        "econml_import_available": bool(module.ECONML_AVAILABLE),
    }
    for label, estimator_class in (
        ("propensity_random_forest", getattr(module, "RandomForestClassifier", None)),
        ("outcome_random_forest", getattr(module, "RandomForestRegressor", None)),
    ):
        if estimator_class is None:
            result[f"{label}_class_module"] = None
            result[f"{label}_class_qualname"] = None
            result[f"{label}_class_module_sha256"] = None
            continue
        estimator_source = inspect.getsourcefile(estimator_class)
        if not estimator_source:
            raise TypeError(f"{label} must come from a file-backed module")
        result[f"{label}_class_module"] = estimator_class.__module__
        result[f"{label}_class_qualname"] = estimator_class.__qualname__
        result[f"{label}_class_module_sha256"] = _sha256_file(
            Path(estimator_source).resolve(strict=True)
        )
    econml_class = module.CausalForestDML
    if econml_class is not None:
        source = inspect.getsourcefile(econml_class)
        if not source:
            raise TypeError("EconML CausalForestDML must come from a file-backed module")
        result.update(
            {
                "econml_class_module": econml_class.__module__,
                "econml_class_qualname": econml_class.__qualname__,
                "econml_class_module_sha256": _sha256_file(Path(source).resolve(strict=True)),
                "econml_fit_code_sha256": _optional_class_method_code(econml_class, "fit"),
                "econml_effect_code_sha256": _optional_class_method_code(econml_class, "effect"),
            }
        )
    else:
        result.update(
            {
                "econml_class_module": None,
                "econml_class_qualname": None,
                "econml_class_module_sha256": None,
                "econml_fit_code_sha256": None,
                "econml_effect_code_sha256": None,
            }
        )
    return MappingProxyType(result)


def _backend_attestation(backend: FinalCausalForestBackend) -> Mapping[str, Any]:
    owner = type(backend)
    if any(name in vars(backend) for name in ("identity", "fit_predict", "fit_audit")):
        raise TypeError("forest backend has unauthenticated per-instance method overrides")
    source_file = inspect.getsourcefile(owner)
    if not source_file:
        raise TypeError("forest backend class must come from a Python source file")
    path = Path(source_file).resolve(strict=True)
    result = {
        "class_module": owner.__module__,
        "class_qualname": owner.__qualname__,
        "module_file_sha256": _sha256_file(path),
        "identity_code_sha256": _method_code_sha256(owner, "identity"),
        "fit_predict_code_sha256": _method_code_sha256(owner, "fit_predict"),
    }
    if callable(getattr(backend, "fit_audit", None)):
        result["fit_audit_code_sha256"] = _method_code_sha256(owner, "fit_audit")
    if owner is FixedCausalForestHeadBackend:
        result["repository_causal_forest_runtime"] = dict(
            _repository_causal_forest_runtime_attestation()
        )
    return result


def _tau_digest(
    *,
    outer_fold: int,
    package_cache_key: str,
    package_manifest_sha256: str,
    backend_identity_sha256: str,
    design_sha256: str,
    heldout_row_ids: tuple[int, ...],
    values: np.ndarray,
    provenance: tuple[FitRowProvenance, ...],
) -> str:
    return _sha256_json(
        {
            "schema_version": FINAL_FOREST_TAU_SCHEMA,
            "outer_fold": outer_fold,
            "package_cache_key": package_cache_key,
            "package_manifest_sha256": package_manifest_sha256,
            "backend_identity_sha256": backend_identity_sha256,
            "design_sha256": design_sha256,
            "heldout_row_ids": list(heldout_row_ids),
            "values_sha256": _matrix_sha256(values[:, None]),
            "recursive_fit_rows": [
                sorted(int(row) for row in lineage.recursive_fit_row_ids())
                for lineage in provenance
            ],
        }
    )


@dataclass(frozen=True)
class SealedOuterHeldoutForestTau:
    """One outer-heldout forest effect vector; no train-OOF counterpart is implied."""

    outer_fold: int
    package_cache_key: str
    package_manifest_sha256: str
    backend_identity_sha256: str
    design_sha256: str
    heldout_row_ids: tuple[int, ...]
    values: np.ndarray = field(repr=False)
    fit_row_provenance: tuple[FitRowProvenance, ...] = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        fold = _positive_int(self.outer_fold, name="outer_fold")
        cache_key = _valid_sha256(self.package_cache_key, name="package_cache_key")
        manifest_sha = _valid_sha256(self.package_manifest_sha256, name="package_manifest_sha256")
        backend_sha = _valid_sha256(self.backend_identity_sha256, name="backend_identity_sha256")
        design_sha = _valid_sha256(self.design_sha256, name="design_sha256")
        rows = _row_ids(self.heldout_row_ids, name="heldout_row_ids")
        values = _finite_vector(self.values, name="values", length=len(rows))
        provenance = tuple(self.fit_row_provenance)
        if len(provenance) != len(rows) or not all(
            isinstance(item, FitRowProvenance) for item in provenance
        ):
            raise TypeError("fit_row_provenance must contain one lineage per heldout row")
        for row, lineage in zip(rows, provenance):
            if row in lineage.recursive_fit_row_ids():
                raise ValueError("outer-heldout forest tau lineage contains its prediction row")
        digest = _tau_digest(
            outer_fold=fold,
            package_cache_key=cache_key,
            package_manifest_sha256=manifest_sha,
            backend_identity_sha256=backend_sha,
            design_sha256=design_sha,
            heldout_row_ids=rows,
            values=values,
            provenance=provenance,
        )
        if _valid_sha256(self.content_sha256, name="content_sha256") != digest:
            raise ValueError("outer-heldout forest tau content SHA-256 mismatch")
        object.__setattr__(self, "outer_fold", fold)
        object.__setattr__(self, "package_cache_key", cache_key)
        object.__setattr__(self, "package_manifest_sha256", manifest_sha)
        object.__setattr__(self, "backend_identity_sha256", backend_sha)
        object.__setattr__(self, "design_sha256", design_sha)
        object.__setattr__(self, "heldout_row_ids", rows)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "fit_row_provenance", provenance)
        object.__setattr__(self, "content_sha256", digest)

    def verify_authenticated_content(self) -> None:
        digest = _tau_digest(
            outer_fold=self.outer_fold,
            package_cache_key=self.package_cache_key,
            package_manifest_sha256=self.package_manifest_sha256,
            backend_identity_sha256=self.backend_identity_sha256,
            design_sha256=self.design_sha256,
            heldout_row_ids=self.heldout_row_ids,
            values=self.values,
            provenance=self.fit_row_provenance,
        )
        if digest != self.content_sha256:
            raise ValueError("outer-heldout forest tau in-memory content was modified")


class StrictOuterHonestFinalCausalForestAdapter:
    """Run one authenticated, fixed final causal forest for outer heldout rows."""

    def __init__(self, *, backend: FinalCausalForestBackend) -> None:
        if not isinstance(backend, FinalCausalForestBackend):
            raise TypeError("backend must implement identity() and fit_predict()")
        self.backend = backend
        self._backend_identity = _closed_identity(backend.identity(), path="backend.identity")
        self._backend_attestation = _backend_attestation(backend)

    def _assert_backend_stable(self) -> None:
        if (
            _closed_identity(self.backend.identity(), path="backend.identity")
            != self._backend_identity
        ):
            raise ValueError("causal-forest backend identity changed")
        if _backend_attestation(self.backend) != self._backend_attestation:
            raise ValueError("causal-forest backend runtime code changed")

    def fit_predict(
        self,
        package: AuthenticatedFinalContextFitUpstreamBank,
        *,
        outer_train_row_ids: Sequence[Any],
        treatment: Sequence[Any],
        outcome: Sequence[Any],
        exact_nuisance: SealedExactNuisanceBankExtension,
        explicit_features: SealedFinalForestExplicitBlock,
    ) -> SealedOuterHeldoutForestTau:
        self._assert_backend_stable()
        design = prepare_final_causal_forest_design(
            package,
            exact_nuisance=exact_nuisance,
            explicit_features=explicit_features,
        )
        requested_rows = _row_ids(outer_train_row_ids, name="outer_train_row_ids")
        if requested_rows != design.train_row_ids:
            raise ValueError("causal-forest fit row identity or order changed")
        treatment_vector = _finite_vector(treatment, name="treatment", length=len(requested_rows))
        outcome_vector = _finite_vector(outcome, name="outcome", length=len(requested_rows))
        if set(np.unique(treatment_vector).tolist()) != {0.0, 1.0}:
            raise ValueError("treatment must contain binary 0/1 values")
        values = self.backend.fit_predict(
            effect_train=np.array(design.effect_train_values, copy=True),
            control_train=np.array(design.control_train_values, copy=True),
            treatment=np.array(treatment_vector, copy=True),
            outcome=np.array(outcome_vector, copy=True),
            effect_heldout=np.array(design.effect_heldout_values, copy=True),
            control_heldout=np.array(design.control_heldout_values, copy=True),
        )
        fit_audit_method = getattr(self.backend, "fit_audit", None)
        backend_fit_audit: Mapping[str, Any] | None = None
        if callable(fit_audit_method):
            backend_fit_audit = _closed_identity(
                fit_audit_method(),
                path="backend.fit_audit",
            )
            if not isinstance(backend_fit_audit, Mapping):
                raise TypeError("causal-forest backend fit audit must be a mapping")
        elif bool(self._backend_identity.get("tune_model", False)):
            raise TypeError(
                "a tuning-enabled causal-forest backend must expose its actual fit audit"
            )
        self._assert_backend_stable()
        package.verify_authenticated_content()
        exact_nuisance.verify_authenticated_content()
        explicit_features.verify_authenticated_content()
        tau = _finite_vector(values, name="forest_tau", length=len(design.heldout_row_ids))
        backend_identity_sha = _sha256_json(
            {
                "identity": self._backend_identity,
                "runtime": self._backend_attestation,
            }
        )
        design_sha = _sha256_json(
            {
                "effect_names": list(design.effect_names),
                "control_names": list(design.control_names),
                "effect_train_sha256": _matrix_sha256(design.effect_train_values),
                "effect_heldout_sha256": _matrix_sha256(design.effect_heldout_values),
                "control_train_sha256": _matrix_sha256(design.control_train_values),
                "control_heldout_sha256": _matrix_sha256(design.control_heldout_values),
                "routing_audit": dict(design.routing_audit),
            }
        )
        full_fit = FitRowProvenance(fit_row_ids=frozenset(design.train_row_ids))
        source = package.calibrated_sources
        raw = package.raw_features
        provenance = tuple(
            FitRowProvenance(
                fit_row_ids=frozenset(design.train_row_ids),
                upstream=(
                    full_fit,
                    *source.outer_heldout_fit_row_provenance[row_index],
                    *raw.outer_heldout_fit_row_provenance[row_index],
                    *exact_nuisance.outer_heldout_fit_row_provenance[row_index],
                ),
            )
            for row_index in range(len(design.heldout_row_ids))
        )
        digest = _tau_digest(
            outer_fold=package.outer_fold,
            package_cache_key=package.cache_key,
            package_manifest_sha256=package.manifest_sha256,
            backend_identity_sha256=backend_identity_sha,
            design_sha256=design_sha,
            heldout_row_ids=design.heldout_row_ids,
            values=tau,
            provenance=provenance,
        )
        result = SealedOuterHeldoutForestTau(
            outer_fold=package.outer_fold,
            package_cache_key=package.cache_key,
            package_manifest_sha256=package.manifest_sha256,
            backend_identity_sha256=backend_identity_sha,
            design_sha256=design_sha,
            heldout_row_ids=design.heldout_row_ids,
            values=tau,
            fit_row_provenance=provenance,
            content_sha256=digest,
        )
        self.last_audit_ = MappingProxyType(
            {
                "adapter": FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID,
                "outer_fold": package.outer_fold,
                "package_cache_key": package.cache_key,
                "package_manifest_sha256": package.manifest_sha256,
                "backend_identity": copy.deepcopy(self._backend_identity),
                "backend_identity_sha256": backend_identity_sha,
                "design_sha256": design_sha,
                "routing": dict(design.routing_audit),
                "forest_tau_content_sha256": result.content_sha256,
                "forest_fit_audit": (
                    None if backend_fit_audit is None else copy.deepcopy(dict(backend_fit_audit))
                ),
                "forest_tuning_from_assembled_oof_bank": bool(
                    backend_fit_audit is not None
                    and backend_fit_audit.get("tuning_attempted", False)
                ),
                "forest_tuning_succeeded": (
                    None if backend_fit_audit is None else backend_fit_audit.get("tuning_succeeded")
                ),
                "forest_tuning_failure_fell_back_to_configured_parameters": bool(
                    backend_fit_audit is not None
                    and backend_fit_audit.get(
                        "tuning_failure_fell_back_to_configured_parameters", False
                    )
                ),
                "forest_effective_parameters": (
                    None
                    if backend_fit_audit is None
                    else copy.deepcopy(backend_fit_audit.get("effective_parameters"))
                ),
                "forest_tuning_confined_to_outer_train": True,
                "forest_tuning_used_outer_heldout_labels": False,
                "single_final_outer_heldout_fit": True,
                "meta_inner_forest_oof_emitted": False,
                "outer_heldout_labels_accepted": False,
                "htr_encoder_or_embedding_policy_changed": False,
            }
        )
        return result

    def emit_meta_inner_oof_tau(self, *_args: Any, **_kwargs: Any) -> None:
        raise NestedFinalForestFeaturesRequired(
            "Current AuthenticatedFinalContextFitUpstreamBank has only the assembled "
            "row-wise meta-inner gate matrix. Honest forest OOF tau additionally needs, "
            "for every meta-inner gate, a complement-only context-inner-OOF fit matrix "
            "plus that same complement fit's gate transform and exact nuisance predictions."
        )

    def audit_record(self) -> Mapping[str, Any]:
        if not hasattr(self, "last_audit_"):
            raise RuntimeError("causal-forest adapter has not produced a prediction")
        return self.last_audit_


__all__ = [
    "FINAL_CONTEXT_FIT_CAUSAL_FOREST_ADAPTER_ID",
    "FINAL_FOREST_EXPLICIT_BLOCK_SCHEMA",
    "FINAL_FOREST_TAU_SCHEMA",
    "FinalCausalForestDesign",
    "FinalCausalForestBackend",
    "FixedCausalForestHeadBackend",
    "NestedFinalForestFeaturesRequired",
    "SealedFinalForestExplicitBlock",
    "SealedOuterHeldoutForestTau",
    "StrictOuterHonestFinalCausalForestAdapter",
    "prepare_final_causal_forest_design",
]
