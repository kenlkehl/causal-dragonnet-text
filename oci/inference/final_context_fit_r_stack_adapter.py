"""Strict R-loss fusion for authenticated final context-fit tau banks.

``AuthenticatedFinalContextFitUpstreamBank`` currently separates calibrated
treatment-effect estimates from role-aware *raw* features.  A raw feature with
the propensity or outcome consumer role is not itself a nuisance prediction.
This module keeps that distinction closed: only the calibrated-source bank is
converted to R-stack signals, and residualization requires a separately sealed
bank of exact cross-fitted ``e_hat`` and ``m_hat`` predictions.

The nuisance extension is intentionally isolated until the final upstream
manifest gains a first-class exact-nuisance record.  It is content addressed
and bound to one exact upstream package, but callers must not mistake it for a
replacement for that future on-disk manifest integration.

No method accepts outer-heldout treatment, outer-heldout outcome, post-hoc
effect targets, or dataset-specific truth metadata.  The adapter delegates the
precommitted, fold-provenance-checked R-loss fit to ``FoldHonestRStack``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from .final_context_fit_upstream_bank import AuthenticatedFinalContextFitUpstreamBank
from .fold_honest_r_stack import (
    INNER_OOF_SCOPE,
    OUTER_HELDOUT_SCOPE,
    FitRowProvenance,
    FoldHonestRStack,
    SignalBundle,
)

FINAL_CONTEXT_FIT_R_STACK_ADAPTER_ID = "strict_final_context_fit_r_stack_adapter_v1"
SEALED_EXACT_NUISANCE_EXTENSION_SCHEMA = "sealed_exact_final_nuisance_extension_v1"
EXACT_PROPENSITY_PREDICTION = "exact_propensity_prediction"
EXACT_OUTCOME_PREDICTION = "exact_outcome_prediction"

_EXACT_SEMANTICS = frozenset({EXACT_PROPENSITY_PREDICTION, EXACT_OUTCOME_PREDICTION})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN = ("true", "oracle", "ground_truth")


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


def _valid_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized != str(value) or _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be one lowercase SHA-256 digest")
    return normalized


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be positive")
    return normalized


def _row_ids(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of row IDs")
    normalized: list[int] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain canonical integer row IDs")
        row_id = int(value)
        if row_id < 0:
            raise ValueError(f"{name} cannot contain negative row IDs")
        normalized.append(row_id)
    result = tuple(normalized)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


def _fold_ids(values: Sequence[Any], *, length: int) -> tuple[Hashable, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("meta_inner_fold_ids must be a sequence")
    normalized: list[Hashable] = []
    for value in tuple(values):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, str)):
            raise TypeError("meta_inner_fold_ids must contain integer or string IDs")
        if isinstance(value, str):
            candidate: Hashable = value.strip()
            if not candidate or any(token in candidate.lower() for token in _FORBIDDEN):
                raise ValueError("meta_inner_fold_ids contains an invalid string ID")
        else:
            candidate = int(value)
            if candidate < 1:
                raise ValueError("integer meta_inner_fold_ids must be positive")
        normalized.append(candidate)
    result = tuple(normalized)
    if len(result) != int(length):
        raise ValueError(f"meta_inner_fold_ids must have length {length}")
    if len(set(result)) < 2:
        raise ValueError("meta_inner_fold_ids must contain at least two folds")
    return result


def _safe_names(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(str(value).strip() for value in tuple(values))
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique non-empty strings")
    if any(any(token in value.lower() for token in _FORBIDDEN) for value in result):
        raise ValueError(f"{name} contains forbidden benchmark metadata")
    return result


def _aligned_strings(values: Sequence[Any], *, name: str, length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(str(value).strip() for value in tuple(values))
    if len(result) != int(length) or any(not value for value in result):
        raise ValueError(f"{name} must align with {length} columns")
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


def _lineage_payload(
    lineage: FitRowProvenance, *, active: set[int] | None = None
) -> Mapping[str, Any]:
    if not isinstance(lineage, FitRowProvenance):
        raise TypeError("lineage entries must be FitRowProvenance")
    stack = set() if active is None else active
    identity = id(lineage)
    if identity in stack:
        raise ValueError("fit-row lineage contains a cycle")
    stack.add(identity)
    try:
        rows: list[int] = []
        for value in lineage.fit_row_ids:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError("fit-row lineage must contain canonical integer IDs")
            row_id = int(value)
            if row_id < 0:
                raise ValueError("fit-row lineage cannot contain negative IDs")
            rows.append(row_id)
        return {
            "fit_row_ids": sorted(rows),
            "upstream": [_lineage_payload(parent, active=stack) for parent in lineage.upstream],
        }
    finally:
        stack.remove(identity)


def _provenance_payload(
    values: tuple[tuple[FitRowProvenance, ...], ...],
) -> list[list[Mapping[str, Any]]]:
    return [[_lineage_payload(lineage) for lineage in row] for row in values]


def _expected_train_fit_rows(
    row_ids: tuple[int, ...], folds: tuple[Hashable, ...]
) -> tuple[frozenset[int], ...]:
    by_fold = {
        fold: frozenset(row for row, candidate in zip(row_ids, folds) if candidate != fold)
        for fold in dict.fromkeys(folds)
    }
    if any(not rows for rows in by_fold.values()):
        raise ValueError("every meta-inner fold must have a non-empty complementary fit")
    return tuple(by_fold[fold] for fold in folds)


def _lineage_matrix(
    values: Sequence[Sequence[FitRowProvenance]],
    *,
    name: str,
    rows: int,
    columns: int,
    expected_fit_rows: Sequence[frozenset[int]],
) -> tuple[tuple[FitRowProvenance, ...], ...]:
    matrix = tuple(tuple(row) for row in tuple(values))
    if len(matrix) != int(rows) or any(len(row) != int(columns) for row in matrix):
        raise ValueError(f"{name} must have provenance shape {(rows, columns)}")
    if len(expected_fit_rows) != int(rows):
        raise ValueError(f"{name} expected-fit specification is misaligned")
    for row_index, (row, expected) in enumerate(zip(matrix, expected_fit_rows)):
        for lineage in row:
            if not isinstance(lineage, FitRowProvenance):
                raise TypeError(f"{name} entries must be FitRowProvenance")
            if lineage.recursive_fit_row_ids() != expected:
                raise ValueError(
                    f"{name} row {row_index} does not carry its exact complementary fit rows"
                )
    return matrix


def _extension_digest(
    *,
    outer_fold: int,
    parent_cache_key: str,
    parent_manifest_sha256: str,
    parent_producer_identity_sha256: str,
    train_row_ids: tuple[int, ...],
    heldout_row_ids: tuple[int, ...],
    meta_inner_fold_ids: tuple[Hashable, ...],
    prediction_names: tuple[str, ...],
    prediction_kinds: tuple[str, ...],
    prediction_semantics: tuple[str, ...],
    train_values: np.ndarray,
    heldout_values: np.ndarray,
    train_provenance: tuple[tuple[FitRowProvenance, ...], ...],
    heldout_provenance: tuple[tuple[FitRowProvenance, ...], ...],
) -> str:
    return _sha256_json(
        {
            "schema_version": SEALED_EXACT_NUISANCE_EXTENSION_SCHEMA,
            "outer_fold": outer_fold,
            "parent_cache_key": parent_cache_key,
            "parent_manifest_sha256": parent_manifest_sha256,
            "parent_producer_identity_sha256": parent_producer_identity_sha256,
            "train_row_ids": list(train_row_ids),
            "heldout_row_ids": list(heldout_row_ids),
            "meta_inner_fold_ids": list(meta_inner_fold_ids),
            "prediction_names": list(prediction_names),
            "prediction_kinds": list(prediction_kinds),
            "prediction_semantics": list(prediction_semantics),
            "train_values_sha256": _matrix_sha256(train_values),
            "heldout_values_sha256": _matrix_sha256(heldout_values),
            "train_provenance": _provenance_payload(train_provenance),
            "heldout_provenance": _provenance_payload(heldout_provenance),
        }
    )


@dataclass(frozen=True)
class SealedExactNuisanceBankExtension:
    """Exact binary nuisance predictions sealed to one final upstream package.

    ``prediction_semantics`` is a closed distinction between conditional
    treatment probabilities and conditional outcome means.  Generic raw
    features and role tags are not accepted here.
    """

    outer_fold: int
    parent_cache_key: str
    parent_manifest_sha256: str
    parent_producer_identity_sha256: str
    train_row_ids: tuple[int, ...]
    heldout_row_ids: tuple[int, ...]
    meta_inner_fold_ids: tuple[Hashable, ...]
    prediction_names: tuple[str, ...]
    prediction_kinds: tuple[str, ...]
    prediction_semantics: tuple[str, ...]
    train_oof_values: np.ndarray = field(repr=False)
    outer_heldout_values: np.ndarray = field(repr=False)
    train_oof_fit_row_provenance: tuple[tuple[FitRowProvenance, ...], ...] = field(repr=False)
    outer_heldout_fit_row_provenance: tuple[tuple[FitRowProvenance, ...], ...] = field(repr=False)
    content_sha256: str

    def __post_init__(self) -> None:
        fold = _positive_int(self.outer_fold, name="outer_fold")
        parent_cache_key = _valid_sha256(self.parent_cache_key, name="parent_cache_key")
        parent_manifest_sha = _valid_sha256(
            self.parent_manifest_sha256, name="parent_manifest_sha256"
        )
        parent_producer_sha = _valid_sha256(
            self.parent_producer_identity_sha256,
            name="parent_producer_identity_sha256",
        )
        train_rows = _row_ids(self.train_row_ids, name="train_row_ids")
        heldout_rows = _row_ids(self.heldout_row_ids, name="heldout_row_ids")
        if set(train_rows) & set(heldout_rows):
            raise ValueError("nuisance train and heldout rows must be disjoint")
        folds = _fold_ids(self.meta_inner_fold_ids, length=len(train_rows))
        names = _safe_names(self.prediction_names, name="prediction_names")
        kinds = _aligned_strings(self.prediction_kinds, name="prediction_kinds", length=len(names))
        semantics = _aligned_strings(
            self.prediction_semantics,
            name="prediction_semantics",
            length=len(names),
        )
        if set(semantics) - _EXACT_SEMANTICS:
            raise ValueError("prediction_semantics contains a non-nuisance semantic type")
        if set(semantics) != _EXACT_SEMANTICS:
            raise ValueError("exact nuisance bank requires propensity and outcome predictions")
        train_values = _finite_matrix(
            self.train_oof_values,
            name="train_oof_values",
            shape=(len(train_rows), len(names)),
        )
        heldout_values = _finite_matrix(
            self.outer_heldout_values,
            name="outer_heldout_values",
            shape=(len(heldout_rows), len(names)),
        )
        propensity_columns = [
            index
            for index, semantic in enumerate(semantics)
            if semantic == EXACT_PROPENSITY_PREDICTION
        ]
        outcome_columns = [
            index
            for index, semantic in enumerate(semantics)
            if semantic == EXACT_OUTCOME_PREDICTION
        ]
        for matrix_name, matrix in (
            ("train_oof_values", train_values),
            ("outer_heldout_values", heldout_values),
        ):
            propensity = matrix[:, propensity_columns]
            outcome = matrix[:, outcome_columns]
            if np.any(propensity <= 0.0) or np.any(propensity >= 1.0):
                raise ValueError(
                    f"{matrix_name} exact propensity predictions must be inside (0, 1)"
                )
            if np.any(outcome < 0.0) or np.any(outcome > 1.0):
                raise ValueError(f"{matrix_name} binary outcome predictions must be inside [0, 1]")
        train_provenance = _lineage_matrix(
            self.train_oof_fit_row_provenance,
            name="train_oof_fit_row_provenance",
            rows=len(train_rows),
            columns=len(names),
            expected_fit_rows=_expected_train_fit_rows(train_rows, folds),
        )
        heldout_provenance = _lineage_matrix(
            self.outer_heldout_fit_row_provenance,
            name="outer_heldout_fit_row_provenance",
            rows=len(heldout_rows),
            columns=len(names),
            expected_fit_rows=tuple(frozenset(train_rows) for _ in heldout_rows),
        )
        digest = _extension_digest(
            outer_fold=fold,
            parent_cache_key=parent_cache_key,
            parent_manifest_sha256=parent_manifest_sha,
            parent_producer_identity_sha256=parent_producer_sha,
            train_row_ids=train_rows,
            heldout_row_ids=heldout_rows,
            meta_inner_fold_ids=folds,
            prediction_names=names,
            prediction_kinds=kinds,
            prediction_semantics=semantics,
            train_values=train_values,
            heldout_values=heldout_values,
            train_provenance=train_provenance,
            heldout_provenance=heldout_provenance,
        )
        if _valid_sha256(self.content_sha256, name="content_sha256") != digest:
            raise ValueError("exact nuisance extension content SHA-256 mismatch")
        object.__setattr__(self, "outer_fold", fold)
        object.__setattr__(self, "parent_cache_key", parent_cache_key)
        object.__setattr__(self, "parent_manifest_sha256", parent_manifest_sha)
        object.__setattr__(self, "parent_producer_identity_sha256", parent_producer_sha)
        object.__setattr__(self, "train_row_ids", train_rows)
        object.__setattr__(self, "heldout_row_ids", heldout_rows)
        object.__setattr__(self, "meta_inner_fold_ids", folds)
        object.__setattr__(self, "prediction_names", names)
        object.__setattr__(self, "prediction_kinds", kinds)
        object.__setattr__(self, "prediction_semantics", semantics)
        object.__setattr__(self, "train_oof_values", train_values)
        object.__setattr__(self, "outer_heldout_values", heldout_values)
        object.__setattr__(self, "train_oof_fit_row_provenance", train_provenance)
        object.__setattr__(self, "outer_heldout_fit_row_provenance", heldout_provenance)
        object.__setattr__(self, "content_sha256", digest)

    @classmethod
    def seal_for_package(
        cls,
        package: AuthenticatedFinalContextFitUpstreamBank,
        *,
        prediction_names: Sequence[Any],
        prediction_kinds: Sequence[Any],
        prediction_semantics: Sequence[Any],
        train_oof_values: Any,
        outer_heldout_values: Any,
        train_oof_fit_row_provenance: Sequence[Sequence[FitRowProvenance]],
        outer_heldout_fit_row_provenance: Sequence[Sequence[FitRowProvenance]],
    ) -> "SealedExactNuisanceBankExtension":
        """Seal already-produced exact nuisance predictions to ``package``."""

        if type(package) is not AuthenticatedFinalContextFitUpstreamBank:
            raise TypeError("package must be the exact authenticated final upstream type")
        package.verify_authenticated_content()
        source = package.calibrated_sources
        train_rows = source.train_row_ids
        heldout_rows = source.heldout_row_ids
        folds = source.meta_inner_fold_ids
        names = _safe_names(prediction_names, name="prediction_names")
        kinds = _aligned_strings(prediction_kinds, name="prediction_kinds", length=len(names))
        semantics = _aligned_strings(
            prediction_semantics, name="prediction_semantics", length=len(names)
        )
        train_values = _finite_matrix(
            train_oof_values,
            name="train_oof_values",
            shape=(len(train_rows), len(names)),
        )
        heldout_values = _finite_matrix(
            outer_heldout_values,
            name="outer_heldout_values",
            shape=(len(heldout_rows), len(names)),
        )
        train_provenance = tuple(tuple(row) for row in train_oof_fit_row_provenance)
        heldout_provenance = tuple(tuple(row) for row in outer_heldout_fit_row_provenance)
        # The constructor performs the exact recursive lineage checks before the
        # returned extension can cross the estimator boundary.
        digest = _extension_digest(
            outer_fold=package.outer_fold,
            parent_cache_key=package.cache_key,
            parent_manifest_sha256=package.manifest_sha256,
            parent_producer_identity_sha256=package.producer_identity_sha256,
            train_row_ids=train_rows,
            heldout_row_ids=heldout_rows,
            meta_inner_fold_ids=folds,
            prediction_names=names,
            prediction_kinds=kinds,
            prediction_semantics=semantics,
            train_values=train_values,
            heldout_values=heldout_values,
            train_provenance=train_provenance,
            heldout_provenance=heldout_provenance,
        )
        result = cls(
            outer_fold=package.outer_fold,
            parent_cache_key=package.cache_key,
            parent_manifest_sha256=package.manifest_sha256,
            parent_producer_identity_sha256=package.producer_identity_sha256,
            train_row_ids=train_rows,
            heldout_row_ids=heldout_rows,
            meta_inner_fold_ids=folds,
            prediction_names=names,
            prediction_kinds=kinds,
            prediction_semantics=semantics,
            train_oof_values=train_values,
            outer_heldout_values=heldout_values,
            train_oof_fit_row_provenance=train_provenance,
            outer_heldout_fit_row_provenance=heldout_provenance,
            content_sha256=digest,
        )
        package.verify_authenticated_content()
        return result

    def verify_authenticated_content(self) -> None:
        digest = _extension_digest(
            outer_fold=self.outer_fold,
            parent_cache_key=self.parent_cache_key,
            parent_manifest_sha256=self.parent_manifest_sha256,
            parent_producer_identity_sha256=self.parent_producer_identity_sha256,
            train_row_ids=self.train_row_ids,
            heldout_row_ids=self.heldout_row_ids,
            meta_inner_fold_ids=self.meta_inner_fold_ids,
            prediction_names=self.prediction_names,
            prediction_kinds=self.prediction_kinds,
            prediction_semantics=self.prediction_semantics,
            train_values=self.train_oof_values,
            heldout_values=self.outer_heldout_values,
            train_provenance=self.train_oof_fit_row_provenance,
            heldout_provenance=self.outer_heldout_fit_row_provenance,
        )
        if digest != self.content_sha256:
            raise ValueError("exact nuisance extension in-memory content was modified")

    def validate_parent(self, package: AuthenticatedFinalContextFitUpstreamBank) -> None:
        if type(package) is not AuthenticatedFinalContextFitUpstreamBank:
            raise TypeError("package must be the exact authenticated final upstream type")
        package.verify_authenticated_content()
        source = package.calibrated_sources
        if (
            self.outer_fold != package.outer_fold
            or self.parent_cache_key != package.cache_key
            or self.parent_manifest_sha256 != package.manifest_sha256
            or self.parent_producer_identity_sha256 != package.producer_identity_sha256
            or self.train_row_ids != source.train_row_ids
            or self.heldout_row_ids != source.heldout_row_ids
            or self.meta_inner_fold_ids != source.meta_inner_fold_ids
        ):
            raise ValueError("exact nuisance extension is not bound to this upstream package")
        self.verify_authenticated_content()

    def mean_prediction(self, semantic: str, *, scope: str) -> np.ndarray:
        normalized = str(semantic).strip()
        if normalized not in _EXACT_SEMANTICS:
            raise ValueError("semantic must name one exact nuisance prediction type")
        if scope == INNER_OOF_SCOPE:
            matrix = self.train_oof_values
        elif scope == OUTER_HELDOUT_SCOPE:
            matrix = self.outer_heldout_values
        else:
            raise ValueError("scope must be inner_oof or outer_heldout")
        columns = [
            index
            for index, candidate in enumerate(self.prediction_semantics)
            if candidate == normalized
        ]
        result = np.mean(matrix[:, columns], axis=1)
        result.setflags(write=False)
        return result


def _tau_bundles(
    package: AuthenticatedFinalContextFitUpstreamBank, *, scope: str
) -> tuple[SignalBundle, ...]:
    source = package.calibrated_sources
    if scope == INNER_OOF_SCOPE:
        row_ids = source.train_row_ids
        values = source.train_oof_values
        provenance = source.train_oof_fit_row_provenance
    elif scope == OUTER_HELDOUT_SCOPE:
        row_ids = source.heldout_row_ids
        values = source.outer_heldout_values
        provenance = source.outer_heldout_fit_row_provenance
    else:
        raise ValueError("scope must be inner_oof or outer_heldout")
    return tuple(
        SignalBundle(
            row_ids=row_ids,
            source_family=f"authenticated_calibrated_tau_{column + 1:03d}",
            tau_predictions=values[:, column],
            prediction_scope=scope,
            fit_row_provenance=tuple(row[column] for row in provenance),
        )
        for column in range(len(source.source_names))
    )


class StrictFinalContextFitRStackAdapter:
    """Fuse exact calibrated tau sources with exact cross-fitted nuisances."""

    def __init__(self, *, ridge_alpha: float = 1.0, nonnegative: bool = True) -> None:
        alpha = float(ridge_alpha)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("ridge_alpha must be finite and non-negative")
        if not isinstance(nonnegative, bool):
            raise TypeError("nonnegative must be boolean")
        self.ridge_alpha = alpha
        self.nonnegative = nonnegative
        self._stack = FoldHonestRStack(
            ridge_alphas=(alpha,),
            nonnegative=nonnegative,
        )

    def fit(
        self,
        package: AuthenticatedFinalContextFitUpstreamBank,
        *,
        outer_train_row_ids: Sequence[Any],
        treatment: Sequence[Any],
        outcome: Sequence[Any],
        exact_nuisance: SealedExactNuisanceBankExtension,
    ) -> "StrictFinalContextFitRStackAdapter":
        if type(package) is not AuthenticatedFinalContextFitUpstreamBank:
            raise TypeError("package must be the exact authenticated final upstream type")
        if type(exact_nuisance) is not SealedExactNuisanceBankExtension:
            raise TypeError("exact_nuisance must use the sealed exact-nuisance type")
        package.verify_authenticated_content()
        exact_nuisance.validate_parent(package)
        requested_rows = _row_ids(outer_train_row_ids, name="outer_train_row_ids")
        source = package.calibrated_sources
        if requested_rows != source.train_row_ids:
            raise ValueError("R-stack fit row identity or order changed")
        treatment_vector = _finite_vector(treatment, name="treatment", length=len(requested_rows))
        outcome_vector = _finite_vector(outcome, name="outcome", length=len(requested_rows))
        propensity = np.array(
            exact_nuisance.mean_prediction(EXACT_PROPENSITY_PREDICTION, scope=INNER_OOF_SCOPE),
            copy=True,
        )
        outcome_prediction = np.array(
            exact_nuisance.mean_prediction(EXACT_OUTCOME_PREDICTION, scope=INNER_OOF_SCOPE),
            copy=True,
        )
        train_signals = _tau_bundles(package, scope=INNER_OOF_SCOPE)
        # Re-authenticate after all estimator inputs have been copied.  Neither
        # role-aware raw feature matrix is ever materialized by this adapter.
        package.verify_authenticated_content()
        exact_nuisance.verify_authenticated_content()
        self._stack.fit(
            row_ids=requested_rows,
            treatment=treatment_vector,
            outcome=outcome_vector,
            propensity=propensity,
            outcome_prediction=outcome_prediction,
            inner_fold_ids=source.meta_inner_fold_ids,
            signals=train_signals,
        )
        self.outer_fold_ = package.outer_fold
        self.package_cache_key_ = package.cache_key
        self.package_manifest_sha256_ = package.manifest_sha256
        self.package_producer_identity_sha256_ = package.producer_identity_sha256
        self.source_bank_content_sha256_ = source.content_sha256
        self.exact_nuisance_content_sha256_ = exact_nuisance.content_sha256
        self.outer_train_row_ids_ = source.train_row_ids
        self.outer_heldout_row_ids_ = source.heldout_row_ids
        self.meta_inner_fold_ids_ = source.meta_inner_fold_ids
        self.source_names_ = source.source_names
        self.source_kinds_ = source.source_kinds
        self.nuisance_names_ = exact_nuisance.prediction_names
        self.nuisance_kinds_ = exact_nuisance.prediction_kinds
        self.nuisance_semantics_ = exact_nuisance.prediction_semantics
        return self

    def predict_effect(
        self,
        package: AuthenticatedFinalContextFitUpstreamBank,
        *,
        exact_nuisance: SealedExactNuisanceBankExtension,
    ) -> np.ndarray:
        self._validate_prediction_inputs(package, exact_nuisance=exact_nuisance)
        predictions = self._stack.predict(
            row_ids=package.calibrated_sources.heldout_row_ids,
            signals=_tau_bundles(package, scope=OUTER_HELDOUT_SCOPE),
        )
        package.verify_authenticated_content()
        exact_nuisance.verify_authenticated_content()
        result = np.asarray(predictions, dtype=float).copy()
        result.setflags(write=False)
        return result

    def predict_bundle(
        self,
        package: AuthenticatedFinalContextFitUpstreamBank,
        *,
        exact_nuisance: SealedExactNuisanceBankExtension,
    ) -> SignalBundle:
        self._validate_prediction_inputs(package, exact_nuisance=exact_nuisance)
        result = self._stack.predict_bundle(
            row_ids=package.calibrated_sources.heldout_row_ids,
            signals=_tau_bundles(package, scope=OUTER_HELDOUT_SCOPE),
            source_family="strict_authenticated_final_r_stack",
        )
        package.verify_authenticated_content()
        exact_nuisance.verify_authenticated_content()
        return result

    def audit_record(self) -> Mapping[str, Any]:
        self._require_fitted()
        source_weights = {
            name: float(weight) for name, weight in zip(self.source_names_, self._stack.weights_)
        }
        return MappingProxyType(
            {
                "adapter": FINAL_CONTEXT_FIT_R_STACK_ADAPTER_ID,
                "outer_fold": self.outer_fold_,
                "package_cache_key": self.package_cache_key_,
                "package_manifest_sha256": self.package_manifest_sha256_,
                "package_producer_identity_sha256": self.package_producer_identity_sha256_,
                "calibrated_source_bank_content_sha256": self.source_bank_content_sha256_,
                "sealed_exact_nuisance_content_sha256": self.exact_nuisance_content_sha256_,
                "calibrated_source_names": list(self.source_names_),
                "calibrated_source_kinds": list(self.source_kinds_),
                "source_weights": source_weights,
                "nuisance_names": list(self.nuisance_names_),
                "nuisance_kinds": list(self.nuisance_kinds_),
                "nuisance_semantics": list(self.nuisance_semantics_),
                "nuisance_ensemble": "precommitted_equal_mean_within_semantic_role",
                "precommitted_ridge_alpha": self.ridge_alpha,
                "nonnegative_tau_weights": self.nonnegative,
                "outer_train_r_loss": float(self._stack.training_r_loss_),
                "raw_feature_bank_used_as_tau": False,
                "raw_feature_bank_used_as_nuisance_predictions": False,
                "outer_heldout_labels_accepted_by_adapter": False,
                "posthoc_effect_targets_accepted_by_adapter": False,
                "htr_embedding_cache_added_by_adapter": False,
                "exact_nuisance_extension_is_first_class_parent_manifest_record": False,
            }
        )

    def _validate_prediction_inputs(
        self,
        package: AuthenticatedFinalContextFitUpstreamBank,
        *,
        exact_nuisance: SealedExactNuisanceBankExtension,
    ) -> None:
        self._require_fitted()
        if type(package) is not AuthenticatedFinalContextFitUpstreamBank:
            raise TypeError("package must be the exact authenticated final upstream type")
        if type(exact_nuisance) is not SealedExactNuisanceBankExtension:
            raise TypeError("exact_nuisance must use the sealed exact-nuisance type")
        exact_nuisance.validate_parent(package)
        source = package.calibrated_sources
        identity = (
            package.outer_fold,
            package.cache_key,
            package.manifest_sha256,
            package.producer_identity_sha256,
            source.content_sha256,
            exact_nuisance.content_sha256,
            source.train_row_ids,
            source.heldout_row_ids,
            source.meta_inner_fold_ids,
            source.source_names,
            source.source_kinds,
        )
        expected = (
            self.outer_fold_,
            self.package_cache_key_,
            self.package_manifest_sha256_,
            self.package_producer_identity_sha256_,
            self.source_bank_content_sha256_,
            self.exact_nuisance_content_sha256_,
            self.outer_train_row_ids_,
            self.outer_heldout_row_ids_,
            self.meta_inner_fold_ids_,
            self.source_names_,
            self.source_kinds_,
        )
        if identity != expected:
            raise ValueError("prediction package or exact nuisance identity changed after fit")

    def _require_fitted(self) -> None:
        if not hasattr(self, "package_cache_key_"):
            raise RuntimeError("StrictFinalContextFitRStackAdapter must be fit before use")


__all__ = [
    "EXACT_OUTCOME_PREDICTION",
    "EXACT_PROPENSITY_PREDICTION",
    "FINAL_CONTEXT_FIT_R_STACK_ADAPTER_ID",
    "SEALED_EXACT_NUISANCE_EXTENSION_SCHEMA",
    "SealedExactNuisanceBankExtension",
    "StrictFinalContextFitRStackAdapter",
]
