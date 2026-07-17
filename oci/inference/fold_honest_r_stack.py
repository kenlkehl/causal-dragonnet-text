"""Fold-honest regularized stacking for treatment-effect signals.

The stack consumes only cross-fitted treatment and outcome residuals and
cross-fitted treatment-effect predictions.  It deliberately has no interface
for oracle treatment effects or oracle nuisance values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Sequence

import numpy as np
from scipy.optimize import lsq_linear


INNER_OOF_SCOPE = "inner_oof"
OUTER_HELDOUT_SCOPE = "outer_heldout"
_VALID_PREDICTION_SCOPES = {INNER_OOF_SCOPE, OUTER_HELDOUT_SCOPE}


def _normalize_key(value: Any, *, name: str) -> Hashable:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        raise ValueError(f"{name} cannot contain missing values")
    try:
        is_missing = bool(value != value)
    except (TypeError, ValueError):
        is_missing = False
    if is_missing:
        raise ValueError(f"{name} cannot contain missing values")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{name} values must be hashable, got {value!r}") from exc
    return value


def _key_tuple(values: Iterable[Any], *, name: str) -> tuple[Hashable, ...]:
    keys = tuple(_normalize_key(value, name=name) for value in values)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must be unique")
    return keys


def _numeric_vector(values: Sequence[float], *, name: str, length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) != int(length):
        raise ValueError(f"{name} must be one-dimensional with length {length}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


@dataclass(frozen=True)
class FitRowProvenance:
    """Recursive target-dependent fit-row lineage for one prediction.

    ``fit_row_ids`` records rows used directly to fit the predicting model.
    ``upstream`` records the corresponding lineage of fitted inputs or targets,
    such as nuisance models used to construct an R-learner pseudo-outcome.
    """

    fit_row_ids: frozenset[Hashable] = field(default_factory=frozenset)
    upstream: tuple["FitRowProvenance", ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized = frozenset(
            _normalize_key(value, name="fit_row_ids") for value in self.fit_row_ids
        )
        upstream = tuple(self.upstream)
        if not all(isinstance(item, FitRowProvenance) for item in upstream):
            raise TypeError("upstream entries must be FitRowProvenance instances")
        object.__setattr__(self, "fit_row_ids", normalized)
        object.__setattr__(self, "upstream", upstream)

    def recursive_fit_row_ids(self) -> frozenset[Hashable]:
        """Return direct and upstream fit rows, rejecting cyclic lineage."""

        result: set[Hashable] = set()
        visited: set[int] = set()
        active: set[int] = set()

        def visit(node: FitRowProvenance) -> None:
            identity = id(node)
            if identity in active:
                raise ValueError("Fit-row provenance contains a cycle")
            if identity in visited:
                return
            active.add(identity)
            result.update(node.fit_row_ids)
            for parent in node.upstream:
                visit(parent)
            active.remove(identity)
            visited.add(identity)

        visit(self)
        return frozenset(result)


@dataclass(frozen=True)
class SignalBundle:
    """One named treatment-effect signal with row-level honesty lineage."""

    row_ids: tuple[Hashable, ...]
    source_family: str
    tau_predictions: np.ndarray = field(repr=False)
    prediction_scope: str
    fit_row_provenance: tuple[FitRowProvenance, ...] = field(repr=False)

    def __post_init__(self) -> None:
        row_ids = _key_tuple(self.row_ids, name="row_ids")
        source_family = str(self.source_family).strip()
        if not source_family:
            raise ValueError("source_family must be non-empty")
        scope = str(self.prediction_scope).strip().lower()
        if scope not in _VALID_PREDICTION_SCOPES:
            raise ValueError(
                f"prediction_scope must be one of {sorted(_VALID_PREDICTION_SCOPES)}"
            )
        predictions = _numeric_vector(
            self.tau_predictions,
            name="tau_predictions",
            length=len(row_ids),
        ).copy()
        predictions.setflags(write=False)
        provenance = tuple(self.fit_row_provenance)
        if len(provenance) != len(row_ids):
            raise ValueError("fit_row_provenance must have one entry per prediction row")
        if not all(isinstance(item, FitRowProvenance) for item in provenance):
            raise TypeError(
                "fit_row_provenance entries must be FitRowProvenance instances"
            )
        recursive_rows_by_lineage: dict[int, frozenset[Hashable]] = {}
        for row_id, lineage in zip(row_ids, provenance):
            lineage_id = id(lineage)
            recursive_rows = recursive_rows_by_lineage.get(lineage_id)
            if recursive_rows is None:
                recursive_rows = lineage.recursive_fit_row_ids()
                recursive_rows_by_lineage[lineage_id] = recursive_rows
            if row_id in recursive_rows:
                raise ValueError(
                    "Fold-honesty violation for source "
                    f"{source_family!r}: prediction row {row_id!r} occurs in its "
                    "recursive fit-row provenance"
                )

        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "source_family", source_family)
        object.__setattr__(self, "tau_predictions", predictions)
        object.__setattr__(self, "prediction_scope", scope)
        object.__setattr__(self, "fit_row_provenance", provenance)

    def aligned_predictions(self, row_ids: Sequence[Hashable]) -> np.ndarray:
        """Return predictions in the requested row order."""

        requested = _key_tuple(row_ids, name="requested row_ids")
        positions = {row_id: index for index, row_id in enumerate(self.row_ids)}
        missing = [row_id for row_id in requested if row_id not in positions]
        if missing:
            raise ValueError(
                f"Signal {self.source_family!r} is missing requested rows {missing[:3]}"
            )
        return np.asarray(
            [self.tau_predictions[positions[row_id]] for row_id in requested],
            dtype=float,
        )

    def provenance_for(self, row_id: Hashable) -> FitRowProvenance:
        normalized = _normalize_key(row_id, name="row_id")
        positions = {value: index for index, value in enumerate(self.row_ids)}
        if normalized not in positions:
            raise ValueError(
                f"Signal {self.source_family!r} has no provenance for row {normalized!r}"
            )
        return self.fit_row_provenance[positions[normalized]]

    def aligned_provenance(
        self,
        row_ids: Sequence[Hashable],
    ) -> tuple[FitRowProvenance, ...]:
        """Return row-level lineage in the requested row order."""

        requested = _key_tuple(row_ids, name="requested row_ids")
        positions = {value: index for index, value in enumerate(self.row_ids)}
        missing = [row_id for row_id in requested if row_id not in positions]
        if missing:
            raise ValueError(
                f"Signal {self.source_family!r} is missing requested rows {missing[:3]}"
            )
        return tuple(self.fit_row_provenance[positions[row_id]] for row_id in requested)


class FoldHonestRStack:
    """Regularized linear R-stack with precommitted regularization.

    Alpha grids are deliberately rejected.  Selecting an alpha after looking
    across the complete OOF bank can adapt to rows that also influenced source
    models for other OOF cells.  A single precommitted alpha is simpler and has
    an auditable data-flow boundary.
    """

    def __init__(
        self,
        *,
        ridge_alphas: Sequence[float] = (1.0,),
        nonnegative: bool = False,
    ) -> None:
        alphas = sorted({float(value) for value in ridge_alphas})
        if not alphas or any(not np.isfinite(value) or value < 0.0 for value in alphas):
            raise ValueError("ridge_alphas must contain finite non-negative values")
        if len(alphas) != 1:
            raise ValueError(
                "adaptive ridge alpha grids are forbidden; supply one precommitted alpha"
            )
        if not isinstance(nonnegative, bool):
            raise TypeError("nonnegative must be a boolean")
        self.ridge_alphas = tuple(alphas)
        self.precommitted_alpha = float(alphas[0])
        self.nonnegative = nonnegative

    def fit(
        self,
        *,
        row_ids: Sequence[Hashable],
        treatment: Sequence[float],
        outcome: Sequence[float],
        propensity: Sequence[float],
        outcome_prediction: Sequence[float],
        inner_fold_ids: Sequence[Hashable],
        signals: Sequence[SignalBundle],
    ) -> "FoldHonestRStack":
        """Fit once using the precommitted alpha and cross-fitted residuals."""

        train_row_ids = _key_tuple(row_ids, name="row_ids")
        n_rows = len(train_row_ids)
        if n_rows < 2:
            raise ValueError("At least two training rows are required")
        treatment_vector = _numeric_vector(treatment, name="treatment", length=n_rows)
        outcome_vector = _numeric_vector(outcome, name="outcome", length=n_rows)
        propensity_vector = _numeric_vector(propensity, name="propensity", length=n_rows)
        outcome_vector_hat = _numeric_vector(
            outcome_prediction,
            name="outcome_prediction",
            length=n_rows,
        )
        if not np.isin(treatment_vector, [0.0, 1.0]).all():
            raise ValueError("treatment must be binary with values 0/1")
        if not np.isin(outcome_vector, [0.0, 1.0]).all():
            raise ValueError("outcome must be binary with values 0/1")
        if np.any(propensity_vector <= 0.0) or np.any(propensity_vector >= 1.0):
            raise ValueError("propensity must be finite and strictly inside (0, 1)")
        if np.any(outcome_vector_hat < 0.0) or np.any(outcome_vector_hat > 1.0):
            raise ValueError("binary outcome_prediction must be inside [0, 1]")
        folds = tuple(
            _normalize_key(value, name="inner_fold_ids") for value in inner_fold_ids
        )
        if len(folds) != n_rows:
            raise ValueError(f"inner_fold_ids must have length {n_rows}")
        unique_folds = tuple(dict.fromkeys(folds))
        if len(unique_folds) < 2:
            raise ValueError("inner_fold_ids must contain at least two folds")

        signal_matrix, source_families, _ = _aligned_signal_matrix(
            signals,
            train_row_ids,
            required_scope=INNER_OOF_SCOPE,
        )
        _validate_inner_fold_provenance(signals, train_row_ids, folds)
        source_means = np.mean(signal_matrix, axis=0)
        source_scales = np.std(signal_matrix, axis=0, ddof=0)
        source_scales = np.where(source_scales > 1e-12, source_scales, 1.0)
        standardized_signal_matrix = (signal_matrix - source_means) / source_scales
        treatment_residual = treatment_vector - propensity_vector
        outcome_residual = outcome_vector - outcome_vector_hat
        if float(np.dot(treatment_residual, treatment_residual)) <= np.finfo(float).eps:
            raise ValueError("Treatment residuals contain no information for R-stacking")

        coefficients = _fit_r_coefficients(
            standardized_signal_matrix,
            treatment_residual,
            outcome_residual,
            alpha=self.precommitted_alpha,
            nonnegative=self.nonnegative,
        )
        standardized_weights = np.asarray(coefficients[1:], dtype=float).copy()
        raw_weights = standardized_weights / source_scales
        raw_constant = float(coefficients[0] - np.dot(source_means, raw_weights))
        for vector in (source_means, source_scales, standardized_weights, raw_weights):
            vector.setflags(write=False)

        self.training_row_ids_ = train_row_ids
        self.source_families_ = source_families
        self.selected_alpha_ = self.precommitted_alpha
        self.regularization_strategy_ = "precommitted_single_alpha"
        self.selected_cv_r_loss_ = None
        self.constant_effect_ = raw_constant
        self.weights_ = raw_weights
        self.standardized_constant_effect_ = float(coefficients[0])
        self.standardized_weights_ = standardized_weights
        self.source_means_ = source_means
        self.source_scales_ = source_scales
        self.source_weights_ = {
            family: float(weight)
            for family, weight in zip(self.source_families_, self.weights_)
        }
        self.cv_results_ = []
        self.training_r_loss_ = _r_loss(
            outcome_residual,
            treatment_residual,
            self.standardized_constant_effect_
            + standardized_signal_matrix @ self.standardized_weights_,
        )
        return self

    def predict(
        self,
        *,
        row_ids: Sequence[Hashable],
        signals: Sequence[SignalBundle],
    ) -> np.ndarray:
        """Predict aligned outer-heldout effects, rejecting meta-fit overlap."""

        self._require_fitted()
        requested = _key_tuple(row_ids, name="row_ids")
        overlap = set(requested) & set(self.training_row_ids_)
        if overlap:
            raise ValueError(
                "Fold-honesty violation: prediction rows overlap R-stack fit rows: "
                f"{list(overlap)[:3]}"
            )
        matrix, _families, _by_family = _aligned_signal_matrix(
            signals,
            requested,
            required_scope=OUTER_HELDOUT_SCOPE,
            expected_source_families=self.source_families_,
        )
        _validate_outer_provenance(signals, requested)
        standardized = (matrix - self.source_means_) / self.source_scales_
        return np.asarray(
            self.standardized_constant_effect_ + standardized @ self.standardized_weights_,
            dtype=float,
        )

    def predict_bundle(
        self,
        *,
        row_ids: Sequence[Hashable],
        signals: Sequence[SignalBundle],
        source_family: str = "regularized_r_stack",
    ) -> SignalBundle:
        """Predict and preserve the stack plus upstream recursive lineage."""

        requested = _key_tuple(row_ids, name="row_ids")
        predictions = self.predict(row_ids=requested, signals=signals)
        _matrix, _families, by_family = _aligned_signal_matrix(
            signals,
            requested,
            required_scope=OUTER_HELDOUT_SCOPE,
            expected_source_families=self.source_families_,
        )
        upstream_by_family = {
            family: by_family[family].aligned_provenance(requested)
            for family in self.source_families_
        }
        provenance = tuple(
            FitRowProvenance(
                fit_row_ids=frozenset(self.training_row_ids_),
                upstream=tuple(
                    upstream_by_family[family][row_index]
                    for family in self.source_families_
                ),
            )
            for row_index, _row_id in enumerate(requested)
        )
        return SignalBundle(
            row_ids=requested,
            source_family=source_family,
            tau_predictions=predictions,
            prediction_scope=OUTER_HELDOUT_SCOPE,
            fit_row_provenance=provenance,
        )

    def _require_fitted(self) -> None:
        required = (
            "training_row_ids_",
            "source_families_",
            "selected_alpha_",
            "constant_effect_",
            "weights_",
            "source_means_",
            "source_scales_",
        )
        if not all(hasattr(self, name) for name in required):
            raise RuntimeError("FoldHonestRStack must be fit before prediction")


def _aligned_signal_matrix(
    signals: Sequence[SignalBundle],
    row_ids: tuple[Hashable, ...],
    *,
    required_scope: str,
    expected_source_families: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, SignalBundle]]:
    bundles = tuple(signals)
    if not bundles:
        raise ValueError("At least one treatment-effect signal is required")
    if not all(isinstance(bundle, SignalBundle) for bundle in bundles):
        raise TypeError("signals must contain only SignalBundle instances")
    by_family: dict[str, SignalBundle] = {}
    expected_rows = set(row_ids)
    for bundle in bundles:
        if bundle.source_family in by_family:
            raise ValueError(
                f"Duplicate signal source_family {bundle.source_family!r}"
            )
        if bundle.prediction_scope != required_scope:
            raise ValueError(
                f"Signal {bundle.source_family!r} has scope "
                f"{bundle.prediction_scope!r}; expected {required_scope!r}"
            )
        if set(bundle.row_ids) != expected_rows:
            raise ValueError(
                f"Signal {bundle.source_family!r} row IDs do not exactly match "
                "the requested rows"
            )
        by_family[bundle.source_family] = bundle

    if expected_source_families is None:
        source_families = tuple(by_family)
    else:
        source_families = tuple(str(value) for value in expected_source_families)
        if set(source_families) != set(by_family):
            missing = sorted(set(source_families) - set(by_family))
            unexpected = sorted(set(by_family) - set(source_families))
            raise ValueError(
                "Outer signal families do not match fitted families; "
                f"missing={missing} unexpected={unexpected}"
            )
    matrix = np.column_stack(
        [by_family[family].aligned_predictions(row_ids) for family in source_families]
    )
    return np.asarray(matrix, dtype=float), source_families, by_family


def _fit_r_coefficients(
    signals: np.ndarray,
    treatment_residual: np.ndarray,
    outcome_residual: np.ndarray,
    *,
    alpha: float,
    nonnegative: bool,
) -> np.ndarray:
    if signals.ndim != 2 or len(signals) != len(treatment_residual):
        raise ValueError("R-stack signal matrix and residual rows must align")
    if float(np.dot(treatment_residual, treatment_residual)) <= np.finfo(float).eps:
        raise ValueError("Treatment residuals contain no information in a fit fold")
    design = np.column_stack(
        [
            treatment_residual,
            treatment_residual[:, None] * signals,
        ]
    )
    n_signals = signals.shape[1]
    if alpha > 0.0:
        penalty = np.zeros((n_signals, n_signals + 1), dtype=float)
        penalty[:, 1:] = np.sqrt(float(alpha)) * np.eye(n_signals)
        design = np.vstack([design, penalty])
        target = np.concatenate([outcome_residual, np.zeros(n_signals, dtype=float)])
    else:
        target = outcome_residual

    if nonnegative:
        lower = np.concatenate([[-np.inf], np.zeros(n_signals, dtype=float)])
        upper = np.full(n_signals + 1, np.inf, dtype=float)
        result = lsq_linear(design, target, bounds=(lower, upper), lsmr_tol="auto")
        if not result.success:
            raise RuntimeError(f"Non-negative R-stack optimization failed: {result.message}")
        return np.asarray(result.x, dtype=float)
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )
    return np.asarray(coefficients, dtype=float)


def _validate_inner_fold_provenance(
    signals: Sequence[SignalBundle],
    row_ids: tuple[Hashable, ...],
    fold_ids: tuple[Hashable, ...],
) -> None:
    rows_by_fold: dict[Hashable, set[Hashable]] = {}
    for row_id, fold_id in zip(row_ids, fold_ids):
        rows_by_fold.setdefault(fold_id, set()).add(row_id)
    for bundle in signals:
        aligned_provenance = bundle.aligned_provenance(row_ids)
        recursive_rows_by_lineage: dict[int, frozenset[Hashable]] = {}
        for row_id, fold_id, lineage in zip(row_ids, fold_ids, aligned_provenance):
            lineage_id = id(lineage)
            recursive_rows = recursive_rows_by_lineage.get(lineage_id)
            if recursive_rows is None:
                recursive_rows = lineage.recursive_fit_row_ids()
                recursive_rows_by_lineage[lineage_id] = recursive_rows
            overlap = recursive_rows & rows_by_fold[fold_id]
            if overlap:
                raise ValueError(
                    "Fold-honesty violation for source "
                    f"{bundle.source_family!r}: prediction row {row_id!r} has "
                    "recursive fit provenance overlapping its supplied inner "
                    f"heldout fold: {list(overlap)[:3]}"
                )


def _validate_outer_provenance(
    signals: Sequence[SignalBundle],
    row_ids: tuple[Hashable, ...],
) -> None:
    heldout_rows = set(row_ids)
    for bundle in signals:
        aligned_provenance = bundle.aligned_provenance(row_ids)
        recursive_rows_by_lineage: dict[int, frozenset[Hashable]] = {}
        for row_id, lineage in zip(row_ids, aligned_provenance):
            lineage_id = id(lineage)
            recursive_rows = recursive_rows_by_lineage.get(lineage_id)
            if recursive_rows is None:
                recursive_rows = lineage.recursive_fit_row_ids()
                recursive_rows_by_lineage[lineage_id] = recursive_rows
            overlap = recursive_rows & heldout_rows
            if overlap:
                raise ValueError(
                    "Fold-honesty violation for source "
                    f"{bundle.source_family!r}: prediction row {row_id!r} has "
                    "recursive fit provenance overlapping the outer-heldout "
                    f"rows: {list(overlap)[:3]}"
                )


def _r_loss(
    outcome_residual: np.ndarray,
    treatment_residual: np.ndarray,
    tau: np.ndarray,
) -> float:
    residual = outcome_residual - treatment_residual * tau
    return float(np.mean(np.square(residual)))


__all__ = [
    "FitRowProvenance",
    "FoldHonestRStack",
    "INNER_OOF_SCOPE",
    "OUTER_HELDOUT_SCOPE",
    "SignalBundle",
]
