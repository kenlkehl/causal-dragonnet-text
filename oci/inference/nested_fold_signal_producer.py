"""Strict nested-fold producers for numerical treatment-effect signals.

The generic orchestrator in this module is deliberately label-blind for the
outer-heldout rows: its public API accepts only heldout row IDs and text.  For
each supplied meta-inner fold it:

1. fits nuisance models on the complementary meta-fit rows and predicts the
   entire meta-heldout fold;
2. asks every signal backend to fit on that same meta-fit partition; and
3. requires the backend's recursive lineage to remain inside the meta-fit
   rows before accepting its meta-heldout predictions.

The BoW weighted-R backend implements the additional nesting needed by an
R-learner.  It creates nuisance OOF predictions through inner-inner folds
entirely inside the meta-fit partition, constructs the weighted R target,
fits a text model on the meta-fit rows, and only then predicts the meta-heldout
rows.  For outer-heldout prediction it repeats that procedure on the full
outer train.  This is stronger than reusing historical Stage-1 OOF matrices,
whose independently generated nuisance and effect folds cannot prove this
recursive exclusion property.
"""

from __future__ import annotations

import hashlib
import base64
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

from .fold_honest_r_stack import (
    INNER_OOF_SCOPE,
    OUTER_HELDOUT_SCOPE,
    FitRowProvenance,
    SignalBundle,
)
from .fold_honest_signal_fusion import (
    AuthenticatedMaterialFile,
    BACKEND_CODE_MATERIAL,
    BACKEND_CONFIG_MATERIAL,
    BOW_R_LOSS,
    CALIBRATED_TAU_ROLE,
    CrossFittedNuisance,
    CrossFittedVector,
    FoldLocalSignal,
    INPUT_MATERIAL,
    MODEL_PROJECTION_MATERIAL,
    NumericalSignalProducerAudit,
    PRODUCER_CODE_MATERIAL,
    PRODUCER_CONFIG_MATERIAL,
    SUPPORTED_SIGNAL_KINDS,
    WrittenFoldNumericalSignalArtifact,
    write_fold_numerical_signal_artifact,
)

NESTED_SIGNAL_PRODUCER_ID = "strict_nested_fold_signal_producer_v1"


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _array_bytes_payload(values: Any) -> Mapping[str, Any]:
    array = np.ascontiguousarray(np.asarray(values))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "order": "C",
        "bytes_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _projection_bundle(**parts: bytes) -> bytes:
    return _canonical_json_bytes(
        {
            name: {
                "sha256": hashlib.sha256(value).hexdigest(),
                "bytes_base64": base64.b64encode(value).decode("ascii"),
            }
            for name, value in sorted(parts.items())
        }
    )


def _write_immutable_bytes(path: Path, payload: bytes, *, label: str) -> Path:
    requested = path.resolve()
    if requested.exists():
        if requested.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite different {label}: {requested}")
    else:
        if not requested.parent.exists():
            raise FileNotFoundError(f"{label} parent does not exist: {requested.parent}")
        with requested.open("xb") as handle:
            handle.write(payload)
    return requested


def _canonical_row_ids(values: Sequence[Any], *, name: str) -> tuple[int, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    if not raw:
        raise ValueError(f"{name} must be non-empty")
    result: list[int] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must contain canonical integer row IDs")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} cannot contain negative row IDs")
        result.append(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique row IDs")
    return tuple(result)


def _finite_vector(values: Sequence[Any], *, name: str, length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) != int(length):
        raise ValueError(f"{name} must be one-dimensional with length {length}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    vector = vector.copy()
    vector.setflags(write=False)
    return vector


def _normalized_texts(values: Sequence[Any], *, name: str, length: int) -> tuple[str, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc
    if len(raw) != int(length):
        raise ValueError(f"{name} must have length {length}")
    return tuple(str(value or "").strip().lower() for value in raw)


@dataclass(frozen=True)
class FoldTrainingRows:
    row_ids: tuple[int, ...]
    texts: tuple[str, ...]
    treatment: np.ndarray
    outcome: np.ndarray
    outcome_type: str = "binary"

    def __post_init__(self) -> None:
        rows = _canonical_row_ids(self.row_ids, name="training.row_ids")
        texts = _normalized_texts(self.texts, name="training.texts", length=len(rows))
        treatment = _finite_vector(self.treatment, name="training.treatment", length=len(rows))
        if not np.isin(treatment, [0.0, 1.0]).all():
            raise ValueError("training.treatment must be binary")
        outcome = _finite_vector(self.outcome, name="training.outcome", length=len(rows))
        outcome_type = str(self.outcome_type).strip().lower()
        if outcome_type not in {"binary", "continuous"}:
            raise ValueError("training.outcome_type must be 'binary' or 'continuous'")
        if outcome_type == "binary" and not np.isin(outcome, [0.0, 1.0]).all():
            raise ValueError("training.outcome must be binary")
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "texts", texts)
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "outcome_type", outcome_type)

    def subset(self, positions: Sequence[int]) -> "FoldTrainingRows":
        selected = np.asarray(positions, dtype=int)
        if selected.ndim != 1 or not len(selected):
            raise ValueError("training subset positions must be a non-empty vector")
        if np.any(selected < 0) or np.any(selected >= len(self.row_ids)):
            raise ValueError("training subset positions are out of range")
        return FoldTrainingRows(
            row_ids=tuple(self.row_ids[int(pos)] for pos in selected),
            texts=tuple(self.texts[int(pos)] for pos in selected),
            treatment=self.treatment[selected],
            outcome=self.outcome[selected],
            outcome_type=self.outcome_type,
        )


@dataclass(frozen=True)
class FoldPredictionRows:
    row_ids: tuple[int, ...]
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        rows = _canonical_row_ids(self.row_ids, name="prediction.row_ids")
        texts = _normalized_texts(self.texts, name="prediction.texts", length=len(rows))
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "texts", texts)

    @classmethod
    def from_training_subset(
        cls, training: FoldTrainingRows, positions: Sequence[int]
    ) -> "FoldPredictionRows":
        selected = np.asarray(positions, dtype=int)
        if selected.ndim != 1 or not len(selected):
            raise ValueError("prediction subset positions must be a non-empty vector")
        return cls(
            row_ids=tuple(training.row_ids[int(pos)] for pos in selected),
            texts=tuple(training.texts[int(pos)] for pos in selected),
        )


@dataclass(frozen=True)
class NuisanceFoldPrediction:
    propensity: np.ndarray
    outcome_prediction: np.ndarray
    propensity_provenance: FitRowProvenance
    outcome_provenance: FitRowProvenance
    model_projection_bytes: bytes = b""


@dataclass(frozen=True)
class SignalFoldPrediction:
    values: np.ndarray
    provenance: FitRowProvenance
    model_projection_bytes: bytes = b""


class NestedNuisanceBackend(Protocol):
    def identity(self) -> Mapping[str, Any]: ...

    def fit_predict(
        self,
        fit_rows: FoldTrainingRows,
        prediction_rows: FoldPredictionRows,
        *,
        random_state: int,
    ) -> NuisanceFoldPrediction: ...


class NestedEffectSignalBackend(Protocol):
    signal_name: str
    source_kind: str

    def identity(self) -> Mapping[str, Any]: ...

    def fit_predict(
        self,
        fit_rows: FoldTrainingRows,
        prediction_rows: FoldPredictionRows,
        *,
        nuisance_backend: NestedNuisanceBackend,
        inner_inner_folds: int,
        random_state: int,
    ) -> SignalFoldPrediction: ...


@dataclass(frozen=True)
class NestedBoWSignalConfig:
    outcome_type: str = "binary"
    inner_inner_folds: int = 3
    ngram_range_min: int = 1
    ngram_range_max: int = 2
    min_df: int = 2
    max_df: float = 1.0
    max_features: int = 20000
    sublinear_tf: bool = True
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    nuisance_ridge_alpha: float = 1.0
    effect_ridge_alpha: float = 1.0
    propensity_clip: float = 0.02
    random_state: int = 42
    signal_name: str = "nested_bow_weighted_r"

    def __post_init__(self) -> None:
        outcome_type = str(self.outcome_type).strip().lower()
        if outcome_type not in {"binary", "continuous"}:
            raise ValueError("outcome_type must be 'binary' or 'continuous'")
        if int(self.inner_inner_folds) < 2:
            raise ValueError("inner_inner_folds must be at least two")
        if int(self.ngram_range_min) < 1 or int(self.ngram_range_max) < int(self.ngram_range_min):
            raise ValueError("invalid ngram range")
        if int(self.min_df) < 1 or int(self.max_features) < 1:
            raise ValueError("min_df and max_features must be positive")
        if not 0.0 < float(self.max_df) <= 1.0:
            raise ValueError("max_df must be in (0, 1]")
        for name in (
            "logistic_c",
            "nuisance_ridge_alpha",
            "effect_ridge_alpha",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        clip = float(self.propensity_clip)
        if not 0.0 < clip < 0.5:
            raise ValueError("propensity_clip must be in (0, 0.5)")
        signal_name = str(self.signal_name).strip()
        if not signal_name:
            raise ValueError("signal_name must be non-empty")
        object.__setattr__(self, "outcome_type", outcome_type)
        object.__setattr__(self, "signal_name", signal_name)

    def content_sha256(self) -> str:
        return _stable_sha256(asdict(self))


class BoWNuisanceBackend:
    def __init__(self, config: NestedBoWSignalConfig) -> None:
        self.config = config

    def identity(self) -> Mapping[str, Any]:
        return {
            "backend": "bow_nuisance_v1",
            "config": asdict(self.config),
        }

    def fit_predict(
        self,
        fit_rows: FoldTrainingRows,
        prediction_rows: FoldPredictionRows,
        *,
        random_state: int,
    ) -> NuisanceFoldPrediction:
        if set(fit_rows.row_ids) & set(prediction_rows.row_ids):
            raise ValueError("nuisance fit and prediction rows must be disjoint")
        if fit_rows.outcome_type != self.config.outcome_type:
            raise ValueError("nuisance backend outcome_type does not match training rows")
        x_fit, x_prediction, vectorizer_projection = _fit_text_matrix(
            fit_rows.texts,
            prediction_rows.texts,
            config=self.config,
        )
        propensity, propensity_projection = _predict_binary(
            x_fit,
            fit_rows.treatment,
            x_prediction,
            c=self.config.logistic_c,
            max_iter=self.config.logistic_max_iter,
            random_state=random_state,
        )
        propensity = np.clip(
            propensity,
            self.config.propensity_clip,
            1.0 - self.config.propensity_clip,
        )
        if self.config.outcome_type == "binary":
            outcome_prediction, outcome_projection = _predict_binary(
                x_fit,
                fit_rows.outcome,
                x_prediction,
                c=self.config.logistic_c,
                max_iter=self.config.logistic_max_iter,
                random_state=random_state + 1,
            )
        else:
            outcome_prediction, outcome_projection = _predict_ridge(
                x_fit,
                fit_rows.outcome,
                x_prediction,
                alpha=self.config.nuisance_ridge_alpha,
            )
        lineage = FitRowProvenance(fit_row_ids=frozenset(fit_rows.row_ids))
        return NuisanceFoldPrediction(
            propensity=np.asarray(propensity, dtype=float),
            outcome_prediction=np.asarray(outcome_prediction, dtype=float),
            propensity_provenance=lineage,
            outcome_provenance=lineage,
            model_projection_bytes=_projection_bundle(
                vectorizer=vectorizer_projection,
                propensity_model=propensity_projection,
                outcome_model=outcome_projection,
            ),
        )


class BoWWeightedRSignalBackend:
    source_kind = BOW_R_LOSS

    def __init__(self, config: NestedBoWSignalConfig) -> None:
        self.config = config
        self.signal_name = config.signal_name

    def identity(self) -> Mapping[str, Any]:
        return {
            "backend": "nested_bow_weighted_r_v1",
            "config": asdict(self.config),
        }

    def fit_predict(
        self,
        fit_rows: FoldTrainingRows,
        prediction_rows: FoldPredictionRows,
        *,
        nuisance_backend: NestedNuisanceBackend,
        inner_inner_folds: int,
        random_state: int,
    ) -> SignalFoldPrediction:
        if set(fit_rows.row_ids) & set(prediction_rows.row_ids):
            raise ValueError("signal fit and prediction rows must be disjoint")
        if fit_rows.outcome_type != self.config.outcome_type:
            raise ValueError("signal backend outcome_type does not match training rows")
        nuisance_e, nuisance_m, nuisance_lineages, nuisance_projections = (
            _inner_inner_nuisance_oof(
            fit_rows,
            nuisance_backend=nuisance_backend,
            requested_folds=inner_inner_folds,
            random_state=random_state + 10_000,
            )
        )
        treatment_residual = fit_rows.treatment - np.clip(
            nuisance_e,
            self.config.propensity_clip,
            1.0 - self.config.propensity_clip,
        )
        outcome_residual = fit_rows.outcome - nuisance_m
        if np.any(np.abs(treatment_residual) < self.config.propensity_clip * 0.99):
            raise RuntimeError("clipped propensity produced a degenerate R residual")
        pseudo_target = outcome_residual / treatment_residual
        sample_weight = np.square(treatment_residual)
        x_fit, x_prediction, vectorizer_projection = _fit_text_matrix(
            fit_rows.texts,
            prediction_rows.texts,
            config=self.config,
        )
        values, effect_projection = _predict_ridge(
            x_fit,
            pseudo_target,
            x_prediction,
            alpha=self.config.effect_ridge_alpha,
            sample_weight=sample_weight,
        )
        lineage = FitRowProvenance(
            fit_row_ids=frozenset(fit_rows.row_ids),
            upstream=tuple(nuisance_lineages),
        )
        return SignalFoldPrediction(
            values=np.asarray(values, dtype=float),
            provenance=lineage,
            model_projection_bytes=_projection_bundle(
                vectorizer=vectorizer_projection,
                effect_model=effect_projection,
                inner_nuisance=_canonical_json_bytes(
                    [
                        {
                            "sha256": hashlib.sha256(value).hexdigest(),
                            "bytes_base64": base64.b64encode(value).decode("ascii"),
                        }
                        for value in nuisance_projections
                    ]
                ),
            ),
        )


def _fit_text_matrix(
    fit_texts: Sequence[str],
    prediction_texts: Sequence[str],
    *,
    config: NestedBoWSignalConfig,
) -> tuple[Any | None, Any | None, bytes]:
    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)[a-z0-9%<>+=-]+",
        ngram_range=(int(config.ngram_range_min), int(config.ngram_range_max)),
        min_df=int(config.min_df),
        max_df=float(config.max_df),
        sublinear_tf=bool(config.sublinear_tf),
        max_features=int(config.max_features),
        dtype=np.float64,
    )
    try:
        x_fit = vectorizer.fit_transform(fit_texts)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower() and "no terms remain" not in str(exc).lower():
            raise
        empty_fit = sparse.csr_matrix((len(fit_texts), 0), dtype=np.float64)
        empty_prediction = sparse.csr_matrix(
            (len(prediction_texts), 0), dtype=np.float64
        )
        projection = _canonical_json_bytes(
            {
                "type": "tfidf_vectorizer",
                "empty_vocabulary": True,
                "config": asdict(config),
                "vocabulary": [],
                "idf": _array_bytes_payload(np.asarray([], dtype=np.float64)),
            }
        )
        return empty_fit, empty_prediction, projection
    projection = _canonical_json_bytes(
        {
            "type": "tfidf_vectorizer",
            "empty_vocabulary": False,
            "config": asdict(config),
            "vocabulary": sorted(
                ((str(token), int(index)) for token, index in vectorizer.vocabulary_.items()),
                key=lambda value: (value[1], value[0]),
            ),
            "idf": _array_bytes_payload(vectorizer.idf_),
        }
    )
    return x_fit, vectorizer.transform(prediction_texts), projection


def _predict_binary(
    x_fit: Any | None,
    fit_values: np.ndarray,
    x_prediction: Any | None,
    *,
    c: float,
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, bytes]:
    values = np.asarray(fit_values, dtype=float)
    prediction_count = int(x_prediction.shape[0])
    if int(x_fit.shape[1]) == 0 or len(np.unique(values)) < 2:
        constant = float(np.mean(values))
        return (
            np.full(prediction_count, constant, dtype=float),
            _canonical_json_bytes(
                {
                    "type": "constant_binary",
                    "constant": constant,
                    "c": float(c),
                    "max_iter": int(max_iter),
                    "random_state": int(random_state),
                }
            ),
        )
    model = LogisticRegression(
        C=float(c),
        solver="liblinear",
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    model.fit(x_fit, values.astype(int))
    predictions = np.asarray(model.predict_proba(x_prediction)[:, 1], dtype=float)
    projection = _canonical_json_bytes(
        {
            "type": "logistic_regression",
            "c": float(c),
            "solver": "liblinear",
            "max_iter": int(max_iter),
            "random_state": int(random_state),
            "classes": _array_bytes_payload(model.classes_),
            "coef": _array_bytes_payload(model.coef_),
            "intercept": _array_bytes_payload(model.intercept_),
            "n_iter": _array_bytes_payload(model.n_iter_),
        }
    )
    return predictions, projection


def _predict_ridge(
    x_fit: Any | None,
    fit_values: np.ndarray,
    x_prediction: Any | None,
    *,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, bytes]:
    if int(x_fit.shape[1]) == 0:
        values = np.asarray(fit_values, dtype=float)
        if sample_weight is None or float(np.sum(sample_weight)) <= 0.0:
            constant = float(np.mean(values))
        else:
            constant = float(np.average(values, weights=np.asarray(sample_weight)))
        return (
            np.full(int(x_prediction.shape[0]), constant, dtype=float),
            _canonical_json_bytes(
                {
                    "type": "constant_ridge",
                    "constant": constant,
                    "alpha": float(alpha),
                }
            ),
        )
    model = Ridge(alpha=float(alpha), solver="lsqr")
    if sample_weight is None:
        model.fit(x_fit, fit_values)
    else:
        model.fit(x_fit, fit_values, sample_weight=np.asarray(sample_weight, dtype=float))
    predictions = np.asarray(model.predict(x_prediction), dtype=float)
    projection = _canonical_json_bytes(
        {
            "type": "ridge",
            "alpha": float(alpha),
            "solver": "lsqr",
            "coef": _array_bytes_payload(model.coef_),
            "intercept": _array_bytes_payload(np.asarray(model.intercept_)),
        }
    )
    return predictions, projection


def _inner_inner_nuisance_oof(
    rows: FoldTrainingRows,
    *,
    nuisance_backend: NestedNuisanceBackend,
    requested_folds: int,
    random_state: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[FitRowProvenance, ...],
    tuple[bytes, ...],
]:
    folds = min(int(requested_folds), len(rows.row_ids))
    if folds < 2:
        raise ValueError("inner-inner nuisance OOF requires at least two folds")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=int(random_state))
    propensity = np.full(len(rows.row_ids), np.nan, dtype=float)
    outcome_prediction = np.full(len(rows.row_ids), np.nan, dtype=float)
    lineages: list[FitRowProvenance] = []
    projections: list[bytes] = []
    for fold_index, (fit_pos, heldout_pos) in enumerate(
        splitter.split(np.arange(len(rows.row_ids))), start=1
    ):
        fit_pos = np.asarray(fit_pos, dtype=int)
        heldout_pos = np.asarray(heldout_pos, dtype=int)
        fit = rows.subset(fit_pos)
        heldout = FoldPredictionRows.from_training_subset(rows, heldout_pos)
        prediction = nuisance_backend.fit_predict(
            fit,
            heldout,
            random_state=int(random_state) + fold_index,
        )
        _validate_nuisance_prediction(prediction, prediction_rows=heldout, fit_rows=fit)
        propensity[heldout_pos] = prediction.propensity
        outcome_prediction[heldout_pos] = prediction.outcome_prediction
        lineages.extend([prediction.propensity_provenance, prediction.outcome_provenance])
        projections.append(prediction.model_projection_bytes)
    if not np.isfinite(propensity).all() or not np.isfinite(outcome_prediction).all():
        raise RuntimeError("inner-inner nuisance OOF predictions are incomplete")
    # Preserve one copy of each exact fold lineage in deterministic order.
    unique: list[FitRowProvenance] = []
    seen: set[int] = set()
    for lineage in lineages:
        if id(lineage) not in seen:
            seen.add(id(lineage))
            unique.append(lineage)
    return propensity, outcome_prediction, tuple(unique), tuple(projections)


_AUTHENTICATED_BOW_NUISANCE_FIT_PREDICT = BoWNuisanceBackend.fit_predict
_AUTHENTICATED_BOW_EFFECT_FIT_PREDICT = BoWWeightedRSignalBackend.fit_predict


class NestedFoldSignalOrchestrator:
    """Generic strict nested-fold producer for one or more signal backends.

    The emitted train OOF values are honest for fitting the final outer-fold
    estimator.  They are not a per-gate nested diagnostic bank: meta-fit OOF
    values may depend on a subsequently selected gate.  ``identity()`` binds
    that limitation explicitly so consumers cannot interpret the current schema as
    supporting adaptive untouched-gate source scoring.
    """

    def __init__(
        self,
        *,
        nuisance_backend: NestedNuisanceBackend,
        signal_backends: Sequence[NestedEffectSignalBackend],
        inner_inner_folds: int,
        random_state: int = 42,
    ) -> None:
        if type(nuisance_backend) is not BoWNuisanceBackend:
            raise TypeError(
                "unauthenticated generic nuisance backends are forbidden; use the "
                "allowlisted concrete BoWNuisanceBackend producer"
            )
        if not signal_backends or any(
            type(backend) is not BoWWeightedRSignalBackend for backend in signal_backends
        ):
            raise TypeError(
                "unauthenticated generic signal backends are forbidden; use allowlisted "
                "concrete producers with exact material capture"
            )
        self.nuisance_backend = nuisance_backend
        self.signal_backends = tuple(signal_backends)
        self.inner_inner_folds = int(inner_inner_folds)
        self.random_state = int(random_state)
        if self.inner_inner_folds < 2:
            raise ValueError("inner_inner_folds must be at least two")
        if not self.signal_backends:
            raise ValueError("at least one signal backend is required")
        names = [str(backend.signal_name).strip() for backend in self.signal_backends]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("signal backend names must be non-empty and unique")
        kinds = [str(backend.source_kind).strip().lower() for backend in self.signal_backends]
        invalid = sorted(set(kinds) - SUPPORTED_SIGNAL_KINDS)
        if invalid:
            raise ValueError(f"unsupported signal backend kinds: {invalid}")
        self._assert_concrete_backends_pristine()

    def _assert_concrete_backends_pristine(self) -> None:
        if (
            BoWNuisanceBackend.fit_predict
            is not _AUTHENTICATED_BOW_NUISANCE_FIT_PREDICT
            or BoWWeightedRSignalBackend.fit_predict
            is not _AUTHENTICATED_BOW_EFFECT_FIT_PREDICT
        ):
            raise TypeError("allowlisted backend class implementation was modified at runtime")
        if type(self.nuisance_backend.config) is not NestedBoWSignalConfig:
            raise TypeError("nuisance backend config has an unauthenticated concrete type")
        nuisance_overrides = set(vars(self.nuisance_backend)) - {"config"}
        if nuisance_overrides:
            raise TypeError(
                "nuisance backend has unauthenticated instance overrides: "
                f"{sorted(nuisance_overrides)}"
            )
        for backend in self.signal_backends:
            if type(backend.config) is not NestedBoWSignalConfig:
                raise TypeError("signal backend config has an unauthenticated concrete type")
            overrides = set(vars(backend)) - {"config", "signal_name"}
            if overrides:
                raise TypeError(
                    "signal backend has unauthenticated instance overrides: "
                    f"{sorted(overrides)}"
                )
            if backend.signal_name != backend.config.signal_name:
                raise ValueError("signal backend declaration no longer matches exact config")

    def identity(self) -> Mapping[str, Any]:
        return {
            "schema": NESTED_SIGNAL_PRODUCER_ID,
            "adaptive_untouched_gate_diagnostic_views": False,
            "inner_inner_folds": self.inner_inner_folds,
            "random_state": self.random_state,
            "nuisance_backend": {
                "backend": "bow_nuisance_v1",
                "config": asdict(self.nuisance_backend.config),
            },
            "signal_backends": [
                {
                    "backend": "nested_bow_weighted_r_v1",
                    "config": asdict(backend.config),
                }
                for backend in self.signal_backends
            ],
        }

    def config_sha256(self) -> str:
        return hashlib.sha256(self.config_bytes()).hexdigest()

    def config_bytes(self) -> bytes:
        return _canonical_json_bytes(self.identity())

    def code_sha256(self) -> str:
        return _module_sha256()

    def producer_audit(
        self, *, input_artifact_paths: Mapping[str, Path | str]
    ) -> NumericalSignalProducerAudit:
        input_hashes = self._input_hashes(input_artifact_paths)
        return NumericalSignalProducerAudit(
            producer_id=NESTED_SIGNAL_PRODUCER_ID,
            producer_code_sha256=self.code_sha256(),
            producer_config_sha256=self.config_sha256(),
            input_artifact_sha256s=input_hashes,
            posthoc_targets_consumed=False,
            outer_heldout_labels_consumed=False,
            dataset_specific_truth_consumed=False,
        )

    @staticmethod
    def _input_hashes(
        input_artifact_paths: Mapping[str, Path | str],
    ) -> Mapping[str, str]:
        if not isinstance(input_artifact_paths, Mapping) or not input_artifact_paths:
            raise ValueError("input_artifact_paths must authenticate at least one real file")
        result: dict[str, str] = {}
        for raw_name, raw_path in input_artifact_paths.items():
            name = str(raw_name).strip()
            if not name or name in result:
                raise ValueError("input artifact names must be unique non-empty strings")
            path = Path(raw_path).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"input artifact is not a regular file: {path}")
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def produce_and_write(
        self,
        path: Path | str,
        *,
        outer_fold: int,
        split_fingerprint: str,
        outer_train: FoldTrainingRows,
        outer_heldout: FoldPredictionRows,
        inner_fold_ids: Sequence[Any],
        input_artifact_paths: Mapping[str, Path | str],
        producer_audit: NumericalSignalProducerAudit | None = None,
    ) -> WrittenFoldNumericalSignalArtifact:
        self._assert_concrete_backends_pristine()
        if set(outer_train.row_ids) & set(outer_heldout.row_ids):
            raise ValueError("outer train and heldout rows must be disjoint")
        if outer_train.outcome_type != "binary":
            raise ValueError(
                "authenticated numerical fusion producer requires binary outcomes"
            )
        if self.nuisance_backend.config.outcome_type != "binary" or any(
            backend.config.outcome_type != "binary" for backend in self.signal_backends
        ):
            raise ValueError(
                "authenticated numerical fusion backends must be configured for binary outcomes"
            )
        folds = _canonical_inner_folds(inner_fold_ids, length=len(outer_train.row_ids))
        external_input_hashes = self._input_hashes(input_artifact_paths)
        external_input_paths = {
            str(name).strip(): Path(input_artifact_paths[name]).resolve(strict=True)
            for name in input_artifact_paths
        }
        if len(external_input_paths) != len(input_artifact_paths):
            raise ValueError("input artifact names collide after normalization")
        if "fold_model_inputs" in external_input_paths:
            raise ValueError("input artifact name 'fold_model_inputs' is reserved")
        output_path = Path(path).resolve()
        model_input_path = _write_immutable_bytes(
            output_path.with_name(output_path.name + ".fold_model_inputs.json"),
            _canonical_json_bytes(
                {
                    "outer_fold": int(outer_fold),
                    "split_fingerprint": str(split_fingerprint),
                    "outer_train": {
                        "row_ids": list(outer_train.row_ids),
                        "texts": list(outer_train.texts),
                        "treatment": _array_bytes_payload(outer_train.treatment),
                        "outcome": _array_bytes_payload(outer_train.outcome),
                        "outcome_type": outer_train.outcome_type,
                        "inner_fold_ids": list(folds),
                    },
                    "outer_heldout": {
                        "row_ids": list(outer_heldout.row_ids),
                        "texts": list(outer_heldout.texts),
                    },
                }
            ),
            label="fold model input material",
        )
        all_input_paths = {
            **external_input_paths,
            "fold_model_inputs": model_input_path,
        }
        external_audit = NumericalSignalProducerAudit(
            producer_id=NESTED_SIGNAL_PRODUCER_ID,
            producer_code_sha256=self.code_sha256(),
            producer_config_sha256=self.config_sha256(),
            input_artifact_sha256s=external_input_hashes,
            posthoc_targets_consumed=False,
            outer_heldout_labels_consumed=False,
            dataset_specific_truth_consumed=False,
        )
        if producer_audit is not None and producer_audit != external_audit:
            raise ValueError(
                "producer audit is self-attested or stale; it must match exact code, "
                "config, and input bytes"
            )
        producer_audit = self.producer_audit(input_artifact_paths=all_input_paths)
        if producer_audit.producer_id != NESTED_SIGNAL_PRODUCER_ID:
            raise ValueError("producer audit identity does not match nested orchestrator")
        if producer_audit.producer_code_sha256 != self.code_sha256():
            raise ValueError("producer audit code hash does not match nested orchestrator")
        if producer_audit.producer_config_sha256 != self.config_sha256():
            raise ValueError("producer audit config hash does not match nested orchestrator")
        unique_folds = tuple(dict.fromkeys(folds))
        propensity = np.full(len(outer_train.row_ids), np.nan, dtype=float)
        outcome_prediction = np.full(len(outer_train.row_ids), np.nan, dtype=float)
        propensity_lineage: list[FitRowProvenance | None] = [None] * len(outer_train.row_ids)
        outcome_lineage: list[FitRowProvenance | None] = [None] * len(outer_train.row_ids)
        signal_values = {
            backend.signal_name: np.full(len(outer_train.row_ids), np.nan, dtype=float)
            for backend in self.signal_backends
        }
        signal_lineage = {
            backend.signal_name: [None] * len(outer_train.row_ids)
            for backend in self.signal_backends
        }
        projection_records: list[Mapping[str, Any]] = []

        for fold_index, fold_id in enumerate(unique_folds, start=1):
            heldout_pos = np.asarray(
                [index for index, value in enumerate(folds) if value == fold_id],
                dtype=int,
            )
            fit_pos = np.asarray(
                [index for index, value in enumerate(folds) if value != fold_id],
                dtype=int,
            )
            if not len(fit_pos) or not len(heldout_pos):
                raise ValueError(f"invalid meta-inner partition {fold_id!r}")
            fit_rows = outer_train.subset(fit_pos)
            prediction_rows = FoldPredictionRows.from_training_subset(outer_train, heldout_pos)
            nuisance = self.nuisance_backend.fit_predict(
                fit_rows,
                prediction_rows,
                random_state=self.random_state + 1_000 * fold_index,
            )
            _validate_nuisance_prediction(
                nuisance,
                prediction_rows=prediction_rows,
                fit_rows=fit_rows,
            )
            projection_records.append(
                {
                    "stage": "meta_inner_nuisance",
                    "fold": str(fold_id),
                    "random_state": self.random_state + 1_000 * fold_index,
                    "bytes_base64": base64.b64encode(
                        nuisance.model_projection_bytes
                    ).decode("ascii"),
                    "sha256": hashlib.sha256(
                        nuisance.model_projection_bytes
                    ).hexdigest(),
                }
            )
            propensity[heldout_pos] = nuisance.propensity
            outcome_prediction[heldout_pos] = nuisance.outcome_prediction
            for position in heldout_pos:
                propensity_lineage[int(position)] = nuisance.propensity_provenance
                outcome_lineage[int(position)] = nuisance.outcome_provenance

            for backend_index, backend in enumerate(self.signal_backends, start=1):
                result = backend.fit_predict(
                    fit_rows,
                    prediction_rows,
                    nuisance_backend=self.nuisance_backend,
                    inner_inner_folds=self.inner_inner_folds,
                    random_state=(self.random_state + 100_000 * backend_index + 1_000 * fold_index),
                )
                _validate_signal_prediction(
                    result,
                    prediction_rows=prediction_rows,
                    fit_rows=fit_rows,
                    signal_name=backend.signal_name,
                )
                projection_records.append(
                    {
                        "stage": "meta_inner_effect",
                        "fold": str(fold_id),
                        "signal_name": backend.signal_name,
                        "random_state": (
                            self.random_state
                            + 100_000 * backend_index
                            + 1_000 * fold_index
                        ),
                        "bytes_base64": base64.b64encode(
                            result.model_projection_bytes
                        ).decode("ascii"),
                        "sha256": hashlib.sha256(
                            result.model_projection_bytes
                        ).hexdigest(),
                    }
                )
                signal_values[backend.signal_name][heldout_pos] = result.values
                for position in heldout_pos:
                    signal_lineage[backend.signal_name][int(position)] = result.provenance

        if not np.isfinite(propensity).all() or not np.isfinite(outcome_prediction).all():
            raise RuntimeError("meta-inner nuisance predictions are incomplete")
        if any(item is None for item in propensity_lineage + outcome_lineage):
            raise RuntimeError("meta-inner nuisance lineage is incomplete")

        fold_signals: list[FoldLocalSignal] = []
        for backend_index, backend in enumerate(self.signal_backends, start=1):
            inner_values = signal_values[backend.signal_name]
            inner_lineage = signal_lineage[backend.signal_name]
            if not np.isfinite(inner_values).all() or any(item is None for item in inner_lineage):
                raise RuntimeError(f"meta-inner signal {backend.signal_name!r} is incomplete")
            outer_result = backend.fit_predict(
                outer_train,
                outer_heldout,
                nuisance_backend=self.nuisance_backend,
                inner_inner_folds=self.inner_inner_folds,
                random_state=self.random_state + 1_000_000 + 100_000 * backend_index,
            )
            _validate_signal_prediction(
                outer_result,
                prediction_rows=outer_heldout,
                fit_rows=outer_train,
                signal_name=backend.signal_name,
            )
            projection_records.append(
                {
                    "stage": "outer_effect",
                    "signal_name": backend.signal_name,
                    "random_state": self.random_state
                    + 1_000_000
                    + 100_000 * backend_index,
                    "bytes_base64": base64.b64encode(
                        outer_result.model_projection_bytes
                    ).decode("ascii"),
                    "sha256": hashlib.sha256(
                        outer_result.model_projection_bytes
                    ).hexdigest(),
                }
            )
            fold_signals.append(
                FoldLocalSignal(
                    signal_name=backend.signal_name,
                    source_kind=backend.source_kind,
                    signal_role=CALIBRATED_TAU_ROLE,
                    inner_oof=SignalBundle(
                        row_ids=outer_train.row_ids,
                        source_family=backend.signal_name,
                        tau_predictions=inner_values,
                        prediction_scope=INNER_OOF_SCOPE,
                        fit_row_provenance=tuple(inner_lineage),
                    ),
                    inner_fold_ids=folds,
                    outer_heldout=SignalBundle(
                        row_ids=outer_heldout.row_ids,
                        source_family=backend.signal_name,
                        tau_predictions=outer_result.values,
                        prediction_scope=OUTER_HELDOUT_SCOPE,
                        fit_row_provenance=tuple(
                            outer_result.provenance for _ in outer_heldout.row_ids
                        ),
                    ),
                )
            )

        nuisance = CrossFittedNuisance(
            propensity=CrossFittedVector(
                name="propensity",
                row_ids=outer_train.row_ids,
                values=propensity,
                inner_fold_ids=folds,
                fit_row_provenance=tuple(propensity_lineage),
            ),
            outcome_prediction=CrossFittedVector(
                name="outcome_prediction",
                row_ids=outer_train.row_ids,
                values=outcome_prediction,
                inner_fold_ids=folds,
                fit_row_provenance=tuple(outcome_lineage),
            ),
        )
        producer_config_path = _write_immutable_bytes(
            output_path.with_name(output_path.name + ".producer_config.json"),
            self.config_bytes(),
            label="producer config material",
        )
        backend_config_materials: list[AuthenticatedMaterialFile] = []
        nuisance_config_path = _write_immutable_bytes(
            output_path.with_name(output_path.name + ".backend_config.nuisance.json"),
            _canonical_json_bytes(
                {
                    "backend": "bow_nuisance_v1",
                    "config": asdict(self.nuisance_backend.config),
                }
            ),
            label="backend config material",
        )
        backend_config_materials.append(
            AuthenticatedMaterialFile(
                category=BACKEND_CONFIG_MATERIAL,
                name="bow_nuisance",
                path=nuisance_config_path,
            )
        )
        for backend_index, backend in enumerate(self.signal_backends, start=1):
            backend_config_path = _write_immutable_bytes(
                output_path.with_name(
                    output_path.name + f".backend_config.effect_{backend_index}.json"
                ),
                _canonical_json_bytes(
                    {
                        "backend": "nested_bow_weighted_r_v1",
                        "config": asdict(backend.config),
                    }
                ),
                label="backend config material",
            )
            backend_config_materials.append(
                AuthenticatedMaterialFile(
                    category=BACKEND_CONFIG_MATERIAL,
                    name=f"bow_weighted_r_{backend_index}",
                    path=backend_config_path,
                )
            )
        projection_path = _write_immutable_bytes(
            output_path.with_name(output_path.name + ".model_projection.json"),
            _canonical_json_bytes(projection_records),
            label="model projection material",
        )
        producer_code_path = Path(__file__).resolve(strict=True)
        materials: list[AuthenticatedMaterialFile] = [
            AuthenticatedMaterialFile(
                category=PRODUCER_CODE_MATERIAL,
                name="nested_fold_orchestrator",
                path=producer_code_path,
            ),
            AuthenticatedMaterialFile(
                category=PRODUCER_CONFIG_MATERIAL,
                name="nested_fold_orchestrator",
                path=producer_config_path,
            ),
            AuthenticatedMaterialFile(
                category=BACKEND_CODE_MATERIAL,
                name="bow_nuisance",
                path=Path(inspect.getsourcefile(type(self.nuisance_backend)) or "").resolve(
                    strict=True
                ),
            ),
            *(
                AuthenticatedMaterialFile(
                    category=BACKEND_CODE_MATERIAL,
                    name=f"bow_weighted_r_{index}",
                    path=Path(inspect.getsourcefile(type(backend)) or "").resolve(
                        strict=True
                    ),
                )
                for index, backend in enumerate(self.signal_backends, start=1)
            ),
            *backend_config_materials,
            AuthenticatedMaterialFile(
                category=MODEL_PROJECTION_MATERIAL,
                name="all_nested_fitted_models",
                path=projection_path,
            ),
            *(
                AuthenticatedMaterialFile(
                    category=INPUT_MATERIAL,
                    name=str(name),
                    path=input_path,
                )
                for name, input_path in all_input_paths.items()
            ),
        ]
        return write_fold_numerical_signal_artifact(
            path,
            outer_fold=outer_fold,
            split_fingerprint=split_fingerprint,
            outer_train_row_ids=outer_train.row_ids,
            outer_heldout_row_ids=outer_heldout.row_ids,
            producer_audit=producer_audit,
            nuisance=nuisance,
            signals=fold_signals,
            authenticated_materials=materials,
            random_seed=self.random_state,
            library_versions={
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__,
            },
        )


def _canonical_inner_folds(values: Sequence[Any], *, length: int) -> tuple[int | str, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError("inner_fold_ids must be a sequence") from exc
    if len(raw) != int(length):
        raise ValueError(f"inner_fold_ids must have length {length}")
    result: list[int | str] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, str)):
            raise TypeError("inner_fold_ids must contain integer or string IDs")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("inner fold string IDs must be non-empty")
        else:
            value = int(value)
            if value < 1:
                raise ValueError("inner fold integer IDs must be positive")
        result.append(value)
    if len(set(result)) < 2:
        raise ValueError("inner_fold_ids must contain at least two folds")
    return tuple(result)


def _validate_nuisance_prediction(
    prediction: NuisanceFoldPrediction,
    *,
    prediction_rows: FoldPredictionRows,
    fit_rows: FoldTrainingRows,
) -> None:
    if not isinstance(prediction, NuisanceFoldPrediction):
        raise TypeError("nuisance backend returned the wrong result type")
    propensity = _finite_vector(
        prediction.propensity,
        name="nuisance.propensity",
        length=len(prediction_rows.row_ids),
    )
    outcome_prediction = _finite_vector(
        prediction.outcome_prediction,
        name="nuisance.outcome_prediction",
        length=len(prediction_rows.row_ids),
    )
    if np.any(propensity <= 0.0) or np.any(propensity >= 1.0):
        raise ValueError("nuisance propensity must be strictly inside (0, 1)")
    if fit_rows.outcome_type == "binary" and (
        np.any(outcome_prediction < 0.0) or np.any(outcome_prediction > 1.0)
    ):
        raise ValueError("binary nuisance outcome prediction must be inside [0, 1]")
    if not isinstance(prediction.model_projection_bytes, bytes) or not prediction.model_projection_bytes:
        raise ValueError("nuisance backend must return exact non-empty model projection bytes")
    for name, lineage in (
        ("propensity", prediction.propensity_provenance),
        ("outcome_prediction", prediction.outcome_provenance),
    ):
        if not isinstance(lineage, FitRowProvenance):
            raise TypeError(f"{name} backend lineage must be FitRowProvenance")
        recursive = set(lineage.recursive_fit_row_ids())
        if not recursive:
            raise ValueError(f"{name} backend lineage is empty")
        if not recursive <= set(fit_rows.row_ids):
            raise ValueError(f"{name} backend lineage leaves the supplied fit partition")
        if recursive & set(prediction_rows.row_ids):
            raise ValueError(f"{name} backend lineage overlaps prediction rows")


def _validate_signal_prediction(
    prediction: SignalFoldPrediction,
    *,
    prediction_rows: FoldPredictionRows,
    fit_rows: FoldTrainingRows,
    signal_name: str,
) -> None:
    if not isinstance(prediction, SignalFoldPrediction):
        raise TypeError(f"Signal backend {signal_name!r} returned the wrong result type")
    _finite_vector(
        prediction.values,
        name=f"{signal_name}.values",
        length=len(prediction_rows.row_ids),
    )
    if not isinstance(prediction.provenance, FitRowProvenance):
        raise TypeError(f"Signal backend {signal_name!r} lineage must be FitRowProvenance")
    recursive = set(prediction.provenance.recursive_fit_row_ids())
    if not recursive:
        raise ValueError(f"Signal backend {signal_name!r} lineage is empty")
    if not recursive <= set(fit_rows.row_ids):
        raise ValueError(
            f"Signal backend {signal_name!r} lineage leaves the supplied fit partition"
        )
    if recursive & set(prediction_rows.row_ids):
        raise ValueError(f"Signal backend {signal_name!r} lineage overlaps prediction rows")
    if not isinstance(prediction.model_projection_bytes, bytes) or not prediction.model_projection_bytes:
        raise ValueError(
            f"Signal backend {signal_name!r} must return exact non-empty model projection bytes"
        )


def make_nested_bow_r_orchestrator(
    config: NestedBoWSignalConfig,
) -> NestedFoldSignalOrchestrator:
    nuisance = BoWNuisanceBackend(config)
    signal = BoWWeightedRSignalBackend(config)
    return NestedFoldSignalOrchestrator(
        nuisance_backend=nuisance,
        signal_backends=[signal],
        inner_inner_folds=config.inner_inner_folds,
        random_state=config.random_state,
    )


__all__ = [
    "BoWNuisanceBackend",
    "BoWWeightedRSignalBackend",
    "FoldPredictionRows",
    "FoldTrainingRows",
    "NESTED_SIGNAL_PRODUCER_ID",
    "NestedBoWSignalConfig",
    "NestedEffectSignalBackend",
    "NestedFoldSignalOrchestrator",
    "NestedNuisanceBackend",
    "NuisanceFoldPrediction",
    "SignalFoldPrediction",
    "make_nested_bow_r_orchestrator",
]
