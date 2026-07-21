"""Non-executable capture and replay for native Stage 1 BoW fits.

The production BoW runner historically retained predictions and feature
importance rows but discarded the fitted vectorizers and learners that
produced them.  This module provides an opt-in proof sink used only by the
production exact-scope wrapper.  It records numerical sklearn state in NPZ and
closed descriptive metadata in JSON; pickle and joblib are never written or
loaded.

Validation replays every inner-fold validation prediction, every registered
held-out transform, and every full-fit importance model from the captured
state.  Stored outputs alone are therefore insufficient to satisfy the proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

from .multi_model_agentic_forest import _make_bow_vectorizer

BOW_NATIVE_CAPTURE_SCHEMA = "production_bow_native_capture_v1"
BOW_NATIVE_CAPTURE_ARRAY_SCHEMA = "production_bow_native_capture_array_v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_KEY = re.compile(r"^[a-z0-9_]+$")
_FORBIDDEN_SUFFIXES = (
    ".joblib",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
)
_NUISANCE_OBJECTIVES = ("treatment_nuisance", "outcome_nuisance")
_EFFECT_OBJECTIVES = ("effect_pseudo_target", "effect_weighted_r")
_FULL_FIT_OBJECTIVES = (
    "treatment_importance",
    "outcome_importance",
    "effect_weighted_r_importance",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema_version": BOW_NATIVE_CAPTURE_ARRAY_SCHEMA,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        ).encode("utf-8")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _text_sha256(row_ids: Sequence[int], texts: Sequence[str]) -> str:
    rows = tuple(map(int, row_ids))
    values = tuple(str(text) for text in texts)
    if len(rows) != len(values):
        raise ValueError("text binding requires one text per row ID")
    digest = hashlib.sha256()
    digest.update(b"production-bow-text-binding-v1\0")
    for row_id, text in zip(rows, values):
        encoded = text.encode("utf-8")
        digest.update(int(row_id).to_bytes(8, byteorder="little", signed=False))
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _row_fingerprint(row_ids: Sequence[int]) -> str:
    rows = tuple(map(int, row_ids))
    if not rows or len(rows) != len(set(rows)) or any(row < 0 for row in rows):
        raise ValueError("row IDs must be unique non-negative integers")
    return _sha256_json({"ordered_row_ids": list(rows)})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable BoW artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable BoW artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_array(value: Any, *, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or (length is not None and len(array) != int(length)):
        raise ValueError(f"{name} must be one vector with the expected length")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _json_safe_view_config(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "name",
        "max_features",
        "min_df",
        "max_df",
        "ngram_range_min",
        "ngram_range_max",
        "sublinear_tf",
        "bow_model",
        "logistic_c",
        "logistic_max_iter",
        "ridge_alpha",
    }
    config = {str(key): value[key] for key in value if str(key) in allowed}
    if set(config) != allowed:
        raise ValueError("BoW proof view configuration is incomplete")
    return json.loads(_canonical_json(config))


class _ArrayStore:
    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.inventory: dict[str, dict[str, Any]] = {}

    def add(self, key: str, value: Any) -> str:
        key = str(key)
        if _SAFE_KEY.fullmatch(key) is None or key in self.arrays:
            raise ValueError(f"invalid or duplicate BoW capture array key: {key}")
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise ValueError("BoW capture arrays cannot use object dtype")
        self.arrays[key] = array
        self.inventory[key] = {
            "dtype": array.dtype.str,
            "shape": [int(value) for value in array.shape],
            "content_sha256": _array_sha256(array),
        }
        return key


def _capture_vectorizer(
    vectorizer: TfidfVectorizer,
    store: _ArrayStore,
    prefix: str,
    *,
    vectorizer_params: Mapping[str, Any],
) -> dict[str, Any]:
    if type(vectorizer) is not TfidfVectorizer:
        raise TypeError("BoW proof capture accepts only the native TfidfVectorizer")
    names = tuple(map(str, vectorizer.get_feature_names_out()))
    if not names or len(names) != len(set(names)):
        raise ValueError("fitted BoW vectorizer has invalid feature names")
    vocabulary = vectorizer.vocabulary_
    if set(vocabulary) != set(names) or any(
        int(vocabulary[name]) != index for index, name in enumerate(names)
    ):
        raise ValueError("fitted BoW vectorizer vocabulary is not index-stable")
    idf = np.asarray(vectorizer.idf_, dtype=np.float64)
    if idf.shape != (len(names),) or not np.isfinite(idf).all():
        raise ValueError("fitted BoW vectorizer has invalid IDF state")
    return {
        "kind": "sklearn_tfidf_vectorizer",
        "params": json.loads(_canonical_json(dict(vectorizer_params))),
        "feature_names": list(names),
        "idf_array": store.add(f"{prefix}_idf", idf),
        "feature_count": len(names),
        "fixed_vocabulary": False,
    }


def _capture_tree_ensemble(
    model: Any,
    store: _ArrayStore,
    prefix: str,
    *,
    classification: bool,
) -> dict[str, Any]:
    estimators = tuple(getattr(model, "estimators_", ()))
    if not estimators:
        raise ValueError("fitted forest has no estimators")
    trees: list[dict[str, Any]] = []
    for index, estimator in enumerate(estimators):
        tree = estimator.tree_
        tree_prefix = f"{prefix}_tree_{index:04d}"
        children_left = np.asarray(tree.children_left, dtype=np.int64)
        children_right = np.asarray(tree.children_right, dtype=np.int64)
        feature = np.asarray(tree.feature, dtype=np.int64)
        threshold = np.asarray(tree.threshold, dtype=np.float64)
        value = np.asarray(tree.value, dtype=np.float64)
        node_count = int(tree.node_count)
        if (
            children_left.shape != (node_count,)
            or children_right.shape != (node_count,)
            or feature.shape != (node_count,)
            or threshold.shape != (node_count,)
            or value.shape[0] != node_count
        ):
            raise ValueError("fitted forest tree state is malformed")
        trees.append(
            {
                "node_count": node_count,
                "children_left": store.add(f"{tree_prefix}_children_left", children_left),
                "children_right": store.add(
                    f"{tree_prefix}_children_right",
                    children_right,
                ),
                "feature": store.add(f"{tree_prefix}_feature", feature),
                "threshold": store.add(f"{tree_prefix}_threshold", threshold),
                "value": store.add(f"{tree_prefix}_value", value),
                "impurity": store.add(
                    f"{tree_prefix}_impurity",
                    np.asarray(tree.impurity, dtype=np.float64),
                ),
                "n_node_samples": store.add(
                    f"{tree_prefix}_n_node_samples",
                    np.asarray(tree.n_node_samples, dtype=np.int64),
                ),
                "weighted_n_node_samples": store.add(
                    f"{tree_prefix}_weighted_n_node_samples",
                    np.asarray(tree.weighted_n_node_samples, dtype=np.float64),
                ),
            }
        )
    state: dict[str, Any] = {
        "kind": (
            "sklearn_tree_ensemble_classifier"
            if classification
            else "sklearn_tree_ensemble_regressor"
        ),
        "class_name": type(model).__name__,
        "estimator_count": len(trees),
        "n_features_in": int(model.n_features_in_),
        "parameters": _safe_estimator_parameters(model),
        "trees": trees,
    }
    if classification:
        classes = np.asarray(model.classes_)
        if classes.ndim != 1 or len(classes) != 2:
            raise ValueError("BoW classifier proof requires exactly two classes")
        state["classes"] = classes.astype(float).tolist()
    return state


def _safe_estimator_parameters(model: Any) -> Mapping[str, Any]:
    try:
        params = model.get_params(deep=False)
    except Exception as exc:
        raise TypeError("captured BoW learner cannot expose its configuration") from exc
    if not isinstance(params, Mapping):
        raise TypeError("captured BoW learner parameters are not a mapping")
    try:
        return json.loads(_canonical_json(dict(params)))
    except (TypeError, ValueError) as exc:
        raise TypeError("captured BoW learner has non-JSON configuration") from exc


def _capture_learner(
    model: Any | None,
    store: _ArrayStore,
    prefix: str,
    *,
    classification: bool,
    constant_prediction: float | None = None,
) -> dict[str, Any]:
    if model is None:
        if constant_prediction is None or not np.isfinite(float(constant_prediction)):
            raise ValueError("constant BoW learner state requires one finite prediction")
        return {
            "kind": "constant_classifier" if classification else "constant_regressor",
            "constant_prediction": float(constant_prediction),
            "parameters": {"constant_prediction": float(constant_prediction)},
        }
    if type(model) is LogisticRegression and classification:
        classes = np.asarray(model.classes_)
        if classes.ndim != 1 or len(classes) != 2:
            raise ValueError("BoW logistic proof requires exactly two classes")
        return {
            "kind": "sklearn_logistic_regression",
            "class_name": type(model).__name__,
            "classes": classes.astype(float).tolist(),
            "n_features_in": int(model.n_features_in_),
            "parameters": _safe_estimator_parameters(model),
            "coef": store.add(f"{prefix}_coef", np.asarray(model.coef_, dtype=np.float64)),
            "intercept": store.add(
                f"{prefix}_intercept",
                np.asarray(model.intercept_, dtype=np.float64),
            ),
            "n_iter": store.add(f"{prefix}_n_iter", np.asarray(model.n_iter_, dtype=np.int64)),
        }
    if type(model) is Ridge and not classification:
        return {
            "kind": "sklearn_ridge",
            "class_name": type(model).__name__,
            "n_features_in": int(model.n_features_in_),
            "parameters": _safe_estimator_parameters(model),
            "coef": store.add(f"{prefix}_coef", np.asarray(model.coef_, dtype=np.float64)),
            "intercept": store.add(
                f"{prefix}_intercept",
                np.asarray(model.intercept_, dtype=np.float64),
            ),
        }
    classifier_types = (ExtraTreesClassifier, RandomForestClassifier)
    regressor_types = (ExtraTreesRegressor, RandomForestRegressor)
    if classification and type(model) in classifier_types:
        return _capture_tree_ensemble(model, store, prefix, classification=True)
    if not classification and type(model) in regressor_types:
        return _capture_tree_ensemble(model, store, prefix, classification=False)
    module = type(model).__module__
    if module.startswith("xgboost."):
        try:
            raw = bytes(model.get_booster().save_raw(raw_format="json"))
        except Exception as exc:  # pragma: no cover - optional dependency
            raise TypeError("XGBoost learner cannot emit safe JSON state") from exc
        if not raw.startswith(b"{"):
            raise ValueError("XGBoost safe capture did not emit JSON")
        json.loads(raw.decode("utf-8"))
        return {
            "kind": "xgboost_json_classifier" if classification else "xgboost_json_regressor",
            "class_name": type(model).__name__,
            "model_json": store.add(f"{prefix}_xgboost_json", np.frombuffer(raw, dtype=np.uint8)),
            "n_features_in": int(model.n_features_in_),
            "parameters": _safe_estimator_parameters(model),
        }
    raise TypeError(
        "unsupported BoW learner for safe non-executable capture: "
        f"{type(model).__module__}.{type(model).__name__}"
    )


def _restore_vectorizer(
    state: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> TfidfVectorizer:
    if state.get("kind") != "sklearn_tfidf_vectorizer":
        raise ValueError("BoW capture has an unsupported vectorizer state")
    params = state.get("params")
    names = state.get("feature_names")
    if not isinstance(params, Mapping) or not isinstance(names, list) or not names:
        raise ValueError("BoW vectorizer state is incomplete")
    names = [str(name) for name in names]
    if len(names) != len(set(names)) or int(state.get("feature_count", 0)) != len(names):
        raise ValueError("BoW vectorizer feature inventory is malformed")
    vectorizer = _make_bow_vectorizer(dict(params))
    vectorizer.vocabulary_ = {name: index for index, name in enumerate(names)}
    vectorizer.fixed_vocabulary_ = True
    # `_make_bow_vectorizer` fixes the native transform dtype (currently
    # float32).  Restore IDF state into that dtype so sparse normalization and
    # the downstream Ridge dot use the same arithmetic as the fitted pipeline.
    vectorizer_dtype = np.dtype(vectorizer.dtype)
    if vectorizer_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("BoW vectorizer has an unsupported arithmetic dtype")
    idf = np.asarray(
        arrays[str(state.get("idf_array"))], dtype=vectorizer_dtype
    )
    if idf.shape != (len(names),) or not np.isfinite(idf).all():
        raise ValueError("BoW vectorizer IDF state is malformed")
    vectorizer.idf_ = idf
    return vectorizer


def _replay_fit_transform(
    state: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    texts: Sequence[str],
) -> sparse.csr_matrix:
    """Recreate native fit-transform ordering for full-fit Ridge replay."""

    restored = _restore_vectorizer(state, arrays)
    params = state.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("BoW vectorizer state has no fitted parameters")
    fitted = _make_bow_vectorizer(dict(params))
    matrix = fitted.fit_transform(tuple(map(str, texts))).tocsr()
    if (
        list(map(str, fitted.get_feature_names_out())) != state.get("feature_names")
        or not np.array_equal(fitted.idf_, restored.idf_)
    ):
        raise RuntimeError("BoW full-fit vectorizer replay differs from captured state")
    return matrix


def _tree_leaf_indices(
    x: sparse.csr_matrix,
    *,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
) -> np.ndarray:
    matrix = sparse.csr_matrix(x, dtype=np.float32)
    leaves = np.empty(matrix.shape[0], dtype=np.int64)
    for row_index in range(matrix.shape[0]):
        row = matrix.getrow(row_index)
        node = 0
        while int(children_left[node]) != int(children_right[node]):
            feature_index = int(feature[node])
            position = int(np.searchsorted(row.indices, feature_index))
            value = (
                float(row.data[position])
                if position < len(row.indices) and int(row.indices[position]) == feature_index
                else 0.0
            )
            node = (
                int(children_left[node])
                if value <= float(threshold[node])
                else int(children_right[node])
            )
        leaves[row_index] = node
    return leaves


def _predict_learner(
    state: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    x: sparse.csr_matrix,
) -> np.ndarray:
    kind = str(state.get("kind") or "")
    n_rows = int(x.shape[0])
    if kind in {"constant_classifier", "constant_regressor"}:
        return np.full(n_rows, float(state["constant_prediction"]), dtype=np.float64)
    if kind == "sklearn_logistic_regression":
        coef = np.asarray(arrays[str(state.get("coef"))], dtype=np.float64)
        intercept = np.asarray(arrays[str(state.get("intercept"))], dtype=np.float64)
        if coef.shape != (1, x.shape[1]) or intercept.shape != (1,):
            raise ValueError("captured logistic state has incompatible dimensions")
        return expit(np.asarray(x @ coef[0], dtype=np.float64).reshape(-1) + intercept[0])
    if kind == "sklearn_ridge":
        # The native BoW vectorizer emits float32 sparse matrices, and sklearn
        # Ridge preserves that dtype for its fitted coefficients and prediction
        # arithmetic.  State is stored losslessly in float64, but replaying the
        # sparse dot in float64 can move a near-zero prediction by a few e-8.
        # Reconstruct the native arithmetic dtype before comparing outputs.
        calculation_dtype = np.dtype(x.dtype)
        if calculation_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError("captured Ridge input has an unsupported arithmetic dtype")
        coef = np.asarray(
            arrays[str(state.get("coef"))], dtype=calculation_dtype
        ).reshape(-1)
        intercept = np.asarray(
            arrays[str(state.get("intercept"))], dtype=calculation_dtype
        ).reshape(-1)
        if coef.shape != (x.shape[1],) or intercept.shape != (1,):
            raise ValueError("captured Ridge state has incompatible dimensions")
        prediction = (
            np.asarray(x @ coef, dtype=calculation_dtype).reshape(-1) + intercept[0]
        )
        return np.asarray(prediction, dtype=np.float64)
    if kind in {"sklearn_tree_ensemble_classifier", "sklearn_tree_ensemble_regressor"}:
        trees = state.get("trees")
        if not isinstance(trees, list) or len(trees) != int(state.get("estimator_count", 0)):
            raise ValueError("captured forest has incomplete trees")
        predictions = np.zeros(n_rows, dtype=np.float64)
        for tree in trees:
            if not isinstance(tree, Mapping):
                raise ValueError("captured forest tree metadata is malformed")
            leaves = _tree_leaf_indices(
                x,
                children_left=np.asarray(arrays[str(tree.get("children_left"))], dtype=np.int64),
                children_right=np.asarray(
                    arrays[str(tree.get("children_right"))],
                    dtype=np.int64,
                ),
                feature=np.asarray(arrays[str(tree.get("feature"))], dtype=np.int64),
                threshold=np.asarray(arrays[str(tree.get("threshold"))], dtype=np.float64),
            )
            value = np.asarray(arrays[str(tree.get("value"))], dtype=np.float64)
            leaf_value = value[leaves]
            if kind.endswith("classifier"):
                if leaf_value.ndim != 3 or leaf_value.shape[2] != 2:
                    raise ValueError("captured classifier tree has incompatible leaf values")
                counts = leaf_value[:, 0, :]
                totals = np.sum(counts, axis=1)
                if np.any(totals <= 0):
                    raise ValueError("captured classifier tree has empty leaves")
                predictions += counts[:, 1] / totals
            else:
                predictions += leaf_value.reshape(n_rows, -1)[:, 0]
        return predictions / float(len(trees))
    if kind in {"xgboost_json_classifier", "xgboost_json_regressor"}:
        try:  # pragma: no cover - optional dependency
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("XGBoost is required to replay this safe JSON state") from exc
        raw = np.asarray(arrays[str(state.get("model_json"))], dtype=np.uint8).tobytes()
        json.loads(raw.decode("utf-8"))
        booster = xgb.Booster()
        booster.load_model(bytearray(raw))
        return np.asarray(booster.predict(xgb.DMatrix(x)), dtype=np.float64)
    raise ValueError(f"unsupported captured BoW learner kind: {kind}")


class NativeBoWProofCaptureSink:
    """Collect actual native BoW fit state for one exact production scope."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        scope_id: str,
        outer_fold: int,
        inner_fold: int,
        fit_row_ids: Sequence[int],
        heldout_row_ids: Sequence[int],
        fit_texts: Sequence[str],
        heldout_texts: Sequence[str],
        text_column: str,
        outcome_type: str,
        e_clip: float,
        nuisance_folds: int,
        effect_folds: int,
        view_configs: Sequence[Mapping[str, Any]],
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        if self.artifact_dir.exists():
            raise RuntimeError("BoW native proof artifact directory must be new")
        self.scope_id = str(scope_id)
        self.outer_fold = int(outer_fold)
        self.inner_fold = int(inner_fold)
        if self.outer_fold < 1 or self.inner_fold < 1:
            raise ValueError("BoW proof capture requires a positive exact-inner scope")
        self.fit_row_ids = tuple(map(int, fit_row_ids))
        self.heldout_row_ids = tuple(map(int, heldout_row_ids))
        if set(self.fit_row_ids) & set(self.heldout_row_ids):
            raise ValueError("BoW proof fit and held-out rows overlap")
        _row_fingerprint(self.fit_row_ids)
        _row_fingerprint(self.heldout_row_ids)
        self.fit_texts = tuple(str(text) for text in fit_texts)
        self.heldout_texts = tuple(str(text) for text in heldout_texts)
        if len(self.fit_texts) != len(self.fit_row_ids) or len(self.heldout_texts) != len(
            self.heldout_row_ids
        ):
            raise ValueError("BoW proof text bindings changed scope length")
        self.text_column = str(text_column)
        self.outcome_type = str(outcome_type).lower()
        self.e_clip = float(e_clip)
        if not 0.0 < self.e_clip < 0.5:
            raise ValueError("BoW proof e_clip must be in (0, 0.5)")
        self.nuisance_folds = int(nuisance_folds)
        self.effect_folds = int(effect_folds)
        self.view_configs = tuple(_json_safe_view_config(value) for value in view_configs)
        names = tuple(str(value["name"]) for value in self.view_configs)
        if not names or len(names) != len(set(names)):
            raise ValueError("BoW proof requires unique configured views")
        self._view_config_by_name = {str(value["name"]): value for value in self.view_configs}
        self._store = _ArrayStore()
        self._folds: list[dict[str, Any]] = []
        self._full_fit: list[dict[str, Any]] = []
        self._scope_outputs: dict[str, str] = {}
        self._scope_output_roles: dict[str, str] = {}
        self._nuisance_source_names: list[str] = []
        self._finalized = False

    def _positions_to_rows(self, positions: Sequence[int], *, name: str) -> tuple[int, ...]:
        values = tuple(map(int, positions))
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} positions must be nonempty and unique")
        if min(values) < 0 or max(values) >= len(self.fit_row_ids):
            raise ValueError(f"{name} positions escape the exact fit scope")
        return tuple(self.fit_row_ids[position] for position in values)

    def record_fold(
        self,
        *,
        family: str,
        objective: str,
        view_name: str,
        view_config: Mapping[str, Any],
        fold: int,
        fit_positions: Sequence[int],
        validation_positions: Sequence[int],
        seed: int,
        target_values: Sequence[float],
        sample_weight: Sequence[float] | None,
        vectorizer_params: Mapping[str, Any],
        vectorizer: TfidfVectorizer | None,
        learner: Any | None,
        classification: bool,
        constant_prediction: float | None,
        validation_prediction: Sequence[float],
        heldout_prediction: Sequence[float],
    ) -> None:
        if self._finalized:
            raise RuntimeError("BoW proof capture is already finalized")
        family = str(family)
        objective = str(objective)
        if family == "bow_nuisance" and objective not in _NUISANCE_OBJECTIVES:
            raise ValueError("BoW nuisance capture received another objective")
        if family == "bow_r_loss" and objective not in _EFFECT_OBJECTIVES:
            raise ValueError("BoW R-loss capture received another objective")
        if family not in {"bow_nuisance", "bow_r_loss"}:
            raise ValueError("BoW proof capture received another family")
        view_name = str(view_name)
        if (
            view_name not in self._view_config_by_name
            or _json_safe_view_config(view_config) != self._view_config_by_name[view_name]
        ):
            raise ValueError("BoW fold changed its configured view")
        fit_positions = tuple(map(int, fit_positions))
        validation_positions = tuple(map(int, validation_positions))
        fit_rows = self._positions_to_rows(fit_positions, name="fold fit")
        validation_rows = self._positions_to_rows(
            validation_positions,
            name="fold validation",
        )
        if set(fit_rows) & set(validation_rows) or set(fit_rows) | set(validation_rows) != set(
            self.fit_row_ids
        ):
            raise ValueError("BoW fold does not partition the exact fit scope")
        targets = _finite_array(target_values, name="fold target", length=len(self.fit_row_ids))
        validation_prediction = _finite_array(
            validation_prediction,
            name="fold validation prediction",
            length=len(validation_rows),
        )
        heldout_prediction = _finite_array(
            heldout_prediction,
            name="fold heldout prediction",
            length=len(self.heldout_row_ids),
        )
        prefix = f"fold_{len(self._folds):04d}"
        state: dict[str, Any] = {
            "schema_version": BOW_NATIVE_CAPTURE_SCHEMA,
            "family": family,
            "objective": objective,
            "view_name": view_name,
            "view_config": self._view_config_by_name[view_name],
            "fold": int(fold),
            "seed": int(seed),
            "classification": bool(classification),
            "fit_row_ids": list(fit_rows),
            "validation_row_ids": list(validation_rows),
            "fit_row_fingerprint": _row_fingerprint(fit_rows),
            "validation_row_fingerprint": _row_fingerprint(validation_rows),
            "fit_target": self._store.add(f"{prefix}_fit_target", targets[list(fit_positions)]),
            "validation_target": self._store.add(
                f"{prefix}_validation_target",
                targets[list(validation_positions)],
            ),
            "validation_prediction": self._store.add(
                f"{prefix}_validation_prediction",
                validation_prediction,
            ),
            "heldout_prediction": self._store.add(
                f"{prefix}_heldout_prediction",
                heldout_prediction,
            ),
            "heldout_columns_read": ["_oci_row_id", self.text_column],
            "heldout_labels_accessed": False,
        }
        if sample_weight is None:
            state["fit_sample_weight"] = None
        else:
            weights = _finite_array(
                sample_weight,
                name="fold sample weight",
                length=len(self.fit_row_ids),
            )
            if np.any(weights < 0):
                raise ValueError("BoW fold sample weights cannot be negative")
            state["fit_sample_weight"] = self._store.add(
                f"{prefix}_fit_sample_weight",
                weights[list(fit_positions)],
            )
        if vectorizer is None:
            if learner is not None:
                raise ValueError("constant BoW fold cannot retain a learner without a vectorizer")
            state["vectorizer"] = None
        else:
            state["vectorizer"] = _capture_vectorizer(
                vectorizer,
                self._store,
                f"{prefix}_vectorizer",
                vectorizer_params=vectorizer_params,
            )
        state["learner"] = _capture_learner(
            learner,
            self._store,
            f"{prefix}_learner",
            classification=classification,
            constant_prediction=constant_prediction,
        )
        self._folds.append(state)

    def record_full_fit(
        self,
        *,
        view_name: str,
        view_config: Mapping[str, Any],
        vectorizer_params: Mapping[str, Any],
        vectorizer: TfidfVectorizer,
        objective: str,
        seed: int,
        target_values: Sequence[float],
        sample_weight: Sequence[float] | None,
        learner: Any | None,
        classification: bool,
        constant_prediction: float | None,
        fit_prediction: Sequence[float],
    ) -> None:
        if self._finalized:
            raise RuntimeError("BoW proof capture is already finalized")
        objective = str(objective)
        if objective not in _FULL_FIT_OBJECTIVES:
            raise ValueError("BoW full-fit capture received another objective")
        view_name = str(view_name)
        if (
            view_name not in self._view_config_by_name
            or _json_safe_view_config(view_config) != self._view_config_by_name[view_name]
        ):
            raise ValueError("BoW full-fit capture changed its configured view")
        targets = _finite_array(
            target_values,
            name="full-fit target",
            length=len(self.fit_row_ids),
        )
        predictions = _finite_array(
            fit_prediction,
            name="full-fit prediction",
            length=len(self.fit_row_ids),
        )
        prefix = f"full_fit_{len(self._full_fit):04d}"
        state: dict[str, Any] = {
            "schema_version": BOW_NATIVE_CAPTURE_SCHEMA,
            "objective": objective,
            "view_name": view_name,
            "view_config": self._view_config_by_name[view_name],
            "seed": int(seed),
            "classification": bool(classification),
            "fit_row_ids": list(self.fit_row_ids),
            "fit_row_fingerprint": _row_fingerprint(self.fit_row_ids),
            "target": self._store.add(f"{prefix}_target", targets),
            "fit_prediction": self._store.add(f"{prefix}_prediction", predictions),
            "vectorizer": _capture_vectorizer(
                vectorizer,
                self._store,
                f"{prefix}_vectorizer",
                vectorizer_params=vectorizer_params,
            ),
            "learner": _capture_learner(
                learner,
                self._store,
                f"{prefix}_learner",
                classification=classification,
                constant_prediction=constant_prediction,
            ),
            "heldout_labels_accessed": False,
        }
        if sample_weight is None:
            state["sample_weight"] = None
        else:
            weights = _finite_array(
                sample_weight,
                name="full-fit sample weight",
                length=len(self.fit_row_ids),
            )
            if np.any(weights < 0):
                raise ValueError("BoW full-fit sample weights cannot be negative")
            state["sample_weight"] = self._store.add(
                f"{prefix}_sample_weight",
                weights,
            )
        self._full_fit.append(state)

    def record_scope_output(self, name: str, values: Sequence[float], *, role: str) -> None:
        if self._finalized:
            raise RuntimeError("BoW proof capture is already finalized")
        name = str(name)
        if _SAFE_KEY.fullmatch(name) is None or name in self._scope_outputs:
            raise ValueError(f"invalid or duplicate BoW scope output: {name}")
        role = str(role)
        allowed_roles = {
            "fit_label",
            "fit_nuisance",
            "heldout_nuisance",
            "fit_residual",
            "fit_pseudo_target",
            "fit_weight",
            "fit_effect_output",
            "heldout_effect_output",
        }
        if role not in allowed_roles:
            raise ValueError("BoW scope output has an unsupported role")
        expected_length = (
            len(self.heldout_row_ids) if role.startswith("heldout_") else len(self.fit_row_ids)
        )
        array = _finite_array(values, name=name, length=expected_length)
        self._scope_outputs[name] = self._store.add(f"scope_{name}", array)
        self._scope_output_roles[name] = role

    def record_nuisance_source(
        self,
        *,
        source_index: int,
        source_name: str,
        e_fit: Sequence[float],
        m_fit: Sequence[float],
        e_heldout: Sequence[float],
        m_heldout: Sequence[float],
    ) -> None:
        source_index = int(source_index)
        if source_index != len(self._nuisance_source_names):
            raise ValueError("BoW nuisance sources must be captured once in native order")
        source_name = str(source_name)
        if not source_name or source_name in self._nuisance_source_names:
            raise ValueError("BoW nuisance source name is empty or duplicated")
        self._nuisance_source_names.append(source_name)
        prefix = f"nuisance_source_{source_index:04d}"
        self.record_scope_output(f"{prefix}_e_fit", e_fit, role="fit_nuisance")
        self.record_scope_output(f"{prefix}_m_fit", m_fit, role="fit_nuisance")
        self.record_scope_output(
            f"{prefix}_e_heldout",
            e_heldout,
            role="heldout_nuisance",
        )
        self.record_scope_output(
            f"{prefix}_m_heldout",
            m_heldout,
            role="heldout_nuisance",
        )

    def finalize(self) -> Mapping[str, Any]:
        if self._finalized:
            raise RuntimeError("BoW proof capture is already finalized")
        self._finalized = True
        body = {
            "schema_version": BOW_NATIVE_CAPTURE_SCHEMA,
            "scope_id": self.scope_id,
            "outer_fold": self.outer_fold,
            "inner_fold": self.inner_fold,
            "fit_row_ids": list(self.fit_row_ids),
            "heldout_row_ids": list(self.heldout_row_ids),
            "fit_row_fingerprint": _row_fingerprint(self.fit_row_ids),
            "heldout_row_fingerprint": _row_fingerprint(self.heldout_row_ids),
            "fit_text_sha256": _text_sha256(self.fit_row_ids, self.fit_texts),
            "heldout_text_sha256": _text_sha256(self.heldout_row_ids, self.heldout_texts),
            "text_column": self.text_column,
            "outcome_type": self.outcome_type,
            "e_clip": self.e_clip,
            "nuisance_folds": self.nuisance_folds,
            "effect_folds": self.effect_folds,
            "view_configs": list(self.view_configs),
            "folds": self._folds,
            "full_fit_models": self._full_fit,
            "scope_outputs": {
                name: {"array": key, "role": self._scope_output_roles[name]}
                for name, key in sorted(self._scope_outputs.items())
            },
            "nuisance_source_names": list(self._nuisance_source_names),
            "array_inventory": self._store.inventory,
            "array_file": "arrays.npz",
            "heldout_columns_read": ["_oci_row_id", self.text_column],
            "heldout_labels_accessed": False,
            "oracle_fields_accessed": False,
            "secrets_accessed": False,
            "executable_serialization_used": False,
            "joblib_or_pickle_used": False,
        }
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        arrays_path = self.artifact_dir / "arrays.npz"
        _atomic_write_npz(arrays_path, self._store.arrays)
        body["array_file_sha256"] = _sha256_file(arrays_path)
        metadata = {**body, "content_sha256": _sha256_json(body)}
        _atomic_write_bytes(
            self.artifact_dir / "metadata.json",
            (json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
        )
        return validate_bow_native_capture(
            self.artifact_dir,
            expected_scope_id=self.scope_id,
            expected_fit_row_ids=self.fit_row_ids,
            expected_heldout_row_ids=self.heldout_row_ids,
            fit_texts=self.fit_texts,
            heldout_texts=self.heldout_texts,
        )


def _load_capture(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("BoW native capture must be one real directory")
    children = sorted(item.name for item in root.iterdir())
    if children != ["arrays.npz", "metadata.json"]:
        raise ValueError("BoW native capture has an open or executable file layout")
    if any(item.is_symlink() for item in root.iterdir()) or any(
        item.name.lower().endswith(_FORBIDDEN_SUFFIXES) for item in root.rglob("*")
    ):
        raise ValueError("BoW native capture contains a symlink or executable serialization")
    metadata_path = root / "metadata.json"
    arrays_path = root / "arrays.npz"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("BoW native capture metadata is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("BoW native capture metadata must be one JSON object")
    body = {key: value for key, value in metadata.items() if key != "content_sha256"}
    if (
        metadata.get("schema_version") != BOW_NATIVE_CAPTURE_SCHEMA
        or metadata.get("content_sha256") != _sha256_json(body)
        or metadata.get("array_file") != arrays_path.name
        or metadata.get("array_file_sha256") != _sha256_file(arrays_path)
        or metadata.get("heldout_labels_accessed") is not False
        or metadata.get("oracle_fields_accessed") is not False
        or metadata.get("secrets_accessed") is not False
        or metadata.get("executable_serialization_used") is not False
        or metadata.get("joblib_or_pickle_used") is not False
    ):
        raise ValueError("BoW native capture has an invalid closed envelope")
    inventory = metadata.get("array_inventory")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("BoW native capture has no numerical inventory")
    try:
        with np.load(arrays_path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(inventory):
                raise ValueError("BoW native capture NPZ inventory is incomplete")
            arrays = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("BoW"):
            raise
        raise ValueError("BoW native capture NPZ is unsafe or malformed") from exc
    for key, record in inventory.items():
        array = arrays[key]
        if (
            not isinstance(record, Mapping)
            or record.get("dtype") != array.dtype.str
            or record.get("shape") != list(array.shape)
            or record.get("content_sha256") != _array_sha256(array)
        ):
            raise RuntimeError(f"BoW native capture array changed: {key}")
    return metadata, arrays


def _fold_predictions(
    fold: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    fit_row_ids: tuple[int, ...],
    fit_text_by_row: Mapping[int, str],
    heldout_texts: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    validation_rows = tuple(map(int, fold.get("validation_row_ids") or ()))
    validation_texts = [fit_text_by_row[row_id] for row_id in validation_rows]
    learner = fold.get("learner")
    if not isinstance(learner, Mapping):
        raise ValueError("BoW fold has no captured learner state")
    vectorizer_state = fold.get("vectorizer")
    if vectorizer_state is None:
        x_validation = sparse.csr_matrix((len(validation_texts), 0), dtype=np.float32)
        x_heldout = sparse.csr_matrix((len(heldout_texts), 0), dtype=np.float32)
    elif isinstance(vectorizer_state, Mapping):
        vectorizer = _restore_vectorizer(vectorizer_state, arrays)
        x_validation = vectorizer.transform(validation_texts).tocsr()
        x_heldout = vectorizer.transform(heldout_texts).tocsr()
    else:
        raise ValueError("BoW fold vectorizer state is malformed")
    validation = _predict_learner(learner, arrays, x_validation)
    heldout = _predict_learner(learner, arrays, x_heldout)
    if fold.get("classification") is True:
        # Native nuisance code clips both validation and held-out predictions.
        e_clip = float(fold.get("e_clip", 0.0) or 0.0)
        if e_clip > 0:
            validation = np.clip(validation, e_clip, 1.0 - e_clip)
            heldout = np.clip(heldout, e_clip, 1.0 - e_clip)
    return validation, heldout


def _assert_close(observed: Any, expected: Any, *, name: str) -> None:
    observed_array = np.asarray(observed, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if observed_array.shape != expected_array.shape or not np.allclose(
        observed_array,
        expected_array,
        rtol=2e-7,
        atol=2e-8,
    ):
        maximum = (
            float(np.max(np.abs(observed_array - expected_array)))
            if observed_array.shape == expected_array.shape and observed_array.size
            else float("inf")
        )
        raise RuntimeError(f"BoW native replay differs for {name}; max_abs={maximum}")


def validate_bow_native_capture(
    artifact_dir: Path,
    *,
    expected_scope_id: str | None = None,
    expected_fit_row_ids: Sequence[int] | None = None,
    expected_heldout_row_ids: Sequence[int] | None = None,
    fit_texts: Sequence[str] | None = None,
    heldout_texts: Sequence[str] | None = None,
    expected_fit_treatment: Sequence[float] | None = None,
    expected_fit_outcome: Sequence[float] | None = None,
) -> Mapping[str, Any]:
    """Validate and replay a closed JSON/NPZ BoW native capture."""

    metadata, arrays = _load_capture(Path(artifact_dir))
    fit_rows = tuple(map(int, metadata.get("fit_row_ids") or ()))
    heldout_rows = tuple(map(int, metadata.get("heldout_row_ids") or ()))
    if (
        not fit_rows
        or not heldout_rows
        or set(fit_rows) & set(heldout_rows)
        or metadata.get("fit_row_fingerprint") != _row_fingerprint(fit_rows)
        or metadata.get("heldout_row_fingerprint") != _row_fingerprint(heldout_rows)
    ):
        raise ValueError("BoW native capture has invalid exact row bindings")
    if expected_scope_id is not None and metadata.get("scope_id") != str(expected_scope_id):
        raise ValueError("BoW native capture belongs to another scope")
    if expected_fit_row_ids is not None and fit_rows != tuple(map(int, expected_fit_row_ids)):
        raise ValueError("BoW native capture changed exact fit row order")
    if expected_heldout_row_ids is not None and heldout_rows != tuple(
        map(int, expected_heldout_row_ids)
    ):
        raise ValueError("BoW native capture changed exact held-out row order")
    if fit_texts is None or heldout_texts is None:
        raise ValueError("BoW native capture validation requires external exact-scope texts")
    fit_texts = tuple(str(text) for text in fit_texts)
    heldout_texts = tuple(str(text) for text in heldout_texts)
    if metadata.get("fit_text_sha256") != _text_sha256(fit_rows, fit_texts) or metadata.get(
        "heldout_text_sha256"
    ) != _text_sha256(heldout_rows, heldout_texts):
        raise ValueError("BoW native capture text binding changed")
    view_configs = metadata.get("view_configs")
    if not isinstance(view_configs, list) or not view_configs:
        raise ValueError("BoW native capture has no configured views")
    view_names = tuple(
        str(value.get("name")) for value in view_configs if isinstance(value, Mapping)
    )
    if len(view_names) != len(view_configs) or len(view_names) != len(set(view_names)):
        raise ValueError("BoW native capture view inventory is malformed")
    folds = metadata.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("BoW native capture has no per-fold learner states")
    fit_text_by_row = dict(zip(fit_rows, fit_texts))
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise ValueError("BoW native capture fold metadata is malformed")
        family = str(fold.get("family") or "")
        objective = str(fold.get("objective") or "")
        view_name = str(fold.get("view_name") or "")
        if (
            view_name not in view_names
            or fold.get("heldout_labels_accessed") is not False
            or fold.get("heldout_columns_read") != ["_oci_row_id", metadata.get("text_column")]
            or family not in {"bow_nuisance", "bow_r_loss"}
        ):
            raise ValueError("BoW native capture fold changed family/view/label policy")
        expected_objectives = (
            _NUISANCE_OBJECTIVES if family == "bow_nuisance" else _EFFECT_OBJECTIVES
        )
        if objective not in expected_objectives:
            raise ValueError("BoW native capture fold has another objective")
        fit_ids = tuple(map(int, fold.get("fit_row_ids") or ()))
        validation_ids = tuple(map(int, fold.get("validation_row_ids") or ()))
        if (
            set(fit_ids) & set(validation_ids)
            or set(fit_ids) | set(validation_ids) != set(fit_rows)
            or fold.get("fit_row_fingerprint") != _row_fingerprint(fit_ids)
            or fold.get("validation_row_fingerprint") != _row_fingerprint(validation_ids)
        ):
            raise ValueError("BoW native capture fold has invalid fit/validation bindings")
        validation, heldout = _fold_predictions(
            fold,
            arrays,
            fit_row_ids=fit_rows,
            fit_text_by_row=fit_text_by_row,
            heldout_texts=heldout_texts,
        )
        if fold.get("classification") is True:
            e_clip = float(metadata.get("e_clip"))
            validation = np.clip(validation, e_clip, 1.0 - e_clip)
            heldout = np.clip(heldout, e_clip, 1.0 - e_clip)
        _assert_close(
            validation,
            arrays[str(fold.get("validation_prediction"))],
            name=f"{view_name}.{objective}.fold{fold.get('fold')}.validation",
        )
        _assert_close(
            heldout,
            arrays[str(fold.get("heldout_prediction"))],
            name=f"{view_name}.{objective}.fold{fold.get('fold')}.heldout",
        )
        grouped.setdefault((view_name, objective), []).append(fold)
    expected_groups = {
        (view_name, objective)
        for view_name in view_names
        for objective in (*_NUISANCE_OBJECTIVES, *_EFFECT_OBJECTIVES)
    }
    if set(grouped) != expected_groups:
        raise ValueError("BoW native capture has missing view/objective fold states")
    for key, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row.get("fold", 0)))
        if tuple(int(row.get("fold", 0)) for row in ordered) != tuple(range(1, len(ordered) + 1)):
            raise ValueError(f"BoW native capture has a missing fold: {key}")
        validation_flat = [
            int(row_id) for row in ordered for row_id in row.get("validation_row_ids") or ()
        ]
        if len(validation_flat) != len(fit_rows) or set(validation_flat) != set(fit_rows):
            raise ValueError(
                f"BoW native capture has a missing fold or overlapping validation rows: {key}"
            )
    full_fit = metadata.get("full_fit_models")
    if not isinstance(full_fit, list):
        raise ValueError("BoW native capture has no full-fit learner inventory")
    full_groups: dict[tuple[str, str], Mapping[str, Any]] = {}
    for state in full_fit:
        if not isinstance(state, Mapping):
            raise ValueError("BoW native capture full-fit metadata is malformed")
        key = (str(state.get("view_name") or ""), str(state.get("objective") or ""))
        if key in full_groups or key[0] not in view_names or key[1] not in _FULL_FIT_OBJECTIVES:
            raise ValueError("BoW native capture has duplicate or unknown full-fit state")
        if state.get("heldout_labels_accessed") is not False:
            raise ValueError("BoW full-fit state changed its label-access policy")
        vectorizer_state = state.get("vectorizer") or {}
        learner_state = state.get("learner") or {}
        if learner_state.get("kind") == "sklearn_ridge":
            # Ridge float32 predictions depend on the native fit-transform CSR
            # accumulation order. Tree and logistic replay are order-invariant
            # and use the ordinary transformed matrix below.
            x_fit = _replay_fit_transform(vectorizer_state, arrays, fit_texts)
        else:
            x_fit = _restore_vectorizer(vectorizer_state, arrays).transform(
                fit_texts
            ).tocsr()
        prediction = _predict_learner(learner_state, arrays, x_fit)
        _assert_close(
            prediction,
            arrays[str(state.get("fit_prediction"))],
            name=f"{key[0]}.{key[1]}.full_fit",
        )
        full_groups[key] = state
    expected_full = {
        (view_name, objective) for view_name in view_names for objective in _FULL_FIT_OBJECTIVES
    }
    if set(full_groups) != expected_full:
        raise ValueError("BoW native capture has missing full-fit objectives")
    scope_outputs = metadata.get("scope_outputs")
    if not isinstance(scope_outputs, Mapping):
        raise ValueError("BoW native capture has no bound scope numerics")

    def output(name: str) -> np.ndarray:
        record = scope_outputs.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"BoW native capture is missing scope output: {name}")
        return np.asarray(arrays[str(record.get("array"))], dtype=np.float64)

    treatment = output("treatment")
    outcome = output("outcome")
    if (expected_fit_treatment is None) != (expected_fit_outcome is None):
        raise ValueError(
            "BoW canonical fit treatment/outcome must be supplied together"
        )
    if expected_fit_treatment is not None:
        canonical_treatment = _finite_array(
            expected_fit_treatment,
            name="canonical fit treatment",
            length=len(fit_rows),
        )
        canonical_outcome = _finite_array(
            expected_fit_outcome,
            name="canonical fit outcome",
            length=len(fit_rows),
        )
        if not np.array_equal(treatment, canonical_treatment):
            raise ValueError("BoW native capture treatment differs from canonical fit labels")
        if not np.array_equal(outcome, canonical_outcome):
            raise ValueError("BoW native capture outcome differs from canonical fit labels")
    ensemble_e_fit = output("ensemble_e_fit")
    ensemble_m_fit = output("ensemble_m_fit")
    ensemble_e_heldout = output("ensemble_e_heldout")
    ensemble_m_heldout = output("ensemble_m_heldout")
    y_residual = output("y_residual")
    t_residual = output("t_residual")
    pseudo_target = output("pseudo_target")
    r_weight = output("r_weight")
    e_clip = float(metadata.get("e_clip"))
    _assert_close(
        treatment - np.clip(ensemble_e_fit, e_clip, 1.0 - e_clip), t_residual, name="t_residual"
    )
    _assert_close(outcome - ensemble_m_fit, y_residual, name="y_residual")
    _assert_close(y_residual / t_residual, pseudo_target, name="pseudo_target")
    _assert_close(np.square(t_residual), r_weight, name="r_weight")
    for view_name in view_names:
        objective_outputs = {
            "treatment_nuisance": (
                output(f"view_{view_names.index(view_name):04d}_e_fit"),
                output(f"view_{view_names.index(view_name):04d}_e_heldout"),
            ),
            "outcome_nuisance": (
                output(f"view_{view_names.index(view_name):04d}_m_fit"),
                output(f"view_{view_names.index(view_name):04d}_m_heldout"),
            ),
            "effect_pseudo_target": (
                output(f"view_{view_names.index(view_name):04d}_pseudo_fit"),
                output(f"view_{view_names.index(view_name):04d}_pseudo_heldout"),
            ),
            "effect_weighted_r": (
                output(f"view_{view_names.index(view_name):04d}_weighted_fit"),
                output(f"view_{view_names.index(view_name):04d}_weighted_heldout"),
            ),
        }
        for objective, (expected_fit, expected_heldout) in objective_outputs.items():
            rows = grouped[(view_name, objective)]
            aggregated_fit = np.full(len(fit_rows), np.nan, dtype=np.float64)
            heldout_predictions = []
            position_by_row = {row_id: index for index, row_id in enumerate(fit_rows)}
            for row in rows:
                validation_ids = tuple(map(int, row["validation_row_ids"]))
                positions = [position_by_row[row_id] for row_id in validation_ids]
                aggregated_fit[positions] = arrays[str(row["validation_prediction"])]
                heldout_predictions.append(
                    np.asarray(arrays[str(row["heldout_prediction"])], dtype=np.float64)
                )
            _assert_close(aggregated_fit, expected_fit, name=f"{view_name}.{objective}.oof")
            _assert_close(
                np.mean(np.vstack(heldout_predictions), axis=0),
                expected_heldout,
                name=f"{view_name}.{objective}.heldout_mean",
            )
    nuisance_source_names = metadata.get("nuisance_source_names")
    if not isinstance(nuisance_source_names, list) or len(nuisance_source_names) < len(view_names):
        raise ValueError("BoW native capture has an incomplete nuisance-source inventory")
    e_fit_sources = np.vstack(
        [
            output(f"nuisance_source_{index:04d}_e_fit")
            for index in range(len(nuisance_source_names))
        ]
    )
    m_fit_sources = np.vstack(
        [
            output(f"nuisance_source_{index:04d}_m_fit")
            for index in range(len(nuisance_source_names))
        ]
    )
    e_heldout_sources = np.vstack(
        [
            output(f"nuisance_source_{index:04d}_e_heldout")
            for index in range(len(nuisance_source_names))
        ]
    )
    m_heldout_sources = np.vstack(
        [
            output(f"nuisance_source_{index:04d}_m_heldout")
            for index in range(len(nuisance_source_names))
        ]
    )
    _assert_close(np.mean(e_fit_sources, axis=0), ensemble_e_fit, name="ensemble_e_fit")
    _assert_close(np.mean(m_fit_sources, axis=0), ensemble_m_fit, name="ensemble_m_fit")
    _assert_close(
        np.mean(e_heldout_sources, axis=0),
        ensemble_e_heldout,
        name="ensemble_e_heldout",
    )
    _assert_close(
        np.mean(m_heldout_sources, axis=0),
        ensemble_m_heldout,
        name="ensemble_m_heldout",
    )
    return json.loads(_canonical_json(metadata))


__all__ = [
    "BOW_NATIVE_CAPTURE_SCHEMA",
    "NativeBoWProofCaptureSink",
    "validate_bow_native_capture",
]
