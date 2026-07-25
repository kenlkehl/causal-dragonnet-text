"""Honest outer-CV causal forest over frozen TF-IDF/NMF topic scores.

This is a fast, non-agentic Stage 2 alternative. It consumes only the exact
full-outer-train contexts produced by :mod:`tfidf_topic_stage1`: treatment and
outcome topic scores enter ``W`` and effect topic scores enter ``X``. Oracle
columns are joined only after the complete prediction parquet has been written
and hashed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import NuisanceCalibrationScientificConfig
from ..models.causal_forest_head import CausalForestHead
from .tfidf_topic_discovery import (
    HANDOFF_SCHEMA_VERSION,
    nuisance_metrics,
    row_set_fingerprint,
    stable_hash,
)
from .tfidf_safe_artifacts import load_named_array_bank

logger = logging.getLogger(__name__)

TOPIC_SCORE_FOREST_SCHEMA_VERSION = "tfidf_topic_score_forest_v1"
_REQUIRED_BANKS = ("treatment", "outcome", "effect")


@dataclass(frozen=True)
class TopicScoreForestConfig:
    """Fixed modeling choices for the direct topic-score forest."""

    n_estimators: int = 200
    max_depth: Optional[int] = None
    min_samples_leaf: int = 10
    max_features: Union[str, int, float] = "sqrt"
    honest: bool = True
    inference: bool = True
    tune_model: bool = False
    standardize: bool = True
    variance_tolerance: float = 1e-12
    include_stacked_nuisance_in_w: bool = False
    random_state: int = 42
    persist_fold_models: bool = True

    def validate(self) -> None:
        if self.n_estimators < 4 or self.n_estimators % 4:
            raise ValueError("n_estimators must be at least 4 and divisible by 4")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be positive")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be positive when provided")
        if self.variance_tolerance < 0:
            raise ValueError("variance_tolerance must be non-negative")


@dataclass(frozen=True)
class TopicBankTransform:
    bank: str
    topic_ids: List[str]
    retained_indices: np.ndarray
    means: np.ndarray
    scales: np.ndarray

    def transform(self, values: np.ndarray, *, standardize: bool) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)[:, self.retained_indices]
        if standardize:
            matrix = (matrix - self.means[None, :]) / self.scales[None, :]
        return np.asarray(matrix, dtype=np.float32)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    temporary.replace(path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_handoff(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != HANDOFF_SCHEMA_VERSION:
                raise ValueError(
                    "The topic-score forest requires a v2 exact-context handoff; "
                    f"line {line_number} has {row.get('schema_version')!r}."
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Stage 1 handoff is empty: {path}")
    return rows


def _validate_outer_contexts(
    rows: Sequence[Dict[str, Any]],
    row_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    outer = sorted(
        (row for row in rows if row.get("scope") == "full_outer_train"),
        key=lambda row: int(row["outer_fold"]),
    )
    if not outer:
        raise ValueError("Handoff contains no full_outer_train contexts")
    folds = [int(row["outer_fold"]) for row in outer]
    if len(set(folds)) != len(folds):
        raise ValueError(f"Handoff contains duplicate full outer folds: {folds}")

    expected_ids = {int(value) for value in row_ids}
    observed_heldout: List[int] = []
    for row in outer:
        fit_ids = [int(value) for value in row.get("fit_row_ids", [])]
        heldout_ids = [int(value) for value in row.get("heldout_row_ids", [])]
        if not fit_ids or not heldout_ids:
            raise ValueError(f"Outer fold {row['outer_fold']} has an empty row set")
        if set(fit_ids) & set(heldout_ids):
            raise ValueError(f"Outer fold {row['outer_fold']} leaks rows across fit/held-out")
        if set(fit_ids) | set(heldout_ids) != expected_ids:
            raise ValueError(
                f"Outer fold {row['outer_fold']} does not partition the supplied dataset"
            )
        if row.get("fit_row_fingerprint") != row_set_fingerprint(fit_ids):
            raise ValueError(f"Outer fold {row['outer_fold']} fit fingerprint mismatch")
        if row.get("heldout_row_fingerprint") != row_set_fingerprint(heldout_ids):
            raise ValueError(f"Outer fold {row['outer_fold']} held-out fingerprint mismatch")
        discovery = row.get("discovery") or {}
        if discovery.get("fit_row_fingerprint") != row.get("fit_row_fingerprint"):
            raise ValueError(f"Outer fold {row['outer_fold']} discovery fit scope mismatch")
        if discovery.get("heldout_row_fingerprint") != row.get("heldout_row_fingerprint"):
            raise ValueError(f"Outer fold {row['outer_fold']} discovery holdout mismatch")
        observed_heldout.extend(heldout_ids)

    if len(observed_heldout) != len(set(observed_heldout)):
        raise ValueError("A row appears in more than one outer held-out fold")
    if set(observed_heldout) != expected_ids:
        raise ValueError("Outer held-out folds do not cover every dataset row exactly once")
    return outer


def _resolve_artifact_path(value: Any, handoff_path: Path, scope_id: str) -> Path:
    requested = Path(str(value)).expanduser()
    candidates = [
        requested,
        Path.cwd() / requested,
        handoff_path.parent / requested,
        handoff_path.parent.parent
        / "stage1_tfidf_topics"
        / "contexts"
        / str(scope_id)
        / requested.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve Stage 1 artifact path: {value}")


def _load_topic_bank(path: Path, *, row_count: int) -> Dict[str, np.ndarray]:
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in load_named_array_bank(
            path,
            expected_row_count=row_count,
        ).items()
    }


def _topic_ids(metadata: Dict[str, Any], bank: str, width: int) -> List[str]:
    bank_metadata = (metadata.get("topic_banks") or {}).get(bank) or {}
    topics = list(bank_metadata.get("topics") or [])
    if len(topics) != width:
        raise ValueError(
            f"{bank} topic metadata has {len(topics)} definitions for {width} score columns"
        )
    return [
        str(topic.get("topic_id") or f"{bank}_topic_{index + 1:03d}")
        for index, topic in enumerate(topics)
    ]


def _fit_bank_transform(
    bank: str,
    train_values: np.ndarray,
    topic_ids: Sequence[str],
    config: TopicScoreForestConfig,
) -> TopicBankTransform:
    values = np.asarray(train_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(topic_ids):
        raise ValueError(f"Invalid {bank} topic matrix shape: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{bank} topic matrix contains non-finite values")
    variances = np.var(values, axis=0)
    retained = np.flatnonzero(variances > float(config.variance_tolerance))
    if not len(retained):
        raise ValueError(f"All {bank} topic columns are constant in the outer training fold")
    retained_values = values[:, retained]
    means = retained_values.mean(axis=0)
    scales = retained_values.std(axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    return TopicBankTransform(
        bank=bank,
        topic_ids=[str(topic_ids[index]) for index in retained],
        retained_indices=retained.astype(int),
        means=np.asarray(means, dtype=np.float64),
        scales=np.asarray(scales, dtype=np.float64),
    )


def prepare_topic_score_matrices(
    *,
    fit_scores: Dict[str, np.ndarray],
    heldout_scores: Dict[str, np.ndarray],
    discovery: Dict[str, Any],
    config: TopicScoreForestConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, TopicBankTransform]]:
    """Build fold-local ``X``/``W`` matrices without fitting on held-out scores."""
    missing = [
        bank for bank in _REQUIRED_BANKS if bank not in fit_scores or bank not in heldout_scores
    ]
    if missing:
        raise ValueError(f"Stage 1 topic arrays are missing required banks: {missing}")
    transforms: Dict[str, TopicBankTransform] = {}
    transformed_fit: Dict[str, np.ndarray] = {}
    transformed_heldout: Dict[str, np.ndarray] = {}
    for bank in _REQUIRED_BANKS:
        fit = np.asarray(fit_scores[bank], dtype=np.float64)
        heldout = np.asarray(heldout_scores[bank], dtype=np.float64)
        if fit.ndim != 2 or heldout.ndim != 2 or fit.shape[1] != heldout.shape[1]:
            raise ValueError(
                f"Incompatible {bank} fit/held-out shapes: {fit.shape} and {heldout.shape}"
            )
        if not np.isfinite(heldout).all():
            raise ValueError(f"{bank} held-out topic matrix contains non-finite values")
        ids = _topic_ids(discovery, bank, fit.shape[1])
        transform = _fit_bank_transform(bank, fit, ids, config)
        transforms[bank] = transform
        transformed_fit[bank] = transform.transform(fit, standardize=config.standardize)
        transformed_heldout[bank] = transform.transform(heldout, standardize=config.standardize)
    x_fit = transformed_fit["effect"]
    x_heldout = transformed_heldout["effect"]
    w_fit = np.column_stack([transformed_fit["treatment"], transformed_fit["outcome"]])
    w_heldout = np.column_stack([transformed_heldout["treatment"], transformed_heldout["outcome"]])
    return x_fit, x_heldout, w_fit, w_heldout, transforms


def _ordered_nuisance_rows(
    nuisance: pd.DataFrame,
    row_ids: Sequence[int],
    scope: str,
) -> pd.DataFrame:
    selected = nuisance[nuisance["prediction_scope"] == scope].copy()
    if selected["_oci_row_id"].duplicated().any():
        raise ValueError(f"Duplicate Stage 1 nuisance rows for scope {scope}")
    indexed = selected.set_index("_oci_row_id", drop=False)
    missing = [int(row_id) for row_id in row_ids if int(row_id) not in indexed.index]
    if missing:
        raise ValueError(f"Stage 1 nuisance artifact is missing {scope} rows: {missing[:5]}")
    return indexed.loc[[int(value) for value in row_ids]].reset_index(drop=True)


def _standardize_extra_w(
    fit: np.ndarray,
    heldout: np.ndarray,
    *,
    standardize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    fit = np.asarray(fit, dtype=np.float64)
    heldout = np.asarray(heldout, dtype=np.float64)
    if not standardize:
        return fit.astype(np.float32), heldout.astype(np.float32)
    means = fit.mean(axis=0)
    scales = fit.std(axis=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    return (
        ((fit - means) / scales).astype(np.float32),
        ((heldout - means) / scales).astype(np.float32),
    )


def _r_loss(
    outcome: np.ndarray,
    treatment: np.ndarray,
    outcome_prediction: np.ndarray,
    propensity: np.ndarray,
    treatment_effect: np.ndarray,
) -> float:
    residual_y = np.asarray(outcome) - np.asarray(outcome_prediction)
    residual_t = np.asarray(treatment) - np.asarray(propensity)
    return float(np.mean((residual_y - np.asarray(treatment_effect) * residual_t) ** 2))


def _safe_pearson(first: Sequence[float], second: Sequence[float]) -> Optional[float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    if mask.sum() < 2 or np.std(first[mask]) == 0 or np.std(second[mask]) == 0:
        return None
    return float(np.corrcoef(first[mask], second[mask])[0, 1])


def _safe_spearman(first: Sequence[float], second: Sequence[float]) -> Optional[float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    if mask.sum() < 2 or np.std(first[mask]) == 0 or np.std(second[mask]) == 0:
        return None
    result = spearmanr(first[mask], second[mask])
    return float(result.statistic) if np.isfinite(result.statistic) else None


def _feature_manifest_rows(
    discovery: Dict[str, Any],
    transforms: Dict[str, TopicBankTransform],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bank in _REQUIRED_BANKS:
        transform = transforms[bank]
        retained_lookup = {
            int(original): position
            for position, original in enumerate(transform.retained_indices.tolist())
        }
        topics = list(((discovery.get("topic_banks") or {}).get(bank) or {}).get("topics") or [])
        for index, topic in enumerate(topics):
            retained_position = retained_lookup.get(index)
            rows.append(
                {
                    "bank": bank,
                    "model_role": "X_heterogeneity" if bank == "effect" else "W_adjustment",
                    "topic_id": topic.get("topic_id"),
                    "source_column_index": index,
                    "retained": retained_position is not None,
                    "training_mean": (
                        float(transform.means[retained_position])
                        if retained_position is not None
                        else None
                    ),
                    "training_scale": (
                        float(transform.scales[retained_position])
                        if retained_position is not None
                        else None
                    ),
                    "terms": topic.get("terms", []),
                }
            )
    return rows


def _heldout_score_frame(
    row_ids: Sequence[int],
    outer_fold: int,
    x_scores: np.ndarray,
    w_scores: np.ndarray,
    transforms: Dict[str, TopicBankTransform],
) -> pd.DataFrame:
    columns: Dict[str, Any] = {
        "_oci_row_id": [int(value) for value in row_ids],
        "outer_fold": np.full(len(row_ids), int(outer_fold), dtype=int),
    }
    adjustment_ids = transforms["treatment"].topic_ids + transforms["outcome"].topic_ids
    for index, topic_id in enumerate(adjustment_ids):
        columns[f"W__{topic_id}"] = w_scores[:, index]
    extra_w_names = ("stage1_stacked_propensity", "stage1_stacked_outcome")
    for offset, name in enumerate(extra_w_names, start=len(adjustment_ids)):
        if offset < w_scores.shape[1]:
            columns[f"W__{name}"] = w_scores[:, offset]
    for index, topic_id in enumerate(transforms["effect"].topic_ids):
        columns[f"X__{topic_id}"] = x_scores[:, index]
    return pd.DataFrame(columns)


def _load_complete_fold_checkpoint(
    fold_dir: Path,
    run_config_hash: str,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
    manifest_path = fold_dir / "fold_manifest.json"
    predictions_path = fold_dir / "predictions.parquet"
    scores_path = fold_dir / "heldout_topic_scores.parquet"
    metrics_path = fold_dir / "metrics.json"
    required_paths = (manifest_path, predictions_path, scores_path, metrics_path)
    if not all(path.exists() for path in required_paths):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_config_hash") != run_config_hash:
            return None
        predictions = pd.read_parquet(predictions_path)
        scores = pd.read_parquet(scores_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return predictions, scores, metrics


def _fit_outer_fold(
    *,
    model_data: pd.DataFrame,
    context: Dict[str, Any],
    handoff_path: Path,
    output_dir: Path,
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    id_columns: Sequence[str],
    config: TopicScoreForestConfig,
    run_config_hash: str,
    forest_factory: Callable[..., Any],
    force: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    fold = int(context["outer_fold"])
    fold_dir = output_dir / f"fold_{fold:03d}"
    if not force:
        checkpoint = _load_complete_fold_checkpoint(fold_dir, run_config_hash)
        if checkpoint is not None:
            logger.info("Reusing complete topic-score forest outer fold %s", fold)
            return checkpoint

    fit_ids = [int(value) for value in context["fit_row_ids"]]
    heldout_ids = [int(value) for value in context["heldout_row_ids"]]
    indexed = model_data.set_index("_oci_row_id", drop=False)
    fit_data = indexed.loc[fit_ids].reset_index(drop=True)
    heldout_data = indexed.loc[heldout_ids].reset_index(drop=True)
    discovery = context["discovery"]
    nuisance_stack_scientific = discovery.get("nuisance_stack_scientific")
    if not isinstance(nuisance_stack_scientific, Mapping):
        raise ValueError(
            "TF-IDF discovery artifact lacks nuisance_stack_scientific"
        )
    calibration_scientific = nuisance_stack_scientific.get("calibration")
    if not isinstance(calibration_scientific, Mapping):
        raise ValueError(
            "TF-IDF discovery artifact lacks nuisance calibration science"
        )
    calibration_config = NuisanceCalibrationScientificConfig(
        **dict(calibration_scientific)
    )
    scope_id = str(discovery.get("scope_id") or f"outer_{fold:03d}_full_train")
    artifacts = discovery.get("artifacts") or {}
    fit_path = _resolve_artifact_path(artifacts.get("fit_topic_values"), handoff_path, scope_id)
    heldout_path = _resolve_artifact_path(
        artifacts.get("heldout_topic_values"), handoff_path, scope_id
    )
    nuisance_path = _resolve_artifact_path(
        artifacts.get("nuisance_predictions"), handoff_path, scope_id
    )
    fit_scores = _load_topic_bank(fit_path, row_count=len(fit_ids))
    heldout_scores = _load_topic_bank(heldout_path, row_count=len(heldout_ids))
    for bank in _REQUIRED_BANKS:
        if bank in fit_scores and fit_scores[bank].shape[0] != len(fit_ids):
            raise ValueError(f"Outer fold {fold} {bank} fit score row count mismatch")
        if bank in heldout_scores and heldout_scores[bank].shape[0] != len(heldout_ids):
            raise ValueError(f"Outer fold {fold} {bank} held-out score row count mismatch")

    x_fit, x_heldout, w_fit, w_heldout, transforms = prepare_topic_score_matrices(
        fit_scores=fit_scores,
        heldout_scores=heldout_scores,
        discovery=discovery,
        config=config,
    )
    nuisance = pd.read_parquet(nuisance_path)
    fit_nuisance = _ordered_nuisance_rows(nuisance, fit_ids, "fit_oof")
    heldout_nuisance = _ordered_nuisance_rows(nuisance, heldout_ids, "external_heldout")
    if config.include_stacked_nuisance_in_w:
        extra_fit, extra_heldout = _standardize_extra_w(
            fit_nuisance[["treatment_stacked", "outcome_stacked"]].to_numpy(dtype=float),
            heldout_nuisance[["treatment_stacked", "outcome_stacked"]].to_numpy(dtype=float),
            standardize=config.standardize,
        )
        w_fit = np.column_stack([w_fit, extra_fit])
        w_heldout = np.column_stack([w_heldout, extra_heldout])

    train_t = fit_data[treatment_column].to_numpy(dtype=float)
    train_y = fit_data[outcome_column].to_numpy(dtype=float)
    forest = forest_factory(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        honest=config.honest,
        inference=config.inference,
        random_state=config.random_state + fold,
        tune_model=config.tune_model,
    )
    logger.info(
        "Fitting topic-score causal forest fold=%s train=%s test=%s X=%s W=%s",
        fold,
        len(fit_data),
        len(heldout_data),
        x_fit.shape[1],
        w_fit.shape[1],
    )
    forest.fit(x_fit, train_t, train_y, W=w_fit)
    forest_prediction = forest.predict(x_heldout, return_ci=config.inference)
    tau = np.asarray(forest_prediction["tau_pred"], dtype=float)

    propensity = np.clip(
        heldout_nuisance["treatment_stacked"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6
    )
    outcome_prediction = heldout_nuisance["outcome_stacked"].to_numpy(dtype=float)
    y0 = outcome_prediction - propensity * tau
    y1 = outcome_prediction + (1.0 - propensity) * tau
    if str(outcome_type).lower() != "continuous":
        y0 = np.clip(y0, 0.0, 1.0)
        y1 = np.clip(y1, 0.0, 1.0)

    keep_columns = [column for column in id_columns if column in heldout_data]
    prediction_columns = [
        "_oci_row_id",
        *keep_columns,
        treatment_column,
        outcome_column,
    ]
    predictions = heldout_data[prediction_columns].copy()
    predictions["pred_ite_prob"] = tau
    predictions["pred_y0_prob"] = y0
    predictions["pred_y1_prob"] = y1
    predictions["pred_propensity_prob"] = propensity
    predictions["pred_outcome_prob"] = outcome_prediction
    predictions["outer_fold"] = fold
    predictions["fit_row_fingerprint"] = context["fit_row_fingerprint"]
    predictions["heldout_row_fingerprint"] = context["heldout_row_fingerprint"]
    predictions["stage1_config_hash"] = context["stage1_config_hash"]
    predictions["topic_score_forest_config_hash"] = run_config_hash
    predictions["prediction_fitting_set_excludes_row_labels"] = True
    for source, target in (
        ("tau_lower", "pred_ite_lower"),
        ("tau_upper", "pred_ite_upper"),
        ("tau_std", "pred_ite_std"),
    ):
        values = forest_prediction.get(source)
        if values is not None:
            predictions[target] = np.asarray(values, dtype=float)

    heldout_t = heldout_data[treatment_column].to_numpy(dtype=float)
    heldout_y = heldout_data[outcome_column].to_numpy(dtype=float)
    metrics = {
        "outer_fold": fold,
        "n_train": int(len(fit_data)),
        "n_test": int(len(heldout_data)),
        "n_x_effect_topics": int(x_fit.shape[1]),
        "n_w_adjustment_inputs": int(w_fit.shape[1]),
        "n_w_treatment_topics": int(len(transforms["treatment"].topic_ids)),
        "n_w_outcome_topics": int(len(transforms["outcome"].topic_ids)),
        "include_stacked_nuisance_in_w": bool(config.include_stacked_nuisance_in_w),
        "ate_estimate": float(np.mean(tau)),
        "ite_standard_deviation": float(np.std(tau)),
        "r_loss_with_stage1_nuisance": _r_loss(
            heldout_y, heldout_t, outcome_prediction, propensity, tau
        ),
        "treatment_nuisance": nuisance_metrics(
            heldout_t,
            propensity,
            binary=True,
            calibration_config=calibration_config,
        ),
        "outcome_nuisance": nuisance_metrics(
            heldout_y,
            outcome_prediction,
            binary=str(outcome_type).lower() != "continuous",
            calibration_config=calibration_config,
        ),
        "oracle_metrics_included": False,
    }
    score_frame = _heldout_score_frame(heldout_ids, fold, x_heldout, w_heldout, transforms)
    feature_rows = _feature_manifest_rows(discovery, transforms)
    if config.include_stacked_nuisance_in_w:
        feature_rows.extend(
            [
                {
                    "bank": "stacked_nuisance",
                    "model_role": "W_adjustment",
                    "topic_id": name,
                    "source_column_index": index,
                    "retained": True,
                    "terms": [],
                }
                for index, name in enumerate(
                    ("stage1_stacked_propensity", "stage1_stacked_outcome")
                )
            ]
        )

    fold_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(fold_dir / "predictions.parquet", predictions)
    _write_parquet(fold_dir / "heldout_topic_scores.parquet", score_frame)
    _write_json(fold_dir / "metrics.json", metrics)
    _write_jsonl(fold_dir / "topic_feature_manifest.jsonl", feature_rows)
    joblib.dump(
        {
            "schema_version": TOPIC_SCORE_FOREST_SCHEMA_VERSION,
            "standardize": config.standardize,
            "transforms": transforms,
        },
        fold_dir / "topic_preprocessing.joblib",
    )
    model_path: Optional[str] = None
    if config.persist_fold_models:
        joblib.dump(forest, fold_dir / "causal_forest.joblib")
        model_path = str(fold_dir / "causal_forest.joblib")
    _write_json(
        fold_dir / "fold_manifest.json",
        {
            "schema_version": TOPIC_SCORE_FOREST_SCHEMA_VERSION,
            "run_config_hash": run_config_hash,
            "outer_fold": fold,
            "fit_row_fingerprint": context["fit_row_fingerprint"],
            "heldout_row_fingerprint": context["heldout_row_fingerprint"],
            "source_fit_topic_values": str(fit_path),
            "source_heldout_topic_values": str(heldout_path),
            "source_nuisance_predictions": str(nuisance_path),
            "forest_model": model_path,
            "heldout_topics_transformed_without_refitting": True,
            "oracle_columns_consumed": False,
        },
    )
    return predictions, score_frame, metrics


def evaluate_frozen_topic_score_predictions(
    *,
    prediction_path: Path,
    oracle_frame: pd.DataFrame,
    oracle_ite_column: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Join oracle ITEs only after the non-oracle prediction file is frozen."""
    prediction_path = Path(prediction_path)
    frozen_hash = _sha256_file(prediction_path)
    predictions = pd.read_parquet(prediction_path)
    if oracle_ite_column in predictions.columns or any(
        str(column).startswith("true_") for column in predictions.columns
    ):
        raise ValueError("The frozen model prediction artifact contains an oracle column")
    oracle = oracle_frame[["_oci_row_id", oracle_ite_column]].copy()
    if oracle["_oci_row_id"].duplicated().any():
        raise ValueError("Oracle frame contains duplicate _oci_row_id values")
    evaluated = predictions.merge(oracle, on="_oci_row_id", how="left", validate="one_to_one")
    if evaluated[oracle_ite_column].isna().any():
        raise ValueError("Oracle ITE is missing for one or more frozen predictions")

    def metrics_for(frame: pd.DataFrame) -> Dict[str, Any]:
        truth = frame[oracle_ite_column].to_numpy(dtype=float)
        estimate = frame["pred_ite_prob"].to_numpy(dtype=float)
        error = estimate - truth
        return {
            "n": int(len(frame)),
            "pearson_correlation": _safe_pearson(truth, estimate),
            "spearman_correlation": _safe_spearman(truth, estimate),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mean_error": float(np.mean(error)),
            "estimated_ate": float(np.mean(estimate)),
            "oracle_ate": float(np.mean(truth)),
            "ate_bias": float(np.mean(estimate) - np.mean(truth)),
            "estimated_ite_standard_deviation": float(np.std(estimate)),
            "oracle_ite_standard_deviation": float(np.std(truth)),
        }

    per_fold = [
        {"outer_fold": int(fold), **metrics_for(frame)}
        for fold, frame in evaluated.groupby("outer_fold", sort=True)
    ]
    payload = {
        "schema_version": TOPIC_SCORE_FOREST_SCHEMA_VERSION,
        "evaluation_is_post_hoc": True,
        "frozen_prediction_path": str(prediction_path),
        "frozen_prediction_sha256": frozen_hash,
        "oracle_ite_column": oracle_ite_column,
        "overall": metrics_for(evaluated),
        "per_fold": per_fold,
    }
    output_dir = Path(output_dir)
    _write_parquet(output_dir / "posthoc_predictions_with_oracle.parquet", evaluated)
    _write_json(output_dir / "posthoc_oracle_metrics.json", payload)
    return payload


def run_tfidf_topic_score_forest(
    *,
    dataset: pd.DataFrame,
    handoff_path: Path,
    output_dir: Path,
    treatment_column: str = "treatment_indicator",
    outcome_column: str = "outcome_indicator",
    outcome_type: str = "binary",
    id_columns: Sequence[str] = ("patient_id",),
    oracle_ite_column: Optional[str] = "true_ite_prob",
    config: Optional[TopicScoreForestConfig] = None,
    forest_factory: Callable[..., Any] = CausalForestHead,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the complete outer-CV direct topic-score forest experiment."""
    config = config or TopicScoreForestConfig()
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = Path(handoff_path)

    data = dataset.reset_index(drop=True).copy()
    data["_oci_row_id"] = np.arange(len(data), dtype=int)
    required = {treatment_column, outcome_column}
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required modeling columns: {missing_columns}")

    # Intentionally exclude text and every oracle column. The model consumes the
    # already-frozen Stage 1 topic arrays plus observed outer-training labels.
    safe_columns = ["_oci_row_id", treatment_column, outcome_column]
    safe_columns.extend(column for column in id_columns if column in data.columns)
    model_data = data[safe_columns].copy()
    handoff_rows = _read_handoff(handoff_path)
    contexts = _validate_outer_contexts(handoff_rows, model_data["_oci_row_id"])
    stage1_hashes = sorted({str(row["stage1_config_hash"]) for row in contexts})
    run_config_hash = stable_hash(
        {
            "schema_version": TOPIC_SCORE_FOREST_SCHEMA_VERSION,
            "config": asdict(config),
            "stage1_config_hashes": stage1_hashes,
            "treatment_column": treatment_column,
            "outcome_column": outcome_column,
            "outcome_type": outcome_type,
            "row_fingerprint": row_set_fingerprint(model_data["_oci_row_id"]),
        }
    )
    _write_json(
        output_dir / "run_config.json",
        {
            "schema_version": TOPIC_SCORE_FOREST_SCHEMA_VERSION,
            "run_config_hash": run_config_hash,
            "handoff_path": str(handoff_path),
            "stage1_config_hashes": stage1_hashes,
            "config": asdict(config),
            "topic_roles": {
                "treatment": "W_adjustment",
                "outcome": "W_adjustment",
                "effect": "X_heterogeneity",
            },
            "raw_text_consumed_by_stage2": False,
            "oracle_columns_consumed_by_modeling": False,
        },
    )

    prediction_frames: List[pd.DataFrame] = []
    score_frames: List[pd.DataFrame] = []
    fold_metrics: List[Dict[str, Any]] = []
    for context in contexts:
        predictions, scores, metrics = _fit_outer_fold(
            model_data=model_data,
            context=context,
            handoff_path=handoff_path,
            output_dir=output_dir,
            treatment_column=treatment_column,
            outcome_column=outcome_column,
            outcome_type=outcome_type,
            id_columns=id_columns,
            config=config,
            run_config_hash=run_config_hash,
            forest_factory=forest_factory,
            force=force,
        )
        prediction_frames.append(predictions)
        score_frames.append(scores)
        fold_metrics.append(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values("_oci_row_id")
    if len(predictions) != len(model_data) or predictions["_oci_row_id"].duplicated().any():
        raise RuntimeError("Outer-fold predictions do not cover every row exactly once")
    if any(str(column).startswith("true_") for column in predictions.columns):
        raise RuntimeError("An oracle column entered the model prediction artifact")
    heldout_scores = pd.concat(score_frames, ignore_index=True).sort_values("_oci_row_id")
    prediction_path = output_dir / "topic_score_predictions.parquet"
    score_path = output_dir / "heldout_topic_scores.parquet"
    _write_parquet(prediction_path, predictions)
    _write_parquet(score_path, heldout_scores)
    _write_jsonl(output_dir / "outer_metrics.jsonl", fold_metrics)
    frozen_hash = _sha256_file(prediction_path)

    oracle_metrics: Optional[Dict[str, Any]] = None
    if oracle_ite_column and oracle_ite_column in data.columns:
        # This is deliberately after the complete prediction artifact exists and
        # has been hashed. No oracle value is visible in any fold modeling call.
        oracle_metrics = evaluate_frozen_topic_score_predictions(
            prediction_path=prediction_path,
            oracle_frame=data[["_oci_row_id", oracle_ite_column]],
            oracle_ite_column=oracle_ite_column,
            output_dir=output_dir,
        )

    summary = {
        "schema_version": TOPIC_SCORE_FOREST_SCHEMA_VERSION,
        "run_config_hash": run_config_hash,
        "prediction_path": str(prediction_path),
        "prediction_sha256_before_oracle_join": frozen_hash,
        "heldout_topic_score_path": str(score_path),
        "n_rows": int(len(predictions)),
        "n_outer_folds": int(len(contexts)),
        "oracle_evaluation_performed": oracle_metrics is not None,
        "posthoc_oracle_metrics_path": (
            str(output_dir / "posthoc_oracle_metrics.json") if oracle_metrics else None
        ),
    }
    _write_json(output_dir / "run_summary.json", summary)
    logger.info("Saved topic-score forest predictions to %s", prediction_path)
    return {**summary, "oracle_metrics": oracle_metrics, "fold_metrics": fold_metrics}
