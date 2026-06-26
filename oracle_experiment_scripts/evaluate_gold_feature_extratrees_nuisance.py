#!/usr/bin/env python3
"""Evaluate ExtraTrees nuisance models on gold synthetic feature values.

This script uses true_<feature_name> columns from a synthetic dataset and
role-tagged feature specs from metadata.json. It reports cross-fitted treatment
and outcome nuisance performance, plus the same unweighted R pseudo-target
diagnostics used by the multi-model agentic BoW path.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit ExtraTrees nuisance models on oracle true_* feature values."
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory or Parquet file.")
    parser.add_argument(
        "--feature-set",
        default="both",
        choices=["confounders", "all", "both"],
        help=(
            "Gold feature values to use. 'confounders' uses true confounders only; "
            "'all' uses true confounders plus true effect modifiers; 'both' reports both."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outcome-type", default="binary", choices=["binary", "continuous"])
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="ExtraTrees max_features value. Use 'none' for sklearn None.",
    )
    parser.add_argument("--model-n-jobs", type=int, default=1)
    parser.add_argument("--e-clip", type=float, default=0.01)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    parquet_path = resolve_parquet_file(dataset_path)
    metadata_path = resolve_metadata_file(dataset_path, parquet_path)
    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    metadata = json.loads(metadata_path.read_text())

    feature_sets = ["confounders", "all"] if args.feature_set == "both" else [args.feature_set]
    results = []
    for feature_set in feature_sets:
        specs = feature_specs(metadata, feature_set)
        result = evaluate_feature_set(
            df=df,
            specs=specs,
            feature_set=feature_set,
            outcome_type=args.outcome_type,
            folds=args.folds,
            seed=args.seed,
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_features=parse_max_features(args.max_features),
            model_n_jobs=args.model_n_jobs,
            e_clip=args.e_clip,
        )
        results.append(result)

    print(json.dumps(results, indent=2, default=json_default))
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2, default=json_default))
    if args.output_csv:
        write_csv(Path(args.output_csv), results)


def resolve_parquet_file(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("dataset.parquet", "dataset_with_extraction.parquet"):
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No dataset parquet found under {path}")


def resolve_metadata_file(dataset_path: Path, parquet_path: Path) -> Path:
    candidates = []
    if dataset_path.is_dir():
        candidates.append(dataset_path / "metadata.json")
    candidates.append(parquet_path.parent / "metadata.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No metadata.json found for {dataset_path}")


def feature_specs(metadata: Dict[str, Any], feature_set: str) -> List[Dict[str, Any]]:
    if feature_set == "confounders":
        specs = list(metadata.get("confounders", []))
    elif feature_set == "all":
        specs = metadata.get("features")
        if specs is None:
            specs = list(metadata.get("confounders", [])) + list(
                metadata.get("effect_modifiers", [])
            )
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return dedupe_specs(specs)


def dedupe_specs(specs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    order = []
    for raw in specs:
        spec = dict(raw)
        name = str(spec["name"])
        if name not in by_name:
            by_name[name] = spec
            order.append(name)
            continue
        roles = []
        for role in list(by_name[name].get("roles") or []) + list(spec.get("roles") or []):
            if role not in roles:
                roles.append(role)
        by_name[name]["roles"] = roles
    return [by_name[name] for name in order]


def parse_max_features(value: str) -> Any:
    normalized = str(value).strip().lower()
    if normalized in {"none", "null"}:
        return None
    try:
        if "." in normalized:
            return float(normalized)
        return int(normalized)
    except ValueError:
        return value


def encoded_feature_names(specs: Sequence[Dict[str, Any]]) -> List[str]:
    names = []
    for spec in specs:
        name = spec["name"]
        if spec["type"] == "continuous":
            names.extend([f"{name}_normalized", f"{name}_missing"])
        else:
            for cat in list(spec.get("categories") or [])[1:]:
                names.append(f"{name}_{cat}")
            names.append(f"{name}_missing")
    return names or ["intercept"]


def evaluate_feature_set(
    *,
    df: pd.DataFrame,
    specs: List[Dict[str, Any]],
    feature_set: str,
    outcome_type: str,
    folds: int,
    seed: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: Any,
    model_n_jobs: int,
    e_clip: float,
) -> Dict[str, Any]:
    missing = [f"true_{spec['name']}" for spec in specs if f"true_{spec['name']}" not in df.columns]
    if missing:
        raise ValueError(f"Missing ground-truth columns for {feature_set}: {missing}")

    y = df["outcome_indicator"].to_numpy(dtype=float)
    t = df["treatment_indicator"].to_numpy(dtype=int)

    split_items = binary_split_items(t, folds, seed)
    e_hat = np.full(len(df), np.nan, dtype=float)
    m_hat = np.full(len(df), np.nan, dtype=float)

    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(split_items, start=1):
        x_train, x_test, feature_names = build_fold_features(df, specs, train_idx, test_idx)

        prop_model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=10_000 + seed + fold,
            n_jobs=model_n_jobs,
        )
        prop_model.fit(x_train, t[train_idx])
        e_hat[test_idx] = prop_model.predict_proba(x_test)[:, 1]

        if outcome_type == "continuous":
            outcome_model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=20_000 + seed + fold,
                n_jobs=model_n_jobs,
            )
            outcome_model.fit(x_train, y[train_idx])
            m_hat[test_idx] = outcome_model.predict(x_test)
        else:
            outcome_model = ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=20_000 + seed + fold,
                n_jobs=model_n_jobs,
            )
            outcome_model.fit(x_train, y[train_idx].astype(int))
            m_hat[test_idx] = outcome_model.predict_proba(x_test)[:, 1]

        fold_metrics.append(
            fold_metric_row(
                fold=fold,
                test_idx=test_idx,
                y=y,
                t=t,
                e_hat=e_hat,
                m_hat=m_hat,
                outcome_type=outcome_type,
                n_features=len(feature_names),
            )
        )

    e_hat = np.clip(e_hat, e_clip, 1.0 - e_clip)
    y_resid = y - m_hat
    t_resid = t.astype(float) - e_hat
    pseudo_target = y_resid / t_resid
    tau_hat = crossfit_tau(
        df=df,
        specs=specs,
        pseudo_target=pseudo_target,
        folds=folds,
        seed=seed,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        model_n_jobs=model_n_jobs,
    )
    r_loss = (y_resid - tau_hat * t_resid) ** 2
    r_loss_zero = y_resid**2

    result: Dict[str, Any] = {
        "feature_set": feature_set,
        "n_rows": int(len(df)),
        "n_specs": int(len(specs)),
        "spec_names": [spec["name"] for spec in specs],
        "n_encoded_features": int(len(encoded_feature_names(specs))),
        "folds": int(folds),
        "treatment_auroc": safe_roc_auc(t, e_hat),
        "treatment_brier": finite_or_none(brier_score_loss(t, e_hat)),
        "treatment_log_loss": finite_or_none(log_loss(t, e_hat)),
        "r_loss_mean": finite_or_none(np.mean(r_loss)),
        "r_loss_at_zero_tau_mean": finite_or_none(np.mean(r_loss_zero)),
        "r_loss_relative_improvement": finite_or_none(1.0 - np.mean(r_loss) / np.mean(r_loss_zero)),
        "tau_hat_pseudo_target_corr": safe_corr(tau_hat, pseudo_target),
        "fold_metrics": fold_metrics,
    }
    if outcome_type == "continuous":
        result["outcome_rmse"] = finite_or_none(np.sqrt(mean_squared_error(y, m_hat)))
    else:
        result["outcome_auroc"] = safe_roc_auc(y, m_hat)
        result["outcome_brier"] = finite_or_none(brier_score_loss(y, m_hat))
        result["outcome_log_loss"] = finite_or_none(log_loss(y, m_hat))

    if "true_treatment_prob" in df.columns:
        result["treatment_true_prob_corr"] = safe_corr(
            e_hat,
            df["true_treatment_prob"].to_numpy(dtype=float),
        )
    if "true_outcome_prob" in df.columns:
        result["outcome_true_prob_corr"] = safe_corr(
            m_hat,
            df["true_outcome_prob"].to_numpy(dtype=float),
        )
    if "true_ite_prob" in df.columns:
        true_ite = df["true_ite_prob"].to_numpy(dtype=float)
        result["tau_hat_true_ite_corr"] = safe_corr(tau_hat, true_ite)
        result["pseudo_target_true_ite_corr"] = safe_corr(pseudo_target, true_ite)
    return result


def build_fold_features(
    df: pd.DataFrame,
    specs: Sequence[Dict[str, Any]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    train_parts = []
    test_parts = []
    names = []

    for spec in specs:
        name = spec["name"]
        col = f"true_{name}"
        if spec["type"] == "continuous":
            train_raw = pd.to_numeric(df.iloc[train_idx][col], errors="coerce").to_numpy(dtype=float)
            test_raw = pd.to_numeric(df.iloc[test_idx][col], errors="coerce").to_numpy(dtype=float)
            missing_train = ~np.isfinite(train_raw)
            missing_test = ~np.isfinite(test_raw)
            observed = train_raw[~missing_train]
            mean = float(np.mean(observed)) if len(observed) else 0.0
            std = float(np.std(observed)) if len(observed) else 1.0
            std = max(std, 1e-6)
            train_value = np.where(missing_train, 0.0, (train_raw - mean) / std)
            test_value = np.where(missing_test, 0.0, (test_raw - mean) / std)
            train_parts.append(train_value.reshape(-1, 1))
            test_parts.append(test_value.reshape(-1, 1))
            train_parts.append(missing_train.astype(float).reshape(-1, 1))
            test_parts.append(missing_test.astype(float).reshape(-1, 1))
            names.extend([f"{name}_normalized", f"{name}_missing"])
        else:
            categories = [str(cat) for cat in spec.get("categories") or []]
            train_values = df.iloc[train_idx][col].astype("string")
            test_values = df.iloc[test_idx][col].astype("string")
            missing_train = train_values.isna().to_numpy()
            missing_test = test_values.isna().to_numpy()
            for cat in categories[1:]:
                train_parts.append((train_values.astype(str).to_numpy() == cat).astype(float).reshape(-1, 1))
                test_parts.append((test_values.astype(str).to_numpy() == cat).astype(float).reshape(-1, 1))
                names.append(f"{name}_{cat}")
            train_parts.append(missing_train.astype(float).reshape(-1, 1))
            test_parts.append(missing_test.astype(float).reshape(-1, 1))
            names.append(f"{name}_missing")

    if not train_parts:
        return (
            np.zeros((len(train_idx), 1), dtype=float),
            np.zeros((len(test_idx), 1), dtype=float),
            ["intercept"],
        )
    return np.hstack(train_parts), np.hstack(test_parts), names


def crossfit_tau(
    *,
    df: pd.DataFrame,
    specs: List[Dict[str, Any]],
    pseudo_target: np.ndarray,
    folds: int,
    seed: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: Any,
    model_n_jobs: int,
) -> np.ndarray:
    oof = np.full(len(df), np.nan, dtype=float)
    splitter = KFold(n_splits=min(folds, len(df)), shuffle=True, random_state=30_000 + seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df), start=1):
        x_train, x_test, _ = build_fold_features(df, specs, train_idx, test_idx)
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=30_000 + seed + fold,
            n_jobs=model_n_jobs,
        )
        model.fit(x_train, pseudo_target[train_idx])
        oof[test_idx] = model.predict(x_test)
    return oof


def binary_split_items(labels: np.ndarray, folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    values, counts = np.unique(labels.astype(int), return_counts=True)
    if len(values) >= 2 and int(np.min(counts)) >= 2:
        n_splits = max(2, min(int(folds), int(np.min(counts)), len(labels)))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [
            (np.asarray(train_idx), np.asarray(test_idx))
            for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels)
        ]
    n_splits = max(2, min(int(folds), len(labels)))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (np.asarray(train_idx), np.asarray(test_idx))
        for train_idx, test_idx in splitter.split(np.zeros(len(labels)))
    ]


def fold_metric_row(
    *,
    fold: int,
    test_idx: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    outcome_type: str,
    n_features: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "fold": int(fold),
        "n_test": int(len(test_idx)),
        "n_encoded_features": int(n_features),
        "treatment_auroc": safe_roc_auc(t[test_idx], e_hat[test_idx]),
        "treatment_brier": finite_or_none(brier_score_loss(t[test_idx], e_hat[test_idx])),
    }
    if outcome_type == "continuous":
        row["outcome_rmse"] = finite_or_none(
            np.sqrt(mean_squared_error(y[test_idx], m_hat[test_idx]))
        )
    else:
        row["outcome_auroc"] = safe_roc_auc(y[test_idx], m_hat[test_idx])
        row["outcome_brier"] = finite_or_none(
            brier_score_loss(y[test_idx], m_hat[test_idx])
        )
    return row


def safe_roc_auc(y_true: np.ndarray, score: np.ndarray) -> Any:
    try:
        return finite_or_none(roc_auc_score(y_true, score))
    except ValueError:
        return None


def safe_corr(left: np.ndarray, right: np.ndarray) -> Any:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return None
    if float(np.std(left[mask])) <= 0.0 or float(np.std(right[mask])) <= 0.0:
        return None
    return finite_or_none(np.corrcoef(left[mask], right[mask])[0, 1])


def finite_or_none(value: Any) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def write_csv(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_keys = sorted(
        {
            key
            for row in results
            for key, value in row.items()
            if not isinstance(value, (list, dict))
        }
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


if __name__ == "__main__":
    main()
