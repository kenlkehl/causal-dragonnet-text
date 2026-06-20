#!/usr/bin/env python
"""Build age-only logistic nuisance predictions for oracle R-learner runs.

This utility assumes the agent has already identified age as the sole
confounder. It fits two one-feature logistic regressions:

* propensity: P[T = 1 | age]
* outcome nuisance: E[Y | age]

The output includes:

* ``nuisance_oof_predictions.parquet``: inner-cross-fit discovery-set nuisance
  predictions in the format consumed by
  ``run_oracle_agentic_attention_r_stage_only.py``.
* ``outer_oof_predictions.parquet``: one honest outer-heldout prediction per
  patient for reporting nuisance performance.
* ``age_only_nuisance_dataset.parquet``: the original dataset plus the
  outer-heldout age-only nuisance predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = Path(__file__).resolve()
for candidate in (SCRIPT_PATH.parents[1], SCRIPT_PATH.parents[2]):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from oci.utils.calibration import (  # noqa: E402
    binary_calibration_metrics,
    clip_probability,
)

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset parquet/csv path or directory containing dataset.parquet.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--age-column", default="true_age")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Optional sample size. If used, pass the same value to downstream "
            "R-stage-only runs so _oci_row_id values align."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    return parser


def _resolve_dataset_file(path: str | Path) -> Path:
    value = Path(path)
    if value.is_file():
        return value
    if not value.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {value}")
    if not value.is_dir():
        raise ValueError(f"Dataset path is neither a file nor a directory: {value}")

    preferred = [
        "dataset.parquet",
        "data.parquet",
        "patients.parquet",
        "dataset.csv",
        "data.csv",
    ]
    for name in preferred:
        candidate = value / name
        if candidate.exists():
            return candidate

    matches = sorted(value.glob("*.parquet")) + sorted(value.glob("*.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No parquet/csv dataset file found under {value}")
    preview = ", ".join(str(match.name) for match in matches[:8])
    raise ValueError(
        f"Multiple candidate dataset files found under {value}: {preview}. "
        "Pass the exact file path."
    )


def _read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset file extension: {path.suffix}")


def _prepare_dataset(args: argparse.Namespace, dataset_file: Path) -> pd.DataFrame:
    df = _read_dataset(dataset_file)
    required = [args.age_column, args.treatment_column, args.outcome_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required column(s): " + ", ".join(missing))

    if args.sample_size is not None and args.sample_size < len(df):
        df = df.sample(
            n=args.sample_size,
            random_state=args.seed + args.repeat_index,
        )
    df = df.reset_index(drop=True).copy()
    df["_oci_row_id"] = np.arange(len(df), dtype=int)

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[required].isna().any().any():
        bad = {
            column: int(df[column].isna().sum())
            for column in required
            if int(df[column].isna().sum()) > 0
        }
        raise ValueError(f"Required columns contain non-numeric/missing values: {bad}")

    return df


def _bounded_fold_count(requested: int, n_rows: int, name: str) -> int:
    if requested < 2:
        raise ValueError(f"{name} must be >= 2")
    if n_rows < 2:
        raise ValueError("At least two rows are required")
    return min(int(requested), int(n_rows))


def _age_matrix(df: pd.DataFrame, age_column: str) -> np.ndarray:
    return df[age_column].to_numpy(dtype=float).reshape(-1, 1)


def _fit_binary_age_model(
    train_df: pd.DataFrame,
    *,
    age_column: str,
    target_column: str,
    max_iter: int,
    regularization_c: float,
):
    x_train = _age_matrix(train_df, age_column)
    y_train = train_df[target_column].to_numpy(dtype=int)
    if np.unique(y_train).size < 2:
        model = DummyClassifier(strategy="constant", constant=int(y_train[0]))
    else:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=float(regularization_c),
                        max_iter=int(max_iter),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    model.fit(x_train, y_train)
    return model


def _predict_positive_probability(model: Any, df: pd.DataFrame, age_column: str) -> np.ndarray:
    probs = model.predict_proba(_age_matrix(df, age_column))
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size == 0 and hasattr(model, "named_steps"):
        classes = np.asarray(getattr(model.named_steps.get("logistic"), "classes_", []))
    matches = np.where(classes == 1)[0]
    if matches.size == 0:
        return np.zeros(len(df), dtype=float)
    return clip_probability(probs[:, int(matches[0])])


def _safe_roc_auc(y_true: Iterable[Any], y_score: Iterable[Any]) -> Optional[float]:
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(score)
    if int(mask.sum()) < 2 or np.unique(y[mask]).size < 2:
        return None
    try:
        return float(roc_auc_score(y[mask], score[mask]))
    except ValueError:
        return None


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _corr_or_none(left: pd.Series, right: pd.Series, method: str = "pearson") -> Optional[float]:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 2:
        return None
    if frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    return _finite_or_none(frame["left"].corr(frame["right"], method=method))


def _add_true_marginal_outcome(df: pd.DataFrame) -> pd.DataFrame:
    required = {"true_treatment_prob", "true_y0_prob", "true_y1_prob"}
    if not required.issubset(df.columns):
        return df
    out = df.copy()
    e_true = out["true_treatment_prob"].to_numpy(dtype=float)
    y0_true = out["true_y0_prob"].to_numpy(dtype=float)
    y1_true = out["true_y1_prob"].to_numpy(dtype=float)
    out["true_marginal_outcome_prob"] = e_true * y1_true + (1.0 - e_true) * y0_true
    return out


def _prediction_frame(
    heldout_df: pd.DataFrame,
    *,
    outer_fold: int,
    nuisance_fold: Optional[int],
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    treatment_column: str,
    outcome_column: str,
    age_column: str,
) -> pd.DataFrame:
    y = heldout_df[outcome_column].to_numpy(dtype=float)
    t = heldout_df[treatment_column].to_numpy(dtype=float)
    y_residual = y - m_hat
    t_residual = t - e_hat
    frame = pd.DataFrame(
        {
            "_oci_row_id": heldout_df["_oci_row_id"].to_numpy(dtype=int),
            "outer_fold": int(outer_fold),
            "age": heldout_df[age_column].to_numpy(dtype=float),
            "e_hat": e_hat,
            "e_hat_raw": e_hat,
            "m_hat": m_hat,
            "m_hat_raw": m_hat,
            "y_residual": y_residual,
            "t_residual": t_residual,
            "r_loss_at_zero_tau": y_residual**2,
            "nuisance_fold": np.nan if nuisance_fold is None else int(nuisance_fold),
            treatment_column: t,
            outcome_column: y,
        }
    )
    passthrough = [
        "patient_id",
        "true_treatment_prob",
        "true_outcome_prob",
        "true_y0_prob",
        "true_y1_prob",
        "true_ite_prob",
        "true_age",
        "true_pdl1_expression",
        "true_marginal_outcome_prob",
    ]
    for column in passthrough:
        if column in heldout_df.columns and column not in frame.columns:
            frame[column] = heldout_df[column].to_numpy()
    return frame


def _fit_and_predict(
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    propensity_model = _fit_binary_age_model(
        train_df,
        age_column=args.age_column,
        target_column=args.treatment_column,
        max_iter=args.max_iter,
        regularization_c=args.regularization_c,
    )
    outcome_model = _fit_binary_age_model(
        train_df,
        age_column=args.age_column,
        target_column=args.outcome_column,
        max_iter=args.max_iter,
        regularization_c=args.regularization_c,
    )
    return {
        "e_hat": _predict_positive_probability(propensity_model, heldout_df, args.age_column),
        "m_hat": _predict_positive_probability(outcome_model, heldout_df, args.age_column),
    }


def _metrics_for_predictions(
    frame: pd.DataFrame,
    *,
    treatment_column: str,
    outcome_column: str,
    prefix: str,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        f"{prefix}_n_rows": int(len(frame)),
        f"{prefix}_propensity_auroc": _safe_roc_auc(frame[treatment_column], frame["e_hat"]),
        f"{prefix}_outcome_auroc": _safe_roc_auc(frame[outcome_column], frame["m_hat"]),
    }
    metrics.update(
        binary_calibration_metrics(
            frame[treatment_column],
            frame["e_hat"],
            prefix=f"{prefix}_propensity",
        )
    )
    metrics.update(
        binary_calibration_metrics(
            frame[outcome_column],
            frame["m_hat"],
            prefix=f"{prefix}_outcome",
        )
    )
    metrics[f"{prefix}_propensity_observed_mean"] = _finite_or_none(
        frame[treatment_column].mean()
    )
    metrics[f"{prefix}_outcome_observed_mean"] = _finite_or_none(frame[outcome_column].mean())
    if "true_treatment_prob" in frame.columns:
        metrics[f"{prefix}_true_propensity_corr"] = _corr_or_none(
            frame["true_treatment_prob"],
            frame["e_hat"],
        )
        metrics[f"{prefix}_true_propensity_spearman_corr"] = _corr_or_none(
            frame["true_treatment_prob"],
            frame["e_hat"],
            method="spearman",
        )
        metrics[f"{prefix}_true_propensity_mae"] = _finite_or_none(
            np.mean(np.abs(frame["true_treatment_prob"] - frame["e_hat"]))
        )
    if "true_marginal_outcome_prob" in frame.columns:
        metrics[f"{prefix}_true_marginal_outcome_corr"] = _corr_or_none(
            frame["true_marginal_outcome_prob"],
            frame["m_hat"],
        )
        metrics[f"{prefix}_true_marginal_outcome_spearman_corr"] = _corr_or_none(
            frame["true_marginal_outcome_prob"],
            frame["m_hat"],
            method="spearman",
        )
        metrics[f"{prefix}_true_marginal_outcome_mae"] = _finite_or_none(
            np.mean(np.abs(frame["true_marginal_outcome_prob"] - frame["m_hat"]))
        )
    return metrics


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_file = _resolve_dataset_file(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_dataset(args, dataset_file)
    df = _add_true_marginal_outcome(df)
    outer_folds = _bounded_fold_count(args.n_folds, len(df), "--n-folds")
    logger.info("Loaded %s rows from %s", len(df), dataset_file)

    outer_splitter = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    nuisance_frames: List[pd.DataFrame] = []
    outer_oof_frames: List[pd.DataFrame] = []
    fold_metric_rows: List[Dict[str, Any]] = []

    for outer_fold, (train_pos, test_pos) in enumerate(outer_splitter.split(df), start=1):
        discovery_df = df.iloc[np.asarray(train_pos)].reset_index(drop=True)
        heldout_df = df.iloc[np.asarray(test_pos)].reset_index(drop=True)
        inner_folds = _bounded_fold_count(
            args.nuisance_folds,
            len(discovery_df),
            "--nuisance-folds",
        )
        logger.info(
            "Outer fold %s/%s: discovery=%s heldout=%s inner_nuisance_folds=%s",
            outer_fold,
            outer_folds,
            len(discovery_df),
            len(heldout_df),
            inner_folds,
        )

        inner_splitter = KFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=10_000 + outer_fold,
        )
        fold_nuisance_frames = []
        for nuisance_fold, (fit_pos, inner_heldout_pos) in enumerate(
            inner_splitter.split(discovery_df),
            start=1,
        ):
            fit_df = discovery_df.iloc[np.asarray(fit_pos)].reset_index(drop=True)
            inner_heldout_df = discovery_df.iloc[np.asarray(inner_heldout_pos)].reset_index(
                drop=True
            )
            preds = _fit_and_predict(fit_df, inner_heldout_df, args)
            fold_nuisance_frames.append(
                _prediction_frame(
                    inner_heldout_df,
                    outer_fold=outer_fold,
                    nuisance_fold=nuisance_fold,
                    e_hat=preds["e_hat"],
                    m_hat=preds["m_hat"],
                    treatment_column=args.treatment_column,
                    outcome_column=args.outcome_column,
                    age_column=args.age_column,
                )
            )

        fold_nuisance = pd.concat(fold_nuisance_frames, ignore_index=True).sort_values(
            "_oci_row_id"
        )
        nuisance_frames.append(fold_nuisance)

        outer_preds = _fit_and_predict(discovery_df, heldout_df, args)
        outer_oof = _prediction_frame(
            heldout_df,
            outer_fold=outer_fold,
            nuisance_fold=None,
            e_hat=outer_preds["e_hat"],
            m_hat=outer_preds["m_hat"],
            treatment_column=args.treatment_column,
            outcome_column=args.outcome_column,
            age_column=args.age_column,
        )
        outer_oof_frames.append(outer_oof)

        fold_metric_rows.append(
            {
                "outer_fold": outer_fold,
                **_metrics_for_predictions(
                    outer_oof,
                    treatment_column=args.treatment_column,
                    outcome_column=args.outcome_column,
                    prefix="outer_oof",
                ),
                **_metrics_for_predictions(
                    fold_nuisance,
                    treatment_column=args.treatment_column,
                    outcome_column=args.outcome_column,
                    prefix="r_stage_nuisance",
                ),
            }
        )

    nuisance_df = (
        pd.concat(nuisance_frames, ignore_index=True)
        .sort_values(["outer_fold", "_oci_row_id"])
        .reset_index(drop=True)
    )
    outer_oof_df = (
        pd.concat(outer_oof_frames, ignore_index=True)
        .sort_values("_oci_row_id")
        .reset_index(drop=True)
    )
    dataset_with_outer_oof = df.merge(
        outer_oof_df[
            [
                "_oci_row_id",
                "outer_fold",
                "e_hat",
                "m_hat",
                "e_hat_raw",
                "m_hat_raw",
                "y_residual",
                "t_residual",
                "r_loss_at_zero_tau",
            ]
        ].rename(
            columns={
                "outer_fold": "age_only_outer_fold",
                "e_hat": "age_only_e_hat",
                "m_hat": "age_only_m_hat",
                "e_hat_raw": "age_only_e_hat_raw",
                "m_hat_raw": "age_only_m_hat_raw",
                "y_residual": "age_only_y_residual",
                "t_residual": "age_only_t_residual",
                "r_loss_at_zero_tau": "age_only_r_loss_at_zero_tau",
            }
        ),
        on="_oci_row_id",
        how="left",
        validate="one_to_one",
    )

    nuisance_path = output_dir / "nuisance_oof_predictions.parquet"
    outer_oof_path = output_dir / "outer_oof_predictions.parquet"
    dataset_path = output_dir / "age_only_nuisance_dataset.parquet"
    fold_metrics_path = output_dir / "fold_metrics.csv"
    metrics_path = output_dir / "metrics.json"
    config_path = output_dir / "config.json"

    nuisance_df.to_parquet(nuisance_path, index=False)
    outer_oof_df.to_parquet(outer_oof_path, index=False)
    dataset_with_outer_oof.to_parquet(dataset_path, index=False)

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    fold_metrics_df.to_parquet(output_dir / "fold_metrics.parquet", index=False)

    metrics = {
        "dataset_file": str(dataset_file),
        "n_rows": int(len(df)),
        "n_folds": int(outer_folds),
        "nuisance_folds": int(args.nuisance_folds),
        "age_column": args.age_column,
        "treatment_column": args.treatment_column,
        "outcome_column": args.outcome_column,
        "outer_oof": _metrics_for_predictions(
            outer_oof_df,
            treatment_column=args.treatment_column,
            outcome_column=args.outcome_column,
            prefix="outer_oof",
        ),
        "r_stage_nuisance": _metrics_for_predictions(
            nuisance_df,
            treatment_column=args.treatment_column,
            outcome_column=args.outcome_column,
            prefix="r_stage_nuisance",
        ),
        "artifacts": {
            "nuisance_oof_predictions": str(nuisance_path),
            "outer_oof_predictions": str(outer_oof_path),
            "age_only_nuisance_dataset": str(dataset_path),
            "fold_metrics": str(fold_metrics_path),
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.n_folds < 2:
        parser.error("--n-folds must be >= 2")
    if args.nuisance_folds < 2:
        parser.error("--nuisance-folds must be >= 2")
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be >= 1")
    if args.max_iter < 1:
        parser.error("--max-iter must be >= 1")
    if args.regularization_c <= 0:
        parser.error("--regularization-c must be > 0")

    metrics = _run(args)
    outer = metrics["outer_oof"]
    nuisance = metrics["r_stage_nuisance"]
    print("Age-only logistic nuisance complete")
    print(f"Rows: {metrics['n_rows']}")
    print(f"Output: {Path(args.output_dir).resolve()}")
    print(
        "Outer OOF: "
        f"treatment AUROC={outer['outer_oof_propensity_auroc']}, "
        f"treatment ECE={outer['outer_oof_propensity_ece']}, "
        f"outcome AUROC={outer['outer_oof_outcome_auroc']}, "
        f"outcome ECE={outer['outer_oof_outcome_ece']}"
    )
    print(
        "R-stage nuisance file: "
        f"rows={nuisance['r_stage_nuisance_n_rows']}, "
        f"treatment AUROC={nuisance['r_stage_nuisance_propensity_auroc']}, "
        f"outcome AUROC={nuisance['r_stage_nuisance_outcome_auroc']}"
    )


if __name__ == "__main__":
    main()
