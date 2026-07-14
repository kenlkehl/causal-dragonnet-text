from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit, logit
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


EPS = 1e-5


def clipped_logit(prob: np.ndarray) -> np.ndarray:
    return logit(np.clip(prob, EPS, 1.0 - EPS))


def make_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, args.ngram_max),
        lowercase=True,
        strip_accents="unicode",
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        sublinear_tf=True,
    )


def make_pair_matrix(
    vectorizer: TfidfVectorizer,
    control_text: np.ndarray,
    treated_text: np.ndarray,
) -> csr_matrix:
    control_x = vectorizer.transform(control_text)
    treated_x = vectorizer.transform(treated_text)
    return hstack([control_x, treated_x], format="csr")


def fit_offset_logistic(
    x: csr_matrix,
    y: np.ndarray,
    offset: np.ndarray,
    alpha: float,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    n_obs, n_features = x.shape
    y = y.astype(float)
    offset = offset.astype(float)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = params[0]
        coef = params[1:]
        eta = offset + intercept + x.dot(coef)
        prob = expit(eta)
        residual = prob - y

        loss = float(np.logaddexp(0.0, eta).mean() - np.mean(y * eta))
        penalty = 0.5 * alpha * float(coef @ coef)
        grad_intercept = np.array([residual.mean()])
        grad_coef = np.asarray(x.T.dot(residual)).ravel() / n_obs + alpha * coef
        grad = np.concatenate([grad_intercept, grad_coef])
        return loss + penalty, grad

    init = np.zeros(n_features + 1)
    result = minimize(
        objective,
        init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": 1e-8, "gtol": 1e-5, "maxls": 30},
    )
    if not result.success:
        print(f"WARNING: offset logistic optimizer ended with: {result.message}")
    return result.x[1:], float(result.x[0])


def load_pair_frame(args: argparse.Namespace) -> pd.DataFrame:
    honest = pd.read_parquet(args.honest_dataset).reset_index(names="source_row_index")
    matched = pd.read_parquet(args.matched_pairs)

    treated = honest.add_prefix("treated_")
    control = honest.add_prefix("control_")

    pairs = matched.merge(
        treated,
        left_on="treated_row_index",
        right_on="treated_source_row_index",
        how="left",
        validate="one_to_one",
    ).merge(
        control,
        left_on="control_row_index",
        right_on="control_source_row_index",
        how="left",
        validate="one_to_one",
    )

    pairs["treated_patient_id"] = pairs["treated_patient_id_x"].astype(int)
    pairs["control_patient_id"] = pairs["control_patient_id_x"].astype(int)
    pairs = pairs.drop(
        columns=[
            "treated_patient_id_x",
            "control_patient_id_x",
            "treated_patient_id_y",
            "control_patient_id_y",
        ]
    )

    pairs["treated_text"] = pairs[f"treated_{args.text_col}"].fillna("").astype(str)
    pairs["control_text"] = pairs[f"control_{args.text_col}"].fillna("").astype(str)
    pairs["treated_outcome"] = pairs[f"treated_{args.outcome_col}"].astype(int)
    pairs["control_outcome"] = pairs[f"control_{args.outcome_col}"].astype(int)
    pairs["control_base_prob"] = np.clip(
        pairs["control_tfidf_outcome_prob"].astype(float), EPS, 1.0 - EPS
    )
    pairs["control_base_logit"] = clipped_logit(pairs["control_base_prob"].to_numpy())

    if {"treated_true_y1_prob", "control_true_y0_prob"}.issubset(pairs.columns):
        pairs["true_pair_delta_prob"] = (
            pairs["treated_true_y1_prob"].astype(float)
            - pairs["control_true_y0_prob"].astype(float)
        )
        pairs["true_pair_delta_logit"] = clipped_logit(
            pairs["treated_true_y1_prob"].astype(float).to_numpy()
        ) - clipped_logit(pairs["control_true_y0_prob"].astype(float).to_numpy())
    if "treated_true_ite_prob" in pairs.columns:
        pairs["true_treated_ite_prob"] = pairs["treated_true_ite_prob"].astype(float)

    return pairs


def metric_block(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, EPS, 1.0 - EPS)
    return {
        "auroc": float(roc_auc_score(y, pred)),
        "brier": float(brier_score_loss(y, pred)),
        "log_loss": float(log_loss(y, pred, labels=[0, 1])),
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(x[mask], y[mask]).statistic),
        "spearman": float(spearmanr(x[mask], y[mask]).statistic),
    }


def top_coefficients(
    vectorizer: TfidfVectorizer,
    coef: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    terms = vectorizer.get_feature_names_out()
    n_terms = len(terms)
    names = np.concatenate(
        [
            np.char.add("control::", terms.astype(str)),
            np.char.add("treated::", terms.astype(str)),
        ]
    )
    if coef.shape[0] != 2 * n_terms:
        raise ValueError("Coefficient vector does not match concatenated pair feature names.")

    order = np.argsort(coef)
    selected = np.concatenate([order[:top_n], order[-top_n:][::-1]])
    return pd.DataFrame(
        {
            "feature": names[selected],
            "coefficient": coef[selected],
            "direction": ["negative"] * top_n + ["positive"] * top_n,
        }
    )


def resolve_treated_true_columns(
    pairs: pd.DataFrame,
    requested_cols: list[str],
) -> list[str]:
    if not requested_cols:
        requested_cols = ["pdl1_expression"]

    resolved = []
    for col in requested_cols:
        candidates = [
            col,
            f"true_{col}",
            f"treated_true_{col}",
        ]
        for candidate in candidates:
            if candidate in pairs.columns:
                resolved.append(candidate)
                break
    return resolved


def summarize_by_modifier(
    pairs: pd.DataFrame,
    modifier_cols: list[str],
) -> pd.DataFrame:
    rows = []

    for modifier_col in modifier_cols:
        values = pairs[modifier_col]
        if pd.api.types.is_numeric_dtype(values):
            nonnull = values.dropna()
            if nonnull.nunique() <= 1:
                bins = values.astype(str).fillna("missing")
            else:
                bins = pd.qcut(values, q=4, duplicates="drop").astype(str)
                bins = bins.where(values.notna(), "missing")
        else:
            bins = values.astype(str).fillna("missing")

        work = pairs.assign(_modifier_bin=bins)
        for level, group in work.groupby("_modifier_bin", dropna=False):
            row = {
                "modifier": modifier_col.removeprefix("treated_true_"),
                "level_or_bin": str(level),
                "n_pairs": int(len(group)),
                "treated_observed_outcome_rate": float(group["treated_outcome"].mean()),
                "control_base_prob_mean": float(group["control_base_prob"].mean()),
                "ridge_delta_prob_mean": float(group["ridge_delta_prob"].mean()),
                "ridge_pred_prob_mean": float(group["ridge_pred_prob"].mean()),
                "offset_logit_delta_mean": float(group["offset_logit_delta"].mean()),
                "offset_logit_pred_prob_mean": float(
                    group["offset_logit_pred_prob"].mean()
                ),
            }
            if "true_treated_ite_prob" in group.columns:
                row["true_treated_ite_prob_mean"] = float(
                    group["true_treated_ite_prob"].mean()
                )
            if "true_pair_delta_prob" in group.columns:
                row["true_pair_delta_prob_mean"] = float(
                    group["true_pair_delta_prob"].mean()
                )
            rows.append(row)

    return pd.DataFrame(rows)


def cross_fit_pair_models(
    pairs: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    y = pairs["treated_outcome"].to_numpy(dtype=int)
    base_prob = pairs["control_base_prob"].to_numpy(dtype=float)
    base_logit = pairs["control_base_logit"].to_numpy(dtype=float)
    control_text = pairs["control_text"].to_numpy()
    treated_text = pairs["treated_text"].to_numpy()

    folds = np.full(len(pairs), -1)
    ridge_delta = np.full(len(pairs), np.nan)
    offset_delta = np.full(len(pairs), np.nan)
    offset_intercepts = []

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        vectorizer = make_vectorizer(args)
        vectorizer.fit(
            np.concatenate([control_text[train_idx], treated_text[train_idx]])
        )
        x_train = make_pair_matrix(
            vectorizer,
            control_text[train_idx],
            treated_text[train_idx],
        )
        x_test = make_pair_matrix(
            vectorizer,
            control_text[test_idx],
            treated_text[test_idx],
        )

        ridge = Ridge(
            alpha=args.ridge_alpha,
            fit_intercept=True,
            solver="lsqr",
            random_state=args.seed,
        )
        ridge.fit(x_train, y[train_idx] - base_prob[train_idx])
        ridge_delta[test_idx] = ridge.predict(x_test)

        coef, intercept = fit_offset_logistic(
            x_train,
            y[train_idx],
            base_logit[train_idx],
            alpha=args.logit_l2_alpha,
            max_iter=args.logit_max_iter,
        )
        offset_delta[test_idx] = intercept + x_test.dot(coef)
        offset_intercepts.append(intercept)
        folds[test_idx] = fold

    if np.isnan(ridge_delta).any() or np.isnan(offset_delta).any() or (folds < 0).any():
        raise RuntimeError("Cross-fitting did not produce predictions for every matched pair.")

    out = pairs.copy()
    out["pair_cv_fold"] = folds
    out["ridge_delta_prob"] = ridge_delta
    out["ridge_pred_prob"] = np.clip(base_prob + ridge_delta, EPS, 1.0 - EPS)
    out["ridge_delta_logit_implied"] = clipped_logit(out["ridge_pred_prob"].to_numpy()) - base_logit
    out["offset_logit_delta"] = offset_delta
    out["offset_logit_pred_logit"] = base_logit + offset_delta
    out["offset_logit_pred_prob"] = expit(out["offset_logit_pred_logit"].to_numpy())
    out["offset_logit_delta_prob_implied"] = out["offset_logit_pred_prob"] - base_prob

    summary: dict[str, object] = {
        "folds": args.folds,
        "seed": args.seed,
        "n_pairs": int(len(out)),
        "treated_outcome_rate": float(y.mean()),
        "baseline_control_prob": metric_block(y, base_prob),
        "ridge_delta_probability_model": metric_block(y, out["ridge_pred_prob"].to_numpy()),
        "offset_logit_delta_model": metric_block(y, out["offset_logit_pred_prob"].to_numpy()),
        "offset_logit_fold_intercepts": [float(x) for x in offset_intercepts],
    }

    if "true_pair_delta_prob" in out.columns:
        summary["ridge_delta_prob_vs_true_pair_delta_prob"] = safe_corr(
            out["ridge_delta_prob"].to_numpy(dtype=float),
            out["true_pair_delta_prob"].to_numpy(dtype=float),
        )
        summary["offset_implied_delta_prob_vs_true_pair_delta_prob"] = safe_corr(
            out["offset_logit_delta_prob_implied"].to_numpy(dtype=float),
            out["true_pair_delta_prob"].to_numpy(dtype=float),
        )
    if "true_treated_ite_prob" in out.columns:
        summary["ridge_delta_prob_vs_true_treated_ite_prob"] = safe_corr(
            out["ridge_delta_prob"].to_numpy(dtype=float),
            out["true_treated_ite_prob"].to_numpy(dtype=float),
        )
        summary["offset_implied_delta_prob_vs_true_treated_ite_prob"] = safe_corr(
            out["offset_logit_delta_prob_implied"].to_numpy(dtype=float),
            out["true_treated_ite_prob"].to_numpy(dtype=float),
        )

    return out, summary


def fit_final_feature_tables(
    pairs: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pairs["treated_outcome"].to_numpy(dtype=int)
    base_prob = pairs["control_base_prob"].to_numpy(dtype=float)
    base_logit = pairs["control_base_logit"].to_numpy(dtype=float)
    control_text = pairs["control_text"].to_numpy()
    treated_text = pairs["treated_text"].to_numpy()

    vectorizer = make_vectorizer(args)
    vectorizer.fit(np.concatenate([control_text, treated_text]))
    x_all = make_pair_matrix(vectorizer, control_text, treated_text)

    ridge = Ridge(
        alpha=args.ridge_alpha,
        fit_intercept=True,
        solver="lsqr",
        random_state=args.seed,
    )
    ridge.fit(x_all, y - base_prob)

    offset_coef, _ = fit_offset_logistic(
        x_all,
        y,
        base_logit,
        alpha=args.logit_l2_alpha,
        max_iter=args.logit_max_iter,
    )

    ridge_coef = np.asarray(ridge.coef_).ravel()
    return (
        top_coefficients(vectorizer, ridge_coef, args.top_features),
        top_coefficients(vectorizer, offset_coef, args.top_features),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train honest matched-pair TF-IDF uplift models."
    )
    parser.add_argument(
        "--honest-dataset",
        default="one_off_uplift/honest_tfidf_nuisance_predictions.parquet",
    )
    parser.add_argument(
        "--matched-pairs",
        default="one_off_uplift/max_one_to_one_matched_pairs.parquet",
    )
    parser.add_argument("--output-dir", default="one_off_uplift")
    parser.add_argument("--text-col", default="clinical_text")
    parser.add_argument("--outcome-col", default="outcome_indicator")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--logit-l2-alpha", type=float, default=1.0)
    parser.add_argument("--logit-max-iter", type=int, default=100)
    parser.add_argument("--top-features", type=int, default=40)
    parser.add_argument("--modifier-cols", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pair_frame(args)
    honest_predictions, summary = cross_fit_pair_models(pairs, args)
    modifier_cols = resolve_treated_true_columns(honest_predictions, args.modifier_cols)
    modifier_summary = summarize_by_modifier(honest_predictions, modifier_cols)
    ridge_features, offset_features = fit_final_feature_tables(pairs, args)

    front_columns = [
        "treated_patient_id",
        "control_patient_id",
        "treated_row_index",
        "control_row_index",
        "pair_cv_fold",
        "treated_outcome",
        "control_outcome",
        "control_base_prob",
        "control_base_logit",
        "ridge_delta_prob",
        "ridge_pred_prob",
        "ridge_delta_logit_implied",
        "offset_logit_delta",
        "offset_logit_pred_logit",
        "offset_logit_pred_prob",
        "offset_logit_delta_prob_implied",
        "true_treated_ite_prob",
        "true_pair_delta_prob",
        "true_pair_delta_logit",
    ]
    for modifier_col in modifier_cols:
        control_col = modifier_col.replace("treated_", "control_", 1)
        front_columns.extend([modifier_col, control_col])
    front_columns = [col for col in front_columns if col in honest_predictions.columns]
    honest_predictions = honest_predictions[
        front_columns + [col for col in honest_predictions.columns if col not in front_columns]
    ]

    pair_predictions_path = output_dir / "pair_uplift_honest_predictions.parquet"
    modifier_summary_path = output_dir / "pair_uplift_by_modifier.csv"
    ridge_feature_path = output_dir / "pair_uplift_ridge_top_features.csv"
    offset_feature_path = output_dir / "pair_uplift_offset_logit_top_features.csv"
    summary_path = output_dir / "pair_uplift_summary.json"

    honest_predictions.to_parquet(pair_predictions_path, index=False)
    modifier_summary.to_csv(modifier_summary_path, index=False)
    ridge_features.to_csv(ridge_feature_path, index=False)
    offset_features.to_csv(offset_feature_path, index=False)

    summary["ngram_range"] = [1, args.ngram_max]
    summary["min_df"] = args.min_df
    summary["max_df"] = args.max_df
    summary["max_features"] = args.max_features
    summary["ridge_alpha"] = args.ridge_alpha
    summary["logit_l2_alpha"] = args.logit_l2_alpha
    summary["outputs"] = {
        "pair_predictions": str(pair_predictions_path),
        "modifier_summary": str(modifier_summary_path),
        "ridge_top_features": str(ridge_feature_path),
        "offset_logit_top_features": str(offset_feature_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
