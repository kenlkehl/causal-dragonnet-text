from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


DEFAULT_DATASET = (
    "synthetic_data/example_synthetic_datasets/"
    "one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet"
)


def make_tfidf_logistic_pipeline(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, args.ngram_max),
                    lowercase=True,
                    strip_accents="unicode",
                    min_df=args.min_df,
                    max_df=args.max_df,
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "logit",
                LogisticRegression(
                    C=args.logit_c,
                    max_iter=args.max_iter,
                    solver="liblinear",
                    random_state=args.seed,
                ),
            ),
        ]
    )


def top_logistic_coefficients(
    model: Pipeline,
    top_n: int,
) -> pd.DataFrame:
    terms = model.named_steps["tfidf"].get_feature_names_out()
    coef = model.named_steps["logit"].coef_.ravel()
    order = np.argsort(coef)
    selected = np.concatenate([order[:top_n], order[-top_n:][::-1]])
    return pd.DataFrame(
        {
            "feature": terms[selected],
            "coefficient": coef[selected],
            "direction": ["negative"] * top_n + ["positive"] * top_n,
        }
    )


def fit_final_nuisance_feature_tables(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    texts = df[args.text_col].fillna("").astype(str).to_numpy()
    treatment = df[args.treatment_col].astype(int).to_numpy()
    outcome = df[args.outcome_col].astype(int).to_numpy()

    prop_model = make_tfidf_logistic_pipeline(args)
    outcome_model = make_tfidf_logistic_pipeline(args)
    prop_model.fit(texts, treatment)
    outcome_model.fit(texts, outcome)

    return (
        top_logistic_coefficients(prop_model, args.top_features),
        top_logistic_coefficients(outcome_model, args.top_features),
    )


def honest_nuisance_predictions(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    texts = df[args.text_col].fillna("").astype(str).to_numpy()
    treatment = df[args.treatment_col].astype(int).to_numpy()
    outcome = df[args.outcome_col].astype(int).to_numpy()

    prop = np.full(len(df), np.nan)
    outcome_prob = np.full(len(df), np.nan)
    folds = np.full(len(df), -1)

    strata = np.array([f"{t}_{y}" for t, y in zip(treatment, outcome)])
    cv = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    for fold, (train_idx, test_idx) in enumerate(cv.split(texts, strata)):
        prop_model = make_tfidf_logistic_pipeline(args)
        outcome_model = make_tfidf_logistic_pipeline(args)

        prop_model.fit(texts[train_idx], treatment[train_idx])
        outcome_model.fit(texts[train_idx], outcome[train_idx])

        prop[test_idx] = prop_model.predict_proba(texts[test_idx])[:, 1]
        outcome_prob[test_idx] = outcome_model.predict_proba(texts[test_idx])[:, 1]
        folds[test_idx] = fold

    if np.isnan(prop).any() or np.isnan(outcome_prob).any() or (folds < 0).any():
        raise RuntimeError("Cross-fitting did not produce predictions for every row.")

    return prop, outcome_prob, folds


def hopcroft_karp(adjacency: Sequence[Sequence[int]], n_right: int) -> tuple[list[int], list[int]]:
    n_left = len(adjacency)
    pair_left = [-1] * n_left
    pair_right = [-1] * n_right
    dist = [0] * n_left

    def bfs() -> bool:
        queue: deque[int] = deque()
        found_free_path = False
        for left in range(n_left):
            if pair_left[left] == -1:
                dist[left] = 0
                queue.append(left)
            else:
                dist[left] = -1

        while queue:
            left = queue.popleft()
            for right in adjacency[left]:
                matched_left = pair_right[right]
                if matched_left == -1:
                    found_free_path = True
                elif dist[matched_left] == -1:
                    dist[matched_left] = dist[left] + 1
                    queue.append(matched_left)
        return found_free_path

    def dfs(left: int) -> bool:
        for right in adjacency[left]:
            matched_left = pair_right[right]
            if matched_left == -1 or (
                dist[matched_left] == dist[left] + 1 and dfs(matched_left)
            ):
                pair_left[left] = right
                pair_right[right] = left
                return True
        dist[left] = -1
        return False

    while bfs():
        for left in range(n_left):
            if pair_left[left] == -1:
                dfs(left)

    return pair_left, pair_right


def build_matches(
    honest: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treated = honest.loc[honest[args.treatment_col] == 1].copy()
    control = honest.loc[honest[args.treatment_col] == 0].copy()

    treated = treated.sort_values(args.id_col).reset_index(drop=False)
    control = control.sort_values(args.id_col).reset_index(drop=False)

    t_prop = treated["tfidf_propensity"].to_numpy()
    t_outcome = treated["tfidf_outcome_prob"].to_numpy()
    c_prop = control["tfidf_propensity"].to_numpy()
    c_outcome = control["tfidf_outcome_prob"].to_numpy()

    eligible_rows = []
    adjacency: list[list[int]] = []

    for i in range(len(treated)):
        prop_diff = np.abs(c_prop - t_prop[i])
        outcome_diff = np.abs(c_outcome - t_outcome[i])
        eligible = np.where(
            (prop_diff <= args.propensity_caliper)
            & (outcome_diff <= args.outcome_caliper)
        )[0]

        order = sorted(
            eligible.tolist(),
            key=lambda j: (
                float(prop_diff[j] + outcome_diff[j]),
                float(prop_diff[j]),
                int(control.loc[j, args.id_col]),
            ),
        )
        adjacency.append(order)

        for j in order:
            eligible_rows.append(
                {
                    "treated_patient_id": treated.loc[i, args.id_col],
                    "control_patient_id": control.loc[j, args.id_col],
                    "treated_row_index": int(treated.loc[i, "index"]),
                    "control_row_index": int(control.loc[j, "index"]),
                    "propensity_treated": float(t_prop[i]),
                    "propensity_control": float(c_prop[j]),
                    "outcome_prob_treated": float(t_outcome[i]),
                    "outcome_prob_control": float(c_outcome[j]),
                    "propensity_abs_diff": float(prop_diff[j]),
                    "outcome_prob_abs_diff": float(outcome_diff[j]),
                    "score_abs_diff_sum": float(prop_diff[j] + outcome_diff[j]),
                }
            )

    eligible_pairs = pd.DataFrame(eligible_rows)
    pair_left, _ = hopcroft_karp(adjacency, len(control))

    matched_rows = []
    for i, j in enumerate(pair_left):
        if j == -1:
            continue
        matched_rows.append(
            {
                "treated_patient_id": treated.loc[i, args.id_col],
                "control_patient_id": control.loc[j, args.id_col],
                "treated_row_index": int(treated.loc[i, "index"]),
                "control_row_index": int(control.loc[j, "index"]),
                "propensity_treated": float(t_prop[i]),
                "propensity_control": float(c_prop[j]),
                "outcome_prob_treated": float(t_outcome[i]),
                "outcome_prob_control": float(c_outcome[j]),
                "propensity_abs_diff": float(abs(t_prop[i] - c_prop[j])),
                "outcome_prob_abs_diff": float(abs(t_outcome[i] - c_outcome[j])),
                "score_abs_diff_sum": float(
                    abs(t_prop[i] - c_prop[j]) + abs(t_outcome[i] - c_outcome[j])
                ),
            }
        )

    matched_pairs = pd.DataFrame(matched_rows).sort_values(
        ["score_abs_diff_sum", "treated_patient_id", "control_patient_id"],
        ignore_index=True,
    )
    return eligible_pairs, matched_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit TF-IDF BoW nuisance models and match treated/control patients "
            "on honest propensity and outcome probabilities."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default="one_off_uplift")
    parser.add_argument("--id-col", default="patient_id")
    parser.add_argument("--text-col", default="clinical_text")
    parser.add_argument("--treatment-col", default="treatment_indicator")
    parser.add_argument("--outcome-col", default="outcome_indicator")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--logit-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--propensity-caliper", type=float, default=0.05)
    parser.add_argument("--outcome-caliper", type=float, default=0.05)
    parser.add_argument("--top-features", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset)
    prop, outcome_prob, folds = honest_nuisance_predictions(df, args)

    honest = df.copy()
    honest["tfidf_cv_fold"] = folds
    honest["tfidf_propensity"] = prop
    honest["tfidf_outcome_prob"] = outcome_prob
    honest["tfidf_propensity_caliper"] = args.propensity_caliper
    honest["tfidf_outcome_caliper"] = args.outcome_caliper

    eligible_pairs, matched_pairs = build_matches(honest, args)
    propensity_features, outcome_features = fit_final_nuisance_feature_tables(df, args)

    honest_path = output_dir / "honest_tfidf_nuisance_predictions.parquet"
    eligible_path = output_dir / "eligible_pairs_within_calipers.parquet"
    matched_path = output_dir / "max_one_to_one_matched_pairs.parquet"
    propensity_features_path = output_dir / "nuisance_propensity_top_features.csv"
    outcome_features_path = output_dir / "nuisance_outcome_top_features.csv"
    summary_path = output_dir / "summary.json"

    honest.to_parquet(honest_path, index=False)
    eligible_pairs.to_parquet(eligible_path, index=False)
    matched_pairs.to_parquet(matched_path, index=False)
    propensity_features.to_csv(propensity_features_path, index=False)
    outcome_features.to_csv(outcome_features_path, index=False)

    summary = {
        "dataset": str(Path(args.dataset)),
        "n_patients": int(len(honest)),
        "n_treated": int((honest[args.treatment_col] == 1).sum()),
        "n_control": int((honest[args.treatment_col] == 0).sum()),
        "text_col": args.text_col,
        "treatment_col": args.treatment_col,
        "outcome_col": args.outcome_col,
        "folds": args.folds,
        "seed": args.seed,
        "ngram_range": [1, args.ngram_max],
        "min_df": args.min_df,
        "max_df": args.max_df,
        "max_features": args.max_features,
        "propensity_caliper": args.propensity_caliper,
        "outcome_caliper": args.outcome_caliper,
        "eligible_cross_treatment_pairs": int(len(eligible_pairs)),
        "max_one_to_one_matched_pairs": int(len(matched_pairs)),
        "propensity_auroc_vs_treatment": float(
            roc_auc_score(honest[args.treatment_col], honest["tfidf_propensity"])
        ),
        "outcome_auroc_vs_outcome": float(
            roc_auc_score(honest[args.outcome_col], honest["tfidf_outcome_prob"])
        ),
        "propensity_prediction_mean": float(honest["tfidf_propensity"].mean()),
        "propensity_prediction_std": float(honest["tfidf_propensity"].std(ddof=0)),
        "outcome_prediction_mean": float(honest["tfidf_outcome_prob"].mean()),
        "outcome_prediction_std": float(honest["tfidf_outcome_prob"].std(ddof=0)),
        "outputs": {
            "honest_dataset": str(honest_path),
            "eligible_pairs": str(eligible_path),
            "matched_pairs": str(matched_path),
            "propensity_top_features": str(propensity_features_path),
            "outcome_top_features": str(outcome_features_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
