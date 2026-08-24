"""Four-arm diagnostic experiment: isolate W vs X source in CausalForestDML.

Implements the four conditions from the diagnosis plan:

  1. Oracle:     W = true_age,   X = true_pdl1_expression
  2. Est W only: W = est_age,    X = true_pdl1_expression
  3. Est X only: W = true_age,   X = est_pdl1_expression
  4. Full est:   W = est_age,    X = est_pdl1_expression

All conditions use identical CV splits and CausalForestDML hyperparameters.
The primary metric is Pearson correlation with true_ite_prob; Spearman is secondary.

Estimated features default to dataset_with_extraction.parquet (oracle-definition
extraction, a proxy until the full Stage I/II pipeline runs). The true values are
taken directly from true_age and true_pdl1_expression in dataset.parquet.

Usage:
  python xmeng_experiments/four_arm_diagnostic.py \\
      --dataset .../one_confounder_one_effect_modifier_nsclc_with_structured/dataset.parquet \\
      [--estimated-features .../dataset_with_extraction.parquet] \\
      [--output-dir /tmp/four_arm] \\
      [--n-folds 5] [--n-repeats 3] [--seed 42]
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oci.config import ExplicitFeatureSpec
from oci.models.explicit_feature_featurizer import get_raw_explicit_features
from oci.models.causal_forest_head import CausalForestHead


# ---------------------------------------------------------------------------
# Feature specification for this fixed DGP
# ---------------------------------------------------------------------------

AGE_SPEC = ExplicitFeatureSpec(
    name="age",
    type="continuous",
    roles=["confounder"],
)

PDL1_SPEC = ExplicitFeatureSpec(
    name="pdl1_expression",
    type="categorical",
    categories=["<1%", "1-49%", "≥50%"],
    roles=["effect_modifier"],
)

ALL_SPECS = [AGE_SPEC, PDL1_SPEC]


# ---------------------------------------------------------------------------
# Feature dict builders
# ---------------------------------------------------------------------------

def _true_age_dicts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build feature dicts for age from true_age column."""
    result = []
    for _, row in df.iterrows():
        val = row.get("true_age")
        missing = val is None or (isinstance(val, float) and np.isnan(val))
        result.append({"age": None if missing else float(val), "age_missing": missing})
    return result


def _true_pdl1_dicts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build feature dicts for pdl1_expression from true_pdl1_expression column."""
    result = []
    for _, row in df.iterrows():
        val = row.get("true_pdl1_expression")
        missing = val is None or (isinstance(val, float) and np.isnan(val))
        result.append({
            "pdl1_expression": None if missing else str(val),
            "pdl1_expression_missing": missing,
        })
    return result


def _est_age_dicts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build feature dicts for age from extracted age column.

    Supports both llm_extracted_age (from dataset_with_extraction.parquet)
    and explicit_feat_age (from full OCI Stage I/II output).
    """
    col = "llm_extracted_age" if "llm_extracted_age" in df.columns else "explicit_feat_age"
    missing_col = f"{col}_missing" if f"{col}_missing" in df.columns else None
    result = []
    for _, row in df.iterrows():
        val = row.get(col)
        if missing_col:
            missing = bool(row.get(missing_col, False))
        else:
            missing = val is None or (isinstance(val, float) and np.isnan(val))
        result.append({"age": None if missing else float(val), "age_missing": missing})
    return result


def _est_pdl1_dicts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build feature dicts for pdl1_expression from extracted pdl1 column.

    Supports both llm_extracted_pdl1_expression (dataset_with_extraction.parquet)
    and explicit_feat_pdl1_expression (full OCI Stage I/II output).
    Treats 'unknown' as missing.
    """
    col = ("llm_extracted_pdl1_expression"
           if "llm_extracted_pdl1_expression" in df.columns
           else "explicit_feat_pdl1_expression")
    missing_col = f"{col}_missing" if f"{col}_missing" in df.columns else None
    result = []
    for _, row in df.iterrows():
        val = row.get(col)
        if missing_col:
            missing = bool(row.get(missing_col, False))
        else:
            missing = (val is None
                       or (isinstance(val, float) and np.isnan(val))
                       or str(val).lower() == "unknown")
        result.append({
            "pdl1_expression": None if missing else str(val),
            "pdl1_expression_missing": missing,
        })
    return result


def _merge_dicts(
    age_dicts: List[Dict], pdl1_dicts: List[Dict]
) -> List[Dict[str, Any]]:
    """Merge per-patient age and pdl1 dicts into one dict per patient."""
    return [{**a, **p} for a, p in zip(age_dicts, pdl1_dicts)]


# ---------------------------------------------------------------------------
# Encoding: produce W (confounder) and X (effect modifier) matrices
# ---------------------------------------------------------------------------

def _encode(
    feature_dicts: List[Dict[str, Any]],
    specs: List[ExplicitFeatureSpec],
    role: str,
    continuous_means: Optional[Dict[str, float]] = None,
    continuous_stds: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[np.ndarray], List[str], Dict[str, float], Dict[str, float]]:
    """Encode feature_dicts for the given role. Returns (matrix_or_None, names, means, stds)."""
    means = {} if continuous_means is None else continuous_means
    stds = {} if continuous_stds is None else continuous_stds

    rows, names = get_raw_explicit_features(
        feature_dicts,
        specs,
        continuous_means=means,
        continuous_stds=stds,
        role=role,
    )
    if not names:
        return None, names, means, stds

    mat = np.array(rows, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[1] == 0:
        return None, names, means, stds
    return mat, names, means, stds


# ---------------------------------------------------------------------------
# One cross-validated run for a single condition
# ---------------------------------------------------------------------------

def _run_condition(
    df: pd.DataFrame,
    age_dicts: List[Dict],
    pdl1_dicts: List[Dict],
    treatment_col: str,
    outcome_col: str,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_estimators: int = 200,
    min_samples_leaf: int = 10,
) -> np.ndarray:
    """Return out-of-fold tau predictions for this condition."""
    feature_dicts = _merge_dicts(age_dicts, pdl1_dicts)
    tau_oof = np.full(len(df), np.nan)

    for train_idx, test_idx in folds:
        train_dicts = [feature_dicts[i] for i in train_idx]
        test_dicts  = [feature_dicts[i] for i in test_idx]

        # Build W (confounder=age) and X (effect_modifier=pdl1)
        W_train, _, w_means, w_stds = _encode(train_dicts, ALL_SPECS, "confounder")
        X_train, _, x_means, x_stds = _encode(train_dicts, ALL_SPECS, "effect_modifier")

        W_test, _, _, _ = _encode(test_dicts, ALL_SPECS, "confounder",
                                   continuous_means=w_means, continuous_stds=w_stds)
        X_test, _, _, _ = _encode(test_dicts, ALL_SPECS, "effect_modifier",
                                   continuous_means=x_means, continuous_stds=x_stds)

        train_T = df[treatment_col].values[train_idx].astype(float)
        train_Y = df[outcome_col].values[train_idx].astype(float)

        forest = CausalForestHead(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            honest=True,
            inference=False,
            random_state=42,
        )
        forest.fit(X_train, train_T, train_Y, W=W_train)
        preds = forest.predict(X_test, return_ci=False)
        tau_oof[test_idx] = preds["tau_pred"]

    return tau_oof


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_four_arm(
    dataset_path: str,
    estimated_features_path: Optional[str],
    output_dir: str,
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    n_estimators: int = 200,
    min_samples_leaf: int = 10,
) -> Dict:
    df = pd.read_parquet(dataset_path)
    print(f"Dataset: {len(df)} patients")
    print(f"  true ITE mean={df['true_ite_prob'].mean():.4f}  "
          f"std={df['true_ite_prob'].std():.4f}")

    has_estimated = estimated_features_path is not None
    df_est = None
    if has_estimated:
        df_est = pd.read_parquet(estimated_features_path)
        # Align by patient_id if present; keep all extracted feature columns
        EST_PREFIXES = ("explicit_feat_", "llm_extracted_")
        if "patient_id" in df.columns and "patient_id" in df_est.columns:
            est_cols = ["patient_id"] + [
                c for c in df_est.columns
                if any(c.startswith(p) for p in EST_PREFIXES)
            ]
            df_est = df.merge(df_est[est_cols], on="patient_id", how="left")
        else:
            df_est = df_est.reset_index(drop=True)
        print(f"Estimated features: {estimated_features_path}")
        est_age_cols = [c for c in df_est.columns if "age" in c and
                        ("explicit" in c or "llm_extracted" in c)]
        est_pdl1_cols = [c for c in df_est.columns if "pdl1" in c and
                         ("explicit" in c or "llm_extracted" in c)]
        print(f"  age cols: {est_age_cols}")
        print(f"  pdl1 cols: {est_pdl1_cols}")
    else:
        print("No estimated features provided — running Condition 1 (oracle) only.")

    true_ite = df["true_ite_prob"].values
    treatment_col = "treatment_indicator"
    outcome_col = "outcome_indicator"

    # Pre-build feature dicts (once, outside the repeat loop)
    true_age_dicts  = _true_age_dicts(df)
    true_pdl1_dicts = _true_pdl1_dicts(df)
    if has_estimated:
        est_age_dicts  = _est_age_dicts(df_est)
        est_pdl1_dicts = _est_pdl1_dicts(df_est)

    conditions = {
        "oracle":     (true_age_dicts,  true_pdl1_dicts),
    }
    if has_estimated:
        conditions["est_W_only"] = (est_age_dicts,  true_pdl1_dicts)
        conditions["est_X_only"] = (true_age_dicts, est_pdl1_dicts)
        conditions["full_est"]   = (est_age_dicts,  est_pdl1_dicts)

    # Collect results across repeats
    all_results: Dict[str, List[Dict]] = {k: [] for k in conditions}

    for rep in range(n_repeats):
        rep_seed = seed + rep * 1000
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)
        folds = list(kf.split(df))

        for cond_name, (age_d, pdl1_d) in conditions.items():
            print(f"\n[repeat {rep+1}/{n_repeats}] Condition: {cond_name}")
            tau_oof = _run_condition(
                df, age_d, pdl1_d,
                treatment_col, outcome_col,
                folds,
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
            )
            valid = ~np.isnan(tau_oof)
            pearson  = float(stats.pearsonr(tau_oof[valid], true_ite[valid])[0])
            spearman = float(stats.spearmanr(tau_oof[valid], true_ite[valid])[0])
            ate_bias = float(abs(tau_oof[valid].mean() - true_ite[valid].mean()))
            ate_pred = float(tau_oof[valid].mean())
            all_results[cond_name].append({
                "pearson":  pearson,
                "spearman": spearman,
                "ate_bias": ate_bias,
                "ate_pred": ate_pred,
            })
            print(f"  pearson={pearson:.4f}  spearman={spearman:.4f}  "
                  f"ate_bias={ate_bias:.4f}  ate_pred={ate_pred:.4f}")

    # Aggregate
    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std across repeats)")
    print("=" * 70)
    print(f"{'Condition':<20} {'Pearson':>10} {'Spearman':>10} {'ATE_bias':>10}")
    print("-" * 70)

    summary = {}
    for cond_name, reps in all_results.items():
        p  = [r["pearson"]  for r in reps]
        sp = [r["spearman"] for r in reps]
        ab = [r["ate_bias"] for r in reps]
        summary[cond_name] = {
            "pearson_mean":  float(np.mean(p)),
            "pearson_std":   float(np.std(p)),
            "spearman_mean": float(np.mean(sp)),
            "spearman_std":  float(np.std(sp)),
            "ate_bias_mean": float(np.mean(ab)),
            "ate_bias_std":  float(np.std(ab)),
        }
        print(
            f"  {cond_name:<18} "
            f"  {np.mean(p):+.4f}±{np.std(p):.4f}"
            f"  {np.mean(sp):+.4f}±{np.std(sp):.4f}"
            f"  {np.mean(ab):.4f}±{np.std(ab):.4f}"
        )

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "dataset": dataset_path,
            "estimated_features": estimated_features_path,
            "n_folds": n_folds,
            "n_repeats": n_repeats,
            "seed": seed,
            "n_estimators": n_estimators,
            "min_samples_leaf": min_samples_leaf,
        },
        "summary": summary,
        "per_repeat": all_results,
    }
    out_path = Path(output_dir) / "four_arm_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Four-arm W/X source diagnostic for CausalForestDML"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to dataset.parquet (must contain true_age, true_pdl1_expression, "
             "true_ite_prob, treatment_indicator, outcome_indicator)"
    )
    parser.add_argument(
        "--estimated-features", default=None,
        help="Path to parquet with explicit_feat_age and explicit_feat_pdl1_expression "
             "columns (e.g. dataset_with_extraction.parquet). If omitted, only Condition 1 runs."
    )
    parser.add_argument(
        "--output-dir", default="/tmp/four_arm_diagnostic",
        help="Directory to write four_arm_results.json"
    )
    parser.add_argument("--n-folds",   type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--n-estimators",    type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    args = parser.parse_args()

    run_four_arm(
        dataset_path=args.dataset,
        estimated_features_path=args.estimated_features,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        seed=args.seed,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
    )


if __name__ == "__main__":
    main()
