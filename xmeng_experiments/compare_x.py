"""Compare true vs extracted PD-L1 (X) from dataset_with_extraction.parquet.

Outputs a confusion-matrix table and a bar-chart plot.

Usage:
  python xmeng_experiments/compare_x.py \
      --dataset   .../dataset.parquet \
      --extracted .../dataset_with_extraction.parquet \
      --output-dir /tmp/compare_x
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CATEGORIES = ["<1%", "1-49%", "≥50%"]


def load_and_align(dataset_path: str, extracted_path: str) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)[["patient_id", "true_pdl1_expression"]]
    de = pd.read_parquet(extracted_path)[["patient_id", "llm_extracted_pdl1_expression"]]
    merged = df.merge(de, on="patient_id", how="inner")
    merged = merged.rename(columns={
        "true_pdl1_expression":          "true_X",
        "llm_extracted_pdl1_expression": "est_X",
    })
    # treat 'unknown' as missing
    merged["est_X"] = merged["est_X"].where(merged["est_X"] != "unknown", other=np.nan)
    return merged


def print_table(df: pd.DataFrame):
    n_total   = len(df)
    n_missing = df["est_X"].isna().sum()
    n_valid   = n_total - n_missing
    df_v = df.dropna(subset=["est_X"])

    overall_acc = (df_v["true_X"] == df_v["est_X"]).mean()

    print(f"\nExtraction summary  (n={n_total})")
    print(f"  Valid extractions : {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  Missing/unknown   : {n_missing} ({100*n_missing/n_total:.1f}%)")
    print(f"  Overall accuracy  : {overall_acc:.4f}")

    print("\nPer-category accuracy:")
    print(f"  {'Category':<12}  {'N_true':>7}  {'N_correct':>10}  {'Accuracy':>9}")
    for cat in CATEGORIES:
        mask = df_v["true_X"] == cat
        n_cat = mask.sum()
        n_correct = (df_v.loc[mask, "est_X"] == cat).sum()
        acc = n_correct / n_cat if n_cat > 0 else float("nan")
        print(f"  {cat:<12}  {n_cat:>7}  {n_correct:>10}  {acc:>9.4f}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    cats_ext = CATEGORIES + ["missing"]
    df_v2 = df.copy()
    df_v2["est_X"] = df_v2["est_X"].fillna("missing")
    header = f"{'':12}" + "".join(f"  {c:>9}" for c in cats_ext)
    print("  " + header)
    for true_cat in CATEGORIES:
        row_str = f"  {true_cat:<12}"
        for pred_cat in cats_ext:
            n = ((df_v2["true_X"] == true_cat) & (df_v2["est_X"] == pred_cat)).sum()
            row_str += f"  {n:>9}"
        print(row_str)


def plot(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: stacked bar — for each true category, proportion of predicted categories
    df_v = df.dropna(subset=["est_X"])
    bottom = np.zeros(len(CATEGORIES))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    pred_cats = CATEGORIES + ["missing"]
    for i, pred_cat in enumerate(pred_cats):
        vals = []
        for true_cat in CATEGORIES:
            mask = df["true_X"] == true_cat
            n_true = mask.sum()
            df_cat = df[mask].copy()
            df_cat["est_X_filled"] = df_cat["est_X"].fillna("missing")
            n_pred = (df_cat["est_X_filled"] == pred_cat).sum()
            vals.append(n_pred / n_true if n_true > 0 else 0)
        axes[0].bar(CATEGORIES, vals, bottom=bottom, label=pred_cat,
                    color=colors[i % len(colors)], alpha=0.85)
        bottom += np.array(vals)
    axes[0].set_xlabel("True PD-L1 category")
    axes[0].set_ylabel("Proportion of patients")
    axes[0].set_title("Predicted category breakdown\n(by true category)")
    axes[0].legend(title="Predicted", fontsize=8)
    axes[0].set_ylim(0, 1.05)

    # Right: count comparison
    true_counts = df["true_X"].value_counts().reindex(CATEGORIES, fill_value=0)
    df_filled = df.copy()
    df_filled["est_X"] = df_filled["est_X"].fillna("missing")
    est_counts  = df_filled["est_X"].value_counts().reindex(CATEGORIES, fill_value=0)
    x = np.arange(len(CATEGORIES))
    w = 0.35
    axes[1].bar(x - w/2, true_counts.values, w, label="True X", color="#4C72B0", alpha=0.85)
    axes[1].bar(x + w/2, est_counts.values,  w, label="Extracted X", color="#DD8452", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CATEGORIES)
    axes[1].set_xlabel("PD-L1 category")
    axes[1].set_ylabel("Count")
    axes[1].set_title("True vs extracted PD-L1\ncategory counts")
    axes[1].legend()

    fig.suptitle("True X vs Extracted X (oracle-definition LLM extraction)", fontsize=11)
    fig.tight_layout()
    out_path = output_dir / "compare_x.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True,
                        help="dataset.parquet with true_pdl1_expression")
    parser.add_argument("--extracted", required=True,
                        help="dataset_with_extraction.parquet with llm_extracted_pdl1_expression")
    parser.add_argument("--output-dir", default="/tmp/compare_x")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_align(args.dataset, args.extracted)
    print_table(df)
    plot(df, out)


if __name__ == "__main__":
    main()
