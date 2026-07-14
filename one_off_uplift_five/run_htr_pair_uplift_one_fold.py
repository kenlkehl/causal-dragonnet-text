from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from oci.inference.multi_model_pair_uplift import fit_htr_pair_uplift_train_test
from oci.models.extractor_factory import create_feature_extractor


EPS = 1e-5


KNOWN_MODIFIERS: Dict[str, Sequence[str]] = {
    "histology_type": (
        "histology",
        "adenocarcinoma",
        "squamous",
    ),
    "egfr_mutation_status": (
        "egfr",
        "mutation",
        "molecular",
        "wild-type",
        "positive",
        "negative",
        "unknown",
    ),
    "baseline_nlr": (
        "neutrophil",
        "lymphocyte",
        "neutrophil-to-lymphocyte",
        "nlr",
    ),
    "brain_metastases_status": (
        "brain metast",
        "intracranial",
        "mri brain",
        "cns",
    ),
    "baseline_hemoglobin": (
        "hemoglobin",
        "hgb",
        "cbc",
        "anemia",
    ),
}


@dataclass
class OneOffHTRRunner:
    device: torch.device
    arch: Any
    training: Any
    avf_config: Any

    def __post_init__(self) -> None:
        self.config = SimpleNamespace(architecture=self.arch, training=self.training)

    def _effect_epochs(self) -> int:
        return max(1, int(getattr(self.avf_config, "effect_epochs", 1) or 1))

    def _cleanup_model(self, model: torch.nn.Module) -> None:
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _create_extractor(self) -> torch.nn.Module:
        return create_feature_extractor(
            extractor_type="hierarchical_transformer",
            device=self.device,
            htr_sentence_model=self.arch.htr_sentence_model,
            htr_freeze_sentence_encoder=self.arch.htr_freeze_sentence_encoder,
            htr_chunk_size_words=self.arch.htr_chunk_size_words,
            htr_chunk_overlap_words=self.arch.htr_chunk_overlap_words,
            htr_max_chunks=self.arch.htr_max_chunks,
            htr_max_chunk_length=self.arch.htr_max_chunk_length,
            htr_num_layers=self.arch.htr_num_layers,
            htr_num_heads=self.arch.htr_num_heads,
            htr_transformer_dim=self.arch.htr_transformer_dim,
            htr_dropout=self.arch.htr_dropout,
            htr_projection_dim=self.arch.htr_projection_dim,
            htr_hash_embedding_dim=self.arch.htr_hash_embedding_dim,
            htr_sentence_encoder_batch_size=self.arch.htr_sentence_encoder_batch_size,
            htr_sentence_encoder_backend=self.arch.htr_sentence_encoder_backend,
            htr_sentence_pooling=self.arch.htr_sentence_pooling,
            htr_normalize_sentence_embeddings=self.arch.htr_normalize_sentence_embeddings,
            htr_trainable_sentence_encoder_layers=(
                self.arch.htr_trainable_sentence_encoder_layers
            ),
            htr_role_attention=self.arch.htr_role_attention,
            htr_w_attention_heads=self.arch.htr_w_attention_heads,
            htr_x_attention_heads=self.arch.htr_x_attention_heads,
        )


def clipped_logit(prob: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(prob, dtype=float), EPS, 1.0 - EPS))


def finite_corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(x[mask], y[mask]).statistic),
        "spearman": float(spearmanr(x[mask], y[mask]).statistic),
    }


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float | int | None]:
    mask = np.isfinite(y) & np.isfinite(pred)
    out: Dict[str, float | int | None] = {"n": int(mask.sum())}
    if int(mask.sum()) == 0:
        out.update({"auroc": None, "brier": None, "log_loss": None})
        return out
    target = y[mask].astype(int)
    prob = np.clip(pred[mask], EPS, 1.0 - EPS)
    out["auroc"] = (
        float(roc_auc_score(target, prob)) if len(np.unique(target)) == 2 else None
    )
    out["brier"] = float(brier_score_loss(target, prob))
    out["log_loss"] = float(log_loss(target, prob, labels=[0, 1]))
    return out


def maybe_clip_texts(texts: Iterable[str], max_chars: int) -> List[str]:
    if max_chars <= 0:
        return [str(text or "") for text in texts]
    return [str(text or "")[:max_chars] for text in texts]


def build_outer_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    strata = (
        df["treatment_indicator"].astype(int).astype(str)
        + "_"
        + df["outcome_indicator"].astype(int).astype(str)
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, strata))
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def summarize_pair_prediction_bases(pair_frame: pd.DataFrame) -> pd.DataFrame:
    if pair_frame.empty:
        return pd.DataFrame(columns=["candidate_row_id", "base_prob_mean"])
    work = pair_frame.copy()
    if "pair_pred_prob" in work.columns and "pair_delta_logit" in work.columns:
        work["base_prob"] = expit(
            clipped_logit(work["pair_pred_prob"].to_numpy(dtype=float))
            - work["pair_delta_logit"].to_numpy(dtype=float)
        )
    elif "base_prob" not in work.columns:
        return pd.DataFrame(columns=["candidate_row_id", "base_prob_mean"])
    return (
        work.groupby("candidate_row_id", as_index=False)["base_prob"]
        .mean()
        .rename(columns={"base_prob": "base_prob_mean"})
    )


def compile_patterns(phrases: Sequence[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(phrase) for phrase in phrases), re.IGNORECASE)


def attention_modifier_summary(attention: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if attention.empty:
        return pd.DataFrame(
            columns=["modifier", "attention_rows", "mean_attention", "top_examples"]
        )
    text_cols = [
        col
        for col in ["attended_token_summary", "highlighted_chunk_text", "chunk_text"]
        if col in attention.columns
    ]
    for modifier, phrases in KNOWN_MODIFIERS.items():
        pattern = compile_patterns(phrases)
        matched = []
        for _, row in attention.iterrows():
            haystack = " ".join(str(row.get(col, "") or "") for col in text_cols)
            if pattern.search(haystack):
                matched.append(row)
        if not matched:
            rows.append(
                {
                    "modifier": modifier,
                    "attention_rows": 0,
                    "mean_attention": None,
                    "top_examples": "",
                }
            )
            continue
        matched_df = pd.DataFrame(matched)
        examples = []
        for _, row in matched_df.sort_values(
            "attention", ascending=False, na_position="last"
        ).head(3).iterrows():
            token_summary = str(row.get("attended_token_summary", "") or "").strip()
            chunk = str(row.get("chunk_text", "") or "").replace("\n", " ")
            examples.append(token_summary or chunk[:180])
        rows.append(
            {
                "modifier": modifier,
                "attention_rows": int(len(matched_df)),
                "mean_attention": float(pd.to_numeric(matched_df["attention"]).mean()),
                "top_examples": " | ".join(examples),
            }
        )
    return pd.DataFrame(rows)


def top_attention_rows(attention: pd.DataFrame, n: int) -> pd.DataFrame:
    if attention.empty:
        return pd.DataFrame()
    cols = [
        col
        for col in [
            "outer_fold",
            "inner_fold",
            "pair_side",
            "row_id",
            "candidate_row_id",
            "control_row_id",
            "pair_delta_logit",
            "pair_pred_prob",
            "pair_base_prob",
            "attention",
            "attended_token_summary",
            "chunk_text",
        ]
        if col in attention.columns
    ]
    return attention.sort_values("attention", ascending=False).head(n)[cols].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=(
            "synthetic_data/example_synthetic_datasets/"
            "five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
        ),
    )
    parser.add_argument(
        "--honest-nuisance",
        default="one_off_uplift_five/honest_tfidf_nuisance_predictions.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="one_off_uplift_five/htr_pair_uplift_one_fold",
    )
    parser.add_argument("--text-col", default="clinical_text")
    parser.add_argument("--text-max-chars", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--effect-folds", type=int, default=2)
    parser.add_argument("--effect-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--propensity-caliper", type=float, default=0.05)
    parser.add_argument("--outcome-caliper", type=float, default=0.05)
    parser.add_argument("--max-controls-per-candidate", type=int, default=3)
    parser.add_argument("--nearest-fallback-controls", type=int, default=1)
    parser.add_argument("--max-attention-pairs", type=int, default=24)
    parser.add_argument("--attention-top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--sentence-model",
        default=(
            "/home/klkehl/.cache/huggingface/hub/models--unsloth--bge-small-en-v1.5/"
            "snapshots/7382f1122c10708a1faa0bbe548674a14b1ffe7e"
        ),
    )
    parser.add_argument("--sentence-backend", default="transformers")
    parser.add_argument("--sentence-pooling", default="token_attention")
    parser.add_argument("--freeze-sentence-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-size-words", type=int, default=96)
    parser.add_argument("--chunk-overlap-words", type=int, default=24)
    parser.add_argument("--max-chunks", type=int, default=96)
    parser.add_argument("--max-chunk-length", type=int, default=128)
    parser.add_argument("--htr-layers", type=int, default=1)
    parser.add_argument("--htr-heads", type=int, default=4)
    parser.add_argument("--htr-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--hash-embedding-dim", type=int, default=256)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--role-attention", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    honest = pd.read_parquet(args.honest_nuisance).reset_index(drop=True)
    if "patient_id" in df.columns and "patient_id" in honest.columns:
        nuisance = honest[
            ["patient_id", "tfidf_propensity", "tfidf_outcome_prob"]
        ].drop_duplicates("patient_id")
        df = df.merge(nuisance, on="patient_id", how="left", validate="one_to_one")
    else:
        df["tfidf_propensity"] = honest["tfidf_propensity"].to_numpy(dtype=float)
        df["tfidf_outcome_prob"] = honest["tfidf_outcome_prob"].to_numpy(dtype=float)
    df["_oci_row_id"] = np.arange(len(df), dtype=int)

    train_idx, test_idx = build_outer_split(df, args.test_size, args.seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    texts = maybe_clip_texts(df[args.text_col].fillna("").astype(str), args.text_max_chars)
    texts_train = [texts[int(pos)] for pos in train_idx]
    texts_test = [texts[int(pos)] for pos in test_idx]

    arch = SimpleNamespace(
        causal_head_hidden_outcome_dim=args.hidden_dim,
        htr_sentence_model=args.sentence_model,
        htr_freeze_sentence_encoder=bool(args.freeze_sentence_encoder),
        htr_chunk_size_words=args.chunk_size_words,
        htr_chunk_overlap_words=args.chunk_overlap_words,
        htr_max_chunks=args.max_chunks,
        htr_max_chunk_length=args.max_chunk_length,
        htr_num_layers=args.htr_layers,
        htr_num_heads=args.htr_heads,
        htr_transformer_dim=args.htr_dim,
        htr_dropout=0.05,
        htr_projection_dim=args.projection_dim,
        htr_hash_embedding_dim=args.hash_embedding_dim,
        htr_sentence_encoder_batch_size=args.encoder_batch_size,
        htr_sentence_encoder_backend=args.sentence_backend,
        htr_sentence_pooling=args.sentence_pooling,
        htr_normalize_sentence_embeddings=True,
        htr_trainable_sentence_encoder_layers=0,
        htr_role_attention=bool(args.role_attention),
        htr_w_attention_heads=1,
        htr_x_attention_heads=1,
    )
    training = SimpleNamespace(
        batch_size=args.batch_size,
        effect_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    avf_config = SimpleNamespace(
        effect_folds=args.effect_folds,
        effect_epochs=args.effect_epochs,
        attention_top_k_chunks=args.attention_top_k,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runner = OneOffHTRRunner(
        device=device,
        arch=arch,
        training=training,
        avf_config=avf_config,
    )

    result = fit_htr_pair_uplift_train_test(
        runner=runner,
        train_df=train_df,
        test_df=test_df,
        texts_train=texts_train,
        texts_test=texts_test,
        y_train=train_df["outcome_indicator"].to_numpy(dtype=float),
        t_train=train_df["treatment_indicator"].to_numpy(dtype=float),
        e_train=train_df["tfidf_propensity"].to_numpy(dtype=float),
        m_train=train_df["tfidf_outcome_prob"].to_numpy(dtype=float),
        e_test=test_df["tfidf_propensity"].to_numpy(dtype=float),
        m_test=test_df["tfidf_outcome_prob"].to_numpy(dtype=float),
        outer_fold=1,
        effect_folds=args.effect_folds,
        propensity_caliper=args.propensity_caliper,
        outcome_caliper=args.outcome_caliper,
        max_controls_per_candidate=args.max_controls_per_candidate,
        nearest_fallback_controls=args.nearest_fallback_controls,
        max_attention_pairs=args.max_attention_pairs,
    )

    pred = test_df[
        [
            "_oci_row_id",
            "patient_id",
            "treatment_indicator",
            "outcome_indicator",
            "true_y0_prob",
            "true_y1_prob",
            "true_ite_prob",
            "true_histology_type",
            "true_egfr_mutation_status",
            "true_baseline_nlr",
            "true_brain_metastases_status",
            "true_baseline_hemoglobin",
            "tfidf_propensity",
            "tfidf_outcome_prob",
        ]
    ].copy()
    pred["htr_delta_logit"] = result.test_delta_logit
    pred["htr_treated_outcome_prob"] = result.test_pred_prob
    pred["candidate_controls_mean"] = result.test_n_controls

    pair_predictions = result.prediction_frame.copy()
    base_summary = summarize_pair_prediction_bases(pair_predictions)
    pred = pred.merge(
        base_summary,
        left_on="_oci_row_id",
        right_on="candidate_row_id",
        how="left",
    ).drop(columns=["candidate_row_id"], errors="ignore")
    fallback_outcome_prob = pd.Series(
        test_df["tfidf_outcome_prob"].to_numpy(dtype=float),
        index=pred.index,
    )
    pred["base_prob_mean"] = pred["base_prob_mean"].fillna(fallback_outcome_prob)
    pred["htr_delta_prob"] = pred["htr_treated_outcome_prob"] - pred["base_prob_mean"]
    pred["true_delta_logit"] = clipped_logit(pred["true_y1_prob"]) - clipped_logit(
        pred["true_y0_prob"]
    )

    treated_mask = pred["treatment_indicator"].to_numpy(dtype=int) == 1
    baseline_for_treated = pred["base_prob_mean"].fillna(fallback_outcome_prob)
    summary: Dict[str, Any] = {
        "dataset": args.dataset,
        "honest_nuisance": args.honest_nuisance,
        "text_col": args.text_col,
        "text_max_chars": int(args.text_max_chars),
        "seed": int(args.seed),
        "device": str(device),
        "outer_fold": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_treated_rows": int(treated_mask.sum()),
            "test_control_rows": int((~treated_mask).sum()),
        },
        "htr": {
            "sentence_model": args.sentence_model,
            "sentence_backend": args.sentence_backend,
            "sentence_pooling": args.sentence_pooling,
            "freeze_sentence_encoder": bool(args.freeze_sentence_encoder),
            "effect_folds": int(args.effect_folds),
            "effect_epochs": int(args.effect_epochs),
            "batch_size": int(args.batch_size),
            "chunk_size_words": int(args.chunk_size_words),
            "chunk_overlap_words": int(args.chunk_overlap_words),
            "max_chunks": int(args.max_chunks),
            "max_chunk_length": int(args.max_chunk_length),
            "htr_layers": int(args.htr_layers),
            "htr_dim": int(args.htr_dim),
        },
        "pair_matching": result.feature_importance.get("pair_matching", {}),
        "inner_evidence": result.evidence_rows,
        "train_oof_metrics": result.metrics,
        "outer_test_actual_treated": {
            "baseline_matched_control_prob": binary_metrics(
                pred.loc[treated_mask, "outcome_indicator"].to_numpy(dtype=float),
                baseline_for_treated.loc[treated_mask].to_numpy(dtype=float),
            ),
            "htr_uplift_prob": binary_metrics(
                pred.loc[treated_mask, "outcome_indicator"].to_numpy(dtype=float),
                pred.loc[treated_mask, "htr_treated_outcome_prob"].to_numpy(dtype=float),
            ),
        },
        "outer_test_delta_vs_truth": {
            "delta_logit_vs_true_delta_logit": finite_corr(
                pred["true_delta_logit"].to_numpy(dtype=float),
                pred["htr_delta_logit"].to_numpy(dtype=float),
            ),
            "delta_prob_vs_true_ite_prob": finite_corr(
                pred["true_ite_prob"].to_numpy(dtype=float),
                pred["htr_delta_prob"].to_numpy(dtype=float),
            ),
        },
        "prediction_distribution": {
            "delta_logit_mean": float(np.nanmean(result.test_delta_logit)),
            "delta_logit_std": float(np.nanstd(result.test_delta_logit)),
            "treated_prob_mean": float(np.nanmean(result.test_pred_prob)),
            "treated_prob_std": float(np.nanstd(result.test_pred_prob)),
            "candidate_controls_mean": float(np.nanmean(result.test_n_controls)),
        },
    }

    attention = pd.DataFrame(result.attention_rows)
    modifier_attention = attention_modifier_summary(attention)
    top_attention = top_attention_rows(attention, n=40)
    summary["attention"] = {
        "rows": int(len(attention)),
        "rows_with_token_summary": int(
            attention.get("attended_token_summary", pd.Series(dtype=object))
            .fillna("")
            .astype(str)
            .str.len()
            .gt(0)
            .sum()
        )
        if not attention.empty
        else 0,
        "known_modifier_hit_rows": {
            str(row["modifier"]): int(row["attention_rows"])
            for _, row in modifier_attention.iterrows()
        },
    }

    pred.to_parquet(out_dir / "htr_pair_uplift_outer_test_predictions.parquet", index=False)
    pair_predictions.to_parquet(
        out_dir / "htr_pair_uplift_pair_predictions.parquet",
        index=False,
    )
    if not attention.empty:
        attention.to_parquet(out_dir / "htr_pair_uplift_attention.parquet", index=False)
    modifier_attention.to_csv(out_dir / "htr_pair_uplift_attention_modifier_hits.csv", index=False)
    top_attention.to_csv(out_dir / "htr_pair_uplift_top_attention_rows.csv", index=False)
    with (out_dir / "htr_pair_uplift_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
