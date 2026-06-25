from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


RANDOM_STATE = 20260625
N_FOLDS = 5
DATASET = Path("dataset.parquet")
REPORT = Path("report.txt")


VECTORIZERS = [
    {
        "label": "unigram",
        "ngram_range": (1, 1),
        "min_df": 5,
        "max_df": 0.95,
        "max_features": 25_000,
        "sublinear_tf": True,
    },
    {
        "label": "broad_1_3",
        "ngram_range": (1, 3),
        "min_df": 5,
        "max_df": 0.95,
        "max_features": 60_000,
        "sublinear_tf": True,
    },
    {
        "label": "phrase_2_4",
        "ngram_range": (2, 4),
        "min_df": 5,
        "max_df": 0.98,
        "max_features": 60_000,
        "sublinear_tf": True,
    },
    {
        "label": "rare_broad",
        "ngram_range": (1, 3),
        "min_df": 2,
        "max_df": 0.98,
        "max_features": 100_000,
        "sublinear_tf": True,
    },
]


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 20) -> str:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"ERROR: {exc!r}"
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if err:
        out = f"{out}\nSTDERR:\n{err}" if out else f"STDERR:\n{err}"
    return out


def safe_auc(y_true: np.ndarray, pred: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, pred))


def safe_ap(y_true: np.ndarray, pred: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, pred))


def metric_block(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    return {
        "auroc": safe_auc(y_true, pred),
        "average_precision": safe_ap(y_true, pred),
        "brier": float(brier_score_loss(y_true, pred)),
        "log_loss": float(log_loss(y_true, pred, labels=[0, 1])),
    }


def top_terms(
    names: np.ndarray,
    coef: np.ndarray,
    *,
    fold: int,
    label: str,
    params: dict,
    source: str,
    n_each: int = 40,
) -> list[dict]:
    coef = np.asarray(coef).ravel()
    if coef.size == 0:
        return []
    n_each = min(n_each, coef.size)
    top_pos = np.argpartition(coef, -n_each)[-n_each:]
    top_neg = np.argpartition(coef, n_each - 1)[:n_each]
    rows = []
    for idx in top_pos[np.argsort(coef[top_pos])[::-1]]:
        rows.append(
            {
                "fold": fold,
                "vectorization_run": label,
                "vectorizer_params": json.dumps(params, sort_keys=True),
                "evidence_source": source,
                "term_or_span": str(names[idx]),
                "direction": "positive",
                "score": float(coef[idx]),
            }
        )
    for idx in top_neg[np.argsort(coef[top_neg])]:
        rows.append(
            {
                "fold": fold,
                "vectorization_run": label,
                "vectorizer_params": json.dumps(params, sort_keys=True),
                "evidence_source": source,
                "term_or_span": str(names[idx]),
                "direction": "negative",
                "score": float(coef[idx]),
            }
        )
    return rows


def make_model() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def inner_oof_nuisance(
    x_train: sparse.csr_matrix,
    a_train: np.ndarray,
    y_train: np.ndarray,
    strata_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    e_inner = np.zeros(len(a_train), dtype=float)
    m_inner = np.zeros(len(y_train), dtype=float)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + 11)
    for inner_tr, inner_va in inner.split(x_train, strata_train):
        e_model = make_model()
        m_model = make_model()
        e_model.fit(x_train[inner_tr], a_train[inner_tr])
        m_model.fit(x_train[inner_tr], y_train[inner_tr])
        e_inner[inner_va] = e_model.predict_proba(x_train[inner_va])[:, 1]
        m_inner[inner_va] = m_model.predict_proba(x_train[inner_va])[:, 1]
    return np.clip(e_inner, 0.02, 0.98), np.clip(m_inner, 0.02, 0.98)


def train_weighted_sgd(
    x_train: sparse.csr_matrix,
    target: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> SGDRegressor:
    reg = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-5,
        max_iter=2500,
        tol=1e-4,
        random_state=RANDOM_STATE,
        learning_rate="invscaling",
        eta0=0.01,
    )
    reg.fit(x_train, target, sample_weight=sample_weight)
    return reg


def summarize_sections(texts: pd.Series) -> dict[str, object]:
    note_counts = texts.str.count("<new_note>").fillna(0).astype(int)
    heading_counter: Counter[str] = Counter()
    for text in texts.head(100):
        for line in str(text).splitlines():
            stripped = line.strip()
            if stripped.startswith("### ") or (
                len(stripped) < 90
                and stripped.endswith(":")
                and any(ch.isalpha() for ch in stripped)
            ):
                heading_counter[stripped[:80]] += 1
    return {
        "new_note_count": note_counts.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
        ).to_dict(),
        "common_headings_sample": heading_counter.most_common(25),
    }


def write_initial_report(df: pd.DataFrame, folds: pd.DataFrame) -> None:
    texts = df["clinical_text"].fillna("")
    char_len = texts.str.len()
    word_len = texts.str.split().str.len()
    section_summary = summarize_sections(texts)
    ctab = pd.crosstab(
        df["treatment_indicator"], df["outcome_indicator"], margins=True, dropna=False
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    torch_probe = run_cmd(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "print('sys.executable', sys.executable); "
                "print('sys.prefix', sys.prefix); "
                "print('VIRTUAL_ENV', os.environ.get('VIRTUAL_ENV')); "
                "print('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES')); "
                "print('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH')); "
                "import torch; "
                "print('torch.__file__', torch.__file__); "
                "print('torch.__version__', torch.__version__); "
                "print('torch.cuda.is_available', torch.cuda.is_available()); "
                "print('torch.cuda.device_count', torch.cuda.device_count()); "
                "print('device_names', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
            ),
        ],
        env=env,
    )
    nvidia = run_cmd(["nvidia-smi"], timeout=15)

    lines = [
        "# Causal DGP Discovery Report",
        "",
        "## Dataset and Environment",
        "",
        f"- Dataset path: `{DATASET.resolve()}`",
        f"- Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        f"- Columns/dtypes: {df.dtypes.astype(str).to_dict()}",
        f"- Missing values: {df.isna().sum().to_dict()}",
        f"- Empty clinical_text rows: {int((char_len == 0).sum())}",
        f"- Treatment rate: {df['treatment_indicator'].mean():.4f} ({int(df['treatment_indicator'].sum())}/{len(df)})",
        f"- Outcome rate: {df['outcome_indicator'].mean():.4f} ({int(df['outcome_indicator'].sum())}/{len(df)})",
        "",
        "Treatment/outcome cross-tab:",
        "",
        "```",
        ctab.to_string(),
        "```",
        "",
        "Clinical text character length summary:",
        "",
        "```",
        char_len.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string(),
        "```",
        "",
        "Clinical text word length summary:",
        "",
        "```",
        word_len.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string(),
        "```",
        "",
        f"- `<new_note>` count summary: {section_summary['new_note_count']}",
        f"- Common heading sample from first 100 notes: {section_summary['common_headings_sample']}",
        f"- Fold counts: {folds['fold'].value_counts().sort_index().to_dict()}",
        "",
        "### GPU / Python Probe",
        "",
        "- Requested GPU assignment for GPU work: physical GPU ID 1 via `CUDA_VISIBLE_DEVICES=1`.",
        "- Shell `nvidia-smi` output:",
        "",
        "```",
        nvidia[:6000],
        "```",
        "",
        "- Sandboxed PyTorch probe from `~/thisenv`:",
        "",
        "```",
        torch_probe[:4000],
        "```",
        "",
        "Sandboxed PyTorch could not use CUDA while `nvidia-smi` saw GPU 1; an escalated probe was run separately and did see one visible RTX A6000 device. Neural HTR jobs therefore require escalation with `CUDA_VISIBLE_DEVICES=1`.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_parquet(DATASET)
    if "patient_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "patient_id"})
    df = df.sort_values("patient_id").reset_index(drop=True)
    texts = df["clinical_text"].fillna("").astype(str).to_numpy()
    a = df["treatment_indicator"].astype(int).to_numpy()
    y = df["outcome_indicator"].astype(int).to_numpy()
    patient_ids = df["patient_id"].to_numpy()
    strata = a.astype(str) + "_" + y.astype(str)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_id = np.full(len(df), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(skf.split(texts, strata)):
        fold_id[test_idx] = fold
    folds = pd.DataFrame({"patient_id": patient_ids, "fold": fold_id})
    folds.to_parquet("fold_assignments.parquet", index=False)
    write_initial_report(df, folds)

    evidence_rows: list[dict] = []
    pred_rows: list[dict] = []
    comparison_rows: list[dict] = []

    for variant in VECTORIZERS:
        label = variant["label"]
        params = {k: v for k, v in variant.items() if k != "label"}
        e_hat = np.zeros(len(df), dtype=float)
        m_hat = np.zeros(len(df), dtype=float)
        r_pred = np.zeros(len(df), dtype=float)
        tau_pred = np.zeros(len(df), dtype=float)

        for fold in range(N_FOLDS):
            train_idx = np.flatnonzero(fold_id != fold)
            test_idx = np.flatnonzero(fold_id == fold)
            vectorizer = TfidfVectorizer(
                ngram_range=params["ngram_range"],
                min_df=params["min_df"],
                max_df=params["max_df"],
                max_features=params["max_features"],
                sublinear_tf=params["sublinear_tf"],
                strip_accents="unicode",
                lowercase=True,
                token_pattern=r"(?u)\b[\w][\w\-]+\b",
            )
            x_train = vectorizer.fit_transform(texts[train_idx])
            x_test = vectorizer.transform(texts[test_idx])
            names = vectorizer.get_feature_names_out()

            e_model = make_model()
            m_model = make_model()
            e_model.fit(x_train, a[train_idx])
            m_model.fit(x_train, y[train_idx])
            e_hat[test_idx] = e_model.predict_proba(x_test)[:, 1]
            m_hat[test_idx] = m_model.predict_proba(x_test)[:, 1]
            evidence_rows.extend(
                top_terms(
                    names,
                    e_model.coef_[0],
                    fold=fold,
                    label=label,
                    params=params,
                    source="bow_treatment",
                )
            )
            evidence_rows.extend(
                top_terms(
                    names,
                    m_model.coef_[0],
                    fold=fold,
                    label=label,
                    params=params,
                    source="bow_outcome",
                )
            )

            e_inner, m_inner = inner_oof_nuisance(
                x_train,
                a[train_idx],
                y[train_idx],
                strata[train_idx],
            )
            residual_y = y[train_idx] - m_inner
            residual_a = a[train_idx] - e_inner
            residual_reg = train_weighted_sgd(x_train, residual_y)
            r_pred[test_idx] = residual_reg.predict(x_test)
            evidence_rows.extend(
                top_terms(
                    names,
                    residual_reg.coef_,
                    fold=fold,
                    label=label,
                    params=params,
                    source="residual_outcome",
                )
            )

            denom = np.where(np.abs(residual_a) < 0.05, np.sign(residual_a) * 0.05, residual_a)
            denom[denom == 0] = 0.05
            pseudo = np.clip(residual_y / denom, -5.0, 5.0)
            weights = np.clip(residual_a**2, 0.0025, 0.25)
            tau_reg = train_weighted_sgd(x_train, pseudo, sample_weight=weights)
            tau_pred[test_idx] = tau_reg.predict(x_test)
            evidence_rows.extend(
                top_terms(
                    names,
                    tau_reg.coef_,
                    fold=fold,
                    label=label,
                    params=params,
                    source="pseudo_outcome",
                )
            )

        e_hat = np.clip(e_hat, 1e-6, 1 - 1e-6)
        m_hat = np.clip(m_hat, 1e-6, 1 - 1e-6)
        residual_a_all = a - e_hat
        residual_y_all = y - m_hat
        denom_all = np.where(
            np.abs(residual_a_all) < 0.05,
            np.sign(residual_a_all) * 0.05,
            residual_a_all,
        )
        denom_all[denom_all == 0] = 0.05
        pseudo_all = np.clip(residual_y_all / denom_all, -5.0, 5.0)

        for i in range(len(df)):
            pred_rows.append(
                {
                    "patient_id": int(patient_ids[i]),
                    "fold": int(fold_id[i]),
                    "e_hat": float(e_hat[i]),
                    "m_hat": float(m_hat[i]),
                    "treatment_residual": float(residual_a_all[i]),
                    "outcome_residual": float(residual_y_all[i]),
                    "pseudo_outcome": float(pseudo_all[i]),
                    "r_loss_target": float(pseudo_all[i]),
                    "fold_specific_ite_estimate": float(tau_pred[i]),
                    "model_family": "tfidf_linear",
                    "iteration": label,
                }
            )

        treatment_metrics = metric_block(a, e_hat)
        outcome_metrics = metric_block(y, m_hat)
        pseudo_mse = float(mean_squared_error(pseudo_all, tau_pred))
        comparison_rows.append(
            {
                "model_family": "tfidf_linear",
                "iteration": label,
                "vectorization_strategy": label,
                "vectorizer_params": json.dumps(params, sort_keys=True),
                "treatment_nuisance": treatment_metrics,
                "outcome_nuisance": outcome_metrics,
                "pseudo_outcome_mse": pseudo_mse,
                "ite_mean": float(np.mean(tau_pred)),
                "ite_sd": float(np.std(tau_pred)),
                "ite_p05": float(np.quantile(tau_pred, 0.05)),
                "ite_p95": float(np.quantile(tau_pred, 0.95)),
            }
        )

    evidence = pd.DataFrame(evidence_rows)
    fold_counts = evidence.groupby(["evidence_source", "term_or_span"])["fold"].nunique()
    variant_counts = evidence.groupby(["evidence_source", "term_or_span"])[
        "vectorization_run"
    ].nunique()
    evidence["recurred_across_folds"] = [
        bool(fold_counts.get((src, term), 0) >= 3)
        for src, term in zip(evidence["evidence_source"], evidence["term_or_span"])
    ]
    evidence["recurred_across_vectorization_strategies"] = [
        bool(variant_counts.get((src, term), 0) >= 2)
        for src, term in zip(evidence["evidence_source"], evidence["term_or_span"])
    ]
    evidence["mapped_candidate_concept"] = ""
    evidence.to_json("text_evidence_bow.jsonl", orient="records", lines=True)

    preds = pd.DataFrame(pred_rows)
    preds.to_parquet("crossfit_predictions_bow.parquet", index=False)

    with open("model_comparison_bow.json", "w", encoding="utf-8") as f:
        json.dump(comparison_rows, f, indent=2)

    top_recurrent = (
        evidence.assign(abs_score=evidence["score"].abs())
        .query("recurred_across_folds or recurred_across_vectorization_strategies")
        .sort_values("abs_score", ascending=False)
        .groupby(["evidence_source", "term_or_span"], as_index=False)
        .agg(
            mean_abs_score=("abs_score", "mean"),
            max_abs_score=("abs_score", "max"),
            folds=("fold", "nunique"),
            variants=("vectorization_run", "nunique"),
            directions=("direction", lambda x: sorted(set(x))),
        )
        .sort_values(["variants", "folds", "max_abs_score"], ascending=False)
        .head(120)
    )

    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n## BoW / TF-IDF Discovery\n\n")
        f.write(
            "Used one fixed 5-fold split stratified by treatment/outcome cells. "
            "Each vectorization run trained out-of-fold treatment and outcome nuisance "
            "models, plus residual-outcome and R-learner-style pseudo-outcome signal models.\n\n"
        )
        f.write("Vectorization suite:\n\n")
        for variant in VECTORIZERS:
            f.write(f"- {variant['label']}: {variant}\n")
        f.write("\nModel comparison summary:\n\n```json\n")
        f.write(json.dumps(comparison_rows, indent=2)[:12000])
        f.write("\n```\n\n")
        f.write("Top recurrent term evidence across folds/vectorizer variants:\n\n")
        f.write("```\n")
        f.write(top_recurrent.to_string(index=False, max_colwidth=80))
        f.write("\n```\n\n")
        f.write(
            "Artifacts written: `fold_assignments.parquet`, `text_evidence_bow.jsonl`, "
            "`crossfit_predictions_bow.parquet`, and `model_comparison_bow.json`.\n"
        )


if __name__ == "__main__":
    main()
