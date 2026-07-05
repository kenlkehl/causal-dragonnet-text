#!/usr/bin/env python3
"""Evaluate TF-IDF-derived tabular features for nuisance modeling.

This script answers a narrow question: can a tabular foundation model consume
TF-IDF features from clinical text and produce useful nuisance predictions for
P[T=1|X] and E[Y|X]?

The TabFM adapter is optional. If the ``tabfm`` package or weights are not
available, its row is reported as skipped while sklearn baselines still run on
the same TF-IDF/SVD features.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


EstimatorFactory = Callable[[str, int], Any]
EPSILON = 1e-6


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: EstimatorFactory
    available: bool = True
    skip_reason: Optional[str] = None


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_file(args.dataset)
    df = read_dataset(dataset_path)
    df = prepare_dataset(df, args)

    model_names = parse_csv_list(args.models)
    svd_components = [int(value) for value in parse_csv_list(args.svd_components)]
    if not svd_components:
        raise ValueError("--svd-components must contain at least one integer")

    all_results: List[Dict[str, Any]] = []
    all_oof_frames: List[pd.DataFrame] = []

    for n_components in svd_components:
        view_name = f"tfidf_svd{n_components}" if n_components > 0 else "tfidf_dense"
        view_results, view_oof = evaluate_feature_view(
            df=df,
            args=args,
            model_names=model_names,
            n_components=n_components,
            view_name=view_name,
        )
        all_results.extend(view_results)
        all_oof_frames.extend(view_oof)

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    oof_path = output_dir / "oof_predictions.parquet"
    config_path = output_dir / "run_config.json"

    summary_json.write_text(json.dumps(all_results, indent=2, default=json_default))
    write_csv(summary_csv, all_results)
    if all_oof_frames:
        pd.concat(all_oof_frames, ignore_index=True).to_parquet(oof_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "n_rows": int(len(df)),
                "args": vars(args),
            },
            indent=2,
            default=json_default,
        )
    )

    print(json.dumps(all_results, indent=2, default=json_default))
    print(f"\nWrote {summary_json}")
    print(f"Wrote {summary_csv}")
    if all_oof_frames:
        print(f"Wrote {oof_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=(
            "synthetic_data/example_synthetic_datasets/"
            "five_confounders_five_effect_modifiers_nsclc_with_structured/dataset.parquet"
        ),
        help="Dataset parquet/csv path or directory containing dataset.parquet.",
    )
    parser.add_argument("--output-dir", default="oci_results_tfidf_tabular_fm_nuisance")
    parser.add_argument("--text-column", default="clinical_text")
    parser.add_argument("--treatment-column", default="treatment_indicator")
    parser.add_argument("--outcome-column", default="outcome_indicator")
    parser.add_argument("--outcome-type", default="binary", choices=["binary", "continuous"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row subsample for fast smoke tests.",
    )
    parser.add_argument(
        "--models",
        default="logistic,extra_trees,hist_gradient_boosting,tabfm",
        help=(
            "Comma-separated model list. Supported: logistic, extra_trees, "
            "hist_gradient_boosting, tabfm, tabfm_ensemble, tabpfn, tabicl."
        ),
    )
    parser.add_argument("--max-tfidf-features", type=int, default=5000)
    parser.add_argument(
        "--svd-components",
        default="128",
        help="Comma-separated dense TF-IDF/SVD dimensions. Use 0 for raw dense TF-IDF.",
    )
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--no-sublinear-tf", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--model-n-jobs", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--tabular-device",
        default="cpu",
        help="Device string passed to tabular foundation model constructors when supported.",
    )
    parser.add_argument(
        "--tabfm-n-estimators",
        type=int,
        default=1,
        help="Number of TabFM feature-shuffle ensemble members. Use 1 for a fast first pass.",
    )
    parser.add_argument(
        "--tabfm-ensemble-n-estimators",
        type=int,
        default=32,
        help="Number of members for the TabFM.ensemble() preset.",
    )
    parser.add_argument(
        "--tabfm-batch-size",
        type=int,
        default=1,
        help="Batch size passed to TabFMClassifier/TabFMRegressor.",
    )
    parser.add_argument(
        "--tabfm-max-rows",
        type=int,
        default=None,
        help="Optional max_num_rows passed to TabFM wrappers.",
    )
    parser.add_argument(
        "--tabfm-use-amp",
        action="store_true",
        help="Enable AMP in TabFM wrappers. Disabled by default for CPU compatibility.",
    )
    return parser


def evaluate_feature_view(
    *,
    df: pd.DataFrame,
    args: argparse.Namespace,
    model_names: Sequence[str],
    n_components: int,
    view_name: str,
) -> Tuple[List[Dict[str, Any]], List[pd.DataFrame]]:
    split_items = split_folds(
        treatment=df[args.treatment_column].to_numpy(dtype=int),
        outcome=df[args.outcome_column].to_numpy(dtype=int)
        if args.outcome_type == "binary"
        else None,
        folds=args.folds,
        seed=args.seed,
    )
    fold_features = build_crossfit_tfidf_features(
        df=df,
        split_items=split_items,
        text_column=args.text_column,
        max_tfidf_features=args.max_tfidf_features,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=not args.no_sublinear_tf,
        n_components=n_components,
        seed=args.seed,
    )

    results: List[Dict[str, Any]] = []
    oof_frames: List[pd.DataFrame] = []
    for model_name in model_names:
        model_spec = get_model_spec(
            model_name=model_name,
            args=args,
            n_features=fold_features["effective_components"],
        )
        if not model_spec.available:
            results.append(
                skipped_result(
                    df=df,
                    args=args,
                    model_name=model_name,
                    feature_view=view_name,
                    skip_reason=model_spec.skip_reason or "model unavailable",
                    n_features=fold_features["effective_components"],
                    vocab_sizes=fold_features["vocab_sizes"],
                )
            )
            continue

        try:
            result, oof_frame = evaluate_model(
                df=df,
                args=args,
                split_items=split_items,
                fold_features=fold_features,
                feature_view=view_name,
                model_spec=model_spec,
            )
        except Exception as exc:
            results.append(
                skipped_result(
                    df=df,
                    args=args,
                    model_name=model_spec.name,
                    feature_view=view_name,
                    skip_reason=f"{type(exc).__name__}: {exc}",
                    n_features=fold_features["effective_components"],
                    vocab_sizes=fold_features["vocab_sizes"],
                )
            )
            continue
        results.append(result)
        oof_frames.append(oof_frame)
    return results, oof_frames


def build_crossfit_tfidf_features(
    *,
    df: pd.DataFrame,
    split_items: Sequence[Tuple[np.ndarray, np.ndarray]],
    text_column: str,
    max_tfidf_features: int,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_df: float,
    sublinear_tf: bool,
    n_components: int,
    seed: int,
) -> Dict[str, Any]:
    texts = df[text_column].astype(str).tolist()
    train_features: List[np.ndarray] = []
    test_features: List[np.ndarray] = []
    vocab_sizes: List[int] = []
    component_counts: List[int] = []

    for fold, (train_idx, test_idx) in enumerate(split_items, start=1):
        vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            dtype=np.float32,
        )
        x_train_sparse = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_test_sparse = vectorizer.transform([texts[i] for i in test_idx])
        vocab_sizes.append(int(len(vectorizer.vocabulary_)))

        if n_components > 0:
            max_components = min(
                int(n_components),
                max(1, int(x_train_sparse.shape[1]) - 1),
                max(1, int(len(train_idx)) - 1),
            )
            reducer = TruncatedSVD(
                n_components=max_components,
                random_state=10_000 + seed + fold,
            )
            x_train = reducer.fit_transform(x_train_sparse)
            x_test = reducer.transform(x_test_sparse)
        else:
            max_components = int(x_train_sparse.shape[1])
            x_train = x_train_sparse.toarray()
            x_test = x_test_sparse.toarray()

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train).astype(np.float32)
        x_test = scaler.transform(x_test).astype(np.float32)
        train_features.append(x_train)
        test_features.append(x_test)
        component_counts.append(int(max_components))

    return {
        "train": train_features,
        "test": test_features,
        "vocab_sizes": vocab_sizes,
        "component_counts": component_counts,
        "effective_components": int(max(component_counts) if component_counts else 0),
    }


def evaluate_model(
    *,
    df: pd.DataFrame,
    args: argparse.Namespace,
    split_items: Sequence[Tuple[np.ndarray, np.ndarray]],
    fold_features: Dict[str, Any],
    feature_view: str,
    model_spec: ModelSpec,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    start_time = time.monotonic()
    y = df[args.outcome_column].to_numpy(dtype=float)
    t = df[args.treatment_column].to_numpy(dtype=int)
    e_hat = np.full(len(df), np.nan, dtype=float)
    m_hat = np.full(len(df), np.nan, dtype=float)
    fold_metrics: List[Dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(split_items, start=1):
        x_train = fold_features["train"][fold - 1]
        x_test = fold_features["test"][fold - 1]

        prop_model = fit_model(
            model_spec,
            task="classification",
            x_train=x_train,
            y_train=t[train_idx],
            seed=10_000 + args.seed + fold,
        )
        e_hat[test_idx] = predict_class_probability(prop_model, x_test)

        if args.outcome_type == "continuous":
            outcome_model = fit_model(
                model_spec,
                task="regression",
                x_train=x_train,
                y_train=y[train_idx],
                seed=20_000 + args.seed + fold,
            )
            m_hat[test_idx] = np.asarray(outcome_model.predict(x_test), dtype=float)
        else:
            outcome_model = fit_model(
                model_spec,
                task="classification",
                x_train=x_train,
                y_train=y[train_idx].astype(int),
                seed=20_000 + args.seed + fold,
            )
            m_hat[test_idx] = predict_class_probability(outcome_model, x_test)

        e_hat[test_idx] = clip_probability(e_hat[test_idx])
        if args.outcome_type == "binary":
            m_hat[test_idx] = clip_probability(m_hat[test_idx])

        fold_metrics.append(
            fold_metric_row(
                fold=fold,
                test_idx=test_idx,
                y=y,
                t=t,
                e_hat=e_hat,
                m_hat=m_hat,
                outcome_type=args.outcome_type,
                n_features=fold_features["component_counts"][fold - 1],
                vocab_size=fold_features["vocab_sizes"][fold - 1],
            )
        )

    result = summarize_predictions(
        df=df,
        args=args,
        model_name=model_spec.name,
        feature_view=feature_view,
        e_hat=e_hat,
        m_hat=m_hat,
        fold_metrics=fold_metrics,
        elapsed_seconds=time.monotonic() - start_time,
        n_features=fold_features["effective_components"],
        vocab_sizes=fold_features["vocab_sizes"],
    )
    oof_frame = pd.DataFrame(
        {
            "_oci_row_id": df["_oci_row_id"].to_numpy(dtype=int),
            "model": model_spec.name,
            "feature_view": feature_view,
            "treatment": t,
            "outcome": y,
            "pred_propensity": e_hat,
            "pred_outcome": m_hat,
        }
    )
    return result, oof_frame


def get_model_spec(model_name: str, args: argparse.Namespace, n_features: int) -> ModelSpec:
    normalized = model_name.strip().lower().replace("-", "_")
    if normalized == "logistic":
        return ModelSpec(
            name="logistic",
            factory=lambda task, seed: make_linear_model(task, args, seed),
        )
    if normalized == "extra_trees":
        return ModelSpec(
            name="extra_trees",
            factory=lambda task, seed: make_extra_trees_model(task, args, seed),
        )
    if normalized == "hist_gradient_boosting":
        return ModelSpec(
            name="hist_gradient_boosting",
            factory=lambda task, seed: make_hist_gradient_boosting_model(task, args, seed),
        )
    if normalized == "tabfm":
        return tabfm_model_spec(args=args, n_features=n_features, use_ensemble=False)
    if normalized == "tabfm_ensemble":
        return tabfm_model_spec(args=args, n_features=n_features, use_ensemble=True)
    if normalized == "tabpfn":
        return tabpfn_model_spec(args=args, n_features=n_features)
    if normalized == "tabicl":
        return tabicl_model_spec(args=args, n_features=n_features)
    return ModelSpec(
        name=normalized,
        factory=lambda task, seed: None,
        available=False,
        skip_reason=f"unsupported model name: {model_name}",
    )


def make_linear_model(task: str, args: argparse.Namespace, seed: int) -> Any:
    if task == "classification":
        return LogisticRegression(
            max_iter=args.max_iter,
            solver="lbfgs",
            random_state=seed,
        )
    return Ridge(alpha=float(args.ridge_alpha), random_state=seed)


def make_extra_trees_model(task: str, args: argparse.Namespace, seed: int) -> Any:
    common = dict(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        random_state=seed,
        n_jobs=args.model_n_jobs,
    )
    if task == "classification":
        return ExtraTreesClassifier(**common)
    return ExtraTreesRegressor(**common)


def make_hist_gradient_boosting_model(task: str, args: argparse.Namespace, seed: int) -> Any:
    common = dict(
        max_iter=min(int(args.max_iter), 300),
        l2_regularization=0.01,
        random_state=seed,
    )
    if task == "classification":
        return HistGradientBoostingClassifier(**common)
    return HistGradientBoostingRegressor(**common)


def tabfm_model_spec(
    args: argparse.Namespace,
    n_features: int,
    *,
    use_ensemble: bool,
) -> ModelSpec:
    spec_name = "tabfm_ensemble" if use_ensemble else "tabfm"
    try:
        module = importlib.import_module("tabfm")
    except Exception as exc:
        return ModelSpec(
            name=spec_name,
            factory=lambda task, seed: None,
            available=False,
            skip_reason=f"tabfm import failed: {exc}",
        )

    classifier_cls = getattr(module, "TabFMClassifier", None)
    regressor_cls = getattr(module, "TabFMRegressor", None)
    loader_module = getattr(module, "tabfm_v1_0_0_pytorch", None)
    loader = getattr(loader_module, "load", None)
    if classifier_cls is None or loader is None:
        return ModelSpec(
            name=spec_name,
            factory=lambda task, seed: None,
            available=False,
            skip_reason="TabFMClassifier or PyTorch loader not found in tabfm package",
        )
    if args.outcome_type == "continuous" and regressor_cls is None:
        return ModelSpec(
            name=spec_name,
            factory=lambda task, seed: None,
            available=False,
            skip_reason="TabFMRegressor not found for continuous outcome",
        )
    if n_features > 500:
        return ModelSpec(
            name=spec_name,
            factory=lambda task, seed: None,
            available=False,
            skip_reason=(
                f"TabFM is optimized for <=500 features; got {n_features}. "
                "Use --svd-components <=500 for this adapter."
            ),
        )

    loaded_models: Dict[str, Any] = {}

    def load_model(task: str) -> Any:
        model_type = "classification" if task == "classification" else "regression"
        if model_type not in loaded_models:
            try:
                loaded_models[model_type] = loader(
                    model_type=model_type,
                    device=args.tabular_device,
                    use_cache=True,
                )
            except FileNotFoundError as exc:
                if "pytorch_model.bin" not in str(exc):
                    raise
                loaded_models[model_type] = load_tabfm_pytorch_safetensors(
                    model_type=model_type,
                    device=args.tabular_device,
                )
        return loaded_models[model_type]

    def factory(task: str, seed: int) -> Any:
        cls = classifier_cls if task == "classification" else regressor_cls
        kwargs = dict(
            n_estimators=int(
                args.tabfm_ensemble_n_estimators if use_ensemble else args.tabfm_n_estimators
            ),
            max_num_features=500,
            max_num_rows=args.tabfm_max_rows,
            use_amp=bool(args.tabfm_use_amp),
            batch_size=int(args.tabfm_batch_size),
            random_state=seed,
            verbose=False,
        )
        if use_ensemble:
            return cls.ensemble(model=load_model(task), **kwargs)
        return cls(
            model=load_model(task),
            **kwargs,
        )

    return ModelSpec(name=spec_name, factory=factory)


def load_tabfm_pytorch_safetensors(model_type: str, device: Optional[str]) -> Any:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch
    from tabfm.src.pytorch import tabfm_v1_0_0 as tabfm_def

    if model_type == "classification":
        config = tabfm_def.ClassificationConfig()
    elif model_type == "regression":
        config = tabfm_def.RegressionConfig()
    else:
        raise ValueError(f"Unsupported TabFM model_type: {model_type}")

    base_path = Path(snapshot_download(repo_id=tabfm_def.HF_REPO_ID))
    safetensors_path = base_path / model_type / "model.safetensors"
    bin_path = base_path / model_type / "pytorch_model.bin"
    model = tabfm_def.TabFM(**config.to_dict())
    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            "TabFM weights not found. Expected one of: "
            f"{safetensors_path}, {bin_path}"
        )
    model.load_state_dict(state_dict, strict=True)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def tabpfn_model_spec(args: argparse.Namespace, n_features: int) -> ModelSpec:
    try:
        module = importlib.import_module("tabpfn")
    except Exception as exc:
        return ModelSpec(
            name="tabpfn",
            factory=lambda task, seed: None,
            available=False,
            skip_reason=f"tabpfn import failed: {exc}",
        )

    classifier_cls = getattr(module, "TabPFNClassifier", None)
    regressor_cls = getattr(module, "TabPFNRegressor", None)
    if classifier_cls is None:
        return ModelSpec(
            name="tabpfn",
            factory=lambda task, seed: None,
            available=False,
            skip_reason="tabpfn.TabPFNClassifier not found",
        )
    if args.outcome_type == "continuous" and regressor_cls is None:
        return ModelSpec(
            name="tabpfn",
            factory=lambda task, seed: None,
            available=False,
            skip_reason="tabpfn.TabPFNRegressor not found for continuous outcome",
        )
    if n_features > 2000:
        return ModelSpec(
            name="tabpfn",
            factory=lambda task, seed: None,
            available=False,
            skip_reason=(
                f"TabPFN adapter expects <=2000 dense features; got {n_features}. "
                "Use --svd-components <=2000."
            ),
        )

    def factory(task: str, seed: int) -> Any:
        cls = classifier_cls if task == "classification" else regressor_cls
        return instantiate_with_supported_kwargs(
            cls,
            {
                "device": args.tabular_device,
                "random_state": seed,
                "seed": seed,
                "ignore_pretraining_limits": True,
            },
        )

    return ModelSpec(name="tabpfn", factory=factory)


def tabicl_model_spec(args: argparse.Namespace, n_features: int) -> ModelSpec:
    try:
        module = importlib.import_module("tabicl")
    except Exception as exc:
        return ModelSpec(
            name="tabicl",
            factory=lambda task, seed: None,
            available=False,
            skip_reason=f"tabicl import failed: {exc}",
        )

    classifier_cls = find_attr(module, ["TabICLClassifier", "TabICL"])
    regressor_cls = find_attr(module, ["TabICLRegressor"])
    if classifier_cls is None:
        return ModelSpec(
            name="tabicl",
            factory=lambda task, seed: None,
            available=False,
            skip_reason="TabICL classifier class not found",
        )
    if args.outcome_type == "continuous" and regressor_cls is None:
        return ModelSpec(
            name="tabicl",
            factory=lambda task, seed: None,
            available=False,
            skip_reason="TabICL regressor class not found for continuous outcome",
        )

    def factory(task: str, seed: int) -> Any:
        cls = classifier_cls if task == "classification" else regressor_cls
        return instantiate_with_supported_kwargs(
            cls,
            {
                "device": args.tabular_device,
                "random_state": seed,
                "seed": seed,
            },
        )

    return ModelSpec(name="tabicl", factory=factory)


def instantiate_with_supported_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls()
    supported = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters and value is not None
    }
    return cls(**supported)


def find_attr(module: Any, names: Sequence[str]) -> Any:
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    return None


def fit_model(
    model_spec: ModelSpec,
    *,
    task: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    if task == "classification" and np.unique(y_train.astype(int)).size < 2:
        model = DummyClassifier(strategy="constant", constant=int(y_train[0]))
    elif task == "regression" and np.nanstd(y_train.astype(float)) <= 0.0:
        model = DummyRegressor(strategy="constant", constant=float(y_train[0]))
    else:
        model = model_spec.factory(task, seed)
    model.fit(x_train, y_train)
    return model


def predict_class_probability(model: Any, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(x_test), dtype=float)
        if probs.ndim == 1:
            return probs
        classes = np.asarray(getattr(model, "classes_", []))
        matches = np.where(classes == 1)[0]
        if matches.size:
            return probs[:, int(matches[0])]
        if probs.shape[1] == 1 and classes.size == 1:
            return np.full(x_test.shape[0], float(classes[0] == 1), dtype=float)
        return probs[:, -1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x_test), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(x_test), dtype=float)


def summarize_predictions(
    *,
    df: pd.DataFrame,
    args: argparse.Namespace,
    model_name: str,
    feature_view: str,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    fold_metrics: List[Dict[str, Any]],
    elapsed_seconds: float,
    n_features: int,
    vocab_sizes: Sequence[int],
) -> Dict[str, Any]:
    y = df[args.outcome_column].to_numpy(dtype=float)
    t = df[args.treatment_column].to_numpy(dtype=int)
    result: Dict[str, Any] = {
        "model": model_name,
        "feature_view": feature_view,
        "status": "ok",
        "n_rows": int(len(df)),
        "folds": int(len(fold_metrics)),
        "n_tfidf_features_max": int(args.max_tfidf_features),
        "n_dense_features": int(n_features),
        "vocab_size_mean": finite_or_none(np.mean(vocab_sizes)),
        "vocab_size_min": int(np.min(vocab_sizes)) if vocab_sizes else None,
        "vocab_size_max": int(np.max(vocab_sizes)) if vocab_sizes else None,
        "elapsed_seconds": finite_or_none(elapsed_seconds),
        "treatment_auroc": safe_roc_auc(t, e_hat),
        "treatment_brier": finite_or_none(brier_score_loss(t, e_hat)),
        "treatment_log_loss": safe_binary_log_loss(t, e_hat),
        "propensity_mean": finite_or_none(np.mean(e_hat)),
        "propensity_min": finite_or_none(np.min(e_hat)),
        "propensity_max": finite_or_none(np.max(e_hat)),
        "propensity_p01": finite_or_none(np.quantile(e_hat, 0.01)),
        "propensity_p99": finite_or_none(np.quantile(e_hat, 0.99)),
        "fold_metrics": fold_metrics,
    }
    if args.outcome_type == "continuous":
        result["outcome_rmse"] = finite_or_none(np.sqrt(mean_squared_error(y, m_hat)))
    else:
        result["outcome_auroc"] = safe_roc_auc(y.astype(int), m_hat)
        result["outcome_brier"] = finite_or_none(brier_score_loss(y.astype(int), m_hat))
        result["outcome_log_loss"] = safe_binary_log_loss(y.astype(int), m_hat)
        result["outcome_mean_pred"] = finite_or_none(np.mean(m_hat))

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
    return result


def skipped_result(
    *,
    df: pd.DataFrame,
    args: argparse.Namespace,
    model_name: str,
    feature_view: str,
    skip_reason: str,
    n_features: int,
    vocab_sizes: Sequence[int],
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "feature_view": feature_view,
        "status": "skipped",
        "skip_reason": skip_reason,
        "n_rows": int(len(df)),
        "folds": int(args.folds),
        "n_tfidf_features_max": int(args.max_tfidf_features),
        "n_dense_features": int(n_features),
        "vocab_size_mean": finite_or_none(np.mean(vocab_sizes)) if vocab_sizes else None,
        "vocab_size_min": int(np.min(vocab_sizes)) if vocab_sizes else None,
        "vocab_size_max": int(np.max(vocab_sizes)) if vocab_sizes else None,
    }


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
    vocab_size: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "fold": int(fold),
        "n_test": int(len(test_idx)),
        "n_dense_features": int(n_features),
        "vocab_size": int(vocab_size),
        "treatment_auroc": safe_roc_auc(t[test_idx], e_hat[test_idx]),
        "treatment_brier": finite_or_none(brier_score_loss(t[test_idx], e_hat[test_idx])),
        "treatment_log_loss": safe_binary_log_loss(t[test_idx], e_hat[test_idx]),
    }
    if outcome_type == "continuous":
        row["outcome_rmse"] = finite_or_none(
            np.sqrt(mean_squared_error(y[test_idx], m_hat[test_idx]))
        )
    else:
        y_fold = y[test_idx].astype(int)
        row["outcome_auroc"] = safe_roc_auc(y_fold, m_hat[test_idx])
        row["outcome_brier"] = finite_or_none(brier_score_loss(y_fold, m_hat[test_idx]))
        row["outcome_log_loss"] = safe_binary_log_loss(y_fold, m_hat[test_idx])
    return row


def resolve_dataset_file(path: str | Path) -> Path:
    value = Path(path)
    if value.is_file():
        return value
    if not value.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {value}")
    preferred = [
        "dataset.parquet",
        "dataset_with_extraction.parquet",
        "data.parquet",
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
        raise FileNotFoundError(f"No parquet/csv file found under {value}")
    preview = ", ".join(match.name for match in matches[:8])
    raise ValueError(
        f"Multiple candidate dataset files found under {value}: {preview}. "
        "Pass the exact file path."
    )


def read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset file extension: {path.suffix}")


def prepare_dataset(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    required = [args.text_column, args.treatment_column, args.outcome_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required column(s): " + ", ".join(missing))
    if args.sample_size is not None and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=args.seed)
    df = df.reset_index(drop=True).copy()
    df["_oci_row_id"] = np.arange(len(df), dtype=int)
    df[args.text_column] = df[args.text_column].fillna("").astype(str)
    for column in [args.treatment_column, args.outcome_column]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    bad = {
        column: int(df[column].isna().sum())
        for column in [args.treatment_column, args.outcome_column]
        if int(df[column].isna().sum()) > 0
    }
    if bad:
        raise ValueError(f"Treatment/outcome columns contain missing values: {bad}")
    return df


def split_folds(
    *,
    treatment: np.ndarray,
    outcome: Optional[np.ndarray],
    folds: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        raise ValueError("--folds must be >= 2")
    n_rows = len(treatment)
    if outcome is not None:
        labels = treatment.astype(int) * 2 + outcome.astype(int)
    else:
        labels = treatment.astype(int)
    values, counts = np.unique(labels, return_counts=True)
    if len(values) >= 2 and int(np.min(counts)) >= 2:
        n_splits = max(2, min(int(folds), int(np.min(counts)), n_rows))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [
            (np.asarray(train_idx), np.asarray(test_idx))
            for train_idx, test_idx in splitter.split(np.zeros(n_rows), labels)
        ]
    n_splits = max(2, min(int(folds), n_rows))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (np.asarray(train_idx), np.asarray(test_idx))
        for train_idx, test_idx in splitter.split(np.zeros(n_rows))
    ]


def parse_csv_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, str):
        raw_values: Iterable[str] = value.split(",")
    else:
        raw_values = value
    return [str(item).strip() for item in raw_values if str(item).strip()]


def safe_roc_auc(y_true: np.ndarray, score: np.ndarray) -> Optional[float]:
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    if int(mask.sum()) < 2 or np.unique(y[mask]).size < 2:
        return None
    try:
        return finite_or_none(roc_auc_score(y[mask], s[mask]))
    except ValueError:
        return None


def safe_binary_log_loss(y_true: np.ndarray, score: np.ndarray) -> Optional[float]:
    try:
        return finite_or_none(log_loss(y_true, clip_probability(score), labels=[0, 1]))
    except ValueError:
        return None


def clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPSILON, 1.0 - EPSILON)


def safe_corr(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 2:
        return None
    if float(np.std(left[mask])) <= 0.0 or float(np.std(right[mask])) <= 0.0:
        return None
    return finite_or_none(np.corrcoef(left[mask], right[mask])[0, 1])


def finite_or_none(value: Any) -> Optional[float]:
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
