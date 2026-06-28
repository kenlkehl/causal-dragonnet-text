#!/usr/bin/env python
"""Oracle runner for the multi-model BoW-guided agentic forest path."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import (  # noqa: E402
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    BoWViewConfig,
    EmbeddingContrastDiscoveryConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ModelArchitectureConfig,
    MultiModelAgenticForestConfig,
)
from oci.inference.multi_model_agentic_forest import (  # noqa: E402
    run_multi_model_agentic_forest,
)
from run_oracle_experiments import _resolve_parquet_file  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MultiModelAgenticOracleConfig:
    dataset_path: str
    dataset_name: str
    n_folds: int = 5
    seed: int = 42
    sample_size: Optional[int] = None
    text_max_chars: Optional[int] = None
    num_workers: int = 1

    nuisance_folds: int = 5
    effect_folds: int = 5
    bow_view_grid: str = "default_broad"
    bow_views_json: Optional[str] = None
    max_features: int = 30000
    min_df: int = 5
    max_df: float = 0.95
    ngram_range_min: int = 1
    ngram_range_max: int = 3
    bow_model: str = "linear"
    prespecified_features_json: Optional[str] = None
    logistic_c: float = 1.0
    ridge_alpha: float = 10.0
    e_clip: float = 0.01
    top_n_features: int = 100
    candidate_proposals_per_fold: int = 30
    candidate_consistency_enabled: bool = True
    candidate_consistency_inner_folds: int = 3
    candidate_consistency_min_folds: int = 2
    candidate_consistency_min_fold_fraction: float = 0.5
    candidate_consistency_recovery_max_candidates: int = 12
    candidate_consistency_parallelism: str = "1"
    extracted_feature_review_enabled: bool = True
    extracted_feature_review_max_rounds: int = 3
    extracted_feature_review_auc_margin: float = 0.02
    extracted_feature_review_loss_relative_margin: float = 0.05
    extracted_feature_review_min_benchmark_auc: float = 0.55
    outer_parallelism: str = "1"
    bow_parallel_backend: str = "processes"
    fold_parallelism: str = "auto"

    embedding_contrast_enabled: bool = True
    embedding_model_name: str = "Qwen/Qwen3-Embedding-8B"
    embedding_cache_dir: Optional[str] = None
    embedding_device: Optional[str] = None
    embedding_batch_size: int = 16
    embedding_chunk_size_words: int = 256
    embedding_chunk_overlap_words: int = 64
    embedding_max_chunks: int = 64
    embedding_chunk_selection: str = "last"
    embedding_top_k_chunks_per_tail: int = 12
    embedding_max_chunks_per_patient: int = 2
    embedding_min_probe_auc: float = 0.0
    embedding_pseudo_target_quantile: float = 0.20
    embedding_pseudo_target_weighted: bool = True
    embedding_include_cell_contrasts: bool = True
    embedding_include_orthogonal_r_score_contrasts: bool = True
    embedding_concept_phrases: List[str] = field(default_factory=list)
    embedding_residualize_columns: List[str] = field(default_factory=list)

    cf_n_estimators: int = 200
    cf_min_samples_leaf: int = 10
    cf_max_depth: Optional[int] = None
    cf_max_features: str = "sqrt"
    cf_honest: bool = True
    cf_inference: bool = True

    min_feature_coverage: float = 0.70
    agent_server_url: str = "http://localhost:8000/v1"
    agent_model_name: str = "Qwen/Qwen3.6-27B"
    agent_api_key: str = "EMPTY"
    agent_temperature: float = 0.0
    agent_max_tokens: int = 25000
    agent_request_max_retries: int = 3
    agent_retry_initial_delay: float = 1.0
    agent_retry_max_delay: float = 30.0
    agent_retry_backoff_factor: float = 2.0
    agent_save_context: bool = False
    agent_save_raw_output: bool = False

    extraction_server_url: str = "http://localhost:8000/v1"
    extraction_model_name: str = "Qwen/Qwen3.6-27B"
    extraction_mode: str = "server"
    extraction_reasoning_parser: Optional[str] = "auto"
    extraction_batch_size: int = 100
    extraction_max_retries: int = 3
    extraction_retry_initial_delay: float = 1.0
    extraction_retry_max_delay: float = 30.0
    extraction_retry_backoff_factor: float = 2.0
    extraction_temperature: float = 0.0
    extraction_max_tokens: int = 25000
    extraction_max_text_length: int = 400000
    extraction_cache_enabled: bool = True
    extraction_cache_dir: Optional[str] = None

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:12]


def _make_applied_config(
    config: MultiModelAgenticOracleConfig,
    parquet_file: Path,
) -> AppliedInferenceConfig:
    bow_views = _bow_views_for_config(config)
    return AppliedInferenceConfig(
        clinical_question=(
            "Estimate heterogeneous treatment effects from clinical text and "
            "identify text-derived confounders and effect modifiers."
        ),
        outcome_type="binary",
        dataset_path=str(parquet_file),
        text_column="clinical_text",
        outcome_column="outcome_indicator",
        treatment_column="treatment_indicator",
        cv_folds=config.n_folds,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=config.cf_n_estimators,
                max_depth=config.cf_max_depth,
                min_samples_leaf=config.cf_min_samples_leaf,
                max_features=config.cf_max_features,
                honest=config.cf_honest,
                inference=config.cf_inference,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=max(2, config.n_folds),
                inner_folds=max(2, config.nuisance_folds),
                max_iterations=1,
                max_additions_per_iter=config.candidate_proposals_per_fold,
                max_removals_per_iter=0,
                min_feature_coverage=config.min_feature_coverage,
                agent_server_url=config.agent_server_url,
                agent_model_name=config.agent_model_name,
                agent_api_key=config.agent_api_key,
                agent_temperature=config.agent_temperature,
                agent_max_tokens=config.agent_max_tokens,
                agent_request_max_retries=config.agent_request_max_retries,
                agent_retry_initial_delay=config.agent_retry_initial_delay,
                agent_retry_max_delay=config.agent_retry_max_delay,
                agent_retry_backoff_factor=config.agent_retry_backoff_factor,
                save_agent_context=config.agent_save_context,
                save_agent_raw_output=config.agent_save_raw_output,
                random_state=config.seed,
            ),
            multi_model_agentic_forest=MultiModelAgenticForestConfig(
                nuisance_folds=config.nuisance_folds,
                effect_folds=config.effect_folds,
                bow_views=bow_views,
                prespecified_features_json=config.prespecified_features_json,
                e_clip=config.e_clip,
                top_n_features=config.top_n_features,
                candidate_proposals_per_fold=config.candidate_proposals_per_fold,
                candidate_consistency_enabled=config.candidate_consistency_enabled,
                candidate_consistency_inner_folds=config.candidate_consistency_inner_folds,
                candidate_consistency_min_folds=config.candidate_consistency_min_folds,
                candidate_consistency_min_fold_fraction=config.candidate_consistency_min_fold_fraction,
                candidate_consistency_recovery_max_candidates=config.candidate_consistency_recovery_max_candidates,
                candidate_consistency_parallelism=config.candidate_consistency_parallelism,
                extracted_feature_review_enabled=config.extracted_feature_review_enabled,
                extracted_feature_review_max_rounds=config.extracted_feature_review_max_rounds,
                extracted_feature_review_auc_margin=config.extracted_feature_review_auc_margin,
                extracted_feature_review_loss_relative_margin=(
                    config.extracted_feature_review_loss_relative_margin
                ),
                extracted_feature_review_min_benchmark_auc=(
                    config.extracted_feature_review_min_benchmark_auc
                ),
                outer_parallelism=config.outer_parallelism,
                bow_parallel_backend=config.bow_parallel_backend,
                fold_parallelism=config.fold_parallelism,
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=config.embedding_contrast_enabled,
                    disable_reason=(
                        None
                        if config.embedding_contrast_enabled
                        else "disabled by oracle multi-model script CLI"
                    ),
                    model_name=config.embedding_model_name,
                    cache_dir=config.embedding_cache_dir,
                    device=config.embedding_device,
                    batch_size=config.embedding_batch_size,
                    chunk_size_words=config.embedding_chunk_size_words,
                    chunk_overlap_words=config.embedding_chunk_overlap_words,
                    max_chunks=config.embedding_max_chunks,
                    chunk_selection=config.embedding_chunk_selection,
                    top_k_chunks_per_tail=config.embedding_top_k_chunks_per_tail,
                    max_chunks_per_patient=config.embedding_max_chunks_per_patient,
                    min_probe_auc=config.embedding_min_probe_auc,
                    pseudo_target_quantile=config.embedding_pseudo_target_quantile,
                    pseudo_target_weighted=config.embedding_pseudo_target_weighted,
                    include_cell_contrasts=config.embedding_include_cell_contrasts,
                    include_orthogonal_r_score_contrasts=(
                        config.embedding_include_orthogonal_r_score_contrasts
                    ),
                    concept_phrases=config.embedding_concept_phrases,
                    residualize_columns=config.embedding_residualize_columns,
                ),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=[],
            vllm_mode=config.extraction_mode,
            vllm_server_url=config.extraction_server_url,
            vllm_model_name=config.extraction_model_name,
            vllm_reasoning_parser=config.extraction_reasoning_parser,
            extraction_batch_size=config.extraction_batch_size,
            extraction_max_retries=config.extraction_max_retries,
            extraction_retry_initial_delay=config.extraction_retry_initial_delay,
            extraction_retry_max_delay=config.extraction_retry_max_delay,
            extraction_retry_backoff_factor=config.extraction_retry_backoff_factor,
            extraction_temperature=config.extraction_temperature,
            extraction_max_tokens=config.extraction_max_tokens,
            extraction_max_text_length=config.extraction_max_text_length,
            cache_enabled=config.extraction_cache_enabled,
            cache_dir=config.extraction_cache_dir,
        ),
    )


def _bow_views_for_config(config: MultiModelAgenticOracleConfig) -> List[BoWViewConfig]:
    if config.bow_views_json:
        with open(config.bow_views_json) as f:
            payload = json.load(f)
        entries = payload.get("bow_views", payload) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError("--bow-views-json must contain a list or {'bow_views': [...]}")
        return [
            entry if isinstance(entry, BoWViewConfig) else BoWViewConfig(**entry)
            for entry in entries
        ]

    grid = str(config.bow_view_grid).strip().lower()
    if grid == "default_broad":
        return []
    if grid == "linear_sweep":
        return [
            BoWViewConfig(
                name="linear_unigram_c0p5",
                bow_model="linear",
                ngram_range_min=1,
                ngram_range_max=1,
                max_features=config.max_features,
                min_df=config.min_df,
                max_df=config.max_df,
                logistic_c=0.5,
                ridge_alpha=max(config.ridge_alpha, 20.0),
            ),
            BoWViewConfig(
                name="linear_1_2",
                bow_model="linear",
                ngram_range_min=1,
                ngram_range_max=2,
                max_features=config.max_features,
                min_df=config.min_df,
                max_df=config.max_df,
                logistic_c=config.logistic_c,
                ridge_alpha=config.ridge_alpha,
            ),
            BoWViewConfig(
                name="linear_1_3",
                bow_model="linear",
                ngram_range_min=1,
                ngram_range_max=3,
                max_features=config.max_features,
                min_df=config.min_df,
                max_df=config.max_df,
                logistic_c=config.logistic_c,
                ridge_alpha=config.ridge_alpha,
            ),
            BoWViewConfig(
                name="linear_2_4_min_df3",
                bow_model="linear",
                ngram_range_min=2,
                ngram_range_max=4,
                max_features=config.max_features,
                min_df=min(config.min_df, 3),
                max_df=config.max_df,
                logistic_c=config.logistic_c,
                ridge_alpha=config.ridge_alpha,
            ),
        ]
    if grid == "cli_single":
        return [
            BoWViewConfig(
                name="cli_view",
                max_features=config.max_features,
                min_df=config.min_df,
                max_df=config.max_df,
                ngram_range_min=config.ngram_range_min,
                ngram_range_max=config.ngram_range_max,
                bow_model=config.bow_model,
                logistic_c=config.logistic_c,
                ridge_alpha=config.ridge_alpha,
            )
        ]
    raise ValueError(
        "--bow-view-grid must be one of default_broad, linear_sweep, or cli_single"
    )


def _load_dataset(config: MultiModelAgenticOracleConfig, parquet_file: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_file).reset_index(drop=True)
    if config.sample_size is not None and config.sample_size < len(df):
        df = (
            df.sample(n=config.sample_size, random_state=config.seed)
            .sort_index()
            .reset_index(drop=True)
        )
    if config.text_max_chars is not None:
        df["clinical_text"] = df["clinical_text"].astype(str).str.slice(0, config.text_max_chars)
    return df


def _metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"n_rows": int(len(results_df))}
    if {"true_ite_prob", "pred_ite_prob"}.issubset(results_df.columns):
        true_ite = results_df["true_ite_prob"].to_numpy(dtype=float)
        pred = results_df["pred_ite_prob"].to_numpy(dtype=float)
        metrics["ite_mse"] = float(mean_squared_error(true_ite, pred))
        metrics["ite_mae"] = float(np.mean(np.abs(true_ite - pred)))
        if np.std(true_ite) > 0 and np.std(pred) > 0:
            metrics["ite_corr"] = float(np.corrcoef(true_ite, pred)[0, 1])
    if {"treatment_indicator", "pred_propensity_prob"}.issubset(results_df.columns):
        try:
            metrics["treatment_auroc"] = float(
                roc_auc_score(
                    results_df["treatment_indicator"],
                    results_df["pred_propensity_prob"],
                )
            )
        except ValueError:
            metrics["treatment_auroc"] = None
    if {"outcome_indicator", "pred_outcome_prob"}.issubset(results_df.columns):
        try:
            metrics["outcome_auroc"] = float(
                roc_auc_score(
                    results_df["outcome_indicator"],
                    results_df["pred_outcome_prob"],
                )
            )
        except ValueError:
            metrics["outcome_auroc"] = None
    if "selected_feature_names" in results_df.columns:
        selected_sets = sorted(set(results_df["selected_feature_names"].fillna("")))
        metrics["selected_feature_sets"] = selected_sets
    if "selected_feature_roles" in results_df.columns:
        selected_role_sets = sorted(
            set(results_df["selected_feature_roles"].fillna(""))
        )
        metrics["selected_feature_role_sets"] = selected_role_sets
    if "selected_confounder_names" in results_df.columns:
        confounder_sets = sorted(
            set(results_df["selected_confounder_names"].fillna(""))
        )
        metrics["selected_confounder_sets"] = confounder_sets
    if "selected_effect_modifier_names" in results_df.columns:
        effect_modifier_sets = sorted(
            set(results_df["selected_effect_modifier_names"].fillna(""))
        )
        metrics["selected_effect_modifier_sets"] = effect_modifier_sets
    return metrics


def _run_one(config: MultiModelAgenticOracleConfig, output_dir: Path) -> Dict[str, Any]:
    parquet_file = _resolve_parquet_file(config.dataset_path)
    df = _load_dataset(config, parquet_file)
    applied = _make_applied_config(config, parquet_file)
    run_hash = config.config_hash()
    prediction_dir = output_dir / "multi_model_agentic_predictions" / run_hash
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / "predictions.parquet"

    logger.info(
        "Running multi-model agentic forest dataset=%s rows=%s hash=%s",
        config.dataset_name,
        len(df),
        run_hash,
    )
    run_multi_model_agentic_forest(
        df,
        applied,
        prediction_path,
        num_workers=config.num_workers,
    )
    results_df = pd.read_parquet(prediction_path)
    result = {
        **asdict(config),
        "config_hash": run_hash,
        "prediction_path": str(prediction_path),
        "metrics": _metrics(results_df),
    }
    return result


def _append_results(output_dir: Path, result_rows: Sequence[Dict[str, Any]]) -> None:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for result in result_rows:
        with open(results_dir / f"{result['config_hash']}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
    flat_rows = []
    for result in result_rows:
        row = {
            key: value
            for key, value in result.items()
            if key not in {"metrics"}
            and not isinstance(value, (list, dict))
        }
        row.update(result.get("metrics", {}))
        flat_rows.append(row)
    frame = pd.DataFrame(flat_rows)
    frame.to_csv(output_dir / "all_results.csv", index=False)
    frame.to_json(output_dir / "all_results.jsonl", orient="records", lines=True)
    frame.to_parquet(output_dir / "all_results.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-model BoW-guided agentic explicit-feature causal forest"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--text-max-chars", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Worker budget used by 'auto' parallelism settings.",
    )

    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--effect-folds", type=int, default=5)
    parser.add_argument(
        "--bow-view-grid",
        default="default_broad",
        choices=["default_broad", "linear_sweep", "cli_single"],
        help=(
            "BoW view preset. default_broad uses the library's multi-view grid; "
            "linear_sweep uses linear TF-IDF views only; cli_single uses the "
            "single view defined by --bow-model and n-gram/vectorizer flags."
        ),
    )
    parser.add_argument(
        "--bow-views-json",
        default=None,
        help="Optional JSON list, or {'bow_views': [...]}, of BoWViewConfig-shaped views.",
    )
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--ngram-range-min", type=int, default=1)
    parser.add_argument("--ngram-range-max", type=int, default=3)
    parser.add_argument(
        "--bow-model",
        default="linear",
        choices=["linear", "extratrees", "random_forest", "xgboost"],
        help=(
            "BoW learner family for nuisance and pseudo-target models. "
            "linear is sparse logistic/ridge; tree options allow feature interactions."
        ),
    )
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument(
        "--prespecified-features-json",
        default=None,
        help=(
            "Optional JSON file with pre-specified variables to extract before "
            "BoW discovery. Accepted keys: features, confounders, effect_modifiers."
        ),
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--top-n-features", type=int, default=100)
    parser.add_argument("--candidate-proposals-per-fold", type=int, default=30)
    parser.add_argument(
        "--no-candidate-consistency",
        action="store_true",
        help="Disable inner-fold candidate consistency selection.",
    )
    parser.add_argument("--candidate-consistency-inner-folds", type=int, default=3)
    parser.add_argument("--candidate-consistency-min-folds", type=int, default=2)
    parser.add_argument(
        "--candidate-consistency-min-fold-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--candidate-consistency-recovery-max-candidates",
        type=int,
        default=12,
        help="Maximum below-threshold candidates shown to the consistency agent for recovery.",
    )
    parser.add_argument(
        "--candidate-consistency-parallelism",
        default="1",
        help=(
            "Parallelism for inner-fold agentic consistency candidate proposal: "
            "'auto' uses runner workers, or pass a positive integer."
        ),
    )
    parser.add_argument(
        "--no-extracted-feature-review",
        action="store_true",
        help="Disable post-extraction simple-model review and agent revision.",
    )
    parser.add_argument("--extracted-feature-review-max-rounds", type=int, default=3)
    parser.add_argument(
        "--extracted-feature-review-auc-margin",
        type=float,
        default=0.02,
        help="Allowed AUC gap below BoW/embedding/HTR benchmarks before review fails.",
    )
    parser.add_argument(
        "--extracted-feature-review-loss-relative-margin",
        type=float,
        default=0.05,
        help="Allowed relative loss/R-loss excess over BoW benchmarks.",
    )
    parser.add_argument(
        "--extracted-feature-review-min-benchmark-auc",
        type=float,
        default=0.55,
        help="Minimum benchmark AUC before an AUC comparison is enforced.",
    )

    parser.add_argument(
        "--enable-embedding-contrast",
        dest="enable_embedding_contrast",
        action="store_true",
        help="Keep patient-level embedding contrast retrieval evidence enabled.",
    )
    parser.add_argument(
        "--disable-embedding-contrast",
        dest="enable_embedding_contrast",
        action="store_false",
        help="Disable required embedding contrast retrieval evidence for this run.",
    )
    parser.set_defaults(enable_embedding_contrast=True)
    parser.add_argument("--embedding-model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument(
        "--embedding-cache-dir",
        default=None,
        help=(
            "Directory for embedding chunk cache. Default: "
            "{dataset_dir}/.oci_cache/embedding_contrast."
        ),
    )
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-chunk-size-words", type=int, default=256)
    parser.add_argument("--embedding-chunk-overlap-words", type=int, default=64)
    parser.add_argument("--embedding-max-chunks", type=int, default=64)
    parser.add_argument(
        "--embedding-chunk-selection",
        default="last",
        choices=["first", "last"],
        help="Which chunks to keep when a patient has more than --embedding-max-chunks.",
    )
    parser.add_argument("--embedding-top-k-chunks-per-tail", type=int, default=12)
    parser.add_argument("--embedding-max-chunks-per-patient", type=int, default=2)
    parser.add_argument("--embedding-min-probe-auc", type=float, default=0.0)
    parser.add_argument("--embedding-pseudo-target-quantile", type=float, default=0.20)
    parser.add_argument(
        "--embedding-unweighted-pseudo-target",
        action="store_true",
        help="Use unweighted R-pseudo contrasts instead of treatment-residual squared weights.",
    )
    parser.add_argument(
        "--embedding-disable-cell-contrasts",
        action="store_true",
        help="Disable within-arm outcome and treatment-outcome 2x2 interaction contrasts.",
    )
    parser.add_argument(
        "--embedding-disable-orthogonal-r-score-contrasts",
        action="store_true",
        help="Disable high-vs-low orthogonal R-score embedding contrasts.",
    )
    parser.add_argument(
        "--embedding-concept-phrase",
        action="append",
        default=[],
        help="Concept phrase to probe against embedding contrast directions. May be repeated.",
    )
    parser.add_argument(
        "--embedding-residualize-column",
        action="append",
        default=[],
        help="Column to residualize out of patient embeddings before contrasts. May be repeated.",
    )
    parser.add_argument(
        "--outer-parallelism",
        default="1",
        help="Parallelism for outer CV folds: 'auto' uses runner workers, or pass a positive integer.",
    )
    parser.add_argument(
        "--bow-parallel-backend",
        default="processes",
        choices=["processes", "threads"],
        help="Backend for BoW cross-fit fold jobs. Processes usually use CPU better for TF-IDF.",
    )
    parser.add_argument(
        "--fold-parallelism",
        default="auto",
        help=(
            "Parallelism for BoW nuisance/effect cross-fit folds: 'auto' uses "
            "num_workers from the runner, or pass a positive integer."
        ),
    )

    parser.add_argument("--cf-n-estimators", type=int, default=200)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--cf-max-depth", type=int, default=None)
    parser.add_argument("--cf-no-inference", action="store_true")

    parser.add_argument(
        "--agent-server-url",
        "--agent-server-urls",
        dest="agent_server_url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible agent endpoint, or comma-separated endpoints.",
    )
    parser.add_argument("--agent-model-name", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--agent-api-key", default="EMPTY")
    parser.add_argument("--agent-max-tokens", type=int, default=25000)
    parser.add_argument("--agent-request-max-retries", type=int, default=3)
    parser.add_argument("--agent-retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--agent-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--agent-retry-backoff-factor", type=float, default=2.0)
    parser.add_argument("--agent-save-context", action="store_true")
    parser.add_argument("--agent-save-raw-output", action="store_true")

    parser.add_argument(
        "--extraction-server-url",
        "--extraction-server-urls",
        dest="extraction_server_url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible extraction endpoint, or comma-separated endpoints.",
    )
    parser.add_argument("--extraction-model-name", default="Qwen/Qwen3.6-27B")
    parser.add_argument(
        "--extraction-mode",
        default="server",
        choices=["server", "start_server", "python_api"],
    )
    parser.add_argument("--extraction-reasoning-parser", default="auto")
    parser.add_argument("--extraction-batch-size", type=int, default=100)
    parser.add_argument("--extraction-max-retries", type=int, default=3)
    parser.add_argument("--extraction-retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--extraction-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--extraction-retry-backoff-factor", type=float, default=2.0)
    parser.add_argument("--extraction-max-tokens", type=int, default=25000)
    parser.add_argument("--extraction-max-text-length", type=int, default=400000)
    parser.add_argument("--extraction-cache-dir", default=None)
    parser.add_argument("--no-extraction-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = MultiModelAgenticOracleConfig(
        dataset_path=args.dataset,
        dataset_name=Path(args.dataset).name,
        n_folds=args.n_folds,
        seed=args.seed,
        sample_size=args.sample_size,
        text_max_chars=args.text_max_chars,
        num_workers=args.num_workers,
        nuisance_folds=args.nuisance_folds,
        effect_folds=args.effect_folds,
        bow_view_grid=args.bow_view_grid,
        bow_views_json=args.bow_views_json,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range_min=args.ngram_range_min,
        ngram_range_max=args.ngram_range_max,
        bow_model=args.bow_model,
        logistic_c=args.logistic_c,
        prespecified_features_json=args.prespecified_features_json,
        ridge_alpha=args.ridge_alpha,
        top_n_features=args.top_n_features,
        candidate_proposals_per_fold=args.candidate_proposals_per_fold,
        candidate_consistency_enabled=not args.no_candidate_consistency,
        candidate_consistency_inner_folds=args.candidate_consistency_inner_folds,
        candidate_consistency_min_folds=args.candidate_consistency_min_folds,
        candidate_consistency_min_fold_fraction=args.candidate_consistency_min_fold_fraction,
        candidate_consistency_recovery_max_candidates=(
            args.candidate_consistency_recovery_max_candidates
        ),
        candidate_consistency_parallelism=args.candidate_consistency_parallelism,
        extracted_feature_review_enabled=not args.no_extracted_feature_review,
        extracted_feature_review_max_rounds=args.extracted_feature_review_max_rounds,
        extracted_feature_review_auc_margin=args.extracted_feature_review_auc_margin,
        extracted_feature_review_loss_relative_margin=(
            args.extracted_feature_review_loss_relative_margin
        ),
        extracted_feature_review_min_benchmark_auc=(
            args.extracted_feature_review_min_benchmark_auc
        ),
        outer_parallelism=args.outer_parallelism,
        bow_parallel_backend=args.bow_parallel_backend,
        fold_parallelism=args.fold_parallelism,
        embedding_contrast_enabled=args.enable_embedding_contrast,
        embedding_model_name=args.embedding_model_name,
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_chunk_size_words=args.embedding_chunk_size_words,
        embedding_chunk_overlap_words=args.embedding_chunk_overlap_words,
        embedding_max_chunks=args.embedding_max_chunks,
        embedding_chunk_selection=args.embedding_chunk_selection,
        embedding_top_k_chunks_per_tail=args.embedding_top_k_chunks_per_tail,
        embedding_max_chunks_per_patient=args.embedding_max_chunks_per_patient,
        embedding_min_probe_auc=args.embedding_min_probe_auc,
        embedding_pseudo_target_quantile=args.embedding_pseudo_target_quantile,
        embedding_pseudo_target_weighted=not args.embedding_unweighted_pseudo_target,
        embedding_include_cell_contrasts=not args.embedding_disable_cell_contrasts,
        embedding_include_orthogonal_r_score_contrasts=(
            not args.embedding_disable_orthogonal_r_score_contrasts
        ),
        embedding_concept_phrases=args.embedding_concept_phrase,
        embedding_residualize_columns=args.embedding_residualize_column,
        cf_n_estimators=args.cf_n_estimators,
        cf_min_samples_leaf=args.cf_min_samples_leaf,
        cf_max_depth=args.cf_max_depth,
        cf_inference=not args.cf_no_inference,
        agent_server_url=args.agent_server_url,
        agent_model_name=args.agent_model_name,
        agent_api_key=args.agent_api_key,
        agent_max_tokens=args.agent_max_tokens,
        agent_request_max_retries=args.agent_request_max_retries,
        agent_retry_initial_delay=args.agent_retry_initial_delay,
        agent_retry_max_delay=args.agent_retry_max_delay,
        agent_retry_backoff_factor=args.agent_retry_backoff_factor,
        agent_save_context=args.agent_save_context,
        agent_save_raw_output=args.agent_save_raw_output,
        extraction_server_url=args.extraction_server_url,
        extraction_model_name=args.extraction_model_name,
        extraction_mode=args.extraction_mode,
        extraction_reasoning_parser=args.extraction_reasoning_parser,
        extraction_batch_size=args.extraction_batch_size,
        extraction_max_retries=args.extraction_max_retries,
        extraction_retry_initial_delay=args.extraction_retry_initial_delay,
        extraction_retry_max_delay=args.extraction_retry_max_delay,
        extraction_retry_backoff_factor=args.extraction_retry_backoff_factor,
        extraction_max_tokens=args.extraction_max_tokens,
        extraction_max_text_length=args.extraction_max_text_length,
        extraction_cache_enabled=not args.no_extraction_cache,
        extraction_cache_dir=args.extraction_cache_dir,
    )

    try:
        result = _run_one(config, output_dir)
        _append_results(output_dir, [result])
        logger.info("Completed multi-model agentic forest: %s", result["metrics"])
    except Exception:
        logger.error("Multi-model agentic forest run failed")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
