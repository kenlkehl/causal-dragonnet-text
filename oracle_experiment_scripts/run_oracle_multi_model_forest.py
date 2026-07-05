#!/usr/bin/env python
"""Oracle runner for the integrated two-stage multi-model forest."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import (  # noqa: E402
    AppliedInferenceConfig,
    MultiModelForestConfig,
    normalize_multi_model_feature_discovery_methods,
)
from oci.inference.multi_model_forest import run_multi_model_forest  # noqa: E402
from run_oracle_multi_model_agentic_forest import (  # noqa: E402
    MultiModelAgenticOracleConfig,
    _append_results,
    _load_dataset,
    _make_applied_config as _make_agentic_applied_config,
    _metrics,
)
from run_oracle_multi_model_forest_agent_optional import (  # noqa: E402
    _normalize_embedding_cache_dir_arg,
    _resolve_oracle_parquet_file_for_optional_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MultiModelForestOracleConfig(MultiModelAgenticOracleConfig):
    stage: str = "all"
    cpus_total: Optional[int] = None
    gpu_ids: Optional[List[int]] = None
    htr_jobs_per_gpu: int = 1
    force_stage1: bool = False
    force_stage2: bool = False

    def primary_hash(self) -> str:
        payload = asdict(self)
        for key in [
            "stage",
            "force_stage1",
            "force_stage2",
            "agent_server_url",
            "agent_model_name",
            "agent_api_key",
            "agent_temperature",
            "agent_max_tokens",
            "agent_request_max_retries",
            "agent_retry_initial_delay",
            "agent_retry_max_delay",
            "agent_retry_backoff_factor",
            "agent_request_timeout",
            "agent_save_context",
            "agent_save_raw_output",
            "extraction_server_url",
            "extraction_model_name",
            "extraction_mode",
            "extraction_reasoning_parser",
            "extraction_batch_size",
            "extraction_max_retries",
            "extraction_retry_initial_delay",
            "extraction_retry_max_delay",
            "extraction_retry_backoff_factor",
            "extraction_request_timeout",
            "extraction_temperature",
            "extraction_max_tokens",
            "extraction_max_text_length",
            "extraction_cache_enabled",
            "extraction_cache_dir",
            "cpus_total",
            "gpu_ids",
            "htr_device",
            "htr_gpu_ids",
            "htr_jobs_per_gpu",
            "num_workers",
        ]:
            payload.pop(key, None)
        return _hash_payload(payload)

    def agentic_hash(self) -> str:
        payload = {
            "primary_hash": self.primary_hash(),
            "agent_server_url": self.agent_server_url,
            "agent_model_name": self.agent_model_name,
            "agent_temperature": self.agent_temperature,
            "agent_max_tokens": self.agent_max_tokens,
            "agent_request_max_retries": self.agent_request_max_retries,
            "agent_retry_initial_delay": self.agent_retry_initial_delay,
            "agent_retry_max_delay": self.agent_retry_max_delay,
            "agent_retry_backoff_factor": self.agent_retry_backoff_factor,
            "extraction_server_url": self.extraction_server_url,
            "extraction_model_name": self.extraction_model_name,
            "extraction_mode": self.extraction_mode,
            "extraction_reasoning_parser": self.extraction_reasoning_parser,
            "extraction_batch_size": self.extraction_batch_size,
            "extraction_max_retries": self.extraction_max_retries,
            "extraction_retry_initial_delay": self.extraction_retry_initial_delay,
            "extraction_retry_max_delay": self.extraction_retry_max_delay,
            "extraction_retry_backoff_factor": self.extraction_retry_backoff_factor,
            "extraction_temperature": self.extraction_temperature,
            "extraction_max_tokens": self.extraction_max_tokens,
            "extraction_max_text_length": self.extraction_max_text_length,
            "extraction_cache_enabled": self.extraction_cache_enabled,
            "extraction_cache_dir": self.extraction_cache_dir,
            "candidate_proposals_per_fold": self.candidate_proposals_per_fold,
            "candidate_consistency_enabled": self.candidate_consistency_enabled,
            "candidate_consistency_inner_folds": self.candidate_consistency_inner_folds,
            "candidate_consistency_min_folds": self.candidate_consistency_min_folds,
            "candidate_consistency_min_fold_fraction": (
                self.candidate_consistency_min_fold_fraction
            ),
            "candidate_consistency_recovery_max_candidates": (
                self.candidate_consistency_recovery_max_candidates
            ),
            "extracted_feature_review_enabled": self.extracted_feature_review_enabled,
            "extracted_feature_review_max_rounds": self.extracted_feature_review_max_rounds,
            "extracted_feature_review_auc_margin": self.extracted_feature_review_auc_margin,
            "extracted_feature_review_loss_relative_margin": (
                self.extracted_feature_review_loss_relative_margin
            ),
            "extracted_feature_review_min_benchmark_auc": (
                self.extracted_feature_review_min_benchmark_auc
            ),
        }
        return _hash_payload(payload)


def _hash_payload(payload: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _make_applied_config(
    config: MultiModelForestOracleConfig,
    parquet_file: Path,
) -> AppliedInferenceConfig:
    applied = _make_agentic_applied_config(config, parquet_file)
    applied.architecture.model_type = "multi_model_forest"
    mm_data = asdict(applied.architecture.multi_model_agentic_forest)
    mm_data["cpus_total"] = config.cpus_total
    mm_data["htr_jobs_per_gpu"] = int(config.htr_jobs_per_gpu)
    mm_config = MultiModelForestConfig(**mm_data)
    applied.architecture.multi_model_forest = mm_config
    applied.architecture.multi_model_agentic_forest = mm_config
    applied.architecture.multi_model_forest_agent_optional = mm_config
    return applied


def _run_one(config: MultiModelForestOracleConfig, output_dir: Path) -> Dict[str, Any]:
    parquet_file = _normalize_run_inputs(config)
    df = _load_dataset(config, parquet_file)
    applied = _make_applied_config(config, parquet_file)
    primary_hash = config.primary_hash()
    agentic_hash = config.agentic_hash()
    run_dir = output_dir / "multi_model_forest" / primary_hash
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "primary_predictions.parquet"

    logger.info(
        "Running multi-model forest dataset=%s rows=%s primary_hash=%s agentic_hash=%s stage=%s",
        config.dataset_name,
        len(df),
        primary_hash,
        agentic_hash,
        config.stage,
    )
    if applied.architecture.multi_model_forest.embedding_contrast.enabled:
        logger.info(
            "Embedding contrast cache root: %s",
            applied.architecture.multi_model_forest.embedding_contrast.cache_dir
            or str(Path(parquet_file).resolve().parent / ".oci_cache" / "embedding_contrast"),
        )

    run_multi_model_forest(
        df,
        applied,
        prediction_path,
        device=config.htr_device,
        gpu_ids=config.gpu_ids,
        num_workers=config.cpus_total or config.num_workers,
        stage=config.stage,
        cpus_total=config.cpus_total,
        htr_jobs_per_gpu=config.htr_jobs_per_gpu,
        force_stage1=config.force_stage1,
        force_stage2=config.force_stage2,
        agentic_run_id=agentic_hash,
    )

    result = {
        **asdict(config),
        "config_hash": _hash_payload(
            {"primary_hash": primary_hash, "agentic_hash": agentic_hash, "stage": config.stage}
        ),
        "primary_hash": primary_hash,
        "agentic_hash": agentic_hash,
        "prediction_path": str(prediction_path),
        "artifact_dir": str(run_dir),
        "stage": config.stage,
    }
    if prediction_path.exists():
        result["metrics"] = _metrics(pd.read_parquet(prediction_path))
    stage2_path = run_dir / "stage2_agentic" / agentic_hash / "agentic_predictions.parquet"
    if stage2_path.exists():
        result["agentic_prediction_path"] = str(stage2_path)
        result["agentic_metrics"] = _metrics(pd.read_parquet(stage2_path))
    return result


def _normalize_run_inputs(config: MultiModelForestOracleConfig) -> Path:
    parquet_file = _resolve_oracle_parquet_file_for_optional_cache(config)
    if config.dataset_path != str(parquet_file):
        logger.info("Resolved --dataset %s to %s", config.dataset_path, parquet_file)
        config.dataset_path = str(parquet_file)
    normalized_cache_dir = _normalize_embedding_cache_dir_arg(config.embedding_cache_dir)
    if normalized_cache_dir != config.embedding_cache_dir:
        logger.info(
            "Using embedding cache root %s from --embedding-cache-dir %s",
            normalized_cache_dir,
            config.embedding_cache_dir,
        )
        config.embedding_cache_dir = normalized_cache_dir
    return parquet_file


def _append_list_arg(parser: argparse.ArgumentParser, name: str, **kwargs: Any) -> None:
    parser.add_argument(name, action="append", default=[], **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run integrated multi-model forest Stage 1/Stage 2 workflow"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", default="all", choices=["all", "stage1", "stage2"])
    parser.add_argument("--force-stage1", action="store_true")
    parser.add_argument("--force-stage2", action="store_true")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--text-max-chars", type=int, default=None)
    parser.add_argument(
        "--cpus-total",
        "--num-workers",
        dest="cpus_total",
        type=int,
        default=None,
        help="Total CPU budget for this run. Replaces outer/fold parallelism flags.",
    )
    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--effect-folds", type=int, default=5)
    parser.add_argument("--feature-discovery-methods", nargs="+", default=None)
    parser.add_argument(
        "--bow-view-grid",
        default="default_broad",
        choices=["default_broad", "linear_sweep", "cli_single"],
    )
    parser.add_argument("--bow-views-json", default=None)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--ngram-range-min", type=int, default=1)
    parser.add_argument("--ngram-range-max", type=int, default=3)
    parser.add_argument(
        "--bow-model",
        default="linear",
        choices=["linear", "extratrees", "random_forest", "xgboost"],
    )
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--top-n-features", type=int, default=100)
    parser.add_argument("--prespecified-features-json", default=None)
    parser.add_argument("--allow-full-data-refit", action="store_true")
    parser.add_argument("--allow-extraction-truncation", action="store_true")

    parser.add_argument("--candidate-proposals-per-fold", type=int, default=30)
    parser.add_argument("--no-candidate-consistency", action="store_true")
    parser.add_argument("--candidate-consistency-inner-folds", type=int, default=3)
    parser.add_argument("--candidate-consistency-min-folds", type=int, default=2)
    parser.add_argument("--candidate-consistency-min-fold-fraction", type=float, default=0.5)
    parser.add_argument("--candidate-consistency-recovery-max-candidates", type=int, default=12)
    parser.add_argument("--no-extracted-feature-review", action="store_true")
    parser.add_argument("--extracted-feature-review-max-rounds", type=int, default=3)
    parser.add_argument("--extracted-feature-review-auc-margin", type=float, default=0.02)
    parser.add_argument(
        "--extracted-feature-review-loss-relative-margin",
        type=float,
        default=0.05,
    )
    parser.add_argument("--extracted-feature-review-min-benchmark-auc", type=float, default=0.55)
    parser.add_argument("--min-feature-coverage", type=float, default=0.70)

    parser.add_argument("--htr-device", default="cuda:0")
    parser.add_argument("--gpu-ids", "--htr-gpu-ids", dest="gpu_ids", type=int, nargs="+")
    parser.add_argument("--htr-jobs-per-gpu", type=int, default=1)

    parser.add_argument(
        "--enable-embedding-contrast",
        dest="embedding_contrast_enabled",
        action="store_true",
    )
    parser.add_argument(
        "--disable-embedding-contrast",
        dest="embedding_contrast_enabled",
        action="store_false",
    )
    parser.set_defaults(embedding_contrast_enabled=True)
    parser.add_argument("--embedding-model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--embedding-cache-dir", default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--embedding-max-seq-length", type=int, default=1024)
    parser.add_argument("--embedding-chunk-size-words", type=int, default=256)
    parser.add_argument("--embedding-chunk-overlap-words", type=int, default=64)
    parser.add_argument("--embedding-max-chunks", type=int, default=64)
    parser.add_argument("--embedding-chunk-selection", default="last", choices=["first", "last"])
    parser.add_argument("--embedding-top-k-chunks-per-tail", type=int, default=12)
    parser.add_argument("--embedding-max-chunks-per-patient", type=int, default=2)
    parser.add_argument("--embedding-min-probe-auc", type=float, default=0.0)
    parser.add_argument("--embedding-pseudo-target-quantile", type=float, default=0.20)
    parser.add_argument("--embedding-unweighted-pseudo-target", action="store_true")
    parser.add_argument("--embedding-disable-cell-contrasts", action="store_true")
    parser.add_argument("--embedding-disable-confounder-vector-contrast", action="store_true")
    parser.add_argument(
        "--embedding-disable-residualized-interaction-contrast",
        action="store_true",
    )
    parser.add_argument("--embedding-disable-orthogonal-r-score-contrasts", action="store_true")
    _append_list_arg(parser, "--embedding-external-cache-dir")
    parser.add_argument("--embedding-external-top-k-chunks-per-tail", type=int, default=12)
    _append_list_arg(parser, "--embedding-concept-phrase")
    _append_list_arg(parser, "--embedding-residualize-column")

    parser.add_argument("--cf-n-estimators", type=int, default=200)
    parser.add_argument("--cf-min-samples-leaf", type=int, default=10)
    parser.add_argument("--cf-max-depth", type=int, default=None)
    parser.add_argument("--cf-max-features", default="sqrt")
    parser.add_argument("--cf-no-inference", action="store_true")

    parser.add_argument(
        "--agent-server-url",
        "--agent-server-urls",
        dest="agent_server_url",
        default="http://localhost:8000/v1",
    )
    parser.add_argument("--agent-model-name", default="auto")
    parser.add_argument("--agent-api-key", default="EMPTY")
    parser.add_argument("--agent-max-tokens", type=int, default=25000)
    parser.add_argument("--agent-request-max-retries", type=int, default=3)
    parser.add_argument("--agent-retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--agent-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--agent-retry-backoff-factor", type=float, default=2.0)
    parser.add_argument("--agent-request-timeout", type=float, default=900.0)
    parser.add_argument("--agent-save-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--agent-save-raw-output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--extraction-server-url",
        "--extraction-server-urls",
        dest="extraction_server_url",
        default="http://localhost:8000/v1",
    )
    parser.add_argument("--extraction-model-name", default="auto")
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
    parser.add_argument("--extraction-request-timeout", type=float, default=900.0)
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
    selected_methods = normalize_multi_model_feature_discovery_methods(
        args.feature_discovery_methods,
        source="feature_discovery_methods",
    )
    embedding_enabled = (
        "embedding_contrast" in selected_methods
        if selected_methods is not None
        else args.embedding_contrast_enabled
    )
    config = MultiModelForestOracleConfig(
        dataset_path=args.dataset,
        dataset_name=Path(args.dataset).name,
        stage=args.stage,
        n_folds=args.n_folds,
        seed=args.seed,
        sample_size=args.sample_size,
        text_max_chars=args.text_max_chars,
        num_workers=args.cpus_total or 1,
        cpus_total=args.cpus_total,
        gpu_ids=args.gpu_ids,
        htr_gpu_ids=args.gpu_ids,
        htr_jobs_per_gpu=args.htr_jobs_per_gpu,
        force_stage1=args.force_stage1,
        force_stage2=args.force_stage2,
        nuisance_folds=args.nuisance_folds,
        effect_folds=args.effect_folds,
        feature_discovery_methods=args.feature_discovery_methods,
        bow_view_grid=args.bow_view_grid,
        bow_views_json=args.bow_views_json,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range_min=args.ngram_range_min,
        ngram_range_max=args.ngram_range_max,
        bow_model=args.bow_model,
        logistic_c=args.logistic_c,
        ridge_alpha=args.ridge_alpha,
        top_n_features=args.top_n_features,
        prespecified_features_json=args.prespecified_features_json,
        candidate_proposals_per_fold=args.candidate_proposals_per_fold,
        candidate_consistency_enabled=not args.no_candidate_consistency,
        candidate_consistency_inner_folds=args.candidate_consistency_inner_folds,
        candidate_consistency_min_folds=args.candidate_consistency_min_folds,
        candidate_consistency_min_fold_fraction=args.candidate_consistency_min_fold_fraction,
        candidate_consistency_recovery_max_candidates=(
            args.candidate_consistency_recovery_max_candidates
        ),
        extracted_feature_review_enabled=not args.no_extracted_feature_review,
        extracted_feature_review_max_rounds=args.extracted_feature_review_max_rounds,
        extracted_feature_review_auc_margin=args.extracted_feature_review_auc_margin,
        extracted_feature_review_loss_relative_margin=(
            args.extracted_feature_review_loss_relative_margin
        ),
        extracted_feature_review_min_benchmark_auc=(
            args.extracted_feature_review_min_benchmark_auc
        ),
        require_honest_outer_split=not args.allow_full_data_refit,
        fail_on_extraction_truncation=not args.allow_extraction_truncation,
        htr_device=args.htr_device,
        embedding_contrast_enabled=embedding_enabled,
        embedding_model_name=args.embedding_model_name,
        embedding_cache_dir=args.embedding_cache_dir,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_max_seq_length=args.embedding_max_seq_length,
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
        embedding_include_confounder_vector_contrast=(
            not args.embedding_disable_confounder_vector_contrast
        ),
        embedding_include_residualized_interaction_contrast=(
            not args.embedding_disable_residualized_interaction_contrast
        ),
        embedding_include_orthogonal_r_score_contrasts=(
            not args.embedding_disable_orthogonal_r_score_contrasts
        ),
        embedding_external_cache_dirs=args.embedding_external_cache_dir,
        embedding_external_top_k_chunks_per_tail=args.embedding_external_top_k_chunks_per_tail,
        embedding_concept_phrases=args.embedding_concept_phrase,
        embedding_residualize_columns=args.embedding_residualize_column,
        cf_n_estimators=args.cf_n_estimators,
        cf_min_samples_leaf=args.cf_min_samples_leaf,
        cf_max_depth=args.cf_max_depth,
        cf_max_features=args.cf_max_features,
        cf_inference=not args.cf_no_inference,
        min_feature_coverage=args.min_feature_coverage,
        agent_server_url=args.agent_server_url,
        agent_model_name=args.agent_model_name,
        agent_api_key=args.agent_api_key,
        agent_max_tokens=args.agent_max_tokens,
        agent_request_max_retries=args.agent_request_max_retries,
        agent_retry_initial_delay=args.agent_retry_initial_delay,
        agent_retry_max_delay=args.agent_retry_max_delay,
        agent_retry_backoff_factor=args.agent_retry_backoff_factor,
        agent_request_timeout=args.agent_request_timeout,
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
        extraction_request_timeout=args.extraction_request_timeout,
        extraction_max_tokens=args.extraction_max_tokens,
        extraction_max_text_length=args.extraction_max_text_length,
        extraction_cache_enabled=not args.no_extraction_cache,
        extraction_cache_dir=args.extraction_cache_dir,
    )

    try:
        result = _run_one(config, output_dir)
        _append_results(output_dir, [result])
        logger.info(
            "Completed multi-model forest stage=%s metrics=%s agentic_metrics=%s",
            config.stage,
            result.get("metrics"),
            result.get("agentic_metrics"),
        )
    except Exception:
        logger.error("Multi-model forest run failed")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
