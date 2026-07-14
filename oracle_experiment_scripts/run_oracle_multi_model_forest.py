#!/usr/bin/env python
"""Oracle runner for the integrated two-stage multi-model forest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
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
    TfidfTopicDiscoveryConfig,
    normalize_tfidf_topic_feature_discovery_methods,
)
from oci.inference.multi_model_forest import run_multi_model_forest  # noqa: E402
from oci.inference.tfidf_topic_agentic_forest import (  # noqa: E402
    validate_tfidf_topic_stage2_handoff,
)
from oci.inference.tfidf_topic_score_selection import (  # noqa: E402
    TOPIC_SCORE_TEST_SCHEMA_VERSION,
)
from run_oracle_multi_model_agentic_forest import (  # noqa: E402
    _LLM_PROVIDER_CLI_CHOICES,
    MultiModelAgenticOracleConfig,
    _append_results,
    _agent_platform_publisher_model_name,
    _load_dataset,
    _make_applied_config as _make_agentic_applied_config,
    _metrics,
    _normalize_llm_provider,
    _parse_codex_extra_args,
    _resolve_oracle_parquet_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_EMBEDDING_CACHE_PREFIX = "cecnn_chunk_embeddings_"
_REQUIRED_EMBEDDING_CACHE_FILES = (
    "metadata.json",
    "chunk_embeddings.npy",
    "offsets.npy",
    "chunk_texts.jsonl",
)


@dataclass
class MultiModelForestOracleConfig(MultiModelAgenticOracleConfig):
    stage: str = "all"
    primary_run_id: Optional[str] = None
    agentic_run_id: Optional[str] = None
    cpus_total: Optional[int] = None
    gpu_ids: Optional[List[int]] = None
    htr_jobs_per_gpu: int = 1
    force_stage1: bool = False
    force_stage2: bool = False
    stage2_preflight_only: bool = False
    htr_device: Optional[str] = None
    htr_gpu_ids: Optional[List[int]] = None
    embedding_contrast_enabled: bool = False
    tfidf_topic_top_fraction: float = 0.10
    tfidf_topic_count: int = 100
    tfidf_topic_seeds: List[int] = None
    tfidf_topic_terms_per_topic: int = 15
    tfidf_topic_max_iter: int = 400
    tfidf_topic_stability_repeats: int = 30
    tfidf_topic_stability_fraction: float = 0.75
    tfidf_topic_score_test_enabled: bool = True
    tfidf_topic_score_test_bootstrap_repeats: int = 500
    tfidf_topic_score_test_bootstrap_top_topics: int = 0
    tfidf_topic_score_test_bootstrap_chunk_size: int = 100
    tfidf_topic_score_test_fdr_level: float = 0.20
    tfidf_topic_score_test_p_threshold: float = 0.10
    tfidf_topic_score_test_min_topics_per_bank: int = 5
    tfidf_topic_score_test_max_topics_per_bank: int = 20
    tfidf_topic_score_test_full_topic_min_inner_folds: int = 1
    tfidf_orphan_ngram_enabled: bool = True
    tfidf_orphan_min_abs_fit_score: float = 2.0
    tfidf_orphan_cluster_similarity_threshold: float = 0.25
    tfidf_orphan_cluster_max_terms: int = 15
    tfidf_orphan_cluster_neighbors: int = 20
    tfidf_orphan_fdr_level: float = 0.20
    tfidf_orphan_p_threshold: float = 0.10
    tfidf_orphan_min_selected_clusters: int = 5
    tfidf_orphan_max_selected_clusters: int = 20
    tfidf_orphan_full_min_inner_folds: int = 1
    topic_label_parallelism: int = 8
    max_variables_per_extraction_request: int = 10
    # Structured schema calls should not spend the response budget on hidden
    # chain-of-thought. These can be explicitly re-enabled for compatible models.
    agent_enable_thinking: Optional[bool] = False
    extraction_enable_thinking: Optional[bool] = False

    def __post_init__(self):
        if self.tfidf_topic_seeds is None:
            self.tfidf_topic_seeds = [42, 43, 44]

    def primary_hash(self) -> str:
        if self.primary_run_id:
            return str(self.primary_run_id).strip()
        data = asdict(self)
        keys = [
            "dataset_path", "dataset_name", "n_folds", "seed", "sample_size",
            "text_max_chars", "nuisance_folds", "feature_discovery_methods",
            "bow_view_grid", "bow_views_json", "max_features", "min_df", "max_df",
            "ngram_range_min", "ngram_range_max", "bow_model", "logistic_c",
            "ridge_alpha", "candidate_consistency_inner_folds",
            "tfidf_topic_top_fraction", "tfidf_topic_count", "tfidf_topic_seeds",
            "tfidf_topic_terms_per_topic", "tfidf_topic_max_iter",
            "tfidf_topic_stability_repeats", "tfidf_topic_stability_fraction",
            "tfidf_topic_score_test_enabled",
            "tfidf_topic_score_test_bootstrap_repeats",
            "tfidf_topic_score_test_bootstrap_top_topics",
            "tfidf_topic_score_test_bootstrap_chunk_size",
            "tfidf_topic_score_test_fdr_level",
            "tfidf_topic_score_test_p_threshold",
            "tfidf_topic_score_test_min_topics_per_bank",
            "tfidf_topic_score_test_max_topics_per_bank",
            "tfidf_topic_score_test_full_topic_min_inner_folds",
            "tfidf_orphan_ngram_enabled",
            "tfidf_orphan_min_abs_fit_score",
            "tfidf_orphan_cluster_similarity_threshold",
            "tfidf_orphan_cluster_max_terms",
            "tfidf_orphan_cluster_neighbors",
            "tfidf_orphan_fdr_level",
            "tfidf_orphan_p_threshold",
            "tfidf_orphan_min_selected_clusters",
            "tfidf_orphan_max_selected_clusters",
            "tfidf_orphan_full_min_inner_folds",
        ]
        payload = {key: data.get(key) for key in keys}
        payload["topic_score_test_schema_version"] = TOPIC_SCORE_TEST_SCHEMA_VERSION
        return _hash_payload(payload)

    def agentic_hash(self) -> str:
        if self.agentic_run_id:
            return str(self.agentic_run_id).strip()
        agent_provider = _normalize_llm_provider(self.agent_provider, source="agent_provider")
        extraction_provider = _normalize_llm_provider(
            self.extraction_provider,
            source="extraction_provider",
        )
        agent_model_name = (
            _agent_platform_publisher_model_name(self.agent_model_name)
            if agent_provider == "agent_platform"
            else self.agent_model_name
        )
        extraction_model_name = (
            _agent_platform_publisher_model_name(self.extraction_model_name)
            if extraction_provider == "agent_platform"
            else self.extraction_model_name
        )
        agent_platform_project = (
            self.agent_platform_project
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
            or os.environ.get("PROJECT_ID")
        )
        extraction_agent_platform_project = (
            self.extraction_agent_platform_project
            or self.agent_platform_project
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCLOUD_PROJECT")
            or os.environ.get("PROJECT_ID")
        )
        payload = {
            "primary_hash": self.primary_hash(),
            "stage2_schema_version": "tfidf_topic_agentic_forest_v7",
            "topic_filter_policy_version": "tfidf_topic_inner_score_policy_v5",
            "topic_score_test_schema_version": TOPIC_SCORE_TEST_SCHEMA_VERSION,
            "topic_prompt_version": "tfidf_topic_label_v2",
            "orphan_ngram_prompt_version": "tfidf_orphan_ngram_label_v1",
            "name_harmonization_prompt_version": "tfidf_topic_name_harmonization_v2",
            "global_dedup_prompt_version": "tfidf_topic_global_dedup_v2",
            "value_harmonization_prompt_version": "tfidf_topic_value_harmonization_v2",
            "canonical_registry_version": "tfidf_topic_canonical_registry_v4",
            "agent_provider": agent_provider,
            "agent_platform_project": agent_platform_project,
            "agent_platform_location": self.agent_platform_location,
            "agent_server_url": self.agent_server_url,
            "agent_model_name": agent_model_name,
            "agent_temperature": self.agent_temperature,
            "agent_max_tokens": self.agent_max_tokens,
            "agent_enable_thinking": self.agent_enable_thinking,
            "agent_request_max_retries": self.agent_request_max_retries,
            "agent_retry_initial_delay": self.agent_retry_initial_delay,
            "agent_retry_max_delay": self.agent_retry_max_delay,
            "agent_retry_backoff_factor": self.agent_retry_backoff_factor,
            "extraction_provider": extraction_provider,
            "extraction_agent_platform_project": extraction_agent_platform_project,
            "extraction_agent_platform_location": self.extraction_agent_platform_location,
            "extraction_server_url": self.extraction_server_url,
            "extraction_model_name": extraction_model_name,
            "extraction_mode": self.extraction_mode,
            "extraction_reasoning_parser": self.extraction_reasoning_parser,
            "extraction_enable_thinking": self.extraction_enable_thinking,
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
            "codex_executable": self.codex_executable,
            "codex_model_name": self.codex_model_name,
            "codex_reasoning_effort": self.codex_reasoning_effort,
            "codex_extra_args": self.codex_extra_args,
            "codex_parallelism": self.codex_parallelism,
            "candidate_proposals_per_fold": self.candidate_proposals_per_fold,
            "concept_inventory_enabled": self.concept_inventory_enabled,
            "concept_inventory_max_concepts": self.concept_inventory_max_concepts,
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
            "parsimony_review_enabled": self.parsimony_review_enabled,
            "topic_label_parallelism": self.topic_label_parallelism,
            "max_variables_per_extraction_request": (
                self.max_variables_per_extraction_request
            ),
        }
        return _hash_payload(payload)


def _hash_payload(payload: Dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _make_applied_config(
    config: MultiModelForestOracleConfig,
    parquet_file: Path,
) -> AppliedInferenceConfig:
    legacy_input = copy.deepcopy(config)
    legacy_input.feature_discovery_methods = ["bow"]
    legacy_input.embedding_contrast_enabled = False
    applied = _make_agentic_applied_config(legacy_input, parquet_file)
    applied.architecture.model_type = "multi_model_forest"
    mm_data = asdict(applied.architecture.multi_model_agentic_forest)
    mm_data["feature_discovery_methods"] = normalize_tfidf_topic_feature_discovery_methods(
        config.feature_discovery_methods,
        source="feature_discovery_methods",
    )
    mm_data["cpus_total"] = config.cpus_total
    mm_data["htr_jobs_per_gpu"] = int(config.htr_jobs_per_gpu)
    mm_data["tfidf_topic"] = asdict(
        TfidfTopicDiscoveryConfig(
            max_features=config.max_features,
            min_df=config.min_df,
            max_df=config.max_df,
            top_fraction=config.tfidf_topic_top_fraction,
            topic_count=config.tfidf_topic_count,
            topic_seeds=list(config.tfidf_topic_seeds),
            terms_per_topic=config.tfidf_topic_terms_per_topic,
            nmf_max_iter=config.tfidf_topic_max_iter,
            stability_repeats=config.tfidf_topic_stability_repeats,
            stability_fraction=config.tfidf_topic_stability_fraction,
            score_test_enabled=config.tfidf_topic_score_test_enabled,
            score_test_bootstrap_repeats=(
                config.tfidf_topic_score_test_bootstrap_repeats
            ),
            score_test_bootstrap_top_topics=(
                config.tfidf_topic_score_test_bootstrap_top_topics
            ),
            score_test_bootstrap_chunk_size=(
                config.tfidf_topic_score_test_bootstrap_chunk_size
            ),
            score_test_fdr_level=config.tfidf_topic_score_test_fdr_level,
            score_test_p_threshold=config.tfidf_topic_score_test_p_threshold,
            score_test_min_topics_per_bank=(
                config.tfidf_topic_score_test_min_topics_per_bank
            ),
            score_test_max_topics_per_bank=(
                config.tfidf_topic_score_test_max_topics_per_bank
            ),
            score_test_full_topic_min_inner_folds=(
                config.tfidf_topic_score_test_full_topic_min_inner_folds
            ),
            orphan_ngram_enabled=config.tfidf_orphan_ngram_enabled,
            orphan_ngram_min_abs_fit_score=(
                config.tfidf_orphan_min_abs_fit_score
            ),
            orphan_ngram_cluster_similarity_threshold=(
                config.tfidf_orphan_cluster_similarity_threshold
            ),
            orphan_ngram_cluster_max_terms=(
                config.tfidf_orphan_cluster_max_terms
            ),
            orphan_ngram_cluster_neighbors=(
                config.tfidf_orphan_cluster_neighbors
            ),
            orphan_ngram_fdr_level=config.tfidf_orphan_fdr_level,
            orphan_ngram_p_threshold=config.tfidf_orphan_p_threshold,
            orphan_ngram_min_selected_clusters=(
                config.tfidf_orphan_min_selected_clusters
            ),
            orphan_ngram_max_selected_clusters=(
                config.tfidf_orphan_max_selected_clusters
            ),
            orphan_ngram_full_min_inner_folds=(
                config.tfidf_orphan_full_min_inner_folds
            ),
            topic_label_parallelism=config.topic_label_parallelism,
            random_state=config.seed,
        )
    )
    mm_config = MultiModelForestConfig(**mm_data)
    applied.architecture.multi_model_forest = mm_config
    applied.architecture.multi_model_agentic_forest = mm_config
    applied.explicit_features.max_variables_per_extraction_request = int(
        config.max_variables_per_extraction_request
    )
    return applied


def _run_one(config: MultiModelForestOracleConfig, output_dir: Path) -> Dict[str, Any]:
    if config.primary_run_id and config.stage != "stage2":
        raise ValueError("--primary-run-id is only supported with --stage stage2")
    if config.agentic_run_id and config.stage != "stage2":
        raise ValueError("--agentic-run-id is only supported with --stage stage2")
    if config.stage2_preflight_only and config.stage != "stage2":
        raise ValueError("--stage2-preflight-only requires --stage stage2")
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

    if config.stage2_preflight_only:
        handoff_path = run_dir / "handoff" / "discovery_contexts.jsonl"
        report = validate_tfidf_topic_stage2_handoff(
            dataset=df,
            config=applied,
            handoff_path=handoff_path,
        )
        report_path = run_dir / "handoff" / "stage2_preflight.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return {
            **asdict(config),
            "config_hash": _hash_payload(
                {
                    "primary_hash": primary_hash,
                    "agentic_hash": agentic_hash,
                    "stage": config.stage,
                    "stage2_preflight_only": True,
                }
            ),
            "primary_hash": primary_hash,
            "agentic_hash": agentic_hash,
            "stage": config.stage,
            "stage2_preflight_only": True,
            "stage2_preflight_path": str(report_path),
            "stage2_preflight": report,
        }

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
        posthoc_oracle_path = (
            stage2_path.parent
            / "tfidf_topic_agentic_forest"
            / "posthoc_oracle_metrics.json"
        )
        if posthoc_oracle_path.exists():
            posthoc_oracle = json.loads(
                posthoc_oracle_path.read_text(encoding="utf-8")
            )
            result["agentic_posthoc_oracle_metrics"] = posthoc_oracle
            overall = posthoc_oracle.get("overall") or {}
            result["agentic_metrics"].update(
                {
                    "ite_corr": overall.get("pearson_correlation"),
                    "ite_spearman_corr": overall.get("spearman_correlation"),
                    "ite_mse": (
                        None
                        if overall.get("rmse") is None
                        else float(overall["rmse"]) ** 2
                    ),
                    "ite_mae": overall.get("mae"),
                    "oracle_evaluation_is_post_hoc": True,
                }
            )
    return result


def _normalize_run_inputs(config: MultiModelForestOracleConfig) -> Path:
    parquet_file = _resolve_oracle_parquet_file_for_cache(config)
    if config.dataset_path != str(parquet_file):
        logger.info("Resolved --dataset %s to %s", config.dataset_path, parquet_file)
        config.dataset_path = str(parquet_file)
    return parquet_file


def _resolve_oracle_parquet_file_for_cache(config: MultiModelForestOracleConfig) -> Path:
    """Resolve the v2 dataset without any embedding-cache side effects."""
    requested = Path(config.dataset_path).expanduser()
    if requested.is_dir():
        base = requested / "dataset.parquet"
        if base.exists():
            return base
    return _resolve_oracle_parquet_file(config.dataset_path)


def _embedding_contrast_requested(config: MultiModelForestOracleConfig) -> bool:
    del config
    return False


def _candidate_oracle_parquet_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return [
        candidate
        for candidate in (
            path / "dataset_with_extraction.parquet",
            path / "dataset.parquet",
        )
        if candidate.exists()
    ]


def _matching_complete_embedding_cache_path(
    config: MultiModelForestOracleConfig,
    parquet_file: Path,
) -> Optional[Path]:
    cache_root = _embedding_cache_root_for_parquet(config, parquet_file)
    cache_hash = _embedding_cache_hash_for_config(config, parquet_file)
    cache_path = cache_root / f"{_EMBEDDING_CACHE_PREFIX}{cache_hash}"
    if not all((cache_path / filename).exists() for filename in _REQUIRED_EMBEDDING_CACHE_FILES):
        return None
    try:
        with open(cache_path / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("cache_hash") != cache_hash:
        return None
    if metadata.get("storage_format") != "variable_length_chunks":
        return None
    return cache_path


def _embedding_cache_root_for_parquet(
    config: MultiModelForestOracleConfig,
    parquet_file: Path,
) -> Path:
    explicit = _normalize_embedding_cache_dir_arg(config.embedding_cache_dir)
    if explicit:
        return Path(explicit).expanduser()
    return _default_embedding_cache_root_for_parquet(parquet_file)


def _default_embedding_cache_root_for_parquet(parquet_file: Path) -> Path:
    return parquet_file.expanduser().resolve().parent / ".oci_cache" / "embedding_contrast"


def _normalize_embedding_cache_dir_arg(cache_dir: Optional[str]) -> Optional[str]:
    if cache_dir is None:
        return None
    path = Path(str(cache_dir)).expanduser()
    if path.name.startswith(_EMBEDDING_CACHE_PREFIX):
        return str(path.parent)
    return str(path)


def _embedding_cache_hash_for_config(
    config: MultiModelForestOracleConfig,
    parquet_file: Path,
) -> str:
    token_limit = (
        int(config.embedding_max_seq_length)
        if config.embedding_max_seq_length is not None
        else "model"
    )
    key = "|".join(
        [
            str(config.embedding_model_name),
            os.path.abspath(str(parquet_file)),
            f"words{int(config.embedding_chunk_size_words)}",
            f"overlap{int(config.embedding_chunk_overlap_words)}",
            f"max{int(config.embedding_max_chunks)}",
            "norm1",
            f"select{str(config.embedding_chunk_selection).strip().lower()}",
            f"tokmax{token_limit}",
            "chunker_word_then_token_bound_v1",
        ]
    )
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _append_list_arg(parser: argparse.ArgumentParser, name: str, **kwargs: Any) -> None:
    parser.add_argument(name, action="append", default=[], **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run integrated multi-model forest Stage 1/Stage 2 workflow"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", default="all", choices=["all", "stage1", "stage2"])
    parser.add_argument(
        "--primary-run-id",
        default=None,
        help=(
            "Existing multi_model_forest/<run_id> directory to reuse for Stage 2. "
            "Use with --stage stage2."
        ),
    )
    parser.add_argument(
        "--agentic-run-id",
        default=None,
        help=(
            "Existing stage2_agentic/<run_id> directory to resume. "
            "Use with --stage stage2; transport endpoints may change only when "
            "the resolved model identity is unchanged."
        ),
    )
    parser.add_argument("--force-stage1", action="store_true")
    parser.add_argument("--force-stage2", action="store_true")
    parser.add_argument(
        "--stage2-preflight-only",
        action="store_true",
        help=(
            "Validate the complete exact-scope Stage 1 handoff for Stage 2 "
            "without constructing an agent or extraction client. Requires "
            "--stage stage2."
        ),
    )
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
    parser.add_argument("--feature-discovery-methods", nargs="+", default=None)
    parser.add_argument(
        "--bow-view-grid",
        default="default_broad",
        choices=["default_broad", "linear_sweep", "cli_single"],
    )
    parser.add_argument("--bow-views-json", default=None)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.98)
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
    parser.add_argument("--tfidf-topic-top-fraction", type=float, default=0.10)
    parser.add_argument("--tfidf-topic-count", type=int, default=100)
    parser.add_argument("--tfidf-topic-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--tfidf-topic-terms-per-topic", type=int, default=15)
    parser.add_argument("--tfidf-topic-max-iter", type=int, default=400)
    parser.add_argument("--tfidf-topic-stability-repeats", type=int, default=30)
    parser.add_argument("--tfidf-topic-stability-fraction", type=float, default=0.75)
    parser.add_argument(
        "--tfidf-topic-score-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter agent-facing topics and their unique n-grams with exact "
            "inner-held-out score tests. The full outer context is mapped from "
            "the fixed inner policy and never uses outer-test labels."
        ),
    )
    parser.add_argument(
        "--tfidf-topic-score-test-bootstrap-repeats", type=int, default=500
    )
    parser.add_argument(
        "--tfidf-topic-score-test-bootstrap-top-topics",
        type=int,
        default=0,
        help=(
            "Number of topics per bank to multiplier-bootstrap; zero (default) "
            "calibrates the complete fitted topic family."
        ),
    )
    parser.add_argument(
        "--tfidf-topic-score-test-bootstrap-chunk-size", type=int, default=100
    )
    parser.add_argument(
        "--tfidf-topic-score-test-fdr-level", type=float, default=0.20
    )
    parser.add_argument(
        "--tfidf-topic-score-test-p-threshold", type=float, default=0.10
    )
    parser.add_argument(
        "--tfidf-topic-score-test-min-topics-per-bank", type=int, default=5
    )
    parser.add_argument(
        "--tfidf-topic-score-test-max-topics-per-bank", type=int, default=20
    )
    parser.add_argument(
        "--tfidf-topic-score-test-full-topic-min-inner-folds", type=int, default=1
    )
    parser.add_argument(
        "--tfidf-orphan-ngram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the fit-defined, held-out-tested raw effect n-gram cluster "
            "branch that complements NMF topic summaries."
        ),
    )
    parser.add_argument(
        "--tfidf-orphan-min-abs-fit-score", type=float, default=2.0
    )
    parser.add_argument(
        "--tfidf-orphan-cluster-similarity-threshold", type=float, default=0.25
    )
    parser.add_argument(
        "--tfidf-orphan-cluster-max-terms", type=int, default=15
    )
    parser.add_argument(
        "--tfidf-orphan-cluster-neighbors", type=int, default=20
    )
    parser.add_argument("--tfidf-orphan-fdr-level", type=float, default=0.20)
    parser.add_argument("--tfidf-orphan-p-threshold", type=float, default=0.10)
    parser.add_argument(
        "--tfidf-orphan-min-selected-clusters", type=int, default=5
    )
    parser.add_argument(
        "--tfidf-orphan-max-selected-clusters", type=int, default=20
    )
    parser.add_argument(
        "--tfidf-orphan-full-min-inner-folds", type=int, default=1
    )
    parser.add_argument("--topic-label-parallelism", type=int, default=8)
    parser.add_argument("--prespecified-features-json", default=None)
    parser.add_argument("--allow-full-data-refit", action="store_true")
    parser.add_argument("--allow-extraction-truncation", action="store_true")

    parser.add_argument("--candidate-proposals-per-fold", type=int, default=80)
    parser.add_argument(
        "--no-concept-inventory",
        action="store_true",
        help="Disable the first-pass shared text-concept inventory before proposal generation.",
    )
    parser.add_argument(
        "--concept-inventory-max-concepts",
        type=int,
        default=60,
        help="Maximum concepts requested from the pre-proposal concept inventory agent.",
    )
    parser.add_argument("--no-candidate-consistency", action="store_true")
    parser.add_argument("--candidate-consistency-inner-folds", type=int, default=3)
    parser.add_argument("--candidate-consistency-min-folds", type=int, default=2)
    parser.add_argument("--candidate-consistency-min-fold-fraction", type=float, default=0.5)
    parser.add_argument("--candidate-consistency-recovery-max-candidates", type=int, default=30)
    parser.add_argument("--no-extracted-feature-review", action="store_true")
    parser.add_argument("--extracted-feature-review-max-rounds", type=int, default=5)
    parser.add_argument("--extracted-feature-review-auc-margin", type=float, default=0.02)
    parser.add_argument(
        "--extracted-feature-review-loss-relative-margin",
        type=float,
        default=0.05,
    )
    parser.add_argument("--extracted-feature-review-min-benchmark-auc", type=float, default=0.55)
    parser.add_argument(
        "--enable-parsimony-review",
        action="store_true",
        help=(
            "Run the final parsimony pruning pass. Disabled by default for the "
            "current oracle discovery stress tests."
        ),
    )
    parser.add_argument("--min-feature-coverage", type=float, default=0.50)

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
        help=(
            "One OpenAI-compatible endpoint or a comma-separated endpoint pool; "
            "with model=auto each endpoint model id is discovered independently."
        ),
    )
    parser.add_argument("--agent-model-name", default="auto")
    parser.add_argument("--agent-api-key", default="EMPTY")
    parser.add_argument(
        "--agent-provider",
        default="openai",
        choices=_LLM_PROVIDER_CLI_CHOICES,
        help="Use local/OpenAI-compatible endpoints or Google Agent Platform.",
    )
    parser.add_argument("--agent-platform-project", default=None)
    parser.add_argument("--agent-platform-location", default="global")
    parser.add_argument("--agent-max-tokens", type=int, default=25000)
    parser.add_argument(
        "--agent-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable chat-template reasoning for proposal/schema requests.",
    )
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
        help=(
            "One extraction endpoint or a comma-separated endpoint pool; "
            "heterogeneous model ids are routed and cached per endpoint."
        ),
    )
    parser.add_argument("--extraction-model-name", default="auto")
    parser.add_argument("--extraction-api-key", default="EMPTY")
    parser.add_argument(
        "--extraction-provider",
        default="openai",
        choices=_LLM_PROVIDER_CLI_CHOICES,
    )
    parser.add_argument("--extraction-agent-platform-project", default=None)
    parser.add_argument("--extraction-agent-platform-location", default="global")
    parser.add_argument(
        "--extraction-mode",
        default="server",
        choices=["server", "start_server", "python_api"],
    )
    parser.add_argument("--extraction-reasoning-parser", default="auto")
    parser.add_argument(
        "--extraction-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable chat-template reasoning for structured extraction requests.",
    )
    parser.add_argument("--extraction-batch-size", type=int, default=100)
    parser.add_argument("--max-variables-per-extraction-request", type=int, default=10)
    parser.add_argument("--extraction-max-retries", type=int, default=3)
    parser.add_argument("--extraction-retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--extraction-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--extraction-retry-backoff-factor", type=float, default=2.0)
    parser.add_argument("--extraction-request-timeout", type=float, default=900.0)
    parser.add_argument("--extraction-max-tokens", type=int, default=25000)
    parser.add_argument("--extraction-max-text-length", type=int, default=400000)
    parser.add_argument("--extraction-cache-dir", default=None)
    parser.add_argument("--no-extraction-cache", action="store_true")
    parser.add_argument(
        "--codex-executable",
        default="codex",
        help="Codex CLI executable used when agent/extraction provider is codex_cli.",
    )
    parser.add_argument(
        "--codex-model-name",
        default="gpt-5.4-mini",
        help=(
            "Model passed to codex exec with -m for codex_cli providers. "
            "Pass an empty string or 'profile' to omit -m and let --profile choose."
        ),
    )
    parser.add_argument(
        "--codex-reasoning-effort",
        default="medium",
        help=(
            "model_reasoning_effort override passed to codex exec. "
            "Pass an empty string or 'default' to omit the override."
        ),
    )
    parser.add_argument(
        "--codex-extra-args",
        default="",
        help=(
            "Additional arguments appended to each codex exec call, e.g. "
            "'--profile local-model'. Parsed with shell-like quoting."
        ),
    )
    parser.add_argument(
        "--codex-parallelism",
        type=int,
        default=4,
        help="Maximum concurrent codex exec calls for extraction. Defaults to 4.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_methods = normalize_tfidf_topic_feature_discovery_methods(
        args.feature_discovery_methods,
        source="feature_discovery_methods",
    )
    config = MultiModelForestOracleConfig(
        dataset_path=args.dataset,
        dataset_name=Path(args.dataset).name,
        stage=args.stage,
        primary_run_id=args.primary_run_id,
        agentic_run_id=args.agentic_run_id,
        n_folds=args.n_folds,
        seed=args.seed,
        sample_size=args.sample_size,
        text_max_chars=args.text_max_chars,
        num_workers=args.cpus_total or 1,
        cpus_total=args.cpus_total,
        force_stage1=args.force_stage1,
        force_stage2=args.force_stage2,
        stage2_preflight_only=args.stage2_preflight_only,
        nuisance_folds=args.nuisance_folds,
        feature_discovery_methods=selected_methods,
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
        tfidf_topic_top_fraction=args.tfidf_topic_top_fraction,
        tfidf_topic_count=args.tfidf_topic_count,
        tfidf_topic_seeds=args.tfidf_topic_seeds,
        tfidf_topic_terms_per_topic=args.tfidf_topic_terms_per_topic,
        tfidf_topic_max_iter=args.tfidf_topic_max_iter,
        tfidf_topic_stability_repeats=args.tfidf_topic_stability_repeats,
        tfidf_topic_stability_fraction=args.tfidf_topic_stability_fraction,
        tfidf_topic_score_test_enabled=args.tfidf_topic_score_test,
        tfidf_topic_score_test_bootstrap_repeats=(
            args.tfidf_topic_score_test_bootstrap_repeats
        ),
        tfidf_topic_score_test_bootstrap_top_topics=(
            args.tfidf_topic_score_test_bootstrap_top_topics
        ),
        tfidf_topic_score_test_bootstrap_chunk_size=(
            args.tfidf_topic_score_test_bootstrap_chunk_size
        ),
        tfidf_topic_score_test_fdr_level=args.tfidf_topic_score_test_fdr_level,
        tfidf_topic_score_test_p_threshold=(
            args.tfidf_topic_score_test_p_threshold
        ),
        tfidf_topic_score_test_min_topics_per_bank=(
            args.tfidf_topic_score_test_min_topics_per_bank
        ),
        tfidf_topic_score_test_max_topics_per_bank=(
            args.tfidf_topic_score_test_max_topics_per_bank
        ),
        tfidf_topic_score_test_full_topic_min_inner_folds=(
            args.tfidf_topic_score_test_full_topic_min_inner_folds
        ),
        tfidf_orphan_ngram_enabled=args.tfidf_orphan_ngram,
        tfidf_orphan_min_abs_fit_score=args.tfidf_orphan_min_abs_fit_score,
        tfidf_orphan_cluster_similarity_threshold=(
            args.tfidf_orphan_cluster_similarity_threshold
        ),
        tfidf_orphan_cluster_max_terms=args.tfidf_orphan_cluster_max_terms,
        tfidf_orphan_cluster_neighbors=args.tfidf_orphan_cluster_neighbors,
        tfidf_orphan_fdr_level=args.tfidf_orphan_fdr_level,
        tfidf_orphan_p_threshold=args.tfidf_orphan_p_threshold,
        tfidf_orphan_min_selected_clusters=(
            args.tfidf_orphan_min_selected_clusters
        ),
        tfidf_orphan_max_selected_clusters=(
            args.tfidf_orphan_max_selected_clusters
        ),
        tfidf_orphan_full_min_inner_folds=(
            args.tfidf_orphan_full_min_inner_folds
        ),
        topic_label_parallelism=args.topic_label_parallelism,
        prespecified_features_json=args.prespecified_features_json,
        candidate_proposals_per_fold=args.candidate_proposals_per_fold,
        concept_inventory_enabled=not args.no_concept_inventory,
        concept_inventory_max_concepts=args.concept_inventory_max_concepts,
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
        parsimony_review_enabled=args.enable_parsimony_review,
        require_honest_outer_split=not args.allow_full_data_refit,
        fail_on_extraction_truncation=not args.allow_extraction_truncation,
        cf_n_estimators=args.cf_n_estimators,
        cf_min_samples_leaf=args.cf_min_samples_leaf,
        cf_max_depth=args.cf_max_depth,
        cf_max_features=args.cf_max_features,
        cf_inference=not args.cf_no_inference,
        min_feature_coverage=args.min_feature_coverage,
        agent_provider=args.agent_provider,
        agent_platform_project=args.agent_platform_project,
        agent_platform_location=args.agent_platform_location,
        agent_server_url=args.agent_server_url,
        agent_model_name=args.agent_model_name,
        agent_api_key=args.agent_api_key,
        agent_max_tokens=args.agent_max_tokens,
        agent_enable_thinking=args.agent_enable_thinking,
        agent_request_max_retries=args.agent_request_max_retries,
        agent_retry_initial_delay=args.agent_retry_initial_delay,
        agent_retry_max_delay=args.agent_retry_max_delay,
        agent_retry_backoff_factor=args.agent_retry_backoff_factor,
        agent_request_timeout=args.agent_request_timeout,
        agent_save_context=args.agent_save_context,
        agent_save_raw_output=args.agent_save_raw_output,
        extraction_provider=args.extraction_provider,
        extraction_agent_platform_project=args.extraction_agent_platform_project,
        extraction_agent_platform_location=args.extraction_agent_platform_location,
        extraction_server_url=args.extraction_server_url,
        extraction_model_name=args.extraction_model_name,
        extraction_api_key=args.extraction_api_key,
        extraction_mode=args.extraction_mode,
        extraction_reasoning_parser=args.extraction_reasoning_parser,
        extraction_enable_thinking=args.extraction_enable_thinking,
        extraction_batch_size=args.extraction_batch_size,
        max_variables_per_extraction_request=(
            args.max_variables_per_extraction_request
        ),
        extraction_max_retries=args.extraction_max_retries,
        extraction_retry_initial_delay=args.extraction_retry_initial_delay,
        extraction_retry_max_delay=args.extraction_retry_max_delay,
        extraction_retry_backoff_factor=args.extraction_retry_backoff_factor,
        extraction_request_timeout=args.extraction_request_timeout,
        extraction_max_tokens=args.extraction_max_tokens,
        extraction_max_text_length=args.extraction_max_text_length,
        extraction_cache_enabled=not args.no_extraction_cache,
        extraction_cache_dir=args.extraction_cache_dir,
        codex_executable=args.codex_executable,
        codex_model_name=args.codex_model_name,
        codex_reasoning_effort=args.codex_reasoning_effort,
        codex_extra_args=_parse_codex_extra_args(args.codex_extra_args),
        codex_parallelism=args.codex_parallelism,
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
