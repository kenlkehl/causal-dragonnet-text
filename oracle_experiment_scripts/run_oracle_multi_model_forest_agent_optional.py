#!/usr/bin/env python
"""Oracle runner for the multi-model W/X forest with optional final agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import (  # noqa: E402
    AgenticAttentionVariableForestConfig,
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    EmbeddingContrastDiscoveryConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ModelArchitectureConfig,
    MultiModelForestAgentOptionalConfig,
    normalize_multi_model_feature_discovery_methods,
)
from oci.inference.multi_model_forest_agent_optional import (  # noqa: E402
    run_multi_model_forest_agent_optional,
)
from run_oracle_multi_model_agentic_forest import (  # noqa: E402
    MultiModelAgenticOracleConfig,
    _append_results,
    _bow_views_for_config,
    _load_dataset,
    _metrics,
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
class MultiModelForestAgentOptionalOracleConfig(MultiModelAgenticOracleConfig):
    """CLI config for the optional-agent multi-model forest oracle run."""

    agentic_explicit_branch_enabled: bool = False


def _make_applied_config(
    config: MultiModelForestAgentOptionalOracleConfig,
    parquet_file: Path,
) -> AppliedInferenceConfig:
    bow_views = _bow_views_for_config(config)
    selected_methods = normalize_multi_model_feature_discovery_methods(
        config.feature_discovery_methods,
        source="feature_discovery_methods",
    )
    embedding_enabled = (
        "embedding_contrast" in selected_methods
        if selected_methods is not None
        else config.embedding_contrast_enabled
    )
    return AppliedInferenceConfig(
        clinical_question=(
            "Estimate heterogeneous treatment effects from clinical text using "
            "non-agentic text-model W/X features; optionally report an explicit "
            "agent-derived branch."
        ),
        outcome_type="binary",
        dataset_path=str(parquet_file),
        text_column="clinical_text",
        outcome_column="outcome_indicator",
        treatment_column="treatment_indicator",
        cv_folds=config.n_folds,
        architecture=ModelArchitectureConfig(
            model_type="multi_model_forest_agent_optional",
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
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=config.nuisance_folds,
                effect_folds=config.effect_folds,
                fold_parallelism=config.fold_parallelism,
            ),
            multi_model_forest_agent_optional=MultiModelForestAgentOptionalConfig(
                nuisance_folds=config.nuisance_folds,
                effect_folds=config.effect_folds,
                feature_discovery_methods=selected_methods,
                bow_views=bow_views,
                prespecified_features_json=config.prespecified_features_json,
                e_clip=config.e_clip,
                top_n_features=config.top_n_features,
                require_honest_outer_split=config.require_honest_outer_split,
                outer_parallelism=config.outer_parallelism,
                bow_parallel_backend=config.bow_parallel_backend,
                fold_parallelism=config.fold_parallelism,
                htr_evidence_enabled=(
                    selected_methods is None or "htr" in selected_methods
                ),
                htr_evidence_disable_reason=(
                    None
                    if selected_methods is None or "htr" in selected_methods
                    else "disabled by --feature-discovery-methods"
                ),
                agentic_explicit_branch_enabled=config.agentic_explicit_branch_enabled,
                embedding_contrast=EmbeddingContrastDiscoveryConfig(
                    enabled=embedding_enabled,
                    disable_reason=(
                        None
                        if embedding_enabled
                        else (
                            "disabled by --feature-discovery-methods"
                            if selected_methods is not None
                            else "disabled by oracle optional-agent script CLI"
                        )
                    ),
                    model_name=config.embedding_model_name,
                    cache_dir=config.embedding_cache_dir,
                    device=config.embedding_device,
                    batch_size=config.embedding_batch_size,
                    max_seq_length=config.embedding_max_seq_length,
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
                    include_confounder_vector_contrast=(
                        config.embedding_include_confounder_vector_contrast
                    ),
                    include_residualized_interaction_contrast=(
                        config.embedding_include_residualized_interaction_contrast
                    ),
                    include_orthogonal_r_score_contrasts=(
                        config.embedding_include_orthogonal_r_score_contrasts
                    ),
                    external_corpus_cache_dirs=config.embedding_external_cache_dirs,
                    external_top_k_chunks_per_tail=(
                        config.embedding_external_top_k_chunks_per_tail
                    ),
                    concept_phrases=config.embedding_concept_phrases,
                    residualize_columns=config.embedding_residualize_columns,
                ),
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=bool(config.agentic_explicit_branch_enabled),
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


def _run_one(
    config: MultiModelForestAgentOptionalOracleConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    parquet_file = _resolve_oracle_parquet_file_for_optional_cache(config)
    if config.dataset_path != str(parquet_file):
        logger.info(
            "Resolved --dataset %s to %s",
            config.dataset_path,
            parquet_file,
        )
        config.dataset_path = str(parquet_file)
    normalized_cache_dir = _normalize_embedding_cache_dir_arg(config.embedding_cache_dir)
    if normalized_cache_dir != config.embedding_cache_dir:
        logger.info(
            "Using embedding cache root %s from --embedding-cache-dir %s",
            normalized_cache_dir,
            config.embedding_cache_dir,
        )
        config.embedding_cache_dir = normalized_cache_dir
    df = _load_dataset(config, parquet_file)
    applied = _make_applied_config(config, parquet_file)
    run_hash = config.config_hash()
    prediction_dir = output_dir / "multi_model_forest_agent_optional_predictions" / run_hash
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / "predictions.parquet"

    logger.info(
        "Running multi-model optional-agent forest dataset=%s rows=%s hash=%s",
        config.dataset_name,
        len(df),
        run_hash,
    )
    if applied.architecture.multi_model_forest_agent_optional.embedding_contrast.enabled:
        cache_dir = (
            applied.architecture.multi_model_forest_agent_optional
            .embedding_contrast
            .cache_dir
        )
        logger.info(
            "Embedding contrast cache root: %s",
            cache_dir
            or _default_embedding_cache_root_for_parquet(parquet_file),
        )
    run_multi_model_forest_agent_optional(
        df,
        applied,
        prediction_path,
        device=config.htr_device,
        gpu_ids=config.htr_gpu_ids,
        num_workers=config.num_workers,
    )
    results_df = pd.read_parquet(prediction_path)
    result = {
        **asdict(config),
        "config_hash": run_hash,
        "prediction_path": str(prediction_path),
        "artifact_dir": str(prediction_path.parent / "multi_model_forest_agent_optional"),
        "metrics": _metrics(results_df),
    }
    branch_summary = (
        prediction_path.parent
        / "multi_model_forest_agent_optional"
        / "agentic_explicit_branch_summary.json"
    )
    if branch_summary.exists():
        with open(branch_summary) as f:
            result["agentic_explicit_branch"] = json.load(f)
    return result


def _append_list_arg(parser: argparse.ArgumentParser, name: str, **kwargs: Any) -> None:
    parser.add_argument(name, action="append", default=[], **kwargs)


def _resolve_oracle_parquet_file_for_optional_cache(
    config: MultiModelForestAgentOptionalOracleConfig,
) -> Path:
    """Resolve a dataset directory, preferring a parquet with a complete chunk cache.

    The shared oracle resolver prefers dataset_with_extraction.parquet. That is a
    good default for agent-extraction paths, but embedding chunk caches are keyed
    to the exact parquet path. For the optional-agent path, prefer whichever
    oracle parquet has a complete matching embedding cache, then fall back to the
    shared resolver.
    """
    path = Path(config.dataset_path).expanduser()
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"--dataset must be a parquet file or dataset directory: {path}")
        return path

    if (
        _embedding_contrast_requested(config)
        and config.sample_size is None
        and config.text_max_chars is None
    ):
        for candidate in _candidate_oracle_parquet_files(path):
            cache_path = _matching_complete_embedding_cache_path(config, candidate)
            if cache_path is not None:
                logger.info(
                    "Resolved dataset directory to %s because matching embedding "
                    "chunk cache exists at %s",
                    candidate,
                    cache_path,
                )
                return candidate
    elif _embedding_contrast_requested(config):
        logger.info(
            "Skipping embedding-cache-aware dataset resolution because sample_size "
            "or text_max_chars changes the in-memory text rows"
        )

    return _resolve_oracle_parquet_file(config.dataset_path)


def _embedding_contrast_requested(
    config: MultiModelForestAgentOptionalOracleConfig,
) -> bool:
    selected_methods = normalize_multi_model_feature_discovery_methods(
        config.feature_discovery_methods,
        source="feature_discovery_methods",
    )
    if selected_methods is not None:
        return "embedding_contrast" in selected_methods
    return bool(config.embedding_contrast_enabled)


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
    config: MultiModelForestAgentOptionalOracleConfig,
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
    config: MultiModelForestAgentOptionalOracleConfig,
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
    config: MultiModelForestAgentOptionalOracleConfig,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-model W/X causal forest with optional final agent branch"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--text-max-chars", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)

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
    parser.add_argument("--outer-parallelism", default="1")
    parser.add_argument(
        "--bow-parallel-backend",
        default="processes",
        choices=["processes", "threads"],
    )
    parser.add_argument("--fold-parallelism", default="auto")
    parser.add_argument("--htr-device", default="cuda:0")
    parser.add_argument("--htr-gpu-ids", type=int, nargs="+", default=None)

    parser.add_argument("--enable-embedding-contrast", dest="embedding_contrast_enabled", action="store_true")
    parser.add_argument("--disable-embedding-contrast", dest="embedding_contrast_enabled", action="store_false")
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
    parser.add_argument("--embedding-disable-residualized-interaction-contrast", action="store_true")
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

    parser.add_argument("--enable-agentic-explicit-branch", action="store_true")
    parser.add_argument("--candidate-proposals-per-fold", type=int, default=30)
    parser.add_argument("--min-feature-coverage", type=float, default=0.70)
    parser.add_argument("--agent-server-url", "--agent-server-urls", dest="agent_server_url", default="http://localhost:8000/v1")
    parser.add_argument("--agent-model-name", default="auto")
    parser.add_argument("--agent-api-key", default="EMPTY")
    parser.add_argument("--agent-max-tokens", type=int, default=25000)
    parser.add_argument("--agent-request-max-retries", type=int, default=3)
    parser.add_argument("--agent-retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--agent-retry-max-delay", type=float, default=30.0)
    parser.add_argument("--agent-retry-backoff-factor", type=float, default=2.0)
    parser.add_argument("--agent-save-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-save-raw-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extraction-server-url", "--extraction-server-urls", dest="extraction_server_url", default="http://localhost:8000/v1")
    parser.add_argument("--extraction-model-name", default="auto")
    parser.add_argument("--extraction-mode", default="server", choices=["server", "start_server", "python_api"])
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
    config = MultiModelForestAgentOptionalOracleConfig(
        dataset_path=args.dataset,
        dataset_name=Path(args.dataset).name,
        n_folds=args.n_folds,
        seed=args.seed,
        sample_size=args.sample_size,
        text_max_chars=args.text_max_chars,
        num_workers=args.num_workers,
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
        require_honest_outer_split=not args.allow_full_data_refit,
        outer_parallelism=args.outer_parallelism,
        bow_parallel_backend=args.bow_parallel_backend,
        fold_parallelism=args.fold_parallelism,
        htr_device=args.htr_device,
        htr_gpu_ids=args.htr_gpu_ids,
        embedding_contrast_enabled=args.embedding_contrast_enabled,
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
        agentic_explicit_branch_enabled=args.enable_agentic_explicit_branch,
        candidate_proposals_per_fold=args.candidate_proposals_per_fold,
        min_feature_coverage=args.min_feature_coverage,
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
        logger.info("Completed multi-model optional-agent forest: %s", result["metrics"])
    except Exception:
        logger.error("Multi-model optional-agent forest run failed")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
