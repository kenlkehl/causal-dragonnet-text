#!/usr/bin/env python
"""Oracle runner for the non-neural BoW-guided agentic forest path."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from oci.config import (  # noqa: E402
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ModelArchitectureConfig,
    NonNeuralAgenticForestConfig,
)
from oci.inference.non_neural_agentic_forest import (  # noqa: E402
    run_non_neural_agentic_forest,
)
from run_oracle_experiments import _resolve_parquet_file  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class NonNeuralAgenticOracleConfig:
    dataset_path: str
    dataset_name: str
    n_folds: int = 5
    seed: int = 42
    sample_size: Optional[int] = None
    text_max_chars: Optional[int] = None

    nuisance_folds: int = 5
    effect_folds: int = 5
    max_features: int = 30000
    min_df: int = 5
    max_df: float = 0.95
    ngram_range_min: int = 1
    ngram_range_max: int = 2
    bow_model: str = "linear"
    logistic_c: float = 1.0
    ridge_alpha: float = 10.0
    e_clip: float = 0.01
    top_n_features: int = 100
    candidate_proposals_per_fold: int = 30
    fold_parallelism: str = "auto"

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
    agent_max_tokens: int = 8192
    agent_save_context: bool = False
    agent_save_raw_output: bool = False

    extraction_server_url: str = "http://localhost:8000/v1"
    extraction_model_name: str = "Qwen/Qwen3.6-27B"
    extraction_mode: str = "server"
    extraction_reasoning_parser: Optional[str] = "auto"
    extraction_batch_size: int = 100
    extraction_max_retries: int = 3
    extraction_temperature: float = 0.0
    extraction_max_tokens: int = 4096
    extraction_max_text_length: int = 8000
    extraction_cache_enabled: bool = True
    extraction_cache_dir: Optional[str] = None

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()[:12]


def _make_applied_config(
    config: NonNeuralAgenticOracleConfig,
    parquet_file: Path,
) -> AppliedInferenceConfig:
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
            model_type="non_neural_agentic_forest",
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
                save_agent_context=config.agent_save_context,
                save_agent_raw_output=config.agent_save_raw_output,
                random_state=config.seed,
            ),
            non_neural_agentic_forest=NonNeuralAgenticForestConfig(
                nuisance_folds=config.nuisance_folds,
                effect_folds=config.effect_folds,
                max_features=config.max_features,
                min_df=config.min_df,
                max_df=config.max_df,
                ngram_range_min=config.ngram_range_min,
                ngram_range_max=config.ngram_range_max,
                bow_model=config.bow_model,
                logistic_c=config.logistic_c,
                ridge_alpha=config.ridge_alpha,
                e_clip=config.e_clip,
                top_n_features=config.top_n_features,
                candidate_proposals_per_fold=config.candidate_proposals_per_fold,
                fold_parallelism=config.fold_parallelism,
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
            extraction_temperature=config.extraction_temperature,
            extraction_max_tokens=config.extraction_max_tokens,
            extraction_max_text_length=config.extraction_max_text_length,
            cache_enabled=config.extraction_cache_enabled,
            cache_dir=config.extraction_cache_dir,
        ),
    )


def _load_dataset(config: NonNeuralAgenticOracleConfig, parquet_file: Path) -> pd.DataFrame:
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
    return metrics


def _run_one(config: NonNeuralAgenticOracleConfig, output_dir: Path) -> Dict[str, Any]:
    parquet_file = _resolve_parquet_file(config.dataset_path)
    df = _load_dataset(config, parquet_file)
    applied = _make_applied_config(config, parquet_file)
    run_hash = config.config_hash()
    prediction_dir = output_dir / "non_neural_agentic_predictions" / run_hash
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / "predictions.parquet"

    logger.info(
        "Running non-neural agentic forest dataset=%s rows=%s hash=%s",
        config.dataset_name,
        len(df),
        run_hash,
    )
    run_non_neural_agentic_forest(df, applied, prediction_path)
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
        description="Run non-neural BoW-guided agentic explicit-feature causal forest"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--text-max-chars", type=int, default=None)

    parser.add_argument("--nuisance-folds", type=int, default=5)
    parser.add_argument("--effect-folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument(
        "--bow-model",
        default="linear",
        choices=["linear", "extratrees", "random_forest", "xgboost"],
        help=(
            "BoW learner family for nuisance and pseudo-target models. "
            "linear is sparse logistic/ridge; tree options allow feature interactions."
        ),
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--top-n-features", type=int, default=100)
    parser.add_argument("--candidate-proposals-per-fold", type=int, default=30)
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

    parser.add_argument("--agent-server-url", default="http://localhost:8000/v1")
    parser.add_argument("--agent-model-name", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--agent-api-key", default="EMPTY")
    parser.add_argument("--agent-max-tokens", type=int, default=8192)
    parser.add_argument("--agent-save-context", action="store_true")
    parser.add_argument("--agent-save-raw-output", action="store_true")

    parser.add_argument("--extraction-server-url", default="http://localhost:8000/v1")
    parser.add_argument("--extraction-model-name", default="Qwen/Qwen3.6-27B")
    parser.add_argument(
        "--extraction-mode",
        default="server",
        choices=["server", "start_server", "python_api"],
    )
    parser.add_argument("--extraction-reasoning-parser", default="auto")
    parser.add_argument("--extraction-batch-size", type=int, default=100)
    parser.add_argument("--extraction-max-tokens", type=int, default=4096)
    parser.add_argument("--extraction-max-text-length", type=int, default=8000)
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
    config = NonNeuralAgenticOracleConfig(
        dataset_path=args.dataset,
        dataset_name=Path(args.dataset).name,
        n_folds=args.n_folds,
        seed=args.seed,
        sample_size=args.sample_size,
        text_max_chars=args.text_max_chars,
        nuisance_folds=args.nuisance_folds,
        effect_folds=args.effect_folds,
        max_features=args.max_features,
        min_df=args.min_df,
        bow_model=args.bow_model,
        ridge_alpha=args.ridge_alpha,
        top_n_features=args.top_n_features,
        candidate_proposals_per_fold=args.candidate_proposals_per_fold,
        fold_parallelism=args.fold_parallelism,
        cf_n_estimators=args.cf_n_estimators,
        cf_min_samples_leaf=args.cf_min_samples_leaf,
        cf_max_depth=args.cf_max_depth,
        cf_inference=not args.cf_no_inference,
        agent_server_url=args.agent_server_url,
        agent_model_name=args.agent_model_name,
        agent_api_key=args.agent_api_key,
        agent_max_tokens=args.agent_max_tokens,
        agent_save_context=args.agent_save_context,
        agent_save_raw_output=args.agent_save_raw_output,
        extraction_server_url=args.extraction_server_url,
        extraction_model_name=args.extraction_model_name,
        extraction_mode=args.extraction_mode,
        extraction_reasoning_parser=args.extraction_reasoning_parser,
        extraction_batch_size=args.extraction_batch_size,
        extraction_max_tokens=args.extraction_max_tokens,
        extraction_max_text_length=args.extraction_max_text_length,
        extraction_cache_enabled=not args.no_extraction_cache,
        extraction_cache_dir=args.extraction_cache_dir,
    )

    try:
        result = _run_one(config, output_dir)
        _append_results(output_dir, [result])
        logger.info("Completed non-neural agentic forest: %s", result["metrics"])
    except Exception:
        logger.error("Non-neural agentic forest run failed")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
