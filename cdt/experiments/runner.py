# cdt/experiments/runner.py

"""Main experiment runner that orchestrates CDT workflows."""

import logging
from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd

from ..config import ExperimentConfig
from ..utils import set_seed, ensure_dir, get_device
from ..data import load_dataset, validate_dataset, create_cache


logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates CDT experiments including pretraining, applied inference, and plasmode."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        ensure_dir(self.output_dir)
        
        set_seed(config.seed)
        
        # Determine device
        if config.device:
            self.device = get_device(config.device)
        elif config.gpu_ids:
            # Default to first GPU if not specified but gpu_ids are present
            import torch
            self.device = torch.device(f"cuda:{config.gpu_ids[0]}")
        else:
            # Default fallback
            self.device = get_device("cuda:0")
        
        logger.info(f"Experiment initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Seed: {config.seed}")
        logger.info(f"Workers: {config.num_workers}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete experiment workflow.
        
        Returns:
            Dictionary with results paths and summaries
        """
        results = {}
        
        self._save_config()
        
        if self.config.pretraining.enabled:
            logger.info("\n" + "="*80)
            logger.info("PHASE 1: PRETRAINING")
            logger.info("="*80)
            pretrained_weights_path = self._run_pretraining()
            results['pretrained_weights'] = str(pretrained_weights_path)
        else:
            pretrained_weights_path = None
            logger.info("Pretraining disabled, skipping")

        if self.config.applied_inference.skip:
            logger.info("\n" + "="*80)
            logger.info("PHASE 2: APPLIED INFERENCE (SKIPPED)")
            logger.info("="*80)
            logger.info("Applied inference skipped via config.applied_inference.skip=True")
        else:
            logger.info("\n" + "="*80)
            logger.info("PHASE 2: APPLIED INFERENCE")
            logger.info("="*80)
            applied_results = self._run_applied_inference(pretrained_weights_path)
            results['applied_inference'] = applied_results

        if self.config.plasmode_experiments.enabled:
            logger.info("\n" + "="*80)
            logger.info("PHASE 3: PLASMODE EXPERIMENTS")
            logger.info("="*80)
            plasmode_results = self._run_plasmode_experiments(pretrained_weights_path)
            results['plasmode_experiments'] = plasmode_results
        else:
            logger.info("Plasmode experiments disabled, skipping")
        
        self._save_results_summary(results)
        
        return results
    
    def _save_config(self) -> None:
        """Save configuration to output directory."""
        config_path = self.output_dir / "config.json"
        self.config.to_json(str(config_path))
        logger.info(f"Configuration saved to: {config_path}")
    
    def _run_pretraining(self) -> Path:
        """
        Run multi-treatment pretraining.
        
        Returns:
            Path to pretrained weights
        """
        from ..training.pretraining import run_pretraining
        
        pretrain_config = self.config.pretraining
        
        logger.info(f"Loading pretraining dataset: {pretrain_config.dataset_path}")
        pretrain_df = load_dataset(pretrain_config.dataset_path)
        
        validate_dataset(
            pretrain_df,
            text_column="clinical_text",
            outcome_column="outcome_indicator",
            treatment_column=pretrain_config.treatment_column
        )
        
        weights_dir = ensure_dir(self.output_dir / "pretrained_weights")
        weights_path = weights_dir / f"pretrained_{self.config.get_hash()}.pt"
        
        if weights_path.exists():
            logger.info(f"Using existing pretrained weights: {weights_path}")
            return weights_path
        
        cache = create_cache(
            self.config.cache_dir,
            pretrain_config.architecture.embedding_model_name,
            pretrain_config.architecture.chunk_size,
            pretrain_config.architecture.chunk_overlap
        )
        
        run_pretraining(
            dataset=pretrain_df,
            config=pretrain_config,
            output_path=weights_path,
            device=self.device,
            cache=cache
        )
        
        logger.info(f"Pretraining complete: {weights_path}")
        return weights_path
    
    def _run_applied_inference(
        self,
        pretrained_weights_path: Path = None
    ) -> str:
        """
        Run applied inference on real data.
        
        Args:
            pretrained_weights_path: Optional path to pretrained weights
        
        Returns:
            Path to predictions file
        """
        from ..inference.applied import run_applied_inference
        
        applied_config = self.config.applied_inference
        
        logger.info(f"Loading dataset: {applied_config.dataset_path}")
        df = load_dataset(applied_config.dataset_path)
        
        validate_dataset(
            df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            split_column=applied_config.split_column
        )
        
        output_dir = ensure_dir(self.output_dir / "applied_inference")
        predictions_path = output_dir / "predictions.parquet"
        
        cache = create_cache(
            self.config.cache_dir,
            applied_config.architecture.embedding_model_name,
            applied_config.architecture.chunk_size,
            applied_config.architecture.chunk_overlap
        )
        
        run_applied_inference(
            dataset=df,
            config=applied_config,
            output_path=predictions_path,
            device=self.device,
            cache=cache,
            pretrained_weights_path=pretrained_weights_path,
            gpu_ids=self.config.gpu_ids,
            num_workers=self.config.num_workers
        )
        
        logger.info(f"Applied inference complete: {predictions_path}")
        return str(predictions_path)
    
    def _run_plasmode_experiments(
        self,
        pretrained_weights_path: Path = None
    ) -> str:
        """
        Run plasmode sensitivity experiments.
        
        Args:
            pretrained_weights_path: Optional path to pretrained weights
        
        Returns:
            Path to plasmode results CSV
        """
        from ..training.plasmode import run_plasmode_experiments
        
        plasmode_config = self.config.plasmode_experiments
        applied_config = self.config.applied_inference
        
        logger.info(f"Loading dataset: {applied_config.dataset_path}")
        df = load_dataset(applied_config.dataset_path)
        
        validate_dataset(
            df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            split_column=applied_config.split_column
        )
        
        output_dir = ensure_dir(self.output_dir / "plasmode_experiments")
        results_path = output_dir / "results.csv"
        
        cache = create_cache(
            self.config.cache_dir,
            applied_config.architecture.embedding_model_name,
            applied_config.architecture.chunk_size,
            applied_config.architecture.chunk_overlap
        )
        
        run_plasmode_experiments(
            dataset=df,
            applied_config=applied_config,
            plasmode_config=plasmode_config,
            output_path=results_path,
            device=self.device,
            cache=cache,
            pretrained_weights_path=pretrained_weights_path,
            num_repeats=plasmode_config.num_repeats,
            num_workers=self.config.num_workers,
            gpu_ids=self.config.gpu_ids
        )
        
        logger.info(f"Plasmode experiments complete: {results_path}")
        return str(results_path)
    
    def _save_results_summary(self, results: Dict[str, Any]) -> None:
        """Save summary of all results."""
        summary_path = self.output_dir / "summary.json"
        
        summary = {
            'config_hash': self.config.get_hash(),
            'seed': self.config.seed,
            'device': str(self.device),
            'results': results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_path}")