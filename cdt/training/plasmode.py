# cdt/training/plasmode.py
"""Plasmode simulation experiments for sensitivity analysis."""

import logging
import random
import json
from pathlib import Path
from dataclasses import asdict
from typing import Optional, List, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer

from ..config import AppliedInferenceConfig, PlasmodeExperimentConfig, PlasmodeConfig
from ..models.causal_dragonnet import CausalDragonnetText
from ..data import ClinicalTextDataset, collate_batch, EmbeddingCache
from ..utils import cuda_cleanup, get_memory_info, set_seed, load_pretrained_with_dimension_matching


logger = logging.getLogger(__name__)


def run_plasmode_experiments(
    dataset: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    plasmode_config: PlasmodeExperimentConfig,
    output_path: Path,
    device: torch.device,
    cache: Optional[EmbeddingCache] = None,
    pretrained_weights_path: Optional[Path] = None,
    num_repeats: int = 3,
    num_workers: int = 1,
    gpu_ids: Optional[List[int]] = None
) -> None:
    """
    Run plasmode sensitivity experiments with parallel execution.
    """
    logger.info("="*80)
    logger.info(f"PLASMODE SENSITIVITY EXPERIMENTS (Workers: {num_workers})")
    logger.info("="*80)
    
    # Split data - We use the real training data as the base for simulation
    train_df = dataset.copy() ##[dataset[applied_config.split_column] == 'train'].copy()
    
    logger.info(f"Using {len(train_df)} training samples for plasmode generation base")
    logger.info(f"Running {len(plasmode_config.plasmode_scenarios)} scenarios × {num_repeats} repeats")
    
    # Dataset saving setup
    save_datasets = getattr(plasmode_config, 'save_datasets', False)
    dataset_dir = None
    if save_datasets:
        dataset_dir = output_path.parent / "simulated_datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Simulated datasets will be saved to: {dataset_dir}")
        
    # Pre-compute Explicit Confounder Embeddings (GPU) to avoid slower CPU encoding in concurrent workers
    gen_texts = plasmode_config.generator_architecture.explicit_confounder_texts
    eval_texts = plasmode_config.evaluator_architecture.explicit_confounder_texts
    
    precomputed_embeddings = {}
    
    unique_text_sets = []
    if gen_texts: unique_text_sets.append(tuple(gen_texts))
    if eval_texts: unique_text_sets.append(tuple(eval_texts))
    unique_text_sets = set(unique_text_sets)
    
    if unique_text_sets:
        logger.info("Pre-computing explicit confounder embeddings on GPU for efficiency...")
        # Use first available GPU or CPU
        compute_device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load temp model
        model_name = plasmode_config.generator_architecture.embedding_model_name
        temp_st = SentenceTransformer(model_name)
        temp_st.to(compute_device)
        
        for text_tuple in unique_text_sets:
            logger.info(f"  Encoding set of {len(text_tuple)} texts...")
            with torch.no_grad():
                embs = temp_st.encode(list(text_tuple), convert_to_tensor=True, device=compute_device, show_progress_bar=False)
                # Move to CPU for sharing across processes
                precomputed_embeddings[text_tuple] = embs.cpu()
        
        # Cleanup
        del temp_st
        cuda_cleanup()
        logger.info("Pre-computation complete.")
    
    # Prepare all tasks
    tasks = []
    for scenario_idx, scenario in enumerate(plasmode_config.plasmode_scenarios):
        logger.info(f"\n{'='*80}")
        logger.info(f"SCENARIO {scenario_idx+1}/{len(plasmode_config.plasmode_scenarios)}")
        logger.info(f"  Mode: {scenario.generation_mode}")
        logger.info(f"  Target ATE: {scenario.target_ate_logit}")
        logger.info(f"  Outcome Scale (SD): {scenario.outcome_heterogeneity_scale}")
        logger.info(f"  ITE Scale (SD): {scenario.ite_heterogeneity_scale}")
        logger.info(f"{'='*80}")
        
        for repeat_idx in range(num_repeats):
            # Determine device for this task
            if gpu_ids:
                # Round-robin assignment based on total task index
                task_global_idx = len(tasks)
                device_id = gpu_ids[task_global_idx % len(gpu_ids)]
                task_device = torch.device(f"cuda:{device_id}")
            else:
                task_device = device

            tasks.append({
                'scenario_idx': scenario_idx,
                'scenario': scenario,
                'repeat_idx': repeat_idx,
                'train_df': train_df,
                'applied_config': applied_config,
                'plasmode_config': plasmode_config,
                'device': task_device,
                'cache': cache,
                'pretrained_weights_path': pretrained_weights_path,
                'dataset_dir': dataset_dir,
                'precomputed_embeddings': precomputed_embeddings
            })

    # Execute tasks in parallel
    logger.info(f"Starting {len(tasks)} experiments on {num_workers} workers...")
    
    results = Parallel(n_jobs=num_workers)(
        delayed(_worker_wrapper)(task) for task in tasks
    )
    
    # Aggregate results
    all_results = []
    all_training_logs = []
    
    for res in results:
        if res is not None:
            metrics, logs = res
            all_results.append(metrics)
            all_training_logs.extend(logs)
            
    # Save aggregated training logs
    if all_training_logs:
        log_path = output_path.parent / "plasmode_training_log_aggregate.csv"
        pd.DataFrame(all_training_logs).to_csv(log_path, index=False)
        logger.info(f"Aggregated plasmode training logs saved to: {log_path}")
    
    # Save results summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"\n{'='*80}")
        logger.info("PLASMODE EXPERIMENTS COMPLETE")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Total experiments: {len(results_df)}")
        logger.info(f"{'='*80}")
        
        # Summary statistics
        summary = results_df.groupby('generation_mode').agg({
            'ate_bias': ['mean', 'std'],
            'ate_rmse': ['mean', 'std'],
            'ite_correlation': ['mean', 'std'],
            'ite_regression_slope': ['mean', 'std']
        }).round(4)
        
        logger.info("\nSummary by generation mode:")
        logger.info(f"\n{summary}")
    else:
        logger.error("No successful experiments completed")


def _worker_wrapper(task: Dict[str, Any]) -> Optional[Tuple[dict, List[Dict[str, Any]]]]:
    """Helper for parallel execution to handle setup and error catching."""
    scenario_idx = task['scenario_idx']
    repeat_idx = task['repeat_idx']
    scenario = task['scenario']
    dataset_dir = task['dataset_dir']
    plasmode_config = task['plasmode_config']
    precomputed_embeddings = task.get('precomputed_embeddings', {})
    
    # Set seed unique to this repeat
    current_seed = random.randint(0, 1000)
    set_seed(current_seed)
    
    # Prepare hyperparameters metadata
    hyperparams = {
        'scenario_idx': scenario_idx,
        'repeat_idx': repeat_idx,
        'seed': current_seed,
        'generation_mode': scenario.generation_mode,
        **asdict(scenario),  # Include all scenario params
        'generator_training': asdict(plasmode_config.generator_training),
        'evaluator_training': asdict(plasmode_config.evaluator_training)
    }
    
    # Determine dataset/config/log path if saving is enabled
    save_dataset_path = None
    save_config_path = None
    save_log_path = None
    
    if dataset_dir:
        base_name = f"scenario_{scenario_idx}_repeat_{repeat_idx}_{scenario.generation_mode}"
        save_dataset_path = dataset_dir / f"{base_name}.parquet"
        save_config_path = dataset_dir / f"{base_name}_params.json"
        save_log_path = dataset_dir / f"{base_name}_log.csv"
        
    try:
        # Single split per repeat: train_fraction determines the split
        metrics, logs = _run_single_plasmode_experiment(
            train_df=task['train_df'],
            scenario=scenario,
            applied_config=task['applied_config'],
            plasmode_config=plasmode_config,
            device=task['device'],
            cache=task['cache'],
            pretrained_weights_path=task['pretrained_weights_path'],
            save_dataset_path=save_dataset_path,
            save_config_path=save_config_path,
            save_log_path=save_log_path,
            hyperparams=hyperparams,
            precomputed_embeddings=precomputed_embeddings
        )
        
        metrics['scenario_idx'] = scenario_idx
        metrics['repeat_idx'] = repeat_idx
        metrics['generation_mode'] = scenario.generation_mode
        metrics['target_ate'] = scenario.target_ate_logit
        metrics['train_fraction'] = plasmode_config.train_fraction
        
        return metrics, logs
        
    except Exception as e:
        logger.error(f"Scenario {scenario_idx} Repeat {repeat_idx} Failed: {e}", exc_info=True)
        return None
    finally:
        cuda_cleanup()


def _run_single_plasmode_experiment(
    train_df: pd.DataFrame,
    scenario: PlasmodeConfig,
    applied_config: AppliedInferenceConfig,
    plasmode_config: PlasmodeExperimentConfig,
    device: torch.device,
    cache: Optional[EmbeddingCache],
    pretrained_weights_path: Optional[Path],
    save_dataset_path: Optional[Path] = None,
    save_config_path: Optional[Path] = None,
    save_log_path: Optional[Path] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    precomputed_embeddings: Dict[Tuple[str], Any] = None
) -> Tuple[dict, List[Dict[str, Any]]]:
    """Run a single plasmode experiment with train/eval split. Returns (metrics, training_logs)
    
    Flow:
    1. Split data into train_split (train_fraction) and eval_split (1 - train_fraction)
    2. Train generator on train_split only
    3. Generate synthetic outcomes for both train_split and eval_split
    4. Train evaluator on train_split (with simulated outcomes)
    5. Predict on eval_split and calculate metrics
    
    This ensures:
    - Consistent DGP within each experiment (one generator per repeat)
    - Out-of-sample evaluation (evaluator never sees eval_split during training)
    - Statistical robustness via multiple repeats across the experiment
    """
    
    # Get train_fraction from config (default 0.8)
    train_fraction = getattr(plasmode_config, 'train_fraction', 0.8)
    seed = hyperparams.get('seed', 42) if hyperparams else 42
    
    # Step 0: Split data into train and eval sets for this experiment
    train_split_df, eval_split_df = train_test_split(
        train_df, train_size=train_fraction, random_state=seed
    )
    train_split_df = train_split_df.reset_index(drop=True)
    eval_split_df = eval_split_df.reset_index(drop=True)
    
    logger.info(f"Single-split experiment: Training on {len(train_split_df)} samples, evaluating on {len(eval_split_df)}")
    
    # Get precomputed embeddings for generator if available
    gen_texts = plasmode_config.generator_architecture.explicit_confounder_texts
    gen_embeddings = precomputed_embeddings.get(tuple(gen_texts)) if gen_texts and precomputed_embeddings else None
    
    # Step 1: Train generator on TRAIN SPLIT only
    generator, gen_history = _train_plasmode_generator(
        train_split_df,
        applied_config,
        plasmode_config,
        device,
        cache,
        pretrained_weights_path,
        explicit_confounder_embeddings=gen_embeddings
    )
    
    # Tag history
    for entry in gen_history:
        entry['model_type'] = 'generator'
        entry['generation_mode'] = scenario.generation_mode
        entry['scenario_idx'] = hyperparams.get('scenario_idx') if hyperparams else -1
        entry['repeat_idx'] = hyperparams.get('repeat_idx') if hyperparams else -1
    
    # Step 2: Generate synthetic outcomes for BOTH splits using the same generator
    oracle_mode = getattr(plasmode_config, 'oracle_mode', False)
    
    # Generate for train split (for evaluator training)
    if oracle_mode:
        train_plasmode_df, train_confounder_features = _generate_plasmode_data(
            train_split_df,
            generator,
            scenario,
            applied_config,
            device,
            cache,
            return_confounders=True
        )
        # Generate for eval split (for metrics)
        eval_plasmode_df, eval_confounder_features = _generate_plasmode_data(
            eval_split_df,
            generator,
            scenario,
            applied_config,
            device,
            cache,
            return_confounders=True
        )
        logger.info("Oracle mode: Evaluator will use generator's confounder_features directly")
    else:
        train_plasmode_df = _generate_plasmode_data(
            train_split_df,
            generator,
            scenario,
            applied_config,
            device,
            cache,
            return_confounders=False
        )
        eval_plasmode_df = _generate_plasmode_data(
            eval_split_df,
            generator,
            scenario,
            applied_config,
            device,
            cache,
            return_confounders=False
        )
        train_confounder_features = None
        eval_confounder_features = None
    
    # Mark splits in dataframes
    train_plasmode_df['sim_split'] = 'train'
    eval_plasmode_df['sim_split'] = 'eval'
    
    # Step 3: Train evaluator on TRAIN SPLIT (no validation holdout)
    if oracle_mode:
        # Oracle mode: train on pre-extracted confounder_features (no standardization)
        evaluator, eval_history = _train_confounder_evaluator(
            train_confounder_features,
            train_plasmode_df,
            None,  # No validation set
            None,
            applied_config,
            plasmode_config,
            device
        )
    else:
        # Get precomputed embeddings for evaluator if available
        eval_texts = plasmode_config.evaluator_architecture.explicit_confounder_texts
        eval_embeddings = precomputed_embeddings.get(tuple(eval_texts)) if eval_texts and precomputed_embeddings else None

        # Realistic mode: train on text, learn own confounders
        evaluator, eval_history = _train_plasmode_evaluator(
            train_plasmode_df,
            None,  # No validation set
            applied_config,
            plasmode_config,
            device,
            cache,
            pretrained_weights_path,
            explicit_confounder_embeddings=eval_embeddings
        )

    # Tag history
    for entry in eval_history:
        entry['model_type'] = 'evaluator'
        entry['generation_mode'] = scenario.generation_mode
        entry['scenario_idx'] = hyperparams.get('scenario_idx') if hyperparams else -1
        entry['repeat_idx'] = hyperparams.get('repeat_idx') if hyperparams else -1
        entry['oracle_mode'] = oracle_mode

    combined_history = gen_history + eval_history

    # Step 4: Generate predictions for EVAL SPLIT only (the held-out data)
    if oracle_mode:
        preds_dict = _predict_confounder_evaluator(
            evaluator,
            eval_confounder_features,
            device
        )
    else:
        preds_dict = _predict_plasmode_evaluator(
            evaluator,
            eval_plasmode_df,
            applied_config,
            plasmode_config,
            device,
            cache
        )
    
    # Add estimated metrics to eval dataframe
    eval_plasmode_df['estimated_ite'] = preds_dict['ite']
    eval_plasmode_df['estimated_y0_logit'] = preds_dict['y0_logit']
    eval_plasmode_df['estimated_y1_logit'] = preds_dict['y1_logit']
    eval_plasmode_df['estimated_propensity'] = preds_dict['propensity']
    
    # Save dataset (eval split only - the held-out data with out-of-sample predictions)
    if save_dataset_path is not None:
        logger.info(f"Saving simulated dataset (eval split only) to {save_dataset_path}")
        eval_plasmode_df.to_parquet(save_dataset_path, index=False)
        
        if save_config_path is not None and hyperparams is not None:
            try:
                hyperparams_copy = hyperparams.copy()
                hyperparams_copy['train_fraction'] = train_fraction
                with open(save_config_path, 'w') as f:
                    json.dump(hyperparams_copy, f, indent=2)
                logger.info(f"Saved params to {save_config_path}")
            except Exception as e:
                logger.warning(f"Failed to save simulation parameters: {e}")
                
        if save_log_path is not None:
            try:
                pd.DataFrame(combined_history).to_csv(save_log_path, index=False)
                logger.info(f"Saved training log to {save_log_path}")
            except Exception as e:
                logger.warning(f"Failed to save training log: {e}")
    
    # Step 5: Evaluate performance on EVAL SPLIT only (out-of-sample)
    metrics = _evaluate_plasmode_performance(
        eval_plasmode_df,
        preds_dict['ite'],
        scenario.target_ate_logit
    )
    
    logger.info(f"Experiment complete: ATE bias={metrics['ate_bias']:.4f}, ITE corr={metrics['ite_correlation']:.4f}")

    # Add metadata to metrics
    metrics['oracle_mode'] = oracle_mode
    metrics['train_fraction'] = train_fraction
    metrics['n_train'] = len(train_split_df)
    metrics['n_eval'] = len(eval_split_df)

    return metrics, combined_history


def _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop):
    """Helper to compute AUROCs from collected batch outputs."""
    y_true = torch.cat(all_targets).numpy()
    t_true = torch.cat(all_treatments).numpy()
    y0_scores = torch.cat(all_y0).numpy()
    y1_scores = torch.cat(all_y1).numpy()
    prop_scores = torch.sigmoid(torch.cat(all_prop)).numpy() # sigmoid for prop score
    
    # Safe AUROC calculation
    def safe_auc(y, score):
        try:
            if len(np.unique(y)) < 2: return None
            return roc_auc_score(y, score)
        except: return None

    # AUROC Y0 (on T=0 samples)
    mask0 = (t_true == 0)
    auroc_y0 = safe_auc(y_true[mask0], y0_scores[mask0]) if mask0.any() else None
    
    # AUROC Y1 (on T=1 samples)
    mask1 = (t_true == 1)
    auroc_y1 = safe_auc(y_true[mask1], y1_scores[mask1]) if mask1.any() else None
    
    # AUROC Propensity
    auroc_prop = safe_auc(t_true, prop_scores)
    
    return {
        'loss': epoch_loss / len(loader),
        'auroc_y0': auroc_y0,
        'auroc_y1': auroc_y1,
        'auroc_prop': auroc_prop
    }


def _train_plasmode_generator(
    train_df: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    plasmode_config: PlasmodeExperimentConfig,
    device: torch.device,
    cache: Optional[EmbeddingCache],
    pretrained_weights_path: Optional[Path],
    explicit_confounder_embeddings: Optional[torch.Tensor] = None
) -> Tuple[CausalDragonnetText, List[Dict[str, Any]]]:
    """Train binary treatment generator. Returns (model, history)."""
    
    generator_df = train_df.copy()
    
    # Create generator model using CausalDragonnetText (Binary)
    gen_arch = plasmode_config.generator_architecture
    generator = CausalDragonnetText(
        sentence_transformer_model_name=gen_arch.embedding_model_name,
        num_latent_confounders=gen_arch.num_latent_confounders,
        features_per_confounder=gen_arch.features_per_confounder,
        explicit_confounder_texts=gen_arch.explicit_confounder_texts,
        aggregator_mode=gen_arch.aggregator_mode,
        dragonnet_representation_dim=gen_arch.dragonnet_representation_dim,
        dragonnet_hidden_outcome_dim=gen_arch.dragonnet_hidden_outcome_dim,
        chunk_size=gen_arch.chunk_size,
        chunk_overlap=gen_arch.chunk_overlap,
        device=str(device),
        model_type=gen_arch.model_type
    )
    
    # Load pretrained weights
    if pretrained_weights_path is not None:
        logger.info("Loading pretrained weights for generator...")
        try:
            pretrained_checkpoint = torch.load(
                pretrained_weights_path, 
                map_location=device,
                weights_only=False
            )
            load_pretrained_with_dimension_matching(
                generator,
                pretrained_checkpoint,
                strict=False,
                auto_adjust=True
            )
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
    
    # Create dataset
    gen_dataset = ClinicalTextDataset(
        data=generator_df,
        text_column=applied_config.text_column,
        outcome_column=applied_config.outcome_column,
        treatment_column=applied_config.treatment_column, 
        model=generator.sentence_transformer_model,
        device=device,
        chunk_size=gen_arch.chunk_size,
        chunk_overlap=gen_arch.chunk_overlap,
        cache=cache
    )
    
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=plasmode_config.generator_training.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0
    )
    
    # Optimizer & Scheduler
    gen_train = plasmode_config.generator_training
    optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=gen_train.learning_rate,
        weight_decay=1e-4  # Match old_cdt behavior
    )

    if gen_train.lr_schedule == "linear":
        total_steps = len(gen_loader) * gen_train.epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
    else:
        scheduler = None
    
    # Training loop
    generator.train()
    history = []
    
    for epoch in range(gen_train.epochs):
        epoch_loss = 0.0
        all_targets = []
        all_treatments = []
        all_y0 = []
        all_y1 = []
        all_prop = []
        
        for batch in tqdm(gen_loader, desc=f"Generator Epoch {epoch+1}", leave=False):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Convert chunks
            batch['chunk_embeddings'] = [
                batch['chunk_embeddings'][i, :, :].contiguous()
                for i in range(batch['chunk_embeddings'].size(0))
            ]
            
            optimizer.zero_grad()
            losses = generator.train_step(
                batch,
                alpha_propensity=gen_train.alpha_propensity,
                beta_targreg=gen_train.beta_targreg
            )
            losses['loss'].backward()
            #torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
                
            epoch_loss += losses['loss'].item()
            
            # Collect
            all_targets.append(batch['outcome'].detach().cpu())
            all_treatments.append(batch['treatment'].detach().cpu())
            all_y0.append(losses['y0_logit'].detach().cpu())
            all_y1.append(losses['y1_logit'].detach().cpu())
            all_prop.append(losses['t_logit'].detach().cpu())
            
        # Calc metrics
        metrics = _compute_epoch_metrics(epoch_loss, gen_loader, all_targets, all_treatments, all_y0, all_y1, all_prop)
        
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': metrics['loss'],
            'train_auroc_y0': metrics['auroc_y0'],
            'train_auroc_y1': metrics['auroc_y1'],
            'train_auroc_prop': metrics['auroc_prop'],
        }
        history.append(epoch_log)
    
    generator.eval()
    return generator, history


def _generate_plasmode_data(
    train_df: pd.DataFrame,
    generator: CausalDragonnetText,
    scenario: PlasmodeConfig,
    applied_config: AppliedInferenceConfig,
    device: torch.device,
    cache: Optional[EmbeddingCache],
    return_confounders: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """Generate synthetic plasmode outcomes using learned confounders.

    Args:
        train_df: Training dataframe with text data
        generator: Trained generator model
        scenario: Plasmode scenario configuration
        applied_config: Applied inference configuration
        device: Torch device
        cache: Optional embedding cache
        return_confounders: If True, also return the extracted confounder matrix

    Returns:
        If return_confounders=False: plasmode_df
        If return_confounders=True: (plasmode_df, confounders_array)
    """

    # Extract confounder representations
    gen_dataset = ClinicalTextDataset(
        data=train_df,
        text_column=applied_config.text_column,
        outcome_column=applied_config.outcome_column,
        treatment_column=applied_config.treatment_column,
        model=generator.sentence_transformer_model,
        device=device,
        chunk_size=generator.chunk_size,
        chunk_overlap=generator.chunk_overlap,
        cache=cache
    )

    gen_loader = DataLoader(gen_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

    all_confounders = []
    generator.eval()
    with torch.no_grad():
        for batch in gen_loader:
            chunk_embeddings = [
                batch['chunk_embeddings'][i, :, :].to(device).contiguous()
                for i in range(batch['chunk_embeddings'].size(0))
            ]
            # Extract raw confounder features (before DragonNet)
            # Use feature_extractor directly to get raw confounders without LayerNorm
            # This ensures generator and evaluator share the same confounder space
            confounders = generator.feature_extractor(chunk_embeddings)
            all_confounders.append(confounders.cpu())

    # Use raw confounder features for ITE generation
    # This is before DragonNet, so both generator and evaluator can learn in this space
    all_confounders = torch.cat(all_confounders, dim=0).numpy()
    plasmode_df = train_df.copy()

    # Treatments
    if scenario.preserve_observed_treatments:
        treatments = train_df[applied_config.treatment_column].values
    else:
        treatments = np.random.binomial(1, 0.5, size=len(train_df))

    # Outcomes and True ITE - generated from raw confounder features
    if scenario.generation_mode == "phi_linear":
        outcomes, true_ite, true_y0, true_y1 = _generate_linear_outcomes(all_confounders, treatments, scenario)
    elif scenario.generation_mode == "deep_nonlinear":
        outcomes, true_ite, true_y0, true_y1 = _generate_deep_nonlinear_outcomes(all_confounders, treatments, scenario, device)
    elif scenario.generation_mode == "uplift_nonlinear":
        outcomes, true_ite, true_y0, true_y1 = _generate_uplift_outcomes(all_confounders, treatments, scenario, device)
    else:
        raise ValueError(f"Unknown generation mode: {scenario.generation_mode}")

    plasmode_df[applied_config.treatment_column] = treatments
    plasmode_df[applied_config.outcome_column] = outcomes
    plasmode_df['true_ite'] = true_ite
    plasmode_df['true_y0_logit'] = true_y0
    plasmode_df['true_y1_logit'] = true_y1

    if return_confounders:
        return plasmode_df, all_confounders
    return plasmode_df


def _standardize(x: np.ndarray) -> np.ndarray:
    """Z-score standardization to mean=0, std=1 (matching old working code)."""
    std = np.std(x)
    if std < 1e-9:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def _generate_linear_outcomes(confounders, treatments, scenario):
    """
    Linear outcomes with explainable heterogeneity.
    Uses unit vector projections and z-score standardization (matching old working code).
    """
    n_samples, n_features = confounders.shape
    np.random.seed(42)

    # Base logit for control outcome rate
    base_logit = np.log(scenario.baseline_control_outcome_rate / (1 - scenario.baseline_control_outcome_rate))

    # 1. Prognostic Component - use normalized unit vector projection
    v0 = np.random.randn(n_features, 1)
    v0 = v0 / np.linalg.norm(v0)  # Unit vector
    s0 = confounders @ v0
    s0 = _standardize(s0)  # Z-score to ensure zero mean, unit variance
    logits_y0 = base_logit + scenario.outcome_heterogeneity_scale * s0.flatten()

    # 2. ITE Component - use normalized unit vector projection
    v_ite = np.random.randn(n_features, 1)
    v_ite = v_ite / np.linalg.norm(v_ite)  # Unit vector
    s_ite = confounders @ v_ite
    s_ite = _standardize(s_ite)  # Z-score to ensure zero mean, unit variance
    true_ite_logits = scenario.target_ate_logit + scenario.ite_heterogeneity_scale * s_ite.flatten()

    # 3. Combine
    logits_y1 = logits_y0 + true_ite_logits
    final_logit = logits_y0 + (true_ite_logits * treatments)

    probs = 1 / (1 + np.exp(-final_logit))
    outcomes = np.random.binomial(1, probs)
    return outcomes, true_ite_logits, logits_y0, logits_y1


def _generate_deep_nonlinear_outcomes(confounders, treatments, scenario, device):
    """
    Deep nonlinear outcomes. Uses z-score standardization then scales (matching old working code).
    """
    # Initialize MLP
    input_dim = confounders.shape[1] + 1
    hidden_dims = scenario.deep_nonlinear_hidden_dims
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(scenario.deep_nonlinear_dropout)
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 1))

    outcome_model = nn.Sequential(*layers).to(device)
    for module in outcome_model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)

    # Base logit for control outcome rate
    base_logit = np.log(scenario.baseline_control_outcome_rate / (1 - scenario.baseline_control_outcome_rate))

    outcome_model.eval()
    with torch.no_grad():
        X_0 = np.concatenate([confounders, np.zeros((len(confounders), 1))], axis=1)
        logits_0_raw = outcome_model(torch.tensor(X_0, dtype=torch.float32, device=device)).squeeze().cpu().numpy()

        X_1 = np.concatenate([confounders, np.ones((len(confounders), 1))], axis=1)
        logits_1_raw = outcome_model(torch.tensor(X_1, dtype=torch.float32, device=device)).squeeze().cpu().numpy()

        ite_raw = logits_1_raw - logits_0_raw

        # Standardize then scale (matching old working code pattern)
        logits_y0 = base_logit + scenario.outcome_heterogeneity_scale * _standardize(logits_0_raw)
        true_ite_logits = scenario.target_ate_logit + scenario.ite_heterogeneity_scale * _standardize(ite_raw)

        logits_y1 = logits_y0 + true_ite_logits
        final_logits = logits_y0 + (true_ite_logits * treatments)

        probs = 1 / (1 + np.exp(-final_logits))
        outcomes = np.random.binomial(1, probs)

    return outcomes, true_ite_logits, logits_y0, logits_y1


def _generate_uplift_outcomes(confounders, treatments, scenario, device):
    """
    Uplift model (Linear Base + Nonlinear ITE).
    Uses unit vector projections and z-score standardization (matching old working code).
    """
    n_samples, n_features = confounders.shape
    np.random.seed(42)

    # Base logit for control outcome rate
    base_logit = np.log(scenario.baseline_control_outcome_rate / (1 - scenario.baseline_control_outcome_rate))

    # 1. Base Model - use normalized unit vector projection
    v0 = np.random.randn(n_features, 1)
    v0 = v0 / np.linalg.norm(v0)  # Unit vector
    s0 = confounders @ v0
    logits_y0 = base_logit + scenario.outcome_heterogeneity_scale * s0.flatten()

    # 2. Uplift Model
    if scenario.uplift_hidden_dims:
        input_dim = confounders.shape[1]
        layers = []
        prev_dim = input_dim
        for hidden_dim in scenario.uplift_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if scenario.uplift_activation == 'relu' else nn.Tanh(),
                nn.Dropout(scenario.uplift_dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        uplift_model = nn.Sequential(*layers).to(device)

        for module in uplift_model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

        uplift_model.eval()
        with torch.no_grad():
            ite_raw = uplift_model(torch.tensor(confounders, dtype=torch.float32, device=device)).squeeze().cpu().numpy()
    else:
        # Use normalized unit vector projection for linear ITE
        v_ite = np.random.randn(n_features, 1)
        v_ite = v_ite / np.linalg.norm(v_ite)  # Unit vector
        ite_raw = (confounders @ v_ite).flatten()

    # Scale raw ITE
    true_ite_logits = scenario.target_ate_logit + scenario.ite_heterogeneity_scale * ite_raw

    logits_y1 = logits_y0 + true_ite_logits
    final_logits = logits_y0 + (true_ite_logits * treatments)

    probs = 1 / (1 + np.exp(-final_logits))
    outcomes = np.random.binomial(1, probs)
    return outcomes, true_ite_logits, logits_y0, logits_y1


def _train_plasmode_evaluator(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    applied_config: AppliedInferenceConfig,
    plasmode_config: PlasmodeExperimentConfig,
    device: torch.device,
    cache: Optional[EmbeddingCache],
    pretrained_weights_path: Optional[Path],
    explicit_confounder_embeddings: Optional[torch.Tensor] = None
) -> Tuple[CausalDragonnetText, List[Dict[str, Any]]]:
    """Train evaluator model on plasmode TRAINING data (realistic mode). Returns (model, history).
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe. If None, skip validation and return None for val metrics.
        applied_config: Applied inference configuration
        plasmode_config: Plasmode experiment configuration
        device: Torch device
        cache: Optional embedding cache
        pretrained_weights_path: Optional path to pretrained weights
        explicit_confounder_embeddings: Optional precomputed confounder embeddings
    """
    
    eval_arch = plasmode_config.evaluator_architecture
    evaluator = CausalDragonnetText(
        sentence_transformer_model_name=eval_arch.embedding_model_name,
        num_latent_confounders=eval_arch.num_latent_confounders,
        features_per_confounder=eval_arch.features_per_confounder,
        explicit_confounder_texts=eval_arch.explicit_confounder_texts,
        aggregator_mode=eval_arch.aggregator_mode,
        dragonnet_representation_dim=eval_arch.dragonnet_representation_dim,
        dragonnet_hidden_outcome_dim=eval_arch.dragonnet_hidden_outcome_dim,
        chunk_size=eval_arch.chunk_size,
        chunk_overlap=eval_arch.chunk_overlap,
        device=str(device),
        model_type=eval_arch.model_type
    )
    
    if pretrained_weights_path is not None:
        try:
            pretrained_checkpoint = torch.load(pretrained_weights_path, map_location=device)
            load_pretrained_with_dimension_matching(evaluator, pretrained_checkpoint, strict=False, auto_adjust=True)
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
    
    # Training dataset
    train_dataset = ClinicalTextDataset(
        data=train_df,
        text_column=applied_config.text_column,
        outcome_column=applied_config.outcome_column,
        treatment_column=applied_config.treatment_column,
        model=evaluator.sentence_transformer_model,
        device=device,
        chunk_size=eval_arch.chunk_size,
        chunk_overlap=eval_arch.chunk_overlap,
        cache=cache
    )
    
    # Validation dataset (optional)
    val_loader = None
    if val_df is not None:
        val_dataset = ClinicalTextDataset(
            data=val_df,
            text_column=applied_config.text_column,
            outcome_column=applied_config.outcome_column,
            treatment_column=applied_config.treatment_column,
            model=evaluator.sentence_transformer_model,
            device=device,
            chunk_size=eval_arch.chunk_size,
            chunk_overlap=eval_arch.chunk_overlap,
            cache=cache
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=plasmode_config.evaluator_training.batch_size,
            shuffle=False,
            collate_fn=collate_batch
        )
    
    # Training loader
    eval_loader = DataLoader(
        train_dataset, 
        batch_size=plasmode_config.evaluator_training.batch_size, 
        shuffle=True, 
        collate_fn=collate_batch
    )
    
    eval_train = plasmode_config.evaluator_training
    optimizer = torch.optim.AdamW(
        evaluator.parameters(), 
        lr=eval_train.learning_rate,
        weight_decay=1e-4  # Match old_cdt behavior
    )
    
    if eval_train.lr_schedule == "linear":
        total_steps = len(eval_loader) * eval_train.epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    else:
        scheduler = None
    
    evaluator.train()
    history = []
    
    for epoch in range(eval_train.epochs):
        epoch_losses = {'total': 0.0, 'outcome': 0.0, 'propensity': 0.0, 'targreg': 0.0}
        
        # TRAIN LOOP
        evaluator.train()
        all_targets = []
        all_treatments = []
        all_y0 = []
        all_y1 = []
        all_prop = []
        
        for batch in eval_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            batch['chunk_embeddings'] = [batch['chunk_embeddings'][i, :, :].contiguous() for i in range(batch['chunk_embeddings'].size(0))]
            
            optimizer.zero_grad()
            losses = evaluator.train_step(batch, alpha_propensity=eval_train.alpha_propensity, beta_targreg=eval_train.beta_targreg)
            losses['loss'].backward()
            #torch.nn.utils.clip_grad_norm_(evaluator.parameters(), max_norm=1.0)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
                
            epoch_losses['total'] += losses['loss'].item()
            epoch_losses['outcome'] += losses['outcome_loss'].item()
            epoch_losses['propensity'] += losses['propensity_loss'].item()
            epoch_losses['targreg'] += (losses['targreg_loss'].item() if torch.is_tensor(losses['targreg_loss']) else losses['targreg_loss'])
            
            all_targets.append(batch['outcome'].detach().cpu())
            all_treatments.append(batch['treatment'].detach().cpu())
            all_y0.append(losses['y0_logit'].detach().cpu())
            all_y1.append(losses['y1_logit'].detach().cpu())
            all_prop.append(losses['t_logit'].detach().cpu())

        train_metrics = _compute_epoch_metrics(epoch_losses['total'], eval_loader, all_targets, all_treatments, all_y0, all_y1, all_prop)

        # VALIDATION LOOP (only if val_df was provided)
        if val_loader is not None:
            val_losses = {'total': 0.0, 'outcome': 0.0, 'propensity': 0.0, 'targreg': 0.0}
            evaluator.eval()
            val_targets = []
            val_treatments = []
            val_y0 = []
            val_y1 = []
            val_prop = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    batch['chunk_embeddings'] = [batch['chunk_embeddings'][i, :, :].contiguous() for i in range(batch['chunk_embeddings'].size(0))]
                    
                    losses = evaluator.train_step(batch, alpha_propensity=eval_train.alpha_propensity, beta_targreg=eval_train.beta_targreg)
                    val_losses['total'] += losses['loss'].item()
                    val_losses['outcome'] += losses['outcome_loss'].item()
                    val_losses['propensity'] += losses['propensity_loss'].item()
                    val_losses['targreg'] += (losses['targreg_loss'].item() if torch.is_tensor(losses['targreg_loss']) else losses['targreg_loss'])
                    
                    val_targets.append(batch['outcome'].detach().cpu())
                    val_treatments.append(batch['treatment'].detach().cpu())
                    val_y0.append(losses['y0_logit'].detach().cpu())
                    val_y1.append(losses['y1_logit'].detach().cpu())
                    val_prop.append(losses['t_logit'].detach().cpu())
            
            val_metrics = _compute_epoch_metrics(val_losses['total'], val_loader, val_targets, val_treatments, val_y0, val_y1, val_prop)
            
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_auroc_y0': train_metrics['auroc_y0'],
                'train_auroc_y1': train_metrics['auroc_y1'],
                'train_auroc_prop': train_metrics['auroc_prop'],
                'val_loss': val_metrics['loss'],
                'val_auroc_y0': val_metrics['auroc_y0'],
                'val_auroc_y1': val_metrics['auroc_y1'],
                'val_auroc_prop': val_metrics['auroc_prop'],
            }
        else:
            # No validation set - report None for validation metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_auroc_y0': train_metrics['auroc_y0'],
                'train_auroc_y1': train_metrics['auroc_y1'],
                'train_auroc_prop': train_metrics['auroc_prop'],
                'val_loss': None,
                'val_auroc_y0': None,
                'val_auroc_y1': None,
                'val_auroc_prop': None,
            }
        history.append(epoch_log)

    return evaluator, history


def _predict_plasmode_evaluator(
    evaluator: CausalDragonnetText,
    df: pd.DataFrame,
    applied_config: AppliedInferenceConfig,
    plasmode_config: PlasmodeExperimentConfig,
    device: torch.device,
    cache: Optional[EmbeddingCache]
) -> dict:
    """Generate ITE predictions (logits) for a dataframe."""
    
    dataset = ClinicalTextDataset(
        data=df,
        text_column=applied_config.text_column,
        outcome_column=applied_config.outcome_column,
        treatment_column=applied_config.treatment_column,
        model=evaluator.sentence_transformer_model,
        device=device,
        chunk_size=plasmode_config.evaluator_architecture.chunk_size,
        chunk_overlap=plasmode_config.evaluator_architecture.chunk_overlap,
        cache=cache
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
    
    evaluator.eval()
    all_ite_logits = []
    all_y0_logits = []
    all_y1_logits = []
    all_propensity = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating predictions", leave=False):
            chunk_embeddings = [batch['chunk_embeddings'][i, :, :].to(device).contiguous() for i in range(batch['chunk_embeddings'].size(0))]
            preds = evaluator.predict(chunk_embeddings)
            
            all_ite_logits.append((preds['y1_logit'] - preds['y0_logit']).cpu().numpy())
            all_y0_logits.append(preds['y0_logit'].cpu().numpy())
            all_y1_logits.append(preds['y1_logit'].cpu().numpy())
            all_propensity.append(preds['propensity'].cpu().numpy())
            
    return {
        'ite': np.concatenate(all_ite_logits),
        'y0_logit': np.concatenate(all_y0_logits),
        'y1_logit': np.concatenate(all_y1_logits),
        'propensity': np.concatenate(all_propensity)
    }


# =============================================================================
# Confounder-based evaluator (oracle mode: uses generator's confounders directly)
# =============================================================================

class ConfounderDataset(torch.utils.data.Dataset):
    """Simple dataset for pre-extracted confounders."""

    def __init__(self, confounders: np.ndarray, treatments: np.ndarray, outcomes: np.ndarray):
        self.confounders = torch.tensor(confounders, dtype=torch.float32)
        self.treatments = torch.tensor(treatments, dtype=torch.float32)
        self.outcomes = torch.tensor(outcomes, dtype=torch.float32)

    def __len__(self):
        return len(self.confounders)

    def __getitem__(self, idx):
        return {
            'confounders': self.confounders[idx],
            'treatment': self.treatments[idx],
            'outcome': self.outcomes[idx]
        }


def _train_confounder_evaluator(
    train_confounder_features: np.ndarray,
    train_df: pd.DataFrame,
    val_confounder_features: Optional[np.ndarray],
    val_df: Optional[pd.DataFrame],
    applied_config: AppliedInferenceConfig,
    plasmode_config: PlasmodeExperimentConfig,
    device: torch.device
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """
    Train outcome heads directly on pre-extracted confounder_features (oracle mode).

    This bypasses the text -> embedding -> confounder pipeline and trains only
    lightweight outcome heads on the generator's confounder_features. Since ITEs are
    generated from confounder_features, this ensures the evaluator can properly recover them.

    Args:
        train_confounder_features: Confounder features matrix for training set
            Shape: (n_train, num_confounders * features_per_confounder)
        train_df: Training dataframe with outcome and treatment columns
        val_confounder_features: Confounder features matrix for validation set. If None, skip validation.
        val_df: Validation dataframe. If None, skip validation.
        applied_config: Applied inference configuration
        plasmode_config: Plasmode experiment configuration
        device: Torch device

    Returns:
        Tuple of (trained outcome heads model, training history)
    """
    from ..models.outcome_heads import OutcomeHeadsOnly, UpliftHeadsOnly

    confounder_dim = train_confounder_features.shape[1]
    eval_arch = plasmode_config.evaluator_architecture
    eval_training = plasmode_config.evaluator_training

    # Create lightweight outcome heads that take confounder_features directly (no representation layers!)
    # Using full DragonNet/UpliftNet here would double-process through 6 more
    # representation layers, causing severe attenuation of the ITE signal.
    if eval_arch.model_type == "uplift":
        model = UpliftHeadsOnly(
            confounder_dim=confounder_dim,
            hidden_outcome_dim=eval_arch.dragonnet_hidden_outcome_dim
        ).to(device)
    else:
        model = OutcomeHeadsOnly(
            confounder_dim=confounder_dim,
            hidden_outcome_dim=eval_arch.dragonnet_hidden_outcome_dim
        ).to(device)

    # Create training dataset
    train_dataset = ConfounderDataset(
        confounders=train_confounder_features,
        treatments=train_df[applied_config.treatment_column].values,
        outcomes=train_df[applied_config.outcome_column].values
    )
    train_loader = DataLoader(train_dataset, batch_size=eval_training.batch_size, shuffle=True)

    # Create validation dataset (optional)
    val_loader = None
    if val_confounder_features is not None and val_df is not None:
        val_dataset = ConfounderDataset(
            confounders=val_confounder_features,
            treatments=val_df[applied_config.treatment_column].values,
            outcomes=val_df[applied_config.outcome_column].values
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Optimizer
    if eval_training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=eval_training.learning_rate,
            weight_decay=1e-4  # Match old_cdt behavior
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=eval_training.learning_rate)

    # Learning rate scheduler
    if eval_training.lr_schedule == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=eval_training.epochs
        )
    else:
        scheduler = None

    history = []
    alpha_propensity = eval_training.alpha_propensity
    beta_targreg = eval_training.beta_targreg

    for epoch in range(eval_training.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            confounders = batch['confounders'].to(device)
            treatments = batch['treatment'].to(device)
            outcomes = batch['outcome'].to(device)

            optimizer.zero_grad()

            if eval_arch.model_type == "uplift":
                y0_logit, tau_logit, t_logit, phi = model(confounders)
                y1_logit = y0_logit + tau_logit
            else:
                y0_logit, y1_logit, t_logit, phi = model(confounders)

            # Propensity loss
            propensity_loss = F.binary_cross_entropy_with_logits(
                t_logit.squeeze(-1), treatments
            )

            # Outcome loss (factual only)
            factual_logit = torch.where(
                treatments.unsqueeze(1) > 0.5,
                y1_logit,
                y0_logit
            )
            outcome_loss = F.binary_cross_entropy_with_logits(
                factual_logit.squeeze(-1), outcomes
            )

            # Targeted regularization
            targreg_loss = torch.tensor(0.0, device=device)
            if beta_targreg > 0:
                propensity = torch.sigmoid(t_logit).clamp(1e-3, 1 - 1e-3)
                H = (treatments.unsqueeze(1) / propensity) - \
                    ((1 - treatments.unsqueeze(1)) / (1 - propensity))
                factual_prob = torch.sigmoid(factual_logit)
                moment = torch.mean((outcomes.unsqueeze(1) - factual_prob) * H)
                targreg_loss = moment ** 2

            total_loss = outcome_loss + alpha_propensity * propensity_loss + beta_targreg * targreg_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        if scheduler:
            scheduler.step()

        # Validation (only if val data was provided)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            all_y0, all_y1, all_prop = [], [], []
            all_targets, all_treatments_val = [], []

            with torch.no_grad():
                for batch in val_loader:
                    confounders = batch['confounders'].to(device)
                    treatments = batch['treatment'].to(device)
                    outcomes = batch['outcome'].to(device)

                    if eval_arch.model_type == "uplift":
                        y0_logit, tau_logit, t_logit, phi = model(confounders)
                        y1_logit = y0_logit + tau_logit
                    else:
                        y0_logit, y1_logit, t_logit, phi = model(confounders)

                    propensity_loss = F.binary_cross_entropy_with_logits(
                        t_logit.squeeze(-1), treatments
                    )
                    factual_logit = torch.where(
                        treatments.unsqueeze(1) > 0.5,
                        y1_logit,
                        y0_logit
                    )
                    outcome_loss = F.binary_cross_entropy_with_logits(
                        factual_logit.squeeze(-1), outcomes
                    )
                    val_loss += (outcome_loss + alpha_propensity * propensity_loss).item()
                    val_batches += 1

                    all_y0.append(torch.sigmoid(y0_logit).cpu())
                    all_y1.append(torch.sigmoid(y1_logit).cpu())
                    all_prop.append(torch.sigmoid(t_logit).cpu())
                    all_targets.append(outcomes.cpu())
                    all_treatments_val.append(treatments.cpu())

            # Compute AUROCs
            y_true = torch.cat(all_targets).numpy()
            t_true = torch.cat(all_treatments_val).numpy()
            y0_scores = torch.cat(all_y0).numpy().squeeze()
            y1_scores = torch.cat(all_y1).numpy().squeeze()
            prop_scores = torch.cat(all_prop).numpy().squeeze()

            try:
                auroc_y0 = roc_auc_score(y_true[t_true == 0], y0_scores[t_true == 0]) if (t_true == 0).sum() > 0 else np.nan
                auroc_y1 = roc_auc_score(y_true[t_true == 1], y1_scores[t_true == 1]) if (t_true == 1).sum() > 0 else np.nan
                auroc_prop = roc_auc_score(t_true, prop_scores)
            except:
                auroc_y0, auroc_y1, auroc_prop = np.nan, np.nan, np.nan

            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': epoch_loss / max(n_batches, 1),
                'val_loss': val_loss / max(val_batches, 1),
                'val_auroc_y0': auroc_y0,
                'val_auroc_y1': auroc_y1,
                'val_auroc_prop': auroc_prop,
            }
        else:
            # No validation set - report None for validation metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': epoch_loss / max(n_batches, 1),
                'val_loss': None,
                'val_auroc_y0': None,
                'val_auroc_y1': None,
                'val_auroc_prop': None,
            }
        history.append(epoch_log)

    return model, history


def _predict_confounder_evaluator(
    model: nn.Module,
    confounder_features: np.ndarray,
    device: torch.device
) -> dict:
    """
    Generate ITE predictions from outcome heads using pre-extracted confounder_features.

    Args:
        model: Trained OutcomeHeadsOnly/UpliftHeadsOnly model
        confounder_features: Confounder features matrix (n_samples, num_confounders * features_per_confounder)
        device: Torch device

    Returns:
        Dictionary with 'ite', 'y0_logit', 'y1_logit', 'propensity' arrays
    """
    from ..models.outcome_heads import UpliftHeadsOnly

    model.eval()
    is_uplift = isinstance(model, UpliftHeadsOnly)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(confounder_features, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_ite_logits = []
    all_y0_logits = []
    all_y1_logits = []
    all_propensity = []

    with torch.no_grad():
        for (batch_cf,) in loader:
            batch_cf = batch_cf.to(device)

            if is_uplift:
                y0_logit, tau_logit, t_logit, _ = model(batch_cf)
                y1_logit = y0_logit + tau_logit
                ite_logit = tau_logit
            else:
                y0_logit, y1_logit, t_logit, _ = model(batch_cf)
                ite_logit = y1_logit - y0_logit

            all_ite_logits.append(ite_logit.squeeze(-1).cpu().numpy())
            all_y0_logits.append(y0_logit.squeeze(-1).cpu().numpy())
            all_y1_logits.append(y1_logit.squeeze(-1).cpu().numpy())
            all_propensity.append(torch.sigmoid(t_logit).squeeze(-1).cpu().numpy())

    return {
        'ite': np.concatenate(all_ite_logits),
        'y0_logit': np.concatenate(all_y0_logits),
        'y1_logit': np.concatenate(all_y1_logits),
        'propensity': np.concatenate(all_propensity)
    }


def _evaluate_plasmode_performance(
    df: pd.DataFrame,
    predicted_ite: np.ndarray,
    target_ate_logit: float
) -> dict:
    """
    Evaluate plasmode performance by comparing predicted ITEs to true ITEs.
    
    Args:
        df: DataFrame with 'true_ite' column containing ground truth ITE logits
        predicted_ite: Array of predicted ITE logits from the evaluator
        target_ate_logit: The target ATE (logit) used in simulation
        
    Returns:
        Dictionary with performance metrics:
        - ate_bias: Difference between estimated and true mean ATE
        - ate_rmse: Root mean squared error of ATE estimation
        - ite_correlation: Pearson correlation between predicted and true ITE
        - ite_regression_slope: Slope of predicted vs true ITE regression
    """
    true_ite = df['true_ite'].values
    
    # ATE metrics
    true_ate = np.mean(true_ite)
    estimated_ate = np.mean(predicted_ite)
    ate_bias = estimated_ate - true_ate
    ate_rmse = np.sqrt(np.mean((predicted_ite - true_ite) ** 2))
    
    # ITE correlation
    if np.std(true_ite) > 1e-9 and np.std(predicted_ite) > 1e-9:
        ite_correlation = np.corrcoef(true_ite, predicted_ite)[0, 1]
    else:
        ite_correlation = np.nan
    
    # ITE regression slope (predicted vs true)
    if np.var(true_ite) > 1e-9:
        ite_regression_slope = np.cov(true_ite, predicted_ite)[0, 1] / np.var(true_ite)
    else:
        ite_regression_slope = np.nan
    
    return {
        'ate_bias': ate_bias,
        'ate_rmse': ate_rmse,
        'ite_correlation': ite_correlation,
        'ite_regression_slope': ite_regression_slope,
        'true_ate': true_ate,
        'estimated_ate': estimated_ate,
        'target_ate': target_ate_logit
    }