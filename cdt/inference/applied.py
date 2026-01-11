# cdt/inference/applied.py
"""Applied causal inference on real clinical data."""

import gc
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

from ..config import AppliedInferenceConfig
from ..models.causal_dragonnet import CausalDragonnetText
from ..models.causal_modernbert import CausalModernBertText
from ..models.causal_cnn import CausalCNNText
from ..data import (
    ClinicalTextDataset, ModernBertClinicalTextDataset,
    collate_batch, collate_modernbert_batch,
    load_dataset, EmbeddingCache
)
from ..utils import (
    cuda_cleanup, get_memory_info, load_pretrained_with_dimension_matching,
    compute_latent_drift, log_latent_drift, compute_confounder_feature_stats, log_confounder_stats
)


logger = logging.getLogger(__name__)


def run_applied_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    cache: Optional[EmbeddingCache] = None,
    pretrained_weights_path: Optional[Path] = None,
    gpu_ids: Optional[List[int]] = None,
    num_workers: int = 1
) -> None:
    """
    Run applied causal inference on real data.
    """
    logger.info("="*80)
    logger.info("APPLIED CAUSAL INFERENCE")
    logger.info("="*80)
    
    # Determine mode
    if config.cv_folds > 1:
        _run_cv_inference(
            dataset, config, output_path, device, cache, pretrained_weights_path, gpu_ids, num_workers
        )
    else:
        _run_fixed_split_inference(
            dataset, config, output_path, device, cache, pretrained_weights_path
        )


def _run_cv_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    cache: Optional[EmbeddingCache],
    pretrained_weights_path: Optional[Path],
    gpu_ids: Optional[List[int]] = None,
    num_workers: int = 1
) -> None:
    """Run K-Fold Cross-Validation inference."""
    k = config.cv_folds
    logger.info(f"Starting {k}-Fold Cross-Validation on {len(dataset)} samples")
    
    # Reset index to ensure KFold works with indices
    dataset = dataset.reset_index(drop=True)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = list(kf.split(dataset))
    
    # Determine devices to use
    if gpu_ids:
        devices = [torch.device(f"cuda:{i}") for i in gpu_ids]
    else:
        devices = [device]
    
    # num_workers is now passed directly
    # num_workers = getattr(config, 'num_workers', 1)
    
    if num_workers > 1:
        logger.info(f"Parallelizing across {num_workers} workers on devices: {devices}")
        
        results = Parallel(n_jobs=num_workers)(
            delayed(_process_fold)(
                fold, train_idx, test_idx, dataset, config, 
                devices[fold % len(devices)], cache, pretrained_weights_path
            )
            for fold, (train_idx, test_idx) in enumerate(splits)
        )
    else:
        results = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            results.append(_process_fold(
                fold, train_idx, test_idx, dataset, config, 
                devices[0], cache, pretrained_weights_path
            ))
    
    # Unpack results
    all_predictions = [r[0] for r in results]
    all_training_logs = [log for r in results for log in r[1]]
    
    # Combine predictions and save
    results_df = pd.concat(all_predictions).sort_index()
    _save_and_summarize(results_df, output_path)

    # Save training logs
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(all_training_logs).to_csv(log_path, index=False)
    logger.info(f"Training logs saved to: {log_path}")


def _process_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device,
    cache: Optional[EmbeddingCache],
    pretrained_weights_path: Optional[Path]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Process a single fold (can be run in parallel)."""
    # Re-configure logger for worker process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"FOLD {fold+1} starting on {device}")

    # 1. Prepare Data for this Fold
    # Use all training indices for training - no internal validation split
    # Validation metrics will be computed on the held-out test fold
    train_df = dataset.iloc[train_idx]
    test_df = dataset.iloc[test_idx]

    # 2. Train Model on this fold
    # Pass test_df as val_df so validation metrics represent held-out fold performance
    model, history = _train_single_model(
        train_df, test_df, config, device, cache, pretrained_weights_path
    )
    
    # Log History
    for entry in history:
        entry['fold'] = fold + 1
    
    # 3. Predict on Held-out Test fold
    preds = _predict_dataset(model, test_df, config, device, cache)
    
    # 4. Store predictions with indices to reconstruct dataframe (logit scale)
    preds_df = test_df.copy()
    preds_df['y0_pred'] = preds['y0_logit']
    preds_df['y1_pred'] = preds['y1_logit']
    preds_df['ite_pred'] = preds['ite_logit']
    preds_df['propensity_pred'] = preds['propensity_logit']
    preds_df['cv_fold'] = fold + 1

    # Aggressive GPU cleanup to prevent OOM across folds
    # Move model to CPU first to release GPU memory before deletion
    model.cpu()
    del model
    del preds
    del train_df, test_df

    # Force garbage collection before CUDA cleanup
    gc.collect()

    # Clear CUDA cache aggressively
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Second round of cleanup
    gc.collect()
    cuda_cleanup()

    logger.info(f"FOLD {fold+1} complete | {get_memory_info()}")
    return preds_df, history


def _run_fixed_split_inference(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    output_path: Path,
    device: torch.device,
    cache: Optional[EmbeddingCache],
    pretrained_weights_path: Optional[Path]
) -> None:
    """Run inference using fixed train/val/test splits."""
    logger.info("Running Fixed Split Inference (Train/Val/Test)")
    
    # Split data
    train_df = dataset[dataset[config.split_column] == 'train'].copy()
    val_df = dataset[dataset[config.split_column] == 'val'].copy()
    test_df = dataset[dataset[config.split_column] == 'test'].copy()
    
    logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train
    model, history = _train_single_model(
        train_df, val_df, config, device, cache, pretrained_weights_path
    )
    
    # Save training logs
    log_path = output_path.parent / "training_log.csv"
    pd.DataFrame(history).to_csv(log_path, index=False)
    logger.info(f"Training logs saved to: {log_path}")
    
    # Predict on Test
    logger.info("Generating predictions on test set...")
    preds = _predict_dataset(model, test_df, config, device, cache)
    
    # Combine (logit scale)
    results_df = test_df.copy()
    results_df['y0_pred'] = preds['y0_logit']
    results_df['y1_pred'] = preds['y1_logit']
    results_df['ite_pred'] = preds['ite_logit']
    results_df['propensity_pred'] = preds['propensity_logit']
    
    _save_and_summarize(results_df, output_path)


def _train_single_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device,
    cache: Optional[EmbeddingCache],
    pretrained_weights_path: Optional[Path]
) -> Tuple[Union[CausalDragonnetText, CausalModernBertText, CausalCNNText], List[Dict[str, Any]]]:
    """Helper to train a single model instance."""

    arch_config = config.architecture
    model_backbone = getattr(arch_config, 'model_backbone', 'sentence_transformer')
    use_modernbert = model_backbone == 'modernbert'
    use_cnn = model_backbone == 'cnn'

    if use_cnn:
        # Create CNN-based model
        model = CausalCNNText(
            embedding_dim=arch_config.cnn_embedding_dim,
            num_filters=arch_config.cnn_num_filters,
            kernel_sizes=arch_config.cnn_kernel_sizes,
            cnn_dropout=arch_config.cnn_dropout,
            max_length=arch_config.cnn_max_length,
            min_word_freq=getattr(arch_config, 'cnn_min_word_freq', 2),
            max_vocab_size=getattr(arch_config, 'cnn_max_vocab_size', 50000),
            projection_dim=arch_config.dragonnet_representation_dim,
            dragonnet_representation_dim=arch_config.dragonnet_representation_dim,
            dragonnet_hidden_outcome_dim=arch_config.dragonnet_hidden_outcome_dim,
            device=str(device),
            model_type=arch_config.model_type
        )
        logger.info("Using CNN backbone")

        # Fit word tokenizer on training texts for this fold
        train_texts = train_df[config.text_column].tolist()
        model.fit_tokenizer(train_texts)
        logger.info(f"Fitted word tokenizer on {len(train_texts)} training texts")

        # No latent tracking for CNN
        initial_latents = None

        # Create datasets for CNN (same as ModernBERT - uses raw text)
        train_dataset = ModernBertClinicalTextDataset(
            data=train_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )

        val_dataset = ModernBertClinicalTextDataset(
            data=val_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )

        collate_fn = collate_modernbert_batch

    elif use_modernbert:
        # Create ModernBERT-based model
        model = CausalModernBertText(
            modernbert_model_name=arch_config.modernbert_model_name,
            projection_dim=arch_config.dragonnet_representation_dim,
            freeze_bert=arch_config.freeze_modernbert,
            max_length=arch_config.modernbert_max_length,
            dragonnet_representation_dim=arch_config.dragonnet_representation_dim,
            dragonnet_hidden_outcome_dim=arch_config.dragonnet_hidden_outcome_dim,
            device=str(device),
            model_type=arch_config.model_type
        )
        logger.info("Using ModernBERT backbone")
        
        # No latent tracking for ModernBERT
        initial_latents = None
        
        # Create datasets for ModernBERT
        train_dataset = ModernBertClinicalTextDataset(
            data=train_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )
        
        val_dataset = ModernBertClinicalTextDataset(
            data=val_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )
        
        collate_fn = collate_modernbert_batch
        
    else:
        # Create SentenceTransformer-based model (original)
        model = CausalDragonnetText(
            sentence_transformer_model_name=arch_config.embedding_model_name,
            num_latent_confounders=arch_config.num_latent_confounders,
            explicit_confounder_texts=arch_config.explicit_confounder_texts,
            value_dim=arch_config.value_dim,
            num_attention_heads=arch_config.num_attention_heads,
            attention_dropout=arch_config.attention_dropout,
            dragonnet_representation_dim=arch_config.dragonnet_representation_dim,
            dragonnet_hidden_outcome_dim=arch_config.dragonnet_hidden_outcome_dim,
            chunk_size=arch_config.chunk_size,
            chunk_overlap=arch_config.chunk_overlap,
            device=str(device),
            model_type=arch_config.model_type
        )
        
        # Load pretrained weights (SentenceTransformer only)
        if pretrained_weights_path is not None and config.use_pretrained_weights:
            try:
                pretrained_checkpoint = torch.load(
                    pretrained_weights_path, map_location=device, weights_only=False
                )
                load_pretrained_with_dimension_matching(
                    model, pretrained_checkpoint, strict=False, auto_adjust=True
                )
                logger.info("    ✓ Loaded pretrained weights")
            except Exception as e:
                logger.warning(f"    ✗ Failed to load pretrained weights: {e}")
        
        # Snapshot initial latent confounders for drift tracking
        initial_latents = None
        if model.feature_extractor.latent_confounders is not None:
            initial_latents = model.feature_extractor.latent_confounders.data.clone()
        
        # Initialize latents if needed
        if pretrained_weights_path is None and config.training.init_latents_from_kmeans:
            _initialize_latents_kmeans(model, train_df, cache, device)
        
        # Create datasets for SentenceTransformer
        train_dataset = ClinicalTextDataset(
            data=train_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            model=model.sentence_transformer_model,
            device=device,
            chunk_size=arch_config.chunk_size,
            chunk_overlap=arch_config.chunk_overlap,
            cache=cache
        )
        
        val_dataset = ClinicalTextDataset(
            data=val_df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            model=model.sentence_transformer_model,
            device=device,
            chunk_size=arch_config.chunk_size,
            chunk_overlap=arch_config.chunk_overlap,
            cache=cache
        )
        
        collate_fn = collate_batch
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Optimization
    train_config = config.training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=1e-4
    )
    
    if train_config.lr_schedule == "linear":
        total_steps = len(train_loader) * train_config.epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
        )
    else:
        scheduler = None
        
    # Training Loop
    best_val_loss = float('inf')
    best_model_state = None
    history = []
    
    # Select training functions based on backbone
    use_text_interface = use_modernbert or use_cnn  # Both use raw text strings

    for epoch in range(train_config.epochs):
        model.train()
        train_stats = _train_epoch_modernbert(model, train_loader, optimizer, scheduler, device, train_config) if use_text_interface else _train_epoch(model, train_loader, optimizer, scheduler, device, train_config)

        model.eval()
        val_stats = _eval_epoch_modernbert(model, val_loader, device, train_config) if use_text_interface else _eval_epoch(model, val_loader, device, train_config)
        
        # Record history
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_stats['loss'],
            'train_auroc_y0': train_stats['auroc_y0'],
            'train_auroc_y1': train_stats['auroc_y1'],
            'train_auroc_prop': train_stats['auroc_prop'],
            'val_loss': val_stats['loss'],
            'val_auroc_y0': val_stats['auroc_y0'],
            'val_auroc_y1': val_stats['auroc_y1'],
            'val_auroc_prop': val_stats['auroc_prop'],
        }
        history.append(epoch_log)
        
        # Save best
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            best_model_state = model.state_dict()
    
    # Restore best
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Compute and log latent drift (SentenceTransformer only)
    if not use_text_interface and initial_latents is not None:
        if hasattr(model.feature_extractor, 'latent_confounders') and model.feature_extractor.latent_confounders is not None:
            drift_df = compute_latent_drift(initial_latents, model.feature_extractor.latent_confounders.data)
            log_latent_drift(drift_df, prefix="Applied ")

    # Cleanup training artifacts to free GPU memory
    del train_loader, val_loader, train_dataset, val_dataset
    del optimizer, best_model_state
    if scheduler is not None:
        del scheduler
    gc.collect()

    return model, history


def _predict_dataset(
    model: Union[CausalDragonnetText, CausalModernBertText, CausalCNNText],
    df: pd.DataFrame,
    config: AppliedInferenceConfig,
    device: torch.device,
    cache: Optional[EmbeddingCache]
) -> dict:
    """Generate predictions for a dataframe."""
    arch_config = config.architecture
    model_backbone = getattr(arch_config, 'model_backbone', 'sentence_transformer')
    use_text_interface = model_backbone in ('modernbert', 'cnn')

    if use_text_interface:
        dataset = ModernBertClinicalTextDataset(
            data=df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column
        )
        collate_fn = collate_modernbert_batch
    else:
        dataset = ClinicalTextDataset(
            data=df,
            text_column=config.text_column,
            outcome_column=config.outcome_column,
            treatment_column=config.treatment_column,
            model=model.sentence_transformer_model,
            device=device,
            chunk_size=arch_config.chunk_size,
            chunk_overlap=arch_config.chunk_overlap,
            cache=cache
        )
        collate_fn = collate_batch

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    if use_text_interface:
        result = _generate_predictions_modernbert(model, loader, device, config)
    else:
        result = _generate_predictions(model, loader, device, config)

    # Cleanup prediction artifacts
    del loader, dataset
    gc.collect()

    return result



def _train_epoch(
    model: CausalDragonnetText,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    config
) -> dict:
    """Train for one epoch."""
    epoch_loss = 0.0
    all_targets = []
    all_treatments = []
    all_y0 = []
    all_y1 = []
    all_prop = []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        
        chunk_embeddings_list = [
            batch['chunk_embeddings'][i, :, :].contiguous()
            for i in range(batch['chunk_embeddings'].size(0))
        ]
        batch['chunk_embeddings'] = chunk_embeddings_list
        
        optimizer.zero_grad()
        
        losses = model.train_step(
            batch,
            alpha_propensity=config.alpha_propensity,
            beta_targreg=config.beta_targreg
        )
        
        losses['loss'].backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        epoch_loss += losses['loss'].item()
        
        # Collect for metrics
        all_targets.append(batch['outcome'].detach().cpu())
        all_treatments.append(batch['treatment'].detach().cpu())
        all_y0.append(losses['y0_logit'].detach().cpu())
        all_y1.append(losses['y1_logit'].detach().cpu())
        all_prop.append(losses['t_logit'].detach().cpu())
    
    return _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop)


def _eval_epoch(
    model: CausalDragonnetText,
    loader: DataLoader,
    device: torch.device,
    config
) -> dict:
    """Evaluate for one epoch."""
    epoch_loss = 0.0
    all_targets = []
    all_treatments = []
    all_y0 = []
    all_y1 = []
    all_prop = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            
            chunk_embeddings_list = [
                batch['chunk_embeddings'][i, :, :].contiguous()
                for i in range(batch['chunk_embeddings'].size(0))
            ]
            batch['chunk_embeddings'] = chunk_embeddings_list
            
            losses = model.train_step(
                batch,
                alpha_propensity=config.alpha_propensity,
                beta_targreg=config.beta_targreg
            )
            
            epoch_loss += losses['loss'].item()
            
            all_targets.append(batch['outcome'].detach().cpu())
            all_treatments.append(batch['treatment'].detach().cpu())
            all_y0.append(losses['y0_logit'].detach().cpu())
            all_y1.append(losses['y1_logit'].detach().cpu())
            all_prop.append(losses['t_logit'].detach().cpu())
    
    return _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop)


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


def _generate_predictions(
    model: CausalDragonnetText,
    loader: DataLoader,
    device: torch.device,
    config: AppliedInferenceConfig = None
) -> dict:
    """Generate predictions on test set."""
    all_y0 = []
    all_y1 = []
    all_propensity = []
    all_confounder_features = []  # For diagnostics
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            chunk_embeddings_list = [
                batch['chunk_embeddings'][i, :, :].to(device).contiguous()
                for i in range(batch['chunk_embeddings'].size(0))
            ]
            
            # Extract confounder features for diagnostics
            features = model.feature_extractor(chunk_embeddings_list)
            all_confounder_features.append(features.cpu().numpy())
            
            preds = model.predict(chunk_embeddings_list)

            # Use logit-scale predictions
            all_y0.append(preds['y0_logit'].cpu().numpy())
            all_y1.append(preds['y1_logit'].cpu().numpy())
            all_propensity.append(preds['t_logit'].cpu().numpy())

    y0_logit = np.concatenate(all_y0)
    y1_logit = np.concatenate(all_y1)
    propensity_logit = np.concatenate(all_propensity)
    ite_logit = y1_logit - y0_logit
    
    # Log confounder feature statistics if config provided
    if config is not None:
        confounder_features = np.concatenate(all_confounder_features, axis=0)
        arch = config.architecture
        stats_df = compute_confounder_feature_stats(
            confounder_features,
            projection_dim=arch.dragonnet_representation_dim
        )
        log_confounder_stats(stats_df, prefix="Applied Inference ")

    return {
        'y0_logit': y0_logit,
        'y1_logit': y1_logit,
        'propensity_logit': propensity_logit,
        'ite_logit': ite_logit
    }


def _save_and_summarize(results_df: pd.DataFrame, output_path: Path) -> None:
    """Save results and print summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)
    
    logger.info(f"Predictions saved to: {output_path}")
    logger.info("\nPrediction Summary:")
    logger.info(f"  Samples: {len(results_df)}")
    logger.info(f"  Mean ITE: {results_df['ite_pred'].mean():.4f}")
    logger.info(f"  Std ITE: {results_df['ite_pred'].std():.4f}")
    logger.info(f"  Mean propensity: {results_df['propensity_pred'].mean():.4f}")


def _initialize_latents_kmeans(
    model: CausalDragonnetText,
    dataset: pd.DataFrame,
    cache: Optional[EmbeddingCache],
    device: torch.device,
    max_samples: int = 5000
) -> None:
    """Initialize latent confounders using k-means clustering."""
    try:
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np
        from ..data.preprocessing import process_text
        
        num_latents = model.feature_extractor.num_latent
        if num_latents == 0:
            return
        
        logger.info(f"Running k-means with k={num_latents}...")
        
        sample_size = min(max_samples, len(dataset))
        sample_df = dataset.sample(n=sample_size, random_state=42)
        
        all_embeddings = []
        for text in tqdm(sample_df['clinical_text'], desc="Computing embeddings", leave=False):
            if cache is not None:
                chunks, embeddings = cache.get_or_compute(
                    text,
                    lambda t: process_text(
                        t,
                        model.sentence_transformer_model,
                        device,
                        model.chunk_size,
                        model.chunk_overlap
                    )
                )
            else:
                chunks, embeddings = process_text(
                    text,
                    model.sentence_transformer_model,
                    device,
                    model.chunk_size,
                    model.chunk_overlap
                )
            
            if embeddings.size(0) > 0:
                all_embeddings.append(embeddings.cpu().numpy())
        
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
            
            kmeans = MiniBatchKMeans(
                n_clusters=num_latents,
                random_state=42,
                batch_size=1000,
                n_init=3
            )
            kmeans.fit(all_embeddings)
            
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            model.feature_extractor.latent_confounders.data.copy_(centers)
            
            logger.info("✓ Latent confounders initialized with k-means")
        
    except ImportError:
        logger.warning("scikit-learn not available, skipping k-means initialization")
    except Exception as e:
        logger.warning(f"K-means initialization failed: {e}")


# =============================================================================
# ModernBERT-specific training and evaluation functions
# =============================================================================

def _train_epoch_modernbert(
    model: CausalModernBertText,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    config
) -> dict:
    """Train for one epoch with ModernBERT model."""
    epoch_loss = 0.0
    all_targets = []
    all_treatments = []
    all_y0 = []
    all_y1 = []
    all_prop = []
    
    for batch in tqdm(loader, desc="Training", leave=False):
        # Move tensors to device
        batch['outcome'] = batch['outcome'].to(device)
        batch['treatment'] = batch['treatment'].to(device)
        # 'texts' stays as list of strings
        
        optimizer.zero_grad()
        
        losses = model.train_step(
            batch,
            alpha_propensity=config.alpha_propensity,
            beta_targreg=config.beta_targreg
        )
        
        losses['loss'].backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        epoch_loss += losses['loss'].item()
        
        # Collect for metrics
        all_targets.append(batch['outcome'].detach().cpu())
        all_treatments.append(batch['treatment'].detach().cpu())
        all_y0.append(losses['y0_logit'].detach().cpu())
        all_y1.append(losses['y1_logit'].detach().cpu())
        all_prop.append(losses['t_logit'].detach().cpu())
    
    return _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop)


def _eval_epoch_modernbert(
    model: CausalModernBertText,
    loader: DataLoader,
    device: torch.device,
    config
) -> dict:
    """Evaluate for one epoch with ModernBERT model."""
    epoch_loss = 0.0
    all_targets = []
    all_treatments = []
    all_y0 = []
    all_y1 = []
    all_prop = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            # Move tensors to device
            batch['outcome'] = batch['outcome'].to(device)
            batch['treatment'] = batch['treatment'].to(device)
            # 'texts' stays as list of strings
            
            losses = model.train_step(
                batch,
                alpha_propensity=config.alpha_propensity,
                beta_targreg=config.beta_targreg
            )
            
            epoch_loss += losses['loss'].item()
            
            all_targets.append(batch['outcome'].detach().cpu())
            all_treatments.append(batch['treatment'].detach().cpu())
            all_y0.append(losses['y0_logit'].detach().cpu())
            all_y1.append(losses['y1_logit'].detach().cpu())
            all_prop.append(losses['t_logit'].detach().cpu())
    
    return _compute_epoch_metrics(epoch_loss, loader, all_targets, all_treatments, all_y0, all_y1, all_prop)


def _generate_predictions_modernbert(
    model: CausalModernBertText,
    loader: DataLoader,
    device: torch.device,
    config: AppliedInferenceConfig = None
) -> dict:
    """Generate predictions on test set with ModernBERT model."""
    all_y0 = []
    all_y1 = []
    all_propensity = []
    all_features = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            texts = batch['texts']
            
            # Extract features for diagnostics
            features = model.get_features(texts)
            all_features.append(features.cpu().numpy())
            
            preds = model.predict(texts)
            
            # Use logit-scale predictions
            all_y0.append(preds['y0_logit'].cpu().numpy())
            all_y1.append(preds['y1_logit'].cpu().numpy())
            all_propensity.append(preds['t_logit'].cpu().numpy())
    
    y0_logit = np.concatenate(all_y0)
    y1_logit = np.concatenate(all_y1)
    propensity_logit = np.concatenate(all_propensity)
    ite_logit = y1_logit - y0_logit
    
    # Log feature statistics if config provided
    if config is not None:
        feature_array = np.concatenate(all_features, axis=0)
        arch = config.architecture
        stats_df = compute_confounder_feature_stats(
            feature_array,
            projection_dim=arch.dragonnet_representation_dim
        )
        log_confounder_stats(stats_df, prefix="Applied Inference ModernBERT ")
    
    return {
        'y0_logit': y0_logit,
        'y1_logit': y1_logit,
        'propensity_logit': propensity_logit,
        'ite_logit': ite_logit
    }