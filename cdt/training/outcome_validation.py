# cdt/training/outcome_validation.py
"""Outcome-only model training for validating text→outcome signal."""

import gc
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import Parallel, delayed

from ..config import AppliedInferenceConfig
from ..models.outcome_model import OutcomeOnlyModel, create_outcome_model_from_config
from ..data import ClinicalTextDataset, collate_batch
from ..utils import cuda_cleanup, get_memory_info


logger = logging.getLogger(__name__)


def train_outcome_model_cv(
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    outcome_validation_config,
    device: torch.device,
    num_workers: int = 1,
    gpu_ids: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train outcome-only model using k-fold CV to validate text→outcome signal.

    For each fold:
    1. Train outcome model on training fold
    2. Predict outcomes on held-out fold
    3. Calculate RMSE, MAE, R² metrics

    This validates whether the CNN can learn meaningful text representations
    for outcome prediction before running full DragonNet training.

    Args:
        dataset: DataFrame with clinical text and outcome columns
        config: AppliedInferenceConfig with architecture settings
        outcome_validation_config: OutcomeValidationConfig with training settings
        device: PyTorch device
        num_workers: Number of parallel workers
        gpu_ids: List of GPU IDs for parallel processing

    Returns:
        Tuple of (DataFrame with 'predicted_outcome' column, training_log DataFrame)
    """
    k = outcome_validation_config.cv_folds

    logger.info(f"=" * 80)
    logger.info("OUTCOME SIGNAL VALIDATION")
    logger.info(f"=" * 80)
    logger.info(f"Training outcome-only model with {k}-fold CV on {len(dataset)} samples")
    logger.info("Purpose: Validate whether CNN can learn text→outcome signal")

    # Reset index to ensure KFold works with indices
    dataset = dataset.reset_index(drop=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    splits = list(kf.split(dataset))

    # Initialize predictions array
    predictions = np.zeros(len(dataset))

    # Determine devices to use
    if gpu_ids:
        devices = [torch.device(f"cuda:{i}") for i in gpu_ids]
    else:
        devices = [device]

    if num_workers > 1:
        logger.info(f"Parallelizing outcome CV across {num_workers} workers on devices: {devices}")

        results = Parallel(n_jobs=num_workers)(
            delayed(_process_outcome_fold)(
                fold, train_idx, test_idx, dataset, config, outcome_validation_config,
                devices[fold % len(devices)]
            )
            for fold, (train_idx, test_idx) in enumerate(splits)
        )
    else:
        results = []
        for fold, (train_idx, test_idx) in enumerate(splits):
            results.append(_process_outcome_fold(
                fold, train_idx, test_idx, dataset, config, outcome_validation_config,
                devices[0]
            ))

    # Combine predictions from all folds
    all_history = []
    all_fold_metrics = []
    for test_idx, fold_preds, fold_metrics, fold_history in results:
        predictions[test_idx] = fold_preds
        all_history.extend(fold_history)
        all_fold_metrics.append(fold_metrics)
        logger.info(f"Fold {fold_metrics['fold']}: RMSE={fold_metrics['rmse']:.4f}, "
                   f"R²={fold_metrics['r2']:.4f}, pred_std={fold_metrics['pred_std']:.4f}")

    # Add predictions to dataset
    dataset = dataset.copy()
    dataset['outcome_validation_pred'] = predictions

    # Create training log DataFrame
    training_log_df = pd.DataFrame(all_history)

    # Log overall summary statistics
    true_outcomes = dataset[config.outcome_column].values
    overall_rmse = np.sqrt(mean_squared_error(true_outcomes, predictions))
    overall_mae = mean_absolute_error(true_outcomes, predictions)
    overall_r2 = r2_score(true_outcomes, predictions)
    pred_std = np.std(predictions)
    true_std = np.std(true_outcomes)

    logger.info(f"=" * 80)
    logger.info("OUTCOME VALIDATION SUMMARY")
    logger.info(f"=" * 80)
    logger.info(f"  Overall RMSE: {overall_rmse:.4f}")
    logger.info(f"  Overall MAE: {overall_mae:.4f}")
    logger.info(f"  Overall R²: {overall_r2:.4f}")
    logger.info(f"  Prediction std: {pred_std:.4f}")
    logger.info(f"  True outcome std: {true_std:.4f}")
    logger.info(f"  Std ratio (pred/true): {pred_std/true_std:.4f}")

    # Assess if model collapsed
    if pred_std < 0.1 * true_std:
        logger.warning("⚠️  MODEL COLLAPSED: Prediction std is < 10% of true std")
        logger.warning("    The model is predicting near-constant values")
        logger.warning("    Text→outcome signal may not be learnable with current setup")
    elif pred_std < 0.5 * true_std:
        logger.warning("⚠️  WEAK SIGNAL: Prediction std is < 50% of true std")
        logger.warning("    The model captures some variation but limited")
    else:
        logger.info("✓ Model captures meaningful variation in outcomes")

    if overall_r2 <= 0:
        logger.warning("⚠️  R² ≤ 0: Model is worse than predicting the mean")
    elif overall_r2 < 0.1:
        logger.warning("⚠️  Low R²: Model explains < 10% of outcome variance")
    else:
        logger.info(f"✓ Model explains {100*overall_r2:.1f}% of outcome variance")

    logger.info(f"=" * 80)

    return dataset, training_log_df


def _process_outcome_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    config: AppliedInferenceConfig,
    outcome_validation_config,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Process a single fold for outcome model training.

    Args:
        fold: Fold index
        train_idx: Training indices
        test_idx: Test indices
        dataset: Full dataset
        config: Applied inference config
        outcome_validation_config: Outcome validation config
        device: PyTorch device

    Returns:
        Tuple of (test_idx, predictions, fold_metrics, training_history)
    """
    # Re-configure logger for worker process
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.info(f"Outcome FOLD {fold + 1} starting on {device}")

    arch_config = config.architecture

    # Prepare data for this fold
    train_df = dataset.iloc[train_idx]
    test_df = dataset.iloc[test_idx]

    # Train outcome model
    model, fold_history = _train_outcome_model(
        train_df, test_df, config, outcome_validation_config, arch_config, device
    )

    # Add fold number to each history entry
    for entry in fold_history:
        entry['fold'] = fold + 1

    # Predict on held-out fold
    predictions = _predict_outcomes(model, test_df, config, outcome_validation_config, device)

    # Calculate fold metrics
    true_outcomes = test_df[config.outcome_column].values
    fold_metrics = {
        'fold': fold + 1,
        'rmse': np.sqrt(mean_squared_error(true_outcomes, predictions)),
        'mae': mean_absolute_error(true_outcomes, predictions),
        'r2': r2_score(true_outcomes, predictions),
        'pred_mean': np.mean(predictions),
        'pred_std': np.std(predictions),
        'true_mean': np.mean(true_outcomes),
        'true_std': np.std(true_outcomes)
    }

    # Cleanup
    model.cpu()
    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()
    cuda_cleanup()

    logger.info(f"Outcome FOLD {fold + 1} complete | {get_memory_info()}")

    return test_idx, predictions, fold_metrics, fold_history


def _train_outcome_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: AppliedInferenceConfig,
    outcome_validation_config,
    arch_config,
    device: torch.device
) -> Tuple[OutcomeOnlyModel, List[Dict[str, Any]]]:
    """
    Train an outcome-only model.

    Args:
        train_df: Training data
        val_df: Validation data
        config: Applied inference config
        outcome_validation_config: Outcome validation config
        arch_config: Model architecture config
        device: PyTorch device

    Returns:
        Tuple of (trained OutcomeOnlyModel, training_history)
    """
    # Get feature extractor type (default to "cnn" for backward compatibility)
    feature_extractor_type = getattr(arch_config, 'feature_extractor_type', 'cnn')

    # Create outcome model
    model = create_outcome_model_from_config(
        arch_config=arch_config,
        representation_dim=arch_config.dragonnet_representation_dim,
        hidden_dim=arch_config.dragonnet_hidden_outcome_dim,
        device=device
    )

    train_texts = train_df[config.text_column].tolist()

    if feature_extractor_type == "cnn":
        # CNN-specific initialization
        model.fit_tokenizer(train_texts)
        logger.info(f"Fitted word tokenizer on {len(train_texts)} training texts")

        # Initialize embeddings from BERT if configured
        use_random_init = getattr(arch_config, 'cnn_use_random_embedding_init', False)
        if not use_random_init and getattr(arch_config, 'cnn_init_embeddings_from', None):
            model.feature_extractor.init_embeddings_from_bert(
                arch_config.cnn_init_embeddings_from,
                freeze=getattr(arch_config, 'cnn_freeze_embeddings', False)
            )

        # Initialize filters from explicit concepts and/or k-means
        if arch_config.cnn_explicit_filter_concepts or arch_config.cnn_num_kmeans_filters > 0:
            model.feature_extractor.init_filters(
                texts=train_texts,
                freeze=arch_config.cnn_freeze_filters
            )
    else:
        logger.info(f"Using BERT feature extractor: {arch_config.bert_model_name}")

    # Create datasets
    train_dataset = ClinicalTextDataset(
        data=train_df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column
    )

    val_dataset = ClinicalTextDataset(
        data=val_df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=outcome_validation_config.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=outcome_validation_config.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=outcome_validation_config.learning_rate,
        weight_decay=1e-4
    )

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    history = []

    for epoch in range(outcome_validation_config.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            batch['outcome'] = batch['outcome'].to(device)

            optimizer.zero_grad()
            losses = model.train_step(batch)
            losses['loss'].backward()
            optimizer.step()

            train_loss += losses['loss'].item()
            train_preds.append(losses['y_pred'].squeeze().cpu().numpy())
            train_targets.append(batch['outcome'].cpu().numpy())

        train_loss = train_loss / len(train_loader)

        # Flatten predictions and targets
        train_preds_flat = np.concatenate(train_preds)
        train_targets_flat = np.concatenate(train_targets)
        train_rmse = np.sqrt(mean_squared_error(train_targets_flat, train_preds_flat))
        train_r2 = r2_score(train_targets_flat, train_preds_flat)
        train_pred_std = np.std(train_preds_flat)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch['outcome'] = batch['outcome'].to(device)
                losses = model.train_step(batch)
                val_loss += losses['loss'].item()
                val_preds.append(losses['y_pred'].squeeze().cpu().numpy())
                val_targets.append(batch['outcome'].cpu().numpy())

        val_loss = val_loss / len(val_loader)

        # Flatten predictions and targets
        val_preds_flat = np.concatenate(val_preds)
        val_targets_flat = np.concatenate(val_targets)
        val_rmse = np.sqrt(mean_squared_error(val_targets_flat, val_preds_flat))
        val_r2 = r2_score(val_targets_flat, val_preds_flat)
        val_pred_std = np.std(val_preds_flat)

        # Record history
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse),
            'train_r2': float(train_r2),
            'val_r2': float(val_r2),
            'train_pred_std': float(train_pred_std),
            'val_pred_std': float(val_pred_std)
        })

        # Log epoch metrics
        logger.info(f"  Epoch {epoch+1}/{outcome_validation_config.epochs}: "
                   f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                   f"val_rmse={val_rmse:.4f}, val_r2={val_r2:.4f}, "
                   f"val_pred_std={val_pred_std:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Cleanup
    del train_loader, val_loader, train_dataset, val_dataset, optimizer
    gc.collect()

    return model, history


def _predict_outcomes(
    model: OutcomeOnlyModel,
    df: pd.DataFrame,
    config: AppliedInferenceConfig,
    outcome_validation_config,
    device: torch.device
) -> np.ndarray:
    """
    Predict outcomes for a dataset.

    Args:
        model: Trained outcome model
        df: DataFrame with texts
        config: Configuration
        outcome_validation_config: Outcome validation config
        device: PyTorch device

    Returns:
        Array of outcome predictions
    """
    dataset = ClinicalTextDataset(
        data=df,
        text_column=config.text_column,
        outcome_column=config.outcome_column,
        treatment_column=config.treatment_column
    )

    loader = DataLoader(
        dataset,
        batch_size=outcome_validation_config.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            texts = batch['texts']
            predictions = model.predict(texts)
            all_predictions.append(predictions.cpu().numpy())

    outcome_predictions = np.concatenate(all_predictions)

    del loader, dataset
    gc.collect()

    return outcome_predictions
