#!/usr/bin/env python
"""
Run applied inference using the token-level causal model.

This script uses a frozen transformer backbone with interpretable
confounder-aligned projection for causal inference from clinical text.

Usage:
    python examples/run_token_level_inference.py --config examples/token_level_config.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from cdt.models import CausalDragonnetTokenLevel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TokenLevelDataset(Dataset):
    """Dataset that returns tokenized text for the token-level model."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        text_column: str = "clinical_text",
        outcome_column: str = "outcome_indicator",
        treatment_column: str = "treatment_indicator",
        max_length: int = 10000,
    ):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.outcome_column = outcome_column
        self.treatment_column = treatment_column
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row[self.text_column])

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'outcome': torch.tensor(row[self.outcome_column], dtype=torch.float32),
            'treatment': torch.tensor(row[self.treatment_column], dtype=torch.float32),
            'idx': idx,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch of samples."""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'outcome': torch.stack([b['outcome'] for b in batch]),
        'treatment': torch.stack([b['treatment'] for b in batch]),
        'idx': torch.tensor([b['idx'] for b in batch]),
    }


def train_epoch(
    model: CausalDragonnetTokenLevel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_outcomes = []
    all_treatments = []
    all_y0 = []
    all_y1 = []
    all_prop = []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        optimizer.zero_grad()

        result = model.train_step(
            batch,
            alpha_propensity=config['training']['alpha_propensity'],
            beta_targreg=config['training']['beta_targreg'],
            outcome_type=config.get('outcome_type', 'binary'),
        )

        result['loss'].backward()
        optimizer.step()

        total_loss += result['loss'].item()

        all_outcomes.append(batch['outcome'].cpu())
        all_treatments.append(batch['treatment'].cpu())
        all_y0.append(result['y0_logit'].cpu())
        all_y1.append(result['y1_logit'].cpu())
        all_prop.append(result['t_logit'].cpu())

    return compute_metrics(total_loss, len(loader), all_outcomes, all_treatments, all_y0, all_y1, all_prop)


def eval_epoch(
    model: CausalDragonnetTokenLevel,
    loader: DataLoader,
    device: torch.device,
    config: Dict,
) -> Dict[str, float]:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_outcomes = []
    all_treatments = []
    all_y0 = []
    all_y1 = []
    all_prop = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            result = model.train_step(
                batch,
                alpha_propensity=config['training']['alpha_propensity'],
                beta_targreg=config['training']['beta_targreg'],
                outcome_type=config.get('outcome_type', 'binary'),
            )

            total_loss += result['loss'].item()

            all_outcomes.append(batch['outcome'].cpu())
            all_treatments.append(batch['treatment'].cpu())
            all_y0.append(result['y0_logit'].cpu())
            all_y1.append(result['y1_logit'].cpu())
            all_prop.append(result['t_logit'].cpu())

    return compute_metrics(total_loss, len(loader), all_outcomes, all_treatments, all_y0, all_y1, all_prop)


def compute_metrics(
    total_loss: float,
    num_batches: int,
    all_outcomes: List[torch.Tensor],
    all_treatments: List[torch.Tensor],
    all_y0: List[torch.Tensor],
    all_y1: List[torch.Tensor],
    all_prop: List[torch.Tensor],
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    outcomes = torch.cat(all_outcomes).numpy()
    treatments = torch.cat(all_treatments).numpy()
    y0_logits = torch.cat(all_y0).squeeze().numpy()
    y1_logits = torch.cat(all_y1).squeeze().numpy()
    prop_logits = torch.cat(all_prop).squeeze().numpy()

    def safe_auroc(y_true, scores):
        try:
            if len(np.unique(y_true)) < 2:
                return None
            return roc_auc_score(y_true, scores)
        except:
            return None

    # Propensity AUROC
    auroc_prop = safe_auroc(treatments, torch.sigmoid(torch.tensor(prop_logits)).numpy())

    # Outcome AUROC (factual only)
    mask_t0 = treatments == 0
    mask_t1 = treatments == 1

    auroc_y0 = safe_auroc(outcomes[mask_t0], y0_logits[mask_t0]) if mask_t0.any() else None
    auroc_y1 = safe_auroc(outcomes[mask_t1], y1_logits[mask_t1]) if mask_t1.any() else None

    return {
        'loss': total_loss / num_batches,
        'auroc_y0': auroc_y0,
        'auroc_y1': auroc_y1,
        'auroc_prop': auroc_prop,
    }


def predict(
    model: CausalDragonnetTokenLevel,
    loader: DataLoader,
    device: torch.device,
    outcome_type: str = "binary",
) -> Dict[str, np.ndarray]:
    """Generate predictions."""
    model.eval()
    all_y0 = []
    all_y1 = []
    all_prop = []
    all_idx = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            preds = model.predict(input_ids, attention_mask, outcome_type=outcome_type)

            if outcome_type == "binary":
                all_y0.append(preds['y0_prob'].cpu().numpy())
                all_y1.append(preds['y1_prob'].cpu().numpy())
            else:
                all_y0.append(preds['y0_pred'].cpu().numpy())
                all_y1.append(preds['y1_pred'].cpu().numpy())

            all_prop.append(preds['propensity'].cpu().numpy())
            all_idx.append(batch['idx'].numpy())

    return {
        'y0_pred': np.concatenate(all_y0),
        'y1_pred': np.concatenate(all_y1),
        'propensity': np.concatenate(all_prop),
        'ite_pred': np.concatenate(all_y1) - np.concatenate(all_y0),
        'idx': np.concatenate(all_idx),
    }


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict,
    device: torch.device,
) -> Tuple[CausalDragonnetTokenLevel, List[Dict]]:
    """Train a single model instance."""

    # Create model
    model_config = config['model']
    max_length = model_config.get('max_length', 10000)
    model = CausalDragonnetTokenLevel(
        model_name=model_config['model_name'],
        explicit_confounder_texts=model_config.get('explicit_confounder_texts'),
        num_latent_confounders=model_config.get('num_latent_confounders', 10),
        aggregation_method=model_config.get('aggregation_method', 'attention'),
        topk=model_config.get('topk', 5),
        anchor_strength=model_config.get('anchor_strength', 0.1),
        dragonnet_representation_dim=model_config.get('dragonnet_representation_dim', 128),
        dragonnet_hidden_outcome_dim=model_config.get('dragonnet_hidden_outcome_dim', 64),
        max_length=max_length,
        model_type=model_config.get('model_type', 'dragonnet'),
        device=str(device),
    )

    # Create datasets
    data_config = config['data']
    train_dataset = TokenLevelDataset(
        train_df,
        model.tokenizer,
        text_column=data_config['text_column'],
        outcome_column=data_config['outcome_column'],
        treatment_column=data_config['treatment_column'],
        max_length=max_length,
    )

    val_dataset = TokenLevelDataset(
        val_df,
        model.tokenizer,
        text_column=data_config['text_column'],
        outcome_column=data_config['outcome_column'],
        treatment_column=data_config['treatment_column'],
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=1e-4,
    )

    # Training loop
    best_val_loss = float('inf')
    best_state = None
    history = []

    for epoch in range(config['training']['epochs']):
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        val_metrics = eval_epoch(model, val_loader, device, config)

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'train_auroc_y0': train_metrics['auroc_y0'],
            'train_auroc_y1': train_metrics['auroc_y1'],
            'train_auroc_prop': train_metrics['auroc_prop'],
            'val_auroc_y0': val_metrics['auroc_y0'],
            'val_auroc_y1': val_metrics['auroc_y1'],
            'val_auroc_prop': val_metrics['auroc_prop'],
        })

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        auroc_str = f"{val_metrics['auroc_prop']:.4f}" if val_metrics['auroc_prop'] else 'N/A'
        logger.info(
            f"Epoch {epoch+1}/{config['training']['epochs']} - "
            f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
            f"Val AUROC Prop: {auroc_str}"
        )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def train_single_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dataset: pd.DataFrame,
    config: Dict,
    device_str: str,
    output_dir: Path,
    k: int,
) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """
    Train a single fold. Designed to be called in parallel.

    Returns:
        Tuple of (predictions_df, history_list, drift_dict)
    """
    # Set up logging for this process
    logger = logging.getLogger(f"fold_{fold+1}")
    logger.info(f"FOLD {fold+1}/{k} starting on {device_str}")

    device = torch.device(device_str)

    train_df = dataset.iloc[train_idx]
    val_df = dataset.iloc[test_idx]  # Use held-out fold for validation

    # Train model
    model, history = train_model(train_df, val_df, config, device)

    # Add fold to history
    for h in history:
        h['fold'] = fold + 1

    # Predict on held-out fold (same as validation)
    data_config = config['data']
    model_config = config['model']
    max_length = model_config.get('max_length', 10000)
    test_dataset = TokenLevelDataset(
        val_df,
        model.tokenizer,
        text_column=data_config['text_column'],
        outcome_column=data_config['outcome_column'],
        treatment_column=data_config['treatment_column'],
        max_length=max_length,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    preds = predict(model, test_loader, device, config.get('outcome_type', 'binary'))

    # Store predictions
    fold_preds = val_df.copy()
    fold_preds['y0_pred'] = preds['y0_pred']
    fold_preds['y1_pred'] = preds['y1_pred']
    fold_preds['ite_pred'] = preds['ite_pred']
    fold_preds['propensity_pred'] = preds['propensity']
    fold_preds['cv_fold'] = fold + 1

    # Get confounder drift
    drift = model.get_confounder_drift()

    # Cleanup
    del model
    torch.cuda.empty_cache()

    logger.info(f"FOLD {fold+1}/{k} completed on {device_str}")

    return fold_preds, history, drift


def _fold_worker(args):
    """Worker function for multiprocessing pool."""
    fold, train_idx, test_idx, dataset, config, device_str, output_dir, k = args
    try:
        return train_single_fold(
            fold, train_idx, test_idx, dataset, config, device_str, output_dir, k
        )
    except Exception as e:
        logger.error(f"Fold {fold+1} failed: {e}")
        raise


def run_cv_inference(
    dataset: pd.DataFrame,
    config: Dict,
    output_dir: Path,
    devices: List[str],
) -> pd.DataFrame:
    """
    Run K-Fold cross-validation with optional parallelization.

    Args:
        dataset: Input DataFrame
        config: Configuration dictionary
        output_dir: Output directory
        devices: List of device strings (e.g., ['cuda:0', 'cuda:1'])
    """
    k = config['data']['cv_folds']
    num_workers = config.get('num_workers', 1)
    logger.info(f"Starting {k}-Fold Cross-Validation on {len(dataset)} samples")
    logger.info(f"Using {num_workers} worker(s) across devices: {devices}")

    dataset = dataset.reset_index(drop=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=config.get('seed', 42))

    # Prepare fold assignments
    fold_splits = list(kf.split(dataset))

    all_predictions = []
    all_history = []

    if num_workers <= 1:
        # Sequential execution (original behavior)
        device = torch.device(devices[0])
        for fold, (train_idx, test_idx) in enumerate(fold_splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"FOLD {fold+1}/{k}")
            logger.info(f"{'='*60}")

            fold_preds, history, drift = train_single_fold(
                fold, train_idx, test_idx, dataset, config, devices[0], output_dir, k
            )

            all_history.extend(history)
            all_predictions.append(fold_preds)

            # Log confounder drift
            if 'explicit_drift' in drift:
                logger.info("Confounder drift from initialization:")
                for name, d in zip(drift['explicit_names'][:5], drift['explicit_drift'][:5].tolist()):
                    logger.info(f"  {name[:50]}: {d:.4f}")
    else:
        # Parallel execution across devices
        logger.info(f"Running {k} folds in parallel with {num_workers} workers")

        # Assign devices to folds in round-robin fashion
        fold_args = []
        for fold, (train_idx, test_idx) in enumerate(fold_splits):
            device_str = devices[fold % len(devices)]
            fold_args.append((
                fold, train_idx, test_idx, dataset, config, device_str, output_dir, k
            ))

        # Use spawn context for CUDA compatibility
        ctx = mp.get_context('spawn')

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            # Submit all folds
            futures = {
                executor.submit(_fold_worker, args): args[0]
                for args in fold_args
            }

            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=k, desc="CV Folds"):
                fold = futures[future]
                try:
                    fold_preds, history, drift = future.result()
                    all_predictions.append(fold_preds)
                    all_history.extend(history)

                    # Log confounder drift
                    if 'explicit_drift' in drift:
                        logger.info(f"Fold {fold+1} confounder drift:")
                        for name, d in zip(drift['explicit_names'][:3], drift['explicit_drift'][:3].tolist()):
                            logger.info(f"  {name[:40]}: {d:.4f}")

                except Exception as e:
                    logger.error(f"Fold {fold+1} failed with error: {e}")
                    raise

    # Combine predictions
    results_df = pd.concat(all_predictions).sort_index()

    # Save history
    history_df = pd.DataFrame(all_history)
    history_df.to_csv(output_dir / "training_log.csv", index=False)

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run token-level causal inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Setup output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Setup devices - support single device string or list of devices
    device_config = config.get('devices', config.get('device', 'cuda:0'))
    if isinstance(device_config, str):
        devices = [device_config]
    else:
        devices = device_config

    num_workers = config.get('num_workers', 1)
    logger.info(f"Using devices: {devices}")
    logger.info(f"Parallel workers: {num_workers}")

    # Load dataset
    data_path = Path(config['data']['dataset_path'])
    logger.info(f"Loading dataset from: {data_path}")
    dataset = pd.read_parquet(data_path)
    logger.info(f"Dataset shape: {dataset.shape}")

    # Run cross-validation
    results_df = run_cv_inference(dataset, config, output_dir, devices)

    # Save predictions
    output_path = output_dir / "applied_predictions.parquet"
    results_df.to_parquet(output_path, index=False)
    logger.info(f"\nPredictions saved to: {output_path}")

    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total samples: {len(results_df)}")
    logger.info(f"Mean predicted ITE: {results_df['ite_pred'].mean():.4f}")
    logger.info(f"Std predicted ITE: {results_df['ite_pred'].std():.4f}")

    if 'true_ite' in results_df.columns:
        from scipy.stats import pearsonr
        r, p = pearsonr(results_df['true_ite'], results_df['ite_pred'])
        logger.info(f"ITE Correlation with ground truth: r={r:.4f}, p={p:.4e}")

        mse = ((results_df['true_ite'] - results_df['ite_pred']) ** 2).mean()
        logger.info(f"ITE MSE: {mse:.4f}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
