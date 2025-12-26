# cdt/training/pretraining.py
"""Multi-treatment pretraining for confounder learning."""

import logging
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from ..config import PretrainingConfig
from ..models.multitreatment import MultiTreatmentDragonnetText
from ..data import ClinicalTextDataset, collate_batch, EmbeddingCache
from ..utils import cuda_cleanup, get_memory_info


logger = logging.getLogger(__name__)


def run_pretraining(
    dataset: pd.DataFrame,
    config: PretrainingConfig,
    output_path: Path,
    device: torch.device,
    cache: Optional[EmbeddingCache] = None
) -> None:
    """
    Run multi-treatment pretraining.
    
    Args:
        dataset: DataFrame with clinical text and treatment labels
        config: Pretraining configuration
        output_path: Path to save pretrained weights
        device: PyTorch device
        cache: Optional embedding cache
    """
    logger.info("="*80)
    logger.info("MULTI-TREATMENT PRETRAINING")
    logger.info("="*80)
    
    # Get unique treatments
    unique_treatments = sorted(dataset[config.treatment_column].unique())
    num_treatments = len(unique_treatments)
    
    logger.info(f"Dataset: {len(dataset)} samples")
    logger.info(f"Treatments: {num_treatments} unique values")
    logger.info(f"  {unique_treatments}")
    
    # Create treatment mapping (original values -> indices)
    treatment_to_idx = {t: i for i, t in enumerate(unique_treatments)}
    dataset_mapped = dataset.copy()
    dataset_mapped[config.treatment_column] = dataset_mapped[config.treatment_column].map(treatment_to_idx)
    
    # Create model
    arch_config = config.architecture
    model = MultiTreatmentDragonnetText(
        num_treatments=num_treatments,
        sentence_transformer_model_name=arch_config.embedding_model_name,
        num_latent_confounders=arch_config.num_latent_confounders,
        features_per_confounder=arch_config.features_per_confounder,
        explicit_confounder_texts=arch_config.explicit_confounder_texts,
        aggregator_mode=arch_config.aggregator_mode,
        dragonnet_representation_dim=arch_config.dragonnet_representation_dim,
        dragonnet_hidden_outcome_dim=arch_config.dragonnet_hidden_outcome_dim,
        chunk_size=arch_config.chunk_size,
        chunk_overlap=arch_config.chunk_overlap,
        device=str(device)
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize latent confounders with k-means if requested
    if config.training.init_latents_from_kmeans:
        logger.info("Initializing latent confounders with k-means...")
        _initialize_latents_kmeans(model, dataset_mapped, cache, device)
    
    # Create dataset and dataloader
    train_dataset = ClinicalTextDataset(
        data=dataset_mapped,
        text_column="clinical_text",
        outcome_column="outcome_indicator",
        treatment_column=config.treatment_column,
        model=model.sentence_transformer_model,
        device=device,
        chunk_size=arch_config.chunk_size,
        chunk_overlap=arch_config.chunk_overlap,
        cache=cache
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=False
    )
    
    # Setup optimizer
    train_config = config.training
    if train_config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate
        )
    
    # Setup learning rate scheduler
    if train_config.lr_schedule == "linear":
        total_steps = len(train_loader) * train_config.epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
    else:
        scheduler = None
    
    # Training loop
    logger.info(f"Starting training for {train_config.epochs} epochs")
    logger.info(f"  Batch size: {train_config.batch_size}")
    logger.info(f"  Learning rate: {train_config.learning_rate}")
    logger.info(f"  Alpha (propensity): {train_config.alpha_propensity}")
    logger.info(f"  Beta (targreg): {train_config.beta_targreg}")
    
    model.train()
    
    for epoch in range(train_config.epochs):
        epoch_losses = {
            'total': 0.0,
            'outcome': 0.0,
            'propensity': 0.0,
            'targreg': 0.0
        }
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{train_config.epochs}",
            leave=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            
            # Convert chunk embeddings list properly
            chunk_embeddings_list = [
                batch['chunk_embeddings'][i, :, :].contiguous()
                for i in range(batch['chunk_embeddings'].size(0))
            ]
            batch['chunk_embeddings'] = chunk_embeddings_list
            
            # Forward and backward
            optimizer.zero_grad()
            
            losses = model.train_step(
                batch,
                alpha_propensity=train_config.alpha_propensity,
                beta_targreg=train_config.beta_targreg
            )
            
            losses['loss'].backward()
            
            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Accumulate losses
            epoch_losses['total'] += losses['loss'].item()
            epoch_losses['outcome'] += losses['outcome_loss'].item()
            epoch_losses['propensity'] += losses['propensity_loss'].item()
            epoch_losses['targreg'] += (
                losses['targreg_loss'].item()
                if torch.is_tensor(losses['targreg_loss'])
                else losses['targreg_loss']
            )
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{losses['loss'].item():.4f}",
                    'outcome': f"{losses['outcome_loss'].item():.4f}",
                    'prop': f"{losses['propensity_loss'].item():.4f}",
                })
        
        # Epoch summary
        n_batches = len(train_loader)
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        logger.info(
            f"Epoch {epoch+1}/{train_config.epochs} - "
            f"Loss: {avg_losses['total']:.4f} "
            f"(outcome: {avg_losses['outcome']:.4f}, "
            f"prop: {avg_losses['propensity']:.4f}, "
            f"targreg: {avg_losses['targreg']:.4f}) "
            f"| {get_memory_info()}"
        )
        
        cuda_cleanup()
    
    # Save checkpoint
    logger.info("Training complete, saving checkpoint...")
    
    model.save_checkpoint(
        path=str(output_path),
        optimizer=optimizer,
        epoch=train_config.epochs,
        metrics={
            'final_loss': avg_losses['total'],
            'num_treatments': num_treatments,
            'treatment_mapping': treatment_to_idx
        }
    )
    
    logger.info(f"Pretrained model saved to: {output_path}")


def _initialize_latents_kmeans(
    model: MultiTreatmentDragonnetText,
    dataset: pd.DataFrame,
    cache: Optional[EmbeddingCache],
    device: torch.device,
    max_samples: int = 5000
) -> None:
    """
    Initialize latent confounders using k-means clustering on embeddings.
    
    Args:
        model: Model to initialize
        dataset: Training dataset
        cache: Embedding cache
        device: Device
        max_samples: Maximum samples for k-means
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np
        
        from ..data.preprocessing import process_text
        
        num_latents = model.feature_extractor.num_latent
        if num_latents == 0:
            return
        
        logger.info(f"Running k-means with k={num_latents} on sample embeddings...")
        
        # Sample texts
        sample_size = min(max_samples, len(dataset))
        sample_df = dataset.sample(n=sample_size, random_state=42)
        
        # Get embeddings
        all_embeddings = []
        for text in tqdm(sample_df['clinical_text'], desc="Computing embeddings"):
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
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        logger.info(f"Collected {len(all_embeddings)} embeddings")
        
        # Run k-means
        kmeans = MiniBatchKMeans(
            n_clusters=num_latents,
            random_state=42,
            batch_size=1000,
            n_init=3
        )
        kmeans.fit(all_embeddings)
        
        # Set cluster centers as initial latent confounders
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        model.feature_extractor.latent_confounders.data.copy_(centers)
        
        logger.info("✓ Latent confounders initialized with k-means centers")
        
    except ImportError:
        logger.warning("scikit-learn not available, skipping k-means initialization")
    except Exception as e:
        logger.warning(f"K-means initialization failed: {e}")
