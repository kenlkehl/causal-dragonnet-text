# cdt/data/dataset.py
"""Dataset classes for CDT."""

import logging
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sentence_transformers import SentenceTransformer

from .preprocessing import process_text
from .cache import EmbeddingCache


logger = logging.getLogger(__name__)


class ClinicalTextDataset(Dataset):
    """Dataset for clinical text with outcomes and treatments."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str,
        outcome_column: str,
        treatment_column: str,
        model: SentenceTransformer,
        device: torch.device,
        chunk_size: int = 128,
        chunk_overlap: int = 32,
        cache: Optional[EmbeddingCache] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data: DataFrame with text, outcomes, and treatments
            text_column: Name of text column
            outcome_column: Name of outcome column
            treatment_column: Name of treatment column
            model: SentenceTransformer for embeddings
            device: PyTorch device
            chunk_size: Words per chunk
            chunk_overlap: Overlapping words
            cache: Optional embedding cache
        """
        self.data = data.reset_index(drop=True)
        self.text_column = text_column
        self.outcome_column = outcome_column
        self.treatment_column = treatment_column
        self.model = model
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache = cache
        
        self.outcomes = torch.tensor(
            data[outcome_column].values, 
            dtype=torch.float32
        )
        self.treatments = torch.tensor(
            data[treatment_column].values,
            dtype=torch.float32
        )
        
        logger.info(f"Dataset created: {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.data.loc[idx, self.text_column]
        
        if self.cache is not None:
            chunks, embeddings = self.cache.get_or_compute(
                text,
                lambda t: process_text(
                    t, self.model, self.device,
                    self.chunk_size, self.chunk_overlap
                )
            )
        else:
            chunks, embeddings = process_text(
                text, self.model, self.device,
                self.chunk_size, self.chunk_overlap
            )
        
        return {
            'chunk_embeddings': embeddings,
            'outcome': self.outcomes[idx],
            'treatment': self.treatments[idx],
            'text_id': idx
        }


class ModernBertClinicalTextDataset(Dataset):
    """Dataset that returns raw text for ModernBERT processing.
    
    Unlike ClinicalTextDataset which pre-computes embeddings, this dataset
    returns raw text strings that are tokenized by the model during forward pass.
    This is more memory-efficient and allows end-to-end fine-tuning.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str,
        outcome_column: str,
        treatment_column: str
    ):
        """
        Initialize dataset.
        
        Args:
            data: DataFrame with text, outcomes, and treatments
            text_column: Name of text column
            outcome_column: Name of outcome column
            treatment_column: Name of treatment column
        """
        self.data = data.reset_index(drop=True)
        self.text_column = text_column
        
        self.texts = data[text_column].tolist()
        self.outcomes = torch.tensor(
            data[outcome_column].values,
            dtype=torch.float32
        )
        self.treatments = torch.tensor(
            data[treatment_column].values,
            dtype=torch.float32
        )
        
        logger.info(f"ModernBertClinicalTextDataset created: {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'text': self.texts[idx],
            'outcome': self.outcomes[idx],
            'treatment': self.treatments[idx],
            'text_id': idx
        }


def collate_modernbert_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch for ModernBERT dataset.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched data with texts as list of strings
    """
    texts = [item['text'] for item in batch]
    outcomes = torch.stack([item['outcome'] for item in batch])
    treatments = torch.stack([item['treatment'] for item in batch])
    text_ids = [item['text_id'] for item in batch]
    
    return {
        'texts': texts,
        'outcome': outcomes,
        'treatment': treatments,
        'text_id': text_ids
    }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate batch with variable-length sequences.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched data with padding
    """
    chunk_embeddings = [item['chunk_embeddings'] for item in batch]
    outcomes = torch.stack([item['outcome'] for item in batch])
    treatments = torch.stack([item['treatment'] for item in batch])
    text_ids = [item['text_id'] for item in batch]
    
    seq_lengths = torch.tensor([len(emb) for emb in chunk_embeddings])
    
    if seq_lengths.sum() > 0:
        padded = pad_sequence(chunk_embeddings, batch_first=True, padding_value=0.0)
        mask = (
            torch.arange(padded.size(1))[None, :] >= seq_lengths[:, None]
        ).unsqueeze(1)
    else:
        batch_size = len(batch)
        # Pass embedding_dim as parameter or get from model config
        embedding_dim = chunk_embeddings[0].size(-1) if chunk_embeddings else None
        if embedding_dim is None:
            raise ValueError("Cannot determine embedding dimension from empty batch")
        padded = torch.zeros(batch_size, 1, embedding_dim)
        mask = torch.ones(batch_size, 1, 1, dtype=torch.bool)
    
    return {
        'chunk_embeddings': padded,
        'mask': mask,
        'outcome': outcomes,
        'treatment': treatments,
        'text_id': text_ids
    }


def load_dataset(
    path: str,
    split: Optional[str] = None,
    split_column: str = 'split'
) -> pd.DataFrame:
    """
    Load dataset from file.
    
    Args:
        path: Path to dataset file (.csv or .parquet)
        split: Optional split to filter (e.g., 'train', 'val', 'test')
        split_column: Name of split column
    
    Returns:
        DataFrame
    """
    logger.info(f"Loading dataset from {path}")
    
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    if split is not None:
        if split_column not in df.columns:
            raise ValueError(f"Split column '{split_column}' not found")
        df = df[df[split_column] == split].copy()
        logger.info(f"Filtered to {split} split: {len(df)} samples")
    
    return df


def validate_dataset(
    df: pd.DataFrame,
    text_column: str,
    outcome_column: str,
    treatment_column: str,
    split_column: Optional[str] = None
) -> None:
    """
    Validate dataset has required columns and correct format.
    
    Args:
        df: DataFrame to validate
        text_column: Expected text column name
        outcome_column: Expected outcome column name
        treatment_column: Expected treatment column name
        split_column: Optional split column name
    
    Raises:
        ValueError: If validation fails
    """
    required = {text_column, outcome_column, treatment_column}
    if split_column:
        required.add(split_column)
    
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if df[text_column].isnull().any():
        raise ValueError(f"Null values in {text_column}")
    
    if df[outcome_column].isnull().any():
        raise ValueError(f"Null values in {outcome_column}")
    
    if df[treatment_column].isnull().any():
        raise ValueError(f"Null values in {treatment_column}")
    
    logger.info(f"Dataset validation passed: {len(df)} samples")
