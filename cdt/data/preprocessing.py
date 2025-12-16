# cdt/data/preprocessing.py

"""Text preprocessing utilities for chunking and embedding."""

import logging
from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util


logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 128,
    chunk_overlap: int = 32
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Number of words per chunk
        chunk_overlap: Number of overlapping words
    
    Returns:
        List of text chunks
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    words = text.split()
    if not words:
        return []
    
    chunks = []
    position = 0
    
    while position < len(words):
        end = min(position + chunk_size, len(words))
        chunk = " ".join(words[position:end])
        chunks.append(chunk)
        
        if end == len(words):
            break
        
        position += (chunk_size - chunk_overlap)
        if position >= end:
            position = end
    
    unique_chunks = list(dict.fromkeys(c for c in chunks if c.strip()))
    return unique_chunks


def embed_chunks(
    chunks: List[str],
    model: SentenceTransformer,
    device: torch.device,
    deduplication_threshold: float = 0.9
) -> Tuple[List[str], torch.Tensor]:
    """
    Embed text chunks and optionally deduplicate.
    
    Args:
        chunks: List of text chunks
        model: SentenceTransformer model
        device: PyTorch device
        deduplication_threshold: Cosine similarity threshold for deduplication
    
    Returns:
        Tuple of (deduplicated chunks, embeddings)
    """
    if not chunks:
        dim = model.get_sentence_embedding_dimension()
        return [], torch.empty(0, dim, device=device)
    
    embeddings = model.encode(
        chunks,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=device
    )
    
    if len(chunks) > 1 and deduplication_threshold is not None:
        chunks, embeddings = deduplicate_chunks(
            chunks, embeddings, deduplication_threshold
        )
    
    return chunks, embeddings



def deduplicate_chunks(
    chunks: List[str],
    embeddings: torch.Tensor,
    threshold: float = 0.9
) -> Tuple[List[str], torch.Tensor]:
    """Remove duplicate chunks based on cosine similarity."""
    if len(chunks) <= 1:
        return chunks, embeddings
    
    similarity = util.cos_sim(embeddings, embeddings)
    n = len(chunks)
    keep_mask = torch.ones(n, dtype=torch.bool, device=embeddings.device)
    
    # Get upper triangle indices (i,j) where i<j - avoids checking pairs twice
    triu_indices = torch.triu_indices(n, n, offset=1, device=embeddings.device)
    triu_similarities = similarity[triu_indices[0], triu_indices[1]]
    
    # Find duplicate pairs
    duplicate_pairs = triu_similarities > threshold
    
    if duplicate_pairs.any():
        # Keep first occurrence (i), remove later occurrences (j)
        keep_mask[triu_indices[1][duplicate_pairs]] = False
    
    # Extract kept chunks
    keep_indices = keep_mask.nonzero(as_tuple=True)[0].cpu().tolist()
    dedup_chunks = [chunks[i] for i in keep_indices]
    dedup_embeddings = embeddings[keep_mask]
    
    if len(dedup_chunks) < len(chunks):
        logger.debug(f"Deduplicated {len(chunks)} -> {len(dedup_chunks)} chunks")
    
    return dedup_chunks, dedup_embeddings


def process_text(
    text: str,
    model: SentenceTransformer,
    device: torch.device,
    chunk_size: int = 128,
    chunk_overlap: int = 32,
    deduplication_threshold: float = 0.9
) -> Tuple[List[str], torch.Tensor]:
    """
    Complete text processing pipeline: chunk, embed, deduplicate.
    
    Args:
        text: Input text
        model: SentenceTransformer model
        device: PyTorch device
        chunk_size: Words per chunk
        chunk_overlap: Overlapping words
        deduplication_threshold: Similarity threshold for deduplication
    
    Returns:
        Tuple of (chunks, embeddings)
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    if not chunks:
        dim = model.get_sentence_embedding_dimension()
        return [], torch.empty(0, dim, device=device)
    
    chunks, embeddings = embed_chunks(
        chunks, model, device, deduplication_threshold
    )
    
    return chunks, embeddings
