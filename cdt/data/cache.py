# cdt/data/cache.py

"""Embedding cache for efficient reuse across experiments."""

import logging
from pathlib import Path
from typing import Callable, Tuple, List, Optional
import torch

from ..utils.io import hash_text, atomic_save, ensure_dir


logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for text embeddings to avoid recomputation."""
    
    def __init__(
        self,
        cache_dir: Path,
        enabled: bool = True
    ):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory for cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        
        if self.enabled:
            ensure_dir(self.cache_dir)
            logger.info(f"Embedding cache initialized: {self.cache_dir}")
    
    def get_cache_path(self, text: str) -> Path:
        """Get cache file path for text."""
        text_hash = hash_text(text)
        return self.cache_dir / f"{text_hash}.pt"
    
    def exists(self, text: str) -> bool:
        """Check if embedding exists in cache."""
        if not self.enabled:
            return False
        return self.get_cache_path(text).exists()
    
    def get(self, text: str) -> Optional[Tuple[List[str], torch.Tensor]]:
        """
        Get cached embedding.
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (chunks, embeddings) or None if not cached
        """
        if not self.enabled:
            return None
        
        cache_path = self.get_cache_path(text)
        if not cache_path.exists():
            return None
        
        try:
            cached = torch.load(cache_path, map_location='cpu')
            chunks = cached['chunks_text_list']
            embeddings = cached['chunk_embeddings']
            return chunks, embeddings
        except Exception as e:
            logger.warning(f"Failed to load cache for {cache_path.name}: {e}")
            return None
    
    def set(
        self,
        text: str,
        chunks: List[str],
        embeddings: torch.Tensor
    ) -> None:
        """
        Save embedding to cache.
        
        Args:
            text: Input text
            chunks: Text chunks
            embeddings: Chunk embeddings
        """
        if not self.enabled:
            return
        
        cache_path = self.get_cache_path(text)
        
        try:
            obj = {
                'chunks_text_list': chunks,
                'chunk_embeddings': embeddings.detach().cpu() if torch.is_tensor(embeddings) else embeddings
            }
            atomic_save(obj, cache_path)
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_path.name}: {e}")
    
    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], Tuple[List[str], torch.Tensor]]
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Get embedding from cache or compute if not cached.
        
        Args:
            text: Input text
            compute_fn: Function to compute embedding if not cached
        
        Returns:
            Tuple of (chunks, embeddings)
        """
        cached = self.get(text)
        if cached is not None:
            return cached
        
        chunks, embeddings = compute_fn(text)
        self.set(text, chunks, embeddings)
        
        return chunks, embeddings
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        if not self.enabled or not self.cache_dir.exists():
            return
        
        count = 0
        for cache_file in self.cache_dir.glob("*.pt"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file.name}: {e}")
        
        logger.info(f"Cleared {count} cached embeddings")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled or not self.cache_dir.exists():
            return {'enabled': False}
        
        cache_files = list(self.cache_dir.glob("*.pt"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'enabled': True,
            'num_entries': len(cache_files),
            'total_size_mb': total_size / (1024**2),
            'cache_dir': str(self.cache_dir)
        }


def create_cache(
    cache_dir: Optional[str],
    model_name: str,
    chunk_size: int,
    chunk_overlap: int
) -> EmbeddingCache:
    """
    Create embedding cache with configuration-specific subdirectory.
    
    Args:
        cache_dir: Base cache directory (None to disable caching)
        model_name: Embedding model name
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
    
    Returns:
        EmbeddingCache instance
    """
    if cache_dir is None:
        return EmbeddingCache(Path(""), enabled=False)
    
    from ..utils.io import safe_filename
    
    model_safe = safe_filename(model_name)
    config_str = f"{model_safe}_cs{chunk_size}_co{chunk_overlap}"
    cache_path = Path(cache_dir) / config_str
    
    return EmbeddingCache(cache_path, enabled=True)
