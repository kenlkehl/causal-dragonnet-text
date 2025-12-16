# cdt/data/__init__.py

"""Data handling modules for CDT."""

from .dataset import (
    ClinicalTextDataset,
    collate_batch,
    load_dataset,
    validate_dataset
)

from .preprocessing import (
    chunk_text,
    embed_chunks,
    process_text
)

from .cache import (
    EmbeddingCache,
    create_cache
)

__all__ = [
    'ClinicalTextDataset',
    'collate_batch',
    'load_dataset',
    'validate_dataset',
    'chunk_text',
    'embed_chunks',
    'process_text',
    'EmbeddingCache',
    'create_cache',
]
