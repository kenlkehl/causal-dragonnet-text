# cdt/utils/__init__.py

"""Utility functions for CDT."""

from .system import (
    limit_threads,
    set_seed,
    cuda_cleanup,
    get_memory_info,
    setup_logging,
    get_device
)

from .io import (
    hash_text,
    safe_filename,
    atomic_save,
    ensure_dir
)

from .model_loading import (
    load_pretrained_with_dimension_matching,
    check_checkpoint_compatibility,
    create_compatible_model_from_checkpoint,
    extract_feature_extractor_config
)

__all__ = [
    'limit_threads',
    'set_seed',
    'cuda_cleanup',
    'get_memory_info',
    'setup_logging',
    'get_device',
    'hash_text',
    'safe_filename',
    'atomic_save',
    'ensure_dir',
    'load_pretrained_with_dimension_matching',
    'check_checkpoint_compatibility',
    'create_compatible_model_from_checkpoint',
    'extract_feature_extractor_config',
]
