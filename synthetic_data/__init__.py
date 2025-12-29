# synthetic_data/__init__.py
"""LLM-based synthetic clinical data generation for causal inference benchmarking."""

from .config import SyntheticDataConfig, LLMConfig
from .generator import generate_synthetic_dataset

__all__ = [
    "SyntheticDataConfig",
    "LLMConfig", 
    "generate_synthetic_dataset",
]
