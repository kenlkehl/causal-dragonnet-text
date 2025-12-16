"""
Causal Dragonnet Text (CDT)

A package for causal inference from clinical text using DragonNet models with multi-treatment pretraining.
"""

__version__ = "0.1.0"

from .config import (
    ExperimentConfig,
    AppliedInferenceConfig,
    PretrainingConfig,
    PlasmodeExperimentConfig,
    create_default_config
)

from .experiments import ExperimentRunner

__all__ = [
    'ExperimentConfig',
    'AppliedInferenceConfig', 
    'PretrainingConfig',
    'PlasmodeExperimentConfig',
    'ExperimentRunner',
    'create_default_config',
]
