"""
Causal Dragonnet Text (CDT)

A package for causal inference from clinical text using CNN-based DragonNet models.
"""

__version__ = "0.1.0"

from .config import (
    ExperimentConfig,
    AppliedInferenceConfig,
    PlasmodeExperimentConfig,
    ModelArchitectureConfig,
    TrainingConfig,
    create_default_config
)

from .experiments import ExperimentRunner

__all__ = [
    'ExperimentConfig',
    'AppliedInferenceConfig',
    'PlasmodeExperimentConfig',
    'ModelArchitectureConfig',
    'TrainingConfig',
    'ExperimentRunner',
    'create_default_config',
]
