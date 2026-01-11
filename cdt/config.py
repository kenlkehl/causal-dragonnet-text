# cdt/config.py
"""Configuration classes for CDT experiments."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import hashlib


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture."""
    model_type: str = "dragonnet"  # "dragonnet" or "uplift"
    # Model backbone selection: "sentence_transformer", "modernbert", or "cnn"
    model_backbone: str = "sentence_transformer"
    # Sentence transformer settings (used when model_backbone="sentence_transformer")
    embedding_model_name: str = "all-MiniLM-L6-v2"
    num_latent_confounders: int = 20
    explicit_confounder_texts: Optional[List[str]] = None
    chunk_size: int = 128
    chunk_overlap: int = 32
    # Cross-attention aggregation parameters (sentence_transformer only)
    value_dim: int = 128  # Output dimension per confounder in cross-attention
    num_attention_heads: int = 4  # Number of attention heads per confounder
    attention_dropout: float = 0.1  # Dropout on attention weights
    # ModernBERT settings (used when model_backbone="modernbert")
    modernbert_model_name: str = "answerdotai/ModernBERT-base"
    freeze_modernbert: bool = False
    modernbert_max_length: int = 8192
    # CNN settings (used when model_backbone="cnn")
    cnn_embedding_dim: int = 128  # Word embedding dimension
    cnn_num_filters: int = 256  # Number of filters per kernel size
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 4, 5, 7])
    cnn_dropout: float = 0.1
    cnn_max_length: int = 2048  # Max sequence length in tokens (words)
    cnn_min_word_freq: int = 2  # Minimum word frequency for vocabulary
    cnn_max_vocab_size: int = 50000  # Maximum vocabulary size
    # Shared settings
    dragonnet_representation_dim: int = 128
    dragonnet_hidden_outcome_dim: int = 64


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    lr_schedule: str = "linear"
    epochs: int = 50
    batch_size: int = 8
    alpha_propensity: float = 1.0
    beta_targreg: float = 0.1
    init_latents_from_kmeans: bool = True


@dataclass
class PretrainingConfig:
    """Configuration for multi-treatment pretraining."""
    enabled: bool = False
    dataset_path: Optional[str] = None
    treatment_column: str = "treatment"
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class PlasmodeConfig:
    """Configuration for plasmode simulation."""
    generation_mode: str = "phi_linear"
    preserve_observed_treatments: bool = True
    baseline_control_outcome_rate: float = 0.20
    target_ate_logit: float = 0.50
    outcome_heterogeneity_scale: float = 1.0
    ite_heterogeneity_scale: float = 1.0
    deep_nonlinear_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    deep_nonlinear_dropout: float = 0.0
    uplift_hidden_dims: List[int] = field(default_factory=list)
    uplift_activation: str = "relu"
    uplift_dropout: float = 0.0


@dataclass
class AppliedInferenceConfig:
    """Configuration for applied inference on real data."""
    dataset_path: str = ""
    text_column: str = "clinical_text"
    outcome_column: str = "outcome_indicator"
    treatment_column: str = "treatment_indicator"
    split_column: str = "split"
    cv_folds: int = 5  # Added: Number of CV folds (0 or 1 = fixed split)
    architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    use_pretrained_weights: bool = True
    skip: bool = False  # Skip applied inference, go straight to plasmode


@dataclass
class PlasmodeExperimentConfig:
    """Configuration for plasmode sensitivity experiments."""
    enabled: bool = False
    num_repeats: int = 1
    save_datasets: bool = False
    train_fraction: float = 0.8  # Fraction of data for training generator/evaluator (rest is eval)
    generator_architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    generator_training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluator_architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    evaluator_training: TrainingConfig = field(default_factory=TrainingConfig)
    plasmode_scenarios: List[PlasmodeConfig] = field(default_factory=list)
    # Oracle mode: evaluator trains directly on generator's confounder_features.
    # - True: Evaluator sees exact confounder features used to generate ITEs (no text processing)
    # - False (default): Evaluator learns its own confounders from text (realistic mode)
    oracle_mode: bool = False


@dataclass
class ExperimentConfig:
    """Main configuration for CDT experiments."""
    output_dir: str = "./cdt_results"
    seed: int = 42
    device: Optional[str] = None
    num_workers: int = 1
    gpu_ids: Optional[List[int]] = None  
    cache_dir: Optional[str] = None
    
    pretraining: PretrainingConfig = field(default_factory=PretrainingConfig)
    applied_inference: AppliedInferenceConfig = field(default_factory=AppliedInferenceConfig)
    plasmode_experiments: PlasmodeExperimentConfig = field(default_factory=PlasmodeExperimentConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        pretraining = PretrainingConfig(
            **{k: ModelArchitectureConfig(**v) if k == 'architecture' 
               else TrainingConfig(**v) if k == 'training' 
               else v 
               for k, v in data.get('pretraining', {}).items()}
        )
        
        applied = AppliedInferenceConfig(
            **{k: ModelArchitectureConfig(**v) if k == 'architecture'
               else TrainingConfig(**v) if k == 'training'
               else v
               for k, v in data.get('applied_inference', {}).items()}
        )
        
        plasmode_data = data.get('plasmode_experiments', {})
        plasmode = PlasmodeExperimentConfig(
            enabled=plasmode_data.get('enabled', False),
            num_repeats=plasmode_data.get('num_repeats', 1),
            save_datasets=plasmode_data.get('save_datasets', False),
            train_fraction=plasmode_data.get('train_fraction', 0.8),
            generator_architecture=ModelArchitectureConfig(**plasmode_data.get('generator_architecture', {})),
            generator_training=TrainingConfig(**plasmode_data.get('generator_training', {})),
            evaluator_architecture=ModelArchitectureConfig(**plasmode_data.get('evaluator_architecture', {})),
            evaluator_training=TrainingConfig(**plasmode_data.get('evaluator_training', {})),
            plasmode_scenarios=[PlasmodeConfig(**s) for s in plasmode_data.get('plasmode_scenarios', [])],
            # Support both old and new config variable names
            oracle_mode=plasmode_data.get('oracle_mode', plasmode_data.get('evaluator_use_generator_confounders', False))
        )
        
        return cls(
            output_dir=data.get('output_dir', './cdt_results'),
            seed=data.get('seed', 42),
            device=data.get('device'),
            num_workers=data.get('num_workers', 1),
            gpu_ids=data.get('gpu_ids'),
            cache_dir=data.get('cache_dir'),
            pretraining=pretraining,
            applied_inference=applied,
            plasmode_experiments=plasmode
        )
    
    def get_hash(self) -> str:
        """Get hash of config for caching."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.applied_inference.dataset_path:
            raise ValueError("applied_inference.dataset_path is required")
        
        if not Path(self.applied_inference.dataset_path).exists():
            raise ValueError(f"Dataset not found: {self.applied_inference.dataset_path}")
        
        if self.pretraining.enabled:
            if not self.pretraining.dataset_path:
                raise ValueError("pretraining.dataset_path required when pretraining.enabled=True")
            if not Path(self.pretraining.dataset_path).exists():
                raise ValueError(f"Pretraining dataset not found: {self.pretraining.dataset_path}")
        
        if self.plasmode_experiments.enabled and not self.plasmode_experiments.plasmode_scenarios:
            raise ValueError("plasmode_experiments.plasmode_scenarios cannot be empty when enabled=True")


def create_default_config(output_path: str) -> None:
    """Create a default configuration file with documentation."""
    config = ExperimentConfig(
        output_dir="./cdt_results",
        seed=42,
        device="cuda:0",
        num_workers=1,
        gpu_ids=[0, 1],  # Example: Use GPUs 0 and 1
        
        pretraining=PretrainingConfig(
            enabled=False,
            dataset_path="./pretrain_dataset.parquet",
            treatment_column="treatment",
            architecture=ModelArchitectureConfig(
                num_latent_confounders=50
            ),
            training=TrainingConfig(
                epochs=10,
                batch_size=8
            )
        ),

        applied_inference=AppliedInferenceConfig(
            dataset_path="./dataset.parquet",
            cv_folds=5,  # Default to 5-fold CV
            architecture=ModelArchitectureConfig(
                num_latent_confounders=20
            ),
            training=TrainingConfig(
                epochs=50,
                batch_size=8
            ),
            use_pretrained_weights=True
        ),
        
        plasmode_experiments=PlasmodeExperimentConfig(
            enabled=False,
            num_repeats=3,
            save_datasets=False,
            plasmode_scenarios=[
                PlasmodeConfig(
                    generation_mode="phi_linear",
                    target_ate_logit=0.5
                ),
                PlasmodeConfig(
                    generation_mode="deep_nonlinear",
                    target_ate_logit=0.5
                )
            ]
        )
    )
    
    config.to_json(output_path)
    print(f"Default configuration saved to: {output_path}")