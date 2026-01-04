# synthetic_data/config.py
"""Configuration classes for synthetic data generation."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import json


DEFAULT_CLINICAL_QUESTION = (
    "Compare the effectiveness of letrozole+palbociclib with letrozole+ribociclib "
    "for the first line treatment of metastatic ER positive breast cancer."
)


@dataclass
class LLMConfig:
    """Configuration for OpenAI-compatible LLM API."""
    api_base_url: str = "http://localhost:8000/v1"  # vLLM default
    api_key: str = ""  # Can be blank for local models
    model_name: str = "openai/gpt-oss-120b"  # Model name to use
    temperature: float = 0.7
    max_tokens: int = 12000
    timeout: float = 300.0  # Request timeout in seconds


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic dataset generation."""
    # Clinical research question
    clinical_question: str = DEFAULT_CLINICAL_QUESTION

    # Dataset parameters
    dataset_size: int = 500
    treatment_coefficient: float = 0.5  # Coefficient for treatment in outcome equation (logit scale)

    # Target rates (intercepts will be calibrated to achieve these)
    target_treatment_rate: float = 0.5  # Proportion of patients receiving treatment=1
    target_control_outcome_rate: float = 0.2  # Outcome rate in control group (treatment=0)

    # Coefficient scaling (to keep logits in reasonable range)
    main_coefficient_scale: float = 0.3  # Scale for main effect coefficients
    interaction_coefficient_scale: float = 0.1  # Scale for interaction coefficients
    target_logit_std: float = 2.0  # Target std for final logits (after rescaling)
    
    # Number of confounders (None = use LLM default of 8-12)
    num_confounders: Optional[int] = None
    
    # Output
    output_dir: str = "./synthetic_output"
    
    # LLM settings
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Reproducibility
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> 'SyntheticDataConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyntheticDataConfig':
        """Create config from dictionary."""
        llm_data = data.pop('llm', {})
        llm_config = LLMConfig(**llm_data) if llm_data else LLMConfig()
        return cls(llm=llm_config, **data)
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.dataset_size < 1:
            raise ValueError("dataset_size must be at least 1")
        
        if not self.clinical_question.strip():
            raise ValueError("clinical_question cannot be empty")
        
        if not self.llm.api_base_url:
            raise ValueError("llm.api_base_url is required")
