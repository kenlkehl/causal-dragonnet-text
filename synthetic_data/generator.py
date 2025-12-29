# synthetic_data/generator.py
"""Main synthetic data generation pipeline."""

import logging
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import SyntheticDataConfig
from .llm_client import LLMClient
from .prompts import (
    CLINICAL_SYSTEM_PROMPT,
    CONFOUNDER_GENERATION_PROMPT,
    REGRESSION_EQUATION_PROMPT,
    SUMMARY_STATISTICS_PROMPT,
    PATIENT_HISTORY_PROMPT,
    format_confounder_list,
    format_patient_characteristics,
)


logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    config: SyntheticDataConfig,
    num_workers: int = 4,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate a synthetic clinical dataset with known causal structure.
    
    This pipeline:
    1. Uses LLM to generate realistic confounders based on clinical question
    2. Uses LLM to generate treatment and outcome regression equations
    3. Uses LLM to generate summary statistics for confounders
    4. For each patient: samples characteristics, computes logits, generates clinical history
    5. Saves dataset and metadata
    
    Args:
        config: Configuration for generation
        num_workers: Number of parallel workers for patient history generation
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (dataset DataFrame, metadata dictionary)
    """
    config.validate()
    
    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Initialize LLM client
    client = LLMClient(config.llm)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting synthetic data generation for: {config.clinical_question[:80]}...")
    
    # Step 1: Generate confounders
    logger.info("Step 1/5: Generating confounders...")
    confounders = _generate_confounders(client, config.clinical_question)
    logger.info(f"Generated {len(confounders)} confounders: {[c['name'] for c in confounders]}")
    
    # Step 2: Generate regression equations
    logger.info("Step 2/5: Generating regression equations...")
    treatment_eq, outcome_eq = _generate_equations(
        client, config.clinical_question, confounders, config.treatment_coefficient
    )
    logger.info(f"Treatment equation has {len(treatment_eq['coefficients'])} terms")
    logger.info(f"Outcome equation has {len(outcome_eq['coefficients'])} terms")
    
    # Step 3: Generate summary statistics
    logger.info("Step 3/6: Generating summary statistics...")
    summary_stats = _generate_summary_statistics(client, config.clinical_question, confounders)
    
    # Step 4: Calibrate intercepts to hit target rates
    logger.info("Step 4/6: Calibrating intercepts to target rates...")
    treatment_eq, outcome_eq = _calibrate_intercepts(
        confounders=confounders,
        summary_stats=summary_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        target_treatment_rate=config.target_treatment_rate,
        target_control_outcome_rate=config.target_control_outcome_rate,
    )
    
    # Step 5: Generate patient data
    logger.info(f"Step 5/6: Generating {config.dataset_size} patients...")
    patient_data = _generate_all_patients(
        client=client,
        config=config,
        confounders=confounders,
        summary_stats=summary_stats,
        treatment_eq=treatment_eq,
        outcome_eq=outcome_eq,
        num_workers=num_workers,
        show_progress=show_progress,
    )
    
    # Step 6: Assemble dataset
    logger.info("Step 6/6: Assembling dataset...")
    df = pd.DataFrame(patient_data)
    
    # Compile metadata
    metadata = {
        "config": asdict(config),
        "confounders": confounders,
        "treatment_equation": treatment_eq,
        "outcome_equation": outcome_eq,
        "summary_statistics": summary_stats,
        "dataset_statistics": {
            "n_patients": len(df),
            "treatment_rate": df["treatment_indicator"].mean(),
            "outcome_rate": df["outcome_indicator"].mean(),
            "mean_treatment_logit": df["true_treatment_logit"].mean(),
            "std_treatment_logit": df["true_treatment_logit"].std(),
            "mean_outcome_logit": df["true_outcome_logit"].mean(),
            "std_outcome_logit": df["true_outcome_logit"].std(),
        }
    }
    
    # Save outputs
    dataset_path = output_dir / "dataset.parquet"
    metadata_path = output_dir / "metadata.json"
    
    df.to_parquet(dataset_path, index=False)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Dataset saved to {dataset_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    
    return df, metadata


def _generate_confounders(client: LLMClient, clinical_question: str) -> List[Dict[str, Any]]:
    """Generate confounders using LLM."""
    prompt = CONFOUNDER_GENERATION_PROMPT.format(clinical_question=clinical_question)
    
    response = client.generate_json(
        prompt=prompt,
        system_prompt=CLINICAL_SYSTEM_PROMPT,
        temperature=0.7,
    )
    
    confounders = response.get("confounders", [])
    
    # Validate structure
    for conf in confounders:
        if "name" not in conf or "type" not in conf:
            raise ValueError(f"Invalid confounder structure: {conf}")
        if conf["type"] == "categorical" and "categories" not in conf:
            raise ValueError(f"Categorical confounder missing categories: {conf}")
    
    return confounders


def _generate_equations(
    client: LLMClient,
    clinical_question: str,
    confounders: List[Dict[str, Any]],
    treatment_coefficient: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate treatment and outcome regression equations."""
    confounder_list = format_confounder_list(confounders)
    
    prompt = REGRESSION_EQUATION_PROMPT.format(
        clinical_question=clinical_question,
        confounder_list=confounder_list,
        treatment_coefficient=treatment_coefficient,
    )
    
    response = client.generate_json(
        prompt=prompt,
        system_prompt=CLINICAL_SYSTEM_PROMPT,
        temperature=0.5,  # Lower temperature for more consistent equations
    )
    
    treatment_eq = response.get("treatment_equation", {})
    outcome_eq = response.get("outcome_equation", {})
    
    # Add fixed treatment coefficient to outcome equation
    outcome_eq["treatment_coefficient"] = treatment_coefficient
    
    return treatment_eq, outcome_eq


def _generate_summary_statistics(
    client: LLMClient,
    clinical_question: str,
    confounders: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate summary statistics for confounders."""
    confounder_list = format_confounder_list(confounders)
    
    prompt = SUMMARY_STATISTICS_PROMPT.format(
        clinical_question=clinical_question,
        confounder_list=confounder_list,
    )
    
    response = client.generate_json(
        prompt=prompt,
        system_prompt=CLINICAL_SYSTEM_PROMPT,
        temperature=0.5,
    )
    
    return response.get("summary_statistics", {})


def _calibrate_intercepts(
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    treatment_eq: Dict[str, Any],
    outcome_eq: Dict[str, Any],
    target_treatment_rate: float,
    target_control_outcome_rate: float,
    n_samples: int = 10000,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Calibrate equation intercepts to achieve target marginal rates.
    
    Uses Monte Carlo sampling and binary search to find intercepts that yield
    the desired treatment rate and control group outcome rate.
    
    Args:
        confounders: List of confounder definitions
        summary_stats: Summary statistics for sampling
        treatment_eq: Treatment assignment equation (will be modified)
        outcome_eq: Outcome equation (will be modified)
        target_treatment_rate: Desired proportion receiving treatment=1
        target_control_outcome_rate: Desired outcome rate when treatment=0
        n_samples: Number of Monte Carlo samples for calibration
        
    Returns:
        Tuple of (calibrated_treatment_eq, calibrated_outcome_eq)
    """
    from scipy.optimize import brentq
    
    # Sample characteristics for calibration
    sampled_chars = [
        _sample_patient_characteristics(confounders, summary_stats)
        for _ in range(n_samples)
    ]
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def compute_linear_predictor(characteristics, equation, confounders, summary_stats, treatment=None):
        """Compute linear predictor WITHOUT intercept."""
        # Temporarily set intercept to 0
        original_intercept = equation.get("intercept", 0.0)
        equation["intercept"] = 0.0
        logit = _compute_logit(characteristics, confounders, summary_stats, equation, treatment=treatment)
        equation["intercept"] = original_intercept
        return logit
    
    # Compute linear predictors for all samples (without intercept)
    treatment_lps = np.array([
        compute_linear_predictor(chars, treatment_eq, confounders, summary_stats)
        for chars in sampled_chars
    ])
    
    outcome_lps = np.array([
        compute_linear_predictor(chars, outcome_eq, confounders, summary_stats, treatment=0)
        for chars in sampled_chars
    ])
    
    # Calibrate treatment intercept
    def treatment_rate_error(intercept):
        probs = sigmoid(intercept + treatment_lps)
        return probs.mean() - target_treatment_rate
    
    try:
        # Search for intercept in reasonable range
        calibrated_treatment_intercept = brentq(treatment_rate_error, -10, 10)
    except ValueError:
        # If target is outside achievable range, use boundary
        logger.warning(f"Could not achieve target treatment rate {target_treatment_rate}, using closest achievable")
        if treatment_rate_error(-10) > 0:
            calibrated_treatment_intercept = -10
        else:
            calibrated_treatment_intercept = 10
    
    # Calibrate outcome intercept (for control group)
    def outcome_rate_error(intercept):
        probs = sigmoid(intercept + outcome_lps)
        return probs.mean() - target_control_outcome_rate
    
    try:
        calibrated_outcome_intercept = brentq(outcome_rate_error, -10, 10)
    except ValueError:
        logger.warning(f"Could not achieve target control outcome rate {target_control_outcome_rate}, using closest achievable")
        if outcome_rate_error(-10) > 0:
            calibrated_outcome_intercept = -10
        else:
            calibrated_outcome_intercept = 10
    
    # Update equations with calibrated intercepts
    original_treatment_intercept = treatment_eq.get("intercept", 0.0)
    original_outcome_intercept = outcome_eq.get("intercept", 0.0)
    
    treatment_eq["intercept"] = calibrated_treatment_intercept
    treatment_eq["original_intercept"] = original_treatment_intercept
    
    outcome_eq["intercept"] = calibrated_outcome_intercept
    outcome_eq["original_intercept"] = original_outcome_intercept
    
    logger.info(f"Calibrated treatment intercept: {original_treatment_intercept:.3f} -> {calibrated_treatment_intercept:.3f}")
    logger.info(f"Calibrated outcome intercept: {original_outcome_intercept:.3f} -> {calibrated_outcome_intercept:.3f}")
    
    # Verify achieved rates
    achieved_treatment_rate = sigmoid(calibrated_treatment_intercept + treatment_lps).mean()
    achieved_outcome_rate = sigmoid(calibrated_outcome_intercept + outcome_lps).mean()
    logger.info(f"Achieved treatment rate: {achieved_treatment_rate:.3f} (target: {target_treatment_rate:.3f})")
    logger.info(f"Achieved control outcome rate: {achieved_outcome_rate:.3f} (target: {target_control_outcome_rate:.3f})")
    
    return treatment_eq, outcome_eq


def _sample_patient_characteristics(
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Sample patient characteristics from summary statistics."""
    characteristics = {}
    
    for conf in confounders:
        name = conf["name"]
        stats = summary_stats.get(name, {})
        
        if conf["type"] == "continuous":
            # Sample from normal distribution
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            value = np.random.normal(mean, std)
            characteristics[name] = value
        else:
            # Sample from categorical distribution
            categories = conf["categories"]
            proportions = stats.get("proportions", {})
            
            # Default to uniform if no proportions
            if not proportions:
                proportions = {cat: 1.0 / len(categories) for cat in categories}
            
            # Ensure proportions sum to 1
            probs = [proportions.get(cat, 0.0) for cat in categories]
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]
            else:
                probs = [1.0 / len(categories)] * len(categories)
            
            chosen = np.random.choice(categories, p=probs)
            characteristics[name] = chosen
    
    return characteristics


def _compute_logit(
    characteristics: Dict[str, Any],
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    equation: Dict[str, Any],
    treatment: Optional[int] = None,
) -> float:
    """
    Compute logit from characteristics using regression equation.
    
    For continuous variables: coefficient * (value - mean) / std (z-scored)
    For categorical variables: coefficient of the dummy for selected category
    """
    logit = equation.get("intercept", 0.0)
    coefficients = equation.get("coefficients", {})
    
    # Build z-scored continuous values map
    z_values = {}
    for conf in confounders:
        name = conf["name"]
        if conf["type"] == "continuous":
            stats = summary_stats.get(name, {})
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            if std == 0:
                std = 1.0
            z_values[name] = (characteristics[name] - mean) / std
    
    # Apply coefficients
    for coef_name, coef_value in coefficients.items():
        # Check if this is a base continuous variable
        if coef_name in z_values:
            logit += coef_value * z_values[coef_name]
            continue
        
        # Check if this is a categorical dummy (format: varname_category)
        matched = False
        for conf in confounders:
            if conf["type"] == "categorical":
                name = conf["name"]
                for cat in conf["categories"][1:]:  # Skip reference category
                    dummy_name = f"{name}_{cat}"
                    if coef_name == dummy_name:
                        # Add coefficient if this category is selected
                        if characteristics.get(name) == cat:
                            logit += coef_value
                        matched = True
                        break
            if matched:
                break
    
    # Apply interactions
    interactions = equation.get("interactions", [])
    for interaction in interactions:
        terms = interaction.get("terms", [])
        coef = interaction.get("coefficient", 0.0)
        
        # Compute product of z-scored values
        product = 1.0
        for term in terms:
            if term in z_values:
                product *= z_values[term]
            elif term in characteristics:
                # For categorical in interaction, use indicator
                product *= 1.0
            else:
                product = 0.0
                break
        
        logit += coef * product
    
    # Add treatment effect if provided
    if treatment is not None:
        treatment_coef = equation.get("treatment_coefficient", 0.0)
        logit += treatment_coef * treatment
    
    return logit


def _generate_single_patient(
    patient_idx: int,
    client: LLMClient,
    config: SyntheticDataConfig,
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    treatment_eq: Dict[str, Any],
    outcome_eq: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate data for a single patient."""
    # Sample characteristics
    characteristics = _sample_patient_characteristics(confounders, summary_stats)
    
    # Compute treatment logit and sample treatment
    treatment_logit = _compute_logit(
        characteristics, confounders, summary_stats, treatment_eq
    )
    treatment_prob = 1.0 / (1.0 + np.exp(-treatment_logit))
    treatment = int(np.random.random() < treatment_prob)
    
    # Compute outcome logit and sample outcome
    outcome_logit = _compute_logit(
        characteristics, confounders, summary_stats, outcome_eq, treatment=treatment
    )
    outcome_prob = 1.0 / (1.0 + np.exp(-outcome_logit))
    outcome = int(np.random.random() < outcome_prob)
    
    # Format patient characteristics as prompt
    patient_prompt = format_patient_characteristics(characteristics, confounders)
    
    # Generate clinical history
    history_prompt = PATIENT_HISTORY_PROMPT.format(
        patient_characteristics=patient_prompt,
        clinical_question=config.clinical_question,
    )
    
    # Reserve tokens for prompt (system prompt + patient history prompt ~1500-2000 tokens)
    history_max_tokens = max(1000, config.llm.max_tokens - 2000)
    
    clinical_history = client.generate(
        prompt=history_prompt,
        system_prompt=CLINICAL_SYSTEM_PROMPT,
        temperature=0.8,  # Higher temperature for more varied text
        max_tokens=history_max_tokens,
    )
    
    return {
        "patient_id": patient_idx,
        "patient_prompt": patient_prompt,
        "clinical_text": clinical_history,
        "treatment_indicator": treatment,
        "outcome_indicator": outcome,
        "true_treatment_logit": treatment_logit,
        "true_outcome_logit": outcome_logit,
    }


def _generate_all_patients(
    client: LLMClient,
    config: SyntheticDataConfig,
    confounders: List[Dict[str, Any]],
    summary_stats: Dict[str, Any],
    treatment_eq: Dict[str, Any],
    outcome_eq: Dict[str, Any],
    num_workers: int = 4,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Generate data for all patients with parallel LLM calls."""
    patient_data = []
    
    if num_workers <= 1:
        # Sequential generation
        iterator = range(config.dataset_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating patients")
        
        for i in iterator:
            data = _generate_single_patient(
                i, client, config, confounders, summary_stats, treatment_eq, outcome_eq
            )
            patient_data.append(data)
    else:
        # Parallel generation
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _generate_single_patient,
                    i, client, config, confounders, summary_stats, treatment_eq, outcome_eq
                ): i
                for i in range(config.dataset_size)
            }
            
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=config.dataset_size, desc="Generating patients")
            
            for future in iterator:
                data = future.result()
                patient_data.append(data)
    
    # Sort by patient_id to ensure reproducibility
    patient_data.sort(key=lambda x: x["patient_id"])
    
    return patient_data
