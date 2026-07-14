"""Focused tests for the one-off cohort contrast calculation."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy import sparse


SCRIPT = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("cohort_contrast_run", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_cohort_score_prioritizes_true_modifier() -> None:
    rng = np.random.default_rng(7)
    n = 4000
    modifier = rng.binomial(1, 0.5, size=n)
    nuisance_terms = rng.binomial(1, 0.2, size=(n, 12))
    x = sparse.csr_matrix(np.column_stack([modifier, nuisance_terms]), dtype=np.float32)
    propensity = np.full(n, 0.5)
    treatment = rng.binomial(1, propensity)
    baseline = np.full(n, 0.35)
    treatment_effect = -0.05 + 0.35 * modifier
    outcome_probability = np.clip(baseline + treatment * treatment_effect, 0.01, 0.99)
    outcome = rng.binomial(1, outcome_probability)

    result = MODULE.cohort_contrast(
        x,
        treatment,
        outcome,
        propensity,
        baseline + propensity * treatment_effect,
        probability_clip=0.01,
    )

    assert result.z_score[0] > 8.0
    assert abs(result.z_score[0]) > np.max(np.abs(result.z_score[1:]))


def test_constant_effect_has_finite_scores() -> None:
    rng = np.random.default_rng(13)
    n = 1000
    x = sparse.csr_matrix(rng.binomial(1, 0.25, size=(n, 20)), dtype=np.float32)
    propensity = np.full(n, 0.5)
    treatment = rng.binomial(1, propensity)
    baseline = np.full(n, 0.4)
    outcome = rng.binomial(1, np.clip(baseline + 0.1 * treatment, 0.01, 0.99))

    result = MODULE.cohort_contrast(
        x,
        treatment,
        outcome,
        propensity,
        baseline + propensity * 0.1,
        probability_clip=0.01,
    )

    assert np.all(np.isfinite(result.z_score))
    assert abs(np.sum(result.patient_contribution)) < 1e-8
