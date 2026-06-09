import math

import pytest


def _config(**kwargs):
    from oracle_experiment_scripts.run_oracle_experiments import ExperimentConfig

    params = dict(
        dataset_path="dummy",
        dataset_name="dummy",
        model_type="rlearner",
        use_explicit_confounders=False,
    )
    params.update(kwargs)
    return ExperimentConfig(**params)


def test_scheduled_gamma_constant_default():
    from oracle_experiment_scripts.run_oracle_experiments import scheduled_gamma_rlearner

    config = _config(gamma_rlearner=7.0)

    assert scheduled_gamma_rlearner(config, 0) == 7.0
    assert scheduled_gamma_rlearner(config, 100) == 7.0


def test_scheduled_gamma_linear_warmup_and_ramp():
    from oracle_experiment_scripts.run_oracle_experiments import scheduled_gamma_rlearner

    config = _config(
        gamma_rlearner=10.0,
        gamma_rlearner_start=0.0,
        gamma_rlearner_warmup_epochs=2,
        gamma_rlearner_ramp_epochs=4,
        gamma_rlearner_schedule="linear",
    )

    values = [scheduled_gamma_rlearner(config, epoch) for epoch in range(8)]
    assert values == [0.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0]


def test_scheduled_gamma_nonconstant_defaults_to_zero_start():
    from oracle_experiment_scripts.run_oracle_experiments import scheduled_gamma_rlearner

    config = _config(
        gamma_rlearner=10.0,
        gamma_rlearner_ramp_epochs=2,
        gamma_rlearner_schedule="linear",
    )

    assert scheduled_gamma_rlearner(config, 0) == 5.0
    assert scheduled_gamma_rlearner(config, 1) == 10.0


def test_scheduled_gamma_cosine_warmup_and_ramp():
    from oracle_experiment_scripts.run_oracle_experiments import scheduled_gamma_rlearner

    config = _config(
        gamma_rlearner=10.0,
        gamma_rlearner_start=0.0,
        gamma_rlearner_warmup_epochs=1,
        gamma_rlearner_ramp_epochs=2,
        gamma_rlearner_schedule="cosine",
    )

    assert scheduled_gamma_rlearner(config, 0) == 0.0
    assert math.isclose(scheduled_gamma_rlearner(config, 1), 5.0)
    assert math.isclose(scheduled_gamma_rlearner(config, 2), 10.0)


def test_scheduled_gamma_rejects_unknown_schedule():
    from oracle_experiment_scripts.run_oracle_experiments import scheduled_gamma_rlearner

    config = _config(
        gamma_rlearner=10.0,
        gamma_rlearner_start=0.0,
        gamma_rlearner_schedule="bad",
    )

    with pytest.raises(ValueError):
        scheduled_gamma_rlearner(config, 0)
