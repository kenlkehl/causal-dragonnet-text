from pathlib import Path

import pytest

from oci.config import ExperimentConfig


def _experiment_config(
    dataset_path: Path,
    *,
    model_type: str,
    methods: list[str] | None = None,
) -> ExperimentConfig:
    architecture = {"model_type": model_type}
    if methods is not None:
        architecture[model_type] = {"feature_discovery_methods": methods}
    return ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": str(dataset_path),
                "architecture": architecture,
            }
        }
    )


def test_multi_model_forest_v2_default_methods_pass_experiment_validation(
    tmp_path: Path,
):
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.touch()
    config = _experiment_config(dataset_path, model_type="multi_model_forest")

    assert (
        config.applied_inference.architecture.multi_model_forest.feature_discovery_methods
        == ["bow", "tfidf_topic_contrast"]
    )
    assert (
        config.applied_inference.architecture.multi_model_forest.structured_effect_estimator
        == "causal_forest"
    )
    config.validate()


def test_multi_model_forest_v2_validation_rejects_legacy_methods(tmp_path: Path):
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.touch()
    config = _experiment_config(
        dataset_path,
        model_type="multi_model_forest",
        methods=["htr"],
    )

    with pytest.raises(ValueError, match="legacy discovery method"):
        config.validate()


def test_multi_model_agentic_validation_keeps_legacy_method_contract(tmp_path: Path):
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.touch()
    config = _experiment_config(
        dataset_path,
        model_type="multi_model_agentic_forest",
        methods=["htr"],
    )

    assert (
        config.applied_inference.architecture.multi_model_agentic_forest
        .feature_discovery_methods
        == ["htr"]
    )
    config.validate()
