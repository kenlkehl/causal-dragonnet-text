from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from oci.config import ExperimentConfig


def test_spdx_license_metadata_requires_a_compatible_setuptools_backend():
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["license"] == "MIT"
    assert "setuptools>=77.0.0" in pyproject["build-system"]["requires"]


def test_generic_experiment_validation_rejects_stage1_only_default(tmp_path: Path):
    dataset = tmp_path / "dataset.parquet"
    dataset.touch()
    config = ExperimentConfig.from_dict(
        {"applied_inference": {"dataset_path": str(dataset)}}
    )

    assert config.applied_inference.architecture.model_type == "multi_model_forest"
    with pytest.raises(ValueError, match="ResearchAllEvidenceWorkflow"):
        config.validate()
