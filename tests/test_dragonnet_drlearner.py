import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from oci.config import (
    AppliedInferenceConfig,
    DragonNetDRLearnerConfig,
    ExperimentConfig,
    ModelArchitectureConfig,
    TrainingConfig,
)
from oci.inference.dragonnet_drlearner import (
    DragonNetDRLearnerRunner,
    dr_pseudo_outcome,
)


def test_dr_pseudo_outcome_uses_aipw_formula_and_clips_propensity():
    y = np.asarray([1.0, 0.0])
    t = np.asarray([1.0, 0.0])
    mu0 = np.asarray([0.2, 0.4])
    mu1 = np.asarray([0.7, 0.6])
    e = np.asarray([0.001, 0.999])

    pseudo = dr_pseudo_outcome(y, t, mu0, mu1, e, e_clip=0.1)

    expected0 = (0.7 - 0.2) + (1.0 - 0.7) / 0.1
    expected1 = (0.6 - 0.4) - (0.0 - 0.4) / 0.1
    np.testing.assert_allclose(pseudo, [expected0, expected1])


def test_dragonnet_dr_config_parses_from_experiment_dict():
    cfg = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "architecture": {
                    "model_type": "dragonnet_drlearner",
                    "dragonnet_drlearner": {
                        "nuisance_folds": 3,
                        "effect_folds": 4,
                        "effect_loss": "mse",
                    },
                }
            }
        }
    )

    dr_cfg = cfg.applied_inference.architecture.dragonnet_drlearner
    assert isinstance(dr_cfg, DragonNetDRLearnerConfig)
    assert dr_cfg.nuisance_folds == 3
    assert dr_cfg.effect_folds == 4
    assert dr_cfg.effect_loss == "mse"


def test_causal_text_passes_htr_kwargs(monkeypatch):
    captured = {}

    class DummyExtractor(nn.Module):
        output_dim = 7

        def forward(self, texts_or_batch):
            if isinstance(texts_or_batch, dict):
                n = len(texts_or_batch.get("texts", []))
            else:
                n = len(texts_or_batch)
            return torch.zeros(n, self.output_dim)

    def fake_create_feature_extractor(**kwargs):
        captured.update(kwargs)
        return DummyExtractor()

    monkeypatch.setattr(
        "oci.models.causal_text.create_feature_extractor",
        fake_create_feature_extractor,
    )
    from oci.models.causal_text import CausalText

    CausalText(
        feature_extractor_type="hierarchical_transformer",
        htr_sentence_model="hash",
        htr_projection_dim=7,
        htr_num_layers=1,
        htr_num_heads=1,
        htr_transformer_dim=8,
        htr_hash_embedding_dim=8,
        causal_head_representation_dim=8,
        causal_head_hidden_outcome_dim=4,
        device="cpu",
    )

    assert captured["htr_sentence_model"] == "hash"
    assert captured["htr_projection_dim"] == 7
    assert captured["htr_num_layers"] == 1


def test_applied_route_dispatches_dragonnet_drlearner(monkeypatch, tmp_path):
    called = {}

    def fake_run_dragonnet_drlearner(dataset, config, output_path, device, num_workers, gpu_ids):
        called["dataset_rows"] = len(dataset)
        called["model_type"] = config.architecture.model_type
        called["output_path"] = output_path

    monkeypatch.setattr(
        "oci.inference.dragonnet_drlearner.run_dragonnet_drlearner",
        fake_run_dragonnet_drlearner,
    )
    from oci.inference.applied import run_applied_inference

    df = pd.DataFrame(
        {
            "clinical_text": ["a", "b"],
            "outcome_indicator": [0, 1],
            "treatment_indicator": [1, 0],
        }
    )
    config = AppliedInferenceConfig(
        architecture=ModelArchitectureConfig(model_type="dragonnet_drlearner"),
        training=TrainingConfig(epochs=1, batch_size=2),
    )
    output_path = tmp_path / "predictions.parquet"

    run_applied_inference(
        dataset=df,
        config=config,
        output_path=output_path,
        device=torch.device("cpu"),
    )

    assert called == {
        "dataset_rows": 2,
        "model_type": "dragonnet_drlearner",
        "output_path": output_path,
    }


def test_crossfit_effect_contract_without_training(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "_oci_row_id": np.arange(6),
            "clinical_text": [f"note {i}" for i in range(6)],
            "outcome_indicator": [0, 1, 0, 1, 0, 1],
            "treatment_indicator": [0, 1, 1, 0, 1, 0],
        }
    )
    nuisance = pd.DataFrame(
        {
            "_oci_row_id": np.arange(6),
            "outer_fold": 1,
            "dr_pseudo_outcome": np.linspace(-0.3, 0.3, 6),
        }
    )
    config = AppliedInferenceConfig(
        architecture=ModelArchitectureConfig(
            model_type="dragonnet_drlearner",
            dragonnet_drlearner=DragonNetDRLearnerConfig(effect_folds=2),
        ),
        training=TrainingConfig(epochs=1, batch_size=2),
    )
    runner = DragonNetDRLearnerRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
    )
    class DummyTauModel:
        extractor = object()

    dummy_model = DummyTauModel()
    monkeypatch.setattr(runner, "_create_tau_model", lambda: dummy_model)
    monkeypatch.setattr(runner, "_train_tau_model", lambda **kwargs: [])
    monkeypatch.setattr(
        runner,
        "_predict_tau_model",
        lambda model, heldout_df: np.full(len(heldout_df), 0.25),
    )
    monkeypatch.setattr(runner, "_cleanup_model", lambda model: None)
    monkeypatch.setattr(
        runner,
        "_attention_evidence",
        lambda **kwargs: [{"stage": kwargs["stage"], "fold": kwargs["fold"]}],
    )

    result = runner.crossfit_effect(df, nuisance, outer_fold=1)

    preds = result["predictions"]
    assert np.all(np.isfinite(preds["tau_hat_r_stage"]))
    assert np.allclose(preds["pred_ite_prob"], 0.25)
    assert set(preds["effect_objective"]) == {"dr_pseudo_outcome_huber"}
    assert {row["stage"] for row in result["attention"]} == {"effect_modifier"}
