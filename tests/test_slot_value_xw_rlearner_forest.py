import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset


class FakeSentenceEncoder:
    def __init__(self, dim=6):
        self.dim = dim

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(
        self,
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        del batch_size, convert_to_numpy, show_progress_bar
        rows = []
        for text in texts:
            base = sum(ord(ch) for ch in str(text))
            row = np.array(
                [((base + 17 * i) % 97) / 97.0 for i in range(self.dim)],
                dtype=np.float32,
            )
            if normalize_embeddings:
                row = row / max(np.linalg.norm(row), 1e-12)
            rows.append(row)
        return np.vstack(rows)


def test_causal_text_forest_slot_value_separate_effect_extractor(monkeypatch):
    import oci.models.slot_value_discovery_extractor as extractor_mod
    from oci.models.causal_text_forest import CausalTextForest

    monkeypatch.setattr(
        extractor_mod,
        "load_sentence_transformer",
        lambda model_name, device=None: FakeSentenceEncoder(dim=6),
    )

    model = CausalTextForest(
        feature_extractor_type="slot_value_discovery",
        svx_cached_embedding_dim=6,
        svx_confounder_concepts=["patient age"],
        svx_effect_modifier_concepts=["PD-L1 expression"],
        svx_num_free_slots=2,
        svx_slot_dim=5,
        svx_num_value_prototypes=2,
        svx_anchor_weight=0.0,
        cf_use_rlearner_representation=True,
        cf_n_estimators=4,
        cf_inference=False,
        device="cpu",
    )

    assert model.effect_feature_extractor is not None
    assert model.effect_feature_extractor is not model.feature_extractor
    assert model.feature_extractor.output_dim == model.effect_feature_extractor.output_dim

    batch = {
        "texts": ["Age 72. PD-L1 >=50%.", "Age 55. PD-L1 <1%."],
        "cached_hidden_states": torch.randn(2, 4, 6),
        "cached_attention_mask": torch.ones(2, 4),
        "treatment": torch.tensor([1.0, 0.0]),
        "outcome": torch.tensor([1.0, 0.0]),
    }
    nuisance = model.train_nuisance_step(batch)
    effect = model.train_effect_r_step(
        batch,
        e_hat=torch.tensor([0.2, 0.8]),
        m_hat=torch.tensor([0.4, 0.3]),
        e_clip=0.05,
    )

    assert nuisance["loss"].ndim == 0
    assert effect["loss"].ndim == 0
    assert effect["r_loss"].ndim == 0


def test_causal_text_forest_slot_value_shared_rlearner_features(monkeypatch):
    import oci.models.slot_value_discovery_extractor as extractor_mod
    from oci.models.causal_text_forest import CausalTextForest

    monkeypatch.setattr(
        extractor_mod,
        "load_sentence_transformer",
        lambda model_name, device=None: FakeSentenceEncoder(dim=6),
    )

    model = CausalTextForest(
        feature_extractor_type="slot_value_discovery",
        svx_cached_embedding_dim=6,
        svx_confounder_concepts=["patient age"],
        svx_effect_modifier_concepts=["PD-L1 expression"],
        svx_num_free_slots=2,
        svx_slot_dim=5,
        svx_num_value_prototypes=2,
        svx_anchor_weight=0.0,
        cf_rlearner_representation_mode="shared_features",
        cf_n_estimators=4,
        cf_inference=False,
        device="cpu",
    )

    assert model.rlearner_representation_mode == "shared_features"
    assert model.effect_feature_extractor is None

    batch = {
        "texts": ["Age 72. PD-L1 >=50%.", "Age 55. PD-L1 <1%."],
        "cached_hidden_states": torch.randn(2, 4, 6),
        "cached_attention_mask": torch.ones(2, 4),
        "treatment": torch.tensor([1.0, 0.0]),
        "outcome": torch.tensor([1.0, 0.0]),
    }
    losses = model.train_representation_step(batch, gamma_rlearner=1.0)
    losses["loss"].backward()

    assert losses["r_loss"].ndim == 0
    assert model.feature_extractor._queries.grad is not None
    assert model.feature_extractor._queries.grad.abs().sum() > 0
    assert model.propensity_head.weight.grad is not None
    assert model.outcome_head.weight.grad is not None
    assert model.effect_tau_head.weight.grad is not None

    x_features, w_features, propensity, outcome = model.extract_forest_features(
        ["Age 72. PD-L1 >=50%.", "Age 55. PD-L1 <1%."],
        batch_size=2,
    )
    assert w_features is None
    assert x_features.shape == (2, model.feature_extractor.output_dim)
    assert propensity.shape == (2,)
    assert outcome.shape == (2,)


def test_causal_text_forest_shared_mode_fits_forest_without_w(monkeypatch):
    import oci.models.slot_value_discovery_extractor as extractor_mod
    from oci.models.causal_text_forest import CausalTextForest

    monkeypatch.setattr(
        extractor_mod,
        "load_sentence_transformer",
        lambda model_name, device=None: FakeSentenceEncoder(dim=6),
    )

    model = CausalTextForest(
        feature_extractor_type="slot_value_discovery",
        svx_cached_embedding_dim=6,
        svx_confounder_concepts=["patient age"],
        svx_effect_modifier_concepts=["PD-L1 expression"],
        svx_num_free_slots=1,
        svx_slot_dim=4,
        svx_num_value_prototypes=1,
        svx_anchor_weight=0.0,
        cf_rlearner_representation_mode="shared_features",
        cf_n_estimators=4,
        cf_inference=False,
        device="cpu",
    )

    captured = {}

    def fake_fit(X, T, Y, W=None, propensity=None, outcome_pred=None):
        del propensity, outcome_pred
        captured["X"] = X
        captured["W"] = W
        captured["T"] = T
        captured["Y"] = Y
        return model.causal_forest

    model.causal_forest.fit = fake_fit
    model.train_causal_forest(
        ["Age 72. PD-L1 >=50%.", "Age 55. PD-L1 <1%."],
        T=np.array([1.0, 0.0]),
        Y=np.array([1.0, 0.0]),
        batch_size=2,
    )

    assert captured["W"] is None
    assert captured["X"].shape == (2, model.feature_extractor.output_dim)
    assert captured["T"].tolist() == [1.0, 0.0]
    assert captured["Y"].tolist() == [1.0, 0.0]


def test_causal_text_forest_shared_mode_appends_explicit_features_to_x(monkeypatch):
    import oci.models.slot_value_discovery_extractor as extractor_mod
    from oci.config import ExplicitFeatureSpec
    from oci.models.causal_text_forest import CausalTextForest

    monkeypatch.setattr(
        extractor_mod,
        "load_sentence_transformer",
        lambda model_name, device=None: FakeSentenceEncoder(dim=6),
    )

    specs = [
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            roles=["confounder"],
        ),
        ExplicitFeatureSpec(
            name="marker",
            type="categorical",
            categories=["low", "high"],
            roles=["effect_modifier"],
        ),
    ]
    values = [
        {"age": 72.0, "age_missing": False, "marker": "high", "marker_missing": False},
        {"age": 55.0, "age_missing": False, "marker": "low", "marker_missing": False},
    ]
    model = CausalTextForest(
        feature_extractor_type="slot_value_discovery",
        svx_cached_embedding_dim=6,
        svx_confounder_concepts=["patient age"],
        svx_effect_modifier_concepts=["PD-L1 expression"],
        svx_num_free_slots=1,
        svx_slot_dim=4,
        svx_num_value_prototypes=1,
        svx_anchor_weight=0.0,
        cf_rlearner_representation_mode="shared_features",
        explicit_feature_specs=specs,
        explicit_feature_output_dim=3,
        cf_n_estimators=4,
        cf_inference=False,
        device="cpu",
    )
    model.fit_explicit_feature_featurizer(values)
    model.fit_explicit_features(values)

    x_features, w_features, _, _ = model.extract_forest_features(
        ["Age 72. PD-L1 >=50%.", "Age 55. PD-L1 <1%."],
        batch_size=2,
        explicit_feature_values=values,
    )

    assert w_features is None
    assert x_features.shape == (2, model.feature_extractor.output_dim + 4)


def test_xw_config_accepts_shared_mode_without_explicit_forest_mode():
    from oracle_experiment_scripts.run_oracle_xw_rlearner_forest_experiments import (
        XWRLearnerForestConfig,
    )

    config = XWRLearnerForestConfig(
        dataset_path="unused",
        dataset_name="unused",
        rlearner_mode="shared_features",
    )

    assert config.rlearner_mode == "shared_features"
    assert config.xw_feature_split is False
    assert config.cf_rlearner_representation_mode == "shared_features"


def test_shared_grid_uses_all_visible_gpus_by_default(monkeypatch):
    from oracle_experiment_scripts import run_slot_value_discovery_shared_rlearner_forest_grid as grid

    monkeypatch.setattr(grid.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(grid.torch.cuda, "device_count", lambda: 3)

    args = SimpleNamespace(device="cuda:0", devices=None)

    assert grid._resolve_run_devices(args) == ["cuda:0", "cuda:1", "cuda:2"]


def test_shared_grid_preserves_explicit_device_choices(monkeypatch):
    from oracle_experiment_scripts import run_slot_value_discovery_shared_rlearner_forest_grid as grid

    monkeypatch.setattr(grid.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(grid.torch.cuda, "device_count", lambda: 3)

    assert grid._resolve_run_devices(
        SimpleNamespace(device="cuda:0", devices=["cuda:1"])
    ) == ["cuda:1"]
    assert grid._resolve_run_devices(
        SimpleNamespace(device="cpu", devices=None)
    ) == ["cpu"]


class TinyEffectDataset(Dataset):
    def __init__(self, n: int):
        self.treatments = torch.tensor([1, 0, 1, 0, 1][:n], dtype=torch.float32)
        self.outcomes = torch.tensor([1, 0, 0, 1, 1][:n], dtype=torch.float32)
        self.data = pd.DataFrame(
            {
                "clinical_text": [f"sample {idx}" for idx in range(n)],
                "true_pdl1_expression": ["<1%", ">=50%", "1-49%", ">=50%", "<1%"][:n],
            }
        )

    def __len__(self):
        return len(self.treatments)

    def __getitem__(self, idx):
        return {
            "texts": self.data["clinical_text"].iloc[idx],
            "treatment": self.treatments[idx],
            "outcome": self.outcomes[idx],
            "text_id": idx,
        }


def tiny_collate(items):
    return {
        "texts": [item["texts"] for item in items],
        "treatment": torch.stack([item["treatment"] for item in items]),
        "outcome": torch.stack([item["outcome"] for item in items]),
        "text_id": [item["text_id"] for item in items],
    }


class TinyEffectModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.train_calls = 0

    def effect_parameters(self):
        return [self.weight]

    def train_effect_r_step(self, batch, e_hat, m_hat, gamma_rlearner=1.0, e_clip=0.01):
        self.train_calls += 1
        e_hat = e_hat.clamp(e_clip, 1.0 - e_clip)
        y_residual = batch["outcome"] - m_hat
        t_residual = batch["treatment"] - e_hat
        tau = self.weight.expand_as(y_residual)
        r_loss = ((y_residual - tau * t_residual) ** 2).mean()
        return {"loss": gamma_rlearner * r_loss, "r_loss": r_loss.detach()}


def _run_tiny_effect_stage(accumulation_steps: int):
    from oracle_experiment_scripts.run_oracle_xw_rlearner_forest_experiments import (
        XWRLearnerForestConfig,
        _train_effect_stage,
    )

    dataset = TinyEffectDataset(n=5)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=tiny_collate)
    model = TinyEffectModel()
    config = XWRLearnerForestConfig(
        dataset_path="unused",
        dataset_name="unused",
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        rlearner_effect_batch_size=2,
        rlearner_effect_accumulation_steps=accumulation_steps,
    )
    diagnostics = _train_effect_stage(
        model=model,
        train_loader=loader,
        nuisance_propensity=np.full(len(dataset), 0.5, dtype=np.float32),
        nuisance_outcome=np.full(len(dataset), 0.5, dtype=np.float32),
        config=config,
        device=torch.device("cpu"),
        use_cached=False,
        gpu_store=None,
    )
    return model, diagnostics


def test_effect_stage_accumulation_steps_final_partial():
    model, diagnostics = _run_tiny_effect_stage(accumulation_steps=2)

    assert model.train_calls == 3
    assert diagnostics["effect_optimizer_steps"] == 2
    assert diagnostics["effect_physical_batch_size"] == 2
    assert diagnostics["effect_effective_batch_size"] == 4
    assert diagnostics["effect_pdl1_cell_summary"]["num_batches"] == 3


def test_effect_stage_default_accumulation_steps_every_batch():
    model, diagnostics = _run_tiny_effect_stage(accumulation_steps=1)

    assert model.train_calls == 3
    assert diagnostics["effect_optimizer_steps"] == 3
    assert diagnostics["effect_effective_batch_size"] == 2
