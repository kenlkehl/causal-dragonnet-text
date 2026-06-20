import json
import logging
import os
import threading

import numpy as np
import pandas as pd
import pytest
import torch

from oci.config import (
    AgenticAttentionVariableForestConfig,
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ModelArchitectureConfig,
    TrainingConfig,
)
from oci.inference.agentic_attention_variable_forest import (
    AgenticAttentionVariableForestRunner,
    consensus_feature_specs,
    _JointRNet,
    _residual_contrastive_label_frame,
    _tail_attention_positions,
    _validate_consensus_disambiguation_response,
    _make_linear_lr_scheduler,
    _run_crossfit_fold_tasks,
    run_agentic_attention_variable_forest,
)
from oci.inference.agentic_explicit_feature_forest import build_agent_prompt
from oci.models import ECONML_AVAILABLE


class FakeAttentionAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context["stage"] == "confounder":
            return [
                {
                    "action": "add",
                    "name": "age_group",
                    "type": "categorical",
                    "categories": ["younger", "older"],
                    "roles": ["confounder"],
                    "description": "Patient age group before treatment",
                }
            ]
        return [
            {
                "action": "add",
                "name": "mutation_status",
                "type": "categorical",
                "categories": ["negative", "positive"],
                "roles": ["effect_modifier"],
                "description": "Presence of a targetable mutation before treatment",
            }
        ]


class FakeExtractionProvider:
    def ensure_features(self, dataset, specs):
        dataset = dataset.copy()
        for spec in specs:
            col = f"explicit_feat_{spec.name}"
            miss_col = f"{col}_missing"
            if spec.name == "age_group":
                dataset[col] = np.where(dataset["clinical_text"].str.contains("older"), "older", "younger")
            elif spec.name == "mutation_status":
                dataset[col] = np.where(
                    dataset["clinical_text"].str.contains("mutation positive"),
                    "positive",
                    "negative",
                )
            else:
                dataset[col] = spec.categories[0] if spec.type == "categorical" else 0.0
            dataset[miss_col] = False
        return dataset


class DummyTensorExtractor(torch.nn.Module):
    output_dim = 3

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 3, bias=False)

    def forward(self, values):
        if isinstance(values, dict):
            features = values["features"]
        else:
            features = torch.stack(list(values)).float()
        return self.projection(features)


def test_consensus_feature_specs_requires_fold_recurrence():
    proposals = {
        1: [
            ExplicitFeatureSpec(
                name="Age Group",
                type="categorical",
                categories=["younger", "older"],
                roles=["confounder"],
            )
        ],
        2: [
            ExplicitFeatureSpec(
                name="age_group",
                type="categorical",
                categories=["younger", "older"],
                roles=["confounder"],
            )
        ],
        3: [
            ExplicitFeatureSpec(
                name="rare_marker",
                type="categorical",
                categories=["no", "yes"],
                roles=["confounder"],
            )
        ],
    }

    selected = consensus_feature_specs(
        proposals,
        min_fold_fraction=2 / 3,
        required_role="confounder",
        min_folds=2,
    )

    assert [spec.name for spec in selected] == ["age_group"]


def test_joint_rlearner_r_loss_detaches_nuisance_heads():
    torch.manual_seed(0)
    model = _JointRNet(
        extractor=DummyTensorExtractor(),
        hidden_dim=5,
        outcome_type="binary",
    )
    inputs = [torch.randn(3) for _ in range(4)]
    treatment = torch.tensor([0.0, 1.0, 0.0, 1.0])
    outcome = torch.tensor([0.0, 1.0, 1.0, 0.0])

    propensity_logit, outcome_logit, tau = model(inputs)
    e_hat = torch.sigmoid(propensity_logit).detach().clamp(0.01, 0.99)
    m_hat = torch.sigmoid(outcome_logit).detach()
    r_loss = ((outcome - m_hat) - tau * (treatment - e_hat)).square().mean()
    r_loss.backward()

    assert model.extractor.projection.weight.grad is not None
    assert model.extractor.projection.weight.grad.abs().sum() > 0
    effect_grads = [
        param.grad
        for param in model.effect_head.parameters()
        if param.requires_grad
    ]
    assert all(grad is not None and grad.abs().sum() > 0 for grad in effect_grads)
    nuisance_params = list(model.nuisance_shared.parameters())
    nuisance_params += list(model.propensity.parameters())
    nuisance_params += list(model.outcome.parameters())
    assert all(param.grad is None for param in nuisance_params)


def test_consensus_feature_specs_uses_agentic_alias_groups():
    proposals = {
        1: [
            ExplicitFeatureSpec(
                name="patient_age",
                type="continuous",
                roles=["confounder"],
                description="Patient age before treatment",
            )
        ],
        2: [
            ExplicitFeatureSpec(
                name="age_at_diagnosis",
                type="continuous",
                roles=["confounder"],
                description="Age when the baseline diagnosis was documented",
            )
        ],
        3: [
            ExplicitFeatureSpec(
                name="tumor_stage",
                type="categorical",
                categories=["early", "advanced"],
                roles=["confounder"],
            )
        ],
        4: [
            ExplicitFeatureSpec(
                name="baseline_age",
                type="continuous",
                roles=["confounder"],
                description="Baseline age before treatment",
            )
        ],
        5: [
            ExplicitFeatureSpec(
                name="rare_marker",
                type="categorical",
                categories=["absent", "present"],
                roles=["confounder"],
            )
        ],
    }
    raw_response = {
        "groups": [
            {
                "canonical_name": "patient_age",
                "member_names": [
                    "patient_age",
                    "age_at_diagnosis",
                    "baseline_age",
                ],
                "member_folds": [1, 2, 4],
                "type": "continuous",
                "description": "Patient age at or before baseline diagnosis",
                "rationale": "All names extract the patient's baseline age.",
            },
            {
                "canonical_name": "rare_marker",
                "member_names": ["rare_marker"],
                "member_folds": [5],
                "type": "categorical",
                "categories": ["absent", "present"],
            },
        ],
        "unmerged": [],
    }

    groups, errors = _validate_consensus_disambiguation_response(
        raw_response,
        proposals_by_fold=proposals,
        required_role="confounder",
    )
    selected = consensus_feature_specs(
        proposals,
        min_fold_fraction=2 / 3,
        min_folds=2,
        required_role="confounder",
        concept_groups=groups,
    )

    assert [spec.name for spec in selected] == ["patient_age"]
    assert selected[0].type == "continuous"
    assert any("at least 2 distinct folds" in error for error in errors)


def test_consensus_disambiguation_rejects_unproposed_and_conflicting_groups():
    proposals = {
        1: [
            ExplicitFeatureSpec(
                name="patient_age",
                type="continuous",
                roles=["confounder"],
            )
        ],
        2: [
            ExplicitFeatureSpec(
                name="baseline_age_bucket",
                type="categorical",
                categories=["younger", "older"],
                roles=["confounder"],
            )
        ],
        3: [
            ExplicitFeatureSpec(
                name="smoking_status",
                type="categorical",
                categories=["never", "ever"],
                roles=["confounder"],
            )
        ],
        4: [
            ExplicitFeatureSpec(
                name="tobacco_use",
                type="categorical",
                categories=["absent", "present"],
                roles=["confounder"],
            )
        ],
    }
    raw_response = {
        "groups": [
            {
                "canonical_name": "patient_age",
                "member_names": ["patient_age", "invented_age"],
                "member_folds": [1, 2],
                "type": "continuous",
            },
            {
                "canonical_name": "patient_age",
                "member_names": ["patient_age", "baseline_age_bucket"],
                "member_folds": [1, 2],
                "type": "continuous",
            },
            {
                "canonical_name": "smoking_status",
                "member_names": ["smoking_status", "tobacco_use"],
                "member_folds": [3, 4],
                "type": "categorical",
                "categories": ["never", "ever"],
            },
        ]
    }

    groups, errors = _validate_consensus_disambiguation_response(
        raw_response,
        proposals_by_fold=proposals,
        required_role="confounder",
    )

    assert groups == []
    assert any("were not proposed" in error for error in errors)
    assert any("conflicting types" in error for error in errors)
    assert any("incompatible categories" in error for error in errors)


def test_config_parses_agentic_attention_variable_forest_block(tmp_path):
    from oci.config import ExperimentConfig, normalize_feature_extractor_type

    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "clinical_text": ["a", "b"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [0, 1],
        }
    ).to_parquet(dataset_path, index=False)

    config = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": str(dataset_path),
                "architecture": {
                    "model_type": "agentic_attention_variable_forest",
                    "feature_extractor_type": "htr",
                    "agentic_attention_variable_forest": {
                        "nuisance_folds": 2,
                        "effect_folds": 2,
                        "fold_parallelism": "1",
                    },
                },
            }
        }
    )

    assert normalize_feature_extractor_type("htr") == "hierarchical_transformer"
    assert (
        config.applied_inference.architecture
        .agentic_attention_variable_forest
        .nuisance_folds
        == 2
    )
    avf = config.applied_inference.architecture.agentic_attention_variable_forest
    assert avf.nuisance_epochs == 20
    assert avf.nuisance_weight_decay == pytest.approx(0.05)
    assert avf.nuisance_label_smoothing == pytest.approx(0.02)
    assert avf.nuisance_calibration == "temperature_isotonic"
    assert avf.effect_objective == "squared_r_loss"
    assert avf.neural_stage_mode == "staged"
    assert avf.joint_rlearner_gamma == pytest.approx(1.0)


def test_agentic_attention_config_accepts_logistic_effect_objective(tmp_path):
    from oci.config import ExperimentConfig

    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "clinical_text": ["a", "b"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [0, 1],
        }
    ).to_parquet(dataset_path, index=False)

    config = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": str(dataset_path),
                "architecture": {
                    "model_type": "agentic_attention_variable_forest",
                    "agentic_attention_variable_forest": {
                        "nuisance_folds": 2,
                        "effect_folds": 2,
                        "effect_objective": "logistic_r_loss",
                    },
                },
            }
        }
    )

    avf = config.applied_inference.architecture.agentic_attention_variable_forest
    assert avf.effect_objective == "logistic_r_loss"


def test_agentic_attention_config_accepts_joint_rlearner_mode(tmp_path):
    from oci.config import ExperimentConfig

    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "clinical_text": ["a", "b"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [0, 1],
        }
    ).to_parquet(dataset_path, index=False)

    config = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": str(dataset_path),
                "architecture": {
                    "model_type": "agentic_attention_variable_forest",
                    "agentic_attention_variable_forest": {
                        "nuisance_folds": 2,
                        "effect_folds": 2,
                        "neural_stage_mode": "joint_rlearner",
                        "joint_rlearner_gamma": "0.5",
                    },
                },
            }
        }
    )

    avf = config.applied_inference.architecture.agentic_attention_variable_forest
    assert avf.neural_stage_mode == "joint_rlearner"
    assert avf.joint_rlearner_gamma == pytest.approx(0.5)


def test_linear_lr_scheduler_spans_fold_training_steps():
    param = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    train_config = TrainingConfig(epochs=2, lr_schedule="linear")
    scheduler = _make_linear_lr_scheduler(optimizer, train_config, steps_per_epoch=3)

    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

    for _ in range(6):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_linear_lr_scheduler_accepts_epoch_override():
    param = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    train_config = TrainingConfig(epochs=50, lr_schedule="linear")
    scheduler = _make_linear_lr_scheduler(
        optimizer,
        train_config,
        steps_per_epoch=2,
        epochs_override=3,
    )

    assert scheduler is not None
    for _ in range(6):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_crossfit_fold_tasks_use_in_process_threads():
    split_items = [
        (1, (np.array([0, 1]), np.array([2]))),
        (2, (np.array([2, 3]), np.array([0]))),
    ]
    parent_pid = os.getpid()
    main_thread = threading.get_ident()

    def run_fold(fold, fit_pos, heldout_pos):
        return {
            "fold": fold,
            "pid": os.getpid(),
            "thread": threading.get_ident(),
            "fit": fit_pos.tolist(),
            "heldout": heldout_pos.tolist(),
        }

    results = _run_crossfit_fold_tasks(run_fold, split_items, n_jobs=2)

    assert [row["fold"] for row in results] == [1, 2]
    assert {row["pid"] for row in results} == {parent_pid}
    assert all(row["thread"] != main_thread for row in results)


def test_nuisance_crossfit_logs_heldout_aurocs(tmp_path, monkeypatch, caplog):
    df = pd.DataFrame(
        {
            "clinical_text": [f"patient {i}" for i in range(8)],
            "treatment_indicator": [0, 1, 0, 1, 0, 1, 0, 1],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=0,
        clinical_question="Compare treatment A versus B.",
        architecture=ModelArchitectureConfig(
            model_type="agentic_attention_variable_forest",
            feature_extractor_type="hierarchical_transformer",
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                fold_parallelism="1",
            ),
        ),
        training=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-3),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=False, features=[]),
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    class TinyExtractor(torch.nn.Module):
        output_dim = 2

        def forward(self, texts):
            return torch.zeros(len(texts), self.output_dim)

    def predict_nuisance(model, heldout):
        del model
        t = heldout[config.treatment_column].to_numpy(dtype=float)
        y = heldout[config.outcome_column].to_numpy(dtype=float)
        return 0.1 + 0.8 * t, 0.1 + 0.8 * y

    monkeypatch.setattr(runner, "_create_extractor", lambda: TinyExtractor())
    monkeypatch.setattr(runner, "_train_nuisance_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_predict_nuisance_model", predict_nuisance)
    monkeypatch.setattr(runner, "_attention_evidence", lambda *args, **kwargs: [])

    caplog.set_level(
        logging.INFO,
        logger="oci.inference.agentic_attention_variable_forest",
    )
    runner._crossfit_nuisance(runner.dataset, outer_fold=1)

    assert "heldout metrics: propensity_auroc=1.0000" in caplog.text
    assert "outcome_auroc=1.0000" in caplog.text
    assert "propensity_ece=" in caplog.text


def test_effect_crossfit_filters_training_rows_by_propensity(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "clinical_text": [f"patient {i}" for i in range(10)],
            "treatment_indicator": [0, 1] * 5,
            "outcome_indicator": [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="agentic_attention_variable_forest",
            feature_extractor_type="hierarchical_transformer",
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                fold_parallelism="1",
                r_stage_min_propensity=0.2,
                r_stage_max_propensity=0.8,
            ),
        ),
        training=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-3),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=False, features=[]),
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    class TinyExtractor(torch.nn.Module):
        output_dim = 2

        def forward(self, texts):
            return torch.zeros(len(texts), self.output_dim)

    nuisance_predictions = pd.DataFrame(
        {
            "_oci_row_id": runner.dataset["_oci_row_id"].to_numpy(),
            "outer_fold": 1,
            "e_hat": [0.05, 0.2, 0.4, 0.85, 0.7, 0.95, 0.3, 0.8, 0.1, 0.6],
            "m_hat": np.full(len(runner.dataset), 0.5),
            "y_residual": runner.dataset["outcome_indicator"].to_numpy(dtype=float) - 0.5,
            "t_residual": runner.dataset["treatment_indicator"].to_numpy(dtype=float) - 0.5,
            "r_loss_at_zero_tau": 0.25,
            "nuisance_fold": 1,
        }
    )
    train_positions = []

    def record_effect_train(model, data, positions, *args, **kwargs):
        del model, data, args, kwargs
        train_positions.extend(int(pos) for pos in positions)

    monkeypatch.setattr(runner, "_create_extractor", lambda: TinyExtractor())
    monkeypatch.setattr(runner, "_train_effect_model", record_effect_train)
    monkeypatch.setattr(
        runner,
        "_predict_effect_model",
        lambda model, heldout: np.full(len(heldout), 0.25),
    )
    monkeypatch.setattr(runner, "_attention_evidence", lambda *args, **kwargs: [])

    result = runner._crossfit_effect(
        runner.dataset,
        nuisance_predictions,
        outer_fold=1,
    )

    e_hat = nuisance_predictions["e_hat"].to_numpy()
    assert train_positions
    assert all(0.2 <= e_hat[pos] <= 0.8 for pos in train_positions)
    assert not any(e_hat[pos] < 0.2 or e_hat[pos] > 0.8 for pos in train_positions)
    assert "r_stage_train_eligible" in result["predictions"].columns
    assert result["predictions"]["r_stage_train_eligible"].tolist() == [
        False,
        True,
        True,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
    ]


def test_logistic_effect_crossfit_reports_probability_scale_tau(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "clinical_text": [f"patient {i}" for i in range(8)],
            "treatment_indicator": [0, 1] * 4,
            "outcome_indicator": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            model_type="agentic_attention_variable_forest",
            feature_extractor_type="hierarchical_transformer",
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                fold_parallelism="1",
                effect_objective="logistic_r_loss",
            ),
        ),
        training=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-3),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=False, features=[]),
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    class TinyExtractor(torch.nn.Module):
        output_dim = 2

        def forward(self, texts):
            return torch.zeros(len(texts), self.output_dim)

    nuisance_predictions = pd.DataFrame(
        {
            "_oci_row_id": runner.dataset["_oci_row_id"].to_numpy(),
            "outer_fold": 1,
            "e_hat": np.full(len(runner.dataset), 0.5),
            "m_hat": np.full(len(runner.dataset), 0.5),
            "y_residual": runner.dataset["outcome_indicator"].to_numpy(dtype=float) - 0.5,
            "t_residual": runner.dataset["treatment_indicator"].to_numpy(dtype=float) - 0.5,
            "r_loss_at_zero_tau": 0.25,
            "nuisance_fold": 1,
        }
    )

    monkeypatch.setattr(runner, "_create_extractor", lambda: TinyExtractor())
    monkeypatch.setattr(runner, "_train_effect_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "_predict_effect_model",
        lambda model, heldout: np.ones(len(heldout)),
    )
    monkeypatch.setattr(runner, "_attention_evidence", lambda *args, **kwargs: [])

    result = runner._crossfit_effect(
        runner.dataset,
        nuisance_predictions,
        outer_fold=1,
    )

    predictions = result["predictions"]
    expected_tau = 1.0 / (1.0 + np.exp(-0.5)) - 1.0 / (1.0 + np.exp(0.5))
    assert predictions["effect_objective"].unique().tolist() == ["logistic_r_loss"]
    assert predictions["tau_logit_modifier"].to_numpy() == pytest.approx(np.ones(len(df)))
    assert predictions["tau_hat_r_stage"].to_numpy() == pytest.approx(
        np.full(len(df), expected_tau)
    )
    assert predictions["effect_loss"].notna().all()
    assert predictions["effect_loss_at_zero_tau"].notna().all()


def test_residual_contrastive_label_frame_builds_tail_vs_neutral_labels():
    scores = np.array([-4.0, -2.0, -0.1, 0.0, 0.1, 2.0, 4.0])
    nuisance_predictions = pd.DataFrame(
        {
            "_oci_row_id": np.arange(len(scores)),
            "outer_fold": 1,
            "e_hat": np.full(len(scores), 0.5),
            "m_hat": np.full(len(scores), 0.5),
            "y_residual": scores,
            "t_residual": np.ones(len(scores)),
            "r_loss_at_zero_tau": scores**2,
            "nuisance_fold": 1,
        }
    )

    labeled = _residual_contrastive_label_frame(
        nuisance_predictions,
        score_name="r_score",
        high_quantile=0.75,
        low_quantile=0.25,
        neutral_abs_quantile=0.30,
    )

    assert labeled["residual_contrastive_group"].tolist() == [
        "low",
        "low",
        "neutral",
        "neutral",
        "neutral",
        "high",
        "high",
    ]
    high_labels = labeled["residual_contrastive_high_vs_neutral_label"].to_numpy()
    low_labels = labeled["residual_contrastive_low_vs_neutral_label"].to_numpy()
    assert np.isnan(high_labels[:2]).all()
    assert high_labels[2:].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert low_labels[:5].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert np.isnan(low_labels[5:]).all()


def test_tail_attention_positions_prefers_positive_tail_by_probability():
    selected = _tail_attention_positions(
        heldout_pos=np.array([0, 1, 2, 3]),
        labels=np.array([1.0, 0.0, 1.0, np.nan]),
        probs=np.array([0.2, 0.9, 0.8, 0.7]),
        max_rows=2,
    )

    assert selected.tolist() == [2, 0]


def test_residual_contrastive_crossfit_outputs_predictions_and_attention(
    tmp_path,
    monkeypatch,
):
    n = 40
    df = pd.DataFrame(
        {
            "clinical_text": [f"patient residual note {i}" for i in range(n)],
            "treatment_indicator": [0, 1] * (n // 2),
            "outcome_indicator": [0, 1, 1, 0] * (n // 4),
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                fold_parallelism="1",
                residual_contrastive_enabled=True,
                residual_contrastive_min_class_count=1,
            ),
        ),
        training=TrainingConfig(epochs=1, batch_size=4, effect_batch_size=4),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=False, features=[]),
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    class TinyExtractor(torch.nn.Module):
        output_dim = 2

        def fit_tokenizer(self, texts):
            del texts

        def forward(self, texts):
            return torch.zeros(len(texts), self.output_dim)

    scores = np.linspace(-2.0, 2.0, n)
    nuisance_predictions = pd.DataFrame(
        {
            "_oci_row_id": runner.dataset["_oci_row_id"].to_numpy(),
            "outer_fold": 1,
            "e_hat": np.full(n, 0.5),
            "m_hat": np.full(n, 0.5),
            "y_residual": scores,
            "t_residual": np.ones(n),
            "r_loss_at_zero_tau": scores**2,
            "nuisance_fold": 1,
        }
    )
    train_calls = []

    def record_train(model, data, positions, labels, contrast_tail, *args, **kwargs):
        del model, data, labels, args, kwargs
        train_calls.append((contrast_tail, len(positions)))

    def fake_predict(model, heldout):
        del model
        return np.linspace(-1.0, 1.0, len(heldout))

    def fake_attention(extractor, heldout, fold, outer_fold, stage, extra):
        del extractor
        rows = []
        for idx, row_id in enumerate(heldout["_oci_row_id"].tolist()):
            rows.append(
                {
                    "row_id": int(row_id),
                    "fold": int(fold),
                    "outer_fold": int(outer_fold),
                    "stage": stage,
                    "chunk_text": "tail evidence",
                    "attention": 1.0,
                    "contrastive_tail": extra["contrastive_tail"][idx],
                    "contrastive_prob": float(extra["contrastive_prob"][idx]),
                }
            )
        return rows

    monkeypatch.setattr(runner, "_create_extractor", lambda: TinyExtractor())
    monkeypatch.setattr(runner, "_train_residual_contrastive_model", record_train)
    monkeypatch.setattr(runner, "_predict_residual_contrastive_model", fake_predict)
    monkeypatch.setattr(runner, "_attention_evidence", fake_attention)

    result = runner._crossfit_residual_contrastive(
        runner.dataset,
        nuisance_predictions,
        outer_fold=1,
    )

    preds = result["predictions"]
    assert {"high", "low"} <= {tail for tail, _ in train_calls}
    assert preds["residual_contrastive_high_prob"].notna().any()
    assert preds["residual_contrastive_low_prob"].notna().any()
    assert "residual_contrastive_high_vs_neutral_auroc" in result["metrics"]
    assert {
        "residual_contrastive_high",
        "residual_contrastive_low",
    } <= {row["stage"] for row in result["attention"]}
    assert (
        tmp_path
        / "agentic_attention_variable_forest"
        / "crossfit_fold_checkpoints"
        / "residual_contrastive"
        / "outer_001_fold_001.done.json"
    ).exists()


def test_nuisance_crossfit_resumes_from_fold_checkpoints(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "clinical_text": [f"patient {i}" for i in range(8)],
            "treatment_indicator": [0, 1, 0, 1, 0, 1, 0, 1],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                fold_parallelism="1",
            ),
        ),
        training=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-3),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=False, features=[]),
    )
    output_path = tmp_path / "predictions.parquet"
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=output_path,
        device=torch.device("cpu"),
        num_workers=0,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    class TinyExtractor(torch.nn.Module):
        output_dim = 2

        def forward(self, texts):
            return torch.zeros(len(texts), self.output_dim)

    train_calls = []

    def predict_nuisance(model, heldout):
        del model
        t = heldout[config.treatment_column].to_numpy(dtype=float)
        y = heldout[config.outcome_column].to_numpy(dtype=float)
        return 0.1 + 0.8 * t, 0.2 + 0.6 * y

    def attention_rows(extractor, heldout, fold, outer_fold, stage, extra):
        del extractor, extra
        return [
            {
                "row_id": int(row_id),
                "fold": int(fold),
                "outer_fold": int(outer_fold),
                "stage": stage,
                "chunk_text": "cached",
                "attention": 1.0,
            }
            for row_id in heldout["_oci_row_id"].tolist()
        ]

    monkeypatch.setattr(runner, "_create_extractor", lambda: TinyExtractor())
    monkeypatch.setattr(runner, "_train_nuisance_model", lambda *args, **kwargs: train_calls.append(1))
    monkeypatch.setattr(runner, "_predict_nuisance_model", predict_nuisance)
    monkeypatch.setattr(runner, "_attention_evidence", attention_rows)

    first = runner._crossfit_nuisance(runner.dataset, outer_fold=1)
    assert len(train_calls) == 2
    assert (
        output_path.parent
        / "agentic_attention_variable_forest"
        / "crossfit_fold_checkpoints"
        / "nuisance"
        / "outer_001_fold_001.done.json"
    ).exists()

    resumed = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=output_path,
        device=torch.device("cpu"),
        num_workers=0,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("cached nuisance fold should not retrain")

    monkeypatch.setattr(resumed, "_create_extractor", fail_if_called)
    monkeypatch.setattr(resumed, "_train_nuisance_model", fail_if_called)
    second = resumed._crossfit_nuisance(resumed.dataset, outer_fold=1)

    pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
    assert first["attention"] == second["attention"]


def test_r_stage_crossfit_resumes_from_fold_checkpoints(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "clinical_text": [f"patient {i}" for i in range(8)],
            "treatment_indicator": [0, 1, 0, 1, 0, 1, 0, 1],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=0,
        architecture=ModelArchitectureConfig(
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                fold_parallelism="1",
            ),
        ),
        training=TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-3),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=False, features=[]),
    )
    output_path = tmp_path / "predictions.parquet"
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=output_path,
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )
    nuisance_predictions = pd.DataFrame(
        {
            "_oci_row_id": runner.dataset["_oci_row_id"].to_numpy(),
            "outer_fold": 1,
            "e_hat": np.full(len(runner.dataset), 0.5),
            "m_hat": np.full(len(runner.dataset), 0.4),
            "y_residual": np.nan,
            "t_residual": np.nan,
            "r_loss_at_zero_tau": np.nan,
            "nuisance_fold": 1,
        }
    )

    class TinyExtractor(torch.nn.Module):
        output_dim = 2

        def forward(self, texts):
            return torch.zeros(len(texts), self.output_dim)

    train_calls = []

    def predict_effect(model, heldout):
        del model
        return np.linspace(0.1, 0.2, len(heldout))

    def attention_rows(extractor, heldout, fold, outer_fold, stage, extra):
        del extractor, extra
        return [
            {
                "row_id": int(row_id),
                "fold": int(fold),
                "outer_fold": int(outer_fold),
                "stage": stage,
                "chunk_text": "cached",
                "attention": 1.0,
            }
            for row_id in heldout["_oci_row_id"].tolist()
        ]

    monkeypatch.setattr(runner, "_create_extractor", lambda: TinyExtractor())
    monkeypatch.setattr(runner, "_train_effect_model", lambda *args, **kwargs: train_calls.append(1))
    monkeypatch.setattr(runner, "_predict_effect_model", predict_effect)
    monkeypatch.setattr(runner, "_attention_evidence", attention_rows)

    first = runner._crossfit_effect(runner.dataset, nuisance_predictions, outer_fold=1)
    assert len(train_calls) == 2
    assert (
        output_path.parent
        / "agentic_attention_variable_forest"
        / "crossfit_fold_checkpoints"
        / "r_stage"
        / "outer_001_fold_001.done.json"
    ).exists()

    resumed = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=output_path,
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("cached R-stage fold should not retrain")

    monkeypatch.setattr(resumed, "_create_extractor", fail_if_called)
    monkeypatch.setattr(resumed, "_train_effect_model", fail_if_called)
    second = resumed._crossfit_effect(resumed.dataset, nuisance_predictions, outer_fold=1)

    pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
    assert first["attention"] == second["attention"]


def test_agent_candidate_output_is_saved_during_discovery(tmp_path):
    class TraceAgent:
        last_raw_response = "{\"proposals\": []}"
        last_response_trace = {"raw_content": "{\"proposals\": []}"}

        def propose(self, context):
            assert context["attention_evidence"][0]["evidence_snippet"] == "important note"
            assert "chunk_text" not in context["attention_evidence"][0]
            return [
                {
                    "action": "add",
                    "name": "baseline_marker",
                    "type": "continuous",
                    "roles": ["confounder"],
                    "description": "Baseline marker before treatment",
                    "rationale": "important note repeatedly mentions the marker",
                }
            ]

    df = pd.DataFrame(
        {
            "clinical_text": ["note a", "note b"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [0, 1],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        architecture=ModelArchitectureConfig(
            agentic_feature_search=AgenticFeatureSearchConfig(
                save_agent_context=True,
                save_agent_raw_output=True,
            ),
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                consensus_min_fold_fraction=1.0,
            ),
        ),
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=TraceAgent(),
        extraction_provider=FakeExtractionProvider(),
    )
    attention_rows = [
        {
            "row_id": int(runner.dataset.loc[0, "_oci_row_id"]),
            "fold": 1,
            "stage": "nuisance",
            "chunk_text": "important note",
            "attention": 0.9,
            "e_hat": 0.2,
            "m_hat": 0.3,
        }
    ]

    selected = runner._discover_variables_from_attention(
        stage="confounder",
        outer_fold=1,
        discovery_df=runner.dataset,
        attention_rows=attention_rows,
        existing_specs=[],
    )

    assert [spec.name for spec in selected] == ["baseline_marker"]
    checkpoint = (
        tmp_path
        / "agentic_attention_variable_forest"
        / "agent_candidate_checkpoints"
        / "confounder"
        / "outer_001_fold_001.json"
    )
    payload = json.loads(checkpoint.read_text())
    assert payload["status"] == "complete"
    assert payload["context"]["attention_evidence"][0]["evidence_snippet"] == "important note"
    assert "chunk_text" not in payload["context"]["attention_evidence"][0]
    assert payload["proposals"][0]["rationale"] == "important note repeatedly mentions the marker"
    assert payload["agent_raw_output"]["raw_content"] == "{\"proposals\": []}"
    jsonl_path = (
        tmp_path
        / "agentic_attention_variable_forest"
        / "confounder_candidates_by_fold.jsonl"
    )
    jsonl_payload = json.loads(jsonl_path.read_text().splitlines()[0])
    assert jsonl_payload["context"]["attention_evidence"][0]["evidence_snippet"] == "important note"
    assert "chunk_text" not in jsonl_payload["context"]["attention_evidence"][0]
    assert jsonl_payload["proposals"][0]["rationale"] == "important note repeatedly mentions the marker"


def test_attention_agent_prompt_is_attention_anchored():
    prompt = build_agent_prompt(
        {
            "prompt_version": "agentic_attention_variable_forest_v1",
            "stage": "confounder",
            "max_proposals": 2,
            "current_features": [],
            "excluded_feature_names": ["low_coverage_marker"],
            "rejected_low_coverage_features": [
                {"name": "low_coverage_marker", "coverage": 0.1}
            ],
            "attention_evidence": [
                {
                    "row_id": 1,
                    "evidence_snippet": "Age 78 years at treatment start.",
                    "attention": 0.9,
                }
            ],
        },
        AgenticFeatureSearchConfig(max_additions_per_iter=6),
    )

    assert "downstream clinical prediction task" in prompt
    assert "highly attended token spans inside highly attended clinical text snippets" in prompt
    assert "mundane patient-level fields count" in prompt
    assert "evidence_snippet" in prompt
    assert '"roles"' not in prompt
    assert "confounders:" not in prompt
    assert "effect modifiers:" not in prompt
    assert "At most 2 add proposals" in prompt
    assert "low_coverage_marker" in prompt


def test_consensus_disambiguation_prompt_is_alias_only():
    prompt = build_agent_prompt(
        {
            "prompt_version": "agentic_attention_consensus_disambiguation_v1",
            "stage": "confounder",
            "consensus_threshold": 2,
            "proposed_variables_by_fold": [
                {
                    "fold": 1,
                    "proposals": [
                        {
                            "name": "patient_age",
                            "type": "continuous",
                            "description": "Patient age",
                            "roles": ["confounder"],
                        }
                    ],
                },
                {
                    "fold": 2,
                    "proposals": [
                        {
                            "name": "baseline_age",
                            "type": "continuous",
                            "description": "Baseline age",
                            "roles": ["confounder"],
                        }
                    ],
                },
            ],
        },
        AgenticFeatureSearchConfig(),
    )

    assert "merge aliases only" in prompt
    assert '"groups"' in prompt
    assert "Use only names that appear in proposed_variables_by_fold" in prompt


def test_attention_agent_context_is_compact_and_filters_blank_chunks(tmp_path):
    df = pd.DataFrame(
        {
            "clinical_text": ["note a", "note b"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [0, 1],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        architecture=ModelArchitectureConfig(
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                attention_top_k_chunks=5,
            ),
        ),
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    chunk = "Patient has baseline age 78 years before treatment. " * 40
    start = chunk.find("age")
    token_spans = json.dumps(
        [
            {
                "text": "baseline age 78 years",
                "focus_token": "age",
                "salience": 0.123456789,
                "token_attention": 0.987654321,
                "char_start": start - len("baseline "),
                "char_end": start + len("age 78 years"),
            }
        ]
    )
    attention_rows = [
        {"row_id": 0, "chunk_text": "   ", "attention": 100.0},
        *[
            {
                "row_id": i,
                "chunk_index": 0,
                "chunk_text": chunk,
                "attention": float(i),
                "top_token_spans_json": token_spans,
                "attended_token_summary": "baseline age 78 years",
            }
            for i in range(1, 101)
        ],
    ]

    context = runner._build_agent_context(
        stage="confounder",
        outer_fold=1,
        inner_fold=1,
        discovery_df=runner.dataset,
        attention_rows=attention_rows,
        existing_specs=[],
    )

    evidence = context["attention_evidence"]
    assert 12 <= len(evidence) < 100
    assert context["attention_evidence_policy"]["source_rows"] == 101
    assert context["attention_evidence_policy"]["usable_source_rows"] == 100
    assert all("chunk_text" not in row for row in evidence)
    assert all(len(row["evidence_snippet"]) <= 480 for row in evidence)
    assert all("baseline age 78 years" in row["evidence_snippet"] for row in evidence)
    span = evidence[0]["top_token_spans"][0]
    assert span == {
        "text": "baseline age 78 years",
        "focus_token": "age",
        "salience": 0.12346,
    }


def test_attention_agent_prompt_allows_missing_roles():
    from oci.inference.agentic_explicit_feature_forest import agent_response_schema_issues

    issues = agent_response_schema_issues(
        [
            {
                "action": "add",
                "name": "attended_patient_field",
                "type": "continuous",
                "description": "A field implied by repeated attended spans",
            }
        ],
        context={"prompt_version": "agentic_attention_variable_forest_v1"},
    )

    assert not issues


def test_low_coverage_attention_candidates_trigger_retry(tmp_path):
    class RetryAgent:
        def __init__(self):
            self.contexts = []

        def propose(self, context):
            self.contexts.append(context)
            if context["proposal_attempt"] == 1:
                return [
                    {
                        "action": "add",
                        "name": "rare_marker",
                        "type": "categorical",
                        "categories": ["absent", "present"],
                        "roles": ["confounder"],
                        "description": "Sparse marker from one note",
                        "rationale": "rare marker appears in a high-attention chunk",
                    }
                ]
            assert "rare_marker" in context["excluded_feature_names"]
            return [
                {
                    "action": "add",
                    "name": "age_group",
                    "type": "categorical",
                    "categories": ["younger", "older"],
                    "roles": ["confounder"],
                    "description": "Age group before treatment",
                    "rationale": "age appears repeatedly in high-attention chunks",
                }
            ]

    class CoverageExtractionProvider:
        def ensure_features(self, dataset, specs):
            dataset = dataset.copy()
            for spec in specs:
                col = f"explicit_feat_{spec.name}"
                miss_col = f"{col}_missing"
                if spec.name == "rare_marker":
                    dataset[col] = ["present", None, None, None]
                    dataset[miss_col] = [False, True, True, True]
                else:
                    dataset[col] = ["older", "younger", "older", "younger"]
                    dataset[miss_col] = False
            return dataset

    df = pd.DataFrame(
        {
            "clinical_text": [
                "Age 78 years.",
                "Age 61 years.",
                "Age 80 years.",
                "Age 55 years.",
            ],
            "treatment_indicator": [0, 1, 0, 1],
            "outcome_indicator": [0, 1, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        architecture=ModelArchitectureConfig(
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                candidate_proposals_per_fold=1,
                coverage_retry_attempts=1,
                consensus_min_fold_fraction=1.0,
                min_extraction_coverage=0.75,
            ),
        ),
    )
    agent = RetryAgent()
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=agent,
        extraction_provider=CoverageExtractionProvider(),
    )
    attention_rows = [
        {
            "row_id": int(runner.dataset.loc[i % len(df), "_oci_row_id"]),
            "fold": fold,
            "stage": "nuisance",
            "chunk_text": f"Age {60 + i} years before treatment.",
            "attention": 0.9,
        }
        for fold in [1, 2]
        for i in [fold]
    ]

    selected = runner._discover_extract_filter_with_retries(
        stage="confounder",
        outer_fold=1,
        discovery_df=runner.dataset,
        train_idx=np.arange(len(runner.dataset)),
        attention_rows=attention_rows,
        existing_specs=[],
    )

    assert [spec.name for spec in selected] == ["age_group"]
    assert {context["proposal_attempt"] for context in agent.contexts} == {1, 2}
    assert all(context["max_proposals"] == 1 for context in agent.contexts)
    coverage_path = (
        tmp_path
        / "agentic_attention_variable_forest"
        / "coverage_filter_by_attempt.jsonl"
    )
    coverage_rows = [json.loads(line) for line in coverage_path.read_text().splitlines()]
    assert coverage_rows[0]["dropped_features"][0]["name"] == "rare_marker"
    assert coverage_rows[1]["kept_features"] == ["age_group"]


def test_attention_candidates_are_filtered_by_association_signal(tmp_path):
    class AssociationAgent:
        def propose(self, context):
            assert context["max_proposals"] == 2
            return [
                {
                    "action": "add",
                    "name": "signal_covariate",
                    "type": "categorical",
                    "categories": ["low", "high"],
                    "roles": ["confounder"],
                    "description": "A baseline covariate associated with treatment and outcome",
                },
                {
                    "action": "add",
                    "name": "noise_covariate",
                    "type": "categorical",
                    "categories": ["a", "b"],
                    "roles": ["confounder"],
                    "description": "A baseline covariate not associated with treatment or outcome",
                },
            ]

    class AssociationExtractionProvider:
        def ensure_features(self, dataset, specs):
            dataset = dataset.copy()
            row_pos = np.arange(len(dataset))
            for spec in specs:
                col = f"explicit_feat_{spec.name}"
                miss_col = f"{col}_missing"
                if spec.name == "signal_covariate":
                    dataset[col] = np.where(row_pos < len(dataset) // 2, "low", "high")
                elif spec.name == "noise_covariate":
                    dataset[col] = np.where(row_pos % 2 == 0, "a", "b")
                else:
                    dataset[col] = spec.categories[0]
                dataset[miss_col] = False
            return dataset

    n = 80
    signal = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
    df = pd.DataFrame(
        {
            "clinical_text": [f"patient {i}" for i in range(n)],
            "treatment_indicator": signal,
            "outcome_indicator": signal,
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        architecture=ModelArchitectureConfig(
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                candidate_proposals_per_fold=2,
                coverage_retry_attempts=0,
                signal_retry_attempts=0,
                consensus_min_fold_fraction=1.0,
                min_extraction_coverage=0.75,
                association_alpha=0.05,
                association_min_n=20,
                association_min_non_missing=10,
                signal_cv_folds=2,
                min_signal_treatment_auroc=0.55,
                min_signal_outcome_auroc=0.55,
            ),
        ),
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=AssociationAgent(),
        extraction_provider=AssociationExtractionProvider(),
    )
    attention_rows = [
        {
            "row_id": int(runner.dataset.loc[fold - 1, "_oci_row_id"]),
            "fold": fold,
            "stage": "nuisance",
            "chunk_text": "high-attention baseline signal chunk",
            "attention": 0.9,
        }
        for fold in [1, 2]
    ]

    selected = runner._discover_extract_filter_with_retries(
        stage="confounder",
        outer_fold=1,
        discovery_df=runner.dataset,
        train_idx=np.arange(len(runner.dataset)),
        attention_rows=attention_rows,
        existing_specs=[],
    )

    assert [spec.name for spec in selected] == ["signal_covariate"]
    assoc_path = (
        tmp_path
        / "agentic_attention_variable_forest"
        / "association_filter_by_attempt.jsonl"
    )
    rows = [json.loads(line) for line in assoc_path.read_text().splitlines()]
    assert rows[0]["kept_features"] == ["signal_covariate"]
    assert rows[0]["dropped_features"][0]["name"] == "noise_covariate"
    assert rows[0]["multivariable_signal"]["adequate"] is True


def test_alias_consensus_reaches_coverage_and_association_filtering(tmp_path):
    class AliasAgent:
        def propose(self, context):
            if (
                context["prompt_version"]
                == "agentic_attention_consensus_disambiguation_v1"
            ):
                return {
                    "groups": [
                        {
                            "canonical_name": "patient_age",
                            "member_names": [
                                "patient_age",
                                "age_at_diagnosis",
                                "baseline_age",
                            ],
                            "member_folds": [1, 2, 3],
                            "type": "continuous",
                            "description": "Patient age before treatment",
                            "rationale": (
                                "All three folds proposed the same age "
                                "extraction target."
                            ),
                        }
                    ],
                    "unmerged": [
                        {"name": "rare_marker", "reason": "Only one fold proposed it."}
                    ],
                }
            names = {
                1: "patient_age",
                2: "age_at_diagnosis",
                3: "baseline_age",
                4: "rare_marker",
                5: "noise_covariate",
            }
            name = names[int(context["fold"])]
            return [
                {
                    "action": "add",
                    "name": name,
                    "type": "continuous" if "age" in name else "categorical",
                    "categories": None if "age" in name else ["absent", "present"],
                    "roles": ["confounder"],
                    "description": f"{name} before treatment",
                    "rationale": "high-attention baseline text",
                }
            ]

    class AliasExtractionProvider:
        def ensure_features(self, dataset, specs):
            dataset = dataset.copy()
            row_pos = np.arange(len(dataset))
            for spec in specs:
                col = f"explicit_feat_{spec.name}"
                miss_col = f"{col}_missing"
                if spec.name == "patient_age":
                    dataset[col] = np.where(row_pos < len(dataset) // 2, 50.0, 80.0)
                else:
                    dataset[col] = spec.categories[0] if spec.categories else 0.0
                dataset[miss_col] = False
            return dataset

    n = 80
    signal = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
    df = pd.DataFrame(
        {
            "clinical_text": [f"patient age note {i}" for i in range(n)],
            "treatment_indicator": signal,
            "outcome_indicator": signal,
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        architecture=ModelArchitectureConfig(
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=5,
                effect_folds=5,
                candidate_proposals_per_fold=1,
                coverage_retry_attempts=0,
                signal_retry_attempts=0,
                consensus_min_folds=2,
                min_extraction_coverage=0.75,
                association_alpha=0.05,
                association_min_n=20,
                association_min_non_missing=10,
                signal_cv_folds=2,
                min_signal_treatment_auroc=0.55,
                min_signal_outcome_auroc=0.55,
            ),
        ),
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        device=torch.device("cpu"),
        num_workers=1,
        proposal_agent=AliasAgent(),
        extraction_provider=AliasExtractionProvider(),
    )
    attention_rows = [
        {
            "row_id": int(runner.dataset.loc[fold - 1, "_oci_row_id"]),
            "fold": fold,
            "stage": "nuisance",
            "chunk_text": f"Age evidence fold {fold}.",
            "attention": 0.9,
        }
        for fold in [1, 2, 3, 4, 5]
    ]

    selected = runner._discover_extract_filter_with_retries(
        stage="confounder",
        outer_fold=1,
        discovery_df=runner.dataset,
        train_idx=np.arange(len(runner.dataset)),
        attention_rows=attention_rows,
        existing_specs=[],
    )

    assert [spec.name for spec in selected] == ["patient_age"]
    disambig_path = (
        tmp_path
        / "agentic_attention_variable_forest"
        / "consensus_disambiguation_by_attempt.jsonl"
    )
    disambig_rows = [
        json.loads(line) for line in disambig_path.read_text().splitlines()
    ]
    assert disambig_rows[0]["status"] == "complete"
    assert disambig_rows[0]["selected_groups"][0]["canonical_name"] == "patient_age"
    coverage_path = (
        tmp_path
        / "agentic_attention_variable_forest"
        / "coverage_filter_by_attempt.jsonl"
    )
    coverage_rows = [json.loads(line) for line in coverage_path.read_text().splitlines()]
    assert coverage_rows[0]["kept_features"] == ["patient_age"]
    assoc_path = (
        tmp_path
        / "agentic_attention_variable_forest"
        / "association_filter_by_attempt.jsonl"
    )
    assoc_rows = [json.loads(line) for line in assoc_path.read_text().splitlines()]
    assert assoc_rows[0]["kept_features"] == ["patient_age"]


def test_fold_parallelism_auto_is_conservative_on_cuda(tmp_path):
    df = pd.DataFrame(
        {
            "clinical_text": ["a", "b", "c", "d"],
            "treatment_indicator": [0, 1, 0, 1],
            "outcome_indicator": [0, 1, 0, 1],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        architecture=ModelArchitectureConfig(
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=4,
                effect_folds=4,
                fold_parallelism="auto",
            ),
        ),
    )

    cpu_runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "cpu.parquet",
        device=torch.device("cpu"),
        num_workers=3,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )
    cuda_runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "cuda.parquet",
        device=torch.device("cuda:0"),
        num_workers=3,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    assert cpu_runner._fold_n_jobs(4) == 3
    assert cuda_runner._fold_n_jobs(4) == 1


def test_explicit_fold_parallelism_overrides_cuda_serial_default(tmp_path):
    df = pd.DataFrame(
        {
            "clinical_text": ["a", "b", "c", "d", "e"],
            "treatment_indicator": [0, 1, 0, 1, 0],
            "outcome_indicator": [0, 1, 0, 1, 0],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        architecture=ModelArchitectureConfig(
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=5,
                effect_folds=5,
                fold_parallelism="2",
            ),
        ),
    )
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "cuda.parquet",
        device=torch.device("cuda:0"),
        num_workers=1,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    assert runner._fold_n_jobs(5) == 2


def test_agentic_attention_config_accepts_inner_fold_parallelism_alias(tmp_path):
    from oci.config import ExperimentConfig

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "applied_inference": {
                    "architecture": {
                        "model_type": "agentic_attention_variable_forest",
                        "agentic_attention_variable_forest": {
                            "nuisance_folds": 2,
                            "effect_folds": 2,
                            "inner_fold_parallelism": "2",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = ExperimentConfig.from_json(config_path)

    assert (
        config.applied_inference.architecture.agentic_attention_variable_forest
        .fold_parallelism
        == "2"
    )


def test_causal_forest_config_accepts_inner_fold_parallelism_alias(tmp_path):
    from oci.config import ExperimentConfig

    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "clinical_text": ["a", "b"],
            "treatment_indicator": [0, 1],
            "outcome_indicator": [0, 1],
        }
    ).to_parquet(dataset_path, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "applied_inference": {
                    "dataset_path": str(dataset_path),
                    "architecture": {
                        "model_type": "causal_forest",
                        "causal_forest": {
                            "use_rlearner_representation": True,
                            "inner_fold_parallelism": "2",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = ExperimentConfig.from_json(config_path)

    assert (
        config.applied_inference.architecture.causal_forest
        .rlearner_inner_fold_parallelism
        == "2"
    )


def test_oracle_agentic_attention_script_builds_configs(tmp_path):
    from oracle_experiment_scripts.run_oracle_agentic_attention_variable_forest_experiments import (
        _make_applied_config,
        _make_configs,
        build_arg_parser,
    )

    args = build_arg_parser().parse_args(
        [
            "--datasets",
            "synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured",
            "--n-repeats",
            "1",
            "--n-folds",
            "2",
            "--nuisance-folds",
            "2",
            "--effect-folds",
            "2",
            "--htr-sentence-model",
            "hash",
            "--htr-sentence-encoder-batch-size",
            "32",
            "--htr-sentence-encoder-backend",
            "transformers",
            "--htr-sentence-pooling",
            "mean",
            "--htr-trainable-sentence-encoder-layers",
            "1",
            "--non-nuisance-epochs",
            "4",
            "--effect-objective",
            "logistic_r_loss",
            "--neural-stage-mode",
            "joint_rlearner",
            "--joint-rlearner-gamma",
            "0.7",
            "--residual-contrastive-enabled",
            "--residual-contrastive-score",
            "r_score_normalized",
            "--residual-contrastive-high-quantile",
            "0.85",
            "--residual-contrastive-low-quantile",
            "0.15",
            "--residual-contrastive-neutral-abs-quantile",
            "0.35",
            "--residual-contrastive-min-class-count",
            "3",
            "--max-experiments",
            "1",
        ]
    )

    configs = _make_configs(args)
    assert len(configs) == 1
    config = configs[0]
    assert config.model_type == "agentic_attention_variable_forest"
    assert config.htr_sentence_model == "hash"
    assert config.htr_sentence_encoder_batch_size == 32
    assert config.htr_sentence_encoder_backend == "transformers"
    assert config.htr_sentence_pooling == "mean"
    assert config.htr_trainable_sentence_encoder_layers == 1
    assert config.non_nuisance_epochs == 4
    assert config.effect_objective == "logistic_r_loss"
    assert config.neural_stage_mode == "joint_rlearner"
    assert config.joint_rlearner_gamma == pytest.approx(0.7)
    assert config.residual_contrastive_enabled is True
    assert config.residual_contrastive_score == "r_score_normalized"
    assert config.residual_contrastive_high_quantile == 0.85
    assert config.residual_contrastive_low_quantile == 0.15
    assert config.residual_contrastive_neutral_abs_quantile == 0.35
    assert config.residual_contrastive_min_class_count == 3
    assert config.nuisance_epochs == 20
    assert config.nuisance_weight_decay == pytest.approx(0.05)
    assert config.nuisance_label_smoothing == pytest.approx(0.02)
    assert config.nuisance_calibration == "temperature_isotonic"

    applied = _make_applied_config(
        config,
        parquet_file=tmp_path / "dataset.parquet",
        initial_specs=[],
    )
    assert applied.architecture.model_type == "agentic_attention_variable_forest"
    assert applied.architecture.feature_extractor_type == "hierarchical_transformer"
    assert applied.architecture.htr_sentence_encoder_batch_size == 32
    assert applied.architecture.htr_sentence_encoder_backend == "transformers"
    assert applied.architecture.htr_sentence_pooling == "mean"
    assert applied.architecture.htr_trainable_sentence_encoder_layers == 1
    assert applied.training.epochs == 4
    assert applied.architecture.agentic_attention_variable_forest.nuisance_folds == 2
    assert applied.architecture.agentic_attention_variable_forest.nuisance_epochs == 20
    assert (
        applied.architecture.agentic_attention_variable_forest.effect_objective
        == "logistic_r_loss"
    )
    assert (
        applied.architecture.agentic_attention_variable_forest.neural_stage_mode
        == "joint_rlearner"
    )
    assert (
        applied.architecture.agentic_attention_variable_forest.joint_rlearner_gamma
        == pytest.approx(0.7)
    )
    assert (
        applied.architecture.agentic_attention_variable_forest.nuisance_weight_decay
        == pytest.approx(0.05)
    )
    assert (
        applied.architecture.agentic_attention_variable_forest
        .residual_contrastive_enabled
        is True
    )
    assert (
        applied.architecture.agentic_attention_variable_forest
        .residual_contrastive_score
        == "r_score_normalized"
    )


def test_applied_router_dispatches_agentic_attention_variable_forest(monkeypatch, tmp_path):
    from oci.inference.applied import run_applied_inference
    import oci.inference.agentic_attention_variable_forest as module

    df = pd.DataFrame(
        {
            "clinical_text": ["a", "b", "c", "d"],
            "treatment_indicator": [0, 1, 0, 1],
            "outcome_indicator": [0, 1, 1, 0],
        }
    )
    called = {}

    def fake_runner(dataset, config, output_path, device, num_workers):
        called["model_type"] = config.architecture.model_type
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.assign(pred_ite_prob=0.0).to_parquet(output_path, index=False)

    monkeypatch.setattr(module, "run_agentic_attention_variable_forest", fake_runner)
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        architecture=ModelArchitectureConfig(
            model_type="agentic_attention_variable_forest",
            feature_extractor_type="hierarchical_transformer",
        ),
    )

    output_path = tmp_path / "predictions.parquet"
    run_applied_inference(
        dataset=df,
        config=config,
        output_path=output_path,
        device=torch.device("cpu"),
    )

    assert called["model_type"] == "agentic_attention_variable_forest"
    assert output_path.exists()


@pytest.mark.skipif(not ECONML_AVAILABLE, reason="econml is required for final forest")
def test_agentic_attention_variable_forest_fixed_split(tmp_path):
    n = 18
    texts = []
    treatment = []
    outcome = []
    true_ite = []
    for i in range(n):
        older = i % 3 == 0
        mutated = i % 2 == 0
        texts.append(
            f"Patient {'older' if older else 'younger'} with NSCLC. "
            f"{'mutation positive' if mutated else 'mutation negative'} before treatment."
        )
        treatment.append(int(older or (i % 4 == 0)))
        outcome.append(int(mutated ^ bool(treatment[-1])))
        true_ite.append(0.2 if mutated else -0.1)

    df = pd.DataFrame(
        {
            "clinical_text": texts,
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
            "true_ite_prob": true_ite,
            "split": ["train"] * 12 + ["test"] * 6,
        }
    )

    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=0,
        clinical_question="Compare treatment A versus B in NSCLC.",
        architecture=ModelArchitectureConfig(
            model_type="agentic_attention_variable_forest",
            feature_extractor_type="hierarchical_transformer",
            htr_sentence_model="hash",
            htr_chunk_size_words=6,
            htr_chunk_overlap_words=1,
            htr_max_chunks=8,
            htr_num_layers=1,
            htr_num_heads=2,
            htr_transformer_dim=24,
            htr_projection_dim=16,
            htr_hash_embedding_dim=24,
            htr_dropout=0.0,
            causal_head_hidden_outcome_dim=12,
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=8,
                min_samples_leaf=2,
                honest=False,
                inference=True,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                agent_max_tokens=1000,
            ),
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                attention_top_k_chunks=2,
                consensus_min_fold_fraction=1.0,
                fold_parallelism="auto",
            ),
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            gradient_clip_norm=1.0,
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=False, features=[]),
    )

    output_path = tmp_path / "applied_inference" / "predictions.parquet"
    run_agentic_attention_variable_forest(
        dataset=df,
        config=config,
        output_path=output_path,
        device=torch.device("cpu"),
        num_workers=2,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )

    results = pd.read_parquet(output_path)
    assert len(results) == 6
    for col in [
        "pred_ite_prob",
        "pred_y0_prob",
        "pred_y1_prob",
        "pred_propensity_prob",
        "selected_feature_names",
    ]:
        assert col in results.columns
        if col != "selected_feature_names":
            assert np.all(np.isfinite(results[col].to_numpy()))
    assert set(results["selected_feature_names"]) == {"age_group,mutation_status"}

    artifact_dir = output_path.parent / "agentic_attention_variable_forest"
    assert (artifact_dir / "nuisance_oof_predictions.parquet").exists()
    assert (artifact_dir / "r_stage_oof_predictions.parquet").exists()
    assert (artifact_dir / "nuisance_attention_evidence.parquet").exists()
    assert (artifact_dir / "r_stage_attention_evidence.parquet").exists()
    assert (artifact_dir / "oracle_metrics.json").exists()

    consensus = json.loads((artifact_dir / "consensus.json").read_text())
    assert consensus[0]["confounders"] == ["age_group"]
    assert consensus[0]["effect_modifiers"] == ["mutation_status"]


def test_joint_rlearner_neural_only_writes_oof_artifacts(tmp_path, monkeypatch):
    texts = [
        f"Patient {idx}. Age {'older' if idx % 2 else 'younger'}. "
        f"PD-L1 {'high' if idx % 3 == 0 else 'low'}."
        for idx in range(16)
    ]
    treatment = np.asarray([idx % 2 for idx in range(16)], dtype=int)
    outcome = np.asarray([(idx // 2) % 2 for idx in range(16)], dtype=int)
    df = pd.DataFrame(
        {
            "clinical_text": texts,
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
            "true_ite_prob": np.linspace(-0.1, 0.3, 16),
        }
    )

    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=2,
        clinical_question="Compare treatment A versus B in NSCLC.",
        architecture=ModelArchitectureConfig(
            model_type="agentic_attention_variable_forest",
            feature_extractor_type="hierarchical_transformer",
            htr_sentence_model="hash",
            htr_chunk_size_words=5,
            htr_chunk_overlap_words=1,
            htr_max_chunks=4,
            htr_num_layers=1,
            htr_num_heads=2,
            htr_transformer_dim=16,
            htr_projection_dim=8,
            htr_hash_embedding_dim=16,
            htr_dropout=0.0,
            causal_head_hidden_outcome_dim=8,
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=4,
                min_samples_leaf=2,
                honest=False,
                inference=False,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                agent_max_tokens=1000,
            ),
            agentic_attention_variable_forest=AgenticAttentionVariableForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                nuisance_epochs=1,
                nuisance_calibration="none",
                attention_top_k_chunks=1,
                neural_only=True,
                neural_stage_mode="joint_rlearner",
                joint_rlearner_gamma=0.5,
                fold_parallelism="auto",
            ),
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            gradient_clip_norm=1.0,
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=False, features=[]),
    )

    output_path = tmp_path / "applied_inference" / "predictions.parquet"
    runner = AgenticAttentionVariableForestRunner(
        dataset=df,
        config=config,
        output_path=output_path,
        device=torch.device("cpu"),
        num_workers=0,
        proposal_agent=FakeAttentionAgent(),
        extraction_provider=FakeExtractionProvider(),
    )
    monkeypatch.setattr(runner, "_create_extractor", lambda: DummyTensorExtractor())
    monkeypatch.setattr(runner, "_train_joint_rlearner_model", lambda *args, **kwargs: None)

    def predict_joint(model, frame):
        n = len(frame)
        scale = np.linspace(0.0, 1.0, n, dtype=float)
        return 0.25 + 0.5 * scale, 0.30 + 0.4 * scale, -0.2 + 0.4 * scale

    def attention_evidence(extractor, frame, fold, outer_fold, stage, extra):
        rows = []
        for offset, row_id in enumerate(frame["_oci_row_id"].tolist()):
            record = {
                "row_id": int(row_id),
                "fold": int(fold),
                "stage": stage,
                "chunk_index": 0,
                "chunk_text": frame.iloc[offset]["clinical_text"],
                "attention": 1.0,
                "outer_fold": int(outer_fold),
            }
            for key, values in extra.items():
                value = np.asarray(values, dtype=object)[offset]
                record[key] = value.item() if hasattr(value, "item") else value
            rows.append(record)
        return rows

    monkeypatch.setattr(runner, "_predict_joint_rlearner_model", predict_joint)
    monkeypatch.setattr(runner, "_attention_evidence", attention_evidence)

    result = runner._crossfit_joint_rlearner(runner.dataset.copy(), outer_fold=1)
    predictions = result["predictions"].copy()
    predictions = predictions.merge(
        runner.dataset.drop(columns=[config.text_column]),
        on="_oci_row_id",
        how="left",
        suffixes=("", "_source"),
    )
    predictions["pred_ite_prob"] = predictions["tau_hat_r_stage"]
    predictions["pred_propensity_prob"] = predictions["e_hat"]
    predictions["pred_outcome_prob"] = predictions["m_hat"]
    runner._save_predictions(predictions)
    runner._save_artifacts(predictions)

    results = pd.read_parquet(output_path)
    assert len(results) == len(df)
    assert set(results["effect_objective"]) == {"squared_r_loss"}
    assert np.all(np.isfinite(results["tau_hat_r_stage"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(results["e_hat"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(results["m_hat"].to_numpy(dtype=float)))

    artifact_dir = output_path.parent / "agentic_attention_variable_forest"
    nuisance = pd.read_parquet(artifact_dir / "nuisance_oof_predictions.parquet")
    r_stage = pd.read_parquet(artifact_dir / "r_stage_oof_predictions.parquet")
    nuisance_attention = pd.read_parquet(
        artifact_dir / "nuisance_attention_evidence.parquet"
    )
    r_stage_attention = pd.read_parquet(
        artifact_dir / "r_stage_attention_evidence.parquet"
    )
    manifest = json.loads((artifact_dir / "run_manifest.json").read_text())

    assert len(nuisance) == len(df)
    assert len(r_stage) == len(df)
    assert set(nuisance_attention["stage"]) == {"nuisance"}
    assert set(r_stage_attention["stage"]) == {"effect_modifier"}
    assert manifest["config"]["neural_stage_mode"] == "joint_rlearner"
    assert manifest["config"]["joint_rlearner_gamma"] == pytest.approx(0.5)
