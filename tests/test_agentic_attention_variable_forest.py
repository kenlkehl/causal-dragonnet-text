import json

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
    consensus_feature_specs,
    run_agentic_attention_variable_forest,
)
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
    )

    assert [spec.name for spec in selected] == ["age_group"]


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
            "--max-experiments",
            "1",
        ]
    )

    configs = _make_configs(args)
    assert len(configs) == 1
    config = configs[0]
    assert config.model_type == "agentic_attention_variable_forest"
    assert config.htr_sentence_model == "hash"

    applied = _make_applied_config(
        config,
        parquet_file=tmp_path / "dataset.parquet",
        initial_specs=[],
    )
    assert applied.architecture.model_type == "agentic_attention_variable_forest"
    assert applied.architecture.feature_extractor_type == "hierarchical_transformer"
    assert applied.architecture.agentic_attention_variable_forest.nuisance_folds == 2


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
