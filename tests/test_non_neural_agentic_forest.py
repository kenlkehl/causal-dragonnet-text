from pathlib import Path

import numpy as np
import pandas as pd

from oci.config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExperimentConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ModelArchitectureConfig,
    NonNeuralAgenticForestConfig,
)
from oci.inference.agentic_explicit_feature_forest import SplitEvaluation
from oci.inference.non_neural_agentic_forest import run_non_neural_agentic_forest


class FakeProposalAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        return [
            {
                "action": "add",
                "name": "age",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "Patient age at treatment initiation in years.",
                "rationale": "Age-bearing terms appear in treatment and outcome models.",
                "expected_signal": "treatment and outcome",
            },
            {
                "action": "add",
                "name": "pd_l1_expression_level",
                "type": "categorical",
                "categories": ["<1%", "1-49%", ">=50%"],
                "roles": ["effect_modifier"],
                "description": "Pretreatment tumor PD-L1 expression category.",
                "rationale": "PD-L1 threshold terms appear in the pseudo-target model.",
                "expected_signal": "pseudo-target",
            },
            {
                "action": "add",
                "name": "pdl1_expression",
                "type": "categorical",
                "categories": ["<1%", "1-49%", ">=50%"],
                "roles": ["effect_modifier"],
                "description": "Pretreatment tumor PD-L1 expression category.",
                "rationale": "Alternative alias for the same PD-L1 concept.",
                "expected_signal": "pseudo-target",
            },
        ]


class FakeExtractionProvider:
    def ensure_features(self, dataset, specs):
        dataset = dataset.copy()
        text = dataset["clinical_text"].astype(str)
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if spec.name == "age":
                dataset[value_col] = text.str.extract(r"age (\d+)").astype(float)
            elif spec.name == "pd_l1_expression":
                dataset[value_col] = np.where(
                    text.str.contains(">=50%"),
                    ">=50%",
                    np.where(text.str.contains("1-49%"), "1-49%", "<1%"),
                )
            else:
                dataset[value_col] = np.nan
            dataset[missing_col] = dataset[value_col].isna()
        return dataset


class FakeEvaluator:
    def __init__(self):
        self.seen_specs = []

    def evaluate_split(self, train_df, test_df, specs, fold_id):
        self.seen_specs.append([spec.name for spec in specs])
        predictions = test_df.copy()
        predictions["pred_ite_prob"] = 0.1
        predictions["pred_y0_prob"] = 0.4
        predictions["pred_y1_prob"] = 0.5
        predictions["pred_propensity_prob"] = 0.5
        predictions["pred_outcome_prob"] = 0.5
        predictions["cv_fold"] = fold_id
        metrics = {
            "fold": fold_id,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_explicit_features": len(specs),
        }
        return SplitEvaluation(predictions=predictions, metrics=metrics)


def test_non_neural_agentic_forest_runs_with_fake_agent_and_extractor(tmp_path: Path):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 baseline note pd-l1 >=50% high marker",
                "age 78 baseline note pd-l1 <1% low marker",
                "age 57 baseline note pd-l1 >=50% high marker",
                "age 76 baseline note pd-l1 <1% low marker",
                "age 61 baseline note pd-l1 1-49% intermediate marker",
                "age 81 baseline note pd-l1 <1% low marker",
                "age 54 baseline note pd-l1 >=50% high marker",
                "age 70 baseline note pd-l1 1-49% intermediate marker",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 0, 0, 1, 0],
            "true_ite_prob": [0.3, 0.0, 0.3, 0.0, -0.1, 0.0, 0.3, -0.1],
        }
    )
    config = AppliedInferenceConfig(
        outcome_type="binary",
        text_column="clinical_text",
        treatment_column="treatment_indicator",
        outcome_column="outcome_indicator",
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="non_neural_agentic_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(inference=False),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                max_iterations=1,
                max_additions_per_iter=4,
                min_feature_coverage=0.1,
                clinical_text_examples_per_prompt=0,
            ),
            non_neural_agentic_forest=NonNeuralAgenticForestConfig(
                nuisance_folds=2,
                effect_folds=2,
                max_features=1000,
                min_df=1,
                top_n_features=5,
                fold_parallelism="2",
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = FakeProposalAgent()
    evaluator = FakeEvaluator()
    output_path = tmp_path / "predictions.parquet"

    run_non_neural_agentic_forest(
        dataset,
        config,
        output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=evaluator,
    )

    predictions = pd.read_parquet(output_path)
    assert len(predictions) == len(dataset)
    assert "selected_feature_names" in predictions.columns
    assert agent.contexts
    assert agent.contexts[0]["prompt_version"] == "non_neural_agentic_forest_v1"
    assert "feature_importance" in agent.contexts[0]
    assert all({"age", "pd_l1_expression"}.issubset(set(names)) for names in evaluator.seen_specs)
    assert all(names.count("pd_l1_expression") == 1 for names in evaluator.seen_specs)
    artifact_dir = output_path.parent / "non_neural_agentic_forest"
    assert (artifact_dir / "bow_oof_predictions.parquet").exists()
    assert (artifact_dir / "agent_candidate_proposals.jsonl").exists()


def test_non_neural_agentic_forest_parses_bow_model_option():
    cfg = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": (
                    "synthetic_data/example_synthetic_datasets/"
                    "one_confounder_one_effect_modifier_nsclc_with_structured/"
                    "dataset.parquet"
                ),
                "architecture": {
                    "model_type": "non_neural_agentic_forest",
                    "non_neural_agentic_forest": {
                        "bow_model": "extratrees",
                        "nuisance_folds": 2,
                        "effect_folds": 2,
                    },
                },
                "explicit_features": {"enabled": True, "features": []},
            }
        }
    )
    assert cfg.applied_inference.architecture.non_neural_agentic_forest.bow_model == "extratrees"
    cfg.validate()
