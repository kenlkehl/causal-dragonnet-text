import json
from pathlib import Path

import numpy as np
import pandas as pd

from oci.config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExperimentConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ModelArchitectureConfig,
    NonNeuralAgenticForestConfig,
)
from oci.inference.agentic_explicit_feature_forest import (
    AgenticFeatureProposal,
    SplitEvaluation,
)
from oci.inference.non_neural_agentic_forest import (
    _candidate_consistency_threshold,
    _fallback_consistency_proposals,
    _fit_binary_bow_fold,
    run_non_neural_agentic_forest,
)


class FakeProposalAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context.get("prompt_version") == "non_neural_agentic_alias_resolution_v1":
            return {
                "groups": [
                    {
                        "canonical_name": "pd_l1_expression",
                        "member_names": [
                            "pd_l1_expression",
                            "pd_l1_expression_level",
                        ],
                        "type": "categorical",
                        "categories": ["<1%", "1-49%", ">=50%"],
                        "description": "Pretreatment tumor PD-L1 expression category.",
                        "roles": ["effect_modifier"],
                        "rationale": "The two names refer to the same extraction target.",
                    }
                ],
                "unmerged": [{"name": "age", "reason": "No alias proposed."}],
            }
        if context.get("prompt_version") == "non_neural_agentic_value_harmonization_v1":
            return {
                "features": [
                    {
                        "name": "age",
                        "type": "continuous",
                        "categories": None,
                        "description": "Patient age at treatment initiation in years.",
                        "missing_values": ["unknown", "not_reported", "high", "low"],
                        "rationale": "Age should remain numeric; qualitative labels are missing.",
                    },
                    {
                        "name": "pd_l1_expression",
                        "type": "categorical",
                        "categories": ["<1%", "1-49%", ">=50%", "unknown"],
                        "description": "Pretreatment tumor PD-L1 expression category.",
                        "value_aliases": {
                            "<1%": ["low negative"],
                            ">=50%": ["high", "50% or greater"],
                        },
                        "missing_values": ["unknown", "not_reported"],
                        "rationale": "Collapse high/low aliases into threshold categories.",
                    },
                ]
            }
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
                "name": "pd_l1_expression",
                "type": "categorical",
                "categories": ["low", "high", "unknown"],
                "roles": ["effect_modifier"],
                "description": "Pretreatment tumor PD-L1 expression category.",
                "rationale": "PD-L1 threshold terms appear in the pseudo-target model.",
                "expected_signal": "pseudo-target",
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
        self.seen_specs.append(specs)
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


class EmptyProposalAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context.get("prompt_version") == "non_neural_agentic_value_harmonization_v1":
            return {"features": context.get("selected_features", [])}
        return []


class RecordingExtractionProvider:
    def __init__(self):
        self.calls = []

    def ensure_features(self, dataset, specs):
        self.calls.append([(spec.name, tuple(spec.roles)) for spec in specs])
        dataset = dataset.copy()
        text = dataset["clinical_text"].astype(str)
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if spec.name == "age":
                dataset[value_col] = text.str.extract(r"age (\d+)").astype(float)
            elif spec.name == "biomarker":
                dataset[value_col] = np.where(
                    text.str.contains("biomarker positive"),
                    "positive",
                    "negative",
                )
            else:
                dataset[value_col] = np.nan
            dataset[missing_col] = dataset[value_col].isna()
        return dataset


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
                candidate_consistency_enabled=False,
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
    assert "selected_feature_roles" in predictions.columns
    assert "selected_confounder_names" in predictions.columns
    assert "selected_effect_modifier_names" in predictions.columns
    assert set(predictions["selected_feature_roles"]) == {
        "age[confounder],pd_l1_expression[effect_modifier]"
    }
    assert set(predictions["selected_confounder_names"]) == {"age"}
    assert set(predictions["selected_effect_modifier_names"]) == {"pd_l1_expression"}
    assert agent.contexts
    assert agent.contexts[0]["prompt_version"] == "non_neural_agentic_forest_v1"
    assert agent.contexts[1]["prompt_version"] == "non_neural_agentic_alias_resolution_v1"
    assert agent.contexts[2]["prompt_version"] == "non_neural_agentic_value_harmonization_v1"
    assert "feature_importance" in agent.contexts[0]
    phrase_features = agent.contexts[0]["feature_importance"]["phrase_features"]
    assert phrase_features
    assert all(2 <= len(row["feature"].split()) <= 4 for row in phrase_features)
    assert "canonical_feature_name_guidance" not in agent.contexts[0]
    assert "true_" not in json.dumps(agent.contexts[0])
    seen_names = [[spec.name for spec in specs] for specs in evaluator.seen_specs]
    assert all({"age", "pd_l1_expression"}.issubset(set(names)) for names in seen_names)
    assert all(names.count("pd_l1_expression") == 1 for names in seen_names)
    pdl1_specs = [
        spec
        for specs in evaluator.seen_specs
        for spec in specs
        if spec.name == "pd_l1_expression"
    ]
    assert pdl1_specs
    assert all(spec.categories == ["<1%", "1-49%", ">=50%"] for spec in pdl1_specs)
    assert all("unknown" not in spec.categories for spec in pdl1_specs)
    assert all(
        spec.value_aliases[">=50%"] == ["high", "50% or greater"]
        for spec in pdl1_specs
    )
    age_specs = [
        spec
        for specs in evaluator.seen_specs
        for spec in specs
        if spec.name == "age"
    ]
    assert age_specs
    assert all(spec.type == "continuous" and spec.categories is None for spec in age_specs)
    assert all("numeric value only" in (spec.description or "") for spec in age_specs)
    artifact_dir = output_path.parent / "non_neural_agentic_forest"
    assert (artifact_dir / "bow_oof_predictions.parquet").exists()
    assert (artifact_dir / "agent_candidate_proposals.jsonl").exists()


def test_non_neural_bow_fold_uses_prespecified_explicit_features():
    texts = ["same baseline note"] * 6
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    explicit_values = [
        {"age": 30.0, "age_missing": False},
        {"age": 31.0, "age_missing": False},
        {"age": 29.0, "age_missing": False},
        {"age": 80.0, "age_missing": False},
        {"age": 81.0, "age_missing": False},
        {"age": 79.0, "age_missing": False},
    ]
    heldout, pred = _fit_binary_bow_fold(
        texts,
        labels,
        fit_pos=np.asarray([0, 1, 3, 4]),
        heldout_pos=np.asarray([2, 5]),
        vectorizer_params={
            "ngram_range_min": 1,
            "ngram_range_max": 1,
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "max_features": 100,
        },
        model_params={
            "bow_model": "linear",
            "logistic_c": 1.0,
            "logistic_max_iter": 1000,
            "ridge_alpha": 1.0,
        },
        explicit_feature_dicts=explicit_values,
        explicit_specs=[
            ExplicitFeatureSpec(
                name="age",
                type="continuous",
                roles=["confounder"],
            )
        ],
        random_state=7,
    )
    assert heldout.tolist() == [2, 5]
    assert pred[1] > pred[0]


def test_non_neural_prespecified_features_extract_before_bow_and_merge_roles(
    tmp_path: Path,
):
    dataset = pd.DataFrame(
        {
            "clinical_text": [
                "age 55 biomarker positive baseline note",
                "age 78 biomarker negative baseline note",
                "age 57 biomarker positive baseline note",
                "age 76 biomarker negative baseline note",
                "age 61 biomarker positive baseline note",
                "age 81 biomarker negative baseline note",
                "age 54 biomarker positive baseline note",
                "age 70 biomarker negative baseline note",
            ],
            "treatment_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
            "outcome_indicator": [1, 0, 1, 0, 1, 0, 1, 0],
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
                candidate_consistency_enabled=False,
                fold_parallelism="1",
                prespecified_confounders=[
                    ExplicitFeatureSpec(
                        name="age",
                        type="continuous",
                        roles=["confounder"],
                    ),
                    {
                        "name": "biomarker",
                        "type": "categorical",
                        "categories": ["negative", "positive"],
                    },
                ],
                prespecified_effect_modifiers=[
                    {
                        "name": "biomarker",
                        "type": "categorical",
                        "categories": ["negative", "positive"],
                    },
                ],
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(enabled=True, features=[]),
    )
    agent = EmptyProposalAgent()
    extractor = RecordingExtractionProvider()
    evaluator = FakeEvaluator()

    run_non_neural_agentic_forest(
        dataset,
        config,
        tmp_path / "predictions.parquet",
        proposal_agent=agent,
        extraction_provider=extractor,
        evaluator=evaluator,
    )

    assert extractor.calls
    assert extractor.calls[0] == [
        ("age", ("confounder",)),
        ("biomarker", ("confounder", "effect_modifier")),
    ]
    first_context = agent.contexts[0]
    assert first_context["prompt_version"] == "non_neural_agentic_forest_v1"
    assert first_context["current_features"] == [
        {
            "name": "age",
            "type": "continuous",
            "categories": None,
            "description": None,
            "roles": ["confounder"],
            "value_aliases": None,
        },
        {
            "name": "biomarker",
            "type": "categorical",
            "categories": ["negative", "positive"],
            "description": None,
            "roles": ["confounder", "effect_modifier"],
            "value_aliases": None,
        },
    ]
    importance = first_context["feature_importance"]
    assert importance["n_prespecified_features"] == 2
    assert "explicit:age_normalized" in importance["prespecified_raw_feature_names"]
    assert "explicit:biomarker_positive" in importance["prespecified_raw_feature_names"]
    seen_roles = {
        spec.name: spec.roles
        for specs in evaluator.seen_specs
        for spec in specs
    }
    assert seen_roles == {
        "age": ["confounder"],
        "biomarker": ["confounder", "effect_modifier"],
    }


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
                        "candidate_consistency_enabled": True,
                        "candidate_consistency_inner_folds": 4,
                        "candidate_consistency_min_folds": 2,
                        "candidate_consistency_min_fold_fraction": 0.5,
                        "candidate_consistency_parallelism": "2",
                        "outer_parallelism": "3",
                        "bow_parallel_backend": "processes",
                    },
                },
                "explicit_features": {"enabled": True, "features": []},
            }
        }
    )
    nn_cfg = cfg.applied_inference.architecture.non_neural_agentic_forest
    assert nn_cfg.bow_model == "extratrees"
    assert nn_cfg.candidate_consistency_enabled is True
    assert nn_cfg.candidate_consistency_inner_folds == 4
    assert nn_cfg.candidate_consistency_min_folds == 2
    assert nn_cfg.candidate_consistency_min_fold_fraction == 0.5
    assert nn_cfg.candidate_consistency_parallelism == "2"
    assert nn_cfg.outer_parallelism == "3"
    assert nn_cfg.bow_parallel_backend == "processes"
    assert nn_cfg.ngram_range_max == 3
    cfg.validate()


def test_non_neural_agentic_forest_parses_prespecified_feature_sources(tmp_path):
    features_json = tmp_path / "features.json"
    features_json.write_text(
        json.dumps(
            {
                "confounders": [
                    {
                        "name": "baseline_risk",
                        "type": "continuous",
                        "description": "Baseline clinical risk score.",
                    }
                ],
                "effect_modifiers": [
                    {
                        "name": "baseline_risk",
                        "type": "continuous",
                        "description": "Baseline clinical risk score.",
                    }
                ],
            }
        )
    )
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
                        "prespecified_confounders": [
                            {
                                "name": "age",
                                "type": "continuous",
                                "description": "Age in years.",
                            }
                        ],
                        "prespecified_effect_modifiers": [
                            {
                                "name": "pd_l1_expression",
                                "type": "categorical",
                                "categories": ["low", "high"],
                                "description": "PD-L1 expression.",
                            }
                        ],
                        "prespecified_features_json": str(features_json),
                    },
                },
                "explicit_features": {"enabled": True, "features": []},
            }
        }
    )
    nn_cfg = cfg.applied_inference.architecture.non_neural_agentic_forest
    assert nn_cfg.prespecified_confounders[0].roles == ["confounder"]
    assert nn_cfg.prespecified_effect_modifiers[0].roles == ["effect_modifier"]
    assert nn_cfg.prespecified_features_json == str(features_json)


def test_non_neural_candidate_consistency_fallback_prefers_stable_candidates():
    assert _candidate_consistency_threshold(
        3,
        min_folds=2,
        min_fold_fraction=0.5,
    ) == 2
    age = AgenticFeatureProposal(
        action="add",
        name="patient_age",
        type="continuous",
        roles=["confounder"],
    )
    noise = AgenticFeatureProposal(
        action="add",
        name="rare_noise",
        type="categorical",
        categories=["present", "absent"],
        roles=["effect_modifier"],
    )
    selected = _fallback_consistency_proposals(
        [
            {
                "name": "patient_age",
                "passes_consistency_gate": True,
                "proposed_on_full_outer_train": True,
            },
            {
                "name": "rare_noise",
                "passes_consistency_gate": False,
                "proposed_on_full_outer_train": True,
            },
        ],
        {"patient_age": age, "rare_noise": noise},
    )
    assert selected == [age]
