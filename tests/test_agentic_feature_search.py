import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oci.config import (
    AgenticFeatureSearchConfig,
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureForestConfig,
    ExplicitFeatureSpec,
    ExperimentConfig,
    ModelArchitectureConfig,
)
from oci.extraction.cache import _compute_config_hash
from oci.inference.agentic_explicit_feature_forest import (
    AgenticFeatureSearchRunner,
    AgenticFeatureProposal,
    OpenAICompatibleFeatureSearchAgent,
    SplitEvaluation,
    VLLMExplicitFeatureExtractionProvider,
    apply_proposals,
    build_iteration_feedback,
    compare_candidate_to_baseline,
    evaluate_candidate_role_diagnostics,
    parse_agent_response,
    run_agentic_explicit_feature_forest,
    validate_agentic_proposals,
)


def _base_specs():
    return [
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            description="Patient age at treatment initiation",
            roles=["confounder"],
        )
    ]


def test_agentic_proposal_validation_rejects_duplicate_without_keyword_gate():
    search_config = AgenticFeatureSearchConfig(max_additions_per_iter=2)
    raw = [
        {
            "action": "add",
            "name": "Age",
            "type": "continuous",
            "roles": ["confounder"],
            "description": "Patient age",
        },
        {
            "action": "add",
            "name": "response_category",
            "type": "categorical",
            "categories": ["response", "no_response"],
            "roles": ["effect_modifier"],
            "description": "Response to treatment after therapy",
        },
        {
            "action": "add",
            "name": "baseline_nlr",
            "type": "continuous",
            "roles": ["effect_modifier"],
            "description": "Baseline neutrophil to lymphocyte ratio before treatment",
        },
    ]

    valid, rejected = validate_agentic_proposals(
        raw,
        current_specs=_base_specs(),
        search_config=search_config,
        allow_removals=False,
    )

    assert [proposal.name for proposal in valid] == ["response_category", "baseline_nlr"]
    assert rejected == [{"proposal": raw[0], "reason": "duplicate_feature"}]


def test_agentic_proposal_validation_does_not_use_allowed_baseline_whitelist():
    raw = [
        {
            "action": "add",
            "name": "patient_age",
            "type": "continuous",
            "roles": ["confounder", "effect_modifier"],
            "description": "The age of the patient at the time of baseline diagnosis or presentation.",
            "rationale": (
                "Age influences treatment selection and may modify physiological "
                "response to therapy."
            ),
            "expected_signal": "treatment, outcome",
        },
        {
            "action": "add",
            "name": "baseline_treatment_response",
            "type": "categorical",
            "categories": ["responder", "non_responder"],
            "roles": ["effect_modifier"],
            "description": "Baseline treatment response category",
        },
    ]

    valid, rejected = validate_agentic_proposals(
        raw,
        current_specs=[],
        search_config=AgenticFeatureSearchConfig(max_additions_per_iter=2),
        allow_removals=False,
    )

    assert [proposal.name for proposal in valid] == [
        "patient_age",
        "baseline_treatment_response",
    ]
    assert rejected == []


def test_apply_proposals_add_remove_and_update_role():
    specs = _base_specs() + [
        ExplicitFeatureSpec(
            name="pdl1",
            type="categorical",
            categories=["low", "high"],
            roles=["effect_modifier"],
        )
    ]
    proposals = [
        AgenticFeatureProposal(
            action="add",
            name="ecog",
            type="categorical",
            categories=["0", "1", "2"],
            roles=["confounder", "effect_modifier"],
            description="Baseline ECOG performance status",
        ),
        AgenticFeatureProposal(action="remove", name="pdl1"),
        AgenticFeatureProposal(action="update_role", name="age", roles=["confounder", "effect_modifier"]),
    ]

    updated = apply_proposals(specs, proposals)

    assert [spec.name for spec in updated] == ["age", "ecog"]
    assert updated[0].roles == ["confounder", "effect_modifier"]
    assert updated[1].roles == ["confounder", "effect_modifier"]


def test_candidate_acceptance_uses_r_loss_and_auc_guardrails():
    search_config = AgenticFeatureSearchConfig(
        min_r_loss_improvement=0.05,
        max_outcome_auroc_drop=0.002,
        max_treatment_auroc_drop=0.002,
        min_improvement_fold_fraction=1.0,
    )
    baseline = [
        {"inner_fold": 1, "r_loss": 1.0, "outcome_auroc": 0.70, "treatment_auroc": 0.75},
        {"inner_fold": 2, "r_loss": 1.0, "outcome_auroc": 0.70, "treatment_auroc": 0.75},
    ]
    good_candidate = [
        {"inner_fold": 1, "r_loss": 0.90, "outcome_auroc": 0.70, "treatment_auroc": 0.75},
        {"inner_fold": 2, "r_loss": 0.92, "outcome_auroc": 0.70, "treatment_auroc": 0.75},
    ]
    bad_outcome_candidate = [
        {"inner_fold": 1, "r_loss": 0.80, "outcome_auroc": 0.60, "treatment_auroc": 0.75},
        {"inner_fold": 2, "r_loss": 0.82, "outcome_auroc": 0.60, "treatment_auroc": 0.75},
    ]

    assert compare_candidate_to_baseline(baseline, good_candidate, search_config)[
        "passes_acceptance"
    ]
    assert not compare_candidate_to_baseline(baseline, bad_outcome_candidate, search_config)[
        "passes_acceptance"
    ]


def test_candidate_role_diagnostics_detect_confounder_and_modifier_signal():
    rng = np.random.default_rng(17)
    n = 400
    age = rng.normal(65, 8, size=n)
    biomarker = rng.normal(0, 1, size=n)
    age_z = (age - age.mean()) / age.std()
    treatment_prob = 1.0 / (1.0 + np.exp(-(-0.1 + 1.3 * biomarker + 0.25 * age_z)))
    treatment = rng.binomial(1, treatment_prob)
    outcome_prob = 1.0 / (
        1.0
        + np.exp(
            -(
                -0.8
                + 0.75 * biomarker
                + 0.25 * age_z
                + 0.15 * treatment
                + 1.4 * treatment * biomarker
            )
        )
    )
    outcome = rng.binomial(1, outcome_prob)
    df = pd.DataFrame(
        {
            "clinical_text": [f"Patient {i}" for i in range(n)],
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
            "explicit_feat_age": age,
            "explicit_feat_age_missing": False,
            "explicit_feat_baseline_biomarker": biomarker,
            "explicit_feat_baseline_biomarker_missing": False,
        }
    )
    current_specs = [
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            roles=["confounder"],
            description="Age at baseline",
        )
    ]
    candidate_specs = [
        ExplicitFeatureSpec(
            name="baseline_biomarker",
            type="continuous",
            roles=["confounder"],
            description="Baseline biomarker value",
        )
    ]

    diagnostics = evaluate_candidate_role_diagnostics(
        dataset=df,
        current_specs=current_specs,
        candidate_specs=candidate_specs,
        config=AppliedInferenceConfig(outcome_type="binary"),
        search_config=AgenticFeatureSearchConfig(
            role_diagnostic_score_delta_threshold=1e-4,
            role_diagnostic_min_n=20,
            role_diagnostic_min_non_missing=20,
        ),
    )

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["status"] == "ok"
    assert diagnostic["confounder_signal"]
    assert diagnostic["effect_modifier_signal"]
    assert diagnostic["recommended_roles"] == ["confounder", "effect_modifier"]
    assert diagnostic["treatment_association"]["score_delta"] > 0
    assert diagnostic["outcome_association"]["score_delta"] > 0
    assert diagnostic["treatment_interaction"]["score_delta"] > 0


def test_iteration_feedback_includes_role_diagnostics():
    recent_decisions = [
        {
            "outer_fold": 1,
            "iteration": 1,
            "event": "candidate_evaluations",
            "payload": [
                {
                    "candidate_id": "baseline_biomarker",
                    "proposals": [
                        {
                            "action": "add",
                            "name": "baseline_biomarker",
                            "type": "continuous",
                            "roles": ["confounder"],
                            "description": "Baseline biomarker value",
                        }
                    ],
                    "summary": {
                        "role_diagnostics": [
                            {
                                "name": "baseline_biomarker",
                                "status": "ok",
                                "proposed_roles": ["confounder"],
                                "recommended_roles": ["confounder", "effect_modifier"],
                                "confounder_signal": True,
                                "effect_modifier_signal": True,
                                "treatment_association": {"score_delta": 0.02},
                                "outcome_association": {"score_delta": 0.03},
                                "treatment_interaction": {"score_delta": 0.04},
                            }
                        ]
                    },
                    "comparison": {
                        "passes_acceptance": False,
                        "r_loss_improvement": 0.0,
                        "outcome_auroc_delta": 0.0,
                        "treatment_auroc_delta": 0.0,
                        "improved_fold_fraction": 0.0,
                    },
                    "accepted": False,
                }
            ],
        }
    ]

    feedback = build_iteration_feedback(recent_decisions, AgenticFeatureSearchConfig())

    assert feedback[0]["role_diagnostics"][0]["recommended_roles"] == [
        "confounder",
        "effect_modifier",
    ]
    assert feedback[0]["role_diagnostics"][0]["interaction_score_delta"] == 0.04


def test_extraction_cache_hash_includes_description_and_prompt_settings():
    spec_a = ExplicitFeatureSpec(
        name="age",
        type="continuous",
        description="Age at diagnosis",
        roles=["confounder"],
    )
    spec_b = ExplicitFeatureSpec(
        name="age",
        type="continuous",
        description="Age at treatment initiation",
        roles=["confounder"],
    )
    base = {
        "features": [spec_a],
        "prompt_template_version": "v1",
        "vllm_model_name": "model",
        "extraction_temperature": 0.0,
        "extraction_max_tokens": 128,
        "extraction_max_text_length": 1000,
    }

    desc_hash = _compute_config_hash({**base, "features": [spec_b]})
    prompt_hash = _compute_config_hash({**base, "prompt_template_version": "v2"})
    parser_hash = _compute_config_hash({**base, "vllm_reasoning_parser": "qwen3"})

    assert _compute_config_hash(base) != desc_hash
    assert _compute_config_hash(base) != prompt_hash
    assert _compute_config_hash(base) != parser_hash


def test_parse_agent_response_strips_inline_reasoning_trace():
    proposals = parse_agent_response(
        '<think>{"proposals": [{"name": "discard_me"}]}</think>\n'
        '{"proposals": [{"name": "age", "type": "continuous"}]}'
    )

    assert proposals == [{"name": "age", "type": "continuous"}]


def test_agentic_vllm_wrapper_adds_autodetected_reasoning_parser():
    from oracle_experiment_scripts.run_oracle_agentic_explicit_forest_experiments import (
        _extract_wrapper_vllm_args,
        _option_value,
        _start_local_vllm_servers,
        _with_expanded_agentic_defaults,
        _vllm_cmd,
    )

    settings = {
        "download_dir": None,
        "gpu_memory_utilization": "0.95",
        "max_num_seqs": "4",
        "max_num_batched_tokens": None,
        "dtype": None,
        "kv_cache_dtype": None,
        "quantization": None,
        "reasoning_parser": "auto",
        "extra_args": [],
    }
    qwen_cmd = _vllm_cmd(
        server_url="http://localhost:8000/v1",
        model_name="nvidia/Qwen3.6-35B-A3B-NVFP4",
        max_model_len="200000",
        settings=settings,
    )
    gemma_cmd = _vllm_cmd(
        server_url="http://localhost:8000/v1",
        model_name="nvidia/Gemma-4-31B-IT-NVFP4",
        max_model_len="200000",
        settings=settings,
    )
    unknown_cmd = _vllm_cmd(
        server_url="http://localhost:8000/v1",
        model_name="unknown/model",
        max_model_len="200000",
        settings=settings,
    )

    assert qwen_cmd[qwen_cmd.index("--reasoning-parser") + 1] == "qwen3"
    assert gemma_cmd[gemma_cmd.index("--reasoning-parser") + 1] == "gemma4"
    assert "--reasoning-parser" not in unknown_cmd

    cleaned, parsed_settings = _extract_wrapper_vllm_args([
        "runner.py",
        "--vllm-reasoning-parser",
        "qwen3",
    ])

    assert parsed_settings["reasoning_parser"] == "qwen3"
    assert cleaned[-2:] == ["--agentic-extraction-reasoning-parser", "qwen3"]

    parsed_settings = _extract_wrapper_vllm_args([
        "runner.py",
        "--vllm-quantization",
        "modelopt",
    ])[1]
    quantized_cmd = _vllm_cmd(
        server_url="http://localhost:8000/v1",
        model_name="nvidia/Qwen3.6-35B-A3B-NVFP4",
        max_model_len="200000",
        settings=parsed_settings,
    )

    assert quantized_cmd[quantized_cmd.index("--quantization") + 1] == "modelopt"

    expanded = _with_expanded_agentic_defaults([
        "runner.py",
        "--agentic-vllm-model-name",
        "legacy/extraction-model",
    ])

    assert "--agentic-extraction-model-name" not in expanded
    assert _option_value(
        expanded[1:],
        "--agentic-agent-model-name",
    ) == "legacy/extraction-model"
    assert _extract_wrapper_vllm_args([
        "runner.py",
        "--agentic-extraction-reasoning-parser",
        "gemma4",
    ])[1]["reasoning_parser"] == "gemma4"

    expanded_from_agent = _with_expanded_agentic_defaults([
        "runner.py",
        "--agentic-agent-model-name",
        "shared/model",
    ])

    assert _option_value(
        expanded_from_agent[1:],
        "--agentic-extraction-model-name",
    ) == "shared/model"

    with pytest.raises(ValueError, match="shared local vLLM server"):
        _start_local_vllm_servers(
            [
                "runner.py",
                "--agentic-agent-model-name",
                "agent/model",
                "--agentic-extraction-model-name",
                "extraction/model",
            ],
            {},
        )


def _provider_config(tmp_path, cache_enabled=False, batch_size=32):
    return AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=[],
            cache_enabled=cache_enabled,
            extraction_batch_size=batch_size,
        ),
    )


def test_agentic_extraction_provider_groups_missing_specs(monkeypatch, tmp_path):
    calls = []

    class FakeVLLMFeatureExtractor:
        def __init__(self, specs, **kwargs):
            self.specs = specs
            calls.append({"spec_names": [spec.name for spec in specs], "kwargs": kwargs})

        def extract_to_dataframe(self, texts, batch_size=32):
            calls[-1]["texts"] = list(texts)
            calls[-1]["batch_size"] = batch_size
            data = {}
            for spec in self.specs:
                value_col = f"explicit_feat_{spec.name}"
                if spec.type == "categorical":
                    data[value_col] = [spec.categories[0]] * len(texts)
                else:
                    data[value_col] = list(range(len(texts)))
                data[f"{value_col}_missing"] = [False] * len(texts)
            return pd.DataFrame(data)

        def cleanup(self):
            calls[-1]["cleanup"] = True

    monkeypatch.setattr(
        "oci.inference.agentic_explicit_feature_forest.VLLMFeatureExtractor",
        FakeVLLMFeatureExtractor,
    )
    provider = VLLMExplicitFeatureExtractionProvider(
        config=_provider_config(tmp_path, cache_enabled=False),
        output_dir=tmp_path,
    )
    df = pd.DataFrame({"clinical_text": ["note a", "note b"]})
    specs = [
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            roles=["confounder"],
            description="Age at baseline",
        ),
        ExplicitFeatureSpec(
            name="ecog",
            type="categorical",
            categories=["0", "1"],
            roles=["confounder"],
            description="Baseline ECOG",
        ),
    ]

    extracted = provider.ensure_features(df, specs)
    provider.ensure_features(extracted, specs)

    assert len(calls) == 1
    assert calls[0]["spec_names"] == ["age", "ecog"]
    assert calls[0]["texts"] == ["note a", "note b"]
    assert calls[0]["cleanup"]
    assert extracted["explicit_feat_age"].tolist() == [0, 1]
    assert extracted["explicit_feat_ecog"].tolist() == ["0", "0"]


def test_agentic_extraction_provider_autodiscovers_server_model(monkeypatch, tmp_path):
    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.models = FakeOpenAIModels(["served-extraction-model"])

    class FakeVLLMFeatureExtractor:
        def __init__(self, specs, **kwargs):
            self.specs = specs
            calls.append({"spec_names": [spec.name for spec in specs], "kwargs": kwargs})

        def extract_to_dataframe(self, texts, batch_size=32):
            calls[-1]["texts"] = list(texts)
            return pd.DataFrame(
                {
                    "explicit_feat_age": [72] * len(texts),
                    "explicit_feat_age_missing": [False] * len(texts),
                }
            )

        def cleanup(self):
            calls[-1]["cleanup"] = True

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        "oci.inference.agentic_explicit_feature_forest.VLLMFeatureExtractor",
        FakeVLLMFeatureExtractor,
    )
    config = _provider_config(tmp_path, cache_enabled=False)
    config.explicit_features.vllm_model_name = "auto"
    provider = VLLMExplicitFeatureExtractionProvider(config=config, output_dir=tmp_path)
    df = pd.DataFrame({"clinical_text": ["note a", "note b"]})
    specs = [
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            roles=["confounder"],
            description="Age at baseline",
        )
    ]

    extracted = provider.ensure_features(df, specs)

    assert calls[0]["kwargs"]["model_name"] == "served-extraction-model"
    assert extracted["explicit_feat_age"].tolist() == [72, 72]


def test_agentic_extraction_provider_saves_grouped_results_as_per_spec_cache(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeVLLMFeatureExtractor:
        def __init__(self, specs, **kwargs):
            self.specs = specs
            calls.append([spec.name for spec in specs])

        def extract_to_dataframe(self, texts, batch_size=32):
            data = {}
            for spec in self.specs:
                value_col = f"explicit_feat_{spec.name}"
                data[value_col] = [float(len(spec.name))] * len(texts)
                data[f"{value_col}_missing"] = [False] * len(texts)
            return pd.DataFrame(data)

        def cleanup(self):
            pass

    monkeypatch.setattr(
        "oci.inference.agentic_explicit_feature_forest.VLLMFeatureExtractor",
        FakeVLLMFeatureExtractor,
    )
    config = _provider_config(tmp_path, cache_enabled=True)
    specs = [
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            roles=["confounder"],
            description="Age at baseline",
        ),
        ExplicitFeatureSpec(
            name="ldh",
            type="continuous",
            roles=["confounder"],
            description="Baseline LDH",
        ),
    ]
    df = pd.DataFrame({"clinical_text": ["note a", "note b", "note c"]})

    first_provider = VLLMExplicitFeatureExtractionProvider(config=config, output_dir=tmp_path)
    first_provider.ensure_features(df, specs)
    second_provider = VLLMExplicitFeatureExtractionProvider(config=config, output_dir=tmp_path)
    cached = second_provider.ensure_features(df, specs)

    assert calls == [["age", "ldh"]]
    assert cached["explicit_feat_age"].tolist() == [3.0, 3.0, 3.0]
    assert cached["explicit_feat_ldh"].tolist() == [3.0, 3.0, 3.0]


def test_agentic_extraction_provider_resumes_from_row_cache(
    monkeypatch,
    tmp_path,
):
    calls = []
    fail_on_call = {"value": 2}

    class FakeVLLMFeatureExtractor:
        def __init__(self, specs, **kwargs):
            self.specs = specs

        def extract_to_dataframe(self, texts, batch_size=32):
            calls.append(list(texts))
            if len(calls) == fail_on_call["value"]:
                raise RuntimeError("simulated extraction interruption")
            data = {}
            for spec in self.specs:
                value_col = f"explicit_feat_{spec.name}"
                data[value_col] = [
                    float(str(text).replace("note ", "")) for text in texts
                ]
                data[f"{value_col}_missing"] = [False] * len(texts)
            return pd.DataFrame(data)

        def cleanup(self):
            pass

    monkeypatch.setattr(
        "oci.inference.agentic_explicit_feature_forest.VLLMFeatureExtractor",
        FakeVLLMFeatureExtractor,
    )
    config = _provider_config(tmp_path, cache_enabled=True, batch_size=2)
    provider = VLLMExplicitFeatureExtractionProvider(config=config, output_dir=tmp_path)
    df = pd.DataFrame({"clinical_text": [f"note {idx}" for idx in range(5)]})
    specs = [
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            roles=["confounder"],
            description="Age at baseline",
        )
    ]

    with pytest.raises(RuntimeError, match="simulated extraction interruption"):
        provider.ensure_features(df, specs)

    row_cache = provider.cache.load_rows_if_valid(
        config.dataset_path,
        provider._cache_config(specs),
        expected_rows=len(df),
    )
    assert row_cache is not None
    assert row_cache["__oci_cache_row_index"].tolist() == [0, 1]

    fail_on_call["value"] = -1
    resumed_provider = VLLMExplicitFeatureExtractionProvider(
        config=config,
        output_dir=tmp_path,
    )
    resumed = resumed_provider.ensure_features(df, specs)

    assert calls == [
        ["note 0", "note 1"],
        ["note 2", "note 3"],
        ["note 2", "note 3"],
        ["note 4"],
    ]
    assert resumed["explicit_feat_age"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    full_cache = resumed_provider.cache.load_if_valid(
        config.dataset_path,
        resumed_provider._cache_config(specs),
        expected_rows=len(df),
    )
    assert full_cache is not None
    assert full_cache["explicit_feat_age"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_agentic_extraction_provider_reextracts_same_name_contract_change(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeVLLMFeatureExtractor:
        def __init__(self, specs, **kwargs):
            self.specs = specs
            calls.append([spec.description for spec in specs])

        def extract_to_dataframe(self, texts, batch_size=32):
            data = {}
            for spec in self.specs:
                value_col = f"explicit_feat_{spec.name}"
                value = 1.0 if "first" in spec.description else 2.0
                data[value_col] = [value] * len(texts)
                data[f"{value_col}_missing"] = [False] * len(texts)
            return pd.DataFrame(data)

        def cleanup(self):
            pass

    monkeypatch.setattr(
        "oci.inference.agentic_explicit_feature_forest.VLLMFeatureExtractor",
        FakeVLLMFeatureExtractor,
    )
    provider = VLLMExplicitFeatureExtractionProvider(
        config=_provider_config(tmp_path, cache_enabled=False),
        output_dir=tmp_path,
    )
    df = pd.DataFrame({"clinical_text": ["note"]})
    first_spec = ExplicitFeatureSpec(
        name="biomarker",
        type="continuous",
        roles=["confounder"],
        description="first definition",
    )
    second_spec = ExplicitFeatureSpec(
        name="biomarker",
        type="continuous",
        roles=["confounder"],
        description="second definition",
    )

    extracted = provider.ensure_features(df, [first_spec])
    reextracted = provider.ensure_features(extracted, [second_spec])

    assert calls == [["first definition"], ["second definition"]]
    assert reextracted["explicit_feat_biomarker"].tolist() == [2.0]


def test_agentic_extraction_provider_reuses_role_only_contract_change(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeVLLMFeatureExtractor:
        def __init__(self, specs, **kwargs):
            self.specs = specs
            calls.append([spec.roles for spec in specs])

        def extract_to_dataframe(self, texts, batch_size=32):
            data = {}
            for spec in self.specs:
                value_col = f"explicit_feat_{spec.name}"
                data[value_col] = [1.0] * len(texts)
                data[f"{value_col}_missing"] = [False] * len(texts)
            return pd.DataFrame(data)

        def cleanup(self):
            pass

    monkeypatch.setattr(
        "oci.inference.agentic_explicit_feature_forest.VLLMFeatureExtractor",
        FakeVLLMFeatureExtractor,
    )
    provider = VLLMExplicitFeatureExtractionProvider(
        config=_provider_config(tmp_path, cache_enabled=False),
        output_dir=tmp_path,
    )
    df = pd.DataFrame({"clinical_text": ["note"]})
    confounder_spec = ExplicitFeatureSpec(
        name="biomarker",
        type="continuous",
        roles=["confounder"],
        description="Baseline biomarker",
    )
    modifier_spec = ExplicitFeatureSpec(
        name="biomarker",
        type="continuous",
        roles=["effect_modifier"],
        description="Baseline biomarker",
    )

    extracted = provider.ensure_features(df, [confounder_spec])
    reused = provider.ensure_features(extracted, [modifier_spec])

    assert calls == [[["confounder"]]]
    assert reused["explicit_feat_biomarker"].tolist() == [1.0]


class FakeOpenAICompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            model="fake-agent",
            id=f"response-{len(self.calls)}",
            created=0,
            usage=None,
        )


class FakeOpenAIModels:
    def __init__(self, model_ids):
        self.model_ids = list(model_ids)
        self.calls = 0

    def list(self):
        self.calls += 1
        return SimpleNamespace(
            data=[SimpleNamespace(id=model_id) for model_id in self.model_ids]
        )


class FakeOpenAIClient:
    def __init__(self, contents, model_ids=None):
        self.completions = FakeOpenAICompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)
        self.models = FakeOpenAIModels(model_ids or ["fake-agent"])


def test_openai_agent_autodiscovers_model_name():
    client = FakeOpenAIClient(
        [json.dumps({"proposals": []})],
        model_ids=["served-agent-model"],
    )
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="auto",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    proposals = agent.propose({"current_features": [], "iteration_feedback": []})

    assert proposals == []
    assert client.models.calls == 1
    assert client.completions.calls[0]["model"] == "served-agent-model"


def test_openai_agent_autodiscovers_legacy_oracle_default_model_name():
    client = FakeOpenAIClient(
        [json.dumps({"proposals": []})],
        model_ids=["served-agent-model"],
    )
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_model_name="Qwen/Qwen3.6-27B",
            agent_schema_repair_attempts=0,
        )
    )
    agent._client = client

    agent.propose({"current_features": [], "iteration_feedback": []})

    assert client.models.calls == 1
    assert client.completions.calls[0]["model"] == "served-agent-model"


def test_openai_agent_retries_next_server(monkeypatch):
    calls = []

    class FakeCompletions:
        def __init__(self, base_url):
            self.base_url = base_url

        def create(self, **kwargs):
            calls.append(self.base_url)
            if self.base_url == "http://server-a/v1":
                raise TimeoutError("server overloaded")
            message = SimpleNamespace(content=json.dumps({"proposals": []}))
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(
                choices=[choice],
                model=kwargs["model"],
                id="response-ok",
                created=0,
                usage=None,
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=FakeCompletions(kwargs["base_url"])
            )
            self.models = FakeOpenAIModels(["unused-model"])

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(
            agent_server_url="http://server-a/v1,http://server-b/v1",
            agent_model_name="served-agent-model",
            agent_schema_repair_attempts=0,
            agent_request_max_retries=1,
            agent_retry_initial_delay=0.0,
        )
    )
    agent._ensure_client()
    agent._client_pool._next_index = 0

    proposals = agent.propose({"current_features": [], "iteration_feedback": []})

    assert proposals == []
    assert calls == ["http://server-a/v1", "http://server-b/v1"]


def test_openai_agent_repairs_missing_required_proposal_fields():
    client = FakeOpenAIClient(
        [
            json.dumps(
                {
                    "proposals": [
                        {
                            "action": "add",
                            "name": "baseline_age",
                            "type": "continuous",
                            "description": "Age in years at treatment initiation",
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "proposals": [
                        {
                            "action": "add",
                            "name": "baseline_age",
                            "type": "continuous",
                            "roles": ["confounder"],
                            "description": "Age in years at treatment initiation",
                            "rationale": "Age may affect treatment selection",
                            "expected_signal": "treatment",
                        }
                    ]
                }
            ),
        ]
    )
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(agent_schema_repair_attempts=1)
    )
    agent._client = client

    proposals = agent.propose({"current_features": [], "iteration_feedback": []})

    assert proposals[0]["roles"] == ["confounder"]
    assert len(client.completions.calls) == 2
    repair_message = client.completions.calls[1]["messages"][-1]["content"]
    assert "baseline_age" in repair_message
    assert "missing roles" in repair_message
    assert len(agent.last_response_trace["repair_attempts"]) == 2


def test_openai_agent_repairs_malformed_json_response():
    client = FakeOpenAIClient(
        [
            "not valid json",
            json.dumps(
                {
                    "proposals": [
                        {
                            "action": "add",
                            "name": "baseline_albumin",
                            "type": "continuous",
                            "roles": ["confounder"],
                            "description": "Baseline serum albumin before treatment",
                        }
                    ]
                }
            ),
        ]
    )
    agent = OpenAICompatibleFeatureSearchAgent(
        AgenticFeatureSearchConfig(agent_schema_repair_attempts=1)
    )
    agent._client = client

    proposals = agent.propose({"current_features": [], "iteration_feedback": []})

    assert proposals[0]["name"] == "baseline_albumin"
    assert len(client.completions.calls) == 2
    repair_message = client.completions.calls[1]["messages"][-1]["content"]
    assert "malformed JSON" in repair_message


class FakeAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        return [
            {
                "action": "add",
                "name": "hidden_modifier",
                "type": "continuous",
                "roles": ["effect_modifier"],
                "description": "Baseline hidden modifier measured before treatment",
                "rationale": "Could explain treatment effect heterogeneity",
                "expected_signal": "tau signal",
            }
        ]


class TracedAgent(FakeAgent):
    def propose(self, context):
        proposals = super().propose(context)
        self.last_response_trace = {
            "raw_content": (
                "I considered baseline variables first.\n"
                '{"proposals": [{"action": "add", "name": "hidden_modifier"}]}'
            ),
            "reasoning_content": "Baseline hidden modifier should improve tau signal.",
            "finish_reason": "stop",
        }
        return proposals


class FakeExtractionProvider:
    def ensure_features(self, dataset, specs):
        dataset = dataset.copy()
        for spec in specs:
            value_col = f"explicit_feat_{spec.name}"
            missing_col = f"{value_col}_missing"
            if value_col in dataset.columns:
                continue
            if spec.type == "categorical":
                dataset[value_col] = spec.categories[0]
            else:
                dataset[value_col] = np.arange(len(dataset), dtype=float)
            dataset[missing_col] = False
        return dataset


class FakeEvaluator:
    def evaluate_split(self, train_df, test_df, specs, fold_id):
        has_hidden = any(spec.name == "hidden_modifier" for spec in specs)
        r_loss = 0.50 if has_hidden else 1.00
        predictions = test_df.copy()
        predictions["pred_ite_prob"] = 0.10 if has_hidden else 0.0
        predictions["pred_y0_prob"] = 0.40
        predictions["pred_y1_prob"] = 0.50
        predictions["pred_propensity_prob"] = 0.50
        predictions["cv_fold"] = fold_id
        metrics = {
            "fold": fold_id,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_explicit_features": len(specs),
            "n_x_features": int(has_hidden),
            "n_w_features": 1,
            "ate_estimate": 0.10 if has_hidden else 0.0,
            "r_loss": r_loss,
            "outcome_auroc": 0.70,
            "treatment_auroc": 0.75,
            "oracle_true_ite_corr": 0.99,
        }
        return SplitEvaluation(predictions=predictions, metrics=metrics)


def test_agentic_runner_resolves_aliases_then_harmonizes_values(tmp_path):
    class HarmonizationAgent:
        supports_alias_resolution = True
        supports_value_harmonization = True

        def __init__(self):
            self.contexts = []

        def propose(self, context):
            self.contexts.append(context)
            if context["prompt_version"] == "multi_model_agentic_alias_resolution_v1":
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
                            "rationale": "Both names refer to the same PD-L1 target.",
                        }
                    ],
                    "unmerged": [],
                }
            if context["prompt_version"] == "multi_model_agentic_value_harmonization_v1":
                return {
                    "features": [
                        {
                            "name": "age",
                            "type": "continuous",
                            "categories": None,
                            "description": "Patient age at treatment initiation in years.",
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
                        },
                    ]
                }
            return []

    df = pd.DataFrame(
        {
            "clinical_text": ["age 55 pd-l1 high", "age 70 pd-l1 low"],
            "treatment_indicator": [1, 0],
            "outcome_indicator": [1, 0],
        }
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=_base_specs(),
        ),
    )
    agent = HarmonizationAgent()
    runner = AgenticFeatureSearchRunner(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )
    selected = [
        *_base_specs(),
        ExplicitFeatureSpec(
            name="pd_l1_expression",
            type="categorical",
            categories=["low", "high", "unknown"],
            description="Pretreatment PD-L1 expression.",
            roles=["effect_modifier"],
        ),
        ExplicitFeatureSpec(
            name="pd_l1_expression_level",
            type="categorical",
            categories=["<1%", "1-49%", ">=50%"],
            description="Pretreatment tumor PD-L1 expression category.",
            roles=["effect_modifier"],
        ),
    ]

    resolved = runner._resolve_selected_aliases(outer_fold=1, selected_specs=selected)
    harmonized = runner._harmonize_value_contracts(
        outer_fold=1,
        selected_specs=resolved,
    )

    assert [spec.name for spec in harmonized] == ["age", "pd_l1_expression"]
    pdl1 = next(spec for spec in harmonized if spec.name == "pd_l1_expression")
    assert pdl1.categories == ["<1%", "1-49%", ">=50%"]
    assert "unknown" not in pdl1.categories
    assert pdl1.value_aliases[">=50%"] == ["high", "50% or greater"]
    age = next(spec for spec in harmonized if spec.name == "age")
    assert age.type == "continuous"
    assert "numeric value only" in age.description
    assert [context["prompt_version"] for context in agent.contexts] == [
        "multi_model_agentic_alias_resolution_v1",
        "multi_model_agentic_value_harmonization_v1",
    ]
    assert [event["event"] for event in runner.decision_events] == [
        "alias_resolution",
        "value_harmonization",
    ]


class FeedbackAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        name = "weak_modifier" if context["iteration"] == 1 else "strong_modifier"
        return [
            {
                "action": "add",
                "name": name,
                "type": "continuous",
                "roles": ["effect_modifier"],
                "description": f"Baseline {name} measured before treatment",
                "rationale": "Could explain treatment effect heterogeneity",
                "expected_signal": "tau signal",
            }
        ]


class RejectThenAcceptEvaluator:
    def evaluate_split(self, train_df, test_df, specs, fold_id):
        names = {spec.name for spec in specs}
        if "strong_modifier" in names:
            r_loss = 0.50
        elif "weak_modifier" in names:
            r_loss = 0.995
        else:
            r_loss = 1.00

        predictions = test_df.copy()
        predictions["pred_ite_prob"] = 0.0
        predictions["pred_y0_prob"] = 0.40
        predictions["pred_y1_prob"] = 0.50
        predictions["pred_propensity_prob"] = 0.50
        predictions["cv_fold"] = fold_id
        metrics = {
            "fold": fold_id,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_explicit_features": len(specs),
            "n_x_features": int(bool(names - {"age"})),
            "n_w_features": 1,
            "ate_estimate": 0.0,
            "r_loss": r_loss,
            "outcome_auroc": 0.70,
            "treatment_auroc": 0.75,
        }
        return SplitEvaluation(predictions=predictions, metrics=metrics)


class BroadCandidateAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        return [
            {
                "action": "add",
                "name": "strong_confounder",
                "type": "continuous",
                "roles": ["confounder", "effect_modifier"],
                "description": "Strong baseline confounder",
            },
            {
                "action": "add",
                "name": "strong_modifier",
                "type": "continuous",
                "roles": ["confounder", "effect_modifier"],
                "description": "Strong baseline effect modifier",
            },
            {
                "action": "add",
                "name": "low_coverage_feature",
                "type": "continuous",
                "roles": ["confounder"],
                "description": "Low coverage baseline variable",
            },
            {
                "action": "add",
                "name": "noise_feature",
                "type": "continuous",
                "roles": ["confounder", "effect_modifier"],
                "description": "Noise baseline variable",
            },
        ]


class BroadScreenEvaluator:
    def evaluate_split(self, train_df, test_df, specs, fold_id):
        names = {spec.name for spec in specs}
        selected_signal_count = int("strong_confounder" in names) + int(
            "strong_modifier" in names
        )
        r_loss = 1.0 - 0.30 * selected_signal_count
        predictions = test_df.copy()
        predictions["pred_ite_prob"] = 0.0
        predictions["pred_y0_prob"] = 0.40
        predictions["pred_y1_prob"] = 0.50
        predictions["pred_propensity_prob"] = 0.50
        predictions["cv_fold"] = fold_id
        metrics = {
            "fold": fold_id,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_explicit_features": len(specs),
            "n_x_features": int("strong_modifier" in names),
            "n_w_features": int("strong_confounder" in names),
            "ate_estimate": 0.0,
            "r_loss": r_loss,
            "outcome_auroc": 0.70,
            "treatment_auroc": 0.75,
        }
        return SplitEvaluation(predictions=predictions, metrics=metrics)


class CountingSourceExtractionProvider:
    def __init__(self):
        self.calls = []
        self.call_descriptions = []

    def ensure_features(self, dataset, specs):
        dataset = dataset.copy()
        missing_specs = [
            spec
            for spec in specs
            if f"explicit_feat_{spec.name}" not in dataset.columns
        ]
        if missing_specs:
            self.calls.append([spec.name for spec in missing_specs])
            self.call_descriptions.append([spec.description for spec in missing_specs])
        for spec in missing_specs:
            value_col = f"explicit_feat_{spec.name}"
            source_col = f"source_{spec.name}"
            values = (
                dataset[source_col].values
                if source_col in dataset.columns
                else np.arange(len(dataset), dtype=float)
            )
            dataset[value_col] = values
            dataset[f"{value_col}_missing"] = pd.isna(values)
        return dataset


class InventoryThenNoneAgent:
    def __init__(self, provider=None, proposals=None):
        self.contexts = []
        self.provider = provider
        self.provider_call_counts = []
        self.proposals = proposals or [
            {
                "action": "add",
                "name": "hidden_modifier",
                "type": "continuous",
                "roles": ["effect_modifier"],
                "description": "Baseline hidden modifier measured before treatment",
            }
        ]

    def propose(self, context):
        self.contexts.append(context)
        if self.provider is not None:
            self.provider_call_counts.append(len(self.provider.calls))
        if context.get("broad_screen_stage") == "inventory":
            return self.proposals
        return [{"action": "none"}]


class FoldSpecificDuplicateAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context.get("broad_screen_stage") != "inventory":
            return [{"action": "none"}]
        if context["outer_fold"] == 1:
            return [
                {
                    "action": "add",
                    "name": "shared_feature",
                    "type": "continuous",
                    "roles": ["confounder"],
                    "description": "First extraction contract",
                }
            ]
        return [
            {
                "action": "add",
                "name": "shared_feature",
                "type": "continuous",
                "roles": ["effect_modifier"],
                "description": "Second extraction contract",
            }
        ]


class SelectByNameAgent:
    def __init__(self, inventory_name="strong_confounder"):
        self.contexts = []
        self.inventory_name = inventory_name

    def propose(self, context):
        self.contexts.append(context)
        if context.get("broad_screen_stage") == "inventory":
            return [
                {
                    "action": "add",
                    "name": self.inventory_name,
                    "type": "continuous",
                    "roles": ["confounder", "effect_modifier"],
                    "description": f"Baseline {self.inventory_name}",
                }
            ]
        return [{"action": "add", "name": self.inventory_name}]


class NewFeatureAfterInventoryAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context.get("broad_screen_stage") == "inventory":
            return [
                {
                    "action": "add",
                    "name": "noise_feature",
                    "type": "continuous",
                    "roles": ["confounder"],
                    "description": "Noise baseline variable",
                }
            ]
        return [
            {
                "action": "add",
                "name": "new_signal",
                "type": "continuous",
                "roles": ["effect_modifier"],
                "description": "New baseline signal not in the extracted shortlist",
            }
        ]


class SequentialBroadSelectionAgent:
    def __init__(self):
        self.contexts = []

    def propose(self, context):
        self.contexts.append(context)
        if context.get("broad_screen_stage") == "inventory":
            return [
                {
                    "action": "add",
                    "name": "strong_confounder",
                    "type": "continuous",
                    "roles": ["confounder", "effect_modifier"],
                    "description": "Strong baseline confounder",
                },
                {
                    "action": "add",
                    "name": "strong_modifier",
                    "type": "continuous",
                    "roles": ["confounder", "effect_modifier"],
                    "description": "Strong baseline effect modifier",
                },
            ]
        name = "strong_confounder" if context["iteration"] == 1 else "strong_modifier"
        return [{"action": "add", "name": name}]


class NewSignalEvaluator(BroadScreenEvaluator):
    def evaluate_split(self, train_df, test_df, specs, fold_id):
        result = super().evaluate_split(train_df, test_df, specs, fold_id)
        if any(spec.name == "new_signal" for spec in specs):
            result.metrics["r_loss"] = 0.40
        return result


def _broad_signal_df(n=120):
    rng = np.random.default_rng(456)
    strong_confounder = rng.normal(size=n)
    strong_modifier = rng.normal(size=n)
    noise = rng.normal(size=n)
    new_signal = rng.normal(size=n)
    treatment = (strong_confounder + rng.normal(scale=0.15, size=n) > 0).astype(int)
    outcome = (
        0.8 * strong_confounder
        + 0.2 * treatment
        + 1.6 * treatment * strong_modifier
        + rng.normal(scale=0.05, size=n)
    )
    low_missing = np.arange(n, dtype=float)
    low_missing_mask = np.arange(n) % 4 != 0
    low_missing[low_missing_mask] = np.nan
    df = pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "clinical_text": [f"Patient {i}" for i in range(n)],
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
            "source_age": np.linspace(50, 80, n),
            "source_hidden_modifier": np.arange(n, dtype=float),
            "source_shared_feature": strong_confounder,
            "source_strong_confounder": strong_confounder,
            "source_strong_modifier": strong_modifier,
            "source_low_coverage_feature": low_missing,
            "source_noise_feature": noise,
            "source_new_signal": new_signal,
        }
    )
    return df


def _broad_config(tmp_path, **overrides):
    search_kwargs = {
        "outer_folds": 2,
        "inner_folds": 2,
        "search_mode": "broad_screen",
        "broad_candidate_count": 4,
        "broad_screen_top_k": 2,
        "min_feature_coverage": 0.70,
        "role_diagnostic_score_delta_threshold": 0.01,
        "min_r_loss_improvement": 0.01,
        "min_improvement_fold_fraction": 1.0,
    }
    search_kwargs.update(overrides.pop("search_overrides", {}))
    initial_specs = overrides.pop("initial_specs", [])
    return AppliedInferenceConfig(
        outcome_type="continuous",
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=8,
                min_samples_leaf=2,
                honest=False,
                inference=False,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(**search_kwargs),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=initial_specs,
            cache_enabled=False,
        ),
    )


def test_agentic_runner_accepts_inner_cv_improvement_without_true_ite_leakage(tmp_path):
    df = pd.DataFrame(
        {
            "patient_id": np.arange(12),
            "clinical_text": [f"Patient {i}" for i in range(12)],
            "treatment_indicator": [0, 1] * 6,
            "outcome_indicator": [0, 0, 1, 1] * 3,
            "true_ite_prob": np.linspace(-0.1, 0.1, 12),
        }
    )
    agent = FakeAgent()
    output_path = tmp_path / "predictions.parquet"
    clinical_question = (
        "Among patients with advanced NSCLC, what is the effect of immunotherapy "
        "receipt on 6-month response?"
    )
    config = AppliedInferenceConfig(
        clinical_question=clinical_question,
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=8,
                min_samples_leaf=2,
                honest=False,
                inference=False,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                search_mode="iterative",
                max_iterations=1,
                min_r_loss_improvement=0.01,
                min_improvement_fold_fraction=1.0,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=_base_specs(),
            cache_enabled=False,
        ),
    )

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=config,
        output_path=output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )

    results = pd.read_parquet(output_path)
    feature_sets = json.loads(
        (tmp_path / "agentic_feature_search" / "feature_sets.json").read_text()
    )
    selected_names = {
        feature["name"]
        for row in feature_sets
        if row["stage"] == "selected"
        for feature in row["features"]
    }

    assert len(results) == len(df)
    assert "hidden_modifier" in selected_names
    assert all(
        context["clinical_question"] == clinical_question
        for context in agent.contexts
    )
    assert all(
        context["estimand"]
        == {
            "treatment_column": "treatment_indicator",
            "outcome_column": "outcome_indicator",
            "outcome_type": "binary",
        }
        for context in agent.contexts
    )
    assert all("true_ite" not in json.dumps(context) for context in agent.contexts)
    decision_lines = (
        tmp_path / "agentic_feature_search" / "agent_decisions.jsonl"
    ).read_text().splitlines()
    persisted_contexts = [
        json.loads(line)["payload"].get("context", {})
        for line in decision_lines
        if json.loads(line)["event"] == "agent_proposals"
    ]
    assert all(context.get("clinical_text_examples") == [] for context in persisted_contexts)


def test_agentic_runner_checks_coverage_only_for_proposed_features(tmp_path):
    df = pd.DataFrame(
        {
            "patient_id": np.arange(12),
            "clinical_text": [f"Patient {i}" for i in range(12)],
            "treatment_indicator": [0, 1] * 6,
            "outcome_indicator": [0, 0, 1, 1] * 3,
            "explicit_feat_baseline_ldh": np.nan,
            "explicit_feat_baseline_ldh_missing": True,
        }
    )
    agent = FakeAgent()
    output_path = tmp_path / "predictions.parquet"
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=8,
                min_samples_leaf=2,
                honest=False,
                inference=False,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                search_mode="iterative",
                max_iterations=1,
                min_feature_coverage=0.70,
                min_r_loss_improvement=0.01,
                min_improvement_fold_fraction=1.0,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=[
                ExplicitFeatureSpec(
                    name="baseline_ldh",
                    type="continuous",
                    roles=["confounder"],
                    description="Baseline LDH",
                )
            ],
            cache_enabled=False,
        ),
    )

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=config,
        output_path=output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )

    feature_sets = json.loads(
        (tmp_path / "agentic_feature_search" / "feature_sets.json").read_text()
    )
    selected_names = {
        feature["name"]
        for row in feature_sets
        if row["stage"] == "selected"
        for feature in row["features"]
    }
    decision_lines = (
        tmp_path / "agentic_feature_search" / "agent_decisions.jsonl"
    ).read_text().splitlines()
    candidate_payloads = [
        json.loads(line)["payload"]
        for line in decision_lines
        if json.loads(line)["event"] == "candidate_evaluations"
    ]
    hidden_comparisons = [
        item["comparison"]
        for payload in candidate_payloads
        for item in payload
        if item["candidate_id"] == "hidden_modifier"
    ]

    assert "hidden_modifier" in selected_names
    assert hidden_comparisons
    assert all(
        comparison.get("rejection_reason") != "low_feature_coverage"
        for comparison in hidden_comparisons
    )


def test_agentic_runner_can_persist_raw_agent_output_when_enabled(tmp_path):
    df = pd.DataFrame(
        {
            "patient_id": np.arange(12),
            "clinical_text": [f"Patient {i}" for i in range(12)],
            "treatment_indicator": [0, 1] * 6,
            "outcome_indicator": [0, 0, 1, 1] * 3,
        }
    )
    output_path = tmp_path / "predictions.parquet"
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=8,
                min_samples_leaf=2,
                honest=False,
                inference=False,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                search_mode="iterative",
                max_iterations=1,
                min_r_loss_improvement=0.01,
                min_improvement_fold_fraction=1.0,
                save_agent_raw_output=True,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=_base_specs(),
            cache_enabled=False,
        ),
    )

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=config,
        output_path=output_path,
        proposal_agent=TracedAgent(),
        extraction_provider=FakeExtractionProvider(),
        evaluator=FakeEvaluator(),
    )

    decision_lines = (
        tmp_path / "agentic_feature_search" / "agent_decisions.jsonl"
    ).read_text().splitlines()
    proposal_payloads = [
        json.loads(line)["payload"]
        for line in decision_lines
        if json.loads(line)["event"] == "agent_proposals"
    ]

    assert proposal_payloads
    for payload in proposal_payloads:
        assert payload["raw_proposals"]
        assert payload["raw_proposals"][0]["name"] == "hidden_modifier"
        trace = payload["agent_raw_output"]
        assert "I considered baseline variables first." in trace["raw_content"]
        assert trace["reasoning_content"] == (
            "Baseline hidden modifier should improve tau signal."
        )


def test_agentic_runner_feeds_rejection_reasons_to_next_iteration(tmp_path):
    df = pd.DataFrame(
        {
            "patient_id": np.arange(12),
            "clinical_text": [f"Patient {i}" for i in range(12)],
            "treatment_indicator": [0, 1] * 6,
            "outcome_indicator": [0, 0, 1, 1] * 3,
        }
    )
    agent = FeedbackAgent()
    output_path = tmp_path / "predictions.parquet"
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=8,
                min_samples_leaf=2,
                honest=False,
                inference=False,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                search_mode="iterative",
                max_iterations=2,
                min_r_loss_improvement=0.01,
                min_improvement_fold_fraction=1.0,
                stop_after_rejected_iteration=False,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=_base_specs(),
            cache_enabled=False,
        ),
    )

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=config,
        output_path=output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=RejectThenAcceptEvaluator(),
    )

    second_iteration_contexts = [
        context for context in agent.contexts if context["iteration"] == 2
    ]
    assert second_iteration_contexts
    for context in second_iteration_contexts:
        weak_feedback = [
            item
            for item in context["iteration_feedback"]
            if item.get("candidate_id") == "weak_modifier"
        ]
        assert weak_feedback
        assert weak_feedback[-1]["status"] == "rejected"
        assert any(
            check.startswith("r_loss_improvement")
            for check in weak_feedback[-1]["failed_checks"]
        )


def test_broad_screen_runner_screens_then_cv_accepts_candidates(tmp_path):
    rng = np.random.default_rng(123)
    n = 120
    strong_confounder = rng.normal(size=n)
    strong_modifier = rng.normal(size=n)
    noise = rng.normal(size=n)
    treatment = (strong_confounder + rng.normal(scale=0.15, size=n) > 0).astype(int)
    outcome = (
        0.8 * strong_confounder
        + 0.2 * treatment
        + 1.6 * treatment * strong_modifier
        + rng.normal(scale=0.05, size=n)
    )
    low_missing = np.arange(n, dtype=float)
    low_missing_mask = np.arange(n) % 4 != 0
    low_missing[low_missing_mask] = np.nan
    df = pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "clinical_text": [f"Patient {i}" for i in range(n)],
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
            "explicit_feat_strong_confounder": strong_confounder,
            "explicit_feat_strong_confounder_missing": False,
            "explicit_feat_strong_modifier": strong_modifier,
            "explicit_feat_strong_modifier_missing": False,
            "explicit_feat_low_coverage_feature": low_missing,
            "explicit_feat_low_coverage_feature_missing": low_missing_mask,
            "explicit_feat_noise_feature": noise,
            "explicit_feat_noise_feature_missing": False,
        }
    )
    agent = BroadCandidateAgent()
    output_path = tmp_path / "predictions.parquet"
    config = AppliedInferenceConfig(
        outcome_type="continuous",
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=8,
                min_samples_leaf=2,
                honest=False,
                inference=False,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                search_mode="broad_screen",
                broad_candidate_count=4,
                broad_screen_top_k=2,
                min_feature_coverage=0.70,
                role_diagnostic_score_delta_threshold=0.01,
                min_r_loss_improvement=0.01,
                min_improvement_fold_fraction=1.0,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=[],
            cache_enabled=False,
        ),
    )

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=config,
        output_path=output_path,
        proposal_agent=agent,
        extraction_provider=FakeExtractionProvider(),
        evaluator=BroadScreenEvaluator(),
    )

    feature_sets = json.loads(
        (tmp_path / "agentic_feature_search" / "feature_sets.json").read_text()
    )
    selected_names = {
        feature["name"]
        for row in feature_sets
        if row["stage"] == "selected"
        for feature in row["features"]
    }
    assert {"strong_confounder", "strong_modifier"} <= selected_names
    assert "low_coverage_feature" not in selected_names

    screening = pd.read_csv(tmp_path / "agentic_feature_search" / "screening_metrics.csv")
    kept = set(screening.loc[screening["kept_for_cv"], "candidate_id"])
    accepted = set(screening.loc[screening["cv_accepted"], "candidate_id"])
    assert {"strong_confounder", "strong_modifier"} <= kept
    assert {"strong_confounder", "strong_modifier"} <= accepted
    low_coverage_reasons = set(
        screening.loc[
            screening["candidate_id"] == "low_coverage_feature",
            "screening_rejection_reason",
        ]
    )
    assert "low_feature_coverage" in low_coverage_reasons

    decision_lines = (
        tmp_path / "agentic_feature_search" / "agent_decisions.jsonl"
    ).read_text().splitlines()
    decisions = [json.loads(line) for line in decision_lines]
    events = [decision["event"] for decision in decisions]
    assert "broad_screening" in events
    broad_payload = [
        item
        for decision in decisions
        if decision["event"] == "broad_screening"
        for item in decision["payload"]
    ]
    assert {
        item["candidate_id"]
        for item in broad_payload
        if item["cv_accepted"]
    } >= {"strong_confounder", "strong_modifier"}
    assert all(context["search_mode"] == "broad_screen" for context in agent.contexts)
    assert all(context["broad_candidate_count"] == 4 for context in agent.contexts)


def test_broad_screen_runner_union_extracts_candidates_once_across_folds(tmp_path):
    rng = np.random.default_rng(456)
    n = 120
    strong_confounder = rng.normal(size=n)
    strong_modifier = rng.normal(size=n)
    noise = rng.normal(size=n)
    treatment = (strong_confounder + rng.normal(scale=0.15, size=n) > 0).astype(int)
    outcome = (
        0.8 * strong_confounder
        + 0.2 * treatment
        + 1.6 * treatment * strong_modifier
        + rng.normal(scale=0.05, size=n)
    )
    low_missing = np.arange(n, dtype=float)
    low_missing_mask = np.arange(n) % 4 != 0
    low_missing[low_missing_mask] = np.nan
    df = pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "clinical_text": [f"Patient {i}" for i in range(n)],
            "treatment_indicator": treatment,
            "outcome_indicator": outcome,
            "source_strong_confounder": strong_confounder,
            "source_strong_modifier": strong_modifier,
            "source_low_coverage_feature": low_missing,
            "source_noise_feature": noise,
        }
    )

    class CountingExtractionProvider:
        def __init__(self):
            self.calls = []

        def ensure_features(self, dataset, specs):
            dataset = dataset.copy()
            missing_specs = [
                spec
                for spec in specs
                if f"explicit_feat_{spec.name}" not in dataset.columns
            ]
            if missing_specs:
                self.calls.append([spec.name for spec in missing_specs])
            for spec in missing_specs:
                value_col = f"explicit_feat_{spec.name}"
                source_col = f"source_{spec.name}"
                values = (
                    dataset[source_col].values
                    if source_col in dataset.columns
                    else np.arange(len(dataset), dtype=float)
                )
                dataset[value_col] = values
                dataset[f"{value_col}_missing"] = pd.isna(values)
            return dataset

    extraction_provider = CountingExtractionProvider()
    config = AppliedInferenceConfig(
        outcome_type="continuous",
        dataset_path=str(tmp_path / "dataset.parquet"),
        cv_folds=2,
        architecture=ModelArchitectureConfig(
            model_type="agentic_explicit_feature_forest",
            explicit_feature_forest=ExplicitFeatureForestConfig(
                n_estimators=8,
                min_samples_leaf=2,
                honest=False,
                inference=False,
            ),
            agentic_feature_search=AgenticFeatureSearchConfig(
                outer_folds=2,
                inner_folds=2,
                search_mode="broad_screen",
                broad_candidate_count=4,
                broad_screen_top_k=2,
                min_feature_coverage=0.70,
                role_diagnostic_score_delta_threshold=0.01,
                min_r_loss_improvement=0.01,
                min_improvement_fold_fraction=1.0,
            ),
        ),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            features=[],
            cache_enabled=False,
        ),
    )

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=config,
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=BroadCandidateAgent(),
        extraction_provider=extraction_provider,
        evaluator=BroadScreenEvaluator(),
    )

    assert extraction_provider.calls == [
        [
            "strong_confounder",
            "strong_modifier",
            "low_coverage_feature",
            "noise_feature",
        ]
    ]


def test_broad_screen_extracts_initial_and_inventory_once_with_initial_first(tmp_path):
    df = _broad_signal_df()
    provider = CountingSourceExtractionProvider()
    agent = InventoryThenNoneAgent(provider=provider)
    initial_specs = [
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            roles=["confounder"],
            description="Age at treatment initiation",
        )
    ]

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=_broad_config(
            tmp_path,
            initial_specs=initial_specs,
            search_overrides={"max_iterations": 1},
        ),
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=agent,
        extraction_provider=provider,
        evaluator=BroadScreenEvaluator(),
    )

    assert provider.calls[0] == ["age", "hidden_modifier"]
    assert provider.calls == [["age", "hidden_modifier"]]

    inventory_contexts = [
        context
        for context in agent.contexts
        if context["broad_screen_stage"] == "inventory"
    ]
    assert inventory_contexts
    assert all(count == 0 for count in agent.provider_call_counts[: len(inventory_contexts)])
    assert all(context["required_features"][0]["name"] == "age" for context in inventory_contexts)
    assert all("current_inner_cv_metrics" not in context for context in inventory_contexts)
    assert all("extraction_summary" not in context for context in inventory_contexts)


def test_broad_screen_canonicalizes_duplicate_name_contracts_across_folds(tmp_path):
    df = _broad_signal_df()
    provider = CountingSourceExtractionProvider()
    agent = FoldSpecificDuplicateAgent()

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=_broad_config(
            tmp_path,
            search_overrides={"max_iterations": 1, "broad_candidate_count": 2},
        ),
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=agent,
        extraction_provider=provider,
        evaluator=BroadScreenEvaluator(),
    )

    assert provider.calls == [["shared_feature"]]
    assert provider.call_descriptions == [["First extraction contract"]]


def test_broad_screen_agent_selects_extracted_candidate_without_reextraction(tmp_path):
    df = _broad_signal_df()
    provider = CountingSourceExtractionProvider()
    agent = SelectByNameAgent()

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=_broad_config(
            tmp_path,
            search_overrides={
                "max_iterations": 1,
                "broad_candidate_count": 1,
                "broad_screen_top_k": 1,
            },
        ),
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=agent,
        extraction_provider=provider,
        evaluator=BroadScreenEvaluator(),
    )

    assert provider.calls == [["strong_confounder"]]
    selection_contexts = [
        context
        for context in agent.contexts
        if context["broad_screen_stage"] == "selection"
    ]
    assert selection_contexts
    assert selection_contexts[0]["available_extracted_features"][0]["name"] == (
        "strong_confounder"
    )


def test_broad_screen_agent_new_feature_triggers_one_on_demand_extraction(tmp_path):
    df = _broad_signal_df()
    provider = CountingSourceExtractionProvider()

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=_broad_config(
            tmp_path,
            search_overrides={
                "max_iterations": 1,
                "broad_candidate_count": 1,
                "broad_screen_top_k": 1,
            },
        ),
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=NewFeatureAfterInventoryAgent(),
        extraction_provider=provider,
        evaluator=NewSignalEvaluator(),
    )

    assert provider.calls == [["noise_feature"], ["new_signal"]]


def test_broad_screen_respects_max_iterations_for_adaptive_rounds(tmp_path):
    df = _broad_signal_df()
    provider = CountingSourceExtractionProvider()
    agent = SequentialBroadSelectionAgent()

    run_agentic_explicit_feature_forest(
        dataset=df,
        config=_broad_config(
            tmp_path,
            search_overrides={
                "max_iterations": 1,
                "broad_candidate_count": 2,
                "broad_screen_top_k": 2,
            },
        ),
        output_path=tmp_path / "predictions.parquet",
        proposal_agent=agent,
        extraction_provider=provider,
        evaluator=BroadScreenEvaluator(),
    )

    feature_sets = json.loads(
        (tmp_path / "agentic_feature_search" / "feature_sets.json").read_text()
    )
    selected_names = {
        feature["name"]
        for row in feature_sets
        if row["stage"] == "selected"
        for feature in row["features"]
    }
    selection_iterations = {
        context["iteration"]
        for context in agent.contexts
        if context["broad_screen_stage"] == "selection"
    }

    assert provider.calls == [["strong_confounder", "strong_modifier"]]
    assert "strong_confounder" in selected_names
    assert "strong_modifier" not in selected_names
    assert selection_iterations == {1}


def test_experiment_config_parses_agentic_search_config(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame(
        {
            "clinical_text": ["note"],
            "treatment_indicator": [0],
            "outcome_indicator": [0],
        }
    ).to_parquet(dataset_path)
    config = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": str(dataset_path),
                "architecture": {
                    "model_type": "agentic_explicit_feature_forest",
                    "agentic_feature_search": {"outer_folds": 3, "inner_folds": 2},
                },
                "explicit_features": {
                    "features": [
                        {
                            "name": "age",
                            "type": "continuous",
                            "roles": ["confounder"],
                        }
                    ]
                },
            }
        }
    )

    search_config = config.applied_inference.architecture.agentic_feature_search
    assert search_config.outer_folds == 3
    assert search_config.search_mode == "broad_screen"
    assert search_config.broad_candidate_count == 80
    assert search_config.broad_screen_top_k == 20
    assert search_config.agent_max_tokens == 25000
    empty_start = ExperimentConfig.from_dict(
        {
            "applied_inference": {
                "dataset_path": str(dataset_path),
                "architecture": {"model_type": "agentic_explicit_feature_forest"},
                "explicit_features": {"enabled": True, "features": []},
            }
        }
    )
    empty_start.validate()

    with pytest.raises(ValueError, match="requires at least one"):
        ExperimentConfig.from_dict(
            {
                "applied_inference": {
                    "dataset_path": str(dataset_path),
                    "architecture": {"model_type": "explicit_feature_forest"},
                    "explicit_features": {"enabled": True, "features": []},
                }
            }
        ).validate()


def test_oracle_grid_propagates_agentic_raw_output_flag():
    from oracle_experiment_scripts.run_oracle_experiments import generate_experiment_grid

    configs = generate_experiment_grid(
        dataset_paths=[
            "./synthetic_data/example_synthetic_datasets/one_confounder_one_effect_modifier_nsclc_with_structured/"
        ],
        filter_model_types=["agentic_explicit_feature_forest"],
        filter_extractor_types=["agentic_explicit_features"],
        agentic_iteration_options=[1, 2],
        agentic_initial_feature_counts=[0],
        agentic_initial_feature_strategies=["true_first"],
        agentic_stop_after_rejected_iteration_options=[True],
        agentic_save_agent_raw_output=True,
    )

    assert configs
    assert all(config.agentic_save_agent_raw_output for config in configs)
    assert all(config.agentic_search_mode == "broad_screen" for config in configs)
    assert {config.agentic_max_iterations for config in configs} == {1, 2}
    assert all(config.agentic_broad_candidate_count == 80 for config in configs)
    assert all(config.agentic_broad_screen_top_k == 20 for config in configs)
