import pytest

from oci.config import ExperimentConfig, ExplicitFeatureSpec
from oci.extraction import (
    VLLMFeatureExtractor,
    infer_vllm_reasoning_parser,
    parse_extraction_response,
    resolve_vllm_reasoning_parser,
)
from oci.models.explicit_feature_featurizer import (
    filter_specs_by_role,
    get_raw_explicit_features,
)


def test_explicit_feature_roles_are_valid_and_deduped():
    spec = ExplicitFeatureSpec(
        name="ecog_status",
        type="categorical",
        categories=["0", "1", "2"],
        roles=["confounder", "effect_modifier", "confounder"],
    )

    assert spec.roles == ["confounder", "effect_modifier"]

    with pytest.raises(ValueError, match="roles required"):
        ExplicitFeatureSpec(name="age", type="continuous")

    with pytest.raises(ValueError, match="invalid roles"):
        ExplicitFeatureSpec(name="age", type="continuous", roles=["instrument"])


def test_raw_explicit_features_split_by_role_with_overlap():
    specs = [
        ExplicitFeatureSpec(
            name="ecog_status",
            type="categorical",
            categories=["0", "1", "2"],
            roles=["confounder", "effect_modifier"],
        ),
        ExplicitFeatureSpec(
            name="age",
            type="continuous",
            roles=["confounder"],
        ),
        ExplicitFeatureSpec(
            name="marker",
            type="continuous",
            roles=["effect_modifier"],
        ),
    ]
    values = [
        {
            "ecog_status": "1",
            "ecog_status_missing": False,
            "age": 60.0,
            "age_missing": False,
            "marker": 2.0,
            "marker_missing": False,
        },
        {
            "ecog_status": "2",
            "ecog_status_missing": False,
            "age": 70.0,
            "age_missing": False,
            "marker": 4.0,
            "marker_missing": False,
        },
    ]

    confounder_specs = filter_specs_by_role(specs, "confounder")
    effect_specs = filter_specs_by_role(specs, "effect_modifier")
    assert [s.name for s in confounder_specs] == ["ecog_status", "age"]
    assert [s.name for s in effect_specs] == ["ecog_status", "marker"]

    w_features, w_names = get_raw_explicit_features(values, specs, role="confounder")
    x_features, x_names = get_raw_explicit_features(values, specs, role="effect_modifier")

    assert w_names == [
        "ecog_status_1",
        "ecog_status_2",
        "ecog_status_missing",
        "age_normalized",
        "age_missing",
    ]
    assert x_names == [
        "ecog_status_1",
        "ecog_status_2",
        "ecog_status_missing",
        "marker_normalized",
        "marker_missing",
    ]
    assert len(w_features[0]) == 5
    assert len(x_features[0]) == 5
    assert w_features[0][0] == 1.0
    assert x_features[0][0] == 1.0


def test_raw_explicit_features_populates_provided_normalization_dicts():
    specs = [
        ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
    ]
    values = [
        {"age": 60.0, "age_missing": False},
        {"age": 70.0, "age_missing": False},
    ]
    means = {}
    stds = {}

    get_raw_explicit_features(values, specs, continuous_means=means, continuous_stds=stds)

    assert means["age"] == 65.0
    assert stds["age"] == 5.0


def test_vllm_reasoning_parser_inference_and_resolution():
    assert infer_vllm_reasoning_parser("nvidia/Qwen3.6-35B-A3B-NVFP4") == "qwen3"
    assert infer_vllm_reasoning_parser("nvidia/Gemma-4-31B-IT-NVFP4") == "gemma4"
    assert infer_vllm_reasoning_parser("openai/gpt-oss-120b") == "openai_gptoss"
    assert infer_vllm_reasoning_parser("meta-llama/Llama-3.1-8B-Instruct") is None

    assert resolve_vllm_reasoning_parser(
        "auto",
        "nvidia/Qwen3.6-35B-A3B-NVFP4",
    ) == "qwen3"
    assert resolve_vllm_reasoning_parser(
        "deepseek_r1",
        "unknown/model",
    ) == "deepseek_r1"
    assert resolve_vllm_reasoning_parser("none", "nvidia/Gemma-4-31B-IT-NVFP4") is None


def test_parse_extraction_response_strips_inline_reasoning_trace():
    specs = [
        ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
    ]

    parsed = parse_extraction_response(
        '<think>{"age": "not the answer"}</think>\n{"age": 71}',
        specs,
    )

    assert parsed["age"].value == 71.0
    assert parsed["age"].is_missing is False


def test_vllm_feature_extractor_server_client_has_no_timeout(monkeypatch):
    calls = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
    )

    extractor._init_server_client()

    assert calls["timeout"] is None
    assert calls["max_retries"] == 0


def test_vllm_feature_extractor_request_does_not_set_timeout():
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)

            class Message:
                content = '{"age": 41}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        class Chat:
            completions = FakeCompletions()

        chat = Chat()

    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
        max_retries=1,
    )
    extractor._client = FakeClient()

    result = extractor._extract_single_server("Age: 41")

    assert "timeout" not in calls
    assert result["age"].value == 41.0
    assert result["age"].is_missing is False


def test_experiment_config_rejects_old_explicit_confounder_keys():
    with pytest.raises(ValueError, match="explicit_confounders"):
        ExperimentConfig.from_dict({
            "applied_inference": {
                "dataset_path": "dataset.parquet",
                "explicit_confounders": {"enabled": True, "confounders": []},
            }
        })

    with pytest.raises(ValueError, match="confounder_forest"):
        ExperimentConfig.from_dict({
            "applied_inference": {
                "dataset_path": "dataset.parquet",
                "architecture": {"model_type": "confounder_forest"},
            }
        })
