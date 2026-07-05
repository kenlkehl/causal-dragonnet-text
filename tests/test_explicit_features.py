import pytest

from oci.config import ExperimentConfig, ExplicitFeatureSpec
from oci.extraction import (
    ExplicitFeatureValue,
    VLLMFeatureExtractor,
    build_extraction_prompt,
    infer_vllm_reasoning_parser,
    parse_extraction_response,
    parse_server_urls,
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

    assert (
        resolve_vllm_reasoning_parser(
            "auto",
            "nvidia/Qwen3.6-35B-A3B-NVFP4",
        )
        == "qwen3"
    )
    assert (
        resolve_vllm_reasoning_parser(
            "deepseek_r1",
            "unknown/model",
        )
        == "deepseek_r1"
    )
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


def test_build_extraction_prompt_truncates_to_note_tail():
    specs = [
        ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
    ]

    prompt = build_extraction_prompt(
        "beginning age 44. " + ("middle " * 20) + "end age 71.",
        specs,
        max_text_length=30,
    )

    assert "end age 71" in prompt
    assert "beginning age 44" not in prompt


def test_parse_extraction_response_maps_categorical_value_aliases():
    specs = [
        ExplicitFeatureSpec(
            name="pd_l1_expression",
            type="categorical",
            categories=["<1%", "1-49%", ">=50%"],
            value_aliases={">=50%": ["high", "50% or greater"]},
            roles=["effect_modifier"],
        ),
    ]

    parsed = parse_extraction_response('{"pd_l1_expression": "high"}', specs)

    assert parsed["pd_l1_expression"].value == ">=50%"
    assert parsed["pd_l1_expression"].is_missing is False


def test_raw_explicit_features_maps_categorical_value_aliases():
    specs = [
        ExplicitFeatureSpec(
            name="pd_l1_expression",
            type="categorical",
            categories=["<1%", "1-49%", ">=50%"],
            value_aliases={">=50%": ["high"]},
            roles=["effect_modifier"],
        ),
    ]
    features, names = get_raw_explicit_features(
        [
            {
                "pd_l1_expression": "high",
                "pd_l1_expression_missing": False,
            }
        ],
        specs,
        role="effect_modifier",
    )

    assert names == [
        "pd_l1_expression_1-49%",
        "pd_l1_expression_>=50%",
        "pd_l1_expression_missing",
    ]
    assert features == [[0.0, 1.0, 0.0]]


def test_vllm_feature_extractor_server_client_uses_request_timeout(monkeypatch):
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
        request_timeout=123.0,
    )

    extractor._init_server_client()

    assert calls["timeout"] == 123.0
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


def test_vllm_feature_extractor_repairs_malformed_extraction_json():
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)

            class Message:
                content = "age: 41"

            if len(calls) == 2:
                Message.content = '{"age": 41}'

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
        max_retries=2,
    )
    extractor._client = FakeClient()

    result = extractor._extract_single_server("Age: 41")

    assert result["age"].value == 41.0
    assert result["age"].is_missing is False
    assert len(calls) == 2
    repair_messages = calls[1]["messages"]
    assert repair_messages[1]["role"] == "assistant"
    assert repair_messages[1]["content"] == "age: 41"
    assert repair_messages[2]["role"] == "user"
    assert "could not be used" in repair_messages[2]["content"]
    assert "malformed JSON" in repair_messages[2]["content"]
    assert '"age": <number-or-null>' in repair_messages[2]["content"]


def test_vllm_feature_extractor_retries_next_server(monkeypatch):
    calls = []

    class FakeCompletions:
        def __init__(self, base_url):
            self.base_url = base_url

        def create(self, **kwargs):
            calls.append(self.base_url)
            if self.base_url == "http://server-a/v1":
                raise TimeoutError("server overloaded")

            class Message:
                content = '{"age": 41}'

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = type(
                "Chat",
                (),
                {"completions": FakeCompletions(kwargs["base_url"])},
            )()

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
        server_url="http://server-a/v1,http://server-b/v1",
        max_retries=2,
        retry_initial_delay=0.0,
    )
    extractor._init_server_client()
    extractor._client_pool._next_index = 0

    result = extractor._extract_single_server("Age: 41")

    assert parse_server_urls("http://server-a/v1, http://server-b/v1") == [
        "http://server-a/v1",
        "http://server-b/v1",
    ]
    assert calls == ["http://server-a/v1", "http://server-b/v1"]
    assert result["age"].value == 41.0
    assert result["age"].is_missing is False


def test_vllm_feature_extractor_server_uses_batch_size_for_concurrency(monkeypatch):
    calls = {"submitted": []}

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers):
            calls["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, text):
            calls["submitted"].append(text)
            return FakeFuture(fn(text))

    monkeypatch.setattr("oci.extraction.explicit_features.ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        "oci.extraction.explicit_features.as_completed",
        lambda futures: list(futures)[::-1],
    )

    extractor = VLLMFeatureExtractor(
        specs=[
            ExplicitFeatureSpec(name="age", type="continuous", roles=["confounder"]),
        ],
        mode="server",
    )
    monkeypatch.setattr(extractor, "_ensure_initialized", lambda: None)

    def fake_extract(text):
        return {
            "age": ExplicitFeatureValue(
                name="age",
                type="continuous",
                value=float(text),
                is_missing=False,
            )
        }

    monkeypatch.setattr(extractor, "_extract_single_server", fake_extract)

    results = extractor.extract(["1", "2", "3"], batch_size=2, show_progress=False)

    assert calls["max_workers"] == 2
    assert calls["submitted"] == ["1", "2", "3"]
    assert [row["age"].value for row in results] == [1.0, 2.0, 3.0]


def test_experiment_config_rejects_old_explicit_confounder_keys():
    with pytest.raises(ValueError, match="explicit_confounders"):
        ExperimentConfig.from_dict(
            {
                "applied_inference": {
                    "dataset_path": "dataset.parquet",
                    "explicit_confounders": {"enabled": True, "confounders": []},
                }
            }
        )

    with pytest.raises(ValueError, match="confounder_forest"):
        ExperimentConfig.from_dict(
            {
                "applied_inference": {
                    "dataset_path": "dataset.parquet",
                    "architecture": {"model_type": "confounder_forest"},
                }
            }
        )
