import pytest
import sys
import types

openai_stub = types.ModuleType("openai")
openai_stub.OpenAI = object
sys.modules.setdefault("openai", openai_stub)

from synthetic_data.config import SyntheticDataConfig
from synthetic_data.cli import _VLLMExtractionClient
from synthetic_data.generator import (
    _build_feature_count_instruction,
    _enforce_feature_count_request,
)
from synthetic_data.vllm_batch_client import VLLMBatchClient, VLLMConfig


def _feature(name, roles):
    return {
        "name": name,
        "type": "continuous",
        "description": name,
        "roles": roles,
    }


def test_role_counts_without_total_default_to_distinct_total():
    features = [
        _feature("age", ["confounder"]),
        _feature("ecog", ["effect_modifier"]),
        _feature("albumin", ["confounder"]),
    ]

    selected = _enforce_feature_count_request(
        features,
        num_features=None,
        num_confounders=1,
        num_effect_modifiers=1,
    )

    assert [feature["name"] for feature in selected] == ["age", "ecog"]


def test_role_counts_with_total_allow_both_role_features():
    features = [
        _feature("ecog", ["confounder", "effect_modifier"]),
        _feature("age", ["confounder"]),
        _feature("pdl1", ["effect_modifier"]),
    ]

    selected = _enforce_feature_count_request(
        features,
        num_features=1,
        num_confounders=1,
        num_effect_modifiers=1,
    )

    assert [feature["name"] for feature in selected] == ["ecog"]


def test_impossible_role_count_request_raises():
    features = [
        _feature("age", ["confounder"]),
        _feature("albumin", ["confounder"]),
    ]

    with pytest.raises(ValueError, match="do not satisfy request"):
        _enforce_feature_count_request(
            features,
            num_features=2,
            num_confounders=1,
            num_effect_modifiers=1,
        )


def test_feature_count_instruction_mentions_separate_roles():
    instruction = _build_feature_count_instruction(
        num_features=8,
        num_confounders=5,
        num_effect_modifiers=5,
    )

    assert "exactly 8 total features" in instruction
    assert "exactly 5 features with the confounder role" in instruction
    assert "exactly 5 features with the effect_modifier role" in instruction
    assert "both roles count toward both role totals" in instruction


def test_config_rejects_impossible_role_counts():
    config = SyntheticDataConfig(
        num_features=3,
        num_confounders=1,
        num_effect_modifiers=1,
    )

    with pytest.raises(ValueError, match="num_features cannot exceed"):
        config.validate()


def test_synthetic_vllm_batch_client_uses_native_reasoning_parser(monkeypatch):
    calls = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr("synthetic_data.vllm_batch_client.LLM", FakeLLM)

    VLLMBatchClient(
        VLLMConfig(
            model_name="openai/gpt-oss-120b",
            tensor_parallel_size=1,
            vllm_reasoning_parser="auto",
        )
    )

    assert calls["reasoning_parser"] == "openai_gptoss"


def test_synthetic_vllm_batch_client_can_disable_reasoning_parser(monkeypatch):
    calls = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr("synthetic_data.vllm_batch_client.LLM", FakeLLM)

    VLLMBatchClient(
        VLLMConfig(
            model_name="openai/gpt-oss-120b",
            tensor_parallel_size=1,
            vllm_reasoning_parser="none",
        )
    )

    assert "reasoning_parser" not in calls


def test_synthetic_vllm_batch_client_strips_harmony_without_marker():
    cleaned = VLLMBatchClient.strip_reasoning_prefix(
        "<|channel|>analysis<|message|>hidden<|end|>"
        "<|channel|>final<|message|>final note"
    )

    assert cleaned == "final note"


def test_synthetic_direct_extraction_client_uses_native_reasoning_parser(monkeypatch):
    calls = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def get_tokenizer(self):
            return object()

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    _VLLMExtractionClient(
        model_name="nvidia/Qwen3.6-35B-A3B-NVFP4",
        tensor_parallel_size=1,
        vllm_reasoning_parser="auto",
    )

    assert calls["reasoning_parser"] == "qwen3"
