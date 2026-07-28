from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from oci.extraction import llm_routing
import oci.inference.openai_compatible_json_discovery_job_runner as runner_module
from oci.inference.hierarchical_all_architecture_discovery import (
    EXTRACTION_DEFINITION_JOB,
    INTERPRET_CHUNK_JOB,
    DiscoveryJobSettings,
    DiscoveryJsonJob,
)
from oci.inference.hierarchical_discovery_response_contract import (
    attach_hierarchical_discovery_response_contract,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    MAX_AUTHENTICATED_RETRIES,
    MINIMUM_DISCOVERY_MAX_TOKENS,
    HIERARCHICAL_GENERATION_JOB_KINDS,
    OpenAICompatibleJsonDiscoveryJobRunner,
    InvalidDiscoveryJsonResponse,
    InvalidDiscoveryTransportResponse,
    Stage2GenerationParameters,
    Stage2GenerationPolicy,
    parse_strict_json_object,
)


def _job(
    *,
    extraction: bool = False,
    selector_thinking_token_budget: int = 5_000,
) -> DiscoveryJsonJob:
    job_kind = EXTRACTION_DEFINITION_JOB if extraction else INTERPRET_CHUNK_JOB
    request = attach_hierarchical_discovery_response_contract(
        job_kind=job_kind,
        request=(
            {
                "job": "define_one_extraction_feature",
                "canonical_name": "age",
                "value_shape_hypothesis": "continuous",
                "supporting_evidence_ids": ["evidence.runner_test"],
            }
            if extraction
            else {
                "job": "interpret_evidence_chunk",
                "evidence": [
                    {
                        "evidence_id": "evidence.runner_test",
                        "member_ids": [],
                    }
                ],
            }
        ),
    )
    return DiscoveryJsonJob.create(
        job_kind=job_kind,
        scope="test_scope",
        dependencies=(),
        settings=(
            DiscoveryJobSettings.extraction()
            if extraction
            else DiscoveryJobSettings.selector(selector_thinking_token_budget)
        ),
        messages=(
            {"role": "system", "content": "Return exactly one JSON object."},
            {"role": "user", "content": runner_module.canonical_json(request)},
        ),
        input_bindings={"catalog_sha256": "1" * 64},
    )


def _response(
    content: str,
    *,
    reasoning_content: str | None = None,
    reasoning: object | None = None,
    model: str = "fixed-model-v3",
    finish_reason: str = "stop",
) -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        reasoning=reasoning,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=41,
        completion_tokens=19,
        total_tokens=60,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
    )
    return SimpleNamespace(
        choices=[choice],
        id="response-1",
        model=model,
        usage=usage,
    )


class _CompletionTransport:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        item = self.owner.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeClient:
    def __init__(self, owner, base_url):
        self.owner = owner
        self.base_url = base_url
        self.chat = SimpleNamespace(completions=_CompletionTransport(owner))
        self.closed = False

    def close(self):
        self.closed = True


class _ClientFactory:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.constructions = []
        self.clients = []

    def __call__(self, **kwargs):
        self.constructions.append(kwargs)
        client = _FakeClient(self, kwargs["base_url"])
        self.clients.append(client)
        return client


def _mutated_client_factory_call(self, **kwargs):
    raise AssertionError("mutated client factory must never be called")


def _post_construction_replacement(self, **kwargs):
    raise AssertionError("replacement client factory must never be called")


class _SelfMutatingClientFactory(_ClientFactory):
    def __call__(self, **kwargs):
        client = _ClientFactory.__call__(self, **kwargs)
        type(self).__call__ = _post_construction_replacement
        return client


def _generation_parameters(
    *,
    max_tokens: int = MINIMUM_DISCOVERY_MAX_TOKENS,
    thinking_enabled: bool,
    thinking_token_budget: int,
    temperature: float = 0.0,
    transport_max_retries: int = 0,
    schema_repair_attempts: int = 1,
) -> Stage2GenerationParameters:
    return Stage2GenerationParameters(
        temperature=temperature,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        seed=42,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        min_tokens=0,
        ignore_eos=False,
        stop_sequences=(),
        stop_token_ids=(),
        include_stop_str_in_output=False,
        logit_bias=(),
        allowed_token_ids=None,
        bad_words=(),
        n=1,
        logprobs=False,
        top_logprobs=0,
        prompt_logprobs=None,
        stream=False,
        use_beam_search=False,
        length_penalty=1.0,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        echo=False,
        add_generation_prompt=True,
        continue_final_message=False,
        add_special_tokens=False,
        include_reasoning=True,
        reasoning_effort=None,
        parallel_tool_calls=False,
        tool_choice="none",
        return_tokens_as_token_ids=False,
        return_token_ids=False,
        return_prompt_text=False,
        thinking_enabled=thinking_enabled,
        thinking_token_budget=thinking_token_budget,
        transport_max_retries=transport_max_retries,
        schema_repair_attempts=schema_repair_attempts,
    )


def _generation_policy(
    *,
    max_tokens: int = MINIMUM_DISCOVERY_MAX_TOKENS,
    selector_thinking_token_budget: int = 5_000,
    transport_max_retries: int = 0,
) -> Stage2GenerationPolicy:
    selector = _generation_parameters(
        max_tokens=max_tokens,
        thinking_enabled=True,
        thinking_token_budget=selector_thinking_token_budget,
        transport_max_retries=transport_max_retries,
    )
    extraction = _generation_parameters(
        max_tokens=max_tokens,
        thinking_enabled=False,
        thinking_token_budget=0,
        transport_max_retries=transport_max_retries,
    )
    return Stage2GenerationPolicy(
        **{
            job_kind: (
                extraction
                if job_kind == EXTRACTION_DEFINITION_JOB
                else selector
            )
            for job_kind in HIERARCHICAL_GENERATION_JOB_KINDS
        },
        feature_proposal_review=selector,
        patient_feature_extraction=extraction,
    )


def _runner(factory, **overrides):
    max_retries = overrides.get("max_retries", 0)
    max_tokens = overrides.pop("max_tokens", MINIMUM_DISCOVERY_MAX_TOKENS)
    selector_budget = overrides.pop("selector_thinking_token_budget", 5_000)
    kwargs = {
        "server_urls": ["http://one.test/v1"],
        "model_name": "fixed-model-v3",
        "api_key": "test-secret-key",
        "request_timeout": 12.5,
        "max_retries": max_retries,
        "retry_initial_delay": 0.0,
        "retry_max_delay": 0.0,
        "retry_jitter_fraction": 0.0,
        "generation_policy": _generation_policy(
            max_tokens=max_tokens,
            selector_thinking_token_budget=selector_budget,
            transport_max_retries=max_retries,
        ),
        "client_factory": factory,
    }
    kwargs.update(overrides)
    return OpenAICompatibleJsonDiscoveryJobRunner(**kwargs)


def test_identity_is_stable_authenticated_secret_free_and_transport_is_lazy():
    factory = _ClientFactory([_response('{"ok":true}')])
    runner = _runner(factory)

    first = runner.identity()
    second = runner.identity()

    assert first == second
    assert factory.constructions == []
    assert first["endpoint_urls"] == ["http://one.test/v1"]
    assert first["model"] == {
        "name": "fixed-model-v3",
        "resolution": "explicit_only_no_autodiscovery",
    }
    assert first["authentication"] == {
        "api_key_mode": "static_api_key",
        "api_key_sha256": hashlib.sha256(b"test-secret-key").hexdigest(),
    }
    assert "test-secret-key" not in json.dumps(first)
    assert first["generation_policy"] == _generation_policy().as_dict()
    assert (
        first["generation_policy_sha256"]
        == _generation_policy().content_sha256
    )
    assert first["generation_policy_resolution"] == "explicit_closed_policy"
    assert first["retry"]["max_attempts"] == 1
    source_path = Path(runner_module.__file__).resolve()
    assert (
        first["implementation"]["file_sha256"]
        == hashlib.sha256(source_path.read_bytes()).hexdigest()
    )
    routing = first["implementation"]["dependencies"]["llm_routing"]
    routing_body = {key: value for key, value in routing.items() if key != "binding_sha256"}
    assert routing == {
        "module": "oci.extraction.llm_routing",
        "file_sha256": hashlib.sha256(
            Path(llm_routing.__file__).resolve().read_bytes()
        ).hexdigest(),
        "binding_sha256": runner_module.content_sha256(routing_body),
    }
    factory_binding = first["client_factory"]
    assert factory_binding["mode"] == "injected_client_factory"
    implementation = factory_binding["implementation"]
    assert implementation["module"] == __name__
    assert implementation["qualname"] == "_ClientFactory"
    assert implementation["kind"] == "callable_instance"
    assert (
        implementation["module_file_sha256"]
        == hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    )
    assert (
        implementation["source_sha256"]
        == hashlib.sha256(inspect.getsource(_ClientFactory).encode()).hexdigest()
    )
    assert {"__init__", "__call__"}.issubset(implementation["code_members"])
    assert implementation["code_members_sha256"] == runner_module.content_sha256(
        implementation["code_members"]
    )
    implementation_body = {
        key: value for key, value in implementation.items() if key != "binding_sha256"
    }
    assert implementation["binding_sha256"] == runner_module.content_sha256(implementation_body)
    factory_body = {key: value for key, value in factory_binding.items() if key != "binding_sha256"}
    assert factory_binding["binding_sha256"] == runner_module.content_sha256(factory_body)
    body = {key: value for key, value in first.items() if key != "identity_sha256"}
    assert first["identity_sha256"] == runner_module.content_sha256(body)


def test_stage2_endpoint_authentication_supports_bearer_and_google_adc_without_secret_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    none = llm_routing.resolve_stage2_endpoint_authentication({})
    assert none.api_key == "EMPTY"
    assert none.identity["mode"] == "none"

    static = llm_routing.resolve_stage2_endpoint_authentication(
        {
            llm_routing.STAGE2_ENDPOINT_AUTH_MODE_ENV: "api_key",
            llm_routing.STAGE2_ENDPOINT_API_KEY_ENV: "remote-secret",
        }
    )
    assert static.api_key == "env:OCI_STAGE2_ENDPOINT_API_KEY"
    assert static.identity == {
        "schema_version": "stage2_endpoint_authentication_v1",
        "mode": "api_key",
        "credential_source": "OCI_STAGE2_ENDPOINT_API_KEY",
    }
    assert "remote-secret" not in json.dumps(static.identity)
    assert llm_routing.resolve_openai_api_key(
        static.api_key,
        {llm_routing.STAGE2_ENDPOINT_API_KEY_ENV: "rotated-secret"},
    ) == "rotated-secret"
    constructions = []
    monkeypatch.setenv(
        llm_routing.STAGE2_ENDPOINT_API_KEY_ENV,
        "runtime-secret",
    )
    pool = llm_routing.OpenAIClientPool(
        server_urls="https://remote.example/v1",
        api_key=static.api_key,
        client_factory=lambda **kwargs: constructions.append(kwargs)
        or SimpleNamespace(close=lambda: None),
    )
    pool.client_for_url("https://remote.example/v1")
    assert constructions[0]["api_key"] == "runtime-secret"

    google = llm_routing.resolve_stage2_endpoint_authentication(
        {llm_routing.STAGE2_ENDPOINT_AUTH_MODE_ENV: "google_adc"}
    )
    assert google.api_key == "GOOGLE_ADC"
    assert google.identity["mode"] == "google_adc"

    with pytest.raises(ValueError, match="required for api_key"):
        llm_routing.resolve_stage2_endpoint_authentication(
            {llm_routing.STAGE2_ENDPOINT_AUTH_MODE_ENV: "api_key"}
        )


def test_stage2_transport_projects_portable_requests_and_keeps_response_identity_separate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        llm_routing.STAGE2_ENDPOINT_TRANSPORT_ENV,
        "openai_compatible",
    )
    transport = llm_routing.resolve_stage2_endpoint_transport()
    assert transport.identity == {
        "schema_version": "stage2_endpoint_transport_v1",
        "mode": "openai_compatible",
    }

    factory = _ClientFactory([_response("{}", model="provider/canonical-v4")])
    pool = llm_routing.OpenAIClientPool(
        server_urls="https://remote.example/v1",
        api_key="EMPTY",
        client_factory=factory,
    )
    response = pool.client_for_url(
        "https://remote.example/v1"
    ).chat.completions.create(
        model="deployment-alias",
        messages=[{"role": "user", "content": "JSON"}],
        max_tokens=100,
        stop=[],
        logprobs=False,
        top_logprobs=0,
        logit_bias={},
        parallel_tool_calls=False,
        tool_choice="none",
        reasoning_effort="medium",
        extra_body={
            "top_k": 20,
            "min_p": 0.1,
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": 5000,
        },
    )

    assert response.model == "provider/canonical-v4"
    assert factory.calls == [
        {
            "model": "deployment-alias",
            "messages": [{"role": "user", "content": "JSON"}],
            "max_tokens": 100,
            "logprobs": False,
            "reasoning_effort": "medium",
        }
    ]
    assert llm_routing.validate_stage2_response_model(
        response.model,
        requested_model="deployment-alias",
    ) == "provider/canonical-v4"
    with pytest.raises(ValueError, match="exact requested vLLM model"):
        llm_routing.validate_stage2_response_model(
            response.model,
            requested_model="deployment-alias",
            transport_mode="vllm",
        )


def test_identity_recomputes_implementation_hash(monkeypatch):
    factory = _ClientFactory([])
    runner = _runner(factory)
    monkeypatch.setattr(runner_module, "_implementation_file_sha256", lambda: "1" * 64)
    first = runner.identity()
    monkeypatch.setattr(runner_module, "_implementation_file_sha256", lambda: "2" * 64)
    second = runner.identity()

    assert first["implementation"]["file_sha256"] == "1" * 64
    assert second["implementation"]["file_sha256"] == "2" * 64
    assert first["identity_sha256"] != second["identity_sha256"]


def test_identity_recomputes_authenticated_llm_routing_hash(monkeypatch):
    runner = _runner(_ClientFactory([]))
    monkeypatch.setattr(runner_module, "_llm_routing_file_sha256", lambda: "1" * 64)
    first = runner.identity()
    monkeypatch.setattr(runner_module, "_llm_routing_file_sha256", lambda: "2" * 64)
    second = runner.identity()

    assert first["implementation"]["dependencies"]["llm_routing"]["file_sha256"] == "1" * 64
    assert second["implementation"]["dependencies"]["llm_routing"]["file_sha256"] == "2" * 64
    assert first["identity_sha256"] != second["identity_sha256"]


def test_generation_policy_closed_schema_rejects_missing_and_extra_fields():
    policy = _generation_policy().as_dict()
    missing_family = dict(policy)
    missing_family.pop("feature_proposal_review")
    with pytest.raises(ValueError, match="missing=.*feature_proposal_review"):
        Stage2GenerationPolicy.from_mapping(missing_family)

    extra_family = {**policy, "unregistered_job_family": {}}
    with pytest.raises(ValueError, match="extra=.*unregistered_job_family"):
        Stage2GenerationPolicy.from_mapping(extra_family)

    missing_parameter = json.loads(json.dumps(policy))
    missing_parameter[INTERPRET_CHUNK_JOB].pop("temperature")
    with pytest.raises(ValueError, match="missing=.*temperature"):
        Stage2GenerationPolicy.from_mapping(missing_parameter)

    extra_parameter = json.loads(json.dumps(policy))
    extra_parameter[INTERPRET_CHUNK_JOB]["unregistered_sampling_knob"] = 0.5
    with pytest.raises(ValueError, match="extra=.*unregistered_sampling_knob"):
        Stage2GenerationPolicy.from_mapping(extra_parameter)


def test_completion_request_generation_schema_rejects_omission_substitution_and_extra():
    parameters = _generation_policy().interpret_architecture_chunk
    request = {
        "model": "fixed-model-v3",
        "messages": [{"role": "user", "content": "complete"}],
        "response_format": {"type": "json_object"},
        **parameters.request_generation_fields(),
    }
    parameters.validate_request_generation_fields(request)

    omitted_null = dict(request)
    omitted_null.pop("reasoning_effort")
    with pytest.raises(ValueError, match="missing=.*reasoning_effort"):
        parameters.validate_request_generation_fields(omitted_null)

    substituted = json.loads(json.dumps(request))
    substituted["extra_body"]["min_p"] = 0.25
    with pytest.raises(ValueError, match="generation controls differ"):
        parameters.validate_request_generation_fields(substituted)

    extra = {**request, "best_of": 2}
    with pytest.raises(ValueError, match="extra=.*best_of"):
        parameters.validate_request_generation_fields(extra)

    nested_extra = json.loads(json.dumps(request))
    nested_extra["extra_body"]["unregistered_sampling_knob"] = True
    with pytest.raises(ValueError, match="generation controls differ"):
        parameters.validate_request_generation_fields(nested_extra)


def test_inherited_internal_request_is_completed_before_closed_validation():
    parameters = _generation_policy().interpret_architecture_chunk
    partial = {
        "model": "fixed-model-v3",
        "messages": [{"role": "user", "content": "complete"}],
        "temperature": parameters.temperature,
        "max_tokens": parameters.max_tokens,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": parameters.thinking_token_budget,
        },
    }
    completed = parameters.complete_inherited_request_generation_fields(partial)
    parameters.validate_request_generation_fields(completed)
    assert {
        key: completed[key]
        for key in parameters.request_generation_fields()
    } == parameters.request_generation_fields()

    substituted = dict(partial)
    substituted["temperature"] = 0.25
    with pytest.raises(ValueError, match="inherited completion request"):
        parameters.complete_inherited_request_generation_fields(substituted)


def test_every_generation_parameter_changes_policy_and_runner_identity():
    base = _generation_policy()
    base_selector = base.interpret_architecture_chunk
    selector_mutations = (
        replace(base_selector, temperature=0.25),
        replace(base_selector, top_p=0.9),
        replace(base_selector, top_k=20),
        replace(base_selector, min_p=0.05),
        replace(base_selector, seed=base_selector.seed + 1),
        replace(base_selector, frequency_penalty=0.1),
        replace(base_selector, presence_penalty=0.1),
        replace(base_selector, repetition_penalty=1.1),
        replace(base_selector, min_tokens=1),
        replace(base_selector, ignore_eos=True),
        replace(base_selector, stop_sequences=("END",)),
        replace(base_selector, stop_token_ids=(7,)),
        replace(base_selector, include_stop_str_in_output=True),
        replace(base_selector, logit_bias=(("7", 1.0),)),
        replace(base_selector, allowed_token_ids=(7, 8)),
        replace(base_selector, bad_words=("forbidden",)),
        replace(base_selector, length_penalty=1.1),
        replace(base_selector, skip_special_tokens=False),
        replace(base_selector, spaces_between_special_tokens=False),
        replace(base_selector, include_reasoning=False),
        replace(base_selector, reasoning_effort="low"),
        replace(
            base_selector,
            thinking_enabled=False,
            thinking_token_budget=0,
        ),
        replace(
            base_selector,
            max_tokens=base_selector.max_tokens + 1,
        ),
        replace(
            base_selector,
            thinking_token_budget=base_selector.thinking_token_budget + 1,
        ),
        replace(base_selector, schema_repair_attempts=0),
    )
    mutations = (
        *(
            replace(
                base,
                interpret_architecture_chunk=selector,
            )
            for selector in selector_mutations
        ),
        _generation_policy(transport_max_retries=1),
    )
    base_identity = _runner(
        _ClientFactory([]),
        generation_policy=base,
    ).identity()["identity_sha256"]
    for mutation in mutations:
        assert mutation.content_sha256 != base.content_sha256
        retries = mutation.interpret_architecture_chunk.transport_max_retries
        if any(
            mutation.for_hierarchical_job(kind).transport_max_retries != retries
            for kind in HIERARCHICAL_GENERATION_JOB_KINDS
        ):
            # A runner has one transport retry loop, so a per-family retry
            # mismatch is intentionally constructor-invalid.  The policy
            # identity still changed above.
            continue
        observed = _runner(
            _ClientFactory([]),
            max_retries=retries,
            generation_policy=mutation,
        ).identity()["identity_sha256"]
        assert observed != base_identity


@pytest.mark.parametrize(
    ("policy_overrides", "match"),
    [
        (
            {
                "thinking_enabled": False,
                "thinking_token_budget": 1,
            },
            "disabled thinking",
        ),
        (
            {
                "thinking_enabled": True,
                "thinking_token_budget": 0,
            },
            "enabled thinking",
        ),
        ({"transport_max_retries": 9}, "cannot exceed"),
        ({"schema_repair_attempts": 2}, "cannot exceed one"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"seed": -1}, "seed"),
        ({"min_tokens": 20_001}, "min_tokens"),
        ({"n": 2}, "n equal to one"),
        ({"logprobs": True}, "logprobs disabled"),
        ({"top_logprobs": 1}, "top_logprobs"),
        ({"prompt_logprobs": 1}, "prompt_logprobs disabled"),
        ({"stream": True}, "stream disabled"),
        ({"use_beam_search": True}, "beam search disabled"),
        ({"echo": True}, "echo disabled"),
        ({"add_generation_prompt": False}, "add_generation_prompt enabled"),
        ({"continue_final_message": True}, "continue_final_message disabled"),
        ({"add_special_tokens": True}, "add_special_tokens disabled"),
        ({"parallel_tool_calls": True}, "parallel_tool_calls disabled"),
        ({"tool_choice": "auto"}, "tool_choice"),
        ({"return_token_ids": True}, "return_token_ids disabled"),
    ],
)
def test_generation_parameter_invariants_fail_closed(policy_overrides, match):
    values = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "seed": 42,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_tokens": 20_000,
        "min_tokens": 0,
        "ignore_eos": False,
        "stop_sequences": (),
        "stop_token_ids": (),
        "include_stop_str_in_output": False,
        "logit_bias": (),
        "allowed_token_ids": None,
        "bad_words": (),
        "n": 1,
        "logprobs": False,
        "top_logprobs": 0,
        "prompt_logprobs": None,
        "stream": False,
        "use_beam_search": False,
        "length_penalty": 1.0,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
        "echo": False,
        "add_generation_prompt": True,
        "continue_final_message": False,
        "add_special_tokens": False,
        "include_reasoning": True,
        "reasoning_effort": None,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "return_tokens_as_token_ids": False,
        "return_token_ids": False,
        "return_prompt_text": False,
        "thinking_enabled": True,
        "thinking_token_budget": 5_000,
        "transport_max_retries": 0,
        "schema_repair_attempts": 1,
        **policy_overrides,
    }
    with pytest.raises(ValueError, match=match):
        Stage2GenerationParameters(**values)


def test_injected_factory_code_mutation_changes_identity_and_fails_before_transport(
    monkeypatch,
):
    factory = _ClientFactory([_response('{"ok":true}')])
    runner = _runner(factory)
    approved = runner.identity()
    monkeypatch.setattr(_ClientFactory, "__call__", _mutated_client_factory_call)

    mutated = runner.identity()
    assert mutated["identity_sha256"] != approved["identity_sha256"]
    with pytest.raises(RuntimeError, match="implementation changed"):
        runner.run_json(job=_job())

    assert factory.constructions == []
    assert factory.calls == []


def test_factory_mutation_during_client_construction_fails_before_remote_call():
    factory = _SelfMutatingClientFactory([_response('{"ok":true}')])
    runner = _runner(factory)
    original_call = _SelfMutatingClientFactory.__call__
    try:
        with pytest.raises(RuntimeError, match="implementation changed"):
            runner.run_json(job=_job())
    finally:
        _SelfMutatingClientFactory.__call__ = original_call

    assert len(factory.constructions) == 1
    assert factory.calls == []


def test_injected_factory_object_replacement_fails_before_transport():
    original = _ClientFactory([_response('{"ok":true}')])
    replacement = _ClientFactory([_response('{"ok":true}')])
    runner = _runner(original)
    runner._client_factory = replacement

    with pytest.raises(RuntimeError, match="object changed"):
        runner.run_json(job=_job())

    assert original.constructions == []
    assert replacement.constructions == []
    assert original.calls == []
    assert replacement.calls == []


def test_uninspectable_injected_factory_fails_closed_at_initialization():
    with pytest.raises(ValueError, match="inspectable source file"):
        _runner(len)


def test_selector_request_preserves_messages_and_records_only_content_reasoning_hashes():
    raw_content = '{"concepts":[],"evidence_dispositions":[]}'
    hidden_reasoning = "private reasoning must never enter metadata"
    alternate_reasoning = {"private": "second private trace"}
    factory = _ClientFactory(
        [
            _response(
                raw_content,
                reasoning_content=hidden_reasoning,
                reasoning=alternate_reasoning,
            )
        ]
    )
    runner = _runner(factory)
    job = _job()

    result = runner.run_json(job=job)

    assert result == {"concepts": [], "evidence_dispositions": []}
    assert len(factory.calls) == 1
    request = factory.calls[0]
    assert request["model"] == "fixed-model-v3"
    assert request["messages"] == job.messages
    assert request["temperature"] == 0
    assert request["max_tokens"] == MINIMUM_DISCOVERY_MAX_TOKENS
    expected_generation = (
        _generation_policy()
        .interpret_architecture_chunk
        .request_generation_fields()
    )
    assert {
        key: request[key]
        for key in expected_generation
    } == expected_generation
    assert set(request) == {
        "model",
        "messages",
        "response_format",
        *expected_generation,
    }
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": f"{job.job_kind}_response",
            "strict": True,
            "schema": job.response_schema,
        },
    }
    assert request["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert (
        request["extra_body"]["thinking_token_budget"]
        == job.settings.thinking_token_budget
    )
    metadata = runner.last_execution_metadata
    assert metadata is not None
    attempt = metadata["attempts"][0]
    assert metadata["parsed_response_sha256"] == runner_module.content_sha256(result)
    assert attempt["parsed_response_sha256"] == runner_module.content_sha256(result)
    assert attempt["content_sha256"] == hashlib.sha256(raw_content.encode()).hexdigest()
    assert attempt["reasoning_hashes"] == {
        "reasoning_content_sha256": hashlib.sha256(hidden_reasoning.encode()).hexdigest(),
        "reasoning_sha256": hashlib.sha256(
            runner_module.canonical_json(alternate_reasoning).encode()
        ).hexdigest(),
    }
    serialized_metadata = json.dumps(metadata)
    assert hidden_reasoning not in serialized_metadata
    assert "second private trace" not in serialized_metadata
    assert raw_content not in serialized_metadata
    assert attempt["usage"] == {
        "completion_tokens": 19,
        "prompt_tokens": 41,
        "reasoning_tokens": 7,
        "total_tokens": 60,
    }


def test_selector_thinking_budget_is_configured_and_identity_bound():
    factory = _ClientFactory(
        [_response('{"concepts":[],"evidence_dispositions":[]}')]
    )
    runner = _runner(
        factory,
        selector_thinking_token_budget=6_000,
        max_tokens=26_000,
    )
    job = _job(selector_thinking_token_budget=6_000)

    runner.run_json(job=job)

    identity = runner.identity()
    configured = identity["generation_policy"][INTERPRET_CHUNK_JOB]
    assert configured["thinking_enabled"] is True
    assert configured["thinking_token_budget"] == 6_000
    assert configured["max_tokens"] == 26_000
    assert factory.calls[0]["extra_body"]["thinking_token_budget"] == 6_000


def test_extraction_request_disables_thinking_and_omits_budget():
    factory = _ClientFactory([_response('{"feature_name":"age"}')])
    runner = _runner(factory)
    job = _job(extraction=True)

    assert runner.run_json(job=job) == {"feature_name": "age"}

    request = factory.calls[0]
    assert request["messages"] == job.messages
    expected_generation = (
        _generation_policy()
        .define_one_extraction_feature
        .request_generation_fields()
    )
    assert {
        key: request[key]
        for key in expected_generation
    } == expected_generation
    assert request["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "thinking_token_budget" not in request["extra_body"]


def test_job_settings_must_match_the_exact_generation_family():
    runner = _runner(
        _ClientFactory([]),
        selector_thinking_token_budget=6_000,
        max_tokens=26_000,
    )
    with pytest.raises(ValueError, match="thinking settings differ"):
        runner._request_kwargs(_job(selector_thinking_token_budget=5_000))


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (_response("{}", model="another-model"), "model differs"),
        (_response("{}", finish_reason="length"), "finish_reason"),
    ],
)
def test_response_requires_exact_model_and_stop_before_parsing(response, match):
    runner = _runner(_ClientFactory([response]))
    with pytest.raises(ValueError, match=match):
        runner.run_json(job=_job())


@pytest.mark.parametrize(
    "content,match",
    [
        ('{"x":1,"x":2}', "duplicate JSON key"),
        ('{"nested":{"x":1,"x":2}}', "duplicate JSON key"),
        ('{"x":NaN}', "non-finite JSON constant"),
        ('{"x":Infinity}', "non-finite JSON constant"),
        ('{"x":1e999}', "non-finite JSON number"),
        ("[1,2,3]", "top level"),
        ('```json\n{"x":1}\n```', "exactly one valid JSON object"),
    ],
)
def test_strict_json_parser_rejects_ambiguous_or_nonfinite_responses(content, match):
    with pytest.raises(ValueError, match=match):
        parse_strict_json_object(content)


def test_invalid_json_is_not_retried_and_metadata_contains_no_response_content():
    bad_content = '{"duplicate":1,"duplicate":2}'
    factory = _ClientFactory(
        [_response(bad_content, reasoning_content="do not record this"), _response('{"ok":true}')]
    )
    runner = _runner(factory, max_retries=3)

    with pytest.raises(
        InvalidDiscoveryJsonResponse,
        match="failed strict JSON parsing",
    ) as captured:
        runner.run_json(job=_job())

    assert len(factory.calls) == 1
    assert captured.value.discovery_response_failure_category == "strict_json_parse_failure"
    assert captured.value.failed_response_content == bad_content
    assert bad_content not in str(captured.value)
    metadata = runner.last_execution_metadata
    assert metadata["outcome"] == "invalid_response"
    assert metadata["attempts"][0]["will_retry"] is False
    assert bad_content not in json.dumps(metadata)
    assert "do not record this" not in json.dumps(metadata)


@pytest.mark.parametrize("extraction", [False, True])
def test_raw_transport_bytes_are_enforced_for_every_job_before_json_parsing(
    extraction,
):
    job = _job(extraction=extraction)
    ceiling = job.identifier_ownership["ownership"]["wire_response_budget"][
        "maximum_transport_bytes"
    ]
    oversized_valid_json = (" " * ceiling) + "{}"
    factory = _ClientFactory([_response(oversized_valid_json), _response("{}")])
    runner = _runner(factory, max_retries=3)

    with pytest.raises(
        InvalidDiscoveryTransportResponse,
        match="raw transport-byte budget",
    ) as captured:
        runner.run_json(job=job)

    assert len(factory.calls) == 1
    assert captured.value.discovery_response_failure_category == (
        "raw_transport_budget_failure"
    )
    assert captured.value.failed_response_content == oversized_valid_json
    assert oversized_valid_json not in str(captured.value)
    metadata = runner.last_execution_metadata
    assert metadata["outcome"] == "invalid_response"
    attempt = metadata["attempts"][0]
    assert attempt["raw_transport_bytes"] == ceiling + 2
    assert attempt["content_sha256"] == hashlib.sha256(
        oversized_valid_json.encode("utf-8")
    ).hexdigest()
    assert oversized_valid_json not in json.dumps(metadata)


def test_transient_retries_are_bounded_rotate_endpoints_and_authenticate_request():
    factory = _ClientFactory([TimeoutError("temporary"), _response('{"ok":true}')])
    runner = _runner(
        factory,
        server_urls=["http://one.test/v1", "http://two.test/v1"],
        max_retries=1,
    )

    assert runner.run_json(job=_job()) == {"ok": True}

    assert len(factory.calls) == 2
    assert factory.calls[0] == factory.calls[1]
    metadata = runner.last_execution_metadata
    attempts = metadata["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["endpoint"] != attempts[1]["endpoint"]
    assert attempts[0]["outcome"] == "transport_error"
    assert attempts[0]["retryable"] is True
    assert attempts[0]["will_retry"] is True
    assert attempts[1]["outcome"] == "success"
    assert attempts[0]["request_sha256"] == attempts[1]["request_sha256"]
    assert attempts[0]["runner_identity_sha256"] == attempts[1]["runner_identity_sha256"]
    assert attempts[0]["request_sha256"] == metadata["request_sha256"]


def test_nonretryable_transport_failure_stops_after_one_attempt():
    factory = _ClientFactory([ValueError("permanent"), _response('{"ok":true}')])
    runner = _runner(factory, max_retries=2)

    with pytest.raises(ValueError, match="permanent"):
        runner.run_json(job=_job())

    assert len(factory.calls) == 1
    metadata = runner.last_execution_metadata
    assert metadata["outcome"] == "transport_error"
    assert metadata["attempts"][0]["retryable"] is False
    assert metadata["attempts"][0]["will_retry"] is False


@pytest.mark.parametrize("model_name", ["", "auto", "AUTODISCOVER", "server", "default"])
def test_model_must_be_explicit(model_name):
    with pytest.raises(ValueError, match="autodiscovery is forbidden"):
        OpenAICompatibleJsonDiscoveryJobRunner(
            server_urls="http://one.test/v1",
            model_name=model_name,
        )


def test_model_must_be_a_string():
    with pytest.raises(TypeError, match="explicit string"):
        OpenAICompatibleJsonDiscoveryJobRunner(
            server_urls="http://one.test/v1",
            model_name=None,
        )


@pytest.mark.parametrize(
    "max_tokens",
    [0, 5000, True],
)
def test_max_tokens_must_leave_a_visible_token_after_selector_thinking(max_tokens):
    exception = TypeError if max_tokens is True else ValueError
    with pytest.raises(exception, match="max_tokens"):
        OpenAICompatibleJsonDiscoveryJobRunner(
            server_urls="http://one.test/v1",
            model_name="fixed-model",
            max_tokens=max_tokens,
        )


def test_each_job_enforces_its_authenticated_dynamic_wire_budget():
    runner = OpenAICompatibleJsonDiscoveryJobRunner(
        server_urls="http://one.test/v1",
        model_name="fixed-model",
        max_tokens=5_001,
    )
    with pytest.raises(ValueError, match="authenticated visible-response"):
        runner._request_kwargs(_job())


def test_retry_count_has_a_hard_upper_bound():
    with pytest.raises(ValueError, match=f"between 0 and {MAX_AUTHENTICATED_RETRIES}"):
        OpenAICompatibleJsonDiscoveryJobRunner(
            server_urls="http://one.test/v1",
            model_name="fixed-model",
            max_retries=MAX_AUTHENTICATED_RETRIES + 1,
        )


def test_close_only_closes_lazily_constructed_clients():
    factory = _ClientFactory([_response('{"ok":true}')])
    runner = _runner(factory)
    runner.close()
    assert factory.clients == []

    runner.run_json(job=_job())
    assert len(factory.clients) == 1
    runner.close()
    assert factory.clients[0].closed is True
