from __future__ import annotations

import hashlib
import inspect
import json
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
    OpenAICompatibleJsonDiscoveryJobRunner,
    InvalidDiscoveryJsonResponse,
    InvalidDiscoveryTransportResponse,
    parse_strict_json_object,
)


def _job(*, extraction: bool = False) -> DiscoveryJsonJob:
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
            DiscoveryJobSettings.extraction() if extraction else DiscoveryJobSettings.selector()
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
) -> SimpleNamespace:
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        reasoning=reasoning,
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=41,
        completion_tokens=19,
        total_tokens=60,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=7),
    )
    return SimpleNamespace(
        choices=[choice],
        id="response-1",
        model="served-model",
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


def _runner(factory, **overrides):
    kwargs = {
        "server_urls": ["http://one.test/v1"],
        "model_name": "fixed-model-v3",
        "api_key": "test-secret-key",
        "request_timeout": 12.5,
        "max_retries": 0,
        "retry_initial_delay": 0.0,
        "retry_max_delay": 0.0,
        "retry_jitter_fraction": 0.0,
        "max_tokens": MINIMUM_DISCOVERY_MAX_TOKENS,
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
    assert first["max_tokens"] == MINIMUM_DISCOVERY_MAX_TOKENS
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
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": f"{job.job_kind}_response",
            "strict": True,
            "schema": job.response_schema,
        },
    }
    assert request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
        "thinking_token_budget": job.settings.thinking_token_budget,
    }
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


def test_extraction_request_disables_thinking_and_omits_budget():
    factory = _ClientFactory([_response('{"feature_name":"age"}')])
    runner = _runner(factory)
    job = _job(extraction=True)

    assert runner.run_json(job=job) == {"feature_name": "age"}

    request = factory.calls[0]
    assert request["messages"] == job.messages
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "thinking_token_budget" not in request["extra_body"]


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
    [0, 5000, 5001, MINIMUM_DISCOVERY_MAX_TOKENS - 1, True],
)
def test_max_tokens_must_cover_visible_response_and_selector_thinking_budget(max_tokens):
    exception = TypeError if max_tokens is True else ValueError
    with pytest.raises(exception, match="max_tokens"):
        OpenAICompatibleJsonDiscoveryJobRunner(
            server_urls="http://one.test/v1",
            model_name="fixed-model",
            max_tokens=max_tokens,
        )


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
