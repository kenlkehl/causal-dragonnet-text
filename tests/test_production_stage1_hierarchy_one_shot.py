from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from oci.config import AppliedInferenceConfig, ExplicitFeatureExtractionConfig
from oci.inference import production_stage1_hierarchy_one_shot as subject
from oci.inference.all_evidence_fusion_runner import (
    AllEvidenceFusionRunResult,
    AllEvidenceFusionRunner,
)
from oci.inference.production_stage1_hierarchy_one_shot import (
    PRODUCTION_STAGE1_HIERARCHY_ONE_SHOT_ATTESTATION_SCHEMA,
    ProductionSingleEndpointFeatureSearchAgent,
    ProductionSingleEndpointJsonDiscoveryJobRunner,
    ProductionStage1HierarchyOneShotOptions,
    _content_sha256,
    _seal_result_attestation,
    _stable_sha256,
    _validate_fresh_roots,
    build_parser,
    run_production_stage1_hierarchy_one_shot,
    validate_exact_model_name,
    validate_production_openai_endpoint,
    validate_single_openai_compatible_endpoint,
)
from oci.inference.neural_query_agentic_forest import NeuralQueryAgenticForestConfig
from oci.inference.production_stage1_hierarchy_handoff import (
    AuthenticatedProductionStage1HierarchyHandoff,
)

TEST_ENDPOINT = "https://llm.example.test:8443/v1"
TEST_MODEL = "publisher/served-model"


def _options(tmp_path: Path) -> ProductionStage1HierarchyOneShotOptions:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / "bundle_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return ProductionStage1HierarchyOneShotOptions(
        bundle_manifest_path=manifest,
        output_dir=tmp_path / "execution",
        preparation_dir=tmp_path / "preparation",
        attestation_dir=tmp_path / "attestation",
        endpoint=TEST_ENDPOINT,
        model_name=TEST_MODEL,
        review_rounds=1,
    )


def _wrapped(path: Path, body: dict[str, object], *, schema: str = "test_v1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": schema,
                "content_sha256": _content_sha256(body),
                "body": body,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_cli_has_no_digest_approval_or_prepare_only_argument() -> None:
    parser = build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    forbidden = {
        option
        for option in options
        if any(token in option for token in ("digest", "approve", "approval", "prepare", "replay"))
    }
    assert forbidden == set()
    assert "--model-identity-json" not in options
    assert "--attestation-dir" in options
    assert {"--endpoint", "--model"}.issubset(options)


@pytest.mark.parametrize(
    "value",
    (
        "http://camus:8010/v1",
        "http://localhost:8010/v1",
        "http://127.0.0.1:8010/v1",
        "https://remote.example:8443/openai/v1",
        "https://[2001:db8::1]:8443/v1",
    ),
)
def test_endpoint_accepts_one_operator_selected_canonical_url(value: str) -> None:
    assert validate_single_openai_compatible_endpoint(value) == value
    assert validate_production_openai_endpoint(value) == value


@pytest.mark.parametrize(
    "value",
    (
        "http://camus:8010/v1/",
        "http://CAMUS:8010/v1",
        "http://user:secret@camus:8010/v1",
        "http://camus:8010/v1?query=1",
        "http://camus:8010/v1#fragment",
        "http://camus:8010/v1,http://camus:8010/v1",
        " http://camus:8010/v1",
        "http://camus:8010/v1 ",
        "http://camus:8010/v1\x00",
        "http://camus..internal:8010/v1",
        "http://camus.internal.:8010/v1",
        "http://camus:99999/v1",
        "http://camus:8010/v1/../other",
        "http://camus:8010/%76%31",
        "ftp://camus:8010/v1",
        "http:///v1",
        "camus:8010/v1",
    ),
)
def test_endpoint_rejects_noncanonical_pool_or_ambiguous_values(value: str) -> None:
    with pytest.raises(ValueError):
        validate_single_openai_compatible_endpoint(value)


@pytest.mark.parametrize(
    "value",
    ("", "auto", "default", " model", "model ", "model,pool", "model\x00name", "a\nb"),
)
def test_model_name_must_be_one_exact_explicit_value(value: str) -> None:
    with pytest.raises(ValueError):
        validate_exact_model_name(value)
    assert validate_exact_model_name(TEST_MODEL) == TEST_MODEL


def test_production_paths_never_reinterpret_literal_model_as_autodiscovery(
    tmp_path: Path,
) -> None:
    literal = "Qwen/Qwen3.6-27B"
    agent = ProductionSingleEndpointFeatureSearchAgent(
        subject.AgenticFeatureSearchConfig(
            agent_server_url=TEST_ENDPOINT,
            agent_model_name=literal,
        )
    )
    assert agent._resolve_agent_model_inventory() == {TEST_ENDPOINT: literal}
    assert agent._resolve_agent_model_name() == literal

    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "cohort.parquet"),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            vllm_mode="server",
            vllm_server_url=TEST_ENDPOINT,
            vllm_model_name=literal,
            cache_dir=str(tmp_path / "cache"),
        ),
    )
    provider = subject.ProductionSingleEndpointExplicitFeatureExtractionProvider(
        config,
        tmp_path / "output",
    )
    assert provider._resolve_vllm_model_inventory() == {TEST_ENDPOINT: literal}
    assert provider._resolve_vllm_model_name() == literal


def _completion_response(
    *,
    model: str = TEST_MODEL,
    finish_reason: str | None = "stop",
    content: str = "{}",
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ],
    )


def test_hierarchy_runner_identity_binds_arbitrary_exact_endpoint_and_model() -> None:
    runner = ProductionSingleEndpointJsonDiscoveryJobRunner(
        server_urls=TEST_ENDPOINT,
        model_name=TEST_MODEL,
        api_key="EMPTY",
    )
    observed = runner.identity()
    assert observed["endpoint_urls"] == [TEST_ENDPOINT]
    assert observed["model"]["name"] == TEST_MODEL
    assert observed["single_endpoint_contract"] == TEST_ENDPOINT
    assert observed["exact_model_contract"] == TEST_MODEL
    assert observed["response_metadata_policy"]["required_finish_reason"] == "stop"
    assert observed["served_deployment_metadata_required"] is False
    assert observed["caller_digest_authority"] is False
    assert observed["external_network_required"] is True
    declared = observed.pop("identity_sha256")
    assert declared == _content_sha256(observed)


@pytest.mark.parametrize(
    ("model", "finish_reason"),
    (("substituted/model", "stop"), (TEST_MODEL, "length"), (TEST_MODEL, None)),
)
def test_hierarchy_response_metadata_is_rejected_before_content(
    model: str,
    finish_reason: str | None,
) -> None:
    runner = ProductionSingleEndpointJsonDiscoveryJobRunner(
        server_urls=TEST_ENDPOINT,
        model_name=TEST_MODEL,
        api_key="EMPTY",
    )
    response = _completion_response(
        model=model,
        finish_reason=finish_reason,
        content="valid or invalid content must remain unread",
    )
    with pytest.raises(ValueError, match="model differs|finish_reason"):
        runner._response_message(response)


@pytest.mark.parametrize(
    ("model", "finish_reason"),
    (("substituted/model", "stop"), (TEST_MODEL, "length")),
)
def test_proposal_review_agent_rejects_response_metadata_before_content(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    finish_reason: str,
) -> None:
    config = subject.AgenticFeatureSearchConfig(
        agent_server_url=TEST_ENDPOINT,
        agent_model_name=TEST_MODEL,
    )
    monkeypatch.setattr(
        subject.OpenAICompatibleFeatureSearchAgent,
        "_create_completion",
        lambda _instance, **_kwargs: _completion_response(
            model=model,
            finish_reason=finish_reason,
            content="must not reach proposal parsing",
        ),
    )
    agent = ProductionSingleEndpointFeatureSearchAgent(config)
    with pytest.raises(ValueError, match="model differs|finish_reason"):
        agent._create_completion(model=TEST_MODEL, messages=[])


@pytest.mark.parametrize(
    ("model", "finish_reason"),
    (("substituted/model", "stop"), (TEST_MODEL, "length")),
)
def test_explicit_extractor_rejects_response_metadata_before_content(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    finish_reason: str,
) -> None:
    response = _completion_response(
        model=model,
        finish_reason=finish_reason,
        content="must not reach extraction parsing",
    )

    class FakeCompletions:
        def create(self, **_kwargs: object) -> object:
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    class FakePool:
        server_urls = [TEST_ENDPOINT]

        @staticmethod
        def client_for_url(url: str) -> object:
            assert url == TEST_ENDPOINT
            return client

        @staticmethod
        def client_for_attempt(_start: int, _attempt: int) -> tuple[str, object]:
            return TEST_ENDPOINT, client

        @staticmethod
        def reserve_start_index() -> int:
            return 0

        @staticmethod
        def close() -> None:
            return None

    def fake_base_init(instance: object) -> None:
        instance._client_pool = FakePool()
        instance._client = client

    monkeypatch.setattr(subject.VLLMFeatureExtractor, "_init_server_client", fake_base_init)
    extractor = subject.ProductionSingleEndpointVLLMFeatureExtractor(
        specs=[],
        mode="server",
        server_url=TEST_ENDPOINT,
        model_name=TEST_MODEL,
        model_names_by_url={TEST_ENDPOINT: TEST_MODEL},
    )
    extractor._init_server_client()
    with pytest.raises(ValueError, match="model differs|finish_reason"):
        extractor._extract_single_server("patient text must remain unread")


def test_roots_are_absolute_fresh_nonnested_and_outside_bundle(tmp_path: Path) -> None:
    options = _options(tmp_path)
    _validate_fresh_roots(options)
    with pytest.raises(ValueError, match="fresh nonexistent"):
        options.attestation_dir.mkdir()
        _validate_fresh_roots(options)
    options.attestation_dir.rmdir()
    nested = ProductionStage1HierarchyOneShotOptions(
        **{
            **options.__dict__,
            "preparation_dir": options.output_dir / "preparation",
        }
    )
    with pytest.raises(ValueError, match="nonnested"):
        _validate_fresh_roots(nested)
    traversing = replace(options, output_dir=tmp_path / "execution" / ".." / "escaped")
    with pytest.raises(ValueError, match="path traversal"):
        _validate_fresh_roots(traversing)
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _validate_fresh_roots(replace(options, output_dir=linked_parent / "execution"))


def test_programmatic_option_validation_precedes_handoff_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = replace(_options(tmp_path), request_timeout=float("nan"))
    loaded = False

    def forbidden_loader(*_args: object, **_kwargs: object) -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError("handoff loader must not run")

    monkeypatch.setattr(subject, "load_production_stage1_hierarchy_handoff", forbidden_loader)
    with pytest.raises(ValueError, match="request_timeout"):
        run_production_stage1_hierarchy_one_shot(options)
    assert loaded is False


def test_failed_handoff_validation_precedes_any_runtime_or_client_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    constructed = False

    def fail_loader(*_args: object, **_kwargs: object) -> object:
        raise ValueError("tampered authenticated bundle")

    def forbidden_builder(**_kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("runtime construction must not start")

    monkeypatch.setattr(
        "oci.inference.production_stage1_hierarchy_one_shot.load_production_stage1_hierarchy_handoff",
        fail_loader,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_hierarchy_one_shot.build_production_stage1_hierarchy_runner",
        forbidden_builder,
    )
    with pytest.raises(ValueError, match="tampered authenticated bundle"):
        run_production_stage1_hierarchy_one_shot(options)
    assert constructed is False


def _binding_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_drift: bool = False,
    htr_drift: bool = False,
    cache_drift: bool = False,
) -> tuple[AuthenticatedProductionStage1HierarchyHandoff, AppliedInferenceConfig]:
    source_config = tmp_path / "source_stage1.json"
    source_config.write_text('{"source":"config"}\n', encoding="utf-8")
    source_sha = hashlib.sha256(source_config.read_bytes()).hexdigest()
    effective_path = tmp_path / "registered_effective_stage1.json"
    effective_path.write_text("{}\n", encoding="utf-8")
    htr_path = tmp_path / "htr_model"
    htr_path.mkdir()
    cache_path = tmp_path / "embedding_cache"
    cache_path.mkdir()
    applied = AppliedInferenceConfig(dataset_path=str(tmp_path / "cohort.parquet"))
    effective_json = json.loads(subject._canonical_json(asdict(applied)))
    query_json = json.loads(subject._canonical_json(asdict(NeuralQueryAgenticForestConfig())))
    expected_cache_identity = {"provider": "sealed-cache", "row_count": 10}
    request = {
        "source_config": {
            "path": str(source_config),
            "sha256": ("0" * 64 if source_drift else source_sha),
        },
        "effective_stage1_config": effective_json,
        "htr_model": {
            "path": str(htr_path),
            "tree_sha256": ("1" * 64 if htr_drift else "2" * 64),
            "sentence_encoder_unfrozen": True,
        },
        "embedding_cache": {
            "path": str(cache_path),
            "identity": expected_cache_identity,
        },
        "query_config": {"effective": query_json, "source": {"provided": False}},
    }
    inputs = SimpleNamespace(
        stage1_config_path=effective_path,
        embedding_cache_dir=cache_path,
        hierarchical_discovery_contract_identity={"content_sha256": "3" * 64},
        _authenticated_registered_json=lambda key: (
            dict(request) if key == "immutable_build_request" else None
        ),
        as_dict=lambda: {"content_sha256": "4" * 64},
    )
    provider = SimpleNamespace(identity=lambda: {"identity_sha256": "5" * 64})
    handoff = AuthenticatedProductionStage1HierarchyHandoff(inputs=inputs, provider=provider)
    snapshot = SimpleNamespace(applied_config=lambda: applied)
    monkeypatch.setattr(
        subject,
        "HistoricalStage1ConfigSnapshot",
        SimpleNamespace(from_path=lambda _path: snapshot),
    )
    monkeypatch.setattr(subject, "_resolve_htr_model_path", lambda _config: htr_path.resolve())
    monkeypatch.setattr(
        subject,
        "PrivateHTRModelTreeSnapshot",
        lambda source: SimpleNamespace(source_path=Path(source).resolve(), sha256="2" * 64),
    )
    observed_cache_identity = (
        {"provider": "drifted-cache", "row_count": 10} if cache_drift else expected_cache_identity
    )
    monkeypatch.setattr(
        subject,
        "SpentOnlyFrozenChunkEmbeddingCache",
        lambda _path: SimpleNamespace(identity=lambda: dict(observed_cache_identity)),
    )
    return handoff, applied


def test_authenticated_runtime_bindings_accept_json_roundtripped_effective_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff, applied = _binding_handoff(tmp_path, monkeypatch)
    (
        request,
        _snapshot,
        observed_applied,
        htr_snapshot,
        cache,
        query_config,
    ) = subject._authenticated_stage1_runtime_bindings(handoff)
    assert json.loads(subject._canonical_json(asdict(observed_applied))) == json.loads(
        subject._canonical_json(request["effective_stage1_config"])
    )
    assert asdict(observed_applied) == asdict(applied)
    assert htr_snapshot.sha256 == "2" * 64
    assert cache.identity() == request["embedding_cache"]["identity"]
    assert (
        json.loads(subject._canonical_json(asdict(query_config)))
        == request["query_config"]["effective"]
    )


def test_fixed_endpoint_is_explicitly_propagated_to_all_three_client_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    dataset_path = tmp_path / "cohort.parquet"
    applied = AppliedInferenceConfig(dataset_path=str(dataset_path), cv_folds=2)
    provider = SimpleNamespace(
        schedule=SimpleNamespace(partitions_by_outer_fold={1: {}, 2: {}}),
        identity=lambda: {"identity_sha256": "5" * 64},
    )
    inputs = SimpleNamespace(
        dataset_path=dataset_path,
        stage1_config_path=tmp_path / "effective_stage1.json",
        embedding_cache_dir=tmp_path / "embedding_cache",
        legacy_handoff_path=tmp_path / "legacy.jsonl",
        tfidf_handoff_path=tmp_path / "tfidf.jsonl",
        primary_splits_path=tmp_path / "primary.parquet",
        hierarchical_discovery_contract_identity={"content_sha256": "3" * 64},
        as_dict=lambda: {"content_sha256": "4" * 64},
    )
    handoff = AuthenticatedProductionStage1HierarchyHandoff(inputs=inputs, provider=provider)
    query_config = NeuralQueryAgenticForestConfig()
    monkeypatch.setattr(
        subject,
        "_authenticated_stage1_runtime_bindings",
        lambda _handoff: (
            {},
            object(),
            applied,
            object(),
            object(),
            query_config,
        ),
    )

    monkeypatch.setattr(subject, "ContextFitNeuralQueryService", lambda **_kwargs: object())
    monkeypatch.setattr(subject, "TfidfTopicOrphanContextBackend", lambda **_kwargs: object())
    monkeypatch.setattr(
        subject,
        "TfidfTopicOrphanSpentDiscoveryBackend",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        subject,
        "build_shared_tfidf_context_fit_backends",
        lambda **_kwargs: SimpleNamespace(context_backend=object()),
    )
    monkeypatch.setattr(subject, "HistoricalStage1ContextBackend", lambda **_kwargs: object())
    monkeypatch.setattr(subject, "NeuralQueryContextBackend", lambda _service: object())
    monkeypatch.setattr(subject, "CompositeContextFitUpstreamBackend", lambda _rows: object())
    monkeypatch.setattr(
        subject,
        "build_coordinate_preserving_final_upstream_schema_config",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        subject,
        "CoordinatePreservingContextFitUpstreamBackend",
        lambda _backend, config: object(),
    )
    gate_provider = object()
    final_producer = object()
    monkeypatch.setattr(
        subject,
        "ContextFitUpstreamGateProvider",
        lambda *_args, **_kwargs: gate_provider,
    )
    monkeypatch.setattr(
        subject,
        "FinalContextFitUpstreamProducer",
        lambda *_args, **_kwargs: final_producer,
    )

    observed: dict[str, object] = {}

    def capture_review_agent(config: object) -> object:
        observed["review_endpoint"] = config.agent_server_url
        observed["review_model"] = config.agent_model_name
        return object()

    def capture_extraction(config: object, _root: Path) -> object:
        observed["extraction_endpoint"] = config.explicit_features.vllm_server_url
        observed["extraction_model"] = config.explicit_features.vllm_model_name
        return object()

    def capture_hierarchy(**kwargs: object) -> object:
        observed["hierarchy_endpoint"] = kwargs["server_urls"]
        observed["hierarchy_model"] = kwargs["model_name"]
        body = {
            "endpoint_urls": [kwargs["server_urls"]],
            "model": {"name": kwargs["model_name"]},
        }
        return SimpleNamespace(identity=lambda: dict(body))

    class CapturedRunner:
        def __init__(self, **kwargs: object):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(subject, "ProductionSingleEndpointFeatureSearchAgent", capture_review_agent)
    monkeypatch.setattr(
        subject,
        "ProductionSingleEndpointExplicitFeatureExtractionProvider",
        capture_extraction,
    )
    monkeypatch.setattr(
        subject,
        "ProductionSingleEndpointJsonDiscoveryJobRunner",
        capture_hierarchy,
    )
    monkeypatch.setattr(subject, "AllEvidenceFusionRunner", CapturedRunner)
    runner = subject.build_production_stage1_hierarchy_runner(
        handoff=handoff,
        options=options,
        endpoint=TEST_ENDPOINT,
    )
    assert observed == {
        "review_endpoint": TEST_ENDPOINT,
        "review_model": TEST_MODEL,
        "extraction_endpoint": TEST_ENDPOINT,
        "extraction_model": TEST_MODEL,
        "hierarchy_endpoint": TEST_ENDPOINT,
        "hierarchy_model": TEST_MODEL,
    }
    assert runner.review_spent_evidence_provider is provider
    assert runner.review_partition_provider is provider
    assert runner.review_gate_source_provider is gate_provider
    assert runner.review_gate_feature_bank_provider is gate_provider


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("source_drift", "source config bytes differ"),
        ("htr_drift", "HTR model tree differs"),
        ("cache_drift", "embedding cache differs"),
    ),
)
def test_request_bound_identity_drift_fails_before_agent_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    handoff, _applied = _binding_handoff(
        tmp_path,
        monkeypatch,
        **{drift: True},
    )
    constructed: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        constructed.append("client-capable")
        raise AssertionError("client-capable constructor must not run")

    monkeypatch.setattr(subject, "ProductionSingleEndpointFeatureSearchAgent", forbidden)
    monkeypatch.setattr(
        subject,
        "ProductionSingleEndpointExplicitFeatureExtractionProvider",
        forbidden,
    )
    monkeypatch.setattr(subject, "ProductionSingleEndpointJsonDiscoveryJobRunner", forbidden)
    with pytest.raises(ValueError, match=message):
        subject.build_production_stage1_hierarchy_runner(
            handoff=handoff,
            options=_options(tmp_path),
            endpoint=TEST_ENDPOINT,
        )
    assert constructed == []


def test_same_provider_and_internal_one_shot_are_the_only_execution_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    provider = object()
    handoff_payload = {
        "manual_digest_approval_required": False,
        "raw_all_architecture_prompt_allowed": False,
        "per_architecture_interpretation_required": True,
        "content_sha256": "a" * 64,
    }
    handoff = SimpleNamespace(
        provider=provider,
        as_dict=lambda: dict(handoff_payload),
    )
    runner = object.__new__(AllEvidenceFusionRunner)
    runner.review_spent_evidence_provider = provider
    runner.review_partition_provider = provider
    runner.hierarchical_discovery_approved_batch_sha256 = None
    called: list[tuple[object, object]] = []

    monkeypatch.setattr(
        "oci.inference.production_stage1_hierarchy_one_shot.load_production_stage1_hierarchy_handoff",
        lambda *_args, **_kwargs: handoff,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_hierarchy_one_shot.build_production_stage1_hierarchy_runner",
        lambda **_kwargs: runner,
    )

    fake_result = AllEvidenceFusionRunResult(
        prediction_path=options.output_dir / "frozen_predictions.parquet",
        run_manifest_path=options.output_dir / "immutable_run_manifest.json",
        fold_manifest_paths=(),
        prediction_sha256="b" * 64,
    )

    def one_shot(*, handoff: object, runner: object) -> AllEvidenceFusionRunResult:
        called.append((handoff, runner))
        return fake_result

    monkeypatch.setattr(
        "oci.inference.production_stage1_hierarchy_one_shot.run_internal_production_stage1_hierarchy_one_shot",
        one_shot,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_hierarchy_one_shot._seal_result_attestation",
        lambda **_kwargs: {"status": "completed"},
    )
    assert run_production_stage1_hierarchy_one_shot(options) == {"status": "completed"}
    assert called == [(handoff, runner)]
    assert runner.review_spent_evidence_provider is provider
    assert runner.review_partition_provider is provider
    assert runner.hierarchical_discovery_approved_batch_sha256 is None


def test_result_attestation_is_closed_and_published_outside_output_tree(tmp_path: Path) -> None:
    options = _options(tmp_path)
    options.output_dir.mkdir()
    options.preparation_dir.mkdir()
    prediction = options.output_dir / "frozen_predictions.parquet"
    prediction.write_bytes(b"frozen-parquet-test-bytes")
    prediction_sha = hashlib.sha256(prediction.read_bytes()).hexdigest()
    run_manifest = _wrapped(
        options.output_dir / "immutable_run_manifest.json",
        {
            "prediction_path": str(prediction.resolve()),
            "prediction_sha256": prediction_sha,
        },
    )
    fold_manifest = _wrapped(
        options.output_dir / "outer_fold_001" / "immutable_fold_manifest.json",
        {"outer_fold": 1},
    )
    _wrapped(
        options.preparation_dir / "authenticated_hierarchical_batch_result.json",
        {"batch_result_sha256": "c" * 64},
    )
    provider = SimpleNamespace(identity=lambda: {"identity_sha256": "d" * 64})
    handoff = SimpleNamespace(
        inputs=SimpleNamespace(
            bundle_manifest_path=options.bundle_manifest_path,
            bundle_sha256="e" * 64,
        ),
        provider=provider,
        as_dict=lambda: {"content_sha256": "f" * 64},
    )
    hierarchy_runner = SimpleNamespace(
        identity=lambda: {
            "identity_sha256": "1" * 64,
            "endpoint_urls": [options.endpoint],
            "model": {"name": options.model_name},
        }
    )
    runner = SimpleNamespace(
        hierarchical_discovery_runner=hierarchy_runner,
        review_spent_evidence_provider=provider,
        review_partition_provider=provider,
    )
    result = AllEvidenceFusionRunResult(
        prediction_path=prediction,
        run_manifest_path=run_manifest,
        fold_manifest_paths=(fold_manifest,),
        prediction_sha256=prediction_sha,
    )
    module_path = Path(
        __import__(
            "oci.inference.production_stage1_hierarchy_one_shot",
            fromlist=["__file__"],
        ).__file__
    ).resolve()
    summary = _seal_result_attestation(
        handoff=handoff,
        runner=runner,
        result=result,
        options=options,
        endpoint=options.endpoint,
        implementation_sha256=_stable_sha256(module_path, label="test module")[1],
    )
    attestation_path = Path(summary["attestation_path"])
    assert attestation_path.parent == options.attestation_dir
    assert not attestation_path.is_relative_to(options.output_dir)
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PRODUCTION_STAGE1_HIERARCHY_ONE_SHOT_ATTESTATION_SCHEMA
    declared = payload.pop("content_sha256")
    assert declared == _content_sha256(payload)
    assert payload["genuine_one_shot_e2e_certified"] is False
    assert payload["production_endpoint"] == TEST_ENDPOINT
    assert payload["production_model"] == TEST_MODEL
    assert payload["remote_runtime_identity"]["endpoint_urls"] == [TEST_ENDPOINT]
    assert payload["remote_runtime_identity"]["model"]["name"] == TEST_MODEL
    assert payload["remote_runtime_identity"]["served_deployment_metadata_required"] is False
    assert payload["run_result_audit_record_is_authorization"] is False
    with pytest.raises(FileExistsError):
        _seal_result_attestation(
            handoff=handoff,
            runner=runner,
            result=result,
            options=options,
            endpoint=options.endpoint,
            implementation_sha256=_stable_sha256(module_path, label="test module")[1],
        )
