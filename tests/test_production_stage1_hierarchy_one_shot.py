from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from oci.config import (
    AppliedInferenceConfig,
    ExplicitFeatureExtractionConfig,
    ExplicitFeatureSpec,
)
from oci.inference import production_stage1_hierarchy_one_shot as subject
from oci.inference.all_evidence_fusion_runner import (
    AllEvidenceFusionRunResult,
    AllEvidenceFusionRunner,
)
from oci.inference.all_evidence_post_extraction_review import (
    GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
    CausalReviewConfig,
)
from oci.inference.production_stage1_hierarchy_one_shot import (
    PRODUCTION_STAGE1_HIERARCHY_ONE_SHOT_ATTESTATION_SCHEMA,
    ProductionSingleEndpointFeatureSearchAgent,
    ProductionSingleEndpointJsonDiscoveryJobRunner,
    ProductionStage1HierarchyOneShotOptions,
    Stage2HierarchyPromptProtocol,
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
from oci.inference.hierarchical_discovery_response_contract import (
    LEGACY_HIERARCHY_WIRE_BUDGET,
)
from oci.inference.openai_compatible_json_discovery_job_runner import (
    HIERARCHICAL_GENERATION_JOB_KINDS,
    Stage2GenerationParameters,
    Stage2GenerationPolicy,
)
from oci.inference.post_extraction_scientific_policy import (
    POST_EXTRACTION_SCIENTIFIC_POLICY_VERSION,
    PostExtractionScientificPolicy,
)
from oci.inference.portable_workflow_spec import (
    PostExtractionCausalReviewSpec,
    Stage2PromptProtocolSpec,
)
from oci.inference.production_stage1_hierarchy_handoff import (
    AuthenticatedProductionStage1HierarchyHandoff,
)
from oci.inference.stage2_prompt_nontruncation import (
    Stage2PromptNonTruncationGuard,
)
from oci.models.strict_causal_forest_runtime import (
    STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
    StrictCausalForestRuntimeConfig,
)
from tests.test_portable_workflow_contracts import (
    _forest_operational,
    _forest_spec,
)
from tests.hierarchy_resource_test_support import (
    FIRST_UNTOUCHED_GATE_BOUNDS,
    HIERARCHY_JOB_CACHE_CONFIG,
)

TEST_ENDPOINT = "https://llm.example.test:8443/v1"
TEST_MODEL = "publisher/served-model"


def _generation_parameters(
    *,
    max_tokens: int,
    thinking_enabled: bool,
    thinking_token_budget: int,
    temperature: float = 0.0,
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
        transport_max_retries=0,
        schema_repair_attempts=1,
    )


def _generation_policy(
    *,
    proposal_max_tokens: int = 26_000,
    extraction_max_tokens: int = 23_000,
    selector_thinking_token_budget: int = 6_000,
) -> Stage2GenerationPolicy:
    selector = _generation_parameters(
        max_tokens=proposal_max_tokens,
        thinking_enabled=True,
        thinking_token_budget=selector_thinking_token_budget,
    )
    definition = _generation_parameters(
        max_tokens=proposal_max_tokens,
        thinking_enabled=False,
        thinking_token_budget=0,
    )
    patient = _generation_parameters(
        max_tokens=extraction_max_tokens,
        thinking_enabled=False,
        thinking_token_budget=0,
    )
    return Stage2GenerationPolicy(
        **{
            job_kind: (definition if job_kind == "define_one_extraction_feature" else selector)
            for job_kind in HIERARCHICAL_GENERATION_JOB_KINDS
        },
        feature_proposal_review=selector,
        patient_feature_extraction=patient,
    )


def _stage2_protocol(
    **overrides: object,
) -> Stage2HierarchyPromptProtocol:
    values: dict[str, object] = {
        "proposal_max_tokens": 26_000,
        "extraction_max_tokens": 23_000,
        "model_context_window_tokens": 131_072,
        "max_rendered_discovery_prompt_bytes": 350_000,
        "selector_thinking_token_budget": 6_000,
        "final_upstream_max_orphan_features": 37,
        "review_neural_query_nuisance_folds": 4,
        "final_upstream_meta_inner_folds": 4,
        "final_upstream_head_regularization": 0.75,
        "query_moment_max_queries": 24,
        "query_moment_max_terms_per_query": 32,
        "query_moment_max_chunks_per_query": 16,
        "query_moment_fallback_chunks_per_query": 8,
        "query_moment_max_excerpt_chars": 1200,
        "query_moment_max_term_chars": 160,
        "query_moment_max_ngram_tokens": 6,
        "extraction_grouping_strategy": "packed",
        "extraction_context_strategy": "complete_paged_v1",
        "extraction_prompt_version": "explicit_features_v5",
        "post_extraction_review_max_operations": 4,
        "post_extraction_review_max_quality_retries": 8,
        "post_extraction_review_min_partition_rows": 8,
        "hierarchical_max_atoms_per_chunk": 2,
        "hierarchical_max_bytes_per_chunk": 48_000,
        "hierarchical_max_semantic_member_ids_per_chunk": 3,
        "hierarchical_max_cross_architecture_lookback_ids": 24,
        "hierarchical_max_cross_architecture_lookback_bytes": 96_000,
        "hierarchical_max_extraction_lookback_ids_per_feature": 8,
        "hierarchical_max_extraction_lookback_bytes_per_feature": 96_000,
        "hierarchical_max_rejection_lookback_ids_per_candidate": 24,
        "hierarchical_max_rejection_lookback_bytes_per_candidate": 48_000,
        "hierarchical_review_max_evidence_ids": 512,
        "hierarchical_review_max_evidence_bytes": 2_000_000,
        "hierarchy_wire_budget": LEGACY_HIERARCHY_WIRE_BUDGET,
        "generation_policy": _generation_policy(),
    }
    values.update(overrides)
    if "generation_policy" not in overrides and {
        "proposal_max_tokens",
        "extraction_max_tokens",
        "selector_thinking_token_budget",
    } & set(overrides):
        values["generation_policy"] = _generation_policy(
            proposal_max_tokens=int(values["proposal_max_tokens"]),
            extraction_max_tokens=int(values["extraction_max_tokens"]),
            selector_thinking_token_budget=int(values["selector_thinking_token_budget"]),
        )
    return Stage2HierarchyPromptProtocol(**values)  # type: ignore[arg-type]


def _post_extraction_scientific_policy() -> PostExtractionScientificPolicy:
    return PostExtractionScientificPolicy.from_mapping(
        {
            "schema_version": POST_EXTRACTION_SCIENTIFIC_POLICY_VERSION,
            "extraction_quality": {
                "minimum_coverage": 0.05,
                "maximum_unknown_category_rate": 0.05,
                "continuous_outlier_minimum_rows": 8,
                "continuous_outlier_iqr_multiplier": 6.0,
                "continuous_outlier_warning_rate": 0.10,
                "fold_coverage_range_warning": 0.35,
                "fold_continuous_scale_epsilon": 1e-8,
            },
            "extraction_redundancy": {
                "association_threshold": 0.80,
                "missingness_jaccard_threshold": 0.90,
                "minimum_pairwise_complete_rows": 3,
            },
            "extraction_grounding": {
                "anchor_group_selection": "all_source_attested_unbounded",
                "maximum_group_span_chars": 96,
                "anchor_value_window_chars": 96,
                "category_assertion_prefix_chars": 64,
                "unit_window_min_chars": 12,
                "unit_window_max_chars": 32,
                "unit_window_divisor": 3,
                "minimum_evaluable_rows": 3,
                "maximum_alternative_category_only_rate": 0.50,
                "unsupported_value_warning_rate": 0.25,
                "minimum_unit_support_rate": 0.50,
            },
            "review_estimator": {
                "standardization_scale_epsilon": 1e-8,
                "logistic_alpha_floor": 1e-12,
                "logistic_solver": "liblinear",
                "logistic_max_iter": 1000,
                "logistic_random_seed": 0,
                "logistic_fit_intercept": True,
                "logistic_class_weight": None,
                "binary_no_features_fallback": "prevalence",
                "binary_single_class_fallback": "prevalence",
                "binary_fit_failure_policy": "prevalence",
                "continuous_minimum_fit_rows": 2,
                "continuous_degenerate_fallback": "mean",
                "effect_minimum_usable_rows": 2,
                "effect_no_usable_fallback": "zero",
                "effect_degenerate_fallback": "weighted_mean",
                "ridge_solver": "auto",
                "ridge_fit_intercept": True,
                "ridge_tolerance": 1e-4,
                "ridge_max_iter": None,
                "ridge_positive": False,
                "ridge_random_seed": None,
            },
        }
    )


def _causal_review_config() -> CausalReviewConfig:
    scientific_policy = _post_extraction_scientific_policy()
    return CausalReviewConfig(
        e_clip=0.04,
        nuisance_ridge_alpha=1.25,
        effect_ridge_alpha=0.8,
        contract_complexity_penalty=0.003,
        encoded_column_complexity_penalty=0.0003,
        minimum_score_improvement=0.001,
        nuisance_relative_tolerance=0.04,
        source_preservation_tolerance=0.03,
        source_context_r_loss_relative_tolerance=0.02,
        feature_bank_preservation_tolerance=0.01,
        estimator_policy=scientific_policy.review_estimator,
    )


class _PromptGuard(Stage2PromptNonTruncationGuard):
    def __init__(self, *, model_name: str = TEST_MODEL) -> None:
        self.model_name = model_name
        self._accepted_client_paths: list[str] = []

    def identity(self) -> dict[str, object]:
        return {"test_prompt_guard": True, "model_name": self.model_name}

    def validate_request(
        self,
        request: object,
        *,
        client_path: str = "unspecified_nonproduction",
    ) -> dict[str, object]:
        return {"request": request, "client_path": client_path}

    def validate_response(
        self,
        response: object,
        *,
        request_audit: object,
    ) -> dict[str, object]:
        if not isinstance(request_audit, dict):
            raise TypeError("test request audit must be one dictionary")
        self._accepted_client_paths.append(str(request_audit["client_path"]))
        return {"request_audit": request_audit, "response_checked": response is not None}

    def execution_audit(self) -> dict[str, object]:
        client_paths = (
            "explicit_feature_extraction",
            "hierarchical_discovery",
            "proposal_and_post_extraction_review",
        )
        records = [{"client_path": client_path} for client_path in self._accepted_client_paths]
        counts = {
            client_path: self._accepted_client_paths.count(client_path)
            for client_path in client_paths
        }
        body: dict[str, object] = {
            "schema_version": "stage2_prompt_nontruncation_execution_audit_v1",
            "guard_identity_sha256": _content_sha256(self.identity()),
            "record_count": len(records),
            "records": records,
            "records_sha256": _content_sha256(records),
            "record_counts_by_client_path": counts,
            "unclassified_record_count": sum(
                1 for client_path in self._accepted_client_paths if client_path not in client_paths
            ),
            "all_records_status": "accepted_nontruncated",
            "all_endpoint_prompt_tokens_exact_match": True,
            "all_request_audits_authenticated": True,
            "all_guard_identities_exact_match": True,
            "all_requests_forbid_truncation_controls": True,
        }
        return {**body, "audit_sha256": _content_sha256(body)}


def _options(tmp_path: Path) -> ProductionStage1HierarchyOneShotOptions:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = bundle / "bundle_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    return ProductionStage1HierarchyOneShotOptions(
        bundle_manifest_path=manifest,
        output_dir=tmp_path / "execution",
        preparation_dir=tmp_path / "preparation",
        attestation_dir=tmp_path / "attestation",
        endpoint=TEST_ENDPOINT,
        model_name=TEST_MODEL,
        review_rounds=1,
        initial_training_partitions=3,
        stage2_protocol=_stage2_protocol(),
        stage2_tokenizer_locator=tokenizer,
        hierarchical_discovery_job_cache_config=HIERARCHY_JOB_CACHE_CONFIG,
        first_untouched_gate_preparation_bounds=FIRST_UNTOUCHED_GATE_BOUNDS,
        post_extraction_review_config=_causal_review_config(),
        post_extraction_scientific_policy=(_post_extraction_scientific_policy()),
        review_stage1_device="cpu",
        review_neural_query_devices=("cpu",),
        source_text_temporally_valid_by_design=True,
        max_candidates=8,
        forest_n_estimators=40,
        forest_max_depth=7,
        forest_min_samples_leaf=4,
        forest_max_features="sqrt",
        forest_honest=True,
        forest_inference=True,
        forest_subforest_size=4,
        forest_tune_model=False,
        forest_nuisance_n_estimators=31,
        forest_nuisance_max_depth=5,
        forest_nuisance_min_samples_leaf=3,
        forest_nuisance_treatment_max_features=0.75,
        forest_nuisance_outcome_max_features="sqrt",
        forest_random_seed=23,
        forest_n_jobs=2,
        proposal_schema_repair_attempts=1,
        request_max_retries=0,
        extraction_max_text_length=119,
        complete_page_core_chars=97,
        complete_page_context_chars=11,
        complete_page_max_chars=119,
        complete_reconciliation_fan_in=7,
    )


def test_final_forest_constructor_receives_every_configured_field_without_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class StrictConstructorSpy:
        def __init__(
            self,
            *,
            n_estimators,
            max_depth,
            min_samples_leaf,
            max_features,
            honest,
            inference,
            subforest_size,
            tune_model,
            nuisance_n_estimators,
            nuisance_max_depth,
            nuisance_min_samples_leaf,
            nuisance_treatment_max_features,
            nuisance_outcome_max_features,
            random_state,
            n_jobs,
        ):
            captured.update(locals())
            captured.pop("self")
            captured.pop("captured")

    monkeypatch.setattr(
        subject,
        "FixedCausalForestHeadBackend",
        StrictConstructorSpy,
    )
    options = _options(tmp_path)
    backend = subject._configured_strict_causal_forest_backend(options)

    assert type(backend) is StrictConstructorSpy
    assert captured == {
        "n_estimators": options.forest_n_estimators,
        "max_depth": options.forest_max_depth,
        "min_samples_leaf": options.forest_min_samples_leaf,
        "max_features": options.forest_max_features,
        "honest": options.forest_honest,
        "inference": options.forest_inference,
        "subforest_size": options.forest_subforest_size,
        "tune_model": options.forest_tune_model,
        "nuisance_n_estimators": options.forest_nuisance_n_estimators,
        "nuisance_max_depth": options.forest_nuisance_max_depth,
        "nuisance_min_samples_leaf": (options.forest_nuisance_min_samples_leaf),
        "nuisance_treatment_max_features": (options.forest_nuisance_treatment_max_features),
        "nuisance_outcome_max_features": (options.forest_nuisance_outcome_max_features),
        "random_state": options.forest_random_seed,
        "n_jobs": options.forest_n_jobs,
    }


def test_portable_one_shot_uses_only_the_typed_v4_forest_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class StrictRuntimeSpy:
        def __init__(self, *, runtime_config):
            captured["runtime_config"] = runtime_config

        def identity(self):
            return {
                "backend": "repository_strict_causal_forest_path_v4",
                "configuration_mode": "portable_strict_runtime_config_v1",
            }

    runtime = StrictCausalForestRuntimeConfig(
        schema_version=STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
        causal_forest=_forest_spec(),
        operational=_forest_operational(2),
    )
    options = replace(
        _options(tmp_path),
        forest_runtime_config=runtime,
        forest_n_estimators=None,
        forest_max_depth=None,
        forest_min_samples_leaf=None,
        forest_max_features=None,
        forest_honest=None,
        forest_inference=None,
        forest_subforest_size=None,
        forest_tune_model=None,
        forest_nuisance_n_estimators=None,
        forest_nuisance_max_depth=None,
        forest_nuisance_min_samples_leaf=None,
        forest_nuisance_treatment_max_features=None,
        forest_nuisance_outcome_max_features=None,
        forest_random_seed=None,
        forest_n_jobs=None,
    )
    monkeypatch.setattr(
        subject,
        "FixedCausalForestHeadBackend",
        StrictRuntimeSpy,
    )

    subject._validate_options(options)
    backend = subject._configured_strict_causal_forest_backend(options)

    assert type(backend) is StrictRuntimeSpy
    assert captured == {"runtime_config": runtime}
    with pytest.raises(ValueError, match="legacy flat forest"):
        subject._validate_options(replace(options, forest_n_estimators=80))


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


def test_stage2_prompt_protocol_has_no_defaults_and_cli_requires_every_field() -> None:
    with pytest.raises(TypeError):
        Stage2HierarchyPromptProtocol()  # type: ignore[call-arg]

    parser = build_parser()
    actions = {action.dest: action for action in parser._actions}
    for field_name in Stage2HierarchyPromptProtocol.__dataclass_fields__:
        assert actions[field_name].required is True
        assert actions[field_name].default is None
    assert actions["review_stage1_device"].required is True
    assert actions["review_stage1_device"].default is None
    assert actions["review_neural_query_device"].required is True
    assert actions["review_neural_query_device"].default is None
    assert actions["hierarchical_job_cache_max_entry_bytes"].required is True
    assert actions["hierarchical_job_cache_max_entry_bytes"].default is None
    for field_name in FIRST_UNTOUCHED_GATE_BOUNDS.__dataclass_fields__:
        action = actions["first_untouched_gate_" + field_name]
        assert action.required is True
        assert action.default is None

    first = _stage2_protocol(
        proposal_max_tokens=26_001,
        extraction_max_tokens=17_003,
    )
    second = _stage2_protocol(
        proposal_max_tokens=26_002,
        extraction_max_tokens=17_003,
    )
    assert first.content_sha256 != second.content_sha256
    assert first.as_dict()["proposal_max_tokens"] == 26_001


def test_stage2_protocol_identity_binds_each_generation_family_policy() -> None:
    base_policy = _generation_policy()
    changed_policy = replace(
        base_policy,
        feature_proposal_review=replace(
            base_policy.feature_proposal_review,
            temperature=0.25,
        ),
    )
    first = _stage2_protocol(generation_policy=base_policy)
    second = _stage2_protocol(generation_policy=changed_policy)

    assert first.content_sha256 != second.content_sha256
    assert first.as_dict()["generation_policy"] == base_policy.as_dict()
    assert second.as_dict()["generation_policy"] == changed_policy.as_dict()


def test_portable_and_one_shot_stage2_protocol_fields_are_exactly_equal() -> None:
    assert set(Stage2PromptProtocolSpec.__dataclass_fields__) == set(
        Stage2HierarchyPromptProtocol.__dataclass_fields__
    )


def test_portable_and_runtime_causal_review_fields_are_exactly_bound_without_option_default(
    tmp_path: Path,
) -> None:
    portable_fields = set(PostExtractionCausalReviewSpec.__dataclass_fields__)
    assert portable_fields - {
        "upstream_review_policy",
        "scientific_policy",
    } == set(
        CausalReviewConfig.__dataclass_fields__
    ) - {"estimator_policy"}
    assert (
        _causal_review_config().estimator_policy
        == _post_extraction_scientific_policy().review_estimator
    )
    assert "upstream_review_policy" in portable_fields
    option_field = ProductionStage1HierarchyOneShotOptions.__dataclass_fields__[
        "post_extraction_review_config"
    ]
    assert option_field.default is MISSING
    assert option_field.default_factory is MISSING
    policy_option_field = ProductionStage1HierarchyOneShotOptions.__dataclass_fields__[
        "upstream_review_policy"
    ]
    assert policy_option_field.default is None
    assert _options(tmp_path).post_extraction_review_config == _causal_review_config()


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
    proposal_generation = _generation_policy().feature_proposal_review
    agent = ProductionSingleEndpointFeatureSearchAgent(
        subject.AgenticFeatureSearchConfig(
            agent_server_url=TEST_ENDPOINT,
            agent_model_name=literal,
            agent_temperature=proposal_generation.temperature,
            agent_max_tokens=proposal_generation.max_tokens,
            agent_enable_thinking=proposal_generation.thinking_enabled,
            agent_thinking_token_budget=(proposal_generation.thinking_token_budget),
            agent_request_max_retries=(proposal_generation.transport_max_retries),
            agent_schema_repair_attempts=(proposal_generation.schema_repair_attempts),
        ),
        prompt_nontruncation_guard=_PromptGuard(model_name=literal),
        generation_parameters=proposal_generation,
    )
    assert agent._resolve_agent_model_inventory() == {TEST_ENDPOINT: literal}
    assert agent._resolve_agent_model_name() == literal

    patient_generation = _generation_policy().patient_feature_extraction
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "cohort.parquet"),
        explicit_features=ExplicitFeatureExtractionConfig(
            enabled=True,
            vllm_mode="server",
            vllm_server_url=TEST_ENDPOINT,
            vllm_model_name=literal,
            vllm_enable_thinking=patient_generation.thinking_enabled,
            extraction_temperature=patient_generation.temperature,
            extraction_max_tokens=patient_generation.max_tokens,
            extraction_max_retries=patient_generation.transport_max_retries,
            cache_dir=str(tmp_path / "cache"),
        ),
    )
    provider = subject.ProductionSingleEndpointExplicitFeatureExtractionProvider(
        config,
        tmp_path / "output",
        prompt_nontruncation_guard=_PromptGuard(model_name=literal),
        generation_parameters=patient_generation,
    )
    assert provider._resolve_vllm_model_inventory() == {TEST_ENDPOINT: literal}
    assert provider._resolve_vllm_model_name() == literal


def test_complete_paged_provider_keeps_two_batch_ledgers_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oci.extraction.complete_paged import (
        COMPLETE_PAGED_RESPONSE_SCHEMA,
        COMPLETE_PAGED_TRANSPORT_SCHEMA,
        CompletePageResponse,
    )

    class NegativeCompletePageExtractor:
        def __init__(self, **kwargs: object) -> None:
            self.model_name = str(kwargs["model_name"])

        def extract_complete_page(
            self,
            *,
            text: str,
            page: object,
            feature: object,
            geometry: object,
        ):
            del feature, geometry
            response = CompletePageResponse.validate(
                {
                    "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                    "status": "negative",
                    "normalized_value": "not_documented",
                    "reason": None,
                    "citations": [],
                },
                text=text,
                page=page,
            )
            attempt = {
                "kind": "initial",
                "request_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "response_sha256": _content_sha256(response.as_dict()),
                "model": self.model_name,
                "finish_reason": "stop",
            }
            body = {
                "schema_version": COMPLETE_PAGED_TRANSPORT_SCHEMA,
                "transport_retry_count": 0,
                "schema_repair_count": 0,
                "configured_model": self.model_name,
                "attempts": [attempt],
            }
            return response, {
                **body,
                "content_sha256": _content_sha256(body),
            }

        def reconcile_complete_pages(self, **_kwargs: object):
            raise AssertionError("the one-page fixture must not invoke reconciliation")

        def cleanup(self) -> None:
            return None

    monkeypatch.setattr(
        subject,
        "ProductionSingleEndpointVLLMFeatureExtractor",
        NegativeCompletePageExtractor,
    )
    spec = ExplicitFeatureSpec(
        name="documented_biomarker",
        type="categorical",
        categories=["documented", "not_documented"],
        description="Whether the prepared note documents the biomarker",
        roles=["confounder"],
    )
    explicit = ExplicitFeatureExtractionConfig(
        enabled=True,
        features=[spec],
        vllm_mode="server",
        vllm_server_url=TEST_ENDPOINT,
        vllm_model_name=TEST_MODEL,
        vllm_enable_thinking=False,
        extraction_batch_size=1,
        max_variables_per_extraction_request=1,
        extraction_max_retries=0,
        extraction_temperature=0.0,
        extraction_max_tokens=23_000,
        extraction_max_text_length=80,
        complete_page_core_chars=64,
        complete_page_context_chars=8,
        complete_page_max_chars=80,
        complete_reconciliation_fan_in=4,
        extraction_grouping_strategy="packed",
        extraction_context_strategy="complete_paged_v1",
        source_text_temporally_valid_by_design=True,
        cache_enabled=False,
        cache_dir=str(tmp_path / "cache"),
    )
    config = AppliedInferenceConfig(
        dataset_path=str(tmp_path / "unused.parquet"),
        text_column="prepared_note",
        explicit_features=explicit,
    )
    provider = subject.ProductionSingleEndpointExplicitFeatureExtractionProvider(
        config,
        tmp_path / "output",
        prompt_nontruncation_guard=_PromptGuard(),
        generation_parameters=(_generation_policy().patient_feature_extraction),
    )

    first = pd.DataFrame(
        {
            "_oci_row_id": [0],
            "prepared_note": ["No biomarker is documented in this note."],
        }
    )
    second = pd.DataFrame(
        {
            "_oci_row_id": [1],
            "prepared_note": ["A different note also has no biomarker."],
        }
    )
    provider._extract_spec_group(first, [spec])
    first_paths = provider.complete_paged_ledger_artifact_paths()
    first_bytes = {path: path.read_bytes() for path in first_paths}
    provider._extract_spec_group(second, [spec])

    manifests = provider.complete_paged_ledger_manifest_paths()
    artifacts = provider.complete_paged_ledger_artifact_paths()
    assert len(manifests) == 2
    assert len(artifacts) == 6
    assert len(set(manifests)) == 2
    assert len(set(artifacts)) == 6
    assert manifests[0].parent != manifests[1].parent
    assert all(path.exists() for path in artifacts)
    assert {path: path.read_bytes() for path in first_paths} == first_bytes


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
        max_retries=0,
        generation_policy=_generation_policy(),
        prompt_nontruncation_guard=_PromptGuard(),
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


def test_production_hierarchy_runner_rejects_missing_legacy_or_retrying_policy() -> None:
    common = {
        "server_urls": TEST_ENDPOINT,
        "model_name": TEST_MODEL,
        "api_key": "EMPTY",
        "prompt_nontruncation_guard": _PromptGuard(),
    }
    with pytest.raises(TypeError, match="requires Stage2GenerationPolicy"):
        ProductionSingleEndpointJsonDiscoveryJobRunner(**common)
    with pytest.raises(ValueError, match="legacy aggregate"):
        ProductionSingleEndpointJsonDiscoveryJobRunner(
            **common,
            generation_policy=_generation_policy(),
            max_retries=0,
            max_tokens=26_000,
        )
    with pytest.raises(ValueError, match="zero transport retries"):
        ProductionSingleEndpointJsonDiscoveryJobRunner(
            **common,
            generation_policy=_generation_policy(),
            max_retries=1,
        )


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
        max_retries=0,
        generation_policy=_generation_policy(),
        prompt_nontruncation_guard=_PromptGuard(),
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
    generation = _generation_policy().feature_proposal_review
    config = subject.AgenticFeatureSearchConfig(
        agent_server_url=TEST_ENDPOINT,
        agent_model_name=TEST_MODEL,
        agent_temperature=generation.temperature,
        agent_max_tokens=generation.max_tokens,
        agent_enable_thinking=generation.thinking_enabled,
        agent_thinking_token_budget=generation.thinking_token_budget,
        agent_request_max_retries=generation.transport_max_retries,
        agent_schema_repair_attempts=generation.schema_repair_attempts,
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
    agent = ProductionSingleEndpointFeatureSearchAgent(
        config,
        prompt_nontruncation_guard=_PromptGuard(),
        generation_parameters=generation,
    )
    with pytest.raises(ValueError, match="model differs|finish_reason"):
        agent._create_completion(
            model=TEST_MODEL,
            messages=[],
            **generation.request_generation_fields(),
        )


def test_proposal_review_constructor_and_request_reject_policy_drift() -> None:
    generation = _generation_policy().feature_proposal_review
    config = subject.AgenticFeatureSearchConfig(
        agent_server_url=TEST_ENDPOINT,
        agent_model_name=TEST_MODEL,
        agent_temperature=generation.temperature,
        agent_max_tokens=generation.max_tokens,
        agent_enable_thinking=generation.thinking_enabled,
        agent_thinking_token_budget=generation.thinking_token_budget,
        agent_request_max_retries=generation.transport_max_retries,
        agent_schema_repair_attempts=generation.schema_repair_attempts,
    )
    with pytest.raises(ValueError, match="configuration differs"):
        ProductionSingleEndpointFeatureSearchAgent(
            replace(config, agent_temperature=0.75),
            prompt_nontruncation_guard=_PromptGuard(),
            generation_parameters=generation,
        )
    agent = ProductionSingleEndpointFeatureSearchAgent(
        config,
        prompt_nontruncation_guard=_PromptGuard(),
        generation_parameters=generation,
    )
    request = {
        "model": TEST_MODEL,
        "messages": [],
        **generation.request_generation_fields(),
    }
    request["temperature"] = 0.75
    with pytest.raises(ValueError, match="generation controls differ"):
        agent._create_completion(**request)


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
    generation = _generation_policy().patient_feature_extraction
    extractor = subject.ProductionSingleEndpointVLLMFeatureExtractor(
        specs=[],
        mode="server",
        server_url=TEST_ENDPOINT,
        model_name=TEST_MODEL,
        model_names_by_url={TEST_ENDPOINT: TEST_MODEL},
        vllm_enable_thinking=generation.thinking_enabled,
        temperature=generation.temperature,
        max_tokens=generation.max_tokens,
        max_retries=generation.transport_max_retries,
        schema_repair_attempts=generation.schema_repair_attempts,
        prompt_nontruncation_guard=_PromptGuard(),
        generation_parameters=generation,
    )
    extractor._init_server_client()
    with pytest.raises(ValueError, match="model differs|finish_reason"):
        extractor._extract_single_server("patient text must remain unread")


def test_complete_page_reconciliation_prompt_strips_authenticated_citation_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oci.extraction.complete_paged import (
        COMPLETE_PAGED_RESPONSE_SCHEMA,
        CompleteFeatureContract,
        CompletePageResponse,
    )

    text = "alpha beta"
    leaf = CompletePageResponse.validate(
        {
            "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
            "status": "positive",
            "normalized_value": "present",
            "reason": None,
            "citations": [
                {
                    "start": 0,
                    "end": 5,
                    "text": "alpha",
                }
            ],
        },
        text=text,
        page=None,
    ).as_dict()
    assert set(leaf["citations"][0]) == {"start", "end", "text", "sha256"}

    generation = _generation_policy().patient_feature_extraction
    extractor = subject.ProductionSingleEndpointVLLMFeatureExtractor(
        specs=[],
        mode="server",
        server_url=TEST_ENDPOINT,
        model_name=TEST_MODEL,
        model_names_by_url={TEST_ENDPOINT: TEST_MODEL},
        vllm_enable_thinking=generation.thinking_enabled,
        temperature=generation.temperature,
        max_tokens=generation.max_tokens,
        max_retries=generation.transport_max_retries,
        schema_repair_attempts=generation.schema_repair_attempts,
        prompt_nontruncation_guard=_PromptGuard(),
        generation_parameters=generation,
    )
    calls: list[dict[str, object]] = []

    def copy_prompt_citation(request: object) -> object:
        assert isinstance(request, dict)
        calls.append(request)
        messages = request["messages"]
        assert isinstance(messages, list)
        prompt = messages[0]["content"]
        assert isinstance(prompt, str)
        prompt_children = json.loads(prompt.split("\nchildren=", 1)[1])
        copied_citation = prompt_children[0]["response"]["citations"][0]
        assert set(copied_citation) == {"start", "end", "text"}
        content = json.dumps(
            {
                "child_ids": ["leaf-a", "leaf-b"],
                "schema_version": COMPLETE_PAGED_RESPONSE_SCHEMA,
                "status": "positive",
                "normalized_value": "present",
                "reason": None,
                "citations": [copied_citation],
            }
        )
        return _completion_response(content=content)

    monkeypatch.setattr(extractor, "_complete_page_call", copy_prompt_citation)
    feature = CompleteFeatureContract(
        name="documented_marker",
        value_type="categorical",
        description="Whether the marker is documented",
        categories=("present", "absent"),
        temporal_rule="use eligible evidence only",
        aggregation_rule="ever documented",
    )
    result, audit = extractor.reconcile_complete_pages(
        text=text,
        feature=feature,
        children=(
            {"node_id": "leaf-a", "response": leaf},
            {"node_id": "leaf-b", "response": leaf},
        ),
    )

    assert len(calls) == 1
    assert audit["schema_repair_count"] == 0
    assert result["citations"] == [
        {
            "start": 0,
            "end": 5,
            "text": "alpha",
            "sha256": hashlib.sha256(b"alpha").hexdigest(),
        }
    ]
    assert set(leaf["citations"][0]) == {"start", "end", "text", "sha256"}


def test_roots_are_absolute_fresh_nonnested_and_outside_bundle(tmp_path: Path) -> None:
    options = _options(tmp_path)
    _validate_fresh_roots(options)
    with pytest.raises(ValueError, match="fresh nonexistent"):
        options.attestation_dir.mkdir()
        _validate_fresh_roots(options)
    _validate_fresh_roots(replace(options, resume=True))
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
    options = replace(
        _options(tmp_path),
        endpoint_api_key="remote-secret",
    )
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
    prompt_guard = _PromptGuard()
    monkeypatch.setattr(
        subject,
        "Stage2PromptNonTruncationGuard",
        lambda **_kwargs: prompt_guard,
    )

    def capture_review_agent(
        config: object,
        *,
        prompt_nontruncation_guard: object,
        generation_parameters: object,
    ) -> object:
        assert prompt_nontruncation_guard is prompt_guard
        assert generation_parameters == _generation_policy().feature_proposal_review
        observed["review_endpoint"] = config.agent_server_url
        observed["review_model"] = config.agent_model_name
        observed["review_api_key"] = config.agent_api_key
        return object()

    def capture_extraction(
        config: object,
        _root: Path,
        *,
        prompt_nontruncation_guard: object,
        generation_parameters: object,
    ) -> object:
        assert prompt_nontruncation_guard is prompt_guard
        assert generation_parameters == _generation_policy().patient_feature_extraction
        observed["extraction_endpoint"] = config.explicit_features.vllm_server_url
        observed["extraction_model"] = config.explicit_features.vllm_model_name
        observed["extraction_api_key"] = config.explicit_features.vllm_api_key
        return object()

    def capture_hierarchy(**kwargs: object) -> object:
        assert kwargs["prompt_nontruncation_guard"] is prompt_guard
        assert kwargs["generation_policy"] == _generation_policy()
        observed["hierarchy_endpoint"] = kwargs["server_urls"]
        observed["hierarchy_model"] = kwargs["model_name"]
        observed["hierarchy_api_key"] = kwargs["api_key"]
        body = {
            "endpoint_urls": [kwargs["server_urls"]],
            "model": {"name": kwargs["model_name"]},
            "prompt_nontruncation_guard": prompt_guard.identity(),
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
        "review_api_key": "remote-secret",
        "extraction_endpoint": TEST_ENDPOINT,
        "extraction_model": TEST_MODEL,
        "extraction_api_key": "remote-secret",
        "hierarchy_endpoint": TEST_ENDPOINT,
        "hierarchy_model": TEST_MODEL,
        "hierarchy_api_key": "remote-secret",
    }
    assert runner.review_spent_evidence_provider is provider
    assert runner.review_partition_provider is provider
    assert runner.review_gate_source_provider is gate_provider
    assert runner.review_gate_feature_bank_provider is gate_provider
    assert (
        runner.hierarchical_discovery_job_cache_config
        is options.hierarchical_discovery_job_cache_config
    )
    assert (
        runner.first_untouched_gate_preparation_bounds
        is options.first_untouched_gate_preparation_bounds
    )


def test_reference_only_builder_propagates_explicit_cache_and_gate_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_path = tmp_path / "portable_prepared.parquet"
    options = replace(
        _options(tmp_path),
        prepared_cohort_path=prepared_path,
        unit_id_column="portable_patient_id",
        text_column="portable_note",
        treatment_column="portable_treatment",
        outcome_column="portable_outcome",
        outcome_type="binary",
        upstream_review_policy=(
            GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY
        ),
    )
    provider = object()
    numerical_bank = object()
    direct_inputs = subject.ReferenceOnlyRoleNeutralStage2Inputs(
        prepared=pd.DataFrame(),
        prepared_cohort_artifact_sha256="1" * 64,
        outer_fold_assignments={
            1: {"fit": (1,), "held_out": (2,)},
            2: {"fit": (2,), "held_out": (1,)},
        },
        prepared_projection_binding=object(),
        runtime_binding=object(),
        numerical_bank=numerical_bank,
    )
    handoff = SimpleNamespace(stage2_provider=provider)
    prompt_guard = _PromptGuard()

    monkeypatch.setattr(
        subject,
        "Stage2PromptNonTruncationGuard",
        lambda **_kwargs: prompt_guard,
    )
    monkeypatch.setattr(
        subject,
        "ProductionSingleEndpointFeatureSearchAgent",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        subject,
        "ProductionSingleEndpointExplicitFeatureExtractionProvider",
        lambda *_args, **_kwargs: object(),
    )

    def hierarchy_runner(**kwargs: object) -> object:
        return SimpleNamespace(
            identity=lambda: {
                "endpoint_urls": [kwargs["server_urls"]],
                "model": {"name": kwargs["model_name"]},
                "prompt_nontruncation_guard": prompt_guard.identity(),
            }
        )

    monkeypatch.setattr(
        subject,
        "ProductionSingleEndpointJsonDiscoveryJobRunner",
        hierarchy_runner,
    )
    monkeypatch.setattr(
        subject,
        "_configured_strict_causal_forest_backend",
        lambda _options: object(),
    )

    class CapturedRunner:
        def __init__(self, **kwargs: object) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setattr(subject, "AllEvidenceFusionRunner", CapturedRunner)
    runner = subject._construct_reference_only_role_neutral_stage2_runner(
        runner_type=CapturedRunner,
        handoff=handoff,
        direct_inputs=direct_inputs,
        options=options,
        endpoint=TEST_ENDPOINT,
    )

    assert (
        runner.hierarchical_discovery_job_cache_config
        is options.hierarchical_discovery_job_cache_config
    )
    assert (
        runner.first_untouched_gate_preparation_bounds
        is options.first_untouched_gate_preparation_bounds
    )


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


def test_reference_only_dispatch_never_enters_legacy_loaders_or_refit_constructors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oci.inference.all_evidence_post_extraction_review import (
        GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
    )
    from oci.inference.production_role_neutral_stage2_handoff import (
        ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
    )

    base = _options(tmp_path)
    base.bundle_manifest_path.write_text(
        json.dumps({"handoff_kind": ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND}) + "\n",
        encoding="utf-8",
    )
    prepared = tmp_path / "arbitrary_prepared_location.parquet"
    prepared.write_bytes(b"dispatch-spy-does-not-decode")
    bank = tmp_path / "arbitrary_bank" / "direct_upstream_numerical_manifest.json"
    bank.parent.mkdir()
    bank.write_text("{}\n", encoding="utf-8")
    options = replace(
        base,
        prepared_cohort_path=prepared,
        unit_id_column="hospital_specific_key",
        text_column="full_longitudinal_narrative",
        treatment_column="received_index_regimen",
        outcome_column="binary_clinical_endpoint",
        outcome_type="binary",
        direct_numerical_bank_manifest_path=bank,
        upstream_review_policy=(GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY),
    )
    provider = object()
    handoff_payload = {
        "offline_handoff_validation_complete": True,
        "independent_runtime_stage1_refit_allowed": False,
        "legacy_bundle_build_invoked": False,
        "content_sha256": "a" * 64,
    }
    handoff = SimpleNamespace(
        handoff_kind=ROLE_NEUTRAL_STAGE1_REFERENCE_HANDOFF_KIND,
        stage2_provider=provider,
        as_dict=lambda: dict(handoff_payload),
    )
    runner = object.__new__(AllEvidenceFusionRunner)
    runner.review_spent_evidence_provider = provider
    runner.review_partition_provider = provider
    runner.hierarchical_discovery_approved_batch_sha256 = None
    fake_result = AllEvidenceFusionRunResult(
        prediction_path=options.output_dir / "frozen_predictions.parquet",
        run_manifest_path=options.output_dir / "immutable_run_manifest.json",
        fold_manifest_paths=(),
        prediction_sha256="b" * 64,
    )
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("portable direct dispatch entered a legacy/refit path")

    monkeypatch.setattr(subject, "load_production_stage1_hierarchy_handoff", forbidden)
    for name in (
        "HistoricalStage1ContextBackend",
        "ContextFitNeuralQueryService",
        "FinalContextFitUpstreamProducer",
        "build_shared_tfidf_context_fit_backends",
    ):
        monkeypatch.setattr(subject, name, forbidden)
    monkeypatch.setattr(
        "oci.inference.production_role_neutral_stage2_handoff."
        "load_reference_only_role_neutral_stage1_handoff",
        lambda *_args, **_kwargs: handoff,
    )
    monkeypatch.setattr(
        subject,
        "build_reference_only_role_neutral_stage2_runner",
        lambda **_kwargs: (calls.append("direct_builder") or runner),
    )
    monkeypatch.setattr(
        subject,
        "run_internal_reference_only_role_neutral_stage2_one_shot",
        lambda **_kwargs: (calls.append("direct_execution") or fake_result),
    )
    monkeypatch.setattr(
        subject,
        "_seal_reference_only_result_attestation",
        lambda **_kwargs: (
            calls.append("direct_attestation") or {"status": "completed", "mode": "reference_only"}
        ),
    )

    assert run_production_stage1_hierarchy_one_shot(options) == {
        "status": "completed",
        "mode": "reference_only",
    }
    assert calls == [
        "direct_builder",
        "direct_execution",
        "direct_attestation",
    ]


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
    prompt_guard = _PromptGuard()
    prompt_guard.validate_response(
        object(),
        request_audit=prompt_guard.validate_request(
            {},
            client_path="hierarchical_discovery",
        ),
    )
    hierarchy_runner = SimpleNamespace(
        identity=lambda: {
            "identity_sha256": "1" * 64,
            "endpoint_urls": [options.endpoint],
            "model": {"name": options.model_name},
        },
        _prompt_nontruncation_guard=prompt_guard,
    )
    runner = SimpleNamespace(
        hierarchical_discovery_runner=hierarchy_runner,
        review_spent_evidence_provider=provider,
        review_partition_provider=provider,
        _production_stage2_prompt_nontruncation_guard=prompt_guard,
        config=SimpleNamespace(post_extraction_review_config=options.post_extraction_review_config),
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
    with pytest.raises(
        RuntimeError,
        match="prompt nontruncation execution audit is incomplete",
    ):
        _seal_result_attestation(
            handoff=handoff,
            runner=runner,
            result=result,
            options=options,
            endpoint=options.endpoint,
            implementation_sha256=_stable_sha256(module_path, label="test module")[1],
        )
    for client_path in (
        "proposal_and_post_extraction_review",
        "explicit_feature_extraction",
    ):
        prompt_guard.validate_response(
            object(),
            request_audit=prompt_guard.validate_request(
                {},
                client_path=client_path,
            ),
        )
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
    assert payload["post_extraction_causal_review"] == asdict(options.post_extraction_review_config)
    assert payload["post_extraction_causal_review_sha256"] == _content_sha256(
        asdict(options.post_extraction_review_config)
    )
    assert payload["remote_runtime_identity"]["endpoint_urls"] == [TEST_ENDPOINT]
    assert payload["remote_runtime_identity"]["model"]["name"] == TEST_MODEL
    assert payload["remote_runtime_identity"]["served_deployment_metadata_required"] is False
    assert payload["run_result_audit_record_is_authorization"] is False
    prompt_execution_audit = payload["prompt_nontruncation_execution_audit"]
    assert prompt_execution_audit["record_count"] == 3
    assert prompt_execution_audit["unclassified_record_count"] == 0
    assert set(prompt_execution_audit["record_counts_by_client_path"].values()) == {1}
    prompt_audit_sha = prompt_execution_audit.pop("audit_sha256")
    assert prompt_audit_sha == _content_sha256(prompt_execution_audit)
    with pytest.raises(FileExistsError):
        _seal_result_attestation(
            handoff=handoff,
            runner=runner,
            result=result,
            options=options,
            endpoint=options.endpoint,
            implementation_sha256=_stable_sha256(module_path, label="test module")[1],
        )
