from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import scripts.canary_production_stage1_hierarchy as canary
from oci.inference.all_evidence_discovery_interfaces import (
    BOW_NUISANCE,
    BOW_R_LOSS,
    DiscoveryEvidenceItem,
    canonical_json,
    content_sha256,
    render_interpret_evidence_chunk_messages,
)
from oci.inference.all_evidence_post_extraction_review import CausalReviewConfig
from oci.inference.approved_hierarchical_discovery_agent import (
    _PerCallMetadataAuthenticatingRunner,
)
from oci.inference.hierarchical_all_architecture_discovery import (
    HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING,
    INTERPRET_CHUNK_JOB,
    DiscoveryJobSettings,
    DiscoveryJsonJob,
    HierarchicalAllArchitectureDiscoveryOrchestrator,
    hierarchical_discovery_implementation_bundle,
)
from oci.inference.production_stage1_hierarchy_one_shot import (
    ProductionStage1HierarchyOneShotOptions,
    Stage2HierarchyPromptProtocol,
)
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
from tests.hierarchy_resource_test_support import (
    FIRST_UNTOUCHED_GATE_BOUNDS,
    HIERARCHY_JOB_CACHE_CONFIG,
)

CAMUS_ENDPOINT = "http://camus:8010/v1"
CAMUS_MODEL = "RedhatAI/gemma-4-26B-A4B-it-FP8-Dynamic"
LOCAL_ENDPOINT = "http://localhost:2345/v1"
LOCAL_MODEL = "local/test-model"


def _generation_parameters(
    *,
    max_tokens: int,
    thinking_enabled: bool,
    thinking_token_budget: int,
) -> Stage2GenerationParameters:
    return Stage2GenerationParameters(
        temperature=0.0,
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


def _generation_policy() -> Stage2GenerationPolicy:
    selector = _generation_parameters(
        max_tokens=26_000,
        thinking_enabled=True,
        thinking_token_budget=6_000,
    )
    definition = _generation_parameters(
        max_tokens=26_000,
        thinking_enabled=False,
        thinking_token_budget=0,
    )
    patient = _generation_parameters(
        max_tokens=18_000,
        thinking_enabled=False,
        thinking_token_budget=0,
    )
    return Stage2GenerationPolicy(
        **{
            job_kind: (
                definition
                if job_kind == "define_one_extraction_feature"
                else selector
            )
            for job_kind in HIERARCHICAL_GENERATION_JOB_KINDS
        },
        feature_proposal_review=selector,
        patient_feature_extraction=patient,
    )


def _stage2_protocol() -> Stage2HierarchyPromptProtocol:
    return Stage2HierarchyPromptProtocol(
        proposal_max_tokens=26_000,
        extraction_max_tokens=18_000,
        model_context_window_tokens=131_072,
        hierarchy_wire_budget=LEGACY_HIERARCHY_WIRE_BUDGET,
        generation_policy=_generation_policy(),
        max_rendered_discovery_prompt_bytes=350_000,
        selector_thinking_token_budget=6_000,
        final_upstream_max_orphan_features=37,
        review_neural_query_nuisance_folds=4,
        final_upstream_meta_inner_folds=4,
        final_upstream_head_regularization=0.75,
        query_moment_max_queries=24,
        query_moment_max_terms_per_query=32,
        query_moment_max_chunks_per_query=16,
        query_moment_fallback_chunks_per_query=8,
        query_moment_max_excerpt_chars=1200,
        query_moment_max_term_chars=160,
        query_moment_max_ngram_tokens=6,
        extraction_grouping_strategy="packed",
        extraction_context_strategy="complete_paged_v1",
        extraction_prompt_version="explicit_features_v5",
        post_extraction_review_max_operations=4,
        post_extraction_review_max_quality_retries=8,
        post_extraction_review_min_partition_rows=8,
        hierarchical_max_atoms_per_chunk=2,
        hierarchical_max_bytes_per_chunk=48_000,
        hierarchical_max_semantic_member_ids_per_chunk=3,
        hierarchical_max_cross_architecture_lookback_ids=24,
        hierarchical_max_cross_architecture_lookback_bytes=96_000,
        hierarchical_max_extraction_lookback_ids_per_feature=8,
        hierarchical_max_extraction_lookback_bytes_per_feature=96_000,
        hierarchical_max_rejection_lookback_ids_per_candidate=24,
        hierarchical_max_rejection_lookback_bytes_per_candidate=48_000,
        hierarchical_review_max_evidence_ids=512,
        hierarchical_review_max_evidence_bytes=2_000_000,
    )


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


def _evidence(
    *,
    evidence_id: str,
    source_family: str = BOW_NUISANCE,
    clue: str = "age",
) -> DiscoveryEvidenceItem:
    suffix = evidence_id.rsplit(".", 1)[-1]
    return DiscoveryEvidenceItem(
        evidence_id=evidence_id,
        source_family=source_family,
        observable_axes=("heterogeneity",),
        content={"readable_clue": clue},
        member_ids=(f"member.canary.{suffix}",),
    )


def _interpret_job(
    *,
    evidence: DiscoveryEvidenceItem,
    chunk_id: str,
    explanation: str = "Interpret the exact architecture-local readable clue.",
) -> DiscoveryJsonJob:
    return DiscoveryJsonJob.create(
        job_kind=INTERPRET_CHUNK_JOB,
        scope=f"{evidence.source_family}.chunk_000",
        dependencies=(),
        settings=DiscoveryJobSettings.selector(6_000),
        messages=render_interpret_evidence_chunk_messages(
            family_explanation=explanation,
            evidence=(evidence,),
        ),
        input_bindings={
            "catalog_sha256": "a" * 64,
            "chunk_plan_sha256": "b" * 64,
            "chunk_id": chunk_id,
            "source_family": evidence.source_family,
            HIERARCHICAL_DISCOVERY_IMPLEMENTATION_BUNDLE_BINDING: (
                hierarchical_discovery_implementation_bundle()["implementation_bundle_sha256"]
            ),
        },
    )


class _Atom:
    def __init__(self, item: DiscoveryEvidenceItem):
        self.evidence_id = item.evidence_id
        self._item = item

    def as_discovery_item(self) -> DiscoveryEvidenceItem:
        return self._item


class _Cache:
    def __init__(self) -> None:
        self.begin_calls: list[dict[str, Any]] = []
        self.replay_calls: list[dict[str, Any]] = []
        self.store_calls: list[dict[str, Any]] = []

    @property
    def execution_metadata(self) -> tuple[()]:
        return ()

    def identity(self) -> dict[str, Any]:
        body = {
            "schema_version": "offline_canary_cache_identity_v1",
            "mode": "read_write_immutable",
            "validated_only": True,
        }
        return {**body, "identity_sha256": content_sha256(body)}

    def begin_execution(self, **kwargs: Any) -> None:
        self.begin_calls.append(copy.deepcopy(kwargs))

    def replay_validated(self, **kwargs: Any) -> None:
        self.replay_calls.append(copy.deepcopy(kwargs))
        return None

    def store_validated(self, **kwargs: Any) -> None:
        self.store_calls.append(copy.deepcopy(kwargs))


class _Orchestrator:
    def __init__(self, *, job: DiscoveryJsonJob, runner_identity: Mapping[str, Any]):
        self.initial_job_ledger = SimpleNamespace(jobs=(job,))
        self.job_cache = _Cache()
        self.precommit = SimpleNamespace(precommit_sha256="c" * 64)
        self.runner_identity = copy.deepcopy(dict(runner_identity))
        self.implementation_bundle_sha256 = hierarchical_discovery_implementation_bundle()[
            "implementation_bundle_sha256"
        ]
        self.config = SimpleNamespace(max_rendered_prompt_bytes=10_000_000)
        self.implementation_checks: list[dict[str, Any]] = []

    def _assert_runner_identity(self, runner: Any) -> None:
        assert canonical_json(runner.identity()) == canonical_json(self.runner_identity)

    def _assert_implementation_bundle_unchanged(self, **kwargs: Any) -> None:
        self.implementation_checks.append(copy.deepcopy(kwargs))

    def _run(self, **kwargs: Any):
        assert isinstance(kwargs["runner"], _PerCallMetadataAuthenticatingRunner)
        return HierarchicalAllArchitectureDiscoveryOrchestrator._run(self, **kwargs)


class _Agent:
    def __init__(self, *, runner: Any, orchestrator: _Orchestrator):
        self.runner = runner
        self._orchestrator = orchestrator

    def _assert_unchanged(self):
        return self.runner.identity(), self._orchestrator


def _fold(
    *,
    outer_fold: int,
    runner: Any,
    evidence: DiscoveryEvidenceItem,
    job: DiscoveryJsonJob,
    orchestrator: _Orchestrator,
) -> Any:
    chunk_id = str(job.input_bindings["chunk_id"])
    chunk = SimpleNamespace(
        chunk_id=chunk_id,
        source_family=evidence.source_family,
        evidence=[{"evidence_id": evidence.evidence_id}],
    )
    return SimpleNamespace(
        outer_fold=outer_fold,
        agent=_Agent(runner=runner, orchestrator=orchestrator),
        catalog=SimpleNamespace(atoms=(_Atom(evidence),)),
        chunk_plan=SimpleNamespace(chunks=(chunk,)),
    )


def _runner_identity(
    *,
    endpoint: str = CAMUS_ENDPOINT,
    model_name: str = CAMUS_MODEL,
    generation_policy: Stage2GenerationPolicy | None = None,
    model_context_window_tokens: int = 131_072,
) -> dict[str, Any]:
    policy = _generation_policy() if generation_policy is None else generation_policy
    prompt_guard_body = {
        "model_name": model_name,
        "model_context_window_tokens": model_context_window_tokens,
        "accounting": {
            "apply_chat_template": True,
            "tokenize": True,
            "add_generation_prompt": True,
            "truncation": False,
            "endpoint_prompt_usage_exact_match_required": True,
            "request_truncation_controls_allowed": False,
        },
    }
    prompt_guard = {
        **prompt_guard_body,
        "identity_sha256": content_sha256(prompt_guard_body),
    }
    body = {
        "schema_version": "offline_canary_runner_v1",
        "endpoint_urls": [endpoint],
        "model": {
            "name": model_name,
            "resolution": "explicit_only_no_autodiscovery",
        },
        "retry": {"max_retries": 0, "max_attempts": 1},
        "generation_policy": policy.as_dict(),
        "generation_policy_sha256": policy.content_sha256,
        "generation_policy_resolution": "explicit_closed_policy",
        "prompt_nontruncation_guard": prompt_guard,
    }
    return {**body, "identity_sha256": content_sha256(body)}


class _MetadataRunner:
    def __init__(
        self,
        *,
        identity: Mapping[str, Any],
        evidence: DiscoveryEvidenceItem,
        endpoint: str,
        model_name: str,
        response_model: str | None = None,
        second_response_model: str | None = None,
        finish_reason: str | None = "stop",
        invalid_first_wire: bool = False,
    ):
        self._identity = copy.deepcopy(dict(identity))
        self._evidence = evidence
        self._endpoint = endpoint
        self._model_name = model_name
        self._response_model = model_name if response_model is None else response_model
        self._second_response_model = second_response_model
        self._finish_reason = finish_reason
        self._invalid_first_wire = invalid_first_wire
        self._metadata: list[dict[str, Any]] = []
        self.calls: list[DiscoveryJsonJob] = []
        self.closed = False

    def identity(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._identity)

    @property
    def execution_metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._metadata))

    def run_json(self, *, job: DiscoveryJsonJob) -> Mapping[str, Any]:
        self.calls.append(job)
        member_id = self._evidence.member_ids[0]
        if self._invalid_first_wire and len(self.calls) == 1:
            response = {"evidence_dispositions": {}}
        else:
            response = {
                "evidence_dispositions": {
                    self._evidence.evidence_id: {
                        "evidence_findings": [],
                        "member_dispositions": {member_id: {"findings": []}},
                        "reason": "No specific patient concept is supported.",
                    }
                }
            }
        request_sha256 = content_sha256(job.as_dict())
        response_sha256 = content_sha256(response)
        raw = canonical_json(response).encode("utf-8")
        attempt = {
            "attempt_number": 1,
            "endpoint": self._endpoint,
            "model": self._model_name,
            "request_sha256": request_sha256,
            "runner_identity_sha256": self._identity["identity_sha256"],
            "outcome": "success",
            "retryable": False,
            "will_retry": False,
            "response_id": "offline-fake-response",
            "response_model": (
                self._second_response_model
                if len(self.calls) == 2 and self._second_response_model is not None
                else self._response_model
            ),
            "usage": {},
            "content_sha256": hashlib.sha256(raw).hexdigest(),
            "reasoning_hashes": {},
            "raw_transport_bytes": len(raw),
            "parsed_response_sha256": response_sha256,
        }
        if self._finish_reason is not None:
            attempt["finish_reason"] = self._finish_reason
        self._metadata.append(
            {
                "job_id": job.job_id,
                "job_kind": job.job_kind,
                "request_sha256": request_sha256,
                "runner_identity_sha256": self._identity["identity_sha256"],
                "outcome": "success",
                "parsed_response_sha256": response_sha256,
                "attempts": [attempt],
            }
        )
        return response

    def close(self) -> None:
        self.closed = True


class _ProductionRunner:
    def __init__(
        self,
        *,
        hierarchy_runner: _MetadataRunner,
        prepared: Any,
        fusion_max_tokens: int,
        fusion_thinking_token_budget: int,
        fusion_enable_thinking: bool,
        extraction_enable_thinking: bool,
        post_extraction_review_config: CausalReviewConfig,
        post_extraction_scientific_policy: PostExtractionScientificPolicy,
    ):
        self.hierarchical_discovery_runner = hierarchy_runner
        self.hierarchical_discovery_approved_batch_sha256 = None
        self.config = SimpleNamespace(
            fusion_enable_thinking=fusion_enable_thinking,
            fusion_thinking_token_budget=fusion_thinking_token_budget,
            fusion_max_tokens=fusion_max_tokens,
            extraction_enable_thinking=extraction_enable_thinking,
            post_extraction_review_config=post_extraction_review_config,
            post_extraction_scientific_policy=(
                post_extraction_scientific_policy
            ),
        )
        self._prepared = prepared
        self.preparation_calls = 0

    def prepare_hierarchical_discovery_batch(self):
        self.preparation_calls += 1
        return self._prepared


class _Handoff:
    def __init__(self) -> None:
        self.inputs = SimpleNamespace(bundle_sha256="d" * 64)
        body = {
            "manual_digest_approval_required": False,
            "raw_all_architecture_prompt_allowed": False,
            "per_architecture_interpretation_required": True,
            "all_ten_architectures_required": True,
        }
        self._value = {**body, "content_sha256": content_sha256(body)}

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._value)


def _options(
    tmp_path: Path,
    *,
    endpoint: str = CAMUS_ENDPOINT,
    model_name: str = CAMUS_MODEL,
) -> ProductionStage1HierarchyOneShotOptions:
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    bundle = bundle_root / "bundle_manifest.json"
    bundle.write_text("{}\n", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    return ProductionStage1HierarchyOneShotOptions(
        bundle_manifest_path=bundle,
        output_dir=tmp_path / "canary_output",
        preparation_dir=tmp_path / "canary_preparation",
        attestation_dir=tmp_path / "canary_attestation",
        endpoint=endpoint,
        model_name=model_name,
        review_rounds=1,
        initial_training_partitions=3,
        stage2_protocol=_stage2_protocol(),
        stage2_tokenizer_locator=tokenizer,
        hierarchical_discovery_job_cache_config=HIERARCHY_JOB_CACHE_CONFIG,
        first_untouched_gate_preparation_bounds=FIRST_UNTOUCHED_GATE_BOUNDS,
        post_extraction_review_config=_causal_review_config(),
        post_extraction_scientific_policy=(
            _post_extraction_scientific_policy()
        ),
        review_stage1_device="cpu",
        review_neural_query_devices=("cpu",),
        source_text_temporally_valid_by_design=True,
        proposal_schema_repair_attempts=1,
        request_max_retries=0,
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
        extraction_max_text_length=119,
        complete_page_core_chars=97,
        complete_page_context_chars=11,
        complete_page_max_chars=119,
        complete_reconciliation_fan_in=7,
    )


def _install_fake_production_graph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    options: ProductionStage1HierarchyOneShotOptions,
    response_model: str | None = None,
    second_response_model: str | None = None,
    finish_reason: str | None = "stop",
    invalid_first_wire: bool = False,
) -> tuple[_MetadataRunner, _Orchestrator, _ProductionRunner]:
    identity = _runner_identity(
        endpoint=options.endpoint,
        model_name=options.model_name,
        generation_policy=options.stage2_protocol.generation_policy,
        model_context_window_tokens=options.model_context_window_tokens,
    )
    evidence = _evidence(evidence_id="evidence.canary.success")
    job = _interpret_job(evidence=evidence, chunk_id="chunk.canary.success")
    hierarchy_runner = _MetadataRunner(
        identity=identity,
        evidence=evidence,
        endpoint=options.endpoint,
        model_name=options.model_name,
        response_model=response_model,
        second_response_model=second_response_model,
        finish_reason=finish_reason,
        invalid_first_wire=invalid_first_wire,
    )
    orchestrator = _Orchestrator(job=job, runner_identity=identity)
    prepared = SimpleNamespace(
        folds=(
            _fold(
                outer_fold=1,
                runner=hierarchy_runner,
                evidence=evidence,
                job=job,
                orchestrator=orchestrator,
            ),
        )
    )
    production_runner = _ProductionRunner(
        hierarchy_runner=hierarchy_runner,
        prepared=prepared,
        fusion_max_tokens=(
            options.stage2_protocol.generation_policy
            .feature_proposal_review.max_tokens
        ),
        fusion_thinking_token_budget=(
            options.stage2_protocol.generation_policy
            .feature_proposal_review.thinking_token_budget
        ),
        fusion_enable_thinking=(
            options.stage2_protocol.generation_policy
            .feature_proposal_review.thinking_enabled
        ),
        extraction_enable_thinking=(
            options.stage2_protocol.generation_policy
            .patient_feature_extraction.thinking_enabled
        ),
        post_extraction_review_config=options.post_extraction_review_config,
        post_extraction_scientific_policy=(
            options.post_extraction_scientific_policy
        ),
    )
    handoff = _Handoff()

    def load_handoff(path: Path, **kwargs: Any):
        assert path == options.bundle_manifest_path
        assert kwargs == {
            "review_rounds": 1,
            "initial_training_partitions": 3,
            "interaction_inner_folds": 3,
            "tfidf_nested_calibration_folds": 3,
        }
        return handoff

    def build_runner(**kwargs: Any):
        assert kwargs["handoff"] is handoff
        assert kwargs["options"] is options
        assert kwargs["endpoint"] == options.endpoint
        assert "model_identity" not in kwargs
        return production_runner

    monkeypatch.setattr(canary, "load_production_stage1_hierarchy_handoff", load_handoff)
    monkeypatch.setattr(canary, "build_production_stage1_hierarchy_runner", build_runner)
    return hierarchy_runner, orchestrator, production_runner


def _walk_mapping(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mapping(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mapping(child)


def test_cli_keeps_safety_controls_fixed_but_requires_scientific_token_budgets(
    tmp_path: Path,
) -> None:
    parser = canary.build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--model-identity-json" not in option_strings
    forbidden_fragments = (
        "approval",
        "digest",
        "replay",
        "oracle",
        "prediction",
        "retry",
        "repair",
    )
    assert not {
        option
        for option in option_strings
        if any(fragment in option for fragment in forbidden_fragments)
    }
    actions = {action.dest: action for action in parser._actions}
    assert actions["proposal_max_tokens"].required is True
    assert actions["extraction_max_tokens"].required is True
    assert actions["review_stage1_device"].required is True
    assert actions["review_neural_query_device"].required is True
    assert actions["stage2_tokenizer_locator"].required is True
    assert actions["hierarchical_job_cache_max_entry_bytes"].required is True
    for field_name in FIRST_UNTOUCHED_GATE_BOUNDS.__dataclass_fields__:
        assert actions["first_untouched_gate_" + field_name].required is True

    options = _options(tmp_path)
    argv = [
        "--bundle-manifest",
        str(options.bundle_manifest_path),
        "--scratch-output-dir",
        str(options.output_dir),
        "--hierarchical-preparation-dir",
        str(options.preparation_dir),
        "--report-dir",
        str(options.attestation_dir),
        "--endpoint",
        CAMUS_ENDPOINT,
        "--model",
        CAMUS_MODEL,
        "--stage2-tokenizer-locator",
        str(options.stage2_tokenizer_locator),
        "--review-rounds",
        "1",
        "--initial-training-partitions",
        "3",
        "--hierarchical-job-cache-max-entry-bytes",
        str(options.hierarchical_discovery_job_cache_config.max_entry_bytes),
        "--source-text-temporally-valid-by-design",
        "--review-stage1-device",
        "cpu",
        "--review-neural-query-device",
        "cpu",
        "--max-candidate-variables",
        str(options.max_candidates),
        "--complete-page-core-chars",
        str(options.complete_page_core_chars),
        "--complete-page-context-chars",
        str(options.complete_page_context_chars),
        "--complete-page-max-chars",
        str(options.complete_page_max_chars),
        "--complete-reconciliation-fan-in",
        str(options.complete_reconciliation_fan_in),
        "--forest-n-estimators",
        str(options.forest_n_estimators),
        "--forest-max-depth",
        str(options.forest_max_depth),
        "--forest-min-samples-leaf",
        str(options.forest_min_samples_leaf),
        "--forest-max-features",
        str(options.forest_max_features),
        "--forest-honest",
        "--forest-inference",
        "--forest-subforest-size",
        str(options.forest_subforest_size),
        "--no-forest-tune-model",
        "--forest-nuisance-n-estimators",
        str(options.forest_nuisance_n_estimators),
        "--forest-nuisance-max-depth",
        str(options.forest_nuisance_max_depth),
        "--forest-nuisance-min-samples-leaf",
        str(options.forest_nuisance_min_samples_leaf),
        "--forest-nuisance-treatment-max-features",
        str(options.forest_nuisance_treatment_max_features),
        "--forest-nuisance-outcome-max-features",
        str(options.forest_nuisance_outcome_max_features),
        "--forest-random-seed",
        str(options.forest_random_seed),
        "--forest-n-jobs",
        str(options.forest_n_jobs),
    ]
    for name, value in asdict(
        options.first_untouched_gate_preparation_bounds
    ).items():
        argv.extend(
            (
                "--first-untouched-gate-" + name.replace("_", "-"),
                str(value),
            )
        )
    wire_budget_path = tmp_path / "hierarchy_wire_budget.json"
    wire_budget_path.write_text(
        json.dumps(
            options.stage2_protocol.hierarchy_wire_budget.as_dict(),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for name, value in options.stage2_protocol.as_dict().items():
        if name in {
            "schema_version",
            "hierarchy_wire_budget",
            "generation_policy",
        }:
            continue
        argv.extend(("--" + name.replace("_", "-"), str(value)))
    argv.extend(("--hierarchy-wire-budget", str(wire_budget_path)))
    generation_policy_path = tmp_path / "generation_policy.json"
    generation_policy_path.write_text(
        json.dumps(
            options.stage2_protocol.generation_policy.as_dict(),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    argv.extend(("--generation-policy", str(generation_policy_path)))
    for name, value in asdict(options.post_extraction_review_config).items():
        if name == "estimator_policy":
            continue
        argv.extend(("--causal-review-" + name.replace("_", "-"), str(value)))
    scientific_policy_path = tmp_path / "post_extraction_scientific_policy.json"
    scientific_policy_path.write_text(
        json.dumps(
            options.post_extraction_scientific_policy.as_dict(),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    argv.extend(
        (
            "--post-extraction-scientific-policy",
            str(scientific_policy_path),
        )
    )
    parsed = canary.options_from_args(parser.parse_args(argv))
    assert parsed.initial_training_partitions == 3
    assert parsed.stage2_protocol == options.stage2_protocol
    assert parsed.post_extraction_review_config == options.post_extraction_review_config
    assert parsed.proposal_max_tokens == 26_000
    assert parsed.extraction_max_tokens == 18_000
    assert parsed.request_max_retries == 0
    assert parsed.proposal_schema_repair_attempts == 1

    local_argv = list(argv)
    local_argv[local_argv.index(CAMUS_ENDPOINT)] = LOCAL_ENDPOINT
    local_argv[local_argv.index(CAMUS_MODEL)] = LOCAL_MODEL
    local = canary.options_from_args(parser.parse_args(local_argv))
    assert local.endpoint == LOCAL_ENDPOINT
    assert local.model_name == LOCAL_MODEL

    bad_endpoint = list(argv)
    bad_endpoint[bad_endpoint.index(CAMUS_ENDPOINT)] = f"{CAMUS_ENDPOINT},{LOCAL_ENDPOINT}"
    with pytest.raises(ValueError, match="single|pool|comma"):
        canary.options_from_args(parser.parse_args(bad_endpoint))


def test_selection_uses_smallest_real_architecture_pure_prompt() -> None:
    identity = _runner_identity()
    runner = SimpleNamespace(identity=lambda: copy.deepcopy(identity))

    long_evidence = _evidence(
        evidence_id="evidence.canary.long",
        clue="a much longer architecture-local readable clue " * 8,
    )
    short_evidence = _evidence(evidence_id="evidence.canary.short", clue="age")
    long_job = _interpret_job(evidence=long_evidence, chunk_id="chunk.canary.long")
    short_job = _interpret_job(evidence=short_evidence, chunk_id="chunk.canary.short")
    long_orchestrator = _Orchestrator(job=long_job, runner_identity=identity)
    short_orchestrator = _Orchestrator(job=short_job, runner_identity=identity)
    prepared = SimpleNamespace(
        folds=(
            _fold(
                outer_fold=1,
                runner=runner,
                evidence=long_evidence,
                job=long_job,
                orchestrator=long_orchestrator,
            ),
            _fold(
                outer_fold=2,
                runner=runner,
                evidence=short_evidence,
                job=short_job,
                orchestrator=short_orchestrator,
            ),
        )
    )

    selected = canary._select_smallest_initial_interpretation_job(
        prepared_batch=prepared,
        production_hierarchy_runner=runner,
    )

    assert selected.job.job_id == short_job.job_id
    assert selected.source_family == BOW_NUISANCE
    assert selected.evidence == (short_evidence,)
    assert selected.rendered_message_bytes < len(long_job.rendered_messages_bytes)


def test_selection_rejects_architecture_mismatch() -> None:
    identity = _runner_identity()
    runner = SimpleNamespace(identity=lambda: copy.deepcopy(identity))
    evidence = _evidence(evidence_id="evidence.canary.mismatch")
    job = _interpret_job(evidence=evidence, chunk_id="chunk.canary.mismatch")
    orchestrator = _Orchestrator(job=job, runner_identity=identity)
    fold = _fold(
        outer_fold=1,
        runner=runner,
        evidence=evidence,
        job=job,
        orchestrator=orchestrator,
    )
    fold.chunk_plan.chunks[0].source_family = BOW_R_LOSS

    with pytest.raises(ValueError, match="different architectures"):
        canary._select_smallest_initial_interpretation_job(
            prepared_batch=SimpleNamespace(folds=(fold,)),
            production_hierarchy_runner=runner,
        )


def test_run_canary_uses_one_authenticated_production_job_and_emits_hashes_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    runner, orchestrator, production_runner = _install_fake_production_graph(
        monkeypatch,
        options=options,
    )

    summary = canary.run_canary(options)

    assert summary["status"] == "accepted"
    assert summary["remote_response_count"] == 1
    assert len(runner.calls) == 1
    assert production_runner.preparation_calls == 1
    assert runner.closed is True
    assert len(orchestrator.job_cache.begin_calls) == 1
    assert len(orchestrator.job_cache.replay_calls) == 1
    assert len(orchestrator.job_cache.store_calls) == 1
    assert not (options.output_dir / "frozen_predictions.parquet").exists()
    assert not (options.output_dir / "immutable_run_manifest.json").exists()

    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    assert report["schema_version"] == canary.CANARY_REPORT_SCHEMA
    body = report["body"]
    assert body["endpoint"] == CAMUS_ENDPOINT
    assert body["model"] == CAMUS_MODEL
    assert "served_deployment_identity" not in body
    assert body["authorization_role"] == "non_authorizing_operational_runtime_check"
    settings = body["settings"]
    assert settings["proposal_max_tokens"] == options.proposal_max_tokens
    assert settings["extraction_max_tokens"] == options.extraction_max_tokens
    assert settings["stage2_hierarchy_prompt_protocol"] == (
        options.stage2_protocol.as_dict()
    )
    assert settings["stage2_hierarchy_prompt_protocol_sha256"] == (
        options.stage2_protocol.content_sha256
    )
    assert settings["extraction_thinking_enabled"] is False
    assert settings["maximum_schema_repairs"] == 1
    assert settings["selector_thinking_enabled"] is True
    assert settings["selector_thinking_token_budget"] == (
        options.selector_thinking_token_budget
    )
    assert settings["max_rendered_discovery_prompt_bytes"] == 350_000
    assert settings["final_upstream_max_orphan_features"] == 37
    assert settings["review_neural_query_nuisance_folds"] == 4
    assert settings["final_upstream_meta_inner_folds"] == 4
    assert settings["final_upstream_head_regularization"] == 0.75
    assert settings["transport_retries"] == 0
    assert body["remote_response_count"] == 1
    assert body["transport_metadata"][0]["attempts"][0]["response_model"] == (CAMUS_MODEL)
    assert body["transport_metadata"][0]["attempts"][0]["finish_reason"] == "stop"
    assert body["prediction_path_constructed"] is False
    assert body["oracle_path_constructed"] is False
    assert body["validation"]["job_cache_identity_sha256"] == (
        orchestrator.job_cache.identity()["identity_sha256"]
    )
    forbidden = canary._FORBIDDEN_OUTPUT_KEYS
    assert all(not (set(row) & forbidden) for row in _walk_mapping(report))
    serialized = canonical_json(report)
    assert "No specific patient concept is supported" not in serialized
    assert "readable_clue" not in serialized


def test_run_canary_binds_an_intentional_local_endpoint_and_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(
        tmp_path,
        endpoint=LOCAL_ENDPOINT,
        model_name=LOCAL_MODEL,
    )
    runner, orchestrator, _production_runner = _install_fake_production_graph(
        monkeypatch,
        options=options,
    )

    summary = canary.run_canary(options)

    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    body = report["body"]
    assert body["endpoint"] == LOCAL_ENDPOINT
    assert body["model"] == LOCAL_MODEL
    attempt = body["transport_metadata"][0]["attempts"][0]
    assert attempt["endpoint"] == LOCAL_ENDPOINT
    assert attempt["model"] == LOCAL_MODEL
    assert attempt["response_model"] == LOCAL_MODEL
    assert attempt["finish_reason"] == "stop"
    assert len(runner.calls) == 1
    assert len(orchestrator.job_cache.store_calls) == 1


def test_run_canary_allows_only_the_single_authenticated_schema_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    runner, orchestrator, _production_runner = _install_fake_production_graph(
        monkeypatch,
        options=options,
        invalid_first_wire=True,
    )

    summary = canary.run_canary(options)

    assert summary["remote_response_count"] == 2
    assert len(runner.calls) == 2
    assert len(orchestrator.job_cache.store_calls) == 1
    report = json.loads(Path(summary["report_path"]).read_text(encoding="utf-8"))
    body = report["body"]
    assert body["validation"]["response_attempt_outcomes"] == [
        "local_json_schema_validation_failure",
        "validated_response",
    ]
    assert all(len(record["attempts"]) == 1 for record in body["transport_metadata"])
    assert all(
        record["attempts"][0]["response_model"] == CAMUS_MODEL
        and record["attempts"][0]["finish_reason"] == "stop"
        for record in body["transport_metadata"]
    )


def test_run_canary_authenticates_the_repair_response_before_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    runner, orchestrator, _production_runner = _install_fake_production_graph(
        monkeypatch,
        options=options,
        invalid_first_wire=True,
        second_response_model="substituted-repair-model",
    )

    with pytest.raises(ValueError, match="response model differs"):
        canary.run_canary(options)

    assert len(runner.calls) == 2
    assert runner.closed is True
    assert orchestrator.job_cache.store_calls == []
    assert not options.attestation_dir.exists()


@pytest.mark.parametrize(
    ("response_model", "finish_reason", "error"),
    [
        ("substituted-model", "stop", "response model differs"),
        (None, "length", "finish_reason must be exactly 'stop'"),
        (None, None, "finish_reason must be exactly 'stop'"),
    ],
)
def test_run_canary_rejects_unauthenticated_response_metadata_before_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_model: str | None,
    finish_reason: str | None,
    error: str,
) -> None:
    options = _options(tmp_path)
    runner, orchestrator, _production_runner = _install_fake_production_graph(
        monkeypatch,
        options=options,
        response_model=response_model,
        finish_reason=finish_reason,
    )

    with pytest.raises(ValueError, match=error):
        canary.run_canary(options)

    assert len(runner.calls) == 1
    assert runner.closed is True
    assert orchestrator.job_cache.store_calls == []
    assert not options.attestation_dir.exists()
    assert not (options.output_dir / "frozen_predictions.parquet").exists()
