from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

import oci.inference.production_authenticated_tree_cache as tree_module
import oci.inference.production_all_evidence_workflow as workflow_module
import oci.inference.role_neutral_benchmark_deployment_selection as selection_module
from oci.inference.portable_artifacts import (
    ArtifactCompatibility,
    publish_portable_artifact,
    validate_portable_artifact,
)
from oci.inference.portable_workflow_spec import (
    DeploymentProfile,
    EVIDENCE_FAMILIES,
    HierarchyWireBudgetSpec,
    PostExtractionCausalReviewSpec,
    ResourcePerformanceSafetyPolicy,
    RunControl,
    SentenceEmbeddingEncoderSpec,
    Stage1ExecutionProfile,
    Stage2PromptProtocolSpec,
)
from oci.inference.production_all_evidence_workflow import (
    EMBEDDING_CACHE_PHASE_SCHEMA,
    PHASES,
    STAGE1_PREFLIGHT_PHASE_SCHEMA,
    STAGE1_ONLY_PHASES,
    ProductionAllEvidenceWorkflow,
    ProductionAllEvidenceWorkflowHooks,
    ProductionAllEvidenceWorkflowOptions,
    build_parser,
    options_from_args,
)
from oci.models.strict_causal_forest_runtime import (
    STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
    StrictCausalForestRuntimeConfig,
)
from oci.inference.all_evidence_post_extraction_review import (
    GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
)
from oci.inference.review_spent_evidence_provider import (
    semantic_witness_config_from_portable_scientific_spec,
)
from tests.test_portable_workflow_contracts import (
    _forest_operational,
    _generation_policy,
    _post_extraction_policy,
    _scientific_spec,
)


def _wire_budget() -> HierarchyWireBudgetSpec:
    return HierarchyWireBudgetSpec(
        budget_version="hierarchy_wire_budget_v1",
        max_opaque_identifier_chars=128,
        max_generated_name_chars=64,
        max_description_chars=128,
        max_reason_chars=128,
        max_ambiguity_chars=128,
        max_free_text_chars=128,
        max_generated_list_items=8,
        max_feature_names_per_member=4,
        max_findings_per_atomic_review=4,
        max_pair_relation_peers_per_page=7,
        max_definition_fold_inputs=8,
        max_group_lookback_ids=8,
        max_adaptive_review_targets=4,
        max_interpret_atoms_per_job=2,
        max_interpret_members_per_job=3,
        max_interpret_name_chars=64,
        max_interpret_description_chars=96,
        max_interpret_ambiguity_chars=96,
        max_interpret_reason_chars=64,
        max_interpret_canonical_json_bytes=20_000,
        max_interpret_transport_bytes=20_000,
        interpret_generation_token_budget=20_000,
        max_response_transport_bytes=20_000,
        generation_token_budget=20_000,
    )


def _stage2_protocol() -> Stage2PromptProtocolSpec:
    return Stage2PromptProtocolSpec(
        proposal_max_tokens=25_000,
        extraction_max_tokens=25_000,
        extraction_grouping_strategy="packed",
        extraction_context_strategy="complete_paged_v1",
        extraction_prompt_version="explicit_features_v5",
        model_context_window_tokens=131_072,
        post_extraction_review_max_operations=4,
        post_extraction_review_max_quality_retries=8,
        post_extraction_review_min_partition_rows=8,
        hierarchical_max_atoms_per_chunk=2,
        hierarchical_max_bytes_per_chunk=96_000,
        hierarchical_max_semantic_member_ids_per_chunk=3,
        hierarchical_max_cross_architecture_lookback_ids=24,
        hierarchical_max_cross_architecture_lookback_bytes=96_000,
        hierarchical_max_extraction_lookback_ids_per_feature=8,
        hierarchical_max_extraction_lookback_bytes_per_feature=96_000,
        hierarchical_max_rejection_lookback_ids_per_candidate=24,
        hierarchical_max_rejection_lookback_bytes_per_candidate=48_000,
        hierarchical_review_max_evidence_ids=512,
        hierarchical_review_max_evidence_bytes=2_000_000,
        max_rendered_discovery_prompt_bytes=220_000,
        selector_thinking_token_budget=5_000,
        final_upstream_max_orphan_features=32,
        review_neural_query_nuisance_folds=3,
        final_upstream_meta_inner_folds=3,
        final_upstream_head_regularization=1.0,
        query_moment_max_queries=24,
        query_moment_max_terms_per_query=32,
        query_moment_max_chunks_per_query=16,
        query_moment_fallback_chunks_per_query=8,
        query_moment_max_excerpt_chars=1200,
        query_moment_max_term_chars=160,
        query_moment_max_ngram_tokens=6,
        hierarchy_wire_budget=_wire_budget(),
        generation_policy=_generation_policy(),
    )


def _causal_review() -> PostExtractionCausalReviewSpec:
    return PostExtractionCausalReviewSpec(
        upstream_review_policy=GATE_ONLY_REFERENCE_PRESERVATION_REVIEW_POLICY,
        e_clip=0.05,
        nuisance_ridge_alpha=1.0,
        effect_ridge_alpha=1.0,
        contract_complexity_penalty=0.002,
        encoded_column_complexity_penalty=0.0002,
        minimum_score_improvement=0.0,
        nuisance_relative_tolerance=0.05,
        source_preservation_tolerance=0.05,
        source_context_r_loss_relative_tolerance=0.05,
        feature_bank_preservation_tolerance=0.05,
        scientific_policy=_post_extraction_policy(),
    )


def _resource_performance_safety() -> ResourcePerformanceSafetyPolicy:
    return ResourcePerformanceSafetyPolicy(
        gpu_max_allocation_fraction=0.85,
        gpu_minimum_headroom_bytes=6 * 1024**3,
        minimum_multi_device_throughput_ratio=1.5,
        maximum_coordination_proof_overhead_ratio=0.3,
        maximum_ordinary_read_amplification=2.0,
        minimum_benchmark_repetitions_per_scope=2,
        read_counter_source="logical_read_bytes",
        fail_on_external_gpu_occupants=True,
    )


def _resource_performance_safety_cli_args(
    policy: ResourcePerformanceSafetyPolicy,
) -> list[str]:
    occupant_flag = (
        "--fail-on-external-gpu-occupants"
        if policy.fail_on_external_gpu_occupants
        else "--no-fail-on-external-gpu-occupants"
    )
    return [
        "--gpu-max-allocation-fraction",
        str(policy.gpu_max_allocation_fraction),
        "--gpu-minimum-headroom-bytes",
        str(policy.gpu_minimum_headroom_bytes),
        "--minimum-multi-device-throughput-ratio",
        str(policy.minimum_multi_device_throughput_ratio),
        "--maximum-coordination-proof-overhead-ratio",
        str(policy.maximum_coordination_proof_overhead_ratio),
        "--maximum-ordinary-read-amplification",
        str(policy.maximum_ordinary_read_amplification),
        "--minimum-benchmark-repetitions-per-scope",
        str(policy.minimum_benchmark_repetitions_per_scope),
        "--performance-read-counter-source",
        policy.read_counter_source,
        occupant_flag,
    ]


def _options(tmp_path: Path, *, endpoint="https://different.example/v1", model="different/model"):
    files = []
    for name in ("dataset.parquet", "stage1.json", "query.json"):
        path = tmp_path / name
        path.write_text(
            name if name == "dataset.parquet" else "{}",
            encoding="utf-8",
        )
        files.append(path)
    embed = (tmp_path / "embed").resolve()
    embed.mkdir(exist_ok=True)
    (embed / "model.safetensors").write_bytes(b"safe embedding model")
    htr = (tmp_path / "htr").resolve()
    htr.mkdir(exist_ok=True)
    # The HTR path intentionally keeps the legacy full-byte validation path.
    (htr / "model.safetensors").write_bytes(b"safe htr model")
    stage2_tokenizer = (tmp_path / "stage2-tokenizer").resolve()
    stage2_tokenizer.mkdir(exist_ok=True)
    (stage2_tokenizer / "tokenizer.json").write_text(
        '{"fixture":"exact stage2 tokenizer"}',
        encoding="utf-8",
    )
    protocol = _stage2_protocol()
    (tmp_path / "stage2_protocol.json").write_text(
        json.dumps(protocol.as_dict(), sort_keys=True),
        encoding="utf-8",
    )
    causal_review = _causal_review()
    (tmp_path / "causal_review.json").write_text(
        json.dumps(causal_review.as_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return ProductionAllEvidenceWorkflowOptions(
        dataset_path=files[0],
        work_root=tmp_path / "run",
        stage1_profile_path=files[1],
        query_profile_path=files[2],
        unit_id_column="id",
        text_column="text",
        treatment_column="a",
        outcome_column="y",
        outcome_type="binary",
        clinical_question="question",
        embedding_model_name="logical/embed",
        embedding_local_model_path=embed,
        htr_local_model_path=htr,
        run_control=RunControl(),
        endpoint=endpoint,
        model_name=model,
        stage2_tokenizer_locator=stage2_tokenizer,
        outer_folds=5,
        review_rounds=2,
        initial_training_partitions=3,
        interaction_inner_folds=3,
        tfidf_nested_calibration_folds=3,
        stage1_device="cpu",
        query_device="cpu",
        review_device="cpu",
        gpu_id=None,
        stage1_seed_policy="canonical_group_sha256_v1",
        max_candidate_variables=7,
        stage2_prompt_protocol=protocol,
        post_extraction_causal_review=causal_review,
        resource_performance_safety=_resource_performance_safety(),
        cluster_preflight_parquet_compression="zstd",
        complete_page_core_chars=97,
        complete_page_context_chars=11,
        complete_page_max_chars=119,
        complete_reconciliation_fan_in=7,
        embedding_chunk_size_words=31,
        embedding_chunk_overlap_words=7,
        embedding_max_chunks=4096,
        embedding_chunk_selection="last",
        embedding_max_seq_length=512,
        embedding_normalize=True,
        embedding_encoder=SentenceEmbeddingEncoderSpec(
            prompt_policy="disabled",
            prompt_name=None,
            output_value="sentence_embedding",
            precision="float32",
            convert_to_numpy=True,
            convert_to_tensor=False,
            truncate_dim=None,
            pooling_output_policy="single_process_sentence_embedding_v1",
            model_dtype="float32",
            stored_array_dtype="float32",
            zero_vector_policy="reject",
        ),
        embedding_batch_size=13,
        seed=42,
        empty_text_policy="marker",
        repeated_character_policy="marker",
        repeated_character_threshold=1000,
        source_text_temporally_valid_by_design=True,
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
        forest_random_seed=19,
    )


def _with_run_control(
    options: ProductionAllEvidenceWorkflowOptions,
    **changes,
) -> ProductionAllEvidenceWorkflowOptions:
    return replace(
        options,
        run_control=replace(options.run_control, **changes),
    )


def _portable_options(
    tmp_path: Path,
) -> ProductionAllEvidenceWorkflowOptions:
    specification = _scientific_spec()
    base = _options(tmp_path)
    runtime = StrictCausalForestRuntimeConfig(
        schema_version=STRICT_CAUSAL_FOREST_RUNTIME_SCHEMA,
        causal_forest=specification.causal_estimator,
        operational=_forest_operational(base.cpu_budget),
    )
    return replace(
        base,
        portable_scientific_spec=specification.identity_payload(),
        stage1_execution_profile=Stage1ExecutionProfile(
            resource_kind="cpu",
            device_count=base.stage1_execution_device_count,
            scope_workers_per_device=base.stage1_scope_workers_per_gpu,
            executor_mode="persistent_slots",
            selection_method="operator_configured",
            selected_candidate=None,
            benchmark_result_sha256=None,
            benchmark_result_locator=None,
            benchmark_workload_deployment_sha256=None,
            benchmark_workload_deployment_locator=None,
        ),
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
    )


def _write_scientific_spec(path: Path) -> Path:
    specification = _scientific_spec()
    payload = asdict(specification)
    payload["stage2_prompt_protocol"] = specification.stage2_prompt_protocol.as_dict()
    payload["post_extraction_causal_review"] = specification.post_extraction_causal_review.as_dict()
    path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _direct_deployment_args(
    *,
    options: ProductionAllEvidenceWorkflowOptions,
    scientific_spec: Path,
    scratch_root: Path,
) -> list[str]:
    forest_operational_path = scratch_root.parent / "forest-operational.json"
    forest_operational_path.write_text(
        json.dumps(
            asdict(_forest_operational(2)),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return [
        "--scientific-spec",
        str(scientific_spec),
        "--dataset",
        str(options.dataset_path),
        "--work-root",
        str(options.work_root),
        "--scratch-root",
        str(scratch_root),
        "--stage1-profile",
        str(options.stage1_profile_path),
        "--query-profile",
        str(options.query_profile_path),
        "--embedding-model-name",
        options.embedding_model_name,
        "--embedding-local-model-path",
        str(options.embedding_local_model_path),
        "--htr-local-model-path",
        str(options.htr_local_model_path),
        "--embedding-batch-size",
        str(options.embedding_batch_size),
        "--devices",
        "cpu",
        "--stage1-device-count",
        "1",
        "--stage1-scope-workers-per-device",
        "1",
        "--cpu-budget",
        "2",
        "--forest-operational",
        str(forest_operational_path),
        "--response-concurrency",
        "3",
        "--storage-backend",
        "posix",
        "--cluster-preflight-parquet-compression",
        str(options.cluster_preflight_parquet_compression),
        "--runtime-compatibility-class",
        "portable-test-runtime-v1",
        *_resource_performance_safety_cli_args(options.resource_performance_safety),
        "--endpoint",
        str(options.endpoint),
        "--model",
        str(options.model_name),
        "--stage2-tokenizer-locator",
        str(options.stage2_tokenizer_locator),
    ]


def _write_deployment_profile(
    path: Path,
    options: ProductionAllEvidenceWorkflowOptions,
) -> Path:
    profile = DeploymentProfile(
        dataset_path=options.dataset_path,
        durable_artifact_root=options.work_root,
        scratch_root=path.parent / "typed-scratch",
        embedding_model_locator=options.embedding_local_model_path,
        htr_model_locator=options.htr_local_model_path,
        stage1_profile_locator=options.stage1_profile_path,
        query_profile_locator=options.query_profile_path,
        embedding_batch_size=options.embedding_batch_size,
        cluster_preflight_parquet_compression=(
            options.cluster_preflight_parquet_compression
        ),
        resource_performance_safety=options.resource_performance_safety,
        forest_operational=_forest_operational(2),
        stage1_execution=Stage1ExecutionProfile(
            resource_kind="cpu",
            device_count=1,
            scope_workers_per_device=1,
            executor_mode="persistent_slots",
            selection_method="operator_configured",
            selected_candidate=None,
            benchmark_result_sha256=None,
            benchmark_result_locator=None,
            benchmark_workload_deployment_sha256=None,
            benchmark_workload_deployment_locator=None,
        ),
        embedding_model_name=options.embedding_model_name,
        endpoint=options.endpoint,
        endpoint_model=options.model_name,
        stage2_tokenizer_locator=options.stage2_tokenizer_locator,
        devices=("cpu",),
        cpu_budget=2,
        response_concurrency=3,
        storage_backend="posix",
        runtime_compatibility_class="portable-test-runtime-v1",
    )
    path.write_text(
        json.dumps(asdict(profile), default=str, sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_stage2_options_propagate_complete_forest_spec_and_cpu_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = replace(_options(tmp_path), cpu_budget=6)
    workflow = ProductionAllEvidenceWorkflow(options)
    bundle = tmp_path / "sealed_stage1" / "bundle_manifest.json"
    bundle.parent.mkdir()
    bundle.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        workflow,
        "_validated_complete",
        lambda phase: {
            "phase": phase,
            "artifacts": [{"path": str(bundle)}],
        },
    )

    stage2 = workflow._stage2_options(tmp_path / "stage2", prefix="test")

    assert {
        "n_estimators": stage2.forest_n_estimators,
        "max_depth": stage2.forest_max_depth,
        "min_samples_leaf": stage2.forest_min_samples_leaf,
        "max_features": stage2.forest_max_features,
        "honest": stage2.forest_honest,
        "inference": stage2.forest_inference,
        "subforest_size": stage2.forest_subforest_size,
        "tune_model": stage2.forest_tune_model,
        "nuisance_n_estimators": stage2.forest_nuisance_n_estimators,
        "nuisance_max_depth": stage2.forest_nuisance_max_depth,
        "nuisance_min_samples_leaf": stage2.forest_nuisance_min_samples_leaf,
        "nuisance_treatment_max_features": (stage2.forest_nuisance_treatment_max_features),
        "nuisance_outcome_max_features": (stage2.forest_nuisance_outcome_max_features),
        "random_state": stage2.forest_random_seed,
    } == {
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
        "nuisance_min_samples_leaf": options.forest_nuisance_min_samples_leaf,
        "nuisance_treatment_max_features": (options.forest_nuisance_treatment_max_features),
        "nuisance_outcome_max_features": (options.forest_nuisance_outcome_max_features),
        "random_state": options.forest_random_seed,
    }
    assert stage2.forest_n_jobs == options.cpu_budget == 6
    assert (
        stage2.post_extraction_scientific_policy
        == options.post_extraction_causal_review.scientific_policy
    )
    assert (
        stage2.post_extraction_review_config.estimator_policy
        == options.post_extraction_causal_review.scientific_policy.review_estimator
    )


def test_endpoint_model_and_phase_resume_are_configuration_bound(tmp_path):
    options = _options(tmp_path)
    calls = []
    overrides = {
        phase: (lambda _attempt, value=phase: calls.append(value) or {"terminal_files": []})
        for phase in PHASES
    }
    ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()
    assert calls == list(PHASES)
    telemetry = json.loads(
        (options.work_root / "execution_attestations" / "performance_telemetry.json").read_text(
            encoding="utf-8"
        )
    )
    assert telemetry["schema_version"] == "portable_workflow_subphase_telemetry_v1"
    assert telemetry["resource_performance_safety"] == options.resource_performance_safety.as_dict()
    assert (
        telemetry["resource_performance_safety_sha256"]
        == options.resource_performance_safety.content_sha256
    )
    assert {row["name"] for row in telemetry["subphases"]} >= {
        f"{phase}.compute" for phase in PHASES
    } | {f"{phase}.proof_and_publication" for phase in PHASES}
    calls.clear()
    ProductionAllEvidenceWorkflow(
        _with_run_control(options, resume=True),
        phase_overrides=overrides,
    ).run()
    assert calls == []
    with pytest.raises(ValueError, match="differs"):
        ProductionAllEvidenceWorkflow(
            replace(
                options,
                run_control=replace(options.run_control, resume=True),
                model_name="substituted/model",
            ),
            phase_overrides=overrides,
        ).run()


def test_run_control_is_excluded_from_immutable_scientific_request(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)
    baseline = ProductionAllEvidenceWorkflow(options)._request_body()
    changed = replace(
        options,
        run_control=RunControl(
            resume=True,
            stop_after="handoff_validation",
            log_level="ERROR",
            validation_depth="fresh_terminal_audit",
        ),
    )
    observed = ProductionAllEvidenceWorkflow(changed)._request_body()

    assert observed == baseline
    assert "run_control" not in observed
    assert observed["scientific_identity"] == baseline["scientific_identity"]


def test_outer_folds_do_not_constrain_configured_inner_review_partitions(
    tmp_path,
):
    options = replace(
        _options(tmp_path),
        outer_folds=3,
        initial_training_partitions=2,
        review_rounds=2,
    )
    workflow = ProductionAllEvidenceWorkflow(options)
    assert workflow.options.outer_folds == 3
    assert workflow.options.initial_training_partitions + workflow.options.review_rounds == 4


def test_workflow_requires_and_identity_binds_complete_stage2_protocol(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)
    with pytest.raises(ValueError, match="stage2_prompt_protocol"):
        ProductionAllEvidenceWorkflow(replace(options, stage2_prompt_protocol=None))
    baseline = ProductionAllEvidenceWorkflow(options)._request_body()
    changed = ProductionAllEvidenceWorkflow(
        replace(
            options,
            stage2_prompt_protocol=replace(
                options.stage2_prompt_protocol,
                hierarchical_max_bytes_per_chunk=95_999,
            ),
        )
    )._request_body()
    assert (
        baseline["stage2_prompt_protocol"]
        == options.stage2_prompt_protocol.as_dict()
    )
    assert (
        baseline["scientific_identity"]["scientific_sha256"]
        != changed["scientific_identity"]["scientific_sha256"]
    )


def test_workflow_requires_and_identity_binds_causal_review_thresholds(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path)
    with pytest.raises(ValueError, match="post_extraction_causal_review"):
        ProductionAllEvidenceWorkflow(replace(options, post_extraction_causal_review=None))
    baseline = ProductionAllEvidenceWorkflow(options)._request_body()
    changed = ProductionAllEvidenceWorkflow(
        replace(
            options,
            post_extraction_causal_review=replace(
                options.post_extraction_causal_review,
                minimum_score_improvement=0.01,
            ),
        )
    )._request_body()
    assert (
        baseline["scientific_identity"]["scientific_sha256"]
        != changed["scientific_identity"]["scientific_sha256"]
    )


def test_stage2_tokenizer_content_is_required_and_path_neutral(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _options(first_root)
    second = _options(second_root)
    first_body = ProductionAllEvidenceWorkflow(first)._request_body()
    second_body = ProductionAllEvidenceWorkflow(second)._request_body()
    assert (
        first_body["scientific_identity"]["scientific_sha256"]
        == second_body["scientific_identity"]["scientific_sha256"]
    )
    with pytest.raises(ValueError, match="stage2_tokenizer_locator"):
        ProductionAllEvidenceWorkflow(replace(first, stage2_tokenizer_locator=None))
    tokenizer_file = second.stage2_tokenizer_locator / "tokenizer.json"
    tokenizer_file.write_text('{"fixture":"changed tokenizer"}', encoding="utf-8")
    changed_body = ProductionAllEvidenceWorkflow(second)._request_body()
    assert (
        first_body["scientific_identity"]["scientific_sha256"]
        != changed_body["scientific_identity"]["scientific_sha256"]
    )


def test_overall_scientific_identity_neutralizes_profile_execution_locators(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "profile-first"
    second_root = tmp_path / "profile-second"
    first_root.mkdir()
    second_root.mkdir()
    first = _options(first_root)
    second = _options(second_root)
    first_profile = {
        "config": {
            "dataset_path": "/first/cohort.parquet",
            "architecture": {
                "htr_sentence_model": "/first/model",
                "htr_chunk_size_words": 73,
                "multi_model_forest": {
                    "fold_parallelism": "1",
                    "embedding_contrast": {
                        "cache_dir": "/first/cache",
                        "device": "cuda:0",
                        "max_chunks": 10_003,
                    },
                },
            },
        }
    }
    second_profile = deepcopy(first_profile)
    second_profile["config"]["dataset_path"] = "/relocated/cohort.parquet"
    second_profile["config"]["architecture"]["htr_sentence_model"] = "/relocated/model"
    second_embedding = second_profile["config"]["architecture"]["multi_model_forest"][
        "embedding_contrast"
    ]
    second_embedding["cache_dir"] = "/relocated/cache"
    second_embedding["device"] = "cuda:7"
    second_profile["config"]["architecture"]["multi_model_forest"]["fold_parallelism"] = "auto"
    first.stage1_profile_path.write_text(
        json.dumps(first_profile),
        encoding="utf-8",
    )
    second.stage1_profile_path.write_text(
        json.dumps(second_profile),
        encoding="utf-8",
    )

    first_identity = ProductionAllEvidenceWorkflow(first)._request_body()["scientific_identity"][
        "scientific_sha256"
    ]
    second_identity = ProductionAllEvidenceWorkflow(second)._request_body()["scientific_identity"][
        "scientific_sha256"
    ]
    assert first_identity == second_identity

    second_profile["config"]["architecture"]["htr_chunk_size_words"] = 74
    second.stage1_profile_path.write_text(
        json.dumps(second_profile),
        encoding="utf-8",
    )
    changed_identity = ProductionAllEvidenceWorkflow(second)._request_body()["scientific_identity"][
        "scientific_sha256"
    ]
    assert changed_identity != first_identity


def test_phase_publication_rejects_hard_linked_payloads(tmp_path):
    options = _options(tmp_path)

    def linked_phase(attempt):
        source = attempt / "payload.bin"
        alias = attempt / "payload-alias.bin"
        source.write_bytes(b"one inode cannot represent two artifact entries")
        os.link(source, alias)
        return {"terminal_files": [str(source)]}

    overrides = {phase: (lambda _attempt: {"terminal_files": []}) for phase in PHASES}
    overrides["input_preparation"] = linked_phase
    with pytest.raises(ValueError, match="hard-linked"):
        ProductionAllEvidenceWorkflow(
            options,
            phase_overrides=overrides,
        ).run()


def test_completed_phases_publish_path_neutral_portable_checkpoint_dag(
    tmp_path: Path,
) -> None:
    oracle = tmp_path / "oracle.parquet"
    oracle.write_bytes(b"configured oracle bytes are not opened by phase overrides")
    options = replace(
        _options(tmp_path),
        evaluate_oracle_posthoc=True,
        oracle_dataset_path=oracle,
        oracle_unit_id_column="oracle_id",
        oracle_ite_column="oracle_ite",
    )
    calls: list[str] = []

    def phase_result(attempt: Path, phase: str) -> dict:
        calls.append(phase)
        payload = attempt / f"{phase}.json"
        payload.write_text(
            json.dumps({"phase": phase}, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "phase_marker": phase,
            "terminal_files": [str(payload.resolve())],
        }

    overrides = {
        phase: (lambda attempt, value=phase: phase_result(attempt, value)) for phase in PHASES
    }
    ProductionAllEvidenceWorkflow(
        options,
        phase_overrides=overrides,
    ).run()
    assert calls == list(PHASES)

    expected = {
        "input_preparation": ("prepared_cohort", ()),
        "embedding_cache": ("embedding_cache", ("input_preparation",)),
        "stage1_preflight": ("clustered_preflight", ("embedding_cache",)),
        "stage1_modeling": ("stage1_handoff", ("stage1_preflight",)),
        "stage2_canary": ("stage2_canary", ("stage1_modeling",)),
        "stage2_inference": (
            "frozen_prediction",
            ("stage1_modeling", "stage2_canary"),
        ),
        "oracle_evaluation": (
            "oracle_evaluation",
            ("stage2_inference",),
        ),
    }
    artifacts = {
        phase: validate_portable_artifact(options.work_root / "portable_checkpoints" / phase)
        for phase in expected
    }
    for phase, (kind, upstream_phases) in expected.items():
        artifact = artifacts[phase]
        assert artifact.manifest["artifact_kind"] == kind
        assert artifact.phase_binding["phase"] == phase
        assert artifact.manifest["upstream_artifact_ids"] == [
            artifacts[parent].artifact_id for parent in upstream_phases
        ]
        assert str(options.work_root) not in artifact.manifest_path.read_text(encoding="utf-8")
        attestation = json.loads(
            (
                options.work_root
                / "execution_attestations"
                / "portable_checkpoint_publications"
                / f"{phase}.json"
            ).read_text(encoding="utf-8")
        )
        assert attestation["artifact_id"] == artifact.artifact_id
        assert (
            attestation["producer_request_sha256"]
            == json.loads(
                (options.work_root / "immutable_run_request.json").read_text(encoding="utf-8")
            )["request_sha256"]
        )

    assert not (options.work_root / "portable_checkpoints" / "handoff_validation").exists()
    assert not (options.work_root / "portable_checkpoints" / "terminal_validation").exists()

    original_ids = {phase: artifact.artifact_id for phase, artifact in artifacts.items()}
    calls.clear()
    ProductionAllEvidenceWorkflow(
        _with_run_control(options, resume=True),
        phase_overrides=overrides,
    ).run()
    assert calls == []
    assert {
        phase: validate_portable_artifact(
            options.work_root / "portable_checkpoints" / phase
        ).artifact_id
        for phase in expected
    } == original_ids
    request = json.loads(
        (options.work_root / "immutable_run_request.json").read_text(encoding="utf-8")
    )
    fresh_validation = workflow_module.validate_published_workflow_checkpoint_dag(
        work_root=options.work_root,
        expected_request_sha256=request["request_sha256"],
        expected_phases=PHASES[:-1],
    )
    assert fresh_validation["status"] == "accepted"
    assert fresh_validation["fresh_full_byte_validation"] is True
    assert fresh_validation["oracle_evaluation_after_frozen_prediction"] is True

    attestation_path = (
        options.work_root
        / "execution_attestations"
        / "portable_checkpoint_publications"
        / "stage2_inference.json"
    )
    original_attestation = attestation_path.read_bytes()
    changed_attestation = json.loads(original_attestation)
    changed_attestation["artifact_id"] = "0" * 64
    attestation_path.chmod(0o644)
    attestation_path.write_text(
        json.dumps(changed_attestation),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="attestation changed"):
        workflow_module.validate_published_workflow_checkpoint_dag(
            work_root=options.work_root,
            expected_request_sha256=request["request_sha256"],
            expected_phases=PHASES[:-1],
        )
    attestation_path.write_bytes(original_attestation)
    attestation_path.chmod(0o444)


def test_oracle_phase_cannot_start_without_frozen_prediction_checkpoint(
    tmp_path: Path,
) -> None:
    oracle = tmp_path / "sealed_oracle.parquet"
    oracle.write_bytes(b"oracle must remain unopened")
    base = _options(tmp_path)
    options = replace(
        base,
        evaluate_oracle_posthoc=True,
        oracle_dataset_path=oracle,
        oracle_unit_id_column="oracle_id",
        oracle_ite_column="oracle_ite",
        run_control=replace(
            base.run_control,
            stop_after="oracle_evaluation",
        ),
    )
    oracle_phase_started: list[bool] = []

    def payload_phase(attempt: Path, phase: str) -> dict:
        payload = attempt / f"{phase}.bin"
        payload.write_bytes(phase.encode("utf-8"))
        return {"terminal_files": [str(payload.resolve())]}

    overrides = {
        phase: (lambda attempt, value=phase: payload_phase(attempt, value))
        for phase in PHASES
        if phase != "stage2_inference"
    }
    overrides["stage2_inference"] = lambda _attempt: {"terminal_files": []}

    def forbidden_oracle_phase(attempt: Path) -> dict:
        oracle_phase_started.append(True)
        return payload_phase(attempt, "oracle_evaluation")

    overrides["oracle_evaluation"] = forbidden_oracle_phase
    with pytest.raises(
        RuntimeError,
        match="required portable checkpoint is absent: stage2_inference",
    ):
        ProductionAllEvidenceWorkflow(
            options,
            phase_overrides=overrides,
        ).run()
    assert oracle_phase_started == []
    assert not (options.work_root / "portable_checkpoints" / "oracle_evaluation").exists()


def test_adopted_checkpoint_substitutes_phase_and_fresh_reader_reopens_bytes(
    tmp_path: Path,
) -> None:
    options = _with_run_control(
        _options(tmp_path),
        stop_after="input_preparation",
    )
    baseline = ProductionAllEvidenceWorkflow(options)._request_body()
    compatibility = ArtifactCompatibility(**baseline["expected_checkpoint_compatibility"])
    checkpoint = tmp_path / "prepared_checkpoint"
    cohort = checkpoint / "prepared" / "modeling_cohort.parquet"
    cohort.parent.mkdir(parents=True)
    cohort.write_bytes(b"portable prepared cohort")
    preparation_manifest = checkpoint / "prepared" / "preparation_manifest.json"
    preparation_manifest.write_text(
        '{"schema_version":"test_preparation_v1"}',
        encoding="utf-8",
    )
    artifact = publish_portable_artifact(
        root=checkpoint,
        artifact_kind="prepared_cohort",
        artifact_schema="prepared_workflow_phase_test_v1",
        compatibility=compatibility,
        upstream_artifact_ids=(),
        payload_paths=(
            "prepared/modeling_cohort.parquet",
            "prepared/preparation_manifest.json",
        ),
        workflow_phase="input_preparation",
        workflow_phase_result={
            "output": {"path": str(cohort.resolve())},
            "terminal_files": [
                str(cohort.resolve()),
                str(preparation_manifest.resolve()),
            ],
        },
    )
    calls: list[str] = []
    adopted_options = _with_run_control(
        options,
        adopt_checkpoints=(artifact.root,),
    )
    runner = ProductionAllEvidenceWorkflow(
        adopted_options,
        phase_overrides={
            "input_preparation": lambda _attempt: calls.append("computed") or {"terminal_files": []}
        },
    )
    result = runner.run()
    assert calls == []
    assert result["status"] == "paused"
    assert result["completed_phases"] == ["input_preparation"]

    phase = workflow_module._validate_phase_manifest_from_paths(
        work_root=adopted_options.work_root.resolve(strict=True),
        phase="input_preparation",
        request_sha256=result["request_sha256"],
    )
    assert phase["adopted_checkpoint"]["artifact_id"] == artifact.artifact_id
    assert phase["adopted_checkpoint"]["fresh_full_byte_validation"] is True
    assert Path(phase["result"]["output"]["path"]) == cohort.resolve()
    prepared, manifest = runner._input_preparation_paths()
    assert prepared == cohort.resolve()
    assert manifest == preparation_manifest.resolve()
    with pytest.raises(ValueError, match="differs from the immutable run request"):
        ProductionAllEvidenceWorkflow(
            replace(
                adopted_options,
                run_control=replace(
                    adopted_options.run_control,
                    resume=True,
                    adopt_checkpoints=(),
                ),
            ),
        ).run()

    attestation = (
        adopted_options.work_root / "checkpoint_adoptions" / f"{artifact.artifact_id}.adoption.json"
    )
    original_attestation = attestation.read_bytes()
    tampered_attestation = json.loads(original_attestation)
    tampered_attestation["consumer_request_sha256"] = "0" * 64
    attestation.chmod(0o644)
    attestation.write_text(
        json.dumps(tampered_attestation),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="attestation is invalid"):
        workflow_module._validate_phase_manifest_from_paths(
            work_root=adopted_options.work_root.resolve(strict=True),
            phase="input_preparation",
            request_sha256=result["request_sha256"],
        )
    attestation.write_bytes(original_attestation)
    attestation.chmod(0o444)

    cohort.write_bytes(b"tampered after adoption")
    with pytest.raises(ValueError, match="changed"):
        ProductionAllEvidenceWorkflow(
            _with_run_control(adopted_options, resume=True),
        ).run()


def test_phase_adoption_rejects_unbound_and_orphaned_checkpoints(
    tmp_path: Path,
) -> None:
    options = _with_run_control(
        _options(tmp_path),
        stop_after="input_preparation",
    )
    baseline = ProductionAllEvidenceWorkflow(options)._request_body()
    compatibility = ArtifactCompatibility(**baseline["expected_checkpoint_compatibility"])

    unbound_root = tmp_path / "unbound_prepared"
    unbound_root.mkdir()
    (unbound_root / "cohort.parquet").write_bytes(b"unbound cohort")
    unbound = publish_portable_artifact(
        root=unbound_root,
        artifact_kind="prepared_cohort",
        artifact_schema="unbound_prepared_test_v1",
        compatibility=compatibility,
        upstream_artifact_ids=(),
        payload_paths=("cohort.parquet",),
    )
    with pytest.raises(ValueError, match="lacks an authenticated phase binding"):
        ProductionAllEvidenceWorkflow(
            _with_run_control(
                options,
                adopt_checkpoints=(unbound.root,),
            )
        ).run()
    assert not options.work_root.exists()

    orphan_root = tmp_path / "orphan_cache"
    orphan_root.mkdir()
    cache_payload = orphan_root / "cache.bin"
    cache_payload.write_bytes(b"orphan cache")
    orphan = publish_portable_artifact(
        root=orphan_root,
        artifact_kind="embedding_cache",
        artifact_schema="orphan_cache_test_v1",
        compatibility=compatibility,
        upstream_artifact_ids=("a" * 64,),
        payload_paths=("cache.bin",),
        workflow_phase="embedding_cache",
        workflow_phase_result={
            "terminal_files": [str(cache_payload.resolve())],
        },
    )
    with pytest.raises(ValueError, match="upstream dependencies are absent"):
        ProductionAllEvidenceWorkflow(
            _with_run_control(
                options,
                adopt_checkpoints=(orphan.root,),
            )
        ).run()
    assert not options.work_root.exists()


def test_adopted_phase_dag_substitutes_dependency_prefix_once(
    tmp_path: Path,
) -> None:
    options = _with_run_control(
        _options(tmp_path),
        stop_after="embedding_cache",
    )
    baseline = ProductionAllEvidenceWorkflow(options)._request_body()
    compatibility = ArtifactCompatibility(**baseline["expected_checkpoint_compatibility"])

    prepared_root = tmp_path / "dag_prepared"
    cohort = prepared_root / "prepared" / "modeling_cohort.parquet"
    cohort.parent.mkdir(parents=True)
    cohort.write_bytes(b"prepared dependency")
    preparation_manifest = prepared_root / "prepared" / "preparation_manifest.json"
    preparation_manifest.write_text("{}", encoding="utf-8")
    prepared = publish_portable_artifact(
        root=prepared_root,
        artifact_kind="prepared_cohort",
        artifact_schema="dag_prepared_test_v1",
        compatibility=compatibility,
        upstream_artifact_ids=(),
        payload_paths=(
            "prepared/modeling_cohort.parquet",
            "prepared/preparation_manifest.json",
        ),
        workflow_phase="input_preparation",
        workflow_phase_result={
            "output": {"path": str(cohort.resolve())},
            "terminal_files": [
                str(cohort.resolve()),
                str(preparation_manifest.resolve()),
            ],
        },
    )

    cache_root = tmp_path / "dag_cache"
    cache_dir = cache_root / "embedding_cache"
    cache_dir.mkdir(parents=True)
    cache_metadata = cache_dir / "metadata.json"
    cache_metadata.write_text("{}", encoding="utf-8")
    cache_cohort = cache_root / "prepared" / "modeling_cohort.parquet"
    cache_cohort.parent.mkdir(parents=True)
    cache_cohort.write_bytes(b"cache-bound cohort")
    cache = publish_portable_artifact(
        root=cache_root,
        artifact_kind="embedding_cache",
        artifact_schema="dag_cache_test_v1",
        compatibility=compatibility,
        upstream_artifact_ids=(prepared.artifact_id,),
        payload_paths=(
            "embedding_cache/metadata.json",
            "prepared/modeling_cohort.parquet",
        ),
        workflow_phase="embedding_cache",
        workflow_phase_result={
            "schema_version": EMBEDDING_CACHE_PHASE_SCHEMA,
            "mode": "fresh_build",
            "cache_path": str(cache_dir.resolve()),
            "prepared_cohort_path": str(cache_cohort.resolve()),
            "cache_identity": {"test": "authenticated"},
            "terminal_files": [
                str(cache_metadata.resolve()),
                str(cache_cohort.resolve()),
            ],
        },
    )
    calls: list[str] = []
    runner = ProductionAllEvidenceWorkflow(
        _with_run_control(
            options,
            # CLI ordering is operational; the authenticated DAG supplies order.
            adopt_checkpoints=(cache.root, prepared.root),
        ),
        phase_overrides={
            phase: (lambda _attempt, value=phase: calls.append(value) or {"terminal_files": []})
            for phase in ("input_preparation", "embedding_cache")
        },
    )
    result = runner.run()
    assert result["completed_phases"] == [
        "input_preparation",
        "embedding_cache",
    ]
    assert calls == []
    observed_cache, observed_prepared = runner._embedding_cache_paths()
    assert observed_cache == cache_dir.resolve()
    assert observed_prepared == cache_cohort.resolve()


def _compact_preflight_adoption_prefix(
    *,
    tmp_path: Path,
    options: ProductionAllEvidenceWorkflowOptions,
    parquet_compression: str,
):
    baseline = ProductionAllEvidenceWorkflow(options)._request_body()
    compatibility = ArtifactCompatibility(
        **baseline["expected_checkpoint_compatibility"]
    )

    prepared_root = tmp_path / f"codec-{parquet_compression}-prepared"
    prepared_payload = prepared_root / "prepared.parquet"
    prepared_root.mkdir()
    prepared_payload.write_bytes(b"prepared")
    prepared = publish_portable_artifact(
        root=prepared_root,
        artifact_kind="prepared_cohort",
        artifact_schema="production_prepared_cohort_checkpoint_v1",
        compatibility=compatibility,
        upstream_artifact_ids=(),
        payload_paths=("prepared.parquet",),
        workflow_phase="input_preparation",
        workflow_phase_result={
            "terminal_files": [str(prepared_payload.resolve())],
        },
    )

    cache_root = tmp_path / f"codec-{parquet_compression}-cache"
    cache_payload = cache_root / "cache.bin"
    cache_root.mkdir()
    cache_payload.write_bytes(b"cache")
    cache = publish_portable_artifact(
        root=cache_root,
        artifact_kind="embedding_cache",
        artifact_schema="production_embedding_cache_checkpoint_v1",
        compatibility=compatibility,
        upstream_artifact_ids=(prepared.artifact_id,),
        payload_paths=("cache.bin",),
        workflow_phase="embedding_cache",
        workflow_phase_result={
            "terminal_files": [str(cache_payload.resolve())],
        },
    )

    preflight_root = tmp_path / f"codec-{parquet_compression}-preflight"
    compact_manifest = (
        preflight_root
        / "cluster_preflight"
        / "cluster_preflight_manifest.json"
    )
    compact_manifest.parent.mkdir(parents=True)
    physical_storage = {
        "owner_concept_payload_format": "parquet",
        "parquet_compression": parquet_compression,
        "parquet_use_dictionary": False,
        "parquet_write_statistics": False,
        "parquet_data_page_version": "1.0",
    }
    compact_body = {
        "schema_version": (
            "production_stage1_cluster_preflight_manifest_v2"
        ),
        "status": "complete",
        "physical_storage": physical_storage,
    }
    compact_manifest.write_text(
        json.dumps(
            {
                **compact_body,
                "content_sha256": workflow_module.identity_sha256(
                    compact_body
                ),
            }
        ),
        encoding="utf-8",
    )
    profile = preflight_root / "effective_stage1_profile.json"
    profile.write_text("{}", encoding="utf-8")
    identity_body = {
        "schema_version": (
            "production_stage1_cluster_preflight_result_v2"
        ),
        "physical_storage": physical_storage,
    }
    preflight = publish_portable_artifact(
        root=preflight_root,
        artifact_kind="clustered_preflight",
        artifact_schema="production_clustered_preflight_checkpoint_v1",
        compatibility=compatibility,
        upstream_artifact_ids=(cache.artifact_id,),
        payload_paths=(
            "cluster_preflight/cluster_preflight_manifest.json",
            "effective_stage1_profile.json",
        ),
        workflow_phase="stage1_preflight",
        workflow_phase_result={
            "schema_version": STAGE1_PREFLIGHT_PHASE_SCHEMA,
            "scientific_cluster_preflight": (
                "accepted_portable_compact_lossless_v2"
            ),
            "cluster_preflight_manifest_path": str(
                compact_manifest.resolve()
            ),
            "effective_profile_path": str(profile.resolve()),
            "cluster_preflight_identity": {
                **identity_body,
                "content_sha256": workflow_module.identity_sha256(
                    identity_body
                ),
            },
            "terminal_files": [
                str(compact_manifest.resolve()),
                str(profile.resolve()),
            ],
        },
    )
    return prepared, cache, preflight


@pytest.mark.parametrize(
    ("produced_codec", "requested_codec", "accepted"),
    (
        ("zstd", "none", False),
        ("none", "zstd", False),
        ("zstd", "zstd", True),
        ("none", "none", True),
    ),
)
def test_public_adopted_compact_preflight_requires_deployment_codec(
    tmp_path: Path,
    produced_codec: str,
    requested_codec: str,
    accepted: bool,
) -> None:
    options = replace(
        _options(tmp_path),
        cluster_preflight_parquet_compression=requested_codec,
    )
    options = _with_run_control(
        options,
        stop_after="stage1_preflight",
    )
    prepared, cache, preflight = _compact_preflight_adoption_prefix(
        tmp_path=tmp_path,
        options=options,
        parquet_compression=produced_codec,
    )
    adopted = _with_run_control(
        options,
        adopt_checkpoints=(
            preflight.root,
            cache.root,
            prepared.root,
        ),
    )
    runner = ProductionAllEvidenceWorkflow(adopted)
    if not accepted:
        with pytest.raises(
            ValueError,
            match=(
                "adopted compact Stage 1 preflight Parquet compression "
                ".* differs from requested deployment compression"
            ),
        ):
            runner.run()
        assert not options.work_root.exists()
        return

    result = runner.run()
    assert result["status"] == "paused"
    assert result["completed_phases"] == [
        "input_preparation",
        "embedding_cache",
        "stage1_preflight",
    ]
    freshly_reopened = workflow_module._validate_phase_manifest_from_paths(
        work_root=options.work_root.resolve(strict=True),
        phase="stage1_preflight",
        request_sha256=result["request_sha256"],
    )
    assert (
        freshly_reopened["result"]["cluster_preflight_identity"][
            "physical_storage"
        ]["parquet_compression"]
        == requested_codec
    )


def test_preflight_parquet_codec_is_excluded_from_scientific_identity(
    tmp_path: Path,
) -> None:
    zstd = _options(tmp_path)
    none = replace(
        zstd,
        cluster_preflight_parquet_compression="none",
    )
    zstd_request = ProductionAllEvidenceWorkflow(zstd)._request_body()
    none_request = ProductionAllEvidenceWorkflow(none)._request_body()

    assert (
        zstd_request["scientific_identity"]
        == none_request["scientific_identity"]
    )
    assert (
        zstd_request["expected_checkpoint_compatibility"]
        == none_request["expected_checkpoint_compatibility"]
    )
    assert zstd_request["cluster_preflight_parquet_compression"] == "zstd"
    assert none_request["cluster_preflight_parquet_compression"] == "none"


def test_terminal_validator_accepts_only_freshly_reopened_adopted_phase(
    tmp_path: Path,
) -> None:
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage1_only=True,
    )
    baseline = ProductionAllEvidenceWorkflow(options)._request_body()
    compatibility = ArtifactCompatibility(**baseline["expected_checkpoint_compatibility"])
    checkpoint = tmp_path / "terminal_adopted_prepared"
    cohort = checkpoint / "prepared" / "modeling_cohort.parquet"
    cohort.parent.mkdir(parents=True)
    cohort.write_bytes(b"terminal validator adopted cohort")
    preparation_manifest = checkpoint / "prepared" / "preparation_manifest.json"
    preparation_manifest.write_text("{}", encoding="utf-8")
    artifact = publish_portable_artifact(
        root=checkpoint,
        artifact_kind="prepared_cohort",
        artifact_schema="terminal_adopted_prepared_test_v1",
        compatibility=compatibility,
        upstream_artifact_ids=(),
        payload_paths=(
            "prepared/modeling_cohort.parquet",
            "prepared/preparation_manifest.json",
        ),
        workflow_phase="input_preparation",
        workflow_phase_result={
            "output": {"path": str(cohort.resolve())},
            "terminal_files": [
                str(cohort.resolve()),
                str(preparation_manifest.resolve()),
            ],
        },
    )
    calls: list[str] = []
    overrides = {
        phase: (lambda _attempt, value=phase: calls.append(value) or {"terminal_files": []})
        for phase in STAGE1_ONLY_PHASES[:-1]
    }
    result = ProductionAllEvidenceWorkflow(
        _with_run_control(
            options,
            adopt_checkpoints=(artifact.root,),
        ),
        phase_overrides=overrides,
    ).run()
    assert calls == [
        "embedding_cache",
        "stage1_preflight",
        "stage1_modeling",
        "handoff_validation",
    ]
    assert result["validated_phase_sequence"] == list(STAGE1_ONLY_PHASES)


def test_phase_uses_configured_scratch_then_publishes_durable_locators(tmp_path):
    scratch_root = tmp_path / "operator-scratch"
    options = replace(_options(tmp_path), scratch_root=scratch_root)
    original = {}

    def input_phase(attempt):
        payload = attempt / "nested" / "payload.bin"
        payload.parent.mkdir()
        payload.write_bytes(b"terminal payload")
        original["attempt"] = attempt
        original["payload"] = payload
        return {
            "primary_path": str(payload),
            "nested": {"attempt_path": str(attempt)},
            "terminal_files": [str(payload)],
        }

    overrides = {phase: (lambda _attempt: {"terminal_files": []}) for phase in PHASES}
    overrides["input_preparation"] = input_phase
    ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()

    manifest = json.loads(
        (options.work_root / "phases" / "input_preparation" / "complete_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    published = Path(manifest["attempt_dir"])
    assert original["attempt"].is_relative_to(scratch_root)
    assert not original["attempt"].exists()
    assert published.parent == (options.work_root / "phases" / "input_preparation").resolve()
    assert manifest["result"]["primary_path"] == str(published / "nested" / "payload.bin")
    assert manifest["result"]["nested"]["attempt_path"] == str(published)
    assert manifest["result"]["terminal_files"] == [str(published / "nested" / "payload.bin")]


def test_direct_only_scientific_cli_fails_before_work_root_creation(tmp_path):
    o = _options(tmp_path)
    args = build_parser().parse_args(
        [
            "--dataset",
            str(o.dataset_path),
            "--work-root",
            str(o.work_root),
            "--stage1-profile",
            str(o.stage1_profile_path),
            "--query-profile",
            str(o.query_profile_path),
            "--unit-id-column",
            "id",
            "--text-column",
            "note",
            "--treatment-column",
            "tx",
            "--outcome-column",
            "y",
            "--outcome-type",
            "binary",
            "--clinical-question",
            "q",
            "--embedding-model-name",
            "embed",
            "--embedding-local-model-path",
            str(o.embedding_local_model_path),
            "--htr-local-model-path",
            str(o.htr_local_model_path),
            *_resource_performance_safety_cli_args(o.resource_performance_safety),
            "--stage2-tokenizer-locator",
            str(o.stage2_tokenizer_locator),
            "--stage1-device",
            "cuda:0",
            "--review-device",
            "cuda:0",
            "--max-candidate-variables",
            "7",
            "--stage2-prompt-protocol",
            str(tmp_path / "stage2_protocol.json"),
            "--post-extraction-causal-review",
            str(tmp_path / "causal_review.json"),
            "--complete-page-core-chars",
            "97",
            "--complete-page-context-chars",
            "11",
            "--complete-page-max-chars",
            "119",
            "--complete-reconciliation-fan-in",
            "7",
            "--embedding-chunk-size-words",
            "31",
            "--embedding-chunk-overlap-words",
            "7",
            "--embedding-max-chunks",
            "4096",
            "--embedding-chunk-selection",
            "last",
            "--embedding-max-seq-length",
            "512",
            "--embedding-batch-size",
            "13",
            "--embedding-normalize",
            "--forest-n-estimators",
            "40",
            "--forest-max-depth",
            "7",
            "--forest-min-samples-leaf",
            "4",
            "--forest-max-features",
            "sqrt",
            "--forest-honest",
            "--forest-inference",
            "--forest-subforest-size",
            "4",
            "--no-forest-tune-model",
            "--forest-nuisance-n-estimators",
            "31",
            "--forest-nuisance-max-depth",
            "5",
            "--forest-nuisance-min-samples-leaf",
            "3",
            "--forest-nuisance-treatment-max-features",
            "0.75",
            "--forest-nuisance-outcome-max-features",
            "sqrt",
            "--forest-random-seed",
            "19",
            "--outer-folds",
            "5",
            "--review-rounds",
            "2",
            "--initial-training-partitions",
            "3",
            "--interaction-inner-folds",
            "3",
            "--tfidf-nested-calibration-folds",
            "3",
            "--seed",
            "42",
            "--empty-text-policy",
            "marker",
            "--repeated-character-policy",
            "marker",
            "--repeated-character-threshold",
            "1000",
            "--source-text-temporally-valid-by-design",
            "--endpoint",
            "https://fake.example/v1",
            "--model",
            "fake/model",
        ]
    )
    with pytest.raises(ValueError, match="--scientific-spec is required"):
        options_from_args(args)
    assert not o.work_root.exists()


def test_scientific_spec_and_complete_direct_deployment_compile_typed_options(
    tmp_path: Path,
) -> None:
    configured = _options(tmp_path)
    scientific_path = _write_scientific_spec(tmp_path / "scientific.json")
    args = build_parser().parse_args(
        _direct_deployment_args(
            options=configured,
            scientific_spec=scientific_path,
            scratch_root=tmp_path / "direct-scratch",
        )
        + [
            "--resume",
            "--stop-after",
            "handoff_validation",
            "--log-level",
            "WARNING",
            "--validation-depth",
            "full",
        ]
    )

    parsed = options_from_args(args)

    assert parsed.portable_scientific_spec is not None
    assert parsed.deployment_profile_path is None
    assert parsed.scientific_spec_path == scientific_path.resolve()
    assert parsed.embedding_model_name == configured.embedding_model_name
    assert parsed.device_policy == ("cpu",)
    assert parsed.cpu_budget == 2
    assert parsed.response_concurrency == 3
    assert parsed.forest_runtime_config is not None
    assert (
        parsed.forest_runtime_config.causal_forest.as_dict()
        == _scientific_spec().causal_estimator.as_dict()
    )
    assert parsed.forest_runtime_config.operational.requested_host_cpu_budget == parsed.cpu_budget
    assert all(
        getattr(parsed, name) is None
        for name in (
            "forest_n_estimators",
            "forest_max_depth",
            "forest_min_samples_leaf",
            "forest_max_features",
            "forest_honest",
            "forest_inference",
            "forest_subforest_size",
            "forest_tune_model",
            "forest_nuisance_n_estimators",
            "forest_nuisance_max_depth",
            "forest_nuisance_min_samples_leaf",
            "forest_nuisance_treatment_max_features",
            "forest_nuisance_outcome_max_features",
            "forest_random_seed",
        )
    )
    assert parsed.run_control == RunControl(
        resume=True,
        stop_after="handoff_validation",
        log_level="WARNING",
        validation_depth="full",
    )
    assert (
        workflow_module._default_portable_role_neutral_hooks(parsed).role_neutral_stage1 is not None
    )
    assert not configured.work_root.exists()


@pytest.mark.parametrize(
    ("flag", "message"),
    (
        ("--cpu-budget", "positive"),
        ("--response-concurrency", "positive"),
        ("--embedding-batch-size", "positive"),
    ),
)
def test_direct_deployment_zero_does_not_fall_back(
    tmp_path: Path,
    flag: str,
    message: str,
) -> None:
    configured = _options(tmp_path)
    scientific_path = _write_scientific_spec(tmp_path / "scientific.json")
    argv = _direct_deployment_args(
        options=configured,
        scientific_spec=scientific_path,
        scratch_root=tmp_path / "direct-scratch",
    )
    argv.extend([flag, "0"])
    with pytest.raises(ValueError, match=message):
        options_from_args(build_parser().parse_args(argv))
    assert not configured.work_root.exists()


@pytest.mark.parametrize(
    ("extra", "field"),
    (
        (["--endpoint", "https://other.invalid/v1"], "endpoint"),
        (["--model", "other/model"], "model"),
        (["--stage2-tokenizer-locator", "other-tokenizer"], "stage2_tokenizer"),
        (["--oracle-dataset", "oracle.parquet"], "oracle_dataset"),
        (["--devices", "cpu"], "devices"),
        (["--stage1-device", "cpu"], "stage1_device"),
    ),
)
def test_typed_deployment_rejects_direct_aliases(
    tmp_path: Path,
    extra: list[str],
    field: str,
) -> None:
    configured = _options(tmp_path)
    scientific_path = _write_scientific_spec(tmp_path / "scientific.json")
    deployment_path = _write_deployment_profile(
        tmp_path / "deployment.json",
        configured,
    )
    argv = [
        "--scientific-spec",
        str(scientific_path),
        "--deployment-profile",
        str(deployment_path),
        *extra,
    ]
    with pytest.raises(ValueError, match=field):
        options_from_args(build_parser().parse_args(argv))
    assert not configured.work_root.exists()


def test_typed_pair_accepts_only_run_control_overrides(
    tmp_path: Path,
) -> None:
    configured = _options(tmp_path)
    scientific_path = _write_scientific_spec(tmp_path / "scientific.json")
    deployment_path = _write_deployment_profile(
        tmp_path / "deployment.json",
        configured,
    )
    checkpoint = tmp_path / "future-checkpoint"
    parsed = options_from_args(
        build_parser().parse_args(
            [
                "--scientific-spec",
                str(scientific_path),
                "--deployment-profile",
                str(deployment_path),
                "--resume",
                "--stop-after",
                "handoff_validation",
                "--adopt-checkpoint",
                str(checkpoint),
                "--log-level",
                "ERROR",
                "--validation-depth",
                "fresh_terminal_audit",
            ]
        )
    )
    assert parsed.run_control == RunControl(
        resume=True,
        stop_after="handoff_validation",
        adopt_checkpoints=(checkpoint,),
        log_level="ERROR",
        validation_depth="fresh_terminal_audit",
    )
    assert parsed.portable_scientific_spec is not None
    assert not configured.work_root.exists()


def test_direct_deployment_rejects_partial_endpoint_group(
    tmp_path: Path,
) -> None:
    configured = _options(tmp_path)
    scientific_path = _write_scientific_spec(tmp_path / "scientific.json")
    argv = _direct_deployment_args(
        options=configured,
        scientific_spec=scientific_path,
        scratch_root=tmp_path / "direct-scratch",
    )
    model_index = argv.index("--model")
    del argv[model_index : model_index + 2]
    with pytest.raises(ValueError, match="endpoint, model, and"):
        options_from_args(build_parser().parse_args(argv))
    assert not configured.work_root.exists()

    oracle_argv = _direct_deployment_args(
        options=configured,
        scientific_spec=scientific_path,
        scratch_root=tmp_path / "direct-scratch",
    )
    oracle_argv.extend(["--oracle-dataset", str(tmp_path / "oracle.parquet")])
    with pytest.raises(ValueError, match="oracle dataset, unit-ID"):
        options_from_args(build_parser().parse_args(oracle_argv))
    assert not configured.work_root.exists()


def test_scientific_spec_rejects_direct_scientific_overrides(
    tmp_path: Path,
) -> None:
    configured = _options(tmp_path)
    scientific_path = _write_scientific_spec(tmp_path / "scientific.json")
    argv = _direct_deployment_args(
        options=configured,
        scientific_spec=scientific_path,
        scratch_root=tmp_path / "direct-scratch",
    )
    argv.extend(["--complete-page-core-chars", "1"])
    with pytest.raises(
        ValueError,
        match="direct scientific shims.*complete_page_core_chars",
    ):
        options_from_args(build_parser().parse_args(argv))
    assert not configured.work_root.exists()


def test_public_compiler_never_derives_model_name_from_locator(
    tmp_path: Path,
) -> None:
    configured = _options(tmp_path)
    scientific_path = _write_scientific_spec(tmp_path / "scientific.json")
    deployment_path = _write_deployment_profile(
        tmp_path / "deployment.json",
        configured,
    )
    payload = json.loads(deployment_path.read_text(encoding="utf-8"))
    payload["embedding_model_name"] = None
    deployment_path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="explicitly configure embedding_model_name",
    ):
        options_from_args(
            build_parser().parse_args(
                [
                    "--scientific-spec",
                    str(scientific_path),
                    "--deployment-profile",
                    str(deployment_path),
                ]
            )
        )
    assert not configured.work_root.exists()


def test_public_cli_has_no_embedded_device_or_seed_policy_default() -> None:
    parser = build_parser()
    parsed = vars(parser.parse_args([]))
    assert "stage1_device" not in parsed
    assert "review_device" not in parsed
    assert "devices" not in parsed
    assert "stage1_seed_policy" not in parsed


def test_stage1_only_needs_no_endpoint_and_stops_after_fresh_handoff_boundary(
    tmp_path,
    monkeypatch,
):
    stage2_module = "oci.inference.production_stage1_hierarchy_one_shot"
    canary_module = "scripts.canary_production_stage1_hierarchy"
    stage2_was_loaded = stage2_module in sys.modules
    canary_was_loaded = canary_module in sys.modules
    openai_was_loaded = "openai" in sys.modules
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage2_tokenizer_locator=None,
        stage1_only=True,
    )
    calls = []
    overrides = {
        phase: (lambda _attempt, value=phase: calls.append(value) or {"terminal_files": []})
        for phase in STAGE1_ONLY_PHASES[:-1]
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Stage 2 construction is forbidden in Stage-1-only mode")

    monkeypatch.setattr(ProductionAllEvidenceWorkflow, "_stage2_options", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    result = ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()

    assert calls == list(STAGE1_ONLY_PHASES[:-1])
    assert result["stage1_only"] is True
    # The override did not perform the real loader subprocess, so the terminal
    # validator must not claim that it did.
    assert result["stage1_handoff_validated_in_fresh_process"] is False
    assert result["validated_phase_sequence"] == list(STAGE1_ONLY_PHASES)
    request = json.loads(
        (options.work_root / "immutable_run_request.json").read_text(encoding="utf-8")
    )
    assert request["endpoint"] is None
    assert request["model_name"] is None
    assert request["phase_sequence"] == list(STAGE1_ONLY_PHASES)
    assert (stage2_module in sys.modules) is stage2_was_loaded
    assert (canary_module in sys.modules) is canary_was_loaded
    assert ("openai" in sys.modules) is openai_was_loaded


def test_canary_descriptor_preparation_is_an_operational_prefix_only(
    tmp_path,
    monkeypatch,
):
    loaded_root = Path(workflow_module.__file__).resolve().parents[2]
    snapshot_sha = "c" * 64
    snapshot_identity = {
        "root": str(loaded_root),
        "manifest_path": str(loaded_root / "source_snapshot_manifest.json"),
        "content_sha256": snapshot_sha,
        "file_count": 1,
    }
    fake_snapshot = SimpleNamespace(
        root=loaded_root,
        content_sha256=snapshot_sha,
        as_dict=lambda: dict(snapshot_identity),
    )
    monkeypatch.setattr(
        "oci.inference.production_source_snapshot.validate_production_source_snapshot",
        lambda _path: fake_snapshot,
    )
    monkeypatch.setenv(
        workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV,
        snapshot_sha,
    )
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage1_only=True,
        gpu_id=None,
        stage1_gpu_ids=(0, 1),
        source_snapshot_root=loaded_root,
    )
    calls = []
    prefix = ("input_preparation", "embedding_cache", "stage1_preflight")
    overrides = {
        phase: (lambda _attempt, value=phase: calls.append(value) or {"terminal_files": []})
        for phase in prefix
    }
    cache = (tmp_path / "prepared_cache").resolve()
    cache.mkdir()
    prepared_path = (tmp_path / "prepared.parquet").resolve()
    prepared_path.write_bytes(b"prepared")
    profile = (tmp_path / "effective_profile.json").resolve()
    profile.write_text("{}", encoding="utf-8")
    preflight = (tmp_path / "cluster_preflight_manifest.json").resolve()
    preflight.write_text("{}", encoding="utf-8")
    descriptor_root = (options.work_root / "recovery" / "descriptor").resolve()
    prepared = SimpleNamespace(
        request_sha256="a" * 64,
        scope_descriptor_root=descriptor_root,
    )

    class _FakeBuilder:
        def __init__(self, _options):
            pass

        def prepare(self):
            return prepared

    selected_root = descriptor_root / "outer_001_full"
    selected_manifest = selected_root / "descriptor_manifest.json"

    def publish(*, prepared, descriptor_root):
        del prepared
        descriptor_root.mkdir(parents=True)
        selected_root.mkdir()
        selected_manifest.write_text("selected", encoding="utf-8")
        set_manifest = descriptor_root / "descriptor_set_manifest.json"
        set_manifest.write_text("set", encoding="utf-8")
        selected = SimpleNamespace(
            scope_id="outer_001_full",
            scope=SimpleNamespace(scope_kind="full_outer"),
            assignment=SimpleNamespace(gpu_id=0),
            manifest_path=selected_manifest,
        )
        return SimpleNamespace(
            root=descriptor_root,
            manifest={"content_sha256": "b" * 64},
            descriptors={"outer_001_full": selected},
        )

    monkeypatch.setattr(
        workflow_module,
        "ProductionStage1BundleBuilder",
        _FakeBuilder,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_legacy_scope_adapter."
        "publish_legacy_stage1_scope_descriptor",
        publish,
    )
    monkeypatch.setattr(
        "oci.inference.production_stage1_legacy_scope_adapter."
        "validate_legacy_stage1_scope_descriptor_set",
        lambda **kwargs: (
            publish(
                prepared=prepared,
                descriptor_root=Path(kwargs["descriptor_root"]),
            )
            if not selected_manifest.exists()
            else SimpleNamespace(
                root=descriptor_root,
                manifest={"content_sha256": "b" * 64},
                descriptors={
                    "outer_001_full": SimpleNamespace(
                        scope_id="outer_001_full",
                        scope=SimpleNamespace(scope_kind="full_outer"),
                        assignment=SimpleNamespace(gpu_id=0),
                        manifest_path=selected_manifest,
                    )
                },
            )
        ),
    )
    workflow = ProductionAllEvidenceWorkflow(
        options,
        phase_overrides=overrides,
    )
    monkeypatch.setattr(
        workflow,
        "_embedding_cache_paths",
        lambda: (cache, prepared_path),
    )
    monkeypatch.setattr(
        workflow,
        "_stage1_preflight_paths",
        lambda: (profile, preflight),
    )
    monkeypatch.setattr(
        workflow,
        "_stage1_build_options",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        workflow,
        "_validate_canary_preparation_in_fresh_process",
        lambda: json.loads(
            (
                options.work_root / "recovery" / "canary_descriptor_preparation_manifest.json"
            ).read_text(encoding="utf-8")
        ),
    )

    result = workflow.prepare_stage1_canary_descriptors_only()

    assert calls == list(prefix)
    assert result["status"] == "complete"
    assert result["selected_scope_id"] == "outer_001_full"
    assert result["selected_logical_gpu_id"] == 0
    assert result["supervised_stage1_fits_started"] is False
    assert not (options.work_root / "phases" / "stage1_modeling").exists()
    request_before = (options.work_root / "immutable_run_request.json").read_bytes()
    resumed = ProductionAllEvidenceWorkflow(
        _with_run_control(options, resume=True),
        phase_overrides=overrides,
    )
    resumed._initialize()
    assert (options.work_root / "immutable_run_request.json").read_bytes() == request_before


def test_canary_descriptor_preparation_requires_source_snapshot(tmp_path):
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage1_only=True,
        gpu_id=None,
        stage1_gpu_ids=(0, 1),
    )
    workflow = ProductionAllEvidenceWorkflow(options)

    with pytest.raises(ValueError, match="requires one authenticated source snapshot"):
        workflow.prepare_stage1_canary_descriptors_only()

    assert not options.work_root.exists()


def test_full_workflow_still_requires_endpoint_and_model(tmp_path):
    with pytest.raises(ValueError, match="requires one endpoint"):
        ProductionAllEvidenceWorkflow(replace(_options(tmp_path), endpoint=None, model_name=None))


def test_legacy_plural_gpu_cli_cannot_select_public_stage1(tmp_path):
    o = _options(tmp_path)
    args = build_parser().parse_args(
        [
            "--dataset",
            str(o.dataset_path),
            "--work-root",
            str(o.work_root),
            "--stage1-profile",
            str(o.stage1_profile_path),
            "--query-profile",
            str(o.query_profile_path),
            "--unit-id-column",
            "id",
            "--text-column",
            "note",
            "--treatment-column",
            "tx",
            "--outcome-column",
            "y",
            "--outcome-type",
            "binary",
            "--clinical-question",
            "q",
            "--embedding-model-name",
            "embed",
            "--embedding-local-model-path",
            str(o.embedding_local_model_path),
            "--htr-local-model-path",
            str(o.htr_local_model_path),
            *_resource_performance_safety_cli_args(o.resource_performance_safety),
            "--stage1-device",
            "cuda:0",
            "--review-device",
            "cuda:0",
            "--max-candidate-variables",
            "7",
            "--stage2-prompt-protocol",
            str(tmp_path / "stage2_protocol.json"),
            "--post-extraction-causal-review",
            str(tmp_path / "causal_review.json"),
            "--complete-page-core-chars",
            "97",
            "--complete-page-context-chars",
            "11",
            "--complete-page-max-chars",
            "119",
            "--complete-reconciliation-fan-in",
            "7",
            "--embedding-chunk-size-words",
            "31",
            "--embedding-chunk-overlap-words",
            "7",
            "--embedding-max-chunks",
            "4096",
            "--embedding-chunk-selection",
            "last",
            "--embedding-max-seq-length",
            "512",
            "--embedding-batch-size",
            "13",
            "--embedding-normalize",
            "--forest-n-estimators",
            "40",
            "--forest-max-depth",
            "7",
            "--forest-min-samples-leaf",
            "4",
            "--forest-max-features",
            "sqrt",
            "--forest-honest",
            "--forest-inference",
            "--forest-subforest-size",
            "4",
            "--no-forest-tune-model",
            "--forest-nuisance-n-estimators",
            "31",
            "--forest-nuisance-max-depth",
            "5",
            "--forest-nuisance-min-samples-leaf",
            "3",
            "--forest-nuisance-treatment-max-features",
            "0.75",
            "--forest-nuisance-outcome-max-features",
            "sqrt",
            "--forest-random-seed",
            "19",
            "--outer-folds",
            "5",
            "--review-rounds",
            "2",
            "--initial-training-partitions",
            "3",
            "--interaction-inner-folds",
            "3",
            "--tfidf-nested-calibration-folds",
            "3",
            "--seed",
            "42",
            "--empty-text-policy",
            "marker",
            "--repeated-character-policy",
            "marker",
            "--repeated-character-threshold",
            "1000",
            "--source-text-temporally-valid-by-design",
            "--stage1-only",
            "--stage1-gpu-id",
            "0",
            "--stage1-gpu-id",
            "1",
            "--query-device",
            "cuda:0",
            "--query-device",
            "cuda:1",
            "--stage1-scope-workers-per-gpu",
            "1",
            "--stage1-preflight-workers",
            "8",
        ]
    )
    with pytest.raises(ValueError, match="--scientific-spec is required"):
        options_from_args(args)
    assert not o.work_root.exists()


def test_singular_gpu_alias_is_accepted_but_conflicts_are_rejected(tmp_path):
    options = _options(tmp_path)
    gpu_options = replace(
        options,
        stage1_device="cuda:0",
        query_device="cuda:0",
        review_device="cuda:0",
        gpu_id=0,
    )
    assert ProductionAllEvidenceWorkflow(gpu_options).stage1_gpu_ids == (0,)
    with pytest.raises(ValueError, match="conflicts"):
        ProductionAllEvidenceWorkflow(
            replace(
                gpu_options,
                gpu_id=1,
                stage1_gpu_ids=(0, 1),
            )
        )


def test_gpu_preflight_checks_every_requested_gpu(monkeypatch, tmp_path):
    workflow = ProductionAllEvidenceWorkflow(
        replace(
            _options(tmp_path),
            gpu_id=None,
            stage1_gpu_ids=(0, 1),
            stage1_device="cuda:1",
        )
    )

    def fake_run(command, **_kwargs):
        if any(value.startswith("--query-gpu=") for value in command):
            return SimpleNamespace(stdout="0, GPU-a, 49140, 15, 0\n1, GPU-b, 49140, 188, 0\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(
        "oci.inference.production_all_evidence_workflow.subprocess.run",
        fake_run,
    )
    report = workflow._gpu_preflight()
    assert report["requested_gpu_ids"] == [0, 1]
    assert report["gpu_uuids"] == {"0": "GPU-a", "1": "GPU-b"}


def test_gpu_preflight_rejects_occupancy_on_either_gpu(monkeypatch, tmp_path):
    workflow = ProductionAllEvidenceWorkflow(
        replace(
            _options(tmp_path),
            gpu_id=None,
            stage1_gpu_ids=(0, 1),
            stage1_device="cuda:1",
        )
    )

    def fake_run(command, **_kwargs):
        if any(value.startswith("--query-gpu=") for value in command):
            return SimpleNamespace(stdout="0, GPU-a, 49140, 15, 0\n1, GPU-b, 49140, 188, 0\n")
        return SimpleNamespace(stdout="GPU-b, 999999, 1024\n")

    monkeypatch.setattr(
        "oci.inference.production_all_evidence_workflow.subprocess.run",
        fake_run,
    )
    with pytest.raises(RuntimeError, match="configured resource safety policy"):
        workflow._gpu_preflight()


def test_gpu_preflight_rejects_large_unreported_memory_occupant(monkeypatch, tmp_path):
    workflow = ProductionAllEvidenceWorkflow(
        replace(
            _options(tmp_path),
            gpu_id=None,
            stage1_gpu_ids=(0, 1),
            stage1_device="cuda:1",
            resource_performance_safety=replace(
                _resource_performance_safety(),
                gpu_max_allocation_fraction=0.2,
            ),
        )
    )

    def fake_run(command, **_kwargs):
        if any(value.startswith("--query-gpu=") for value in command):
            return SimpleNamespace(stdout="0, GPU-a, 49140, 15, 0\n1, GPU-b, 49140, 12000, 0\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(
        "oci.inference.production_all_evidence_workflow.subprocess.run",
        fake_run,
    )
    with pytest.raises(
        RuntimeError,
        match="existing_allocation_exceeds_configured_fraction",
    ):
        workflow._gpu_preflight()


def test_gpu_preflight_external_process_policy_is_explicit(
    monkeypatch,
    tmp_path,
):
    workflow = ProductionAllEvidenceWorkflow(
        replace(
            _options(tmp_path),
            gpu_id=None,
            stage1_gpu_ids=(0,),
            stage1_device="cuda:0",
            resource_performance_safety=replace(
                _resource_performance_safety(),
                fail_on_external_gpu_occupants=False,
            ),
        )
    )

    def fake_run(command, **_kwargs):
        if any(value.startswith("--query-gpu=") for value in command):
            return SimpleNamespace(stdout="0, GPU-a, 49140, 15, 0\n")
        return SimpleNamespace(stdout="GPU-a, 999999, 1\n")

    monkeypatch.setattr(
        "oci.inference.production_all_evidence_workflow.subprocess.run",
        fake_run,
    )
    report = workflow._gpu_preflight()
    assert report["exclusive_gpu_check_required"] is False
    assert report["observed_compute_processes"][0][0]["pid"] == 999999


def test_cuda_devices_must_be_covered_by_exclusive_gpu_ids(tmp_path):
    with pytest.raises(ValueError, match="included in the exclusive"):
        ProductionAllEvidenceWorkflow(
            replace(
                _options(tmp_path),
                gpu_id=None,
                stage1_gpu_ids=(0,),
                stage1_device="cuda:1",
                query_device="cuda:0",
            )
        )
    with pytest.raises(ValueError, match="included in the exclusive"):
        ProductionAllEvidenceWorkflow(
            replace(
                _options(tmp_path),
                gpu_id=None,
                stage1_gpu_ids=(0,),
                stage1_device="cuda:0",
                query_device=None,
                query_devices=("cuda:1",),
            )
        )


def test_cache_import_rejects_partial_explicit_source_preparation(tmp_path):
    with pytest.raises(ValueError, match="requires both"):
        ProductionAllEvidenceWorkflow(
            replace(
                _options(tmp_path),
                embedding_cache_import=tmp_path / "cache",
                embedding_cache_import_source_prepared_path=tmp_path / "prepared.parquet",
            )
        )


def test_cache_import_can_discover_its_authenticated_source_preparation(tmp_path):
    options = _options(tmp_path)
    source = tmp_path / "source_prepared"
    source.mkdir()
    prepared = source / "modeling_cohort.parquet"
    manifest = source / "preparation_manifest.json"
    prepared.write_bytes(b"prepared")
    manifest.write_text("{}", encoding="utf-8")
    cache = tmp_path / "source_cache"
    cache.mkdir()
    (cache / "metadata.json").write_text(
        json.dumps({"production_provenance": {"dataset": {"path": str(prepared.resolve())}}}),
        encoding="utf-8",
    )
    workflow = ProductionAllEvidenceWorkflow(replace(options, embedding_cache_import=cache))
    assert workflow._resolved_cache_import_sources() == (
        prepared.resolve(),
        manifest.resolve(),
    )


def test_parallel_cache_preflight_and_modeling_hooks_receive_immutable_context(
    tmp_path,
):
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage1_only=True,
        gpu_id=None,
        stage1_gpu_ids=(),
        stage1_device="cpu",
        query_device="cpu",
        stage1_scope_workers_per_gpu=1,
        stage1_preflight_workers=8,
    )
    observed = {}

    def prepare(attempt):
        prepared = attempt / "prepared"
        prepared.mkdir()
        cohort = prepared / "modeling_cohort.parquet"
        manifest = prepared / "preparation_manifest.json"
        cohort.write_bytes(b"cohort")
        manifest.write_text("{}", encoding="utf-8")
        return {
            "output": {"path": str(cohort)},
            "terminal_files": [str(cohort), str(manifest)],
        }

    def cache(attempt, context):
        observed["cache"] = context
        cache_dir = attempt / "embedding_cache"
        prepared_dir = attempt / "prepared"
        cache_dir.mkdir()
        prepared_dir.mkdir()
        cache_file = cache_dir / "metadata.json"
        cache_file.write_text("{}", encoding="utf-8")
        cohort = prepared_dir / "modeling_cohort.parquet"
        cohort.write_bytes(b"cohort")
        return {
            "schema_version": EMBEDDING_CACHE_PHASE_SCHEMA,
            "cache_path": str(cache_dir),
            "prepared_cohort_path": str(cohort),
            "cache_identity": {"test_identity": True},
            "terminal_files": [str(cache_file), str(cohort)],
        }

    def preflight(attempt, context):
        observed["preflight"] = context
        profile = attempt / "effective_stage1_profile.json"
        artifact = attempt / "cluster_preflight" / "cluster_preflight_manifest.json"
        artifact.parent.mkdir()
        profile.write_text("{}", encoding="utf-8")
        artifact.write_text("{}", encoding="utf-8")
        return {
            "schema_version": STAGE1_PREFLIGHT_PHASE_SCHEMA,
            "effective_profile_path": str(profile),
            "cluster_preflight_manifest_path": str(artifact),
            "terminal_files": [str(profile), str(artifact)],
        }

    def modeling(attempt, context):
        observed["modeling"] = context
        # This hook tests immutable context propagation only; it must not
        # impersonate the closed production Stage 1 bundle schema.
        manifest = attempt / "hook_modeling_marker.json"
        manifest.write_text("{}", encoding="utf-8")
        return {"terminal_files": [str(manifest)]}

    workflow = ProductionAllEvidenceWorkflow(
        options,
        phase_overrides={
            "input_preparation": prepare,
            "handoff_validation": lambda _attempt: {"terminal_files": []},
        },
        hooks=ProductionAllEvidenceWorkflowHooks(
            embedding_cache=cache,
            stage1_preflight=preflight,
            stage1_modeling=modeling,
        ),
    )
    result = workflow.run()
    assert result["stage1_only"] is True
    for phase in ("cache", "preflight", "modeling"):
        assert observed[phase]["request_sha256"]
        assert observed[phase]["stage1_scope_workers_per_gpu"] == 1
        assert observed[phase]["stage1_preflight_workers"] == 8
        assert observed[phase]["resource_preflight"]["requested_gpu_ids"] == []
        assert observed[phase]["stage1_scope_attempt_root"].endswith(
            "/recovery/stage1_scope_attempts"
        )
        assert observed[phase]["stage1_scope_progress_path"].endswith(
            "/recovery/stage1_scope_progress.json"
        )
    assert observed["modeling"]["embedding_cache_path"].endswith("/embedding_cache")


def test_typed_portable_stage1_fails_before_legacy_bundle_build(
    tmp_path,
    monkeypatch,
):
    options = replace(
        _portable_options(tmp_path),
    )
    legacy_builder_constructed = False

    class _ForbiddenLegacyBuilder:
        def __init__(self, _options):
            nonlocal legacy_builder_constructed
            legacy_builder_constructed = True
            raise AssertionError("legacy Stage 1 builder must not be constructed")

    monkeypatch.setattr(
        workflow_module,
        "ProductionStage1BundleBuilder",
        _ForbiddenLegacyBuilder,
    )
    workflow = ProductionAllEvidenceWorkflow(options)
    attempt = tmp_path / "typed_portable_stage1_attempt"
    attempt.mkdir()

    with pytest.raises(
        RuntimeError,
        match="explicit role-neutral.*legacy 40-attempt",
    ):
        workflow._run_default("stage1_modeling", attempt)

    assert legacy_builder_constructed is False
    assert not tuple(attempt.iterdir())


def test_typed_portable_modeling_reopens_preflight_context_without_prepare(
    tmp_path,
    monkeypatch,
) -> None:
    options = _portable_options(tmp_path)
    workflow = ProductionAllEvidenceWorkflow(
        options,
        hooks=workflow_module._default_portable_role_neutral_hooks(options),
    )
    prepared_cohort = (tmp_path / "prepared.parquet").resolve()
    profile = (tmp_path / "effective_stage1_profile.json").resolve()
    cache = (tmp_path / "cache").resolve()
    preflight = (tmp_path / "cluster_preflight_manifest.json").resolve()
    state = (tmp_path / "cluster_state_bundle_manifest.json").resolve()
    context_manifest = (
        tmp_path / "prepared_stage1_context_manifest.json"
    ).resolve()
    for path in (prepared_cohort, profile, preflight, state, context_manifest):
        path.write_text("{}\n", encoding="utf-8")
    cache.mkdir()

    monkeypatch.setattr(
        workflow,
        "_embedding_cache_paths",
        lambda: (cache, prepared_cohort),
    )
    monkeypatch.setattr(
        workflow,
        "_stage1_preflight_paths",
        lambda: (profile, preflight),
    )
    monkeypatch.setattr(
        workflow,
        "_stage1_preflight_state_bundle_path",
        lambda: state,
    )
    monkeypatch.setattr(
        workflow,
        "_stage1_prepared_context_path",
        lambda: context_manifest,
    )

    def forbidden_prepare(_self):
        raise AssertionError(
            "modeling must not rerun monolithic Stage 1 preparation"
        )

    monkeypatch.setattr(
        workflow_module.ProductionStage1BundleBuilder,
        "prepare",
        forbidden_prepare,
    )
    fake_prepared = SimpleNamespace(
        options=SimpleNamespace(
            dataset_path=prepared_cohort,
            config_path=profile,
            cluster_preflight_state_bundle_manifest_path=state,
        ),
        embedding_cache_path=cache,
        cluster_preflight_manifest_path=preflight,
        stage1_scope_plan=SimpleNamespace(physical_scopes=()),
        cluster_preflight_state_bundle=None,
    )
    fake_context = SimpleNamespace(
        reconstruct=lambda **_kwargs: (fake_prepared, object())
    )
    monkeypatch.setattr(
        "oci.inference.prepared_stage1_context."
        "load_prepared_stage1_context",
        lambda _path: fake_context,
    )
    attempt = (tmp_path / "modeling_attempt").resolve()
    attempt.mkdir()
    with pytest.raises(
        RuntimeError,
        match="authenticated no-refit clustered state",
    ):
        workflow._run_portable_role_neutral_stage1_modeling(attempt)


def test_adopted_prepared_context_rebinds_all_consumer_execution_roots(
    tmp_path,
    monkeypatch,
) -> None:
    options = _portable_options(tmp_path)
    workflow = ProductionAllEvidenceWorkflow(options)
    cohort = (tmp_path / "prepared.parquet").resolve()
    profile = (tmp_path / "effective_stage1_profile.json").resolve()
    cache = (tmp_path / "embedding_cache").resolve()
    preflight = (tmp_path / "cluster_preflight_manifest.json").resolve()
    state = (tmp_path / "cluster_state_bundle_manifest.json").resolve()
    context_manifest = (
        tmp_path
        / "producer_checkpoint"
        / "prepared_stage1_context_manifest.json"
    ).resolve()
    rebound_manifest = (
        options.work_root
        / "recovery"
        / "adopted_prepared_stage1_context"
        / "prepared_stage1_context_manifest.json"
    ).resolve()
    for path in (cohort, profile, preflight, state, context_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    cache.mkdir()
    rebound_manifest.parent.mkdir(parents=True, exist_ok=True)
    rebound_manifest.write_text("{}\n", encoding="utf-8")

    from oci.inference.production_stage1_bundle import (
        Stage1BundleBuildOptions,
    )

    current_options = Stage1BundleBuildOptions(
        dataset_path=cohort,
        config_path=profile,
        embedding_cache_dir=cache,
        output_dir=(
            options.work_root
            / "recovery"
            / "prepared_stage1_context_runtime"
        ).resolve(),
        unit_id_column=options.unit_id_column,
        initial_training_partitions=options.initial_training_partitions,
        query_config_path=options.query_profile_path,
        stage1_scope_descriptor_root=(
            options.work_root / "recovery" / "consumer_descriptors"
        ).resolve(),
        stage1_scope_attempt_root=(
            options.work_root / "recovery" / "stage1_scope_attempts"
        ).resolve(),
        stage1_scope_progress_path=(
            options.work_root
            / "recovery"
            / "stage1_scope_progress.json"
        ).resolve(),
        cluster_preflight_manifest_path=preflight,
        cluster_preflight_state_bundle_manifest_path=state,
    )
    from oci.inference.prepared_stage1_context import (
        serialize_stage1_build_options,
    )

    current_mapping = serialize_stage1_build_options(current_options)
    sealed_mapping = deepcopy(current_mapping)
    producer_root = (
        tmp_path / "producer_checkpoint" / "runtime"
    ).resolve()
    sealed_mapping.update(
        {
            "output_dir": str(producer_root),
            "stage1_scope_descriptor_root": str(
                producer_root / "descriptor"
            ),
            "stage1_scope_attempt_root": str(
                producer_root / "attempts"
            ),
            "stage1_scope_progress_path": str(
                producer_root / "progress.json"
            ),
        }
    )
    input_locator_keys = {
        "dataset_path",
        "config_path",
        "embedding_cache_dir",
        "query_config_path",
        "cluster_preflight_manifest_path",
        "cluster_preflight_state_bundle_manifest_path",
    }
    assert all(
        sealed_mapping[key] == current_mapping[key]
        for key in input_locator_keys
    )
    assert sealed_mapping != current_mapping

    htr = options.htr_local_model_path.resolve(strict=True)
    request_body = {
        "dataset": {"path": str(cohort)},
        "source_config": {"path": str(profile)},
        "embedding_cache": {"path": str(cache)},
        "htr_model": {"path": str(htr)},
        "runtime": {
            "device": "cpu",
            "gpu_ids": [],
            "num_workers": 1,
            "tfidf_workers": 1,
            "tfidf_parallel_backend": "loky",
            "query_devices": [],
            "query_nuisance_folds": 3,
            "scope_workers_per_gpu": 1,
            "preflight_workers": 1,
            "scope_descriptor_root": str(producer_root / "descriptor"),
            "scope_attempt_root": str(producer_root / "attempts"),
            "scope_progress_path": str(producer_root / "progress.json"),
        },
    }
    exact_request = {
        **request_body,
        "request_sha256": workflow_module._sha(request_body),
    }
    scientific_root = "c" * 64
    fake_context = SimpleNamespace(
        execution_locators={
            "stage1_build_options": sealed_mapping,
            "exact_stage1_request": exact_request,
        },
        content_root_sha256=scientific_root,
    )
    phase = {
        "result": {
            "prepared_stage1_context_manifest_path": str(
                context_manifest
            )
        },
        "artifacts": [{"path": str(context_manifest)}],
    }
    monkeypatch.setattr(
        workflow,
        "_validated_complete",
        lambda phase_name: (
            phase if phase_name == "stage1_preflight" else None
        ),
    )
    monkeypatch.setattr(
        workflow,
        "_embedding_cache_paths",
        lambda: (cache, cohort),
    )
    monkeypatch.setattr(
        workflow,
        "_stage1_preflight_paths",
        lambda: (profile, preflight),
    )
    monkeypatch.setattr(
        workflow,
        "_stage1_preflight_state_bundle_path",
        lambda: state,
    )
    monkeypatch.setattr(
        workflow,
        "_stage1_build_options",
        lambda **_kwargs: current_options,
    )
    monkeypatch.setattr(
        "oci.inference.prepared_stage1_context."
        "load_prepared_stage1_context",
        lambda _path: fake_context,
    )
    rebound_calls = []

    def rebind(**kwargs):
        rebound_calls.append(kwargs)
        return SimpleNamespace(
            manifest_path=rebound_manifest,
            content_root_sha256=scientific_root,
        )

    monkeypatch.setattr(
        "oci.inference.prepared_stage1_context."
        "rebind_prepared_stage1_context_locators",
        rebind,
    )

    observed = workflow._stage1_prepared_context_path()

    assert observed == rebound_manifest
    assert len(rebound_calls) == 1
    rebound_call = rebound_calls[0]
    assert rebound_call["stage1_build_options"] == current_mapping
    assert rebound_call["stage1_build_options"]["output_dir"] != (
        sealed_mapping["output_dir"]
    )
    rebound_runtime = rebound_call["exact_stage1_request"]["runtime"]
    assert rebound_runtime["scope_attempt_root"].startswith(
        str(options.work_root)
    )
    assert rebound_runtime["scope_progress_path"].startswith(
        str(options.work_root)
    )
    assert not rebound_runtime["scope_attempt_root"].startswith(
        str(producer_root)
    )


def test_typed_portable_scope_concurrency_is_deployment_selected(
    tmp_path,
) -> None:
    portable_root = tmp_path / "portable"
    portable_root.mkdir()
    portable = replace(
        _portable_options(portable_root),
        stage1_scope_workers_per_gpu=2,
        stage1_execution_profile=Stage1ExecutionProfile(
            resource_kind="cpu",
            device_count=1,
            scope_workers_per_device=2,
            executor_mode="persistent_slots",
            selection_method="operator_configured",
            selected_candidate=None,
            benchmark_result_sha256=None,
            benchmark_result_locator=None,
            benchmark_workload_deployment_sha256=None,
            benchmark_workload_deployment_locator=None,
        ),
    )
    workflow = ProductionAllEvidenceWorkflow(portable)

    assert workflow.options.stage1_scope_workers_per_gpu == 2

    historical_root = tmp_path / "historical"
    historical_root.mkdir()
    historical = replace(
        _options(historical_root),
        stage1_scope_workers_per_gpu=2,
    )
    with pytest.raises(ValueError, match="historical Stage 1 requires exactly one"):
        ProductionAllEvidenceWorkflow(historical)


def test_resume_revalidation_reopens_measured_benchmark_authorities(
    tmp_path,
    monkeypatch,
) -> None:
    paths = {}
    for name in ("dataset", "stage1-profile", "query-profile", "scientific"):
        path = (tmp_path / f"{name}.json").resolve()
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    safety = ResourcePerformanceSafetyPolicy(
        gpu_max_allocation_fraction=0.85,
        gpu_minimum_headroom_bytes=6 * 1024**3,
        minimum_multi_device_throughput_ratio=1.5,
        maximum_coordination_proof_overhead_ratio=0.3,
        maximum_ordinary_read_amplification=2.0,
        minimum_benchmark_repetitions_per_scope=2,
        read_counter_source="process_read_bytes",
        fail_on_external_gpu_occupants=True,
    )
    request = {
        "dataset_path": str(paths["dataset"]),
        "source_sha256": workflow_module.stable_file_sha256(
            paths["dataset"]
        )[0],
        "stage1_profile_path": str(paths["stage1-profile"]),
        "stage1_profile_sha256": workflow_module.stable_file_sha256(
            paths["stage1-profile"]
        )[0],
        "query_profile_path": str(paths["query-profile"]),
        "query_profile_sha256": workflow_module.stable_file_sha256(
            paths["query-profile"]
        )[0],
        "scientific_spec_path": str(paths["scientific"]),
        "scientific_spec_source_sha256": (
            workflow_module.stable_file_sha256(paths["scientific"])[0]
        ),
        "deployment_profile_path": None,
        "stage1_execution_profile": {
            "schema_version": "portable_stage1_execution_profile_v3",
            "resource_kind": "accelerator",
            "device_count": 2,
            "scope_workers_per_device": 2,
            "executor_mode": "persistent_slots",
            "selection_method": "measured_role_neutral_benchmark_v1",
            "selected_candidate": "measured-x2",
            "benchmark_result_sha256": "a" * 64,
            "benchmark_result_locator": str(
                (tmp_path / "benchmark-result.json").resolve()
            ),
            "benchmark_workload_deployment_sha256": "b" * 64,
            "benchmark_workload_deployment_locator": str(
                (tmp_path / "workload-deployment.json").resolve()
            ),
        },
        "resource_performance_safety": safety.as_dict(),
        "cpu_budget": 8,
    }

    def reached(**kwargs):
        assert kwargs["profile"].scope_workers_per_device == 2
        assert kwargs["resource_performance_safety"] == safety
        raise RuntimeError("measured-benchmark-revalidation-reached")

    monkeypatch.setattr(
        selection_module,
        "validate_benchmarked_stage1_execution_profile",
        reached,
    )
    with pytest.raises(
        RuntimeError,
        match="measured-benchmark-revalidation-reached",
    ):
        workflow_module._revalidate_request_bound_external_inputs(request)


@pytest.mark.parametrize("injection_kind", ("hook", "override"))
def test_typed_portable_stage1_rejects_generic_modeling_injections(
    tmp_path,
    injection_kind,
):
    options = replace(
        _portable_options(tmp_path),
    )
    modeling = lambda _attempt, *_args: {"terminal_files": []}
    kwargs = (
        {"hooks": ProductionAllEvidenceWorkflowHooks(stage1_modeling=modeling)}
        if injection_kind == "hook"
        else {"phase_overrides": {"stage1_modeling": modeling}}
    )

    with pytest.raises(
        ValueError,
        match="forbids generic modeling hooks and phase overrides",
    ):
        ProductionAllEvidenceWorkflow(options, **kwargs)


def test_role_neutral_handoff_publication_cannot_attest_legacy_build(
    tmp_path,
):
    with pytest.raises(ValueError, match="cannot invoke the legacy"):
        workflow_module.RoleNeutralStage1HandoffPublication(
            bundle_manifest_path=tmp_path / "bundle_manifest.json",
            source_role_neutral_execution_content_sha256="a" * 64,
            legacy_bundle_build_invoked=True,
            all_ten_role_neutral_execution_is_exclusive_evidence_source=True,
        )


@pytest.mark.parametrize(
    ("portable_spec", "expected_direct_loader"),
    ((None, False), ({"schema_version": "portable-test-v1"}, True)),
)
def test_handoff_validation_dispatches_only_portable_mode_to_direct_loader(
    tmp_path,
    monkeypatch,
    portable_spec,
    expected_direct_loader,
):
    options = _options(tmp_path) if portable_spec is None else _portable_options(tmp_path)
    workflow = ProductionAllEvidenceWorkflow(options)
    manifest = (tmp_path / "stage1_bundle" / "bundle_manifest.json").resolve()
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        workflow,
        "_validated_complete",
        lambda phase: {
            "phase": phase,
            "artifacts": [{"path": str(manifest)}],
        },
    )
    observed = {}

    def validate(
        *,
        bundle_manifest,
        report_path,
        portable_role_neutral,
    ):
        observed.update(
            {
                "bundle_manifest": bundle_manifest,
                "report_path": report_path,
                "portable_role_neutral": portable_role_neutral,
            }
        )
        return {"status": "accepted"}

    monkeypatch.setattr(
        workflow,
        "_validate_handoff_in_fresh_process",
        validate,
    )
    attempt = (tmp_path / "handoff_attempt").resolve()
    attempt.mkdir()
    result = workflow._run_default("handoff_validation", attempt)

    assert observed["bundle_manifest"] == manifest
    assert observed["portable_role_neutral"] is expected_direct_loader
    assert result["fresh_process_validation"] == {"status": "accepted"}


def test_public_main_binds_default_role_neutral_integration_for_typed_mode(
    tmp_path,
    monkeypatch,
    capsys,
):
    portable = _portable_options(tmp_path)
    profiles = deepcopy(
        portable.portable_scientific_spec["architecture_profiles"]
    )
    portable_spec = dict(portable.portable_scientific_spec)
    portable_spec["architecture_profiles"] = profiles
    portable_spec["stage2_prompt_protocol"] = _stage2_protocol().as_dict()
    options = replace(
        portable,
        portable_scientific_spec=portable_spec,
    )
    observed = {}

    class Parser:
        def parse_args(self, _argv):
            return SimpleNamespace(prepare_stage1_canary_descriptors_only=False)

        def error(self, message):
            raise AssertionError(message)

    class Workflow:
        def __init__(self, supplied_options, *, hooks):
            observed["options"] = supplied_options
            observed["hooks"] = hooks

        def run(self):
            return {"status": "not_started_test"}

    monkeypatch.setattr(workflow_module, "build_parser", lambda: Parser())
    monkeypatch.setattr(
        workflow_module,
        "_reexec_from_source_snapshot",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        workflow_module,
        "options_from_args",
        lambda _args: options,
    )
    monkeypatch.setattr(
        workflow_module,
        "ProductionAllEvidenceWorkflow",
        Workflow,
    )

    assert workflow_module.main([]) == 0
    integration = observed["hooks"].role_neutral_stage1
    assert integration is not None
    assert set(integration.producer_factories_builder.architecture_profiles) == set(
        EVIDENCE_FAMILIES
    )
    assert type(integration.executor).__name__ == (
        "PersistentSpawnRoleNeutralPhysicalOwnerExecutor"
    )
    assert integration.executor.process_isolated_physical_owners is True
    assert type(integration.handoff_publisher).__name__ == (
        "ReferenceOnlyRoleNeutralStage1HandoffPublisher"
    )
    assert "not_started_test" in capsys.readouterr().out


def test_default_typed_integration_rejects_missing_profile_before_fitting(
    tmp_path,
) -> None:
    profiles = {family: {"configured_profile": family} for family in EVIDENCE_FAMILIES[:-1]}
    portable = _portable_options(tmp_path)
    portable_spec = dict(portable.portable_scientific_spec)
    portable_spec["architecture_profiles"] = profiles
    options = replace(
        portable,
        portable_scientific_spec=portable_spec,
    )

    with pytest.raises(
        ValueError,
        match="profiles differ from all ten",
    ):
        workflow_module._default_portable_role_neutral_hooks(options)


def test_relocated_cache_attestation_is_propagated_to_stage1_builder(
    tmp_path,
    monkeypatch,
):
    source_cache = tmp_path / "source_cache"
    source_cache.mkdir()
    source_prepared = tmp_path / "source_prepared.parquet"
    source_manifest = tmp_path / "source_preparation_manifest.json"
    source_prepared.write_bytes(b"source")
    source_manifest.write_text("{}", encoding="utf-8")
    options = replace(
        _options(tmp_path),
        embedding_cache_import=source_cache,
        embedding_cache_import_source_prepared_path=source_prepared,
        embedding_cache_import_source_preparation_manifest_path=source_manifest,
    )
    workflow = ProductionAllEvidenceWorkflow(options)
    sentinel = object()
    monkeypatch.setattr(
        workflow,
        "_embedding_cache_relocation_options",
        lambda **_kwargs: sentinel,
    )
    profile = tmp_path / "effective.json"
    profile.write_text("{}", encoding="utf-8")
    cache = tmp_path / "relocated" / "embedding_cache"
    cache.mkdir(parents=True)
    prepared = tmp_path / "relocated" / "prepared" / "modeling_cohort.parquet"
    prepared.parent.mkdir()
    prepared.write_bytes(b"prepared")
    built = workflow._stage1_build_options(
        dataset=prepared,
        profile=profile,
        cache=cache,
        output=tmp_path / "bundle",
        dry_run=False,
    )
    assert built.embedding_cache_relocation is sentinel
    assert (
        built.stage1_scope_attempt_root
        == (options.work_root / "recovery/stage1_scope_attempts").resolve()
    )
    assert (
        built.stage1_scope_progress_path
        == (options.work_root / "recovery/stage1_scope_progress.json").resolve()
    )


def test_portable_semantic_witness_profile_is_bound_to_stage1_request(
    tmp_path,
) -> None:
    options = _portable_options(tmp_path)
    portable_identity = deepcopy(options.portable_scientific_spec)
    benchmark = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "example_configs"
            / "portable_all_evidence_scientific_nsclc.json"
        ).read_text(encoding="utf-8")
    )
    portable_identity["architecture_profiles"][
        "lexical_semantic_retrieval"
    ] = deepcopy(
        benchmark["architecture_profiles"][
            "lexical_semantic_retrieval"
        ]
    )
    options = replace(
        options,
        portable_scientific_spec=portable_identity,
    )
    workflow = ProductionAllEvidenceWorkflow(options)
    cache = tmp_path / "embedding-cache"
    cache.mkdir()
    output = tmp_path / "bundle"

    built = workflow._stage1_build_options(
        dataset=options.dataset_path,
        profile=options.stage1_profile_path,
        cache=cache,
        output=output,
        dry_run=True,
    )
    expected = semantic_witness_config_from_portable_scientific_spec(
        options.portable_scientific_spec
    )

    assert (
        built.semantic_witness_scientific_config.identity_sha256
        == expected.identity_sha256
    )
    assert (
        built.semantic_witness_scientific_config.as_dict()
        == options.portable_scientific_spec["architecture_profiles"][
            "lexical_semantic_retrieval"
        ]["producer_configuration"]
    )
    assert built.portable_cluster_preflight_v2 is True


def test_effective_profile_binds_review_tfidf_and_interaction_cli_settings(
    tmp_path,
):
    options = replace(
        _options(tmp_path),
        outer_folds=7,
        review_rounds=4,
        initial_training_partitions=3,
        tfidf_nested_calibration_folds=6,
        interaction_inner_folds=7,
    )
    profile = {
        "config": {
            "architecture": {
                "htr_sentence_model": "old",
                "htr_chunk_size_words": 37,
                "htr_chunk_overlap_words": 5,
                "htr_max_chunks": 999,
                "htr_max_chunk_length": 71,
                "multi_model_forest": {
                    "candidate_consistency_inner_folds": 2,
                    "tfidf_nested_calibration_folds": 2,
                    "embedding_contrast": {},
                },
                "multi_model_agentic_forest": {
                    "candidate_consistency_inner_folds": 2,
                    "tfidf_nested_calibration_folds": 2,
                    "embedding_contrast": {},
                },
                "explicit_feature_forest": {"interaction_inner_folds": 2},
                "causal_forest": {},
            }
        }
    }
    options.stage1_profile_path.write_text(json.dumps(profile), encoding="utf-8")
    workflow = ProductionAllEvidenceWorkflow(options)
    attempt = tmp_path / "effective"
    attempt.mkdir()
    cache = tmp_path / "cache_for_profile"
    cache.mkdir()
    path = workflow._effective_stage1_profile(
        attempt,
        dataset_path=options.dataset_path,
        embedding_cache_dir=cache,
    )
    effective = json.loads(path.read_text(encoding="utf-8"))["config"]
    architecture = effective["architecture"]
    assert effective["clinical_question"] == options.clinical_question
    for section_name in ("multi_model_forest", "multi_model_agentic_forest"):
        assert architecture[section_name]["candidate_consistency_inner_folds"] == 7
        assert architecture[section_name]["tfidf_nested_calibration_folds"] == 6
    assert architecture["explicit_feature_forest"]["interaction_inner_folds"] == 7


def test_effective_profile_rejects_implicit_htr_text_window_defaults(tmp_path):
    options = _options(tmp_path)
    profile = {
        "config": {
            "architecture": {
                "htr_sentence_model": "old",
                "htr_chunk_size_words": 37,
                "htr_chunk_overlap_words": 5,
                "htr_max_chunk_length": 71,
            }
        }
    }
    options.stage1_profile_path.write_text(json.dumps(profile), encoding="utf-8")
    workflow = ProductionAllEvidenceWorkflow(options)
    attempt = tmp_path / "effective_missing_htr_window"
    attempt.mkdir()
    cache = tmp_path / "cache_for_missing_htr_window"
    cache.mkdir()

    with pytest.raises(ValueError, match="htr_max_chunks"):
        workflow._effective_stage1_profile(
            attempt,
            dataset_path=options.dataset_path,
            embedding_cache_dir=cache,
        )


@pytest.mark.parametrize("mutation", ["change", "extra"])
def test_resume_rejects_any_change_to_a_sealed_attempt_tree(tmp_path, mutation):
    options = _options(tmp_path)
    payload_path = None

    def input_phase(attempt):
        nonlocal payload_path
        payload_path = attempt / "unlisted" / "nested.bin"
        payload_path.parent.mkdir()
        payload_path.write_bytes(b"sealed")
        return {"terminal_files": []}

    overrides = {phase: (lambda _attempt: {"terminal_files": []}) for phase in PHASES}
    overrides["input_preparation"] = input_phase
    ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()
    assert payload_path is not None
    manifest = json.loads(
        (options.work_root / "phases/input_preparation/complete_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert any(row["relative_path"] == "unlisted/nested.bin" for row in manifest["artifacts"])
    assert not payload_path.exists()
    payload_path = Path(manifest["attempt_dir"]) / "unlisted" / "nested.bin"
    assert payload_path.is_file()
    assert payload_path.is_relative_to(options.work_root / "phases" / "input_preparation")
    if mutation == "change":
        payload_path.write_bytes(b"changed")
    else:
        (payload_path.parent / "extra.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="attempt tree changed"):
        ProductionAllEvidenceWorkflow(
            _with_run_control(options, resume=True),
            phase_overrides=overrides,
        ).run()


def test_source_snapshot_option_reexecs_from_authenticated_tree(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshot"
    entrypoint = snapshot_root / "scripts/run_production_all_evidence_workflow.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# fixture", encoding="utf-8")
    snapshot = SimpleNamespace(
        root=snapshot_root.resolve(),
        content_sha256="a" * 64,
    )
    monkeypatch.setattr(
        "oci.inference.production_source_snapshot.validate_production_source_snapshot",
        lambda _path: snapshot,
    )
    monkeypatch.delenv(workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV, raising=False)
    observed = {}

    class ReexecObserved(Exception):
        pass

    def fake_execve(executable, arguments, environment):
        observed.update(
            executable=executable,
            arguments=arguments,
            environment=environment,
        )
        raise ReexecObserved

    monkeypatch.setattr(workflow_module.os, "execve", fake_execve)
    scientific_path = _write_scientific_spec(tmp_path / "scientific.json")
    parsed = SimpleNamespace(
        source_snapshot_root=snapshot_root,
        scientific_spec=scientific_path,
    )
    with pytest.raises(ReexecObserved):
        workflow_module._reexec_from_source_snapshot(
            parsed_args=parsed,
            raw_argv=(
                "--source-snapshot-root",
                str(snapshot_root),
                "--scientific-spec",
                str(scientific_path),
            ),
        )
    assert observed["arguments"][:3] == [
        workflow_module.sys.executable,
        "-P",
        "-u",
    ]
    assert observed["arguments"][3] == str(entrypoint)
    assert observed["environment"]["PYTHONPATH"] == str(snapshot_root.resolve())
    assert (
        observed["environment"][workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV]
        == snapshot.content_sha256
    )
    assert observed["environment"]["PYTHONHASHSEED"] == str(_scientific_spec().seed)


def test_source_snapshot_reexec_rejects_changed_parent_python_hash_seed(tmp_path, monkeypatch):
    snapshot_root = tmp_path / "snapshot"
    entrypoint = snapshot_root / "scripts/run_production_all_evidence_workflow.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("# fixture", encoding="utf-8")
    snapshot = SimpleNamespace(
        root=snapshot_root.resolve(),
        content_sha256="b" * 64,
    )
    monkeypatch.setattr(
        "oci.inference.production_source_snapshot.validate_production_source_snapshot",
        lambda _path: snapshot,
    )
    monkeypatch.setattr(
        workflow_module,
        "__file__",
        str(snapshot_root / "oci/inference/production_all_evidence_workflow.py"),
    )
    monkeypatch.setenv(
        workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV,
        snapshot.content_sha256,
    )
    monkeypatch.setenv("PYTHONHASHSEED", "41")
    scientific_path = _write_scientific_spec(tmp_path / "scientific.json")

    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        workflow_module._reexec_from_source_snapshot(
            parsed_args=SimpleNamespace(
                source_snapshot_root=snapshot_root,
                scientific_spec=scientific_path,
            ),
            raw_argv=(
                "--source-snapshot-root",
                str(snapshot_root),
                "--scientific-spec",
                str(scientific_path),
            ),
        )


def test_fresh_canary_validator_sets_and_verifies_snapshot_environment(
    tmp_path,
    monkeypatch,
):
    options = replace(
        _options(tmp_path),
        endpoint=None,
        model_name=None,
        stage1_only=True,
        seed=42,
    )
    snapshot_root = (tmp_path / "snapshot").resolve()
    module_path = snapshot_root / "oci" / "inference" / "production_all_evidence_workflow.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# snapshot fixture\n", encoding="utf-8")
    snapshot_sha = "d" * 64
    workflow = ProductionAllEvidenceWorkflow(options)
    workflow.request = {
        "source_snapshot": {
            "root": str(snapshot_root),
            "content_sha256": snapshot_sha,
        }
    }
    options.work_root.mkdir()
    expected_result = {"status": "complete"}
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        observed["environment"] = dict(kwargs["env"])
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "result": expected_result,
                    "validator_module_path": str(module_path),
                    "source_snapshot_marker": kwargs["env"][
                        workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV
                    ],
                    "python_hash_seed": kwargs["env"]["PYTHONHASHSEED"],
                    "python_path": kwargs["env"]["PYTHONPATH"],
                    "python_no_user_site": kwargs["env"]["PYTHONNOUSERSITE"],
                }
            )
        )

    monkeypatch.setattr(workflow_module.subprocess, "run", fake_run)

    assert workflow._validate_canary_preparation_in_fresh_process() == expected_result
    assert observed["command"][1] == "-P"
    assert observed["environment"]["PYTHONHASHSEED"] == "42"
    assert observed["environment"]["PYTHONPATH"] == str(snapshot_root)
    assert observed["environment"]["PYTHONNOUSERSITE"] == "1"
    assert observed["environment"][workflow_module.SOURCE_SNAPSHOT_EXECUTION_ENV] == snapshot_sha


def test_interrupted_initial_request_publication_preserves_attempt_and_fresh_root(
    tmp_path,
    monkeypatch,
):
    options = _options(tmp_path)
    workflow = ProductionAllEvidenceWorkflow(options)
    original_atomic_write = workflow_module._atomic_write_json

    def interrupt_request(path, value):
        if path.name == "immutable_run_request.json":
            raise KeyboardInterrupt("fixture interruption")
        return original_atomic_write(path, value)

    monkeypatch.setattr(workflow_module, "_atomic_write_json", interrupt_request)
    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        workflow._initialize()

    assert not options.work_root.exists()
    attempts = tuple(
        options.work_root.parent.glob(f".{options.work_root.name}.initialization_attempt_*")
    )
    assert len(attempts) == 1
    assert attempts[0].is_dir()

    monkeypatch.setattr(workflow_module, "_atomic_write_json", original_atomic_write)
    ProductionAllEvidenceWorkflow(options)._initialize()
    assert (options.work_root / "immutable_run_request.json").is_file()
    assert attempts[0].is_dir()


@pytest.mark.parametrize(
    ("field", "expected_message"),
    (
        ("query_profile_path", "neural-query profile changed"),
        ("embedding_local_model_path", "embedding model tree changed"),
    ),
)
def test_phase_boundary_rejects_request_bound_external_input_changes(
    tmp_path,
    field,
    expected_message,
):
    options = _options(tmp_path)
    target = Path(getattr(options, field))
    if target.is_dir():
        target = target / "model.safetensors"

    def mutate_bound_input(_attempt):
        target.write_text("changed after immutable request", encoding="utf-8")
        return {"terminal_files": []}

    overrides = {phase: (lambda _attempt: {"terminal_files": []}) for phase in PHASES}
    overrides["input_preparation"] = mutate_bound_input
    workflow = ProductionAllEvidenceWorkflow(
        options,
        phase_overrides=overrides,
    )

    with pytest.raises(RuntimeError, match=expected_message):
        workflow.run()
    assert not (
        options.work_root / "phases" / "input_preparation" / "complete_manifest.json"
    ).exists()


def test_imported_cache_workflow_hashes_embedding_tree_once_per_process(
    tmp_path,
    monkeypatch,
):
    options = _options(tmp_path)
    source_cache = (tmp_path / "source-cache").resolve()
    source_cache.mkdir()
    (source_cache / "metadata.json").write_text("{}\n", encoding="utf-8")
    source_prepared = (tmp_path / "source-prepared.parquet").resolve()
    source_prepared.write_bytes(b"prepared")
    source_manifest = (tmp_path / "source-preparation.json").resolve()
    source_manifest.write_text("{}\n", encoding="utf-8")
    options = replace(
        options,
        embedding_cache_import=source_cache,
        embedding_cache_import_source_prepared_path=source_prepared,
        embedding_cache_import_source_preparation_manifest_path=source_manifest,
    )
    tree_module.clear_authenticated_directory_tree_cache()
    calls: dict[Path, int] = {}
    original = tree_module._stable_file_authentication

    def counted(root: Path, relative_path: str):
        calls[root] = calls.get(root, 0) + 1
        return original(root, relative_path)

    monkeypatch.setattr(tree_module, "_stable_file_authentication", counted)
    overrides = {phase: (lambda _attempt: {"terminal_files": []}) for phase in PHASES}
    ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()

    assert calls[options.embedding_local_model_path] == 1
    assert calls[source_cache] == 1
    request = json.loads(
        (options.work_root / "immutable_run_request.json").read_text(encoding="utf-8")
    )
    assert (
        request["embedding_model_revalidation_policy"]
        == tree_module.AUTHENTICATED_DIRECTORY_TREE_POLICY
    )


def test_fresh_cache_build_keeps_full_model_tree_validation_path(
    tmp_path,
    monkeypatch,
):
    options = _options(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fresh embedding-cache builds cannot use process-local tree reuse")

    monkeypatch.setattr(workflow_module, "authenticate_directory_tree", forbidden)
    overrides = {phase: (lambda _attempt: {"terminal_files": []}) for phase in PHASES}
    ProductionAllEvidenceWorkflow(options, phase_overrides=overrides).run()
    request = json.loads(
        (options.work_root / "immutable_run_request.json").read_text(encoding="utf-8")
    )
    assert request["embedding_model_revalidation_policy"] == "full_byte_tree_reauthentication_v1"
